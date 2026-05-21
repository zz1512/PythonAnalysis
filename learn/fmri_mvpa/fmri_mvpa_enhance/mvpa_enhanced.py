#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA分析模块 - 增强版
基于fmri_mvpa_roi_pipeline_enhanced.py进行优化
集成FDR校正、并行处理、内存优化和HTML报告
"""

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import json
from datetime import datetime
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedMVPAAnalyzer:
    """增强版MVPA分析器"""
    
    def __init__(self, config):
        self.config = config
        self.mvpa_config = config.mvpa
        
        # 创建结果目录
        Path(self.mvpa_config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # 性能监控
        self.performance_stats = {
            'start_time': None,
            'end_time': None,
            'memory_usage': [],
            'processing_times': {},
            'failed_analyses': []
        }
        
        # 结果存储
        self.results = {
            'individual_results': [],
            'group_results': None,
            'statistics': None
        }
        
    def log(self, message: str):
        """增强的日志函数"""
        logging.info(message)
        
        # 保存到日志文件
        log_file = Path(self.mvpa_config.results_dir) / "analysis.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [MVPA] {message}\n")
    
    def memory_cleanup(self):
        """内存清理"""
        gc.collect()
    
    def load_roi_masks(self) -> Dict[str, str]:
        """加载ROI masks"""
        roi_dir = Path(self.mvpa_config.roi_dir)
        
        if not roi_dir.exists():
            raise FileNotFoundError(f"ROI目录不存在: {roi_dir}")
        
        roi_masks = {}
        for roi_file in roi_dir.glob("*.nii*"):
            roi_name = roi_file.stem.replace('.nii', '')
            roi_masks[roi_name] = str(roi_file)
        
        self.log(f"加载了 {len(roi_masks)} 个ROI masks")
        return roi_masks
    
    def load_lss_trial_data(self, subject: str, run: int) -> Optional[Tuple[pd.DataFrame, Dict[str, List[str]]]]:
        """加载LSS试验数据"""
        lss_dir = Path(self.mvpa_config.lss_root) / subject / f"run{run}"
        
        if not lss_dir.exists():
            self.log(f"LSS目录不存在: {lss_dir}")
            return None
        
        # 加载试验信息
        trial_info_file = lss_dir / "trial_info.csv"
        if not trial_info_file.exists():
            self.log(f"试验信息文件不存在: {trial_info_file}")
            return None
        
        trial_info = pd.read_csv(trial_info_file)
        
        # 只保留成功的试验
        successful_trials = trial_info[trial_info['success'] == True].copy()
        
        if len(successful_trials) == 0:
            self.log(f"没有成功的试验: {subject} run{run}")
            return None
        
        # 构建beta文件路径
        beta_images = {}
        for condition in successful_trials['condition'].unique():
            condition_trials = successful_trials[successful_trials['condition'] == condition]
            beta_files = []
            
            for _, trial in condition_trials.iterrows():
                trial_idx = trial['trial_idx']
                beta_file = lss_dir / f"trial_{trial_idx:03d}_{condition}_beta.nii.gz"
                
                if beta_file.exists():
                    beta_files.append(str(beta_file))
                else:
                    self.log(f"Beta文件不存在: {beta_file}")
            
            if beta_files:
                beta_images[condition] = beta_files
        
        return successful_trials, beta_images
    
    def load_multi_run_lss_data(self, subject: str, runs: List[int]) -> Optional[Tuple[pd.DataFrame, Dict[str, List[str]]]]:
        """加载多个run的LSS数据"""
        all_trial_info = []
        all_beta_images = {}
        
        for run in runs:
            data = self.load_lss_trial_data(subject, run)
            if data is None:
                continue
            
            trial_info, beta_images = data
            
            # 添加run信息
            trial_info = trial_info.copy()
            trial_info['run'] = run
            trial_info['subject'] = subject
            
            all_trial_info.append(trial_info)
            
            # 合并beta images
            for condition, files in beta_images.items():
                if condition not in all_beta_images:
                    all_beta_images[condition] = []
                all_beta_images[condition].extend(files)
        
        if not all_trial_info:
            return None
        
        combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
        
        return combined_trial_info, all_beta_images
    
    def extract_roi_timeseries_optimized(self, functional_imgs: List[str], 
                                        roi_mask_path: str,
                                        confounds: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """优化版ROI时间序列提取"""
        try:
            # 使用NiftiMasker进行高效提取
            masker = NiftiMasker(
                mask_img=roi_mask_path,
                standardize=True,
                detrend=True,
                low_pass=None,
                high_pass=None,
                t_r=None,
                memory=self.mvpa_config.memory_cache,
                memory_level=self.mvpa_config.memory_level,
                verbose=0
            )
            
            # 提取时间序列
            timeseries_list = []
            for img_path in functional_imgs:
                try:
                    img = image.load_img(img_path)
                    ts = masker.fit_transform(img, confounds=confounds)
                    # 对于单个试验的beta图，取平均值
                    timeseries_list.append(ts.mean(axis=0))
                except Exception as e:
                    self.log(f"提取时间序列失败 {img_path}: {e}")
                    continue
            
            if not timeseries_list:
                return None
            
            return np.array(timeseries_list)
            
        except Exception as e:
            self.log(f"ROI时间序列提取失败: {e}")
            return None
    
    def run_roi_classification_enhanced(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """增强版ROI分类分析"""
        try:
            # 数据验证
            if len(X) != len(y):
                raise ValueError("特征和标签长度不匹配")
            
            if len(np.unique(y)) < 2:
                raise ValueError("标签类别少于2个")
            
            # 创建分类器pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(**self.mvpa_config.svm_params))
            ])
            
            # 网格搜索最优参数
            param_grid = {f'svm__{k}': v for k, v in self.mvpa_config.svm_param_grid.items()}
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=self.mvpa_config.cv_folds, 
                                 shuffle=True, 
                                 random_state=self.mvpa_config.cv_random_state),
                scoring='accuracy',
                n_jobs=1,  # 避免嵌套并行
                verbose=0
            )
            
            # 拟合模型
            grid_search.fit(X, y)
            best_pipeline = grid_search.best_estimator_
            
            # 交叉验证评估
            cv_scores = cross_val_score(
                best_pipeline,
                X, y,
                cv=StratifiedKFold(n_splits=self.mvpa_config.cv_folds, 
                                 shuffle=True, 
                                 random_state=self.mvpa_config.cv_random_state),
                scoring='accuracy',
                n_jobs=1
            )
            
            # 置换检验
            permutation_scores = []
            np.random.seed(self.mvpa_config.permutation_random_state)
            
            for _ in range(self.mvpa_config.n_permutations):
                y_perm = np.random.permutation(y)
                perm_score = cross_val_score(
                    best_pipeline, X, y_perm,
                    cv=StratifiedKFold(n_splits=self.mvpa_config.cv_folds, 
                                     shuffle=True, 
                                     random_state=self.mvpa_config.cv_random_state),
                    scoring='accuracy',
                    n_jobs=1
                ).mean()
                permutation_scores.append(perm_score)
            
            # 计算p值
            p_value = (np.sum(np.array(permutation_scores) >= cv_scores.mean()) + 1) / (self.mvpa_config.n_permutations + 1)
            
            return {
                'cv_scores': cv_scores,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'permutation_scores': permutation_scores,
                'p_value': p_value,
                'n_samples': len(X),
                'n_features': X.shape[1] if X.ndim > 1 else 1,
                'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
            }
            
        except Exception as e:
            self.log(f"ROI分类分析失败: {e}")
            return {
                'cv_scores': np.array([]),
                'mean_accuracy': np.nan,
                'std_accuracy': np.nan,
                'best_params': {},
                'permutation_scores': [],
                'p_value': 1.0,
                'n_samples': len(X) if X is not None else 0,
                'n_features': 0,
                'class_distribution': {},
                'error': str(e)
            }
    
    def prepare_classification_data(self, trial_info: pd.DataFrame, 
                                  beta_images: Dict[str, List[str]], 
                                  cond1: str, cond2: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """准备分类数据"""
        try:
            if cond1 not in beta_images or cond2 not in beta_images:
                self.log(f"条件 {cond1} 或 {cond2} 的数据不存在")
                return None
            
            # 获取两个条件的文件
            files_cond1 = beta_images[cond1]
            files_cond2 = beta_images[cond2]
            
            if len(files_cond1) == 0 or len(files_cond2) == 0:
                self.log(f"条件 {cond1} 或 {cond2} 没有有效文件")
                return None
            
            # 平衡样本数量
            min_samples = min(len(files_cond1), len(files_cond2))
            if min_samples < 3:  # 最少需要3个样本进行交叉验证
                self.log(f"样本数量不足: {cond1}={len(files_cond1)}, {cond2}={len(files_cond2)}")
                return None
            
            # 随机选择平衡的样本
            np.random.seed(42)
            selected_files_cond1 = np.random.choice(files_cond1, min_samples, replace=False)
            selected_files_cond2 = np.random.choice(files_cond2, min_samples, replace=False)
            
            # 合并文件和标签
            all_files = list(selected_files_cond1) + list(selected_files_cond2)
            labels = [0] * min_samples + [1] * min_samples  # 0: cond1, 1: cond2
            
            return all_files, np.array(labels)
            
        except Exception as e:
            self.log(f"准备分类数据失败: {e}")
            return None
    
    def analyze_single_roi_parallel(self, args: Tuple) -> Dict[str, Any]:
        """并行分析单个ROI"""
        roi_name, roi_path, subject, runs, contrasts = args
        
        try:
            # 加载LSS数据
            data = self.load_multi_run_lss_data(subject, runs)
            if data is None:
                return {
                    'roi_name': roi_name,
                    'subject': subject,
                    'success': False,
                    'error': 'LSS数据加载失败'
                }
            
            trial_info, beta_images = data
            
            roi_results = {
                'roi_name': roi_name,
                'subject': subject,
                'success': True,
                'contrasts': {}
            }
            
            # 分析每个对比
            for contrast_name, cond1, cond2 in contrasts:
                # 准备分类数据
                class_data = self.prepare_classification_data(trial_info, beta_images, cond1, cond2)
                if class_data is None:
                    roi_results['contrasts'][contrast_name] = {
                        'success': False,
                        'error': '分类数据准备失败'
                    }
                    continue
                
                all_files, labels = class_data
                
                # 提取ROI特征
                X = self.extract_roi_timeseries_optimized(all_files, roi_path)
                if X is None:
                    roi_results['contrasts'][contrast_name] = {
                        'success': False,
                        'error': 'ROI特征提取失败'
                    }
                    continue
                
                # 运行分类分析
                classification_results = self.run_roi_classification_enhanced(X, labels)
                classification_results['contrast_name'] = contrast_name
                classification_results['conditions'] = (cond1, cond2)
                classification_results['success'] = True
                
                roi_results['contrasts'][contrast_name] = classification_results
            
            return roi_results
            
        except Exception as e:
            return {
                'roi_name': roi_name,
                'subject': subject,
                'success': False,
                'error': str(e)
            }
    
    def analyze_single_subject_roi_enhanced(self, subject: str, runs: List[int], 
                                          contrasts: List[Tuple[str, str, str]], 
                                          roi_masks: Dict[str, str]) -> Dict[str, Any]:
        """增强版单被试ROI分析"""
        self.log(f"分析被试: {subject}")
        
        subject_results = {
            'subject': subject,
            'runs': runs,
            'roi_results': {},
            'summary': {
                'total_rois': len(roi_masks),
                'successful_rois': 0,
                'failed_rois': 0
            }
        }
        
        if self.mvpa_config.use_parallel and self.mvpa_config.n_jobs > 1:
            # 并行处理ROI
            args_list = [(roi_name, roi_path, subject, runs, contrasts) 
                        for roi_name, roi_path in roi_masks.items()]
            
            with Pool(processes=min(self.mvpa_config.n_jobs, len(roi_masks))) as pool:
                roi_results = pool.map(self.analyze_single_roi_parallel, args_list)
        else:
            # 串行处理ROI
            roi_results = []
            for roi_name, roi_path in roi_masks.items():
                args = (roi_name, roi_path, subject, runs, contrasts)
                result = self.analyze_single_roi_parallel(args)
                roi_results.append(result)
        
        # 整理结果
        for result in roi_results:
            roi_name = result['roi_name']
            subject_results['roi_results'][roi_name] = result
            
            if result['success']:
                subject_results['summary']['successful_rois'] += 1
            else:
                subject_results['summary']['failed_rois'] += 1
                self.performance_stats['failed_analyses'].append((subject, roi_name, result.get('error', 'Unknown')))
        
        self.log(f"被试 {subject} 完成: {subject_results['summary']['successful_rois']}/{subject_results['summary']['total_rois']} ROI成功")
        
        return subject_results
    
    def perform_group_statistics_enhanced(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """增强版组水平统计分析"""
        self.log("执行组水平统计分析")
        
        # 按ROI和对比分组
        grouped = group_df.groupby(['roi_name', 'contrast_name'])
        
        stats_results = []
        
        for (roi_name, contrast_name), group in grouped:
            if len(group) < 2:
                continue
            
            accuracies = group['mean_accuracy'].values
            
            # 单样本t检验（与机会水平0.5比较）
            t_stat, p_value = stats.ttest_1samp(accuracies, 0.5)
            
            # 效应量（Cohen's d）
            cohens_d = (accuracies.mean() - 0.5) / accuracies.std()
            
            # 置换检验的组水平p值
            perm_p_values = group['p_value'].values
            combined_perm_p = stats.combine_pvalues(perm_p_values, method='fisher')[1]
            
            stats_results.append({
                'roi_name': roi_name,
                'contrast_name': contrast_name,
                'n_subjects': len(group),
                'mean_accuracy': accuracies.mean(),
                'std_accuracy': accuracies.std(),
                'sem_accuracy': accuracies.std() / np.sqrt(len(accuracies)),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'combined_perm_p': combined_perm_p,
                'min_accuracy': accuracies.min(),
                'max_accuracy': accuracies.max()
            })
        
        stats_df = pd.DataFrame(stats_results)
        
        if len(stats_df) > 0:
            # 多重比较校正
            for contrast in stats_df['contrast_name'].unique():
                contrast_mask = stats_df['contrast_name'] == contrast
                p_values = stats_df.loc[contrast_mask, 'p_value'].values
                
                # FDR校正
                rejected, p_corrected, _, _ = multipletests(
                    p_values, 
                    alpha=self.mvpa_config.alpha_level, 
                    method=self.mvpa_config.correction_method
                )
                
                stats_df.loc[contrast_mask, 'p_corrected'] = p_corrected
                stats_df.loc[contrast_mask, 'significant'] = rejected
        
        return stats_df
    
    def create_enhanced_visualizations(self, group_df: pd.DataFrame, stats_df: pd.DataFrame) -> Dict[str, str]:
        """创建增强版可视化"""
        self.log("创建可视化图表")
        
        viz_dir = Path(self.mvpa_config.results_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        saved_plots = {}
        
        try:
            # 设置绘图风格
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. 准确率分布图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 整体准确率分布
            axes[0, 0].hist(group_df['mean_accuracy'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Chance Level')
            axes[0, 0].set_xlabel('Mean Accuracy')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Classification Accuracies')
            axes[0, 0].legend()
            
            # 按ROI的准确率
            if len(group_df['roi_name'].unique()) <= 20:  # 只在ROI数量不太多时绘制
                sns.boxplot(data=group_df, x='roi_name', y='mean_accuracy', ax=axes[0, 1])
                axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
                axes[0, 1].set_title('Accuracy by ROI')
            
            # 按对比的准确率
            sns.boxplot(data=group_df, x='contrast_name', y='mean_accuracy', ax=axes[1, 0])
            axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Accuracy by Contrast')
            
            # p值分布
            if len(stats_df) > 0:
                axes[1, 1].hist(stats_df['p_value'], bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
                axes[1, 1].set_xlabel('p-value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Distribution of p-values')
                axes[1, 1].legend()
            
            plt.tight_layout()
            overview_path = viz_dir / "mvpa_overview.png"
            plt.savefig(overview_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots['overview'] = str(overview_path)
            
            # 2. 显著性结果热图
            if len(stats_df) > 0 and len(stats_df['roi_name'].unique()) > 1:
                pivot_data = stats_df.pivot(index='roi_name', columns='contrast_name', values='mean_accuracy')
                pivot_sig = stats_df.pivot(index='roi_name', columns='contrast_name', values='significant')
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 创建热图
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                           center=0.5, ax=ax, cbar_kws={'label': 'Mean Accuracy'})
                
                # 添加显著性标记
                for i in range(len(pivot_sig.index)):
                    for j in range(len(pivot_sig.columns)):
                        if pivot_sig.iloc[i, j]:
                            ax.text(j+0.5, i+0.5, '*', ha='center', va='center', 
                                   color='white', fontsize=20, fontweight='bold')
                
                ax.set_title('MVPA Results Heatmap\n(* indicates significant after correction)')
                plt.tight_layout()
                
                heatmap_path = viz_dir / "significance_heatmap.png"
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                saved_plots['heatmap'] = str(heatmap_path)
            
            # 3. 效应量图
            if len(stats_df) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 按效应量排序
                stats_sorted = stats_df.sort_values('cohens_d', ascending=True)
                
                colors = ['red' if sig else 'gray' for sig in stats_sorted['significant']]
                
                bars = ax.barh(range(len(stats_sorted)), stats_sorted['cohens_d'], color=colors, alpha=0.7)
                
                ax.set_yticks(range(len(stats_sorted)))
                ax.set_yticklabels([f"{row['roi_name']}_{row['contrast_name']}" 
                                   for _, row in stats_sorted.iterrows()])
                ax.set_xlabel("Cohen's d")
                ax.set_title("Effect Sizes (Cohen's d)\nRed = Significant after correction")
                ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                
                # 添加效应量解释线
                for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
                    ax.axvline(threshold, color='blue', linestyle='--', alpha=0.5)
                    ax.axvline(-threshold, color='blue', linestyle='--', alpha=0.5)
                    ax.text(threshold, len(stats_sorted), label, ha='center', va='bottom', 
                           color='blue', fontsize=8)
                
                plt.tight_layout()
                
                effect_path = viz_dir / "effect_sizes.png"
                plt.savefig(effect_path, dpi=300, bbox_inches='tight')
                plt.close()
                saved_plots['effect_sizes'] = str(effect_path)
            
            self.log(f"可视化完成，保存了 {len(saved_plots)} 个图表")
            
        except Exception as e:
            self.log(f"可视化创建失败: {e}")
        
        return saved_plots
    
    def generate_html_report(self, group_df: pd.DataFrame, stats_df: pd.DataFrame, 
                           visualizations: Dict[str, str]) -> str:
        """生成HTML报告"""
        self.log("生成HTML报告")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced MVPA Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .significant {{ background-color: #ffeb3b; }}
                .image {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced MVPA Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis Version: {self.config.version}</p>
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <ul>
                    <li>Total Subjects: {len(group_df['subject'].unique()) if len(group_df) > 0 else 0}</li>
                    <li>Total ROIs: {len(group_df['roi_name'].unique()) if len(group_df) > 0 else 0}</li>
                    <li>Total Contrasts: {len(group_df['contrast_name'].unique()) if len(group_df) > 0 else 0}</li>
                    <li>Significant Results: {len(stats_df[stats_df['significant'] == True]) if len(stats_df) > 0 else 0}</li>
                </ul>
            </div>
        """
        
        # 添加可视化
        if visualizations:
            html_content += '<div class="section"><h2>Visualizations</h2>'
            for name, path in visualizations.items():
                rel_path = Path(path).name
                html_content += f'<div class="image"><h3>{name.title()}</h3><img src="visualizations/{rel_path}" style="max-width: 100%; height: auto;"></div>'
            html_content += '</div>'
        
        # 添加统计结果表
        if len(stats_df) > 0:
            html_content += '<div class="section"><h2>Statistical Results</h2>'
            html_content += '<table class="table"><tr>'
            
            # 表头
            for col in ['roi_name', 'contrast_name', 'n_subjects', 'mean_accuracy', 'p_value', 'p_corrected', 'cohens_d', 'significant']:
                html_content += f'<th>{col.replace("_", " ").title()}</th>'
            html_content += '</tr>'
            
            # 数据行
            for _, row in stats_df.iterrows():
                row_class = 'significant' if row['significant'] else ''
                html_content += f'<tr class="{row_class}">'
                html_content += f'<td>{row["roi_name"]}</td>'
                html_content += f'<td>{row["contrast_name"]}</td>'
                html_content += f'<td>{row["n_subjects"]}</td>'
                html_content += f'<td>{row["mean_accuracy"]:.3f}</td>'
                html_content += f'<td>{row["p_value"]:.4f}</td>'
                html_content += f'<td>{row["p_corrected"]:.4f}</td>'
                html_content += f'<td>{row["cohens_d"]:.3f}</td>'
                html_content += f'<td>{"Yes" if row["significant"] else "No"}</td>'
                html_content += '</tr>'
            
            html_content += '</table></div>'
        
        html_content += """
            <div class="section">
                <h2>Analysis Parameters</h2>
                <ul>
                    <li>Cross-validation folds: {cv_folds}</li>
                    <li>Permutations: {n_permutations}</li>
                    <li>Correction method: {correction_method}</li>
                    <li>Alpha level: {alpha_level}</li>
                </ul>
            </div>
        </body>
        </html>
        """.format(
            cv_folds=self.mvpa_config.cv_folds,
            n_permutations=self.mvpa_config.n_permutations,
            correction_method=self.mvpa_config.correction_method,
            alpha_level=self.mvpa_config.alpha_level
        )
        
        # 保存HTML报告
        report_path = Path(self.mvpa_config.results_dir) / "mvpa_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log(f"HTML报告已保存: {report_path}")
        return str(report_path)
    
    def run_group_roi_analysis_enhanced(self) -> Dict[str, Any]:
        """运行增强版组水平ROI分析"""
        self.performance_stats['start_time'] = datetime.now()
        self.log("开始增强版MVPA分析")
        
        try:
            # 加载ROI masks
            roi_masks = self.load_roi_masks()
            
            if not roi_masks:
                raise ValueError("没有找到ROI masks")
            
            # 分析每个被试
            all_results = []
            
            for subject in self.mvpa_config.subjects:
                try:
                    subject_results = self.analyze_single_subject_roi_enhanced(
                        subject, self.mvpa_config.runs, self.mvpa_config.contrasts, roi_masks
                    )
                    all_results.append(subject_results)
                    
                except Exception as e:
                    self.log(f"被试 {subject} 分析失败: {e}")
                    self.performance_stats['failed_analyses'].append((subject, 'all_rois', str(e)))
                    continue
            
            if not all_results:
                raise ValueError("没有成功的分析结果")
            
            # 整理组水平数据
            group_data = []
            
            for subject_result in all_results:
                subject = subject_result['subject']
                
                for roi_name, roi_result in subject_result['roi_results'].items():
                    if not roi_result['success']:
                        continue
                    
                    for contrast_name, contrast_result in roi_result['contrasts'].items():
                        if not contrast_result.get('success', False):
                            continue
                        
                        group_data.append({
                            'subject': subject,
                            'roi_name': roi_name,
                            'contrast_name': contrast_name,
                            'mean_accuracy': contrast_result['mean_accuracy'],
                            'std_accuracy': contrast_result['std_accuracy'],
                            'p_value': contrast_result['p_value'],
                            'n_samples': contrast_result['n_samples'],
                            'n_features': contrast_result['n_features']
                        })
            
            if not group_data:
                raise ValueError("没有有效的组水平数据")
            
            group_df = pd.DataFrame(group_data)
            
            # 执行组水平统计
            stats_df = self.perform_group_statistics_enhanced(group_df)
            
            # 保存结果
            results_dir = Path(self.mvpa_config.results_dir)
            group_df.to_csv(results_dir / "group_results.csv", index=False)
            stats_df.to_csv(results_dir / "statistical_results.csv", index=False)
            
            # 创建可视化
            visualizations = self.create_enhanced_visualizations(group_df, stats_df)
            
            # 生成HTML报告
            html_report = self.generate_html_report(group_df, stats_df, visualizations)
            
            # 保存配置
            config_path = results_dir / "analysis_config.json"
            self.mvpa_config.save_config(str(config_path))
            
            self.performance_stats['end_time'] = datetime.now()
            
            # 生成最终总结
            summary = {
                'total_subjects': len(self.mvpa_config.subjects),
                'successful_subjects': len(all_results),
                'total_rois': len(roi_masks),
                'total_analyses': len(group_data),
                'significant_results': len(stats_df[stats_df['significant'] == True]) if len(stats_df) > 0 else 0,
                'duration': str(self.performance_stats['end_time'] - self.performance_stats['start_time']),
                'failed_analyses': len(self.performance_stats['failed_analyses'])
            }
            
            self.log(f"\n{'='*60}")
            self.log(f"MVPA分析完成")
            self.log(f"成功被试: {summary['successful_subjects']}/{summary['total_subjects']}")
            self.log(f"总分析数: {summary['total_analyses']}")
            self.log(f"显著结果: {summary['significant_results']}")
            self.log(f"总耗时: {summary['duration']}")
            self.log(f"{'='*60}")
            
            return {
                'group_results': group_df,
                'statistical_results': stats_df,
                'visualizations': visualizations,
                'html_report': html_report,
                'summary': summary,
                'performance_stats': self.performance_stats
            }
            
        except Exception as e:
            self.log(f"MVPA分析失败: {e}")
            raise

def run_enhanced_mvpa_analysis(config) -> Dict[str, Any]:
    """运行增强版MVPA分析的便捷函数"""
    analyzer = EnhancedMVPAAnalyzer(config)
    return analyzer.run_group_roi_analysis_enhanced()

if __name__ == "__main__":
    from .global_config import GlobalConfig
    
    # 创建配置
    config = GlobalConfig.create_default_config()
    
    # 运行分析
    results = run_enhanced_mvpa_analysis(config)
    print("增强版MVPA分析完成")
    print(f"显著结果: {results['summary']['significant_results']}")
    print(f"HTML报告: {results['html_report']}")