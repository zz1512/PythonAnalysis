#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MVPA Searchlight分析脚本
专注于隐喻vs空间分类的灰质脑区分析

主要功能:
1. 加载LSS数据
2. 在灰质区域进行searchlight分析
3. 识别分类准确的脑区
4. 生成脑区地图和报告

优化特点:
- 代码简洁，易于理解和维护
- 专注核心功能，移除冗余代码
- 优化内存使用和性能
- 清晰的错误处理和日志
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime
import warnings

# 科学计算库
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.ndimage import label

# 神经影像库
from nilearn.decoding import SearchLight
from nilearn.maskers import NiftiMasker
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_glass_brain

# 本地模块
from optimized_config import OptimizedMVPAConfig, QUICK_TEST_CONFIG, FULL_ANALYSIS_CONFIG

# 设置环境变量优化
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

class SimplifiedSearchlightAnalyzer:
    """简化的Searchlight分析器"""
    
    def __init__(self, config: OptimizedMVPAConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 结果存储
        self.results = {
            'subject_results': [],
            'group_results': None,
            'significant_clusters': [],
            'analysis_summary': {}
        }
    
    def load_subject_data(self, subject_id: str) -> Tuple[Optional[List[str]], Optional[pd.DataFrame]]:
        """加载单个被试的LSS数据"""
        self.logger.info(f"加载被试 {subject_id} 的数据")
        
        all_beta_images = []
        all_trial_info = []
        
        for run in self.config.runs:
            # LSS数据目录
            lss_dir = self.config.lss_root / subject_id / f"run-{run}_LSS"
            
            if not lss_dir.exists():
                self.logger.warning(f"LSS目录不存在: {lss_dir}")
                continue
            
            # 加载trial信息
            trial_info_path = lss_dir / "trial_info.csv"
            if not trial_info_path.exists():
                self.logger.warning(f"trial_info.csv不存在: {trial_info_path}")
                continue
            
            trial_info = pd.read_csv(trial_info_path)
            
            # 加载beta图像
            beta_images = []
            valid_trials = []
            
            for _, trial in trial_info.iterrows():
                beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
                
                if beta_path.exists():
                    beta_images.append(str(beta_path))
                    trial_copy = trial.copy()
                    trial_copy['run'] = run
                    trial_copy['subject'] = subject_id
                    valid_trials.append(trial_copy)
            
            if beta_images:
                all_beta_images.extend(beta_images)
                all_trial_info.extend(valid_trials)
                self.logger.info(f"  run {run}: 加载了 {len(beta_images)} 个trial")
        
        if not all_beta_images:
            self.logger.error(f"被试 {subject_id} 没有找到有效数据")
            return None, None
        
        combined_trial_info = pd.DataFrame(all_trial_info)
        self.logger.info(f"被试 {subject_id} 总共加载 {len(all_beta_images)} 个trial")
        
        return all_beta_images, combined_trial_info
    
    def prepare_classification_data(self, trial_info: pd.DataFrame, contrast: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """准备分类数据"""
        contrast_name, cond1, cond2 = contrast
        
        # 筛选目标条件的trial
        cond1_trials = trial_info[trial_info['original_condition'] == cond1]
        cond2_trials = trial_info[trial_info['original_condition'] == cond2]
        
        self.logger.info(f"条件 {cond1}: {len(cond1_trials)} trials")
        self.logger.info(f"条件 {cond2}: {len(cond2_trials)} trials")
        
        # 检查数据充足性
        if len(cond1_trials) < self.config.min_trials_per_condition or len(cond2_trials) < self.config.min_trials_per_condition:
            raise ValueError(f"数据不足: {cond1}={len(cond1_trials)}, {cond2}={len(cond2_trials)}")
        
        # 平衡数据（如果启用）
        if self.config.balance_trials:
            min_trials = min(len(cond1_trials), len(cond2_trials))
            cond1_trials = cond1_trials.sample(n=min_trials, random_state=self.config.cv_random_state)
            cond2_trials = cond2_trials.sample(n=min_trials, random_state=self.config.cv_random_state)
            self.logger.info(f"平衡后每个条件 {min_trials} trials")
        
        # 合并数据
        selected_trials = pd.concat([cond1_trials, cond2_trials], ignore_index=True)
        
        # 创建标签
        labels = np.array([0] * len(cond1_trials) + [1] * len(cond2_trials))
        
        # 获取trial索引
        trial_indices = selected_trials.index.values
        
        return trial_indices, labels
    
    def run_searchlight_classification(self, beta_images: List[str], trial_indices: np.ndarray, 
                                     labels: np.ndarray) -> Tuple[nib.Nifti1Image, Dict]:
        """运行searchlight分类分析"""
        self.logger.info("开始Searchlight分类分析")
        
        # 选择对应的beta图像
        selected_beta_images = [beta_images[i] for i in trial_indices]
        
        # 加载灰质mask
        mask_path = self.config.get_gray_matter_mask_path()
        self.logger.info(f"使用灰质mask: {mask_path}")
        
        # 创建分类器pipeline
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**self.config.svm_params))
        ])
        
        # 创建交叉验证策略
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )
        
        # 创建Searchlight对象
        searchlight = SearchLight(
            mask_img=str(mask_path),
            radius=self.config.searchlight_radius,
            estimator=classifier,
            cv=cv,
            scoring='accuracy',
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
        # 拟合Searchlight
        self.logger.info("拟合Searchlight模型...")
        searchlight.fit(selected_beta_images, labels)
        
        # 获取准确率地图
        accuracy_img = searchlight.scores_
        
        # 计算统计信息
        accuracy_data = accuracy_img.get_fdata()
        mask_data = nib.load(mask_path).get_fdata()
        
        # 只在mask内计算统计
        masked_accuracy = accuracy_data[mask_data > 0]
        
        stats_info = {
            'mean_accuracy': np.mean(masked_accuracy),
            'std_accuracy': np.std(masked_accuracy),
            'max_accuracy': np.max(masked_accuracy),
            'min_accuracy': np.min(masked_accuracy),
            'above_chance': np.sum(masked_accuracy > self.config.chance_level),
            'total_voxels': len(masked_accuracy)
        }
        
        self.logger.info(f"Searchlight完成 - 平均准确率: {stats_info['mean_accuracy']:.3f}")
        
        return accuracy_img, stats_info
    
    def identify_significant_clusters(self, accuracy_img: nib.Nifti1Image, 
                                    threshold: float = 0.55) -> List[Dict]:
        """识别显著的分类准确区域"""
        self.logger.info(f"识别显著簇 (阈值: {threshold})")
        
        accuracy_data = accuracy_img.get_fdata()
        
        # 创建二值mask
        significant_mask = accuracy_data > threshold
        
        # 连通性分析
        labeled_array, num_clusters = label(significant_mask)
        
        clusters = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = labeled_array == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size >= self.config.min_region_size:
                # 计算簇的统计信息
                cluster_accuracies = accuracy_data[cluster_mask]
                
                # 找到峰值坐标
                peak_idx = np.unravel_index(
                    np.argmax(accuracy_data * cluster_mask),
                    accuracy_data.shape
                )
                
                # 转换为MNI坐标
                peak_mni = nib.affines.apply_affine(
                    accuracy_img.affine,
                    peak_idx
                )
                
                cluster_info = {
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'peak_accuracy': np.max(cluster_accuracies),
                    'mean_accuracy': np.mean(cluster_accuracies),
                    'peak_mni': peak_mni,
                    'peak_voxel': peak_idx
                }
                
                clusters.append(cluster_info)
        
        # 按峰值准确率排序
        clusters.sort(key=lambda x: x['peak_accuracy'], reverse=True)
        
        self.logger.info(f"发现 {len(clusters)} 个显著簇")
        
        return clusters
    
    def process_single_subject(self, subject_id: str) -> Optional[Dict]:
        """处理单个被试"""
        self.logger.info(f"\n=== 处理被试 {subject_id} ===")
        
        try:
            # 加载数据
            beta_images, trial_info = self.load_subject_data(subject_id)
            if beta_images is None:
                return None
            
            subject_results = {'subject_id': subject_id, 'contrasts': {}}
            
            # 处理每个对比
            for contrast in self.config.contrasts:
                contrast_name = contrast[0]
                self.logger.info(f"处理对比: {contrast_name}")
                
                try:
                    # 准备分类数据
                    trial_indices, labels = self.prepare_classification_data(trial_info, contrast)
                    
                    # 运行searchlight分析
                    accuracy_img, stats_info = self.run_searchlight_classification(
                        beta_images, trial_indices, labels
                    )
                    
                    # 识别显著簇
                    significant_clusters = self.identify_significant_clusters(accuracy_img)
                    
                    # 保存结果
                    subject_result_dir = self.config.results_dir / "subject_results" / subject_id
                    subject_result_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存准确率地图
                    accuracy_path = subject_result_dir / f"{contrast_name}_accuracy.nii.gz"
                    nib.save(accuracy_img, accuracy_path)
                    
                    # 保存簇信息
                    if significant_clusters:
                        clusters_df = pd.DataFrame(significant_clusters)
                        clusters_path = subject_result_dir / f"{contrast_name}_clusters.csv"
                        clusters_df.to_csv(clusters_path, index=False)
                    
                    subject_results['contrasts'][contrast_name] = {
                        'stats': stats_info,
                        'clusters': significant_clusters,
                        'accuracy_map': str(accuracy_path)
                    }
                    
                    self.logger.info(f"对比 {contrast_name} 完成")
                    
                except Exception as e:
                    self.logger.error(f"对比 {contrast_name} 失败: {e}")
                    continue
            
            return subject_results
            
        except Exception as e:
            self.logger.error(f"被试 {subject_id} 处理失败: {e}")
            return None
    
    def run_group_analysis(self, subject_results: List[Dict]) -> Dict:
        """运行组水平分析"""
        self.logger.info("\n=== 组水平分析 ===")
        
        if not subject_results:
            raise ValueError("没有有效的被试结果")
        
        group_results = {}
        
        for contrast in self.config.contrasts:
            contrast_name = contrast[0]
            self.logger.info(f"组分析: {contrast_name}")
            
            # 收集所有被试的准确率地图
            accuracy_maps = []
            all_clusters = []
            
            for subject_result in subject_results:
                if contrast_name in subject_result['contrasts']:
                    map_path = subject_result['contrasts'][contrast_name]['accuracy_map']
                    accuracy_maps.append(map_path)
                    
                    clusters = subject_result['contrasts'][contrast_name]['clusters']
                    for cluster in clusters:
                        cluster['subject'] = subject_result['subject_id']
                        all_clusters.append(cluster)
            
            if not accuracy_maps:
                self.logger.warning(f"对比 {contrast_name} 没有有效数据")
                continue
            
            # 计算组平均地图
            self.logger.info(f"计算组平均地图 ({len(accuracy_maps)} 个被试)")
            
            # 加载所有地图
            imgs = [nib.load(path) for path in accuracy_maps]
            
            # 计算平均
            mean_img = image.mean_img(imgs)
            
            # 单样本t检验
            # 创建与chance水平的差值图像
            chance_level = self.config.chance_level
            diff_imgs = [image.math_img(f'img - {chance_level}', img=img) for img in imgs]
            
            # 计算t统计量
            diff_data = np.array([img.get_fdata() for img in diff_imgs])
            
            # 只在mask区域计算
            mask_path = self.config.get_gray_matter_mask_path()
            mask_data = nib.load(mask_path).get_fdata()
            
            t_stats = np.zeros_like(mask_data)
            p_values = np.ones_like(mask_data)
            
            for i, j, k in np.ndindex(mask_data.shape):
                if mask_data[i, j, k] > 0:
                    voxel_data = diff_data[:, i, j, k]
                    if np.std(voxel_data) > 0:  # 避免除零
                        t_stat, p_val = stats.ttest_1samp(voxel_data, 0)
                        t_stats[i, j, k] = t_stat
                        p_values[i, j, k] = p_val
            
            # 创建统计地图
            t_img = nib.Nifti1Image(t_stats, mean_img.affine, mean_img.header)
            p_img = nib.Nifti1Image(p_values, mean_img.affine, mean_img.header)
            
            # FDR校正
            from scipy.stats import false_discovery_control
            p_flat = p_values[mask_data > 0]
            p_fdr = false_discovery_control(p_flat, alpha=self.config.alpha_level)
            
            # 创建显著性mask
            fdr_threshold = np.percentile(p_flat[p_fdr], 95) if np.any(p_fdr) else 0.05
            significant_mask = (p_values < fdr_threshold) & (mask_data > 0)
            
            # 保存组结果
            group_result_dir = self.config.results_dir / "statistical_maps"
            group_result_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存地图
            mean_path = group_result_dir / f"{contrast_name}_group_mean.nii.gz"
            t_path = group_result_dir / f"{contrast_name}_group_tstat.nii.gz"
            p_path = group_result_dir / f"{contrast_name}_group_pval.nii.gz"
            
            nib.save(mean_img, mean_path)
            nib.save(t_img, t_path)
            nib.save(p_img, p_path)
            
            # 识别组水平显著簇
            group_clusters = self.identify_significant_clusters(
                mean_img, 
                threshold=self.config.chance_level + 0.05
            )
            
            group_results[contrast_name] = {
                'mean_map': str(mean_path),
                't_map': str(t_path),
                'p_map': str(p_path),
                'group_clusters': group_clusters,
                'all_subject_clusters': all_clusters,
                'n_subjects': len(accuracy_maps),
                'fdr_threshold': fdr_threshold
            }
            
            self.logger.info(f"组分析 {contrast_name} 完成")
        
        return group_results
    
    def generate_brain_maps(self, group_results: Dict) -> Dict:
        """生成脑区地图可视化"""
        self.logger.info("\n=== 生成脑区地图 ===")
        
        import matplotlib.pyplot as plt
        
        visualization_results = {}
        
        for contrast_name, results in group_results.items():
            self.logger.info(f"生成 {contrast_name} 的脑区地图")
            
            # 创建可视化目录
            viz_dir = self.config.results_dir / "brain_maps" / contrast_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载统计地图
            mean_img = nib.load(results['mean_map'])
            t_img = nib.load(results['t_map'])
            
            # 生成不同视图的脑图
            views = ['lateral', 'medial', 'dorsal', 'ventral']
            
            for view in views:
                # 平均准确率地图
                fig = plt.figure(figsize=(12, 8))
                plot_stat_map(
                    mean_img,
                    threshold=self.config.chance_level + 0.02,
                    title=f'{contrast_name} - 组平均准确率 ({view})',
                    cut_coords=5,
                    display_mode=view[0] if view != 'dorsal' else 'z',
                    colorbar=True,
                    cmap='hot'
                )
                
                mean_map_path = viz_dir / f"mean_accuracy_{view}.png"
                plt.savefig(mean_map_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # t统计地图
                fig = plt.figure(figsize=(12, 8))
                plot_stat_map(
                    t_img,
                    threshold=2.0,  # t > 2
                    title=f'{contrast_name} - t统计地图 ({view})',
                    cut_coords=5,
                    display_mode=view[0] if view != 'dorsal' else 'z',
                    colorbar=True,
                    cmap='cold_hot'
                )
                
                t_map_path = viz_dir / f"t_stat_{view}.png"
                plt.savefig(t_map_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 玻璃脑图
            fig = plt.figure(figsize=(15, 5))
            plot_glass_brain(
                mean_img,
                threshold=self.config.chance_level + 0.02,
                title=f'{contrast_name} - 组平均准确率 (玻璃脑)',
                colorbar=True,
                cmap='hot'
            )
            
            glass_brain_path = viz_dir / "glass_brain.png"
            plt.savefig(glass_brain_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_results[contrast_name] = {
                'brain_maps_dir': str(viz_dir),
                'glass_brain': str(glass_brain_path)
            }
        
        return visualization_results
    
    def generate_summary_report(self, group_results: Dict, visualization_results: Dict) -> str:
        """生成分析摘要报告"""
        self.logger.info("\n=== 生成摘要报告 ===")
        
        report_path = self.config.results_dir / "reports" / "analysis_summary.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MVPA Searchlight 分析报告\n\n")
            f.write(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 配置信息
            f.write("## 分析配置\n\n")
            f.write(f"- **被试数量**: {len(self.config.subjects)}\n")
            f.write(f"- **分析run**: {self.config.runs}\n")
            f.write(f"- **Searchlight半径**: {self.config.searchlight_radius}mm\n")
            f.write(f"- **灰质mask**: {self.config.gray_matter_mask}\n")
            f.write(f"- **分类器**: 线性SVM (C={self.config.svm_params['C']})\n")
            f.write(f"- **交叉验证**: {self.config.cv_folds}折\n")
            f.write(f"- **显著性水平**: {self.config.alpha_level}\n\n")
            
            # 结果摘要
            f.write("## 分析结果\n\n")
            
            for contrast_name, results in group_results.items():
                f.write(f"### {contrast_name}\n\n")
                f.write(f"- **参与被试数**: {results['n_subjects']}\n")
                f.write(f"- **显著簇数量**: {len(results['group_clusters'])}\n")
                f.write(f"- **FDR阈值**: {results['fdr_threshold']:.4f}\n\n")
                
                # 显著簇详情
                if results['group_clusters']:
                    f.write("#### 显著分类准确区域\n\n")
                    f.write("| 簇ID | 大小(体素) | 峰值准确率 | 平均准确率 | MNI坐标(x,y,z) |\n")
                    f.write("|------|------------|------------|------------|----------------|\n")
                    
                    for cluster in results['group_clusters'][:10]:  # 只显示前10个
                        mni_str = f"({cluster['peak_mni'][0]:.0f}, {cluster['peak_mni'][1]:.0f}, {cluster['peak_mni'][2]:.0f})"
                        f.write(f"| {cluster['cluster_id']} | {cluster['size']} | {cluster['peak_accuracy']:.3f} | {cluster['mean_accuracy']:.3f} | {mni_str} |\n")
                    
                    f.write("\n")
                
                # 脑图路径
                if contrast_name in visualization_results:
                    f.write("#### 脑区地图\n\n")
                    f.write(f"- 脑图目录: `{visualization_results[contrast_name]['brain_maps_dir']}`\n")
                    f.write(f"- 玻璃脑图: `{visualization_results[contrast_name]['glass_brain']}`\n\n")
            
            # 文件路径
            f.write("## 输出文件\n\n")
            f.write(f"- **结果目录**: `{self.config.results_dir}`\n")
            f.write(f"- **统计地图**: `{self.config.results_dir}/statistical_maps/`\n")
            f.write(f"- **被试结果**: `{self.config.results_dir}/subject_results/`\n")
            f.write(f"- **脑区地图**: `{self.config.results_dir}/brain_maps/`\n")
        
        self.logger.info(f"摘要报告已保存: {report_path}")
        return str(report_path)
    
    def run_complete_analysis(self) -> Dict:
        """运行完整的分析流程"""
        self.logger.info("\n" + "="*50)
        self.logger.info("开始MVPA Searchlight分析")
        self.logger.info("="*50)
        
        # 验证配置
        if not self.config.validate_paths():
            raise ValueError("配置验证失败")
        
        # 处理所有被试
        subject_results = []
        for subject_id in self.config.subjects:
            result = self.process_single_subject(subject_id)
            if result:
                subject_results.append(result)
        
        if not subject_results:
            raise ValueError("没有成功处理的被试")
        
        self.logger.info(f"成功处理 {len(subject_results)} 个被试")
        
        # 组水平分析
        group_results = self.run_group_analysis(subject_results)
        
        # 生成脑区地图
        visualization_results = self.generate_brain_maps(group_results)
        
        # 生成报告
        report_path = self.generate_summary_report(group_results, visualization_results)
        
        # 汇总结果
        final_results = {
            'subject_results': subject_results,
            'group_results': group_results,
            'visualization_results': visualization_results,
            'report_path': report_path,
            'config': self.config
        }
        
        self.logger.info("\n" + "="*50)
        self.logger.info("分析完成！")
        self.logger.info(f"结果保存在: {self.config.results_dir}")
        self.logger.info(f"摘要报告: {report_path}")
        self.logger.info("="*50)
        
        return final_results

def main():
    """主函数"""
    # 选择配置
    print("选择分析配置:")
    print("1. 快速测试 (2个被试, 1个run)")
    print("2. 完整分析 (所有被试, 所有run)")
    print("3. 自定义配置")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        config = OptimizedMVPAConfig(QUICK_TEST_CONFIG)
    elif choice == '2':
        config = OptimizedMVPAConfig(FULL_ANALYSIS_CONFIG)
    elif choice == '3':
        # 自定义配置示例
        custom_config = {
            'subjects': [f"sub-{i:02d}" for i in range(1, 6)],  # 前5个被试
            'runs': [3, 4],
            'n_permutations': 200,
            'debug_mode': True
        }
        config = OptimizedMVPAConfig(custom_config)
    else:
        print("无效选择，使用快速测试配置")
        config = OptimizedMVPAConfig(QUICK_TEST_CONFIG)
    
    print("\n使用配置:")
    print(config.get_config_summary())
    
    # 确认开始分析
    confirm = input("\n确认开始分析? (y/N): ").strip().lower()
    if confirm != 'y':
        print("分析已取消")
        return
    
    # 创建分析器并运行
    analyzer = SimplifiedSearchlightAnalyzer(config)
    
    try:
        results = analyzer.run_complete_analysis()
        print(f"\n✅ 分析成功完成！")
        print(f"结果目录: {config.results_dir}")
        return results
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        logging.exception("详细错误信息:")
        return None

if __name__ == "__main__":
    main()