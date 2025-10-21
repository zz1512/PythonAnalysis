#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fMRI MVPA ROI Pipeline - 合并版本

这是将所有模块合并到一个文件中的版本，包含以下功能：
- 配置管理
- 工具函数
- 数据加载
- 特征提取
- 分类分析
- 组水平统计分析
- 可视化
- HTML报告生成
- 主分析流程

使用方法：
    python fmri_mvpa_roi_pipeline_merged.py

作者: AI Assistant
日期: 2024
"""

import os
import gc
import json
import base64
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ============================================================================
# 配置管理模块 (config.py)
# ============================================================================

class MVPAConfig:
    """MVPA分析配置类"""
    
    def __init__(self):
        # 基础路径配置
        self.base_dir = Path("/Users/bytedance/PycharmProjects/PythonAnalysis/learn/fmri_mvpa")
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results" / "roi_mvpa_results"
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 被试列表
        self.subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
        
        # run列表
        self.runs = [1, 2, 3, 4]
        
        # ROI配置
        self.roi_dir = self.data_dir / "rois"
        self.roi_pattern = "*.nii.gz"
        
        # LSS数据路径模板
        self.lss_dir_template = self.data_dir / "derivatives" / "lss" / "{subject}" / "func"
        self.lss_file_template = "{subject}_task-faces_run-{run:02d}_space-MNI152NLin2009cAsym_desc-lss_beta.nii.gz"
        self.trial_info_template = "{subject}_task-faces_run-{run:02d}_desc-lss_trial_info.csv"
        
        # 分类器参数
        self.svm_params = {
            'kernel': 'linear',
            'C': 1.0,
            'random_state': 42
        }
        
        # 参数网格搜索
        self.svm_param_grid = {
            'C': [0.1, 1.0, 10.0]
        }
        
        # 交叉验证参数
        self.cv_folds = 5
        self.cv_random_state = 42
        
        # 置换检验参数
        self.n_permutations = 1000
        self.permutation_random_state = 42
        
        # 统计参数
        self.alpha_level = 0.05
        
        # 对比条件
        self.contrasts = [
            ('faces_vs_objects', 'faces', 'objects'),
            ('faces_vs_scenes', 'faces', 'scenes'),
            ('objects_vs_scenes', 'objects', 'scenes')
        ]
        
        # 内存和缓存设置
        self.memory_cache = None
        self.memory_level = 1
        
        # 日志设置
        self.log_level = logging.INFO
        self.log_file = self.results_dir / "analysis.log"
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def save_config(self, filepath):
        """保存配置到JSON文件"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

# ============================================================================
# 工具函数模块 (utils.py)
# ============================================================================

def log(message, config):
    """记录日志信息"""
    logging.info(message)
    print(message)

def memory_cleanup():
    """内存清理"""
    gc.collect()

# ============================================================================
# 数据加载模块 (data_loader.py)
# ============================================================================

def load_roi_masks(config):
    """加载ROI mask文件"""
    roi_masks = {}
    roi_files = list(config.roi_dir.glob(config.roi_pattern))
    
    if not roi_files:
        log(f"未找到ROI文件: {config.roi_dir / config.roi_pattern}", config)
        return roi_masks
    
    for roi_file in roi_files:
        roi_name = roi_file.stem.replace('.nii', '')
        roi_masks[roi_name] = str(roi_file)
        log(f"加载ROI: {roi_name} -> {roi_file}", config)
    
    return roi_masks

def load_lss_trial_data(subject, run, config):
    """加载单个run的LSS trial数据和beta图像"""
    # 构建文件路径
    lss_dir = Path(str(config.lss_dir_template).format(subject=subject))
    
    # LSS beta文件路径
    beta_file = lss_dir / config.lss_file_template.format(subject=subject, run=run)
    
    # trial信息文件路径
    trial_info_file = lss_dir / config.trial_info_template.format(subject=subject, run=run)
    
    # 检查文件是否存在
    if not beta_file.exists():
        log(f"  LSS beta文件不存在: {beta_file}", config)
        return None, None
    
    if not trial_info_file.exists():
        log(f"  trial信息文件不存在: {trial_info_file}", config)
        return None, None
    
    try:
        # 加载trial信息
        trial_info = pd.read_csv(trial_info_file)
        
        # 验证trial信息
        if trial_info.empty:
            log(f"  trial信息为空: {trial_info_file}", config)
            return None, None
        
        # 构建beta图像路径列表
        beta_images = []
        for trial_idx in trial_info['trial_index']:
            # LSS beta文件命名约定：原文件名_trial-{trial_idx}.nii.gz
            trial_beta_file = beta_file.parent / f"{beta_file.stem}_trial-{trial_idx:03d}.nii.gz"
            beta_images.append(str(trial_beta_file))
        
        log(f"  成功加载run {run}: {len(trial_info)} trials, {len(beta_images)} beta文件", config)
        
        return beta_images, trial_info
        
    except Exception as e:
        log(f"  加载run {run}数据失败: {str(e)}", config)
        return None, None

def load_multi_run_lss_data(subject, runs, config):
    """加载多个run的LSS数据"""
    log(f"加载被试 {subject} 的多run LSS数据", config)
    
    all_beta_images = []
    all_trial_info = []
    global_trial_index = 0
    
    for run in runs:
        log(f"  加载run {run}...", config)
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        
        if beta_images is not None and trial_info is not None:
            # 为每个trial添加全局索引和run信息
            trial_info = trial_info.copy()
            trial_info['run'] = run
            trial_info['combined_trial_index'] = range(global_trial_index, global_trial_index + len(trial_info))
            
            all_trial_info.append(trial_info)
            all_beta_images.extend(beta_images)
            global_trial_index += len(trial_info)
        else:
            log(f"  警告: run {run} 数据加载失败", config)
    
    if not all_trial_info:
        log(f"错误: 被试 {subject} 没有成功加载任何run的数据", config)
        return None, None
    
    # 合并所有run的数据
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    
    log(f"多run数据合并完成: 总共 {len(combined_trial_info)} 个trial, {len(all_beta_images)} 个beta文件", config)
    
    # 验证数据一致性
    if len(combined_trial_info) != len(all_beta_images):
        log(f"警告: trial数量({len(combined_trial_info)})与beta文件数量({len(all_beta_images)})不匹配!", config)
    
    # 打印每个条件的trial数量统计
    if 'original_condition' in combined_trial_info.columns:
        condition_counts = combined_trial_info['original_condition'].value_counts()
        log(f"  条件统计: {dict(condition_counts)}", config)
    
    return all_beta_images, combined_trial_info

# ============================================================================
# 特征提取模块 (feature_extraction.py)
# ============================================================================

def extract_roi_features(beta_images, roi_mask_path, config):
    """从beta图像中提取ROI特征"""
    try:
        # 使用NiftiMasker进行特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )
        
        # 提取特征
        X = masker.fit_transform(beta_images)
        
        return X
        
    except Exception as e:
        log(f"ROI特征提取失败: {str(e)}", config)
        return None

# ============================================================================
# 分类分析模块 (classification.py)
# ============================================================================

def run_roi_classification_enhanced(X, y, config):
    """增强的ROI分类分析"""
    # 创建基础分类器
    base_svm = SVC(random_state=config.svm_params['random_state'])
    
    # 创建分类pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', base_svm)
    ])
    
    # 交叉验证
    cv = StratifiedKFold(
        n_splits=config.cv_folds, 
        shuffle=True, 
        random_state=config.cv_random_state
    )
    
    # 参数网格搜索
    param_grid = {'svm__' + key: value for key, value in config.svm_param_grid.items()}
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=1  # 避免嵌套并行
    )
    
    # 拟合网格搜索
    grid_search.fit(X, y)
    
    # 使用最佳参数进行交叉验证
    best_pipeline = grid_search.best_estimator_
    cv_scores = cross_val_score(best_pipeline, X, y, cv=cv, scoring='accuracy')
    
    # 根据专家建议：单被试不做置换检验，只使用交叉验证结果
    observed_score = np.mean(cv_scores)
    
    # 不进行置换检验，设置默认值
    null_distribution = np.array([])
    p_value = np.nan  # 单被试不计算p值
    
    return {
        'cv_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'p_value_permutation': p_value,  # 被试内置换检验p值
        'null_distribution': null_distribution,
        'chance_level': 0.5,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

def prepare_classification_data(trial_info, beta_images, cond1, cond2, config):
    """准备分类数据"""
    # 筛选特定条件的trial - 使用original_condition列
    condition_col = 'original_condition' if 'original_condition' in trial_info.columns else 'condition'
    cond1_trials = trial_info[trial_info[condition_col] == cond1]
    cond2_trials = trial_info[trial_info[condition_col] == cond2]
    
    if len(cond1_trials) == 0 or len(cond2_trials) == 0:
        log(f"条件数据不足: {cond1}={len(cond1_trials)}, {cond2}={len(cond2_trials)}", config)
        return None, None, None
    
    log(f"准备分类数据: {cond1} vs {cond2} (总beta文件: {len(beta_images)}, trial数: {len(trial_info)})", config)
    
    X_indices = []
    y_labels = []
    trial_details = []
    
    # 条件1的trial
    for i, (_, trial) in enumerate(cond1_trials.iterrows()):
        # 对于多run合并数据，使用combined_trial_index；对于单run数据，使用trial_index-1
        if 'combined_trial_index' in trial:
            trial_idx = trial['combined_trial_index']
            index_type = 'combined_trial_index'
        else:
            trial_idx = trial['trial_index'] - 1
            index_type = 'trial_index-1'
            
        if trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(0)
            trial_detail = {
                'trial_index': trial['trial_index'],
                'condition': cond1,
                'run': trial.get('run', 'unknown'),
                'beta_path': beta_images[trial_idx]
            }
            trial_details.append(trial_detail)
            
            # 只记录前3个trial的详细对应关系
            if i < 3:
                log(f"  {cond1}[{i+1}]: trial_index={trial['trial_index']}, {index_type}={trial_idx}, "
                    f"run={trial.get('run', 'unknown')}", config)
        else:
            log(f"  警告: trial_idx={trial_idx} 超出beta_images范围 (len={len(beta_images)})", config)
    
    # 条件2的trial
    for i, (_, trial) in enumerate(cond2_trials.iterrows()):
        if 'combined_trial_index' in trial:
            trial_idx = trial['combined_trial_index']
            index_type = 'combined_trial_index'
        else:
            trial_idx = trial['trial_index'] - 1
            index_type = 'trial_index-1'
            
        if trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(1)
            trial_detail = {
                'trial_index': trial['trial_index'],
                'condition': cond2,
                'run': trial.get('run', 'unknown'),
                'beta_path': beta_images[trial_idx]
            }
            trial_details.append(trial_detail)
            
            # 只记录前3个trial的详细对应关系
            if i < 3:
                log(f"  {cond2}[{i+1}]: trial_index={trial['trial_index']}, {index_type}={trial_idx}, "
                    f"run={trial.get('run', 'unknown')}", config)
        else:
            log(f"  警告: trial_idx={trial_idx} 超出beta_images范围 (len={len(beta_images)})", config)
    
    if len(X_indices) == 0:
        log(f"没有有效的trial数据用于分类", config)
        return None, None, None
    
    # 获取对应的beta图像路径
    selected_beta_images = [beta_images[i] for i in X_indices]
    
    # 汇总日志
    cond1_count = sum(1 for y in y_labels if y == 0)
    cond2_count = sum(1 for y in y_labels if y == 1)
    log(f"分类数据准备完成: {cond1}={cond1_count}, {cond2}={cond2_count}, 总计{len(selected_beta_images)}个beta文件", config)
    
    # 检查是否有重复的beta文件
    unique_betas = set(selected_beta_images)
    if len(unique_betas) != len(selected_beta_images):
        log(f"警告: 发现重复的beta文件! 唯一文件数: {len(unique_betas)}, 总选择数: {len(selected_beta_images)}", config)
    
    return selected_beta_images, np.array(y_labels), pd.DataFrame(trial_details)

# ============================================================================
# 组水平统计分析模块 (group_statistics.py)
# ============================================================================

def permutation_test_group_level(group_accuracies, n_permutations=1000, random_state=None):
    """组水平置换检验 - 修正版本，解决随机种子问题"""
    # 只有在random_state不为None时才设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 观察到的组平均准确率
    observed_mean = np.mean(group_accuracies)
    n_subjects = len(group_accuracies)
    
    # 数据质量检查
    if np.any(group_accuracies > 1.0) or np.any(group_accuracies < 0.0):
        print(f"⚠️  警告: 准确率数据超出合理范围 [0,1]: min={np.min(group_accuracies):.3f}, max={np.max(group_accuracies):.3f}")
    
    if np.mean(group_accuracies) > 0.95:
        print(f"⚠️  警告: 平均准确率过高 ({np.mean(group_accuracies):.3f})，可能存在过拟合或数据泄露")
    
    # 修正的置换策略：通过对符号进行随机翻转来构建零分布
    null_distribution = []
    
    # 将数据以0.5为中心
    data_centered = group_accuracies - 0.5
    
    for _ in range(n_permutations):
        # 随机翻转符号 - 这是正确的置换方法
        random_signs = np.random.choice([-1, 1], size=n_subjects, replace=True)
        permuted_accuracies = 0.5 + random_signs * data_centered
        null_distribution.append(np.mean(permuted_accuracies))
    
    null_distribution = np.array(null_distribution)
    
    # 计算p值（单尾检验，因为我们预期准确率高于随机水平）
    if observed_mean > 0.5:
        p_value = np.mean(null_distribution >= observed_mean)
    else:
        p_value = np.mean(null_distribution <= observed_mean)
    
    # 修正：如果p值为0，使用保守估计
    if p_value == 0:
        p_value = 1.0 / (n_permutations + 1)  # 保守估计
        print(f"📊 注意: 使用保守估计 p = 1/(n_permutations+1) = {p_value:.6f}")
    
    # 添加诊断信息
    extreme_count = np.sum(null_distribution >= observed_mean) if observed_mean > 0.5 else np.sum(null_distribution <= observed_mean)
    print(f"📈 置换检验诊断: 观察值={observed_mean:.4f}, 零分布范围=[{np.min(null_distribution):.4f}, {np.max(null_distribution):.4f}], 极端值数量={extreme_count}/{n_permutations}")
    
    return p_value, null_distribution

def perform_group_statistics_corrected(group_df, config):
    """执行修正的组水平统计分析"""
    log("开始组水平统计分析", config)
    
    stats_results = []
    
    # 按对比和ROI分组
    for (contrast, roi), group_data in group_df.groupby(['contrast', 'roi']):
        accuracies = group_data['mean_accuracy'].values
        n_subjects = len(accuracies)
        
        if n_subjects < 3:
            log(f"跳过 {contrast}-{roi}: 被试数量不足 (n={n_subjects})", config)
            continue
        
        # 基本统计量
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        sem_acc = std_acc / np.sqrt(n_subjects)
        
        # 数据质量检查
        if np.any(accuracies > 1.0) or np.any(accuracies < 0.0):
            log(f"警告 {contrast}-{roi}: 准确率数据超出合理范围 [0,1]", config)
            accuracies = np.clip(accuracies, 0.0, 1.0)  # 限制在合理范围内
        
        # 单样本t检验（vs 0.5随机水平）
        t_stat, t_p_value = stats.ttest_1samp(accuracies, 0.5)
        
        # 计算单尾t检验p值
        t_p_value_one_tailed = t_p_value / 2 if mean_acc > 0.5 else 1 - t_p_value / 2
        
        # 修正：如果t检验p值为0，使用更精确的计算
        if t_p_value == 0:
            # 使用t分布的累积分布函数
            from scipy.stats import t as t_dist
            t_p_value_precise = 2 * (1 - t_dist.cdf(abs(t_stat), n_subjects - 1))
            t_p_value_one_tailed_precise = 1 - t_dist.cdf(abs(t_stat), n_subjects - 1) if mean_acc > 0.5 else t_dist.cdf(-abs(t_stat), n_subjects - 1)
            log(f"使用精确t检验计算: {contrast}-{roi}, 精确双尾p={t_p_value_precise:.2e}, 精确单尾p={t_p_value_one_tailed_precise:.2e}", config)
            t_p_value = t_p_value_precise
            t_p_value_one_tailed = t_p_value_one_tailed_precise
        
        # Cohen's d效应量
        cohens_d = (mean_acc - 0.5) / std_acc
        
        # 95%置信区间
        ci_margin = stats.t.ppf(0.975, n_subjects - 1) * sem_acc
        ci_lower = mean_acc - ci_margin
        ci_upper = mean_acc + ci_margin
        
        # 为每个ROI-对比组合生成不同的随机种子
        roi_contrast_seed = hash(f"{contrast}_{roi}") % (2**31) + config.permutation_random_state
        
        # 组水平置换检验（现在使用单尾检验）
        p_permutation, null_dist = permutation_test_group_level(
            accuracies, 
            n_permutations=config.n_permutations,
            random_state=roi_contrast_seed
        )
        
        # 效应量分类
        effect_size_category = "小" if abs(cohens_d) < 0.2 else "中" if abs(cohens_d) < 0.8 else "大"
        
        result = {
            'contrast': contrast,
            'roi': roi,
            'n_subjects': n_subjects,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'sem_accuracy': sem_acc,
            't_statistic': t_stat,
            'p_value_ttest_two_tailed': t_p_value,  # 双尾t检验p值
            'p_value_ttest_one_tailed': t_p_value_one_tailed,  # 单尾t检验p值
            'p_value_permutation': p_permutation,  # 单尾置换检验p值
            'cohens_d': cohens_d,
            'effect_size_category': effect_size_category,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'null_distribution_mean': np.mean(null_dist),
            'null_distribution_std': np.std(null_dist),
            'null_distribution_range': f"[{np.min(null_dist):.4f}, {np.max(null_dist):.4f}]"
        }
        
        stats_results.append(result)
        
        # 判断显著性（基于置换检验）
        is_significant = p_permutation < config.alpha_level
        
        # 详细的结果报告
        log(f"\n=== 组统计结果 {contrast}-{roi} ===", config)
        log(f"📊 基本统计: 平均准确率={mean_acc:.4f} (±{std_acc:.4f}), n={n_subjects}", config)
        log(f"📈 t检验: t({n_subjects-1})={t_stat:.3f}, 双尾p={t_p_value:.2e}, 单尾p={t_p_value_one_tailed:.2e}", config)
        log(f"🔄 置换检验: p={p_permutation:.6f} (基于{config.n_permutations}次置换)", config)
        log(f"📏 效应量: Cohen's d={cohens_d:.3f} ({effect_size_category}效应)", config)
        log(f"🎯 95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]", config)
        log(f"🔍 零分布: 均值={np.mean(null_dist):.4f}, 范围={result['null_distribution_range']}", config)
        log(f"✅ 显著性 (α={config.alpha_level}): {'显著' if is_significant else '不显著'}", config)
        
        # 质量诊断
        if mean_acc > 0.95:
            log(f"⚠️  质量警告: 准确率过高，建议检查是否存在过拟合或数据泄露", config)
        if p_permutation == 1.0 / (config.n_permutations + 1):
            log(f"📊 注意: 使用了保守p值估计，建议增加置换次数", config)
    
    # 转换为DataFrame
    stats_df = pd.DataFrame(stats_results)
    
    if stats_df.empty:
        log("警告：没有生成任何统计结果", config)
        return stats_df
    
    # 使用置换检验p值作为最终显著性判断（单尾检验）
    stats_df['significant'] = stats_df['p_value_permutation'] < config.alpha_level
    
    # 报告结果 - 明确说明使用单尾检验
    significant_results = stats_df[stats_df['significant']]
    
    log(f"\n组水平置换检验显著结果数(单尾检验): {len(significant_results)}/{len(stats_df)}", config)
    
    if not significant_results.empty:
        log("\n显著结果(单尾置换检验):", config)
        for _, result in significant_results.iterrows():
            log(f"  {result['contrast']} - {result['roi']}: "
                f"accuracy = {result['mean_accuracy']:.3f}, "
                f"t({result['n_subjects'] - 1}) = {result['t_statistic']:.3f}, "
                f"p_t(单尾) = {result['p_value_ttest_one_tailed']:.4f}, "
                f"p_permutation(单尾) = {result['p_value_permutation']:.4f}, "
                f"d = {result['cohens_d']:.3f}", config)
    else:
        log("\n无显著结果", config)
    
    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistical_results_corrected.csv", index=False)
    
    return stats_df

# ============================================================================
# 可视化模块 (visualization.py)
# ============================================================================

def setup_chinese_fonts():
    """设置中文字体，增加跨平台兼容性"""
    # 尝试多种字体，提高兼容性
    font_candidates = [
        'SimHei',           # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'PingFang SC',      # macOS 苹方
        'Hiragino Sans GB', # macOS 冬青黑体
        'Arial Unicode MS', # 跨平台 Unicode 字体
        'WenQuanYi Micro Hei', # Linux 文泉驿微米黑
        'DejaVu Sans',      # Linux 默认字体
        'sans-serif'        # 系统默认无衬线字体
    ]
    
    font_found = False
    for font_name in font_candidates:
        try:
            # 检查字体是否可用
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if font_name in available_fonts or font_name == 'sans-serif':
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"成功设置字体: {font_name}")
                font_found = True
                break
        except Exception as e:
            print(f"字体 {font_name} 设置失败: {e}")
            continue
    
    if not font_found:
        print("警告：未找到合适的中文字体，可能影响中文显示")
        # 使用默认设置
        plt.rcParams['axes.unicode_minus'] = False
    
    return font_found

def create_enhanced_visualizations(group_df, stats_df, config):
    """创建增强的可视化"""
    log("创建可视化", config)
    
    # 设置中文字体
    setup_chinese_fonts()
    
    # 设置图形样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建主要可视化
    main_viz_path = create_main_visualizations(group_df, stats_df, config)
    
    # 创建个体ROI图
    individual_viz_path = create_individual_roi_plots(group_df, stats_df, config)
    
    return main_viz_path, individual_viz_path

def create_main_visualizations(group_df, stats_df, config):
    """创建主要可视化图表"""
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('fMRI MVPA ROI Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. 准确率热图
    ax1 = axes[0, 0]
    pivot_data = group_df.pivot_table(values='mean_accuracy', index='roi', columns='contrast', aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5, ax=ax1)
    ax1.set_title('平均分类准确率热图')
    ax1.set_xlabel('对比条件')
    ax1.set_ylabel('ROI')
    
    # 2. 显著性结果条形图
    ax2 = axes[0, 1]
    if not stats_df.empty:
        sig_counts = stats_df.groupby('contrast')['significant'].sum()
        sig_counts.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title('各对比条件显著ROI数量')
        ax2.set_xlabel('对比条件')
        ax2.set_ylabel('显著ROI数量')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. 效应量分布
    ax3 = axes[1, 0]
    if not stats_df.empty:
        stats_df['cohens_d'].hist(bins=20, ax=ax3, alpha=0.7, color='lightgreen')
        ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Cohen\'s d 效应量分布')
        ax3.set_xlabel('Cohen\'s d')
        ax3.set_ylabel('频次')
    
    # 4. P值分布
    ax4 = axes[1, 1]
    if not stats_df.empty:
        stats_df['p_value_permutation'].hist(bins=20, ax=ax4, alpha=0.7, color='orange')
        ax4.axvline(config.alpha_level, color='red', linestyle='--', alpha=0.7, label=f'α = {config.alpha_level}')
        ax4.set_title('置换检验P值分布')
        ax4.set_xlabel('P值')
        ax4.set_ylabel('频次')
        ax4.legend()
    
    plt.tight_layout()
    
    # 保存图形
    viz_path = config.results_dir / "main_visualizations.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"主要可视化已保存: {viz_path}", config)
    return viz_path

def create_individual_roi_plots(group_df, stats_df, config):
    """为每个ROI创建详细图表"""
    rois = group_df['roi'].unique()
    n_rois = len(rois)
    
    # 计算子图布局
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rois == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, roi in enumerate(rois):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        roi_data = group_df[group_df['roi'] == roi]
        roi_stats = stats_df[stats_df['roi'] == roi] if not stats_df.empty else pd.DataFrame()
        
        # 创建箱线图
        sns.boxplot(data=roi_data, x='contrast', y='mean_accuracy', ax=ax)
        
        # 添加显著性标记
        if not roi_stats.empty:
            for j, (_, stat) in enumerate(roi_stats.iterrows()):
                if stat['significant']:
                    ax.text(j, ax.get_ylim()[1] * 0.95, '*', 
                           ha='center', va='center', fontsize=16, color='red')
        
        # 添加随机水平线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='随机水平')
        
        ax.set_title(f'{roi}')
        ax.set_xlabel('对比条件')
        ax.set_ylabel('分类准确率')
        ax.tick_params(axis='x', rotation=45)
        
        if i == 0:
            ax.legend()
    
    # 隐藏多余的子图
    for i in range(n_rois, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('各ROI分类准确率详细分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图形
    individual_viz_path = config.results_dir / "individual_roi_plots.png"
    plt.savefig(individual_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log(f"个体ROI图表已保存: {individual_viz_path}", config)
    return individual_viz_path

# ============================================================================
# HTML报告生成模块 (report_generator.py)
# ============================================================================

def image_to_base64(image_path):
    """将图片转换为base64编码，增强错误处理"""
    try:
        # 检查文件是否存在
        if not Path(image_path).exists():
            print(f"警告：图片文件不存在: {image_path}")
            return None
        
        # 检查文件大小，避免加载过大的文件
        file_size = Path(image_path).stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB限制
            print(f"警告：图片文件过大 ({file_size/1024/1024:.1f}MB): {image_path}")
            return None
        
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            print(f"成功编码图片: {image_path} ({file_size/1024:.1f}KB)")
            return encoded
            
    except PermissionError:
        print(f"权限错误：无法读取图片文件: {image_path}")
        return None
    except MemoryError:
        print(f"内存错误：图片文件过大无法加载: {image_path}")
        return None
    except Exception as e:
        print(f"图片编码失败: {image_path}, 错误类型: {type(e).__name__}, 详情: {e}")
        return None

def generate_image_html(image_b64, alt_text):
    """生成图片HTML代码，处理加载失败的情况"""
    if image_b64:
        return f'<img src="data:image/png;base64,{image_b64}" alt="{alt_text}" style="max-width: 100%; height: auto; margin: 10px 0;">'
    else:
        return f'<div class="image-placeholder" style="border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 10px 0;">图片加载失败: {alt_text}</div>'

def generate_html_report(group_df, stats_df, config, main_viz_b64=None, individual_viz_b64=None):
    """生成HTML报告"""
    log("生成HTML报告", config)
    
    # 如果没有提供图片，尝试加载
    if main_viz_b64 is None:
        main_viz_path = config.results_dir / "main_visualizations.png"
        main_viz_b64 = image_to_base64(main_viz_path)
    
    if individual_viz_b64 is None:
        individual_viz_path = config.results_dir / "individual_roi_plots.png"
        individual_viz_b64 = image_to_base64(individual_viz_path)
    
    # 计算汇总统计
    n_subjects = len(group_df['subject'].unique()) if not group_df.empty else 0
    n_rois = len(group_df['roi'].unique()) if not group_df.empty else 0
    n_contrasts = len(group_df['contrast'].unique()) if not group_df.empty else 0
    n_significant = len(stats_df[stats_df['significant']]) if not stats_df.empty else 0
    n_total_tests = len(stats_df) if not stats_df.empty else 0
    
    # 生成HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>fMRI MVPA ROI Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            .summary-box {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #3498db;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background-color: #ffffff;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-top: 3px solid #3498db;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #e8f4f8;
            }}
            .significant {{
                background-color: #d4edda !important;
                font-weight: bold;
            }}
            .not-significant {{
                background-color: #f8d7da !important;
            }}
            .image-container {{
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background-color: #fafafa;
                border-radius: 8px;
            }}
            .image-placeholder {{
                border: 2px dashed #ccc;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #f9f9f9;
                border-radius: 8px;
                color: #666;
            }}
            .method-box {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
            .warning {{
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                color: #721c24;
            }}
            .info {{
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                color: #0c5460;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 fMRI MVPA ROI Analysis Report</h1>
            
            <div class="summary-box">
                <h2>📊 分析概览</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{n_subjects}</div>
                        <div class="stat-label">被试数量</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{n_rois}</div>
                        <div class="stat-label">ROI数量</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{n_contrasts}</div>
                        <div class="stat-label">对比条件</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{n_significant}/{n_total_tests}</div>
                        <div class="stat-label">显著结果</div>
                    </div>
                </div>
            </div>
            
            <div class="method-box">
                <h3>🔬 分析方法</h3>
                <ul>
                    <li><strong>分类器</strong>: 线性支持向量机 (Linear SVM)</li>
                    <li><strong>交叉验证</strong>: {config.cv_folds}折分层交叉验证</li>
                    <li><strong>统计检验</strong>: 组水平置换检验 ({config.n_permutations}次置换)</li>
                    <li><strong>显著性水平</strong>: α = {config.alpha_level}</li>
                    <li><strong>效应量</strong>: Cohen's d</li>
                </ul>
            </div>
    """
    
    # 添加主要可视化
    html_content += f"""
            <h2>📈 主要结果可视化</h2>
            <div class="image-container">
                {generate_image_html(main_viz_b64, "主要分析结果可视化")}
            </div>
    """
    
    # 添加个体ROI可视化
    html_content += f"""
            <h2>🎯 各ROI详细分析</h2>
            <div class="image-container">
                {generate_image_html(individual_viz_b64, "各ROI详细分析结果")}
            </div>
    """
    
    # 添加统计结果表格
    if not stats_df.empty:
        html_content += """
            <h2>📋 详细统计结果</h2>
            <table>
                <thead>
                    <tr>
                        <th>对比条件</th>
                        <th>ROI</th>
                        <th>被试数</th>
                        <th>平均准确率</th>
                        <th>标准差</th>
                        <th>t统计量</th>
                        <th>t检验p值(单尾)</th>
                        <th>置换检验p值</th>
                        <th>Cohen's d</th>
                        <th>效应量</th>
                        <th>显著性</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in stats_df.iterrows():
            row_class = "significant" if row['significant'] else "not-significant"
            significance_text = "✅ 显著" if row['significant'] else "❌ 不显著"
            
            html_content += f"""
                    <tr class="{row_class}">
                        <td>{row['contrast']}</td>
                        <td>{row['roi']}</td>
                        <td>{row['n_subjects']}</td>
                        <td>{row['mean_accuracy']:.4f}</td>
                        <td>{row['std_accuracy']:.4f}</td>
                        <td>{row['t_statistic']:.3f}</td>
                        <td>{row['p_value_ttest_one_tailed']:.4f}</td>
                        <td>{row['p_value_permutation']:.4f}</td>
                        <td>{row['cohens_d']:.3f}</td>
                        <td>{row['effect_size_category']}</td>
                        <td>{significance_text}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        """
    
    # 添加个体结果汇总
    if not group_df.empty:
        html_content += """
            <h2>👥 个体结果汇总</h2>
            <div class="info">
                <p>以下表格显示每个被试在各ROI和对比条件下的分类准确率。这些结果用于组水平统计分析。</p>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>被试</th>
                        <th>ROI</th>
                        <th>对比条件</th>
                        <th>分类准确率</th>
                        <th>标准差</th>
                        <th>特征数</th>
                        <th>样本数</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # 只显示前50行，避免表格过长
        display_df = group_df.head(50)
        for _, row in display_df.iterrows():
            html_content += f"""
                    <tr>
                        <td>{row['subject']}</td>
                        <td>{row['roi']}</td>
                        <td>{row['contrast']}</td>
                        <td>{row['mean_accuracy']:.4f}</td>
                        <td>{row['std_accuracy']:.4f}</td>
                        <td>{row['n_features']}</td>
                        <td>{row['n_samples']}</td>
                    </tr>
            """
        
        if len(group_df) > 50:
            html_content += f"""
                    <tr>
                        <td colspan="7" style="text-align: center; font-style: italic; color: #666;">
                            ... 还有 {len(group_df) - 50} 行数据（完整数据请查看CSV文件）
                        </td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        """
    
    # 添加方法说明和注意事项
    html_content += """
            <h2>📝 方法说明</h2>
            <div class="method-box">
                <h3>🔍 分析流程</h3>
                <ol>
                    <li><strong>数据预处理</strong>: 加载LSS beta图像和trial信息</li>
                    <li><strong>特征提取</strong>: 使用ROI mask提取大脑活动模式</li>
                    <li><strong>分类分析</strong>: 线性SVM进行二分类，网格搜索优化参数</li>
                    <li><strong>交叉验证</strong>: 5折分层交叉验证评估分类性能</li>
                    <li><strong>组水平统计</strong>: 置换检验评估组水平显著性</li>
                </ol>
                
                <h3>📊 统计方法</h3>
                <ul>
                    <li><strong>单样本t检验</strong>: 检验准确率是否显著高于随机水平(0.5)</li>
                    <li><strong>置换检验</strong>: 通过随机翻转标签构建零分布，更稳健的统计推断</li>
                    <li><strong>效应量</strong>: Cohen's d量化效应大小</li>
                    <li><strong>多重比较</strong>: 建议使用FDR校正控制假阳性率</li>
                </ul>
            </div>
            
            <div class="warning">
                <h3>⚠️ 注意事项</h3>
                <ul>
                    <li>本分析未进行多重比较校正，解释结果时需谨慎</li>
                    <li>样本量较小时，置换检验结果更可靠</li>
                    <li>分类准确率过高(>95%)可能提示过拟合或数据泄露</li>
                    <li>建议结合其他验证方法确认结果的可重复性</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>分析软件: Python fMRI MVPA Pipeline</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    report_path = config.results_dir / "analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"HTML报告已保存: {report_path}", config)
    return report_path

# ============================================================================
# 主分析流程
# ============================================================================

def run_group_roi_analysis_corrected(config):
    """运行组水平ROI分析（修正版本）"""
    log("开始组水平ROI分析", config)
    
    # 保存配置
    config.save_config(config.results_dir / "analysis_config.json")
    
    # 加载ROI masks
    roi_masks = load_roi_masks(config)
    if not roi_masks:
        raise ValueError("未找到有效的ROI mask文件")
    
    log(f"成功加载 {len(roi_masks)} 个ROI masks", config)
    
    # 初始化结果存储
    all_results = []
    
    # 对每个被试进行分析
    for subject_id in config.subjects:
        log(f"\n处理被试: {subject_id}", config)
        
        try:
            # 加载该被试的LSS数据
            beta_images, trial_info = load_multi_run_lss_data(subject_id, config.runs, config)
            if beta_images is None or trial_info is None:
                log(f"被试 {subject_id} 的LSS数据加载失败，跳过", config)
                continue
            
            # 对每个对比条件进行分析
            for contrast_name, cond1, cond2 in config.contrasts:
                log(f"  分析对比: {contrast_name} ({cond1} vs {cond2})", config)
                
                # 准备分类数据
                selected_betas, labels, trial_details = prepare_classification_data(
                    trial_info, beta_images, cond1, cond2, config
                )
                
                if selected_betas is None:
                    log(f"跳过 {subject_id} {cond1} vs {cond2}: 数据准备失败", config)
                    continue
                
                # 对每个ROI进行分类分析
                for roi_name, roi_mask_path in roi_masks.items():
                    try:
                        # 提取ROI特征
                        X = extract_roi_features(selected_betas, roi_mask_path, config)
                        
                        if X is None or X.shape[1] == 0:
                            continue
                        
                        # 运行分类分析
                        result = run_roi_classification_enhanced(X, labels, config)
                        
                        if result is not None:
                            result.update({
                                'subject': subject_id,
                                'roi': roi_name,
                                'contrast': contrast_name,
                                'condition1': cond1,
                                'condition2': cond2,
                                'n_trials_cond1': np.sum(labels == 0),
                                'n_trials_cond2': np.sum(labels == 1)
                            })
                            all_results.append(result)
                            
                    except Exception as e:
                        log(f"      {roi_name} 分析出错: {str(e)}", config)
                        continue
            
            # 内存清理
            memory_cleanup()
            
        except Exception as e:
            log(f"被试 {subject_id} 处理失败: {str(e)}", config)
            continue
    
    # 如果有结果，进行组水平分析
    if all_results:
        log(f"\n开始组水平统计分析，共 {len(all_results)} 个结果", config)
        
        # 转换为DataFrame
        group_df = pd.DataFrame(all_results)
        
        # 保存个体结果
        group_df.to_csv(config.results_dir / "individual_roi_mvpa_results.csv", index=False)
        
        # 执行组水平统计分析
        stats_df = perform_group_statistics_corrected(group_df, config)
        
        # 保存组水平统计结果
        stats_df.to_csv(config.results_dir / "group_roi_mvpa_statistics.csv", index=False)
        
        # 创建可视化
        main_viz_path, individual_viz_path = create_enhanced_visualizations(group_df, stats_df, config)
        
        # 生成HTML报告
        generate_html_report(group_df, stats_df, config)
        
        log(f"\n分析完成！结果保存在: {config.results_dir}", config)
        log(f"分析被试数: {len(set(group_df['subject']))}", config)
        log(f"ROI数量: {len(roi_masks)}", config)
        log(f"对比数量: {len(config.contrasts)}", config)
        
        return group_df, stats_df
    else:
        log("没有有效的分析结果", config)
        return None, None

def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()
    
    try:
        # 运行分析
        group_df, stats_df = run_group_roi_analysis_corrected(config)
        if group_df is not None:
            print(f"\n✅ 分析完成！结果保存在: {config.results_dir}")
            return config.results_dir
        else:
            print("\n❌ 分析失败：没有有效结果")
            return None
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()