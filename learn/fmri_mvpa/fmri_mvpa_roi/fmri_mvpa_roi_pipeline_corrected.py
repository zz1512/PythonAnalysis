# fmri_mvpa_roi_pipeline_corrected.py

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
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 配置类 ==========
class MVPAConfig:
    """MVPA分析配置参数"""
    def __init__(self):
        # 基本参数
        self.subjects = [f"sub-{i:02d}" for i in range(1, 29) if i not in [14, 24]]
        self.runs = [3, 4]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.roi_dir = Path(r"../../../learn_mvpa/full_roi_mask")
        self.results_dir = Path(r"../../../learn_mvpa/metaphor_ROI_MVPA_corrected")
        
        # 分类器参数
        self.svm_params = {
            'random_state': 42
        }
        
        # 参数网格搜索配置
        self.svm_param_grid = {
            'C': [0.1, 1.0, 10.0],  # 可选的C值
            'kernel': ['linear']     # 保持线性核
        }
        
        # 交叉验证参数
        self.cv_folds = 5
        self.cv_random_state = 42
        
        # 置换检验参数
        self.n_permutations = 1000  # 增加到1000以获得更稳定的p值
        self.permutation_random_state = 42
        
        # 多重比较校正参数
        self.correction_method = 'fdr_bh'  # 'fdr_bh', 'bonferroni', 'holm'
        self.alpha_level = 0.05
        
        # 新增：多重比较校正方案选择
        self.correction_approach = 'parametric'  # 'parametric', 'permutation_based', 'cluster_based'
        
        # 并行处理参数
        self.n_jobs = min(4, cpu_count() - 1)
        self.use_parallel = True
        
        # 内存优化参数
        self.batch_size = 50
        self.memory_cache = 'nilearn_cache'
        self.memory_level = 1
        
        # 对比条件
        self.contrasts = [
            ('metaphor_vs_space', 'yy', 'kj'),  # 隐喻 vs 空间
        ]
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, filepath):
        """保存配置到JSON文件"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

# ========== 日志和内存管理 ==========
def log(message, config=None):
    """增强的日志函数"""
    logging.info(message)
    
    # 可选：保存到日志文件
    if config and hasattr(config, 'results_dir'):
        log_file = config.results_dir / "analysis.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [MVPA] {message}\n")

def memory_cleanup():
    """内存清理"""
    gc.collect()

# ========== 数据加载函数 ==========
def load_roi_masks(config):
    """加载ROI mask文件"""
    log("加载ROI masks", config)
    
    roi_masks = {}
    roi_files = list(config.roi_dir.glob("*.nii.gz"))
    
    if not roi_files:
        raise FileNotFoundError(f"在{config.roi_dir}中未找到ROI mask文件")
    
    for roi_file in roi_files:
        roi_name = roi_file.stem.replace('.nii', '')
        roi_masks[roi_name] = str(roi_file)
        log(f"  加载ROI: {roi_name}", config)
    
    log(f"总共加载了{len(roi_masks)}个ROI masks", config)
    return roi_masks

def load_lss_trial_data(subject, run, config):
    """加载LSS分析的trial数据和beta图像"""
    lss_dir = config.lss_root / subject / f"run-{run}_LSS"
    
    if not lss_dir.exists():
        log(f"LSS目录不存在: {lss_dir}", config)
        return None, None
    
    # 加载trial信息
    trial_map_path = lss_dir / "trial_info.csv"
    if not trial_map_path.exists():
        log(f"trial_info.csv不存在: {trial_map_path}", config)
        return None, None
    
    trial_info = pd.read_csv(trial_map_path)
    
    # 加载所有beta图像
    beta_images = []
    valid_trials = []
    
    for _, trial in trial_info.iterrows():
        beta_path = lss_dir / f"beta_trial_{trial['trial_index']:03d}.nii.gz"
        if beta_path.exists():
            beta_images.append(str(beta_path))
            # 添加run信息到trial数据中
            trial_with_run = trial.copy()
            trial_with_run['run'] = run
            trial_with_run['global_trial_index'] = f"run{run}_trial{trial['trial_index']}"
            valid_trials.append(trial_with_run)
        else:
            log(f"Beta图像缺失: {beta_path}", config)
    
    if len(beta_images) == 0:
        log(f"没有找到有效的beta图像: {subject} run-{run}", config)
        return None, None
    
    valid_trials_df = pd.DataFrame(valid_trials)
    log(f"加载 {subject} run-{run}: {len(beta_images)}个trial", config)
    
    return beta_images, valid_trials_df

def load_multi_run_lss_data(subject, runs, config):
    """加载并合并多个run的LSS trial数据"""
    log(f"开始加载 {subject} 的多run数据: runs {runs}", config)
    
    all_beta_images = []
    all_trial_info = []
    
    for run in runs:
        beta_images, trial_info = load_lss_trial_data(subject, run, config)
        if beta_images is not None and trial_info is not None:
            all_beta_images.extend(beta_images)
            all_trial_info.append(trial_info)
            log(f"  成功加载 run-{run}: {len(beta_images)}个trial", config)
        else:
            log(f"  跳过 run-{run}: 数据加载失败", config)
    
    if not all_beta_images:
        log(f"错误: {subject} 没有加载到任何有效的trial数据", config)
        return None, None
    
    # 合并所有run的trial信息
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    
    # 重新索引trial，确保每个trial有唯一标识
    combined_trial_info['combined_trial_index'] = range(len(combined_trial_info))
    
    log(f"合并完成 {subject}: 总共{len(all_beta_images)}个trial (来自{len(all_trial_info)}个run)", config)
    
    # 打印每个条件的trial数量统计
    if 'original_condition' in combined_trial_info.columns:
        condition_counts = combined_trial_info['original_condition'].value_counts()
        log(f"  条件统计: {dict(condition_counts)}", config)
    
    return all_beta_images, combined_trial_info

# ========== 特征提取函数 ==========
def extract_roi_timeseries_optimized(functional_imgs, roi_mask_path, config, confounds=None):
    """优化的ROI时间序列提取"""
    try:
        # 使用NiftiMasker进行高效的特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            detrend=True,
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )
        
        # 提取时间序列
        timeseries = masker.fit_transform(functional_imgs, confounds=confounds)
        
        return timeseries
        
    except Exception as e:
        log(f"ROI特征提取失败: {str(e)}", config)
        return None

# ========== 分类分析函数 ==========
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
    
    # 置换检验（被试内水平）
    np.random.seed(config.permutation_random_state)
    null_distribution = []
    observed_score = np.mean(cv_scores)
    
    for i in range(config.n_permutations):
        y_perm = np.random.permutation(y)
        cv_perm = StratifiedKFold(
            n_splits=config.cv_folds, 
            shuffle=True, 
            random_state=config.cv_random_state + i + 1
        )
        perm_scores = cross_val_score(best_pipeline, X, y_perm, cv=cv_perm, scoring='accuracy')
        null_distribution.append(np.mean(perm_scores))
    
    # 计算p值
    null_distribution = np.array(null_distribution)
    p_value = (np.sum(null_distribution >= observed_score) + 1) / (config.n_permutations + 1)
    
    # 确保p值不为0
    min_p_value = 1.0 / (config.n_permutations + 1)
    p_value = max(p_value, min_p_value)
    
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
    
    X_indices = []
    y_labels = []
    trial_details = []
    
    # 条件1的trial
    for _, trial in cond1_trials.iterrows():
        # 对于多run合并数据，使用combined_trial_index；对于单run数据，使用trial_index-1
        if 'combined_trial_index' in trial:
            trial_idx = trial['combined_trial_index']
        else:
            trial_idx = trial['trial_index'] - 1
            
        if trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(0)
            trial_detail = {
                'trial_index': trial['trial_index'],
                'condition': trial[condition_col],
                'label': 0
            }
            # 添加run信息（如果存在）
            if 'run' in trial:
                trial_detail['run'] = trial['run']
            if 'global_trial_index' in trial:
                trial_detail['global_trial_index'] = trial['global_trial_index']
            if 'combined_trial_index' in trial:
                trial_detail['combined_trial_index'] = trial['combined_trial_index']
            trial_details.append(trial_detail)
    
    # 条件2的trial
    for _, trial in cond2_trials.iterrows():
        # 对于多run合并数据，使用combined_trial_index；对于单run数据，使用trial_index-1
        if 'combined_trial_index' in trial:
            trial_idx = trial['combined_trial_index']
        else:
            trial_idx = trial['trial_index'] - 1
            
        if trial_idx < len(beta_images):
            X_indices.append(trial_idx)
            y_labels.append(1)
            trial_detail = {
                'trial_index': trial['trial_index'],
                'condition': trial[condition_col],
                'label': 1
            }
            # 添加run信息（如果存在）
            if 'run' in trial:
                trial_detail['run'] = trial['run']
            if 'global_trial_index' in trial:
                trial_detail['global_trial_index'] = trial['global_trial_index']
            if 'combined_trial_index' in trial:
                trial_detail['combined_trial_index'] = trial['combined_trial_index']
            trial_details.append(trial_detail)
    
    if len(X_indices) < 6:
        log(f"有效样本数不足: {len(X_indices)}", config)
        return None, None, None
    
    selected_betas = [beta_images[i] for i in X_indices]
    return selected_betas, np.array(y_labels), pd.DataFrame(trial_details)

# ========== 并行处理函数 ==========
def analyze_single_roi_parallel(args):
    """单个ROI的并行分析"""
    roi_name, roi_mask_path, selected_betas, labels, config = args
    
    try:
        # 提取ROI特征
        X = extract_roi_timeseries_optimized(selected_betas, roi_mask_path, config)
        
        if X is None or X.shape[1] == 0:
            return None
        
        # 运行分类分析
        results = run_roi_classification_enhanced(X, labels, config)
        results['roi'] = roi_name
        
        return results
        
    except Exception as e:
        log(f"ROI {roi_name} 分析失败: {str(e)}", config)
        return None

# ========== 主要分析函数 ==========
def analyze_single_subject_roi_enhanced(subject, runs, contrasts, roi_masks, config):
    """增强的单被试ROI分析（支持多run合并）"""
    log(f"开始分析 {subject}", config)
    
    # 加载多run合并数据
    if isinstance(runs, list) and len(runs) > 1:
        beta_images, trial_info = load_multi_run_lss_data(subject, runs, config)
    else:
        # 兼容单run分析
        single_run = runs[0] if isinstance(runs, list) else runs
        beta_images, trial_info = load_lss_trial_data(subject, single_run, config)
    
    if beta_images is None:
        return None
    
    results = []
    
    for contrast_name, cond1, cond2 in contrasts:
        log(f"  分析对比: {contrast_name} ({cond1} vs {cond2})", config)
        
        # 准备分类数据
        selected_betas, labels, trial_details = prepare_classification_data(
            trial_info, beta_images, cond1, cond2, config
        )
        
        if selected_betas is None:
            continue
        
        # 并行处理ROI分析
        if config.use_parallel and len(roi_masks) > 1:
            # 准备并行参数
            parallel_args = [
                (roi_name, roi_mask, selected_betas, labels, config)
                for roi_name, roi_mask in roi_masks.items()
            ]
            
            # 并行执行
            with Pool(processes=config.n_jobs) as pool:
                roi_results = pool.map(analyze_single_roi_parallel, parallel_args)
        else:
            # 串行处理
            roi_results = [
                analyze_single_roi_parallel((roi_name, roi_mask, selected_betas, labels, config))
                for roi_name, roi_mask in roi_masks.items()
            ]
        
        # 收集结果
        for result in roi_results:
            if result is not None:
                result.update({
                    'subject': subject,
                    'contrast': contrast_name,
                    'condition1': cond1,
                    'condition2': cond2
                })
                results.append(result)
    
    memory_cleanup()
    return results

# ========== 修正的统计分析函数 ==========
def perform_group_statistics_corrected(group_df, config):
    """修正的组水平统计检验 - 解决统计概念混淆问题"""
    log("执行修正的组水平统计检验", config)
    log("修正内容：使用组水平t检验p值进行多重比较校正", config)
    
    stats_results = []
    
    for contrast_name, cond1, cond2 in config.contrasts:
        contrast_data = group_df[group_df['contrast'] == contrast_name]
        
        for roi in group_df['roi'].unique():
            roi_subset = contrast_data[contrast_data['roi'] == roi]
            roi_accuracies = roi_subset['mean_accuracy']  # 使用准确率而不是p值
            roi_pvalues_permutation = roi_subset['p_value_permutation']  # 被试内置换检验p值
            
            if len(roi_accuracies) >= 3:
                # 组水平单样本t检验 vs 随机水平(0.5)
                t_stat, group_p_value = stats.ttest_1samp(roi_accuracies, 0.5)
                cohens_d = (np.mean(roi_accuracies) - 0.5) / np.std(roi_accuracies)
                
                # 计算被试内置换检验p值的平均值（作为参考）
                mean_permutation_p = np.mean(roi_pvalues_permutation)
                
                # 计算置换检验显著的被试比例
                significant_subjects_perm = np.sum(roi_pvalues_permutation < 0.05)
                prop_significant_perm = significant_subjects_perm / len(roi_pvalues_permutation)
                
                stats_results.append({
                    'contrast': contrast_name,
                    'roi': roi,
                    'mean_accuracy': np.mean(roi_accuracies),
                    'std_accuracy': np.std(roi_accuracies),
                    'sem_accuracy': np.std(roi_accuracies) / np.sqrt(len(roi_accuracies)),
                    't_statistic': t_stat,
                    'p_value_group_ttest': group_p_value,  # 组水平t检验p值（用于多重比较校正）
                    'p_value_permutation_mean': mean_permutation_p,  # 被试内置换检验p值平均
                    'prop_significant_permutation': prop_significant_perm,  # 置换检验显著的被试比例
                    'n_significant_permutation': significant_subjects_perm,  # 置换检验显著的被试数量
                    'cohens_d': cohens_d,
                    'n_subjects': len(roi_accuracies),
                    'ci_lower': np.mean(roi_accuracies) - 1.96 * np.std(roi_accuracies) / np.sqrt(len(roi_accuracies)),
                    'ci_upper': np.mean(roi_accuracies) + 1.96 * np.std(roi_accuracies) / np.sqrt(len(roi_accuracies))
                })
    
    stats_df = pd.DataFrame(stats_results)
    
    # 修正：使用组水平t检验p值进行多重比较校正
    if len(stats_df) > 1:
        if config.correction_approach == 'parametric':
            log(f"应用{config.correction_method}多重比较校正（基于组水平t检验p值）", config)
            
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                stats_df['p_value_group_ttest'],  # 修正：使用组水平t检验p值
                alpha=config.alpha_level, 
                method=config.correction_method
            )
            
            stats_df['p_corrected'] = p_corrected
            stats_df['significant_corrected'] = rejected
            stats_df['correction_method'] = f"{config.correction_method}_parametric"
            
        elif config.correction_approach == 'permutation_based':
            # 方案1：基于置换的多重比较校正
            log("应用基于置换的多重比较校正（方案1）", config)
            stats_df = apply_permutation_based_correction(stats_df, group_df, config)
            
        elif config.correction_approach == 'cluster_based':
            # 方案2：簇水平分析框架（占位符）
            log("簇水平分析框架尚未实现（方案2）", config)
            stats_df['p_corrected'] = stats_df['p_value_group_ttest']
            stats_df['significant_corrected'] = stats_df['p_value_group_ttest'] < config.alpha_level
            stats_df['correction_method'] = 'cluster_based_placeholder'
        
        # 报告结果
        significant_uncorrected = stats_df[stats_df['p_value_group_ttest'] < config.alpha_level]
        significant_corrected = stats_df[stats_df['significant_corrected']]
        
        log(f"\n未校正显著结果数（组水平t检验）: {len(significant_uncorrected)}", config)
        log(f"校正后显著结果数: {len(significant_corrected)}", config)
        
        if not significant_corrected.empty:
            log("\n校正后显著结果:", config)
            for _, result in significant_corrected.iterrows():
                log(f"  {result['contrast']} - {result['roi']}: "
                    f"accuracy = {result['mean_accuracy']:.3f}, "
                    f"t({result['n_subjects'] - 1}) = {result['t_statistic']:.3f}, "
                    f"p_group = {result['p_value_group_ttest']:.3f}, "
                    f"p_corrected = {result['p_corrected']:.3f}, "
                    f"d = {result['cohens_d']:.3f}", config)
        else:
            log("\n校正后无显著结果", config)
    else:
        log("\n只有一个比较，无需多重比较校正", config)
        stats_df['p_corrected'] = stats_df['p_value_group_ttest']
        stats_df['significant_corrected'] = stats_df['p_value_group_ttest'] < config.alpha_level
        stats_df['correction_method'] = 'none'
    
    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistical_results_corrected.csv", index=False)
    
    return stats_df

def apply_permutation_based_correction(stats_df, group_df, config):
    """改进的基于置换的多重比较校正"""
    log("实施基于置换的多重比较校正", config)
    
    # 按对比和ROI组织数据
    contrast_roi_data = {}
    for contrast_name, _, _ in config.contrasts:
        contrast_data = group_df[group_df['contrast'] == contrast_name]
        contrast_roi_data[contrast_name] = {}
        
        for roi in contrast_data['roi'].unique():
            roi_data = contrast_data[contrast_data['roi'] == roi]
            if len(roi_data) >= 3:
                contrast_roi_data[contrast_name][roi] = roi_data['mean_accuracy'].values
    
    # 构建最大统计量的零分布
    max_t_null = np.zeros(config.n_permutations)
    
    np.random.seed(config.permutation_random_state)
    
    for perm_i in range(config.n_permutations):
        if perm_i % 100 == 0:
            log(f"  置换进度: {perm_i}/{config.n_permutations}", config)
        
        perm_max_t = -np.inf
        
        for contrast_name, roi_data_dict in contrast_roi_data.items():
            for roi, accuracies in roi_data_dict.items():
                # 更合理的置换方法：在零假设下生成数据
                # 零假设：准确率围绕0.5随机波动
                null_mean = 0.5
                null_std = np.std(accuracies)
                
                # 从零分布中生成置换数据
                perm_accuracies = np.random.normal(null_mean, null_std, size=len(accuracies))
                
                # 计算置换后的t统计量
                t_stat_perm, _ = stats.ttest_1samp(perm_accuracies, 0.5)
                perm_max_t = max(perm_max_t, abs(t_stat_perm))
        
        max_t_null[perm_i] = perm_max_t
    
    # 计算校正后的p值
    corrected_results = []
    for contrast_name, roi_data_dict in contrast_roi_data.items():
        for roi, accuracies in roi_data_dict.items():
            # 计算观察到的t统计量
            t_obs, _ = stats.ttest_1samp(accuracies, 0.5)
            
            # 计算校正p值
            p_corrected = (np.sum(max_t_null >= abs(t_obs)) + 1) / (config.n_permutations + 1)
            
            # 查找对应的统计结果行
            mask = (stats_df['contrast'] == contrast_name) & (stats_df['roi'] == roi)
            if mask.any():
                idx = mask.idxmax()
                stats_df.loc[idx, 'p_corrected'] = p_corrected
                stats_df.loc[idx, 'significant_corrected'] = p_corrected < config.alpha_level
    
    stats_df['correction_method'] = 'permutation_based_maxT'
    return stats_df

# ========== 主要运行函数 ==========
def run_group_roi_analysis_corrected(config):
    """运行修正的组水平ROI分析"""
    log("开始修正的组水平ROI MVPA分析", config)
    log(f"分析被试: {config.subjects}", config)
    log(f"分析runs: {config.runs}", config)
    log(f"对比条件: {config.contrasts}", config)
    log(f"多重比较校正方案: {config.correction_approach}", config)
    
    # 保存配置
    config.save_config(config.results_dir / "analysis_config_corrected.json")
    
    # 加载ROI masks
    roi_masks = load_roi_masks(config)
    
    # 分析所有被试
    all_results = []
    
    for subject in config.subjects:
        try:
            subject_results = analyze_single_subject_roi_enhanced(
                subject, config.runs, config.contrasts, roi_masks, config
            )
            all_results.extend(subject_results)
        except Exception as e:
            log(f"被试{subject}分析失败: {str(e)}", config)
            continue
    
    if not all_results:
        log("错误：没有成功分析的被试数据", config)
        return
    
    # 创建组水平数据框
    group_df = pd.DataFrame(all_results)
    group_df.to_csv(config.results_dir / "individual_results_corrected.csv", index=False)
    
    log(f"成功分析了{len(group_df['subject'].unique())}个被试的数据", config)
    
    # 执行修正的组水平统计检验
    stats_df = perform_group_statistics_corrected(group_df, config)
    
    # 创建可视化
    create_enhanced_visualizations(group_df, stats_df, config)
    
    # 生成HTML报告
    generate_html_report(group_df, stats_df, config)
    
    log(f"分析完成！结果保存在: {config.results_dir}", config)
    
    return group_df, stats_df

# ========== 可视化和报告函数（简化版本） ==========
def create_enhanced_visualizations(group_df, stats_df, config):
    """创建增强的可视化"""
    log("创建可视化", config)
    
    # 简化的可视化实现
    plt.figure(figsize=(12, 8))
    
    # ROI分类准确率条形图
    roi_means = group_df.groupby('roi')['mean_accuracy'].mean().sort_values(ascending=False)
    
    plt.subplot(2, 2, 1)
    roi_means.plot(kind='bar')
    plt.title('ROI分类准确率')
    plt.ylabel('准确率')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='随机水平')
    plt.legend()
    plt.xticks(rotation=45)
    
    # 显著性结果
    plt.subplot(2, 2, 2)
    if 'significant_corrected' in stats_df.columns:
        sig_counts = stats_df['significant_corrected'].value_counts()
        sig_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('校正后显著性结果')
    
    plt.tight_layout()
    plt.savefig(config.results_dir / "corrected_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    log("可视化已保存", config)

def generate_html_report(group_df, stats_df, config):
    """生成增强的中文HTML报告"""
    log("生成增强的中文HTML报告", config)
    
    # 数据预处理和排序：按照>0.5且具有显著效应排序
    # 首先筛选准确率>0.5的结果
    high_accuracy_df = stats_df[stats_df['mean_accuracy'] > 0.5].copy()
    
    # 按照显著性和效应量排序
    high_accuracy_df['sort_key'] = (
        high_accuracy_df['significant_corrected'].astype(int) * 1000 +  # 显著性优先
        high_accuracy_df['mean_accuracy'] * 100 +  # 准确率次之
        (1 - high_accuracy_df['p_corrected']) * 10  # p值越小排序越前
    )
    sorted_df = high_accuracy_df.sort_values('sort_key', ascending=False)
    
    # 统计摘要
    total_rois = len(stats_df)
    high_acc_rois = len(high_accuracy_df)
    significant_rois = len(stats_df[stats_df['significant_corrected']])
    significant_high_acc = len(sorted_df[sorted_df['significant_corrected']])
    
    # 生成详细的结果表格HTML
    def generate_results_table(df):
        if df.empty:
            return "<p>无符合条件的结果</p>"
        
        table_html = """
        <table class="results-table">
            <thead>
                <tr>
                    <th>排名</th>
                    <th>ROI区域</th>
                    <th>分类准确率</th>
                    <th>标准误</th>
                    <th>t统计量</th>
                    <th>组水平p值</th>
                    <th>校正后p值</th>
                    <th>Cohen's d</th>
                    <th>95%置信区间</th>
                    <th>显著性</th>
                    <th>效应解释</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            # 效应量解释
            d_value = abs(row['cohens_d'])
            if d_value < 0.2:
                effect_size = "微小效应"
            elif d_value < 0.5:
                effect_size = "小效应"
            elif d_value < 0.8:
                effect_size = "中等效应"
            else:
                effect_size = "大效应"
            
            # 显著性标记
            significance = "✓ 显著" if row['significant_corrected'] else "不显著"
            row_class = "significant-row" if row['significant_corrected'] else ""
            
            table_html += f"""
                <tr class="{row_class}">
                    <td>{idx}</td>
                    <td><strong>{row['roi']}</strong></td>
                    <td>{row['mean_accuracy']:.3f}</td>
                    <td>{row['sem_accuracy']:.3f}</td>
                    <td>{row['t_statistic']:.3f}</td>
                    <td>{row['p_value_group_ttest']:.4f}</td>
                    <td>{row['p_corrected']:.4f}</td>
                    <td>{row['cohens_d']:.3f}</td>
                    <td>[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]</td>
                    <td>{significance}</td>
                    <td>{effect_size}</td>
                </tr>
            """
        
        table_html += "</tbody></table>"
        return table_html
    
    # 生成详细分析文本
    def generate_detailed_analysis():
        analysis_text = """
        <div class="analysis-section">
            <h3>🔍 详细结果分析</h3>
        """
        
        if not sorted_df.empty:
            # 最佳结果分析
            best_result = sorted_df.iloc[0]
            analysis_text += f"""
            <div class="highlight-box">
                <h4>🏆 最佳分类结果</h4>
                <p><strong>{best_result['roi']}</strong> 区域表现最佳：</p>
                <ul>
                    <li>分类准确率：<strong>{best_result['mean_accuracy']:.1%}</strong>（高于随机水平{(best_result['mean_accuracy']-0.5)*100:.1f}个百分点）</li>
                    <li>统计显著性：t({best_result['n_subjects']-1}) = {best_result['t_statistic']:.3f}, p = {best_result['p_corrected']:.4f}</li>
                    <li>效应量：Cohen's d = {best_result['cohens_d']:.3f}（{"大效应" if abs(best_result['cohens_d']) >= 0.8 else "中等效应" if abs(best_result['cohens_d']) >= 0.5 else "小效应"}）</li>
                    <li>置信区间：[{best_result['ci_lower']:.3f}, {best_result['ci_upper']:.3f}]</li>
                </ul>
            </div>
            """
            
            # 显著结果总结
            if significant_high_acc > 0:
                analysis_text += f"""
                <div class="summary-box">
                    <h4>📊 显著结果总结</h4>
                    <p>在{high_acc_rois}个准确率超过随机水平的ROI中，有<strong>{significant_high_acc}个</strong>达到统计显著性：</p>
                    <ul>
                """
                
                for _, row in sorted_df[sorted_df['significant_corrected']].iterrows():
                    analysis_text += f"""
                        <li><strong>{row['roi']}</strong>：准确率{row['mean_accuracy']:.1%}，p = {row['p_corrected']:.4f}</li>
                    """
                
                analysis_text += "</ul></div>"
            
            # 趋势分析
            analysis_text += f"""
            <div class="trend-box">
                <h4>📈 整体趋势分析</h4>
                <ul>
                    <li>总体表现：{high_acc_rois}/{total_rois}个ROI的分类准确率超过随机水平（50%）</li>
                    <li>显著性比例：{significant_rois}/{total_rois}个ROI达到统计显著性（{significant_rois/total_rois*100:.1f}%）</li>
                    <li>平均准确率：{stats_df['mean_accuracy'].mean():.1%}（全部ROI），{high_accuracy_df['mean_accuracy'].mean():.1%}（>50%的ROI）</li>
                    <li>最高准确率：{stats_df['mean_accuracy'].max():.1%}</li>
                </ul>
            </div>
            """
        else:
            analysis_text += "<p>⚠️ 没有ROI的分类准确率超过随机水平（50%）</p>"
        
        analysis_text += "</div>"
        return analysis_text
    
    # 生成优化建议
    def generate_optimization_suggestions():
        suggestions = """
        <div class="optimization-section">
            <h3>🚀 优化建议</h3>
            <div class="suggestions-grid">
        """
        
        # 数据质量优化
        suggestions += """
            <div class="suggestion-box">
                <h4>📊 数据质量优化</h4>
                <ul>
                    <li><strong>增加样本量</strong>：当前被试数量可能限制了统计功效</li>
                    <li><strong>数据预处理</strong>：考虑更严格的运动校正和信号去噪</li>
                    <li><strong>特征选择</strong>：可尝试基于univariate分析的特征预选择</li>
                    <li><strong>时间窗口</strong>：优化LSS模型的时间窗口参数</li>
                </ul>
            </div>
        """
        
        # 分析方法优化
        suggestions += """
            <div class="suggestion-box">
                <h4>🔬 分析方法优化</h4>
                <ul>
                    <li><strong>分类器调优</strong>：尝试不同的SVM参数或其他分类器（如随机森林）</li>
                    <li><strong>交叉验证</strong>：考虑leave-one-subject-out交叉验证</li>
                    <li><strong>多模态融合</strong>：结合多个ROI的信息进行集成分类</li>
                    <li><strong>时间序列分析</strong>：考虑时间动态信息</li>
                </ul>
            </div>
        """
        
        # 统计分析优化
        suggestions += """
            <div class="suggestion-box">
                <h4>📈 统计分析优化</h4>
                <ul>
                    <li><strong>贝叶斯分析</strong>：补充贝叶斯统计方法评估证据强度</li>
                    <li><strong>效应量报告</strong>：更多关注效应量而非仅p值</li>
                    <li><strong>置信区间</strong>：报告准确率的置信区间</li>
                    <li><strong>重复性验证</strong>：在独立数据集上验证结果</li>
                </ul>
            </div>
        """
        
        # 可视化优化
        suggestions += """
            <div class="suggestion-box">
                <h4>🎨 可视化优化</h4>
                <ul>
                    <li><strong>脑图可视化</strong>：生成ROI结果的脑图展示</li>
                    <li><strong>交互式图表</strong>：使用plotly等工具创建交互式结果展示</li>
                    <li><strong>个体差异</strong>：展示被试间的个体差异模式</li>
                    <li><strong>时间序列图</strong>：如果有多个时间点，展示时间变化</li>
                </ul>
            </div>
            </div>
        </div>
        """
        
        return suggestions
    
    # 主HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>fMRI MVPA ROI分析详细报告</title>
        <style>
            body {{
                font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                font-weight: bold;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #007bff;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
                margin-bottom: 5px;
            }}
            .stat-label {{
                color: #666;
                font-size: 0.9em;
            }}
            .results-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
            }}
            .results-table th {{
                background-color: #343a40;
                color: white;
                padding: 12px 8px;
                text-align: center;
                font-weight: bold;
            }}
            .results-table td {{
                padding: 10px 8px;
                text-align: center;
                border-bottom: 1px solid #dee2e6;
            }}
            .results-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .significant-row {{
                background-color: #d4edda !important;
                font-weight: bold;
            }}
            .highlight-box {{
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #e17055;
            }}
            .summary-box {{
                background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #00b894;
            }}
            .trend-box {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #2196f3;
            }}
            .suggestions-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .suggestion-box {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            .suggestion-box h4 {{
                color: #28a745;
                margin-top: 0;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .section h2 {{
                color: #343a40;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .section h3 {{
                color: #495057;
                margin-top: 25px;
            }}
            ul {{
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 8px;
            }}
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .config-item {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 3px solid #6c757d;
            }}
            .config-label {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 5px;
            }}
            .config-value {{
                color: #007bff;
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 fMRI MVPA ROI分析详细报告</h1>
                <p>生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                <p>多重比较校正方案：{config.correction_approach}</p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-number">{total_rois}</div>
                    <div class="stat-label">总ROI数量</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{high_acc_rois}</div>
                    <div class="stat-label">准确率>50%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{significant_rois}</div>
                    <div class="stat-label">统计显著</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats_df['mean_accuracy'].max():.1%}</div>
                    <div class="stat-label">最高准确率</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📋 分析配置信息</h2>
                <div class="config-grid">
                    <div class="config-item">
                        <div class="config-label">被试数量</div>
                        <div class="config-value">{len(group_df['subject'].unique())}人</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">ROI区域数量</div>
                        <div class="config-value">{len(group_df['roi'].unique())}个</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">对比条件</div>
                        <div class="config-value">{', '.join([f'{c[0]} ({c[1]} vs {c[2]})' for c in config.contrasts])}</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">校正方法</div>
                        <div class="config-value">{config.correction_method}</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">显著性水平</div>
                        <div class="config-value">α = {config.alpha_level}</div>
                    </div>
                    <div class="config-item">
                        <div class="config-label">交叉验证</div>
                        <div class="config-value">{config.cv_folds}折交叉验证</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🏆 主要结果（按效应排序）</h2>
                <p><strong>排序规则：</strong>优先显示准确率>50%且具有显著效应的ROI，按显著性和效应量排序</p>
                {generate_results_table(sorted_df)}
            </div>
            
            {generate_detailed_analysis()}
            
            {generate_optimization_suggestions()}
            
            <div class="section">
                <h2>📊 完整统计结果</h2>
                <p>以下为所有ROI的完整统计结果：</p>
                {generate_results_table(stats_df.sort_values('mean_accuracy', ascending=False))}
            </div>
            
            <div class="section">
                <h2>📝 方法学说明</h2>
                <div class="highlight-box">
                    <h4>🔬 分析流程</h4>
                    <ol>
                        <li><strong>数据预处理</strong>：LSS（Least Squares Separate）模型提取单试次beta值</li>
                        <li><strong>特征提取</strong>：从每个ROI提取BOLD信号特征</li>
                        <li><strong>分类分析</strong>：使用支持向量机（SVM）进行二分类</li>
                        <li><strong>交叉验证</strong>：{config.cv_folds}折分层交叉验证评估分类性能</li>
                        <li><strong>统计检验</strong>：组水平单样本t检验（vs 50%随机水平）</li>
                        <li><strong>多重比较校正</strong>：{config.correction_method}方法校正多重比较</li>
                    </ol>
                </div>
                
                <div class="summary-box">
                    <h4>📈 统计指标解释</h4>
                    <ul>
                        <li><strong>分类准确率</strong>：正确分类的试次比例，>50%表示超过随机水平</li>
                        <li><strong>t统计量</strong>：组水平t检验统计量，绝对值越大效应越强</li>
                        <li><strong>Cohen's d</strong>：标准化效应量，0.2/0.5/0.8分别对应小/中/大效应</li>
                        <li><strong>95%置信区间</strong>：真实准确率的可能范围</li>
                        <li><strong>校正后p值</strong>：经多重比较校正的p值，<0.05为显著</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>⚠️ 局限性与注意事项</h2>
                <ul>
                    <li>结果的泛化性需要在独立样本中验证</li>
                    <li>ROI定义可能影响结果，建议尝试不同的ROI划分方案</li>
                    <li>分类准确率受到噪声、个体差异等多种因素影响</li>
                    <li>统计显著性不等同于实际意义，需结合效应量综合判断</li>
                    <li>多重比较校正可能过于保守，可考虑FDR等方法</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存报告
    report_path = config.results_dir / "enhanced_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"增强的中文HTML报告已生成：{report_path}", config)
    
    # 同时生成简化版本的CSV报告
    summary_df = sorted_df[[
        'roi', 'mean_accuracy', 'sem_accuracy', 't_statistic', 
        'p_value_group_ttest', 'p_corrected', 'cohens_d', 
        'significant_corrected', 'ci_lower', 'ci_upper'
    ]].copy()
    summary_df.columns = [
        'ROI区域', '分类准确率', '标准误', 't统计量', 
        '组水平p值', '校正后p值', 'Cohen\'s d', 
        '是否显著', '置信区间下限', '置信区间上限'
    ]
    summary_df.to_csv(config.results_dir / "results_summary_chinese.csv", 
                      index=False, encoding='utf-8-sig')
    
    return report_path

# ========== 主函数 ==========
def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()
    
    # 可以在这里修改配置
    # config.correction_approach = 'permutation_based'  # 选择校正方案
    
    # 运行分析
    try:
        group_df, stats_df = run_group_roi_analysis_corrected(config)
        print("\n=== 修正的分析完成 ===")
        print(f"结果保存在: {config.results_dir}")
        
    except Exception as e:
        print(f"分析失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()