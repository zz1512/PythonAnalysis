# fmri_mvpa_roi_pipeline_enhanced.py
# Enhanced version with FDR correction, parallel processing, memory optimization, and HTML reporting

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
        self.subjects = [f"sub-{i:02d}" for i in range(1, 5)]
        self.runs = [3]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.roi_dir = Path(r"../../../learn_mvpa/full_roi_mask")
        self.results_dir = Path(r"../../../learn_mvpa/metaphor_ROI_MVPA_enhanced")
        
        # 分类器参数
        self.svm_params = {
            'random_state': 42
        }
        
        # 参数网格搜索配置
        self.svm_param_grid = {
            'C': [0.1, 1.0, 10.0],  # 可选的C值
            'kernel': ['linear']     # 保持线性核
        }
        
        # 交叉验证参数，小样本可用3，大样本使用5
        self.cv_folds = 5
        self.cv_random_state = 42
        
        # 置换检验参数，对于严格的统计推断，n_permutations可以增加到1000
        self.n_permutations = 100
        self.permutation_random_state = 42
        
        # 多重比较校正参数
        self.correction_method = 'fdr_bh'  # 'fdr_bh', 'bonferroni', 'holm'
        self.alpha_level = 0.05
        
        # 并行处理参数
        self.n_jobs = min(4, cpu_count() - 1)  # 保留一个核心
        self.use_parallel = True
        
        # 内存优化参数
        self.batch_size = 50  # 批处理大小
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
    """加载预生成的ROI掩模"""
    log("加载ROI掩模...", config)
    roi_masks = {}
    
    roi_files = list(config.roi_dir.glob("*.nii.gz"))
    
    for roi_file in roi_files:
        roi_name = roi_file.stem.replace("cluster_", "")
        try:
            roi_mask = image.load_img(str(roi_file))
            roi_masks[roi_name] = roi_mask
            log(f"加载ROI: {roi_name}", config)
        except Exception as e:
            log(f"加载ROI失败 {roi_file}: {e}", config)
    
    if not roi_masks:
        log("错误: 没有找到可用的ROI掩模文件", config)
        return None
    
    log(f"成功加载 {len(roi_masks)} 个ROI掩模", config)
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
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True, 
            detrend=False,
            memory=config.memory_cache,
            memory_level=config.memory_level
        )
        
        # 批处理加载大数据集
        if len(functional_imgs) > config.batch_size:
            timeseries_list = []
            for i in range(0, len(functional_imgs), config.batch_size):
                batch = functional_imgs[i:i+config.batch_size]
                batch_ts = masker.fit_transform(batch, confounds=confounds)
                timeseries_list.append(batch_ts)
                memory_cleanup()  # 清理内存
            
            timeseries = np.vstack(timeseries_list)
        else:
            timeseries = masker.fit_transform(functional_imgs, confounds=confounds)
        
        log(f"提取时间序列形状: {timeseries.shape}", config)
        return timeseries, masker
    
    except Exception as e:
        log(f"提取ROI时间序列失败: {e}", config)
        return None, None

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
    
    # 置换检验（使用最佳参数）
    np.random.seed(config.permutation_random_state)
    null_distribution = []
    observed_score = np.mean(cv_scores)
    
    for i in range(config.n_permutations):
        # 使用不同的随机种子确保每次置换都不同
        np.random.seed(config.permutation_random_state + i + 1)
        y_perm = np.random.permutation(y)
        perm_scores = cross_val_score(best_pipeline, X, y_perm, cv=cv, scoring='accuracy')
        null_distribution.append(np.mean(perm_scores))
    
    # 计算p值 - 修正计算逻辑
    null_distribution = np.array(null_distribution)
    p_value = (np.sum(null_distribution >= observed_score) + 1) / (config.n_permutations + 1)
    
    # 确保p值不为0（最小值为1/(n_permutations+1)）
    min_p_value = 1.0 / (config.n_permutations + 1)
    p_value = max(p_value, min_p_value)
    
    return {
        'cv_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'p_value': p_value,
        'null_distribution': null_distribution,
        'chance_level': 0.5,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }

def prepare_classification_data(trial_info, beta_images, cond1, cond2, config):
    """准备分类数据"""
    # 筛选特定条件的trial
    cond1_trials = trial_info[trial_info['original_condition'].str.contains(cond1, case=False, na=False)]
    cond2_trials = trial_info[trial_info['original_condition'].str.contains(cond2, case=False, na=False)]
    
    if len(cond1_trials) < 3 or len(cond2_trials) < 3:
        log(f"条件 {cond1} vs {cond2}: 样本数不足 ({len(cond1_trials)} vs {len(cond2_trials)})", config)
        return None, None, None
    
    # 准备数据和标签
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
                'condition': trial['original_condition'],
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
                'condition': trial['original_condition'],
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
    """单个ROI的并行分析函数"""
    roi_name, roi_mask, selected_betas, labels, config = args
    
    try:
        # 提取ROI时间序列
        X, masker = extract_roi_timeseries_optimized(selected_betas, roi_mask, config)
        if X is None:
            return roi_name, None
        
        # 运行分类
        classification_result = run_roi_classification_enhanced(X, labels, config)
        
        result = {
            **classification_result,
            'n_trials_cond1': np.sum(labels == 0),
            'n_trials_cond2': np.sum(labels == 1),
            'n_voxels': X.shape[1]
        }
        
        memory_cleanup()  # 清理内存
        return roi_name, result
    
    except Exception as e:
        log(f"ROI {roi_name} 分析失败: {e}")
        return roi_name, None

def analyze_single_subject_roi_enhanced(subject, runs, contrasts, roi_masks, config):
    """增强的单被试ROI MVPA分析（支持多run合并）"""
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
    
    results = {}
    
    for contrast_name, cond1, cond2 in contrasts:
        log(f"  {subject}: {cond1} vs {cond2}", config)
        
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
                parallel_results = pool.map(analyze_single_roi_parallel, parallel_args)
            
            # 整理结果
            contrast_results = {}
            for roi_name, result in parallel_results:
                if result is not None:
                    contrast_results[roi_name] = result
        else:
            # 串行处理
            contrast_results = {}
            for roi_name, roi_mask in roi_masks.items():
                log(f"    ROI: {roi_name}", config)
                roi_name, result = analyze_single_roi_parallel(
                    (roi_name, roi_mask, selected_betas, labels, config)
                )
                if result is not None:
                    contrast_results[roi_name] = result
        
        if contrast_results:
            results[contrast_name] = contrast_results
            # 保存trial详细信息（包含run信息）
            runs_str = "_".join([f"run{r}" for r in (runs if isinstance(runs, list) else [runs])])
            trial_details.to_csv(
                config.results_dir / f"{subject}_{contrast_name}_{runs_str}_trial_details.csv",
                index=False
            )
    
    memory_cleanup()  # 清理内存
    return results

# ========== 统计分析函数 ==========
def perform_group_statistics_enhanced(group_df, config):
    """增强的组水平统计检验"""
    log("执行组水平统计检验", config)
    
    stats_results = []
    
    for contrast_name, cond1, cond2 in config.contrasts:
        contrast_data = group_df[group_df['contrast'] == contrast_name]
        
        for roi in group_df['roi'].unique():
            roi_subset = contrast_data[contrast_data['roi'] == roi]
            roi_accuracies = roi_subset['accuracy']
            roi_pvalues = roi_subset['p_value']  # 被试内置换检验p值
            
            if len(roi_accuracies) >= 3:  # 改为>=3，因为3个被试也可以做统计
                # 单样本t检验 vs 随机水平(0.5) - 用于组水平统计
                t_stat, group_p_value = stats.ttest_1samp(roi_accuracies, 0.5)
                cohens_d = (np.mean(roi_accuracies) - 0.5) / np.std(roi_accuracies)
                
                # 计算被试内置换检验p值的平均值（作为参考）
                mean_permutation_p = np.mean(roi_pvalues)
                
                stats_results.append({
                    'contrast': contrast_name,
                    'roi': roi,
                    'mean_accuracy': np.mean(roi_accuracies),
                    'std_accuracy': np.std(roi_accuracies),
                    'sem_accuracy': np.std(roi_accuracies) / np.sqrt(len(roi_accuracies)),
                    't_statistic': t_stat,
                    'p_value': group_p_value,  # 组水平t检验p值
                    'p_value_permutation': mean_permutation_p,  # 被试内置换检验p值平均
                    'cohens_d': cohens_d,
                    'n_subjects': len(roi_accuracies),
                    'ci_lower': np.mean(roi_accuracies) - 1.96 * np.std(roi_accuracies) / np.sqrt(len(roi_accuracies)),
                    'ci_upper': np.mean(roi_accuracies) + 1.96 * np.std(roi_accuracies) / np.sqrt(len(roi_accuracies))
                })
    
    stats_df = pd.DataFrame(stats_results)
    
    # 多重比较校正 - 使用置换检验p值
    if len(stats_df) > 1:
        log(f"应用{config.correction_method}多重比较校正（基于置换检验p值）", config)
        
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            stats_df['p_value_permutation'], 
            alpha=config.alpha_level, 
            method=config.correction_method
        )
        
        stats_df['p_corrected'] = p_corrected
        stats_df['significant_corrected'] = rejected
        stats_df['correction_method'] = config.correction_method
        
        # 报告结果
        significant_uncorrected = stats_df[stats_df['p_value_permutation'] < config.alpha_level]
        significant_corrected = stats_df[stats_df['significant_corrected']]
        
        log(f"\n未校正显著结果数（置换检验）: {len(significant_uncorrected)}", config)
        log(f"校正后显著结果数: {len(significant_corrected)}", config)
        
        if not significant_corrected.empty:
            log("\n校正后显著结果:", config)
            for _, result in significant_corrected.iterrows():
                log(f"  {result['contrast']} - {result['roi']}: "
                    f"accuracy = {result['mean_accuracy']:.3f}, "
                    f"t({result['n_subjects'] - 1}) = {result['t_statistic']:.3f}, "
                    f"p_raw = {result['p_value']:.3f}, "
                    f"p_corrected = {result['p_corrected']:.3f}, "
                    f"d = {result['cohens_d']:.3f}", config)
        else:
            log("\n校正后无显著结果", config)
    else:
        log("\n只有一个比较，无需多重比较校正", config)
        stats_df['p_corrected'] = stats_df['p_value_permutation']
        stats_df['significant_corrected'] = stats_df['p_value_permutation'] < config.alpha_level
        stats_df['correction_method'] = 'none'
    
    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistical_results.csv", index=False)
    
    return stats_df

# ========== 可视化函数 ==========
def create_enhanced_visualizations(group_df, stats_df, config):
    """创建增强的可视化"""
    log("创建可视化", config)
    
    try:
        # 设置绘图风格
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # 如果seaborn-v0_8不可用，尝试其他样式
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
                log("警告：使用默认matplotlib样式，因为seaborn样式不可用", config)
        sns.set_palette("husl")
        
        # 1. ROI分类准确率热图（带显著性标记）
        plt.figure(figsize=(14, 8))
        
        # 准备热图数据
        contrast_names = [c[0] for c in config.contrasts]
        roi_names = sorted(group_df['roi'].unique())
        
        heatmap_data = []
        significance_mask = []
        
        for contrast_name, _, _ in config.contrasts:
            contrast_row = []
            sig_row = []
            for roi in roi_names:
                roi_data = group_df[(group_df['contrast'] == contrast_name) & 
                                   (group_df['roi'] == roi)]['accuracy']
                mean_acc = np.mean(roi_data) if len(roi_data) > 0 else 0
                contrast_row.append(mean_acc)
                
                # 检查显著性
                roi_stats = stats_df[(stats_df['contrast'] == contrast_name) & 
                                    (stats_df['roi'] == roi)]
                is_sig = len(roi_stats) > 0 and roi_stats.iloc[0]['significant_corrected']
                sig_row.append(is_sig)
            
            heatmap_data.append(contrast_row)
            significance_mask.append(sig_row)
        
        heatmap_data = np.array(heatmap_data)
        significance_mask = np.array(significance_mask)
        
        # 绘制热图
        ax = sns.heatmap(heatmap_data,
                        xticklabels=roi_names,
                        yticklabels=contrast_names,
                        annot=True, fmt='.3f', cmap='RdYlBu_r',
                        vmin=0.4, vmax=0.8, center=0.5,
                        cbar_kws={'label': 'Classification Accuracy'})
        
        # 添加显著性标记
        for i in range(len(contrast_names)):
            for j in range(len(roi_names)):
                if significance_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                             edgecolor='black', lw=3))
        
        plt.title('ROI MVPA Classification Accuracy\n(Black borders indicate significance after correction)')
        plt.tight_layout()
        plt.savefig(config.results_dir / "roi_accuracy_heatmap_enhanced.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 详细的统计结果图
        for contrast_name, _, _ in config.contrasts:
            contrast_data = group_df[group_df['contrast'] == contrast_name]
            contrast_stats = stats_df[stats_df['contrast'] == contrast_name]
            
            if not contrast_data.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # 左图：准确率条形图带误差线
                roi_order = contrast_stats.sort_values('mean_accuracy', ascending=False)['roi']
                
                bars = ax1.bar(range(len(roi_order)), 
                              [contrast_stats[contrast_stats['roi']==roi]['mean_accuracy'].iloc[0] 
                               for roi in roi_order],
                              yerr=[contrast_stats[contrast_stats['roi']==roi]['sem_accuracy'].iloc[0] 
                                   for roi in roi_order],
                              capsize=5, alpha=0.7)
                
                # 标记显著性
                for i, roi in enumerate(roi_order):
                    roi_stat = contrast_stats[contrast_stats['roi']==roi].iloc[0]
                    if roi_stat['significant_corrected']:
                        bars[i].set_color('red')
                        ax1.text(i, roi_stat['mean_accuracy'] + roi_stat['sem_accuracy'] + 0.01, 
                                '*', ha='center', va='bottom', fontsize=16, fontweight='bold')
                
                ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Chance Level')
                ax1.set_xlabel('ROI')
                ax1.set_ylabel('Classification Accuracy')
                ax1.set_title(f'Classification Accuracy: {contrast_name}')
                ax1.set_xticks(range(len(roi_order)))
                ax1.set_xticklabels(roi_order, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 右图：效应量
                effect_sizes = [contrast_stats[contrast_stats['roi']==roi]['cohens_d'].iloc[0] 
                               for roi in roi_order]
                colors = ['red' if contrast_stats[contrast_stats['roi']==roi]['significant_corrected'].iloc[0] 
                         else 'blue' for roi in roi_order]
                
                ax2.barh(range(len(roi_order)), effect_sizes, color=colors, alpha=0.7)
                ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                ax2.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
                ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium Effect')
                ax2.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large Effect')
                ax2.set_ylabel('ROI')
                ax2.set_xlabel("Cohen's d")
                ax2.set_title('Effect Sizes')
                ax2.set_yticks(range(len(roi_order)))
                ax2.set_yticklabels(roi_order)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(config.results_dir / f"detailed_results_{contrast_name}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        log("可视化完成", config)
        
    except Exception as e:
        log(f"可视化失败: {e}", config)

# ========== HTML报告生成 ==========
def generate_html_report(group_df, stats_df, config):
    """生成详细的HTML报告"""
    log("生成HTML报告", config)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>fMRI MVPA ROI Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .significant {{ background-color: #ffeeee; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                      background-color: #e6f3ff; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>fMRI MVPA ROI Analysis Report</h1>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Subjects:</strong> {len(config.subjects)} ({', '.join(config.subjects)})</p>
            <p><strong>Runs:</strong> {', '.join([f'run-{r}' for r in config.runs])} (Multi-run Combined Analysis)</p>
            <p><strong>ROIs:</strong> {len(group_df['roi'].unique())}</p>
            <p><strong>Contrasts:</strong> {len(config.contrasts)}</p>
        </div>
        
        <div class="section">
            <h2>Analysis Configuration</h2>
            <div class="metric"><strong>Cross-validation:</strong> {config.cv_folds}-fold</div>
            <div class="metric"><strong>Permutations:</strong> {config.n_permutations}</div>
            <div class="metric"><strong>Correction Method:</strong> {config.correction_method}</div>
            <div class="metric"><strong>Alpha Level:</strong> {config.alpha_level}</div>
            <div class="metric"><strong>SVM Kernel:</strong> {', '.join(config.svm_param_grid['kernel'])}</div>
            <div class="metric"><strong>SVM C:</strong> {', '.join(map(str, config.svm_param_grid['C']))}</div>
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
    """
    
    # 添加汇总统计
    total_tests = len(stats_df)
    significant_uncorrected = len(stats_df[stats_df['p_value'] < config.alpha_level])
    significant_corrected = len(stats_df[stats_df['significant_corrected']])
    
    html_content += f"""
            <div class="metric"><strong>Total Tests:</strong> {total_tests}</div>
            <div class="metric"><strong>Significant (uncorrected):</strong> {significant_uncorrected}</div>
            <div class="metric"><strong>Significant (corrected):</strong> {significant_corrected}</div>
            <div class="metric"><strong>False Discovery Rate:</strong> {(significant_uncorrected - significant_corrected) / max(significant_uncorrected, 1):.2%}</div>
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Contrast</th>
                    <th>ROI</th>
                    <th>Mean Accuracy</th>
                    <th>SEM</th>
                    <th>95% CI</th>
                    <th>t-statistic</th>
                    <th>p-value (permutation)</th>
                    <th>p-value (corrected)</th>
                    <th>Cohen's d</th>
                    <th>N Subjects</th>
                    <th>Significant</th>
                </tr>
    """
    
    # 添加每行结果
    for _, row in stats_df.iterrows():
        row_class = "significant" if row['significant_corrected'] else ""
        significance_mark = "✓" if row['significant_corrected'] else ""
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{row['contrast']}</td>
                    <td>{row['roi']}</td>
                    <td>{row['mean_accuracy']:.3f}</td>
                    <td>{row['sem_accuracy']:.3f}</td>
                    <td>[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]</td>
                    <td>{row['t_statistic']:.3f}</td>
                    <td>{row['p_value_permutation']:.4f}</td>
                    <td>{row['p_corrected']:.4f}</td>
                    <td>{row['cohens_d']:.3f}</td>
                    <td>{row['n_subjects']}</td>
                    <td>{significance_mark}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <h3>ROI Accuracy Heatmap</h3>
            <img src="roi_accuracy_heatmap_enhanced.png" alt="ROI Accuracy Heatmap" style="max-width: 100%;">
    """
    
    # 添加每个对比的详细图
    for contrast_name, _, _ in config.contrasts:
        html_content += f"""
            <h3>Detailed Results: {contrast_name}</h3>
            <img src="detailed_results_{contrast_name}.png" alt="Detailed Results {contrast_name}" style="max-width: 100%;">
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Methods</h2>
            <p><strong>Classification:</strong> Support Vector Machine with linear kernel</p>
            <p><strong>Cross-validation:</strong> Stratified k-fold cross-validation</p>
            <p><strong>Statistical Testing:</strong> Permutation test followed by one-sample t-test against chance level (0.5)</p>
            <p><strong>Multiple Comparison Correction:</strong> False Discovery Rate (FDR) using Benjamini-Hochberg procedure</p>
            <p><strong>Effect Size:</strong> Cohen's d calculated as (mean - 0.5) / standard deviation</p>
        </div>
        
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open(config.results_dir / "analysis_report.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log("HTML报告生成完成", config)

# ========== 主执行函数 ==========
def run_group_roi_analysis_enhanced(config):
    """运行增强的组水平ROI分析"""
    log("开始增强版隐喻ROI MVPA分析", config)
    
    # 保存配置
    config.save_config(config.results_dir / "analysis_config.json")
    
    # 加载ROI掩模
    roi_masks = load_roi_masks(config)
    if roi_masks is None:
        return None
    
    # 分析每个被试
    all_subject_results = {}
    
    for subject in config.subjects:
        subject_results = analyze_single_subject_roi_enhanced(
            subject, config.runs, config.contrasts, roi_masks, config
        )
        if subject_results is not None:
            all_subject_results[subject] = subject_results
        else:
            log(f"跳过被试 {subject}: 数据加载失败", config)
    
    if not all_subject_results:
        log("错误: 没有获得任何分析结果", config)
        return None
    
    # 收集组水平数据
    group_data = []
    for subject, subject_results in all_subject_results.items():
        for contrast_name, contrast_results in subject_results.items():
            for roi_name, roi_result in contrast_results.items():
                group_data.append({
                    'subject': subject,
                    'contrast': contrast_name,
                    'roi': roi_name,
                    'accuracy': roi_result['mean_accuracy'],
                    'std_accuracy': roi_result['std_accuracy'],
                    'p_value': roi_result['p_value'],
                    'n_trials_cond1': roi_result['n_trials_cond1'],
                    'n_trials_cond2': roi_result['n_trials_cond2'],
                    'n_voxels': roi_result['n_voxels'],
                    'n_features': roi_result['n_features'],
                    'n_samples': roi_result['n_samples'],
                    'best_params': str(roi_result.get('best_params', {})),
                    'best_score': roi_result.get('best_score', np.nan)
                })
    
    group_df = pd.DataFrame(group_data)
    group_df.to_csv(config.results_dir / "group_roi_mvpa_results.csv", index=False)
    
    # 组水平统计分析
    stats_df = perform_group_statistics_enhanced(group_df, config)
    
    # 创建可视化
    create_enhanced_visualizations(group_df, stats_df, config)
    
    # 生成HTML报告
    generate_html_report(group_df, stats_df, config)
    
    log(f"\n增强版MVPA分析完成!", config)
    log(f"结果保存在: {config.results_dir}", config)
    log(f"分析被试数: {len(all_subject_results)}", config)
    log(f"ROI数量: {len(roi_masks)}", config)
    log(f"对比数量: {len(config.contrasts)}", config)
    log(f"HTML报告: {config.results_dir / 'analysis_report.html'}", config)
    
    return group_df, stats_df

def main():
    """主函数"""
    # 创建配置
    config = MVPAConfig()
    
    # 运行分析
    results = run_group_roi_analysis_enhanced(config)
    
    if results is not None:
        group_df, stats_df = results
        logging.info("\n=== 分析完成 ===")
        logging.info(f"总测试数: {len(stats_df)}")
        logging.info(f"显著结果数 (校正后): {len(stats_df[stats_df['significant_corrected']])}")
        logging.info(f"详细报告: {config.results_dir / 'analysis_report.html'}")
    else:
        logging.error("分析失败")

if __name__ == "__main__":
    main()