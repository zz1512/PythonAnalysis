# fmri_rsa_pipeline.py
# RSA (Representational Similarity Analysis) Pipeline
# 基于MVPA分析的配置和数据路径结构

import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image
from nilearn.maskers import NiftiMasker
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
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
from sklearn.metrics import pairwise_distances
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 配置类 ==========
class RSAConfig:
    """RSA分析配置参数"""
    def __init__(self):
        # 基本参数
        self.subjects = [f"sub-{i:02d}" for i in range(1, 29) if i not in [14, 24]]
        self.runs = [3, 4]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.roi_dir = Path(r"../../../learn_mvpa/full_roi_mask")
        self.results_dir = Path(r"../../../learn_rsa/metaphor_ROI_RSA")
        
        # RSA特定参数
        self.distance_metrics = ['correlation', 'euclidean', 'cosine']  # 距离度量方法
        self.correlation_method = 'pearson'  # 'pearson', 'spearman'
        
        # 条件设置
        self.conditions = ['yy', 'kj']  # 隐喻条件和空间条件
        self.condition_labels = {'yy': 'metaphor', 'kj': 'spatial'}
        
        # 统计参数
        self.n_permutations = 1000
        self.permutation_random_state = 42
        self.correction_method = 'fdr_bh'  # 'fdr_bh', 'bonferroni', 'holm'
        self.alpha_level = 0.05
        
        # 并行处理参数
        self.n_jobs = min(4, cpu_count() - 1)
        self.use_parallel = True
        
        # 内存优化参数
        self.memory_cache = 'nilearn_cache'
        self.memory_level = 1
        
        # RDM分析类型
        self.rdm_analyses = [
            'within_condition',  # 条件内RDM
            'between_condition',  # 条件间RDM
            'cross_condition'     # 交叉条件RDM
        ]
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'rdms').mkdir(exist_ok=True)
        (self.results_dir / 'statistics').mkdir(exist_ok=True)
        (self.results_dir / 'visualizations').mkdir(exist_ok=True)
    
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
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [RSA] {message}\n")

def memory_cleanup():
    """内存清理"""
    gc.collect()

# ========== 数据加载函数 ==========
def load_roi_masks(config):
    """加载ROI mask文件"""
    roi_masks = {}
    roi_files = list(config.roi_dir.glob("*.nii.gz"))
    
    for roi_file in roi_files:
        roi_name = roi_file.stem.replace('.nii', '')
        roi_masks[roi_name] = roi_file
        log(f"加载ROI mask: {roi_name}", config)
    
    return roi_masks

def load_lss_trial_data(subject, run, config):
    """加载单个被试单个run的LSS trial数据"""
    subject_dir = config.lss_root / subject / "func"
    
    # 查找trial info文件
    trial_info_pattern = f"{subject}_task-*_run-{run:02d}_*trial_info.csv"
    trial_info_files = list(subject_dir.glob(trial_info_pattern))
    
    if not trial_info_files:
        log(f"警告：未找到{subject} run-{run}的trial info文件", config)
        return None, None
    
    trial_info_file = trial_info_files[0]
    trial_info = pd.read_csv(trial_info_file)
    
    # 查找beta images目录
    beta_dir_pattern = f"{subject}_task-*_run-{run:02d}_*trial_betas"
    beta_dirs = list(subject_dir.glob(beta_dir_pattern))
    
    if not beta_dirs:
        log(f"警告：未找到{subject} run-{run}的beta images目录", config)
        return None, None
    
    beta_dir = beta_dirs[0]
    beta_files = sorted(list(beta_dir.glob("trial_*.nii.gz")))
    
    if len(beta_files) != len(trial_info):
        log(f"警告：{subject} run-{run}的beta文件数量({len(beta_files)})与trial数量({len(trial_info)})不匹配", config)
        return None, None
    
    return trial_info, beta_files

def load_multi_run_lss_data(subject, runs, config):
    """加载多个run的LSS数据并合并"""
    all_trial_info = []
    all_beta_files = []
    
    for run in runs:
        trial_info, beta_files = load_lss_trial_data(subject, run, config)
        if trial_info is not None and beta_files is not None:
            # 添加run信息
            trial_info['run'] = run
            all_trial_info.append(trial_info)
            all_beta_files.extend(beta_files)
    
    if not all_trial_info:
        return None, None
    
    # 合并所有run的数据
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    
    return combined_trial_info, all_beta_files

# ========== 特征提取函数 ==========
def extract_roi_patterns(functional_imgs, roi_mask_path, config):
    """从ROI中提取神经活动模式"""
    try:
        # 使用NiftiMasker进行特征提取
        masker = NiftiMasker(
            mask_img=roi_mask_path,
            standardize=True,
            detrend=False,  # RSA通常不需要去趋势
            memory=config.memory_cache,
            memory_level=config.memory_level,
            verbose=0
        )
        
        # 提取模式
        patterns = masker.fit_transform(functional_imgs)
        
        return patterns
        
    except Exception as e:
        log(f"ROI模式提取失败: {str(e)}", config)
        return None

# ========== RDM计算函数 ==========
def compute_rdm(patterns, metric='correlation'):
    """计算表征相似性矩阵(RDM)"""
    if metric == 'correlation':
        # 计算相关距离 (1 - correlation)
        rdm = 1 - np.corrcoef(patterns)
    elif metric == 'euclidean':
        # 计算欧几里得距离
        rdm = pairwise_distances(patterns, metric='euclidean')
    elif metric == 'cosine':
        # 计算余弦距离
        rdm = pairwise_distances(patterns, metric='cosine')
    else:
        raise ValueError(f"不支持的距离度量: {metric}")
    
    return rdm

def compute_condition_rdms(patterns, trial_info, conditions, metric='correlation'):
    """计算不同条件的RDM"""
    rdms = {}
    
    for condition in conditions:
        # 获取该条件的试次索引
        condition_indices = trial_info[trial_info['condition'] == condition].index.tolist()
        
        if len(condition_indices) > 1:
            condition_patterns = patterns[condition_indices]
            rdm = compute_rdm(condition_patterns, metric)
            rdms[condition] = rdm
    
    return rdms

def compute_cross_condition_rdm(patterns, trial_info, condition1, condition2, metric='correlation'):
    """计算跨条件RDM"""
    # 获取两个条件的试次索引
    cond1_indices = trial_info[trial_info['condition'] == condition1].index.tolist()
    cond2_indices = trial_info[trial_info['condition'] == condition2].index.tolist()
    
    if len(cond1_indices) == 0 or len(cond2_indices) == 0:
        return None
    
    cond1_patterns = patterns[cond1_indices]
    cond2_patterns = patterns[cond2_indices]
    
    if metric == 'correlation':
        # 计算跨条件相关距离
        cross_corr = np.corrcoef(cond1_patterns, cond2_patterns)
        # 提取跨条件部分
        rdm = 1 - cross_corr[:len(cond1_indices), len(cond1_indices):]
    elif metric == 'euclidean':
        rdm = pairwise_distances(cond1_patterns, cond2_patterns, metric='euclidean')
    elif metric == 'cosine':
        rdm = pairwise_distances(cond1_patterns, cond2_patterns, metric='cosine')
    else:
        raise ValueError(f"不支持的距离度量: {metric}")
    
    return rdm

# ========== RSA统计分析函数 ==========
def compute_rdm_summary_stats(rdm):
    """计算RDM的汇总统计量"""
    # 提取上三角部分（排除对角线）
    upper_tri_indices = np.triu_indices_from(rdm, k=1)
    distances = rdm[upper_tri_indices]
    
    stats_dict = {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'median_distance': np.median(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances)
    }
    
    return stats_dict

def compare_rdms(rdm1, rdm2, method='spearman'):
    """比较两个RDM的相似性"""
    # 提取上三角部分
    upper_tri_indices = np.triu_indices_from(rdm1, k=1)
    vec1 = rdm1[upper_tri_indices]
    vec2 = rdm2[upper_tri_indices]
    
    if method == 'spearman':
        correlation, p_value = spearmanr(vec1, vec2)
    elif method == 'pearson':
        correlation, p_value = pearsonr(vec1, vec2)
    else:
        raise ValueError(f"不支持的相关方法: {method}")
    
    return correlation, p_value

def permutation_test_rdm_difference(rdm1, rdm2, n_permutations=1000, random_state=42):
    """置换检验比较两个RDM的差异"""
    np.random.seed(random_state)
    
    # 计算观察到的差异
    upper_tri_indices = np.triu_indices_from(rdm1, k=1)
    vec1 = rdm1[upper_tri_indices]
    vec2 = rdm2[upper_tri_indices]
    observed_diff = np.mean(vec1 - vec2)
    
    # 置换检验
    permuted_diffs = []
    combined_vec = np.concatenate([vec1, vec2])
    n_samples = len(vec1)
    
    for _ in range(n_permutations):
        np.random.shuffle(combined_vec)
        perm_vec1 = combined_vec[:n_samples]
        perm_vec2 = combined_vec[n_samples:]
        perm_diff = np.mean(perm_vec1 - perm_vec2)
        permuted_diffs.append(perm_diff)
    
    # 计算p值
    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value, permuted_diffs

# ========== 单被试RSA分析 ==========
def analyze_single_subject_rsa(subject, runs, roi_masks, config):
    """单被试RSA分析"""
    log(f"开始分析被试: {subject}", config)
    
    # 加载LSS数据
    trial_info, beta_files = load_multi_run_lss_data(subject, runs, config)
    
    if trial_info is None or beta_files is None:
        log(f"跳过被试{subject}：数据加载失败", config)
        return None
    
    # 过滤目标条件
    target_conditions = config.conditions
    trial_info_filtered = trial_info[trial_info['condition'].isin(target_conditions)].copy()
    
    if len(trial_info_filtered) == 0:
        log(f"跳过被试{subject}：没有目标条件的数据", config)
        return None
    
    # 获取对应的beta文件
    filtered_indices = trial_info_filtered.index.tolist()
    filtered_beta_files = [beta_files[i] for i in filtered_indices]
    
    subject_results = {
        'subject': subject,
        'roi_results': {}
    }
    
    # 对每个ROI进行分析
    for roi_name, roi_mask_path in roi_masks.items():
        log(f"  分析ROI: {roi_name}", config)
        
        try:
            # 提取ROI模式
            patterns = extract_roi_patterns(filtered_beta_files, roi_mask_path, config)
            
            if patterns is None:
                continue
            
            roi_result = {
                'roi_name': roi_name,
                'n_trials': len(patterns),
                'n_voxels': patterns.shape[1]
            }
            
            # 计算不同类型的RDM
            for metric in config.distance_metrics:
                metric_results = {}
                
                # 1. 条件内RDM
                if 'within_condition' in config.rdm_analyses:
                    condition_rdms = compute_condition_rdms(
                        patterns, trial_info_filtered, target_conditions, metric
                    )
                    
                    for condition, rdm in condition_rdms.items():
                        stats = compute_rdm_summary_stats(rdm)
                        metric_results[f'{condition}_within'] = {
                            'rdm': rdm,
                            'stats': stats
                        }
                
                # 2. 跨条件RDM
                if 'cross_condition' in config.rdm_analyses and len(target_conditions) >= 2:
                    cross_rdm = compute_cross_condition_rdm(
                        patterns, trial_info_filtered, 
                        target_conditions[0], target_conditions[1], metric
                    )
                    
                    if cross_rdm is not None:
                        stats = {
                            'mean_distance': np.mean(cross_rdm),
                            'std_distance': np.std(cross_rdm),
                            'median_distance': np.median(cross_rdm)
                        }
                        metric_results['cross_condition'] = {
                            'rdm': cross_rdm,
                            'stats': stats
                        }
                
                # 3. 全局RDM
                if 'between_condition' in config.rdm_analyses:
                    global_rdm = compute_rdm(patterns, metric)
                    stats = compute_rdm_summary_stats(global_rdm)
                    metric_results['global'] = {
                        'rdm': global_rdm,
                        'stats': stats
                    }
                
                roi_result[f'{metric}_results'] = metric_results
            
            subject_results['roi_results'][roi_name] = roi_result
            
        except Exception as e:
            log(f"  ROI {roi_name}分析失败: {str(e)}", config)
            continue
    
    memory_cleanup()
    return subject_results

# ========== 组水平统计分析 ==========
def perform_group_rsa_statistics(all_results, config):
    """组水平RSA统计分析"""
    log("开始组水平统计分析", config)
    
    # 收集所有ROI的结果
    roi_names = set()
    for result in all_results:
        if result and 'roi_results' in result:
            roi_names.update(result['roi_results'].keys())
    
    group_stats = {}
    
    for roi_name in roi_names:
        log(f"  分析ROI: {roi_name}", config)
        
        roi_group_stats = {
            'roi_name': roi_name,
            'n_subjects': 0,
            'metric_stats': {}
        }
        
        for metric in config.distance_metrics:
            metric_stats = {}
            
            # 收集所有被试的统计量
            for analysis_type in ['within', 'cross_condition', 'global']:
                subject_stats = []
                
                for result in all_results:
                    if (result and 'roi_results' in result and 
                        roi_name in result['roi_results']):
                        
                        roi_result = result['roi_results'][roi_name]
                        metric_key = f'{metric}_results'
                        
                        if metric_key in roi_result:
                            metric_result = roi_result[metric_key]
                            
                            # 根据分析类型收集数据
                            if analysis_type == 'within':
                                # 收集条件内的平均距离
                                within_distances = []
                                for condition in config.conditions:
                                    key = f'{condition}_within'
                                    if key in metric_result:
                                        within_distances.append(
                                            metric_result[key]['stats']['mean_distance']
                                        )
                                if within_distances:
                                    subject_stats.append(np.mean(within_distances))
                            
                            elif analysis_type in metric_result:
                                subject_stats.append(
                                    metric_result[analysis_type]['stats']['mean_distance']
                                )
                
                if len(subject_stats) > 1:
                    # 计算组统计量
                    subject_stats = np.array(subject_stats)
                    
                    # t检验（与0比较）
                    t_stat, p_value = stats.ttest_1samp(subject_stats, 0)
                    
                    # 效应量（Cohen's d）
                    cohens_d = np.mean(subject_stats) / np.std(subject_stats, ddof=1)
                    
                    metric_stats[analysis_type] = {
                        'n_subjects': len(subject_stats),
                        'mean': np.mean(subject_stats),
                        'std': np.std(subject_stats, ddof=1),
                        'sem': stats.sem(subject_stats),
                        't_stat': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'subject_values': subject_stats.tolist()
                    }
            
            roi_group_stats['metric_stats'][metric] = metric_stats
        
        # 计算参与分析的被试数量
        roi_group_stats['n_subjects'] = len([
            r for r in all_results 
            if r and 'roi_results' in r and roi_name in r['roi_results']
        ])
        
        group_stats[roi_name] = roi_group_stats
    
    return group_stats

def apply_multiple_comparison_correction(group_stats, config):
    """应用多重比较校正"""
    log("应用多重比较校正", config)
    
    # 收集所有p值
    all_p_values = []
    p_value_info = []  # 存储p值的来源信息
    
    for roi_name, roi_stats in group_stats.items():
        for metric, metric_stats in roi_stats['metric_stats'].items():
            for analysis_type, stats_dict in metric_stats.items():
                if 'p_value' in stats_dict:
                    all_p_values.append(stats_dict['p_value'])
                    p_value_info.append({
                        'roi': roi_name,
                        'metric': metric,
                        'analysis': analysis_type
                    })
    
    if not all_p_values:
        return group_stats
    
    # 应用校正
    rejected, corrected_p_values, _, _ = multipletests(
        all_p_values, 
        alpha=config.alpha_level, 
        method=config.correction_method
    )
    
    # 将校正后的p值写回结果
    for i, (corrected_p, is_significant) in enumerate(zip(corrected_p_values, rejected)):
        info = p_value_info[i]
        roi_stats = group_stats[info['roi']]
        metric_stats = roi_stats['metric_stats'][info['metric']]
        analysis_stats = metric_stats[info['analysis']]
        
        analysis_stats['p_corrected'] = corrected_p
        analysis_stats['significant_corrected'] = is_significant
    
    return group_stats

# ========== 可视化函数 ==========
def create_rdm_visualization(rdm, title, save_path=None):
    """创建RDM可视化"""
    plt.figure(figsize=(8, 6))
    
    # 创建热图
    sns.heatmap(rdm, 
                annot=False, 
                cmap='viridis', 
                square=True,
                cbar_kws={'label': 'Distance'})
    
    plt.title(title)
    plt.xlabel('Trials')
    plt.ylabel('Trials')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_group_results_visualization(group_stats, config):
    """创建组结果可视化"""
    log("创建组结果可视化", config)
    
    # 为每个度量创建可视化
    for metric in config.distance_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Group RSA Results - {metric.title()} Distance', fontsize=16)
        
        # 收集数据用于可视化
        roi_names = list(group_stats.keys())
        analysis_types = ['within', 'cross_condition', 'global']
        
        for i, analysis_type in enumerate(analysis_types[:3]):
            ax = axes[i//2, i%2]
            
            means = []
            sems = []
            p_values = []
            
            for roi_name in roi_names:
                roi_stats = group_stats[roi_name]
                if (metric in roi_stats['metric_stats'] and 
                    analysis_type in roi_stats['metric_stats'][metric]):
                    
                    stats_dict = roi_stats['metric_stats'][metric][analysis_type]
                    means.append(stats_dict['mean'])
                    sems.append(stats_dict['sem'])
                    p_values.append(stats_dict.get('p_corrected', stats_dict['p_value']))
                else:
                    means.append(0)
                    sems.append(0)
                    p_values.append(1)
            
            # 创建柱状图
            bars = ax.bar(range(len(roi_names)), means, yerr=sems, 
                         capsize=5, alpha=0.7)
            
            # 标记显著性
            for j, (bar, p_val) in enumerate(zip(bars, p_values)):
                if p_val < 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + sems[j] + 0.01,
                           '***', ha='center', va='bottom')
                elif p_val < 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + sems[j] + 0.01,
                           '**', ha='center', va='bottom')
                elif p_val < 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + sems[j] + 0.01,
                           '*', ha='center', va='bottom')
            
            ax.set_title(f'{analysis_type.replace("_", " ").title()} Analysis')
            ax.set_xlabel('ROI')
            ax.set_ylabel('Mean Distance')
            ax.set_xticks(range(len(roi_names)))
            ax.set_xticklabels(roi_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        # 移除空的子图
        if len(analysis_types) < 4:
            axes[1, 1].remove()
        
        plt.tight_layout()
        
        # 保存图片
        save_path = config.results_dir / 'visualizations' / f'group_results_{metric}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log(f"保存可视化: {save_path}", config)

# ========== 报告生成函数 ==========
def generate_rsa_report(group_stats, config):
    """生成RSA分析报告"""
    log("生成RSA分析报告", config)
    
    report_path = config.results_dir / 'RSA_Analysis_Report.html'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RSA Analysis Report</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .significant {{ background-color: #ffeb3b; }}
            .very-significant {{ background-color: #4caf50; color: white; }}
        </style>
    </head>
    <body>
        <h1>RSA (Representational Similarity Analysis) Report</h1>
        
        <h2>Analysis Configuration</h2>
        <ul>
            <li>Subjects: {len(config.subjects)} subjects</li>
            <li>Runs: {config.runs}</li>
            <li>Conditions: {config.conditions}</li>
            <li>Distance Metrics: {config.distance_metrics}</li>
            <li>Correction Method: {config.correction_method}</li>
            <li>Alpha Level: {config.alpha_level}</li>
        </ul>
        
        <h2>Results Summary</h2>
    """
    
    # 为每个ROI创建结果表格
    for roi_name, roi_stats in group_stats.items():
        html_content += f"""
        <h3>ROI: {roi_name}</h3>
        <p>Number of subjects: {roi_stats['n_subjects']}</p>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Analysis Type</th>
                <th>Mean Distance</th>
                <th>SEM</th>
                <th>t-statistic</th>
                <th>p-value</th>
                <th>p-corrected</th>
                <th>Cohen's d</th>
                <th>Significance</th>
            </tr>
        """
        
        for metric, metric_stats in roi_stats['metric_stats'].items():
            for analysis_type, stats_dict in metric_stats.items():
                if 'p_value' in stats_dict:
                    p_corrected = stats_dict.get('p_corrected', stats_dict['p_value'])
                    
                    # 确定显著性级别
                    if p_corrected < 0.001:
                        sig_class = 'very-significant'
                        sig_text = '***'
                    elif p_corrected < 0.01:
                        sig_class = 'significant'
                        sig_text = '**'
                    elif p_corrected < 0.05:
                        sig_class = 'significant'
                        sig_text = '*'
                    else:
                        sig_class = ''
                        sig_text = 'n.s.'
                    
                    html_content += f"""
                    <tr class="{sig_class}">
                        <td>{metric}</td>
                        <td>{analysis_type.replace('_', ' ').title()}</td>
                        <td>{stats_dict['mean']:.4f}</td>
                        <td>{stats_dict['sem']:.4f}</td>
                        <td>{stats_dict['t_stat']:.3f}</td>
                        <td>{stats_dict['p_value']:.4f}</td>
                        <td>{p_corrected:.4f}</td>
                        <td>{stats_dict['cohens_d']:.3f}</td>
                        <td>{sig_text}</td>
                    </tr>
                    """
        
        html_content += "</table>"
    
    html_content += """
        <h2>Notes</h2>
        <ul>
            <li>* p < 0.05, ** p < 0.01, *** p < 0.001</li>
            <li>p-corrected values use {config.correction_method} correction</li>
            <li>Distance values: higher values indicate greater dissimilarity</li>
        </ul>
        
        <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"报告已保存: {report_path}", config)

# ========== 主分析函数 ==========
def run_group_rsa_analysis(config):
    """运行组水平RSA分析"""
    log("开始RSA分析", config)
    
    # 保存配置
    config.save_config(config.results_dir / 'rsa_config.json')
    
    # 加载ROI masks
    roi_masks = load_roi_masks(config)
    
    if not roi_masks:
        log("错误：未找到ROI mask文件", config)
        return
    
    log(f"找到{len(roi_masks)}个ROI masks", config)
    
    # 分析所有被试
    all_results = []
    
    if config.use_parallel and config.n_jobs > 1:
        # 并行处理
        log(f"使用并行处理，进程数: {config.n_jobs}", config)
        
        analyze_func = partial(
            analyze_single_subject_rsa,
            runs=config.runs,
            roi_masks=roi_masks,
            config=config
        )
        
        with Pool(config.n_jobs) as pool:
            all_results = pool.map(analyze_func, config.subjects)
    else:
        # 串行处理
        for subject in config.subjects:
            result = analyze_single_subject_rsa(subject, config.runs, roi_masks, config)
            all_results.append(result)
    
    # 过滤有效结果
    valid_results = [r for r in all_results if r is not None]
    log(f"成功分析{len(valid_results)}个被试", config)
    
    if len(valid_results) == 0:
        log("错误：没有有效的分析结果", config)
        return
    
    # 保存个体结果
    results_file = config.results_dir / 'individual_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = []
        for result in valid_results:
            serializable_result = json.loads(json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
            serializable_results.append(serializable_result)
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    log(f"个体结果已保存: {results_file}", config)
    
    # 组水平统计分析
    group_stats = perform_group_rsa_statistics(valid_results, config)
    
    # 多重比较校正
    group_stats = apply_multiple_comparison_correction(group_stats, config)
    
    # 保存组统计结果
    group_stats_file = config.results_dir / 'statistics' / 'group_statistics.json'
    with open(group_stats_file, 'w', encoding='utf-8') as f:
        serializable_stats = json.loads(json.dumps(group_stats, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    log(f"组统计结果已保存: {group_stats_file}", config)
    
    # 创建可视化
    create_group_results_visualization(group_stats, config)
    
    # 生成报告
    generate_rsa_report(group_stats, config)
    
    log("RSA分析完成", config)
    
    return group_stats

# ========== 主函数 ==========
def main():
    """主函数"""
    # 创建配置
    config = RSAConfig()
    
    # 运行分析
    try:
        group_stats = run_group_rsa_analysis(config)
        
        if group_stats:
            print("\n=== RSA分析完成 ===")
            print(f"结果保存在: {config.results_dir}")
            print(f"分析了{len(config.subjects)}个被试")
            print(f"分析了{len(group_stats)}个ROI")
            
            # 显示显著结果摘要
            print("\n=== 显著结果摘要 ===")
            for roi_name, roi_stats in group_stats.items():
                significant_results = []
                for metric, metric_stats in roi_stats['metric_stats'].items():
                    for analysis_type, stats_dict in metric_stats.items():
                        if 'significant_corrected' in stats_dict and stats_dict['significant_corrected']:
                            significant_results.append(f"{metric}-{analysis_type}")
                
                if significant_results:
                    print(f"{roi_name}: {', '.join(significant_results)}")
        
    except Exception as e:
        logging.error(f"分析过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()