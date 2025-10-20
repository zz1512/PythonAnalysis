# statistics.py
# 统计分析模块

import numpy as np
import pandas as pd
from scipy import stats
from utils import log

def permutation_test_group_level(group_accuracies, n_permutations=1000, random_state=42):
    """组水平置换检验"""
    np.random.seed(random_state)
    
    # 观察到的组平均准确率
    observed_mean = np.mean(group_accuracies)
    
    # 置换检验：每次随机翻转一些被试的标签（相对于随机水平0.5）
    null_distribution = []
    
    for _ in range(n_permutations):
        # 生成随机的准确率（围绕0.5）
        permuted_accuracies = np.random.normal(0.5, np.std(group_accuracies), len(group_accuracies))
        null_distribution.append(np.mean(permuted_accuracies))
    
    null_distribution = np.array(null_distribution)
    
    # 计算p值（双尾检验）
    if observed_mean >= 0.5:
        p_value = np.mean(null_distribution >= observed_mean) * 2
    else:
        p_value = np.mean(null_distribution <= observed_mean) * 2
    
    p_value = min(p_value, 1.0)  # 确保p值不超过1
    
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
        
        # 单样本t检验（vs 0.5随机水平）
        t_stat, t_p_value = stats.ttest_1samp(accuracies, 0.5)
        
        # Cohen's d效应量
        cohens_d = (mean_acc - 0.5) / std_acc
        
        # 95%置信区间
        ci_margin = stats.t.ppf(0.975, n_subjects - 1) * sem_acc
        ci_lower = mean_acc - ci_margin
        ci_upper = mean_acc + ci_margin
        
        # 组水平置换检验
        p_permutation, null_dist = permutation_test_group_level(
            accuracies, 
            n_permutations=config.n_permutations,
            random_state=config.permutation_random_state
        )
        
        result = {
            'contrast': contrast,
            'roi': roi,
            'n_subjects': n_subjects,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'sem_accuracy': sem_acc,
            't_statistic': t_stat,
            'p_value_ttest': t_p_value,
            'p_value_permutation': p_permutation,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'null_distribution_mean': np.mean(null_dist),
            'null_distribution_std': np.std(null_dist)
        }
        
        stats_results.append(result)
        
        log(f"统计完成: {contrast}-{roi} (n={n_subjects}): "
            f"acc={mean_acc:.3f}, t={t_stat:.3f}, p_perm={p_permutation:.3f}, d={cohens_d:.3f}", config)
    
    # 转换为DataFrame
    stats_df = pd.DataFrame(stats_results)
    
    if stats_df.empty:
        log("警告：没有生成任何统计结果", config)
        return stats_df
    
    # 使用置换检验p值作为最终显著性判断
    stats_df['significant'] = stats_df['p_value_permutation'] < config.alpha_level
    
    # 报告结果
    significant_results = stats_df[stats_df['significant']]
    
    log(f"\n组水平置换检验显著结果数: {len(significant_results)}/{len(stats_df)}", config)
    
    if not significant_results.empty:
        log("\n显著结果:", config)
        for _, result in significant_results.iterrows():
            log(f"  {result['contrast']} - {result['roi']}: "
                f"accuracy = {result['mean_accuracy']:.3f}, "
                f"t({result['n_subjects'] - 1}) = {result['t_statistic']:.3f}, "
                f"p_permutation = {result['p_value_permutation']:.3f}, "
                f"d = {result['cohens_d']:.3f}", config)
    else:
        log("\n无显著结果", config)
    
    # 保存统计结果
    stats_df.to_csv(config.results_dir / "group_statistical_results_corrected.csv", index=False)
    
    return stats_df