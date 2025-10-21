import numpy as np
import pandas as pd
from scipy import stats
from utils import log

def permutation_test_group_level(group_accuracies, n_permutations=1000, random_state=42):
    """组水平置换检验 - 修正版本，解决P值为0的问题"""
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
        
        # 组水平置换检验（现在使用单尾检验）
        p_permutation, null_dist = permutation_test_group_level(
            accuracies, 
            n_permutations=config.n_permutations,
            random_state=config.permutation_random_state
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