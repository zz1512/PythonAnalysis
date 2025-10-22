# visualization.py
# 可视化模块 - Searchlight版本

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def log(message, config):
    """简单的日志函数"""
    print(f"[LOG] {message}")

def setup_chinese_fonts():
    """设置中文字体支持"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体设置成功")
    except Exception as e:
        print(f"中文字体设置失败，使用默认字体: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def create_enhanced_visualizations(group_df, stats_df, config):
    """
    创建增强的Searchlight MVPA可视化图表
    
    Args:
        group_df: 个体结果DataFrame
        stats_df: 组水平统计结果DataFrame
        config: 配置对象
    
    Returns:
        tuple: (主要可视化路径, 个体可视化路径)
    """
    
    # 设置中文字体
    setup_chinese_fonts()
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建主要可视化
    main_viz_path = create_main_visualization(group_df, stats_df, config)
    
    # 创建个体结果可视化
    individual_viz_path = create_individual_visualization(group_df, config)
    
    return main_viz_path, individual_viz_path

def create_main_visualization(group_df, stats_df, config):
    """
    创建主要的组水平结果可视化
    
    Args:
        group_df: 个体结果DataFrame
        stats_df: 组水平统计结果DataFrame
        config: 配置对象
    
    Returns:
        Path: 保存的图片路径
    """
    
    if group_df.empty or stats_df.empty:
        print("数据为空，跳过主要可视化")
        return None
    
    try:
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        # 1. 组水平准确率条形图
        ax1 = fig.add_subplot(gs[0, :])
        create_group_accuracy_barplot(stats_df, ax1)
        
        # 2. 个体准确率分布
        ax2 = fig.add_subplot(gs[1, 0])
        create_accuracy_distribution(group_df, ax2)
        
        # 3. 效应量可视化
        ax3 = fig.add_subplot(gs[1, 1])
        create_effect_size_plot(stats_df, ax3)
        
        # 4. p值比较
        ax4 = fig.add_subplot(gs[1, 2])
        create_pvalue_comparison(stats_df, ax4)
        
        # 5. 被试间一致性热图
        ax5 = fig.add_subplot(gs[2, 0])
        create_subject_consistency_heatmap(group_df, ax5)
        
        # 6. 准确率vs样本量散点图
        ax6 = fig.add_subplot(gs[2, 1])
        create_accuracy_vs_sample_size(group_df, ax6)
        
        # 7. 统计显著性总结
        ax7 = fig.add_subplot(gs[2, 2])
        create_significance_summary(stats_df, ax7)
        
        # 设置总标题
        fig.suptitle('fMRI Searchlight MVPA 分析结果总览', fontsize=20, fontweight='bold', y=0.98)
        
        # 保存图片
        viz_path = config.results_dir / "searchlight_main_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        log(f"主要可视化已保存: {viz_path}", config)
        return viz_path
        
    except Exception as e:
        print(f"创建主要可视化失败: {str(e)}")
        plt.close('all')
        return None

def create_individual_visualization(group_df, config):
    """
    创建个体结果可视化
    
    Args:
        group_df: 个体结果DataFrame
        config: 配置对象
    
    Returns:
        Path: 保存的图片路径
    """
    
    if group_df.empty:
        print("个体数据为空，跳过个体可视化")
        return None
    
    try:
        # 计算需要的子图数量
        n_contrasts = len(group_df['contrast'].unique())
        n_cols = min(3, n_contrasts)
        n_rows = (n_contrasts + n_cols - 1) // n_cols
        
        # 创建图形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_contrasts == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 为每个对比条件创建个体结果图
        for i, contrast in enumerate(group_df['contrast'].unique()):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col] if n_cols > 1 else axes[0]
            else:
                ax = axes[row, col]
            
            contrast_data = group_df[group_df['contrast'] == contrast]
            create_individual_contrast_plot(contrast_data, ax, contrast)
        
        # 隐藏多余的子图
        for i in range(n_contrasts, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        # 设置总标题
        fig.suptitle('个体被试 Searchlight MVPA 结果', fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        viz_path = config.results_dir / "searchlight_individual_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        log(f"个体可视化已保存: {viz_path}", config)
        return viz_path
        
    except Exception as e:
        print(f"创建个体可视化失败: {str(e)}")
        plt.close('all')
        return None

def create_group_accuracy_barplot(stats_df, ax):
    """创建组水平准确率条形图"""
    try:
        contrasts = stats_df['contrast'].tolist()
        accuracies = stats_df['mean_accuracy'].tolist()
        errors = stats_df['sem_accuracy'].tolist()
        
        # 根据显著性设置颜色
        colors = ['#2E8B57' if row.get('significant_permutation', False) or row.get('significant_t_test', False) 
                 else '#CD5C5C' for _, row in stats_df.iterrows()]
        
        bars = ax.bar(contrasts, accuracies, yerr=errors, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加准确率数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加机会水平线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='机会水平 (50%)')
        
        ax.set_title('组水平分类准确率', fontsize=14, fontweight='bold')
        ax.set_ylabel('平均准确率', fontsize=12)
        ax.set_xlabel('对比条件', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围
        ax.set_ylim(0.4, max(accuracies) + 0.1)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_accuracy_distribution(group_df, ax):
    """创建准确率分布图"""
    try:
        # 创建小提琴图
        sns.violinplot(data=group_df, x='contrast', y='accuracy', ax=ax, inner='box')
        
        # 添加散点
        sns.stripplot(data=group_df, x='contrast', y='accuracy', ax=ax, 
                     color='red', alpha=0.6, size=4)
        
        # 添加机会水平线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title('个体准确率分布', fontsize=12, fontweight='bold')
        ax.set_ylabel('准确率', fontsize=10)
        ax.set_xlabel('对比条件', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_effect_size_plot(stats_df, ax):
    """创建效应量可视化"""
    try:
        contrasts = stats_df['contrast'].tolist()
        effect_sizes = stats_df['cohens_d'].tolist()
        
        # 根据效应量大小设置颜色
        colors = ['#8B0000' if abs(d) >= 0.8 else '#FF6347' if abs(d) >= 0.5 
                 else '#FFA500' if abs(d) >= 0.2 else '#90EE90' for d in effect_sizes]
        
        bars = ax.bar(contrasts, effect_sizes, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # 添加效应量数值标签
        for bar, d in zip(bars, effect_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.1,
                   f'{d:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # 添加效应量参考线
        for threshold, label, color in [(0.2, '小', 'green'), (0.5, '中', 'orange'), (0.8, '大', 'red')]:
            ax.axhline(y=threshold, color=color, linestyle=':', alpha=0.5)
            ax.axhline(y=-threshold, color=color, linestyle=':', alpha=0.5)
        
        ax.set_title('效应量 (Cohen\'s d)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cohen\'s d', fontsize=10)
        ax.set_xlabel('对比条件', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_pvalue_comparison(stats_df, ax):
    """创建p值比较图"""
    try:
        contrasts = stats_df['contrast'].tolist()
        t_pvalues = stats_df['t_pvalue'].tolist()
        perm_pvalues = stats_df['permutation_pvalue'].tolist()
        
        x = np.arange(len(contrasts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, t_pvalues, width, label='t检验', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, perm_pvalues, width, label='置换检验', alpha=0.8, color='lightcoral')
        
        # 添加显著性水平线
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        
        ax.set_title('p值比较', fontsize=12, fontweight='bold')
        ax.set_ylabel('p值', fontsize=10)
        ax.set_xlabel('对比条件', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(contrasts, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_subject_consistency_heatmap(group_df, ax):
    """创建被试间一致性热图"""
    try:
        # 创建被试×对比条件的准确率矩阵
        pivot_df = group_df.pivot(index='subject', columns='contrast', values='accuracy')
        
        # 创建热图
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0.5, ax=ax, cbar_kws={'label': '准确率'})
        
        ax.set_title('被试间准确率一致性', fontsize=12, fontweight='bold')
        ax.set_xlabel('对比条件', fontsize=10)
        ax.set_ylabel('被试', fontsize=10)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_accuracy_vs_sample_size(group_df, ax):
    """创建准确率vs样本量散点图"""
    try:
        # 计算总样本量
        group_df['total_trials'] = group_df['n_trials_cond1'] + group_df['n_trials_cond2']
        
        # 为不同对比条件使用不同颜色
        contrasts = group_df['contrast'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(contrasts)))
        
        for i, contrast in enumerate(contrasts):
            contrast_data = group_df[group_df['contrast'] == contrast]
            ax.scatter(contrast_data['total_trials'], contrast_data['accuracy'], 
                      c=[colors[i]], label=contrast, alpha=0.7, s=50)
        
        # 添加机会水平线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='机会水平')
        
        ax.set_title('准确率 vs 样本量', fontsize=12, fontweight='bold')
        ax.set_xlabel('总试次数', fontsize=10)
        ax.set_ylabel('准确率', fontsize=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_significance_summary(stats_df, ax):
    """创建统计显著性总结"""
    try:
        # 计算显著性统计
        total_contrasts = len(stats_df)
        sig_t_test = sum(stats_df.get('significant_t_test', [False] * total_contrasts))
        sig_permutation = sum(stats_df.get('significant_permutation', [False] * total_contrasts))
        sig_both = sum([t and p for t, p in zip(
            stats_df.get('significant_t_test', [False] * total_contrasts),
            stats_df.get('significant_permutation', [False] * total_contrasts)
        )])
        
        # 创建饼图
        labels = ['t检验显著', '置换检验显著', '两者都显著', '都不显著']
        sizes = [sig_t_test - sig_both, sig_permutation - sig_both, sig_both, 
                total_contrasts - sig_t_test - sig_permutation + sig_both]
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightgray']
        
        # 只显示非零的部分
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
        if non_zero_indices:
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_sizes = [sizes[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            ax.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                  autopct='%1.0f', startangle=90)
        
        ax.set_title('显著性检验总结', fontsize=12, fontweight='bold')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_individual_contrast_plot(contrast_data, ax, contrast_name):
    """为单个对比条件创建个体结果图"""
    try:
        subjects = contrast_data['subject'].tolist()
        accuracies = contrast_data['accuracy'].tolist()
        
        # 根据准确率设置颜色
        colors = ['#2E8B57' if acc >= 0.6 else '#FFA500' if acc >= 0.55 else '#CD5C5C' 
                 for acc in accuracies]
        
        bars = ax.bar(subjects, accuracies, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # 添加准确率数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 添加机会水平线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        # 计算平均值线
        mean_acc = np.mean(accuracies)
        ax.axhline(y=mean_acc, color='blue', linestyle='-', alpha=0.7, 
                  label=f'平均: {mean_acc:.3f}')
        
        ax.set_title(f'{contrast_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('准确率', fontsize=9)
        ax.set_xlabel('被试', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围
        ax.set_ylim(0.4, max(accuracies) + 0.1)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_individual_searchlight_plots(group_df, config):
    """
    为每个被试创建单独的searchlight结果图
    
    Args:
        group_df: 个体结果DataFrame
        config: 配置对象
    
    Returns:
        list: 生成的图片路径列表
    """
    
    if group_df.empty:
        print("个体数据为空，跳过个体searchlight图")
        return []
    
    setup_chinese_fonts()
    plot_paths = []
    
    try:
        # 为每个被试创建图
        for subject in group_df['subject'].unique():
            subject_data = group_df[group_df['subject'] == subject]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            contrasts = subject_data['contrast'].tolist()
            accuracies = subject_data['accuracy'].tolist()
            
            # 根据准确率设置颜色
            colors = ['#2E8B57' if acc >= 0.6 else '#FFA500' if acc >= 0.55 else '#CD5C5C' 
                     for acc in accuracies]
            
            bars = ax.bar(contrasts, accuracies, color=colors, alpha=0.8, 
                         edgecolor='black', linewidth=1)
            
            # 添加准确率数值标签
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 添加机会水平线
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='机会水平 (50%)')
            
            ax.set_title(f'被试 {subject} - Searchlight MVPA 结果', fontsize=14, fontweight='bold')
            ax.set_ylabel('分类准确率', fontsize=12)
            ax.set_xlabel('对比条件', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # 设置y轴范围
            ax.set_ylim(0.4, max(accuracies) + 0.1)
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = config.results_dir / f"searchlight_{subject}_results.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            plot_paths.append(plot_path)
            log(f"被试 {subject} 的searchlight图已保存: {plot_path}", config)
        
        return plot_paths
        
    except Exception as e:
        print(f"创建个体searchlight图失败: {str(e)}")
        plt.close('all')
        return plot_paths