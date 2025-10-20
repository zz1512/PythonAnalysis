# visualization.py
# 可视化模块

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import log

def create_enhanced_visualizations(group_df, stats_df, config):
    """创建增强的可视化"""
    log("创建可视化", config)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建多个图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 单个ROI带数据点的条形图
    ax1 = plt.subplot(3, 2, 1)
    roi_stats = stats_df.sort_values('mean_accuracy', ascending=False)
    
    # 绘制条形图
    bars = ax1.bar(range(len(roi_stats)), roi_stats['mean_accuracy'], 
                   color=['#2E8B57' if sig else '#CD5C5C' for sig in roi_stats['significant']],
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    # 添加误差线
    ax1.errorbar(range(len(roi_stats)), roi_stats['mean_accuracy'], 
                yerr=roi_stats['sem_accuracy'], fmt='none', color='black', capsize=5)
    
    # 添加个体数据点
    for i, roi in enumerate(roi_stats['roi']):
        roi_data = group_df[group_df['roi'] == roi]['mean_accuracy']
        x_jitter = np.random.normal(i, 0.05, len(roi_data))
        ax1.scatter(x_jitter, roi_data, alpha=0.6, s=30, color='black', zorder=3)
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='随机水平(50%)')
    ax1.set_xlabel('ROI区域', fontsize=12)
    ax1.set_ylabel('分类准确率', fontsize=12)
    ax1.set_title('各ROI分类准确率（带个体数据点）', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(roi_stats)))
    ax1.set_xticklabels(roi_stats['roi'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加显著性标记
    for i, (_, row) in enumerate(roi_stats.iterrows()):
        if row['significant']:
            ax1.text(i, row['mean_accuracy'] + row['sem_accuracy'] + 0.02, 
                    f'p={row["p_value_permutation"]:.3f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. 多ROI横向比较条形图
    ax2 = plt.subplot(3, 2, 2)
    significant_rois = roi_stats[roi_stats['significant']]
    if len(significant_rois) > 0:
        bars2 = ax2.barh(range(len(significant_rois)), significant_rois['mean_accuracy'],
                        color='#2E8B57', alpha=0.7, edgecolor='black')
        ax2.errorbar(significant_rois['mean_accuracy'], range(len(significant_rois)),
                    xerr=significant_rois['sem_accuracy'], fmt='none', color='black', capsize=5)
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_yticks(range(len(significant_rois)))
        ax2.set_yticklabels(significant_rois['roi'])
        ax2.set_xlabel('分类准确率', fontsize=12)
        ax2.set_title('显著ROI横向比较', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加效应量信息
        for i, (_, row) in enumerate(significant_rois.iterrows()):
            ax2.text(row['mean_accuracy'] + 0.01, i, 
                    f'd={row["cohens_d"]:.2f}', 
                    va='center', fontsize=9)
    else:
        ax2.text(0.5, 0.5, '无显著ROI', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16)
        ax2.set_title('显著ROI横向比较', fontsize=14, fontweight='bold')
    
    # 3. 效应量分布图
    ax3 = plt.subplot(3, 2, 3)
    effect_sizes = np.abs(roi_stats['cohens_d'])
    colors = ['#8B0000' if d >= 0.8 else '#FF6347' if d >= 0.5 else '#FFA500' if d >= 0.2 else '#D3D3D3' 
              for d in effect_sizes]
    bars3 = ax3.bar(range(len(roi_stats)), effect_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.2, color='orange', linestyle=':', alpha=0.8, label='小效应(0.2)')
    ax3.axhline(y=0.5, color='red', linestyle=':', alpha=0.8, label='中等效应(0.5)')
    ax3.axhline(y=0.8, color='darkred', linestyle=':', alpha=0.8, label='大效应(0.8)')
    ax3.set_xlabel('ROI区域', fontsize=12)
    ax3.set_ylabel('效应量 |Cohen\'s d|', fontsize=12)
    ax3.set_title('各ROI效应量分布', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(roi_stats)))
    ax3.set_xticklabels(roi_stats['roi'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 统计显著性概览
    ax4 = plt.subplot(3, 2, 4)
    sig_counts = roi_stats['significant'].value_counts()
    colors_pie = ['#2E8B57', '#CD5C5C']
    labels = ['显著', '不显著']
    wedges, texts, autotexts = ax4.pie(sig_counts.values, labels=labels, autopct='%1.1f%%', 
                                      colors=colors_pie, startangle=90)
    ax4.set_title('统计显著性分布', fontsize=14, fontweight='bold')
    
    # 5. 准确率分布直方图
    ax5 = plt.subplot(3, 2, 5)
    all_accuracies = group_df['mean_accuracy']
    ax5.hist(all_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='随机水平')
    ax5.axvline(x=all_accuracies.mean(), color='green', linestyle='-', alpha=0.8, linewidth=2, 
               label=f'平均值({all_accuracies.mean():.3f})')
    ax5.set_xlabel('分类准确率', fontsize=12)
    ax5.set_ylabel('频次', fontsize=12)
    ax5.set_title('所有数据点准确率分布', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. ROI间相关性热图
    ax6 = plt.subplot(3, 2, 6)
    roi_pivot = group_df.pivot_table(values='mean_accuracy', index='subject', columns='roi')
    if roi_pivot.shape[1] > 1:
        corr_matrix = roi_pivot.corr()
        im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax6.set_xticks(range(len(corr_matrix.columns)))
        ax6.set_yticks(range(len(corr_matrix.columns)))
        ax6.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax6.set_yticklabels(corr_matrix.columns)
        ax6.set_title('ROI间准确率相关性', fontsize=14, fontweight='bold')
        
        # 添加相关系数文本
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax6, shrink=0.8)
    else:
        ax6.text(0.5, 0.5, '需要多个ROI\n才能计算相关性', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('ROI间准确率相关性', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    viz_path = config.results_dir / "enhanced_analysis_visualizations.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"可视化图表已保存: {viz_path}", config)
    
    # 创建单个ROI详细图表
    create_individual_roi_plots(group_df, stats_df, config)
    
    return viz_path

def create_individual_roi_plots(group_df, stats_df, config):
    """为每个ROI创建详细的个体图表"""
    log("创建单个ROI详细图表", config)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取ROI列表
    rois = stats_df['roi'].unique()
    n_rois = len(rois)
    
    # 计算子图布局
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rois == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, roi in enumerate(rois):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # 获取该ROI的数据
        roi_data = group_df[group_df['roi'] == roi]['mean_accuracy']
        roi_stats = stats_df[stats_df['roi'] == roi].iloc[0]
        
        # 绘制条形图
        color = '#2E8B57' if roi_stats['significant'] else '#CD5C5C'
        bar = ax.bar([0], [roi_stats['mean_accuracy']], color=color, alpha=0.7, 
                    edgecolor='black', linewidth=2, width=0.6)
        
        # 添加误差线
        ax.errorbar([0], [roi_stats['mean_accuracy']], 
                   yerr=[roi_stats['sem_accuracy']], 
                   fmt='none', color='black', capsize=10, capthick=2)
        
        # 添加个体数据点
        x_jitter = np.random.normal(0, 0.05, len(roi_data))
        scatter = ax.scatter(x_jitter, roi_data, alpha=0.8, s=50, 
                           color='darkblue', edgecolor='white', linewidth=1, zorder=3)
        
        # 添加随机水平线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                  label='随机水平(50%)')
        
        # 设置标题和标签
        significance_text = "显著" if roi_stats['significant'] else "不显著"
        ax.set_title(f'{roi}\n({significance_text})', fontsize=12, fontweight='bold')
        ax.set_ylabel('分类准确率', fontsize=10)
        ax.set_xlim(-0.4, 0.4)
        ax.set_xticks([])
        
        # 添加统计信息文本框
        stats_text = f"""均值±标准误: {roi_stats['mean_accuracy']:.3f}±{roi_stats['sem_accuracy']:.3f}
t({roi_stats['n_subjects']-1}) = {roi_stats['t_statistic']:.3f}
p = {roi_stats['p_value_permutation']:.3f}
Cohen's d = {roi_stats['cohens_d']:.3f}
95% CI: [{roi_stats['ci_lower']:.3f}, {roi_stats['ci_upper']:.3f}]"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=8)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_rois, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        elif n_cols > 1:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图像
    individual_viz_path = config.results_dir / "individual_roi_detailed_plots.png"
    plt.savefig(individual_viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"单个ROI详细图表已保存: {individual_viz_path}", config)
    
    return individual_viz_path