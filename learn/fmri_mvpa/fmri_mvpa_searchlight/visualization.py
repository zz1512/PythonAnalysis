# visualization.py
# 可视化模块 - Searchlight版本
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False
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

def setup_chinese_fonts(config=None):
    """设置中文字体支持，优先使用配置文件提供的字体。"""

    custom_font = None
    if config is not None:
        custom_path = getattr(config, 'chinese_font_path', None)
        if custom_path:
            font_path = Path(custom_path)
            if font_path.exists():
                try:
                    fm.fontManager.addfont(str(font_path))
                    custom_font = fm.FontProperties(fname=str(font_path)).get_name()
                    plt.rcParams['font.sans-serif'] = [custom_font] + list(plt.rcParams.get('font.sans-serif', []))
                    plt.rcParams['font.family'] = ['sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"使用自定义中文字体: {custom_font}")
                    return custom_font
                except Exception as e:
                    print(f"加载自定义中文字体失败 {font_path}: {e}")

    font_file_candidates = [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf'),
        Path('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf'),
        Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'),
        Path.home() / 'fonts' / 'NotoSansCJK-Regular.otf',
    ]

    for font_file in font_file_candidates:
        try:
            if font_file and font_file.exists():
                fm.fontManager.addfont(str(font_file))
        except Exception as e:
            print(f"加载字体文件失败 {font_file}: {e}")

    font_candidates = [
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Arial Unicode MS',
        'DejaVu Sans',
        'sans-serif'
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in font_candidates:
        if font_name == 'sans-serif' or font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + list(plt.rcParams.get('font.sans-serif', []))
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {font_name}")
            return font_name

    print("警告：未找到合适的中文字体，图表中的中文可能显示异常")
    plt.rcParams['axes.unicode_minus'] = False
    return None

def create_enhanced_visualizations(group_df, stats_df, config, significance_summary=None):
    """
    改进版：生成更美观、更有解释力的静态可视化结果
    - 支持中文字体
    - 组平均准确率与个体准确率并列显示
    - 添加显著性标注（星号或阈值线）
    - 输出base64编码图像以供HTML报告使用
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64

    # ======================
    # 通用绘图样式设置
    # ======================
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")

    # ======================
    # 一、组水平统计图
    # ======================
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    plt.subplots_adjust(hspace=0.4)

    # 上图：组水平准确率
    ax1 = axes[0]
    if stats_df is not None and not stats_df.empty:
        sns.barplot(
            data=stats_df,
            x='contrast',
            y='mean_accuracy',
            hue=None,
            ci='sd',
            ax=ax1,
            palette=sns.color_palette("RdYlBu_r", len(stats_df))
        )
        ax1.axhline(0.5, color='gray', linestyle='--', lw=1.5, label='机会水平 (0.5)')
        ax1.set_title("组水平平均准确率（Mean Accuracy by Contrast）", fontsize=14, weight='bold')
        ax1.set_ylabel("平均准确率", fontsize=12)
        ax1.set_xlabel("对比条件", fontsize=11)
        ax1.legend(loc='upper right')

        # 添加显著性标注（p<0.05加星号）
        for i, row in enumerate(stats_df.itertuples()):
            pval = getattr(row, 't_pvalue', 1.0)
            y = getattr(row, 'mean_accuracy', 0.0)
            if pval < 0.001:
                mark = '***'
            elif pval < 0.01:
                mark = '**'
            elif pval < 0.05:
                mark = '*'
            else:
                mark = ''
            if mark:
                ax1.text(i, y + 0.02, mark, ha='center', va='bottom', color='red', fontsize=12)

    else:
        ax1.text(0.5, 0.5, "无有效组水平结果", ha='center', va='center', fontsize=13)
        ax1.axis('off')

    # ======================
    # 二、单被试结果散点图
    # ======================
    ax2 = axes[1]
    if group_df is not None and not group_df.empty:
        sns.stripplot(
            data=group_df,
            x='contrast',
            y='accuracy',
            hue='subject',
            jitter=True,
            dodge=True,
            size=6,
            alpha=0.8,
            ax=ax2
        )
        ax2.axhline(0.5, color='gray', linestyle='--', lw=1.3)
        ax2.set_title("单被试准确率分布（Individual Accuracy Scatter）", fontsize=14, weight='bold')
        ax2.set_ylabel("准确率", fontsize=12)
        ax2.set_xlabel("对比条件", fontsize=11)
        ax2.legend(title='被试', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "无单被试数据", ha='center', va='center', fontsize=13)
        ax2.axis('off')

    # ======================
    # 保存主图（组水平 + 单被试）
    # ======================
    main_path = config.results_dir / "searchlight_main_visualization.png"
    plt.tight_layout()
    fig.savefig(main_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ======================
    # 三、显著性图（如果有）
    # ======================
    significance_path = None
    if significance_summary is not None and not significance_summary.empty:
        plt.figure(figsize=(9, 5))
        sig_data = significance_summary.copy()
        sns.barplot(
            data=sig_data,
            x='contrast',
            y='n_significant_voxels',
            hue='correction',
            palette='coolwarm'
        )
        plt.axhline(0, color='gray', lw=1)
        plt.title("显著性体素数（Significant Voxels per Contrast）", fontsize=14, weight='bold')
        plt.ylabel("显著体素数")
        plt.xlabel("对比条件")
        plt.legend(title="校正类型")
        significance_path = config.results_dir / "searchlight_significance_visualization.png"
        plt.tight_layout()
        plt.savefig(significance_path, dpi=200, bbox_inches='tight')
        plt.close()

    # ======================
    # 转换为base64以便HTML报告嵌入
    # ======================
    def fig_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8") if path.exists() else None

    main_b64 = fig_to_b64(main_path)
    individual_b64 = main_b64  # 同一张图兼容旧字段
    significance_b64 = fig_to_b64(significance_path) if significance_path else None

    log(f"✅ 可视化图像已生成: {main_path}", config)
    return main_path, main_path, significance_path


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
        chance_series = stats_df.get('mean_chance_accuracy', pd.Series([0.5]*len(stats_df)))
        chance_levels = chance_series.tolist()
        if 'mean_delta_accuracy' in stats_df.columns:
            deltas = stats_df['mean_delta_accuracy'].tolist()
        else:
            deltas = (stats_df['mean_accuracy'] - chance_series).tolist()
        errors_series = stats_df.get('sem_delta_accuracy', stats_df.get('sem_accuracy', pd.Series([0.0]*len(stats_df))))
        errors = errors_series.tolist()

        x = np.arange(len(contrasts))

        colors = ['#2E8B57' if row.get('significant_permutation', False) or row.get('significant_t_test', False)
                 else '#CD5C5C' for _, row in stats_df.iterrows()]

        bars = ax.bar(x, accuracies, yerr=errors, capsize=5,
                     color=colors, alpha=0.85, edgecolor='black', linewidth=1)

        ax.scatter(x, chance_levels, color='black', marker='D', s=60, label='机会水平', zorder=5)

        for idx, (bar, acc, delta) in enumerate(zip(bars, accuracies, deltas)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}\nΔ={delta:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(contrasts, rotation=0)

        ymin = min(min(chance_levels), min(accuracies)) - 0.05
        ymax = max(max(accuracies), max(chance_levels)) + 0.05
        ax.set_ylim(ymin, ymax)

        ax.set_title('组水平准确率与差值', fontsize=14, fontweight='bold')
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_xlabel('对比条件', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    except Exception as e:
        ax.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center', transform=ax.transAxes)

def create_accuracy_distribution(group_df, ax):
    """创建准确率分布图"""
    try:
        value_col = 'mean_delta_accuracy' if 'mean_delta_accuracy' in group_df.columns else 'accuracy'
        sns.violinplot(data=group_df, x='contrast', y=value_col, ax=ax, inner='box', color='#87CEEB')

        sns.stripplot(data=group_df, x='contrast', y=value_col, ax=ax,
                     color='darkred', alpha=0.7, size=4)

        if value_col == 'mean_delta_accuracy':
            ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.7)
            ax.set_title('个体准确率差值分布 (准确率-机会水平)', fontsize=12, fontweight='bold')
            ax.set_ylabel('准确率差值', fontsize=10)
        else:
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
        perm_pvalues = stats_df.get('sign_flip_pvalue', stats_df.get('permutation_pvalue', pd.Series([np.nan]*len(stats_df)))).tolist()
        
        x = np.arange(len(contrasts))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, t_pvalues, width, label='t检验', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, perm_pvalues, width, label='符号翻转置换', alpha=0.8, color='lightcoral')
        
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
        value_col = 'mean_delta_accuracy' if 'mean_delta_accuracy' in group_df.columns else 'accuracy'
        pivot_df = group_df.pivot(index='subject', columns='contrast', values=value_col)

        center = 0 if value_col == 'mean_delta_accuracy' else 0.5
        label = '准确率差值' if value_col == 'mean_delta_accuracy' else '准确率'

        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   center=center, ax=ax, cbar_kws={'label': label})

        title = '被试间准确率差值一致性' if value_col == 'mean_delta_accuracy' else '被试间准确率一致性'
        ax.set_title(title, fontsize=12, fontweight='bold')
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
        
        value_col = 'mean_delta_accuracy' if 'mean_delta_accuracy' in group_df.columns else 'accuracy'

        for i, contrast in enumerate(contrasts):
            contrast_data = group_df[group_df['contrast'] == contrast]
            ax.scatter(contrast_data['total_trials'], contrast_data[value_col],
                      c=[colors[i]], label=contrast, alpha=0.7, s=50)

        if value_col == 'mean_delta_accuracy':
            ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.7, label='Δ = 0')
            ax.set_ylabel('准确率差值', fontsize=10)
            ax.set_title('准确率差值 vs 样本量', fontsize=12, fontweight='bold')
        else:
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='机会水平')
            ax.set_ylabel('准确率', fontsize=10)
            ax.set_title('准确率 vs 样本量', fontsize=12, fontweight='bold')

        ax.set_xlabel('总试次数', fontsize=10)
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

def create_significant_voxel_visualization(significance_summary, config):
    """创建显著体素的统计可视化"""

    if significance_summary is None:
        log("显著性汇总为空，跳过显著性可视化", config)
        return None

    try:
        summary_df = pd.DataFrame(significance_summary).copy()
        if summary_df.empty:
            log("显著性汇总数据为空，跳过显著性可视化", config)
            return None

        setup_chinese_fonts(config)

        # 构建透视表用于绘图
        count_pivot = summary_df.pivot_table(
            index='contrast',
            columns='correction',
            values='n_significant_voxels',
            aggfunc='first'
        ).fillna(0)

        proportion_pivot = summary_df.pivot_table(
            index='contrast',
            columns='correction',
            values='proportion_significant',
            aggfunc='first'
        ).fillna(0) * 100.0

        threshold_info = summary_df.groupby('contrast')[
            ['classification_threshold', 'n_voxels_above_threshold']
        ].first()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        count_pivot.plot(kind='bar', ax=axes[0], alpha=0.85, edgecolor='black')
        axes[0].set_title('显著体素数量 (准确率 > 阈值)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('对比条件', fontsize=10)
        axes[0].set_ylabel('显著体素数', fontsize=10)
        axes[0].grid(True, axis='y', alpha=0.3)
        axes[0].legend(title='校正方式')

        proportion_pivot.plot(kind='bar', ax=axes[1], alpha=0.85, edgecolor='black')
        axes[1].set_title('显著体素占比 (相对于>阈值体素)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('对比条件', fontsize=10)
        axes[1].set_ylabel('占比 (%)', fontsize=10)
        axes[1].grid(True, axis='y', alpha=0.3)
        axes[1].legend(title='校正方式')

        # 在第二个子图添加阈值信息表
        table_data = [
            [contrast,
             f"> {row['classification_threshold']:.2f}",
             int(row['n_voxels_above_threshold'])]
            for contrast, row in threshold_info.iterrows()
        ]
        table = axes[1].table(
            cellText=table_data,
            colLabels=['对比条件', '分类阈值', '>阈值体素数'],
            loc='bottom',
            bbox=[0.0, -0.55, 1.0, 0.45]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        plt.tight_layout(rect=[0, 0.15, 1, 1])

        viz_path = config.results_dir / "searchlight_significance_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        log(f"显著性汇总可视化已保存: {viz_path}", config)
        return viz_path

    except Exception as e:
        print(f"创建显著性可视化失败: {str(e)}")
        plt.close('all')
        return None

def create_individual_contrast_plot(contrast_data, ax, contrast_name):
    """为单个对比条件创建个体结果图"""
    try:
        subjects = contrast_data['subject'].tolist()
        accuracies = contrast_data.get('mean_accuracy', contrast_data['accuracy']).tolist()
        chance_values = contrast_data.get('mean_chance_accuracy', pd.Series([0.5]*len(contrast_data))).tolist()
        deltas = contrast_data.get('mean_delta_accuracy', (contrast_data.get('mean_accuracy', contrast_data['accuracy']) - contrast_data.get('mean_chance_accuracy', 0.5))).tolist()

        colors = ['#2E8B57' if acc - chance >= 0.05 else '#FFA500' if acc - chance >= 0.02 else '#CD5C5C'
                 for acc, chance in zip(accuracies, chance_values)]

        bars = ax.bar(subjects, accuracies, color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1)

        ax.scatter(subjects, chance_values, color='black', marker='D', s=40, label='机会水平', zorder=5)

        for bar, acc, delta in zip(bars, accuracies, deltas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.3f}\nΔ={delta:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        mean_acc = np.mean(accuracies)
        mean_chance = np.mean(chance_values)
        ax.axhline(y=mean_chance, color='gray', linestyle='--', alpha=0.6, label=f'平均机会水平 {mean_chance:.3f}')
        ax.axhline(y=mean_acc, color='blue', linestyle='-', alpha=0.7,
                  label=f'平均准确率 {mean_acc:.3f}')

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
    
    setup_chinese_fonts(config)
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