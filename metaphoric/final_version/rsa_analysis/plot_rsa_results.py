#!/usr/bin/env python3
"""
plot_rsa_results.py
RSA 结果可视化脚本
功能：
1. 计算 Delta Similarity (Post - Pre)
2. 绘制 ROI x Condition 交互作用图
3. 绘制 Item-wise 漂移散点图

输入（默认相对于 `${PYTHON_METAPHOR_ROOT}`）
- `rsa_results_optimized/rsa_summary_stats.csv`
- `rsa_results_optimized/rsa_itemwise_details.csv`

输出
- `rsa_results_optimized/figures/`：图表文件（png 等）

注意
- 如果你修改了 rsa_config.py 的输出目录，请同步修改本脚本的 RESULTS_DIR，或把 RESULTS_DIR 改成从环境变量读取。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# ================= 配置 =================
BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
RESULTS_DIR = BASE_DIR / "rsa_results_optimized"
SUMMARY_FILE = RESULTS_DIR / "rsa_summary_stats.csv"
ITEMWISE_FILE = RESULTS_DIR / "rsa_itemwise_details.csv"
OUTPUT_DIR = RESULTS_DIR / "figures"

# 绘图风格设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150


# =======================================

def prep_delta_data(df):
    """
    数据清洗：计算 Delta (Post - Pre)
    """
    # 1. 转换宽表为长表 (Melt)
    df_long = df.melt(
        id_vars=['subject', 'roi', 'stage'],
        value_vars=['Sim_Metaphor', 'Sim_Spatial', 'Sim_Baseline'],
        var_name='Condition', value_name='Similarity'
    )

    # 清洗 Condition 名称 (去掉 'Sim_' 前缀)
    df_long['Condition'] = df_long['Condition'].str.replace('Sim_', '')

    # 2. Pivot 计算 Post - Pre
    # 将 stage 转为列
    df_pivot = df_long.pivot_table(
        index=['subject', 'roi', 'Condition'],
        columns='stage',
        values='Similarity'
    ).reset_index()

    # 计算增量 Delta
    df_pivot['Delta_Similarity'] = df_pivot['Post'] - df_pivot['Pre']

    return df_pivot


def plot_roi_interaction(df_delta):
    """
    Figure 3A: 交互作用柱状图 (Bar Plot with Error Bars)
    核心假设验证：IFG 里 Metaphor 高，Precuneus 里 Spatial 高
    """
    plt.figure(figsize=(10, 6))

    # 定义颜色：隐喻=红，空间=蓝，基线=灰
    palette = {"Metaphor": "#e74c3c", "Spatial": "#3498db", "Baseline": "#95a5a6"}

    # 绘图
    ax = sns.barplot(
        data=df_delta,
        x="roi",
        y="Delta_Similarity",
        hue="Condition",
        palette=palette,
        capsize=.1,
        err_kws={'linewidth': 1.5},
        alpha=0.9
    )

    # 装饰
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # 0值线
    plt.title("Similarity Change (Post - Pre) by ROI", fontweight='bold')
    plt.ylabel("Δ Similarity (Fisher's z)", fontsize=12)
    plt.xlabel("")
    plt.legend(title='Condition', loc='upper right')

    # 保存
    out_path = OUTPUT_DIR / "Fig3A_Interaction_Barplot.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"图表已保存: {out_path}")
    plt.show()


def plot_itemwise_scatter(df_item, roi_name="L_IFG"):
    """
    Figure 4A: 项目级散点图
    展示 40 个隐喻对是否都变了
    """
    # 1. 筛选特定 ROI 和 隐喻条件
    subset = df_item[
        (df_item['roi'] == roi_name) &
        (df_item['condition'] == 'Metaphor')
        ]

    if subset.empty:
        print(f"Warning: No data for ROI {roi_name}")
        return

    # 2. 聚合：计算每个 Pair 在所有被试上的均值
    pair_stats = subset.groupby(['word_label', 'stage'])['similarity'].mean().reset_index()

    # 3. Pivot 为 Pre vs Post
    pair_pivot = pair_stats.pivot(index='word_label', columns='stage', values='similarity').reset_index()

    # 4. 绘图
    plt.figure(figsize=(8, 8))

    # 画对角线 y=x
    min_val = min(pair_pivot['Pre'].min(), pair_pivot['Post'].min()) - 0.05
    max_val = max(pair_pivot['Pre'].max(), pair_pivot['Post'].max()) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Change')

    # 画散点
    sns.scatterplot(
        data=pair_pivot, x='Pre', y='Post',
        s=100, color="#e74c3c", edgecolor='w', alpha=0.8
    )

    # 标注几个离得最远的词 (变化最大的)
    pair_pivot['Change'] = pair_pivot['Post'] - pair_pivot['Pre']
    top_changers = pair_pivot.sort_values('Change', ascending=False).head(5)

    for _, row in top_changers.iterrows():
        plt.text(row['Pre'] + 0.005, row['Post'], row['word_label'], fontsize=9, alpha=0.8)

    plt.title(f"Metaphor Pair Similarity Shift in {roi_name}", fontweight='bold')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Pre-test Similarity")
    plt.ylabel("Post-test Similarity")

    # 统计检验 (Paired t-test across items)
    t, p = stats.ttest_rel(pair_pivot['Post'], pair_pivot['Pre'])
    plt.text(min_val + 0.02, max_val - 0.05, f"Paired t-test (N=40):\nt={t:.2f}, p={p:.4f}",
             bbox=dict(facecolor='white', alpha=0.8))

    out_path = OUTPUT_DIR / f"Fig4A_ItemScatter_{roi_name}.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"图表已保存: {out_path}")
    plt.show()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("正在加载数据...")
    if not SUMMARY_FILE.exists() or not ITEMWISE_FILE.exists():
        print("错误：找不到 CSV 文件，请先运行 rsa 分析脚本。")
        return

    df_sum = pd.read_csv(SUMMARY_FILE)
    df_item = pd.read_csv(ITEMWISE_FILE)

    print("1. 绘制交互作用图...")
    df_delta = prep_delta_data(df_sum)
    plot_roi_interaction(df_delta)

    print("2. 绘制 IFG 隐喻对散点图...")
    # 这里默认画 L_IFG，如果你有别的 ROI 名字，请修改
    # 检查一下 df_item 里 roi 列的名字
    roi_list = df_item['roi'].unique()
    for roi in roi_list:
        if "IFG" in roi or "Metaphor" in roi:  # 尝试找 IFG
            plot_itemwise_scatter(df_item, roi_name=roi)

    print("完成！")


if __name__ == "__main__":
    main()
