#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_isc_results.py
绘制 被试*被试相关矩阵 分析结果的热图和直方图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import MDS
import matplotlib.transforms as transforms

# ================= 配置区 =================
# CSV 所在的文件夹：全脑：similarity_matrices_final；具体roi：similarity_matrices_roi
INPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final")
# INPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_roi")

# 图片输出文件夹
FIGURE_DIR = INPUT_DIR / "figures"

# 绘图风格设置 (Paper 风格)
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


# ==========================================

def plot_heatmap(df, out_path, title):
    """绘制增强对比度的热图"""
    # 【修复】使用 copy() 防止修改原数据影响后续步骤
    matrix = df.values.copy()

    # 将对角线设为 NaN，防止 1.0 影响配色范围
    np.fill_diagonal(matrix, np.nan)

    # 自动计算合适的显示范围 (Robust scaling)
    valid_vals = matrix[~np.isnan(matrix)]

    if len(valid_vals) == 0:
        print("  ⚠️ 警告：矩阵全为空，跳过 Heatmap")
        return

    vmax = np.percentile(valid_vals, 99)
    vmin = np.percentile(valid_vals, 1)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, cmap="RdBu_r", center=0,
                     vmin=vmin, vmax=vmax,
                     xticklabels=False, yticklabels=False,
                     cbar_kws={'label': "Pearson Correlation (r)"})

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(f"Subjects (N={len(df)})")
    plt.ylabel(f"Subjects (N={len(df)})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图1] Heatmap 已保存")


def plot_histogram(df, out_path, title):
    """绘制分布直方图"""
    matrix = df.values
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    values = matrix[mask]

    # 移除 NaN
    values = values[~np.isnan(values)]

    if len(values) == 0:
        print("  ⚠️ 警告：没有有效数值，跳过 Histogram")
        return

    mean_val = np.mean(values)
    std_val = np.std(values)

    plt.figure(figsize=(8, 6))
    sns.histplot(values, bins=100, kde=True, color="#4c72b0", edgecolor=None, alpha=0.6)

    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.axvline(mean_val, color='#c44e52', linestyle='-', linewidth=2, label=f"Mean r={mean_val:.3f}")

    plt.title(f"Distribution of Inter-Subject Similarity\n({title})", fontsize=14)
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Count (Pairwise Combinations)")
    plt.legend()

    text_str = f"Mean = {mean_val:.3f}\nSD = {std_val:.3f}\nN Pairs = {len(values)}"
    plt.gca().text(0.95, 0.95, text_str, transform=plt.gca().transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图2] Histogram 已保存")


def plot_mds(df, out_path, title):
    """绘制 MDS 降维图"""
    print("  正在计算 MDS (可能需要几分钟)...")

    sim_matrix = df.values.copy()

    # 【核心修复】处理 NaN
    # 1. 填充对角线为 1.0 (自己和自己相关是1)
    np.fill_diagonal(sim_matrix, 1.0)

    # 2. 如果还有其他 NaN，说明某些被试数据有问题，填充为 0 (无相关)
    if np.isnan(sim_matrix).any():
        print("  ⚠️ 注意：矩阵包含 NaN，MDS计算前已自动替换为 0")
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

    # 3. 转换为距离矩阵
    dist_matrix = 1 - sim_matrix

    # 确保距离非负 (浮点误差可能导致微小的负数)
    dist_matrix[dist_matrix < 0] = 0

    try:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, max_iter=300)
        embedding = mds.fit_transform(dist_matrix)

        plt.figure(figsize=(9, 9))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=15, c="#55a868", edgecolors='none')

        plt.title(f"MDS Embedding of Subjects\n({title})", fontsize=14)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xticks([])
        plt.yticks([])

        plt.gca().text(0.02, 0.02, "Points closer together = More similar brains",
                       transform=plt.gca().transAxes, fontsize=10, color='gray')

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [图3] MDS Plot 已保存")

    except Exception as e:
        print(f"  ❌ MDS 计算失败: {e}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(INPUT_DIR.glob("similarity_*.csv"))

    if not csv_files:
        print(f"未在 {INPUT_DIR} 找到 CSV 文件！")
        return

    print(f"找到 {len(csv_files)} 个矩阵文件，开始绘图...")

    for f in csv_files:
        # 提取 clean title
        name_clean = f.stem.replace("similarity_", "").replace("_ContentAligned", "")

        if "surface" in name_clean.lower():
            display_title = f"{name_clean} Hemisphere"
        else:
            display_title = "Subcortical Volume"

        print(f"\n>>> 处理: {f.name}")

        # 读取数据
        try:
            df = pd.read_csv(f, index_col=0)

            # 1. Heatmap
            plot_heatmap(df, FIGURE_DIR / f"Heatmap_{f.stem}.png", display_title)

            # 2. Histogram
            plot_histogram(df, FIGURE_DIR / f"Hist_{f.stem}.png", display_title)

            # 3. MDS
            plot_mds(df, FIGURE_DIR / f"MDS_{f.stem}.png", display_title)

        except Exception as e:
            print(f"  ❌ 处理文件失败: {e}")

    print(f"\n所有图片已保存至: {FIGURE_DIR}")


if __name__ == "__main__":
    main()