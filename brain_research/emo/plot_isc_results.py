#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_isc_results.py
功能：读取相似性矩阵 CSV，绘制发表级图表 (Heatmap, Histogram, MDS)。
适配：1000+ 被试，数值较低的情况。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import MDS
import matplotlib.transforms as transforms

# ================= 配置区 =================
# CSV 所在的文件夹
INPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_combined")

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
    matrix = df.values
    # 将对角线设为 NaN，防止 1.0 影响配色范围
    np.fill_diagonal(matrix, np.nan)
    
    # 自动计算合适的显示范围 (Robust scaling)
    # 取 98 分位数作为最大值，防止个别极值让图变黑
    # 取 2 分位数作为最小值
    valid_vals = matrix[~np.isnan(matrix)]
    vmax = np.percentile(valid_vals, 99)
    vmin = np.percentile(valid_vals, 1)
    
    plt.figure(figsize=(10, 8))
    
    # 使用 RdBu_r (红蓝) 配色，中心对齐 0 (白色)
    # 针对数值偏正的情况，使用 mako 或 magma 可能更好，但 RdBu_r 是相关性标准色
    ax = sns.heatmap(matrix, cmap="RdBu_r", center=0, 
                     vmin=vmin, vmax=vmax,
                     xticklabels=False, yticklabels=False, # 1000个标签太多，关掉
                     cbar_kws={'label': "Pearson Correlation (r)"})
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(f"Subjects (N={len(df)})")
    plt.ylabel(f"Subjects (N={len(df)})")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图1] Heatmap 已保存")

def plot_histogram(df, out_path, title):
    """绘制分布直方图 (证明 Signal > Noise)"""
    # 提取上三角矩阵 (不含对角线)
    matrix = df.values
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    values = matrix[mask]
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    plt.figure(figsize=(8, 6))
    
    # 绘制直方图 + KDE 曲线
    sns.histplot(values, bins=100, kde=True, color="#4c72b0", edgecolor=None, alpha=0.6)
    
    # 绘制 0 线
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # 绘制均值线
    plt.axvline(mean_val, color='#c44e52', linestyle='-', linewidth=2, label=f"Mean r={mean_val:.3f}")
    
    plt.title(f"Distribution of Inter-Subject Similarity\n({title})", fontsize=14)
    plt.xlabel("Pearson Correlation (r)")
    plt.ylabel("Count (Pairwise Combinations)")
    plt.legend()
    
    # 添加统计信息文本
    text_str = f"Mean = {mean_val:.3f}\nSD = {std_val:.3f}\nN Pairs = {len(values)}"
    plt.gca().text(0.95, 0.95, text_str, transform=plt.gca().transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图2] Histogram 已保存")

def plot_mds(df, out_path, title):
    """绘制 MDS 降维图 (展示人群结构)"""
    print("  正在计算 MDS (可能需要几分钟)...")
    
    # 距离矩阵 = 1 - 相关矩阵
    # 越相关，距离越近
    sim_matrix = df.values
    dist_matrix = 1 - sim_matrix
    
    # 使用 MDS 将 1000维 降到 2维
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, max_iter=300)
    embedding = mds.fit_transform(dist_matrix)
    
    plt.figure(figsize=(9, 9))
    
    # 散点图
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=15, c="#55a868", edgecolors='none')
    
    plt.title(f"MDS Embedding of Subjects\n({title})", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # 移除刻度数字，因为 MDS 坐标绝对值无意义，只有相对距离有意义
    plt.xticks([])
    plt.yticks([])
    
    # 添加说明
    plt.gca().text(0.02, 0.02, "Points closer together = More similar brains", 
                   transform=plt.gca().transAxes, fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [图3] MDS Plot 已保存")

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 扫描所有的 CSV 结果
    csv_files = list(INPUT_DIR.glob("similarity_*.csv"))
    
    if not csv_files:
        print(f"未在 {INPUT_DIR} 找到 CSV 文件！")
        return
        
    print(f"找到 {len(csv_files)} 个矩阵文件，开始绘图...")
    
    for f in csv_files:
        # 解析名字，例如 similarity_surface_L_Combined_Paired.csv
        # 提取 clean title: "Surface L (Combined)"
        name_clean = f.stem.replace("similarity_", "").replace("_Paired", "").replace("_Combined", "")
        # 美化标题
        if "surface" in name_clean:
            display_title = f"{name_clean} Hemisphere"
        else:
            display_title = "Subcortical Volume"
            
        print(f"\n>>> 处理: {f.name}")
        
        # 读取数据 (Index 是被试ID)
        df = pd.read_csv(f, index_col=0)
        
        # 1. Heatmap
        plot_heatmap(df, FIGURE_DIR / f"Heatmap_{f.stem}.png", display_title)
        
        # 2. Histogram
        plot_histogram(df, FIGURE_DIR / f"Hist_{f.stem}.png", display_title)
        
        # 3. MDS (可选，如果人太多太慢可以注释掉)
        plot_mds(df, FIGURE_DIR / f"MDS_{f.stem}.png", display_title)

    print(f"\n所有图片已保存至: {FIGURE_DIR}")

if __name__ == "__main__":
    main()