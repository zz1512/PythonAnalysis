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
import os
import re

# ================= 配置区 =================
# CSV 所在的文件夹：全脑：similarity_matrices_final；具体roi：similarity_matrices_roi
INPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
# INPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_roi_age_sorted")

# 图片输出文件夹
FIGURE_DIR = INPUT_DIR / "figures"

# 被试信息表路径（用于年龄注释；环境变量 SUBJECT_AGE_TABLE 可覆盖）
SUBJECT_INFO_PATH = Path(
    os.environ.get(
        "SUBJECT_AGE_TABLE",
        "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv",
    )
)
COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

# 绘图风格设置 (Paper 风格)
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


# ==========================================

def _parse_chinese_age_exact(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y = re.search(r"(\d+)\s*岁", s)
    m = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d = re.search(r"(\d+)\s*天", s)
    years = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    days = int(d.group(1)) if d else 0
    return years + (months / 12.0) + (days / 365.0)


def load_subject_ages_map(subject_info_path: Path) -> dict:
    path_str = str(subject_info_path)
    if not subject_info_path.exists():
        return {}
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        return {}

    if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        return {}

    age_map = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        age_map[sub_id] = _parse_chinese_age_exact(row[age_col])
    return age_map


def _ages_for_df(df: pd.DataFrame, age_map: dict) -> np.ndarray:
    subs = [str(x) for x in df.index.tolist()]
    ages = np.array([age_map.get(s, np.nan) for s in subs], dtype=float)
    return ages


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
    sns.heatmap(matrix, cmap="RdBu_r", center=0,
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


def plot_heatmap_with_age(df, ages, out_path, title):
    matrix = df.values.copy()
    np.fill_diagonal(matrix, np.nan)
    valid_vals = matrix[~np.isnan(matrix)]
    if len(valid_vals) == 0:
        print("  ⚠️ 警告：矩阵全为空，跳过 HeatmapWithAge")
        return

    vmax = np.percentile(valid_vals, 99)
    vmin = np.percentile(valid_vals, 1)
    ages = np.asarray(ages, dtype=float).reshape(-1)
    has_age = ages.size == df.shape[0] and np.isfinite(ages).all()

    fig_w, fig_h = 10, 9
    strip_h = 0.35 if has_age else 0.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=2 if has_age else 1, ncols=1, height_ratios=[strip_h, 1.0] if has_age else [1.0], hspace=0.05)

    ax_strip = None
    if has_age:
        ax_strip = fig.add_subplot(gs[0, 0])
        cmap = plt.get_cmap("viridis")
        age_img = ages[np.newaxis, :]
        im = ax_strip.imshow(age_img, aspect="auto", cmap=cmap)
        ax_strip.set_yticks([])
        ax_strip.set_xticks([])
        cbar = fig.colorbar(im, ax=ax_strip, orientation="horizontal", fraction=0.9, pad=0.25)
        cbar.set_label("Age")

    ax = fig.add_subplot(gs[1, 0] if has_age else gs[0, 0])
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson Correlation (r)"},
    )
    if has_age and df.shape[0] >= 8:
        n = int(df.shape[0])
        bounds = [int(round(n * k / 4.0)) for k in (1, 2, 3)]
        bounds = [b for b in bounds if 0 < b < n]
        for b in bounds:
            ax.axvline(b, color="black", linewidth=0.9, alpha=0.65)
            ax.axhline(b, color="black", linewidth=0.9, alpha=0.65)
    ax.set_title(title, fontsize=16, pad=16)
    ax.set_xlabel(f"Subjects (N={len(df)})")
    ax.set_ylabel(f"Subjects (N={len(df)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图1b] Heatmap(+Age) 已保存")


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


def plot_mds(df, out_path, title, ages=None):
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
        ages_arr = None if ages is None else np.asarray(ages, dtype=float).reshape(-1)
        if ages_arr is not None and ages_arr.size == embedding.shape[0] and np.isfinite(ages_arr).any():
            sc = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                alpha=0.75,
                s=22,
                c=ages_arr,
                cmap="viridis",
                edgecolors="none",
            )
            cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
            cbar.set_label("Age")
            m = np.isfinite(ages_arr)
            if m.sum() >= 10:
                r1 = float(np.corrcoef(embedding[m, 0], ages_arr[m])[0, 1])
                r2 = float(np.corrcoef(embedding[m, 1], ages_arr[m])[0, 1])
                title2 = f"{title}\nMDS Dim-Age corr: r(dim1)={r1:.2f}, r(dim2)={r2:.2f}"
            else:
                title2 = title
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=15, c="#55a868", edgecolors="none")
            title2 = title

        plt.title(f"MDS Embedding of Subjects\n({title2})", fontsize=14)
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


def plot_umap(df, out_path, title, ages=None):
    try:
        import umap  # type: ignore
    except Exception:
        print("  ⚠️ UMAP 不可用（未安装 umap-learn），跳过 UMAP 图")
        return

    sim_matrix = df.values.copy()
    np.fill_diagonal(sim_matrix, 1.0)
    if np.isnan(sim_matrix).any():
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)
    dist_matrix = 1 - sim_matrix
    dist_matrix[dist_matrix < 0] = 0

    try:
        reducer = umap.UMAP(n_components=2, metric="precomputed", random_state=42, n_neighbors=20, min_dist=0.2)
        emb = reducer.fit_transform(dist_matrix)

        plt.figure(figsize=(9, 9))
        ages_arr = None if ages is None else np.asarray(ages, dtype=float).reshape(-1)
        if ages_arr is not None and ages_arr.size == emb.shape[0] and np.isfinite(ages_arr).any():
            sc = plt.scatter(emb[:, 0], emb[:, 1], c=ages_arr, cmap="viridis", s=22, alpha=0.78, edgecolors="none")
            cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
            cbar.set_label("Age")
        else:
            plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.7, c="#55a868", edgecolors="none")

        plt.title(f"UMAP Embedding of Subjects\n({title})", fontsize=14)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("  [图3b] UMAP Plot 已保存")
    except Exception as e:
        print(f"  ❌ UMAP 计算失败: {e}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(INPUT_DIR.glob("similarity_*AgeSorted.csv"))
    if not csv_files:
        csv_files = list(INPUT_DIR.glob("similarity_*.csv"))

    if not csv_files:
        print(f"未在 {INPUT_DIR} 找到 CSV 文件！")
        return

    print(f"找到 {len(csv_files)} 个矩阵文件，开始绘图...")
    age_map = load_subject_ages_map(SUBJECT_INFO_PATH)
    if age_map:
        print(f"已加载年龄表: {SUBJECT_INFO_PATH}")
    else:
        print(f"⚠️ 未能加载年龄表（将跳过年龄注释）: {SUBJECT_INFO_PATH}")

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
            ages = _ages_for_df(df, age_map) if age_map else None

            # 1. Heatmap
            plot_heatmap(df, FIGURE_DIR / f"Heatmap_{f.stem}.png", display_title)
            plot_heatmap_with_age(df, ages, FIGURE_DIR / f"HeatmapAge_{f.stem}.png", display_title)

            # 2. Histogram
            plot_histogram(df, FIGURE_DIR / f"Hist_{f.stem}.png", display_title)

            # 3. MDS
            plot_mds(df, FIGURE_DIR / f"MDS_{f.stem}.png", display_title, ages=ages)
            plot_umap(df, FIGURE_DIR / f"UMAP_{f.stem}.png", display_title, ages=ages)

        except Exception as e:
            print(f"  ❌ 处理文件失败: {e}")

    print(f"\n所有图片已保存至: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
