#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_supplement_similarity_distributions.py

补充材料推荐图：8 个相似性矩阵的上三角分布汇总（Facet）
默认零参数运行：从默认 final/roi AgeSorted 目录中读取 similarity_*AgeSorted.csv。
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FINAL_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
ROI_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_roi_age_sorted")
MAX_SAMPLES_PER_MATRIX = 200000


def _upper_values(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    v = mat[iu]
    v = v[np.isfinite(v)]
    return v.astype(float, copy=False)


def _load_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=float, copy=False)


def main() -> None:
    files: List[Path] = []
    if FINAL_DIR.exists():
        files.extend(sorted(FINAL_DIR.glob("similarity_*AgeSorted.csv")))
    if ROI_DIR.exists():
        files.extend(sorted(ROI_DIR.glob("similarity_*AgeSorted.csv")))

    if not files:
        print("未找到 similarity_*AgeSorted.csv（请先运行 calc_isc 脚本）")
        return

    rng = np.random.default_rng(42)
    rows = []
    for f in files:
        try:
            mat = _load_matrix(f)
            vals = _upper_values(mat)
            if vals.size == 0:
                continue
            if int(vals.size) > int(MAX_SAMPLES_PER_MATRIX):
                idx = rng.choice(int(vals.size), size=int(MAX_SAMPLES_PER_MATRIX), replace=False)
                vals = vals[idx]
            for x in vals:
                rows.append({"matrix": f.stem.replace("similarity_", ""), "r": float(x)})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("未能提取任何有效上三角值")
        return

    mats = sorted(df["matrix"].unique().tolist(), key=lambda s: (0 if s.startswith("surface_") else 1, s))
    df["matrix"] = pd.Categorical(df["matrix"], categories=mats, ordered=True)

    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    n = len(mats)
    ncols = 4 if n >= 4 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), sharex=True, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, m in enumerate(mats):
        ax = axes[i]
        sub = df[df["matrix"] == m]["r"].to_numpy(dtype=float)
        sns.histplot(sub, bins=60, kde=True, ax=ax, color="#4c72b0", alpha=0.65)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("r")
        ax.set_ylabel("Pairs")
        if sub.size > 0:
            mu = float(np.mean(sub))
            ax.text(0.98, 0.92, f"mean={mu:.3f}\nN={sub.size}", transform=ax.transAxes, ha="right", va="top", fontsize=8)

    for j in range(i + 1, axes.size):
        axes[j].axis("off")

    fig.suptitle("Supplementary: Distribution of Inter-Subject Similarity (Upper Triangle)", y=1.01)
    plt.tight_layout()
    out_dir = Path("./figures_supplement")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "S_similarity_distributions.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
