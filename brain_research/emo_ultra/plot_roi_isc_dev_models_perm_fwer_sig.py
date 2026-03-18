#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_roi_isc_dev_models_perm_fwer_sig.py

Step 4:
汇总所有刺激类型的显著 ROI（正效应 + FWER），绘制热图并导出汇总表。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制 emo_ultra ROI ISC 发育模型显著结果热图")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def collect_results(matrix_dir: Path, alpha: float) -> pd.DataFrame:
    by_stim = matrix_dir / "by_stimulus"
    rows: List[pd.DataFrame] = []
    for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        p = d / "roi_isc_dev_models_perm_fwer.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sub = df[(df["r_obs"] > 0) & (df["p_fwer_model_wise"] <= float(alpha))].copy()
        if sub.empty:
            continue
        sub["stimulus_type"] = d.name
        rows.append(sub)
    if not rows:
        raise ValueError("无显著结果可绘图")
    return pd.concat(rows, axis=0, ignore_index=True)


def draw(df: pd.DataFrame, out_dir: Path, alpha: float, dpi: int) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_csv = out_dir / f"emo_ultra_sig_results_pos_fwer_a{alpha:g}.csv"
    df.sort_values(["stimulus_type", "model", "roi"]).to_csv(sig_csv, index=False)

    models = ["M_nn", "M_conv", "M_div"]
    stim_types = sorted(df["stimulus_type"].unique().tolist())

    # 以最显著 p 值排序 ROI
    roi_order = (
        df.groupby("roi", as_index=False)["p_fwer_model_wise"].min().sort_values("p_fwer_model_wise")["roi"].astype(str).tolist()
    )

    fig_h = max(4.5, 0.25 * len(roi_order) + 2.0)
    fig_w = max(6.0, 4.0 * len(stim_types))
    fig, axes = plt.subplots(1, len(stim_types), figsize=(fig_w, fig_h), sharey=True)
    if len(stim_types) == 1:
        axes = [axes]

    vmax = max(0.1, float(np.nanmax(np.abs(df["r_obs"].to_numpy()))))

    for i, stim in enumerate(stim_types):
        ax = axes[i]
        sub = df[df["stimulus_type"] == stim]
        mat = sub.pivot(index="roi", columns="model", values="r_obs").reindex(index=roi_order, columns=models)
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar=(i == len(stim_types) - 1),
            cbar_kws={"label": "r (positive significant)"} if i == len(stim_types) - 1 else None,
        )
        ax.set_title(stim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel("")
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle(f"emo_ultra significant ROI ISC effects (positive only, FWER<= {alpha:g})", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = out_dir / f"emo_ultra_sig_heatmap_pos_fwer_a{alpha:g}.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {"sig_csv": sig_csv, "figure": fig_path}


def main() -> None:
    args = parse_args()
    df = collect_results(args.matrix_dir, float(args.alpha))
    draw(df, args.matrix_dir / "figures", float(args.alpha), int(args.dpi))


if __name__ == "__main__":
    main()
