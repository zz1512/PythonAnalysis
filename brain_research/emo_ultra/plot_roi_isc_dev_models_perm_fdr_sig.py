#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_roi_isc_dev_models_perm_fdr_sig.py

Step 4 (FDR version):
对每个刺激类型，基于 Step3 的置换原始 p 值计算 BH-FDR，并绘制显著结果热图与导出汇总表。
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
    p = argparse.ArgumentParser(description="绘制 emo_ultra ROI ISC 发育模型显著结果热图（BH-FDR）")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fdr-mode", type=str, default="model_wise", choices=["model_wise", "global"])
    p.add_argument("--positive-only", action="store_true")
    return p.parse_args()


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float).reshape(-1)
    q = np.full(p.shape, np.nan, dtype=float)
    m = np.isfinite(p)
    if int(m.sum()) == 0:
        return q
    pv = p[m]
    n = int(pv.size)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * float(n) / np.arange(1, n + 1, dtype=float)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty_like(pv)
    q_valid[order] = q_ranked
    q[m] = q_valid
    return q


def calc_fdr(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["p_fdr_bh"] = np.nan
    if "p_perm_one_tailed" not in out.columns:
        raise ValueError("结果文件缺少列: p_perm_one_tailed")
    if mode == "model_wise":
        for _, idx in out.groupby("model").groups.items():
            out.loc[idx, "p_fdr_bh"] = bh_fdr(out.loc[idx, "p_perm_one_tailed"].to_numpy(dtype=float))
    elif mode == "global":
        out["p_fdr_bh"] = bh_fdr(out["p_perm_one_tailed"].to_numpy(dtype=float))
    else:
        raise ValueError(f"不支持 fdr mode: {mode}")
    return out


def collect_results(matrix_dir: Path, alpha: float, fdr_mode: str, positive_only: bool) -> pd.DataFrame:
    by_stim = matrix_dir / "by_stimulus"
    rows: List[pd.DataFrame] = []
    for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        p = d / "roi_isc_dev_models_perm_fwer.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        need = {"roi", "model", "r_obs", "p_perm_one_tailed"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{p} 缺少列: {sorted(miss)}")
        work = calc_fdr(df, mode=str(fdr_mode))
        sub = work[np.isfinite(work["r_obs"]) & np.isfinite(work["p_fdr_bh"])].copy()
        if bool(positive_only):
            sub = sub[sub["r_obs"] > 0]
        sub = sub[sub["p_fdr_bh"] <= float(alpha)]
        if sub.empty:
            continue
        sub["stimulus_type"] = d.name
        rows.append(sub)
    if not rows:
        raise ValueError("无显著结果可绘图")
    return pd.concat(rows, axis=0, ignore_index=True)


def draw(df: pd.DataFrame, out_dir: Path, alpha: float, dpi: int, fdr_mode: str, positive_only: bool) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fdr_{fdr_mode}"
    if bool(positive_only):
        tag += "_pos"
    sig_csv = out_dir / f"emo_ultra_sig_results_{tag}_a{alpha:g}.csv"
    df.sort_values(["stimulus_type", "model", "p_fdr_bh", "roi"]).to_csv(sig_csv, index=False)

    models = ["M_nn", "M_conv", "M_div"]
    stim_types = sorted(df["stimulus_type"].unique().tolist())
    roi_order = (
        df.groupby("roi", as_index=False)["p_fdr_bh"].min().sort_values("p_fdr_bh")["roi"].astype(str).tolist()
    )

    fig_h = max(4.5, 0.25 * len(roi_order) + 2.0)
    fig_w = max(6.0, 4.0 * len(stim_types))
    fig, axes = plt.subplots(1, len(stim_types), figsize=(fig_w, fig_h), sharey=True)
    if len(stim_types) == 1:
        axes = [axes]

    vmax = max(0.1, float(np.nanmax(np.abs(df["r_obs"].to_numpy(dtype=float)))))

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
            cbar_kws={"label": "r (FDR significant)"} if i == len(stim_types) - 1 else None,
        )
        ax.set_title(stim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel("")
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    title = f"emo_ultra significant ROI ISC effects (BH-FDR {fdr_mode}, q<= {alpha:g})"
    if bool(positive_only):
        title += " (positive only)"
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = out_dir / f"emo_ultra_sig_heatmap_{tag}_a{alpha:g}.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {"sig_csv": sig_csv, "figure": fig_path}


def main() -> None:
    args = parse_args()
    df = collect_results(Path(args.matrix_dir), float(args.alpha), str(args.fdr_mode), bool(args.positive_only))
    draw(df, Path(args.matrix_dir) / "figures", float(args.alpha), int(args.dpi), str(args.fdr_mode), bool(args.positive_only))


if __name__ == "__main__":
    main()

