#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_roi_isc_dev_models_perm_sig.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制 emo_ultra ROI ISC 发育模型显著结果热图（FWER/FDR）")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--sig-method", type=str, default=None, choices=["fwer", "fdr_model_wise", "fdr_global"])
    p.add_argument("--fdr-mode", type=str, default="model_wise", choices=["model_wise", "global"])
    p.add_argument("--positive-only", action="store_true")
    return p.parse_args()


def resolve_sig_method(sig_method: Optional[str], fdr_mode: str) -> str:
    if sig_method is not None:
        return str(sig_method)
    fm = str(fdr_mode).strip().lower()
    if fm == "model_wise":
        return "fdr_model_wise"
    if fm == "global":
        return "fdr_global"
    raise ValueError(f"不支持 fdr mode: {fdr_mode}")


def collect_results(matrix_dir: Path, stimulus_dir_name: str, alpha: float, sig_method: str, positive_only: bool) -> pd.DataFrame:
    by_stim = matrix_dir / str(stimulus_dir_name)
    rows: List[pd.DataFrame] = []
    for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        p = d / "roi_isc_dev_models_perm_fwer.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        need = {"roi", "model", "r_obs"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{p} 缺少列: {sorted(miss)}")
        sm = str(sig_method)
        if sm == "fwer":
            pcol = "p_fwer_model_wise"
            if pcol not in df.columns:
                raise ValueError(f"{p} 缺少列: {pcol}")
            sub = df[np.isfinite(df["r_obs"]) & np.isfinite(df[pcol])].copy()
            sub = sub[sub["r_obs"] > 0]
            sub = sub[sub[pcol] <= float(alpha)]
            sub = sub.rename(columns={pcol: "p_sig"}).copy()
        elif sm == "fdr_model_wise":
            pcol = "p_fdr_bh_model_wise"
            if pcol not in df.columns:
                raise ValueError(f"{p} 缺少列: {pcol}")
            sub = df[np.isfinite(df["r_obs"]) & np.isfinite(df[pcol])].copy()
            if bool(positive_only):
                sub = sub[sub["r_obs"] > 0]
            sub = sub[sub[pcol] <= float(alpha)]
            sub = sub.rename(columns={pcol: "p_sig"}).copy()
        elif sm == "fdr_global":
            pcol = "p_fdr_bh_global"
            if pcol not in df.columns:
                raise ValueError(f"{p} 缺少列: {pcol}")
            sub = df[np.isfinite(df["r_obs"]) & np.isfinite(df[pcol])].copy()
            if bool(positive_only):
                sub = sub[sub["r_obs"] > 0]
            sub = sub[sub[pcol] <= float(alpha)]
            sub = sub.rename(columns={pcol: "p_sig"}).copy()
        else:
            raise ValueError(f"不支持 sig-method: {sig_method}")
        if sub.empty:
            continue
        sub["stimulus_type"] = d.name
        rows.append(sub)
    if not rows:
        raise ValueError("无显著结果可绘图")
    return pd.concat(rows, axis=0, ignore_index=True)


def draw(df: pd.DataFrame, out_dir: Path, alpha: float, dpi: int, sig_method: str, positive_only: bool) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(sig_method)
    if bool(positive_only):
        tag += "_pos"
    sig_csv = out_dir / f"emo_ultra_sig_results_{tag}_a{alpha:g}.csv"
    df.sort_values(["stimulus_type", "model", "p_sig", "roi"]).to_csv(sig_csv, index=False)

    models = ["M_nn", "M_conv", "M_div"]
    stim_types = sorted(df["stimulus_type"].unique().tolist())
    roi_order = df.groupby("roi", as_index=False)["p_sig"].min().sort_values("p_sig")["roi"].astype(str).tolist()

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
            cbar_kws={"label": "r (significant)"} if i == len(stim_types) - 1 else None,
        )
        ax.set_title(stim)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel("")
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    title = f"emo_ultra significant ROI ISC effects ({sig_method}, <= {alpha:g})"
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
    sig_method = resolve_sig_method(args.sig_method, str(args.fdr_mode))
    df = collect_results(Path(args.matrix_dir), str(args.stimulus_dir_name), float(args.alpha), str(sig_method), bool(args.positive_only))
    draw(df, Path(args.matrix_dir) / "figures", float(args.alpha), int(args.dpi), str(sig_method), bool(args.positive_only))


if __name__ == "__main__":
    main()
