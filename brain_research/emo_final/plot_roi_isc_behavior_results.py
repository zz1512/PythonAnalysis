#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot brain-behavior ROI association results")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus", choices=("by_stimulus", "by_emotion"))
    p.add_argument("--result-file", type=str, default="roi_isc_behavior_perm_fwer.csv")
    p.add_argument("--p-col", type=str, default="p_fdr_bh_global", choices=("p_fdr_bh_global", "p_fwer_model_wise", "p_perm", "p_perm_one_tailed"))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-rois-heatmap", type=int, default=40)
    p.add_argument("--min-abs-r", type=float, default=0.0)
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--negative-only", action="store_true")
    p.add_argument("--no-show-all-if-no-sig", action="store_false", dest="show_all_if_no_sig", default=True)
    return p.parse_args()


def _branch_tag(stimulus_dir_name: str) -> str:
    return "trial" if str(stimulus_dir_name) == "by_stimulus" else "emotion"


def _sign_filter(df: pd.DataFrame, positive_only: bool, negative_only: bool) -> pd.DataFrame:
    out = df.copy()
    if bool(positive_only) and bool(negative_only):
        raise ValueError("Cannot use both --positive-only and --negative-only")
    if bool(positive_only):
        out = out[out["r_obs"].astype(float) > 0].copy()
    if bool(negative_only):
        out = out[out["r_obs"].astype(float) < 0].copy()
    return out


def collect_results(
    matrix_dir: Path,
    stimulus_dir_name: str,
    result_file: str,
    p_col: str,
    alpha: float,
    min_abs_r: float,
    positive_only: bool,
    negative_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    rows_all: List[pd.DataFrame] = []
    rows_sig: List[pd.DataFrame] = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        csv_path = stim_dir / str(result_file)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        need = {"roi", "r_obs", str(p_col)}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{csv_path} missing columns: {sorted(miss)}")

        sub = df[np.isfinite(df["r_obs"].to_numpy(dtype=float)) & np.isfinite(df[str(p_col)].to_numpy(dtype=float))].copy()
        if sub.empty:
            continue
        sub["stimulus_type"] = stim_dir.name
        sub = _sign_filter(sub, positive_only=bool(positive_only), negative_only=bool(negative_only))
        sub = sub[np.abs(sub["r_obs"].to_numpy(dtype=float)) >= float(min_abs_r)].copy()
        if sub.empty:
            continue

        rows_all.append(sub.copy())
        sig = sub[sub[str(p_col)].astype(float) <= float(alpha)].copy()
        if not sig.empty:
            rows_sig.append(sig)

    if not rows_all:
        raise ValueError("No usable brain-behavior result files found")
    df_all = pd.concat(rows_all, axis=0, ignore_index=True)
    df_sig = pd.concat(rows_sig, axis=0, ignore_index=True) if rows_sig else pd.DataFrame(columns=df_all.columns)
    return df_all, df_sig


def _rank_rois(df: pd.DataFrame) -> List[str]:
    rank_df = (
        df.groupby("roi", as_index=False)
        .agg(
            p_best=("p_sig", "min"),
            r_abs_max=("r_obs", lambda x: float(np.nanmax(np.abs(pd.to_numeric(x, errors="coerce"))))),
        )
        .sort_values(["p_best", "r_abs_max", "roi"], ascending=[True, False, True])
    )
    return rank_df["roi"].astype(str).tolist()


def draw_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    branch_tag: str,
    p_col: str,
    alpha: float,
    max_rois: int,
    dpi: int,
    positive_only: bool,
    negative_only: bool,
) -> Dict[str, Path]:
    if df.empty:
        raise ValueError("No rows available to draw heatmap")

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.rename(columns={str(p_col): "p_sig"}).copy()
    roi_order = _rank_rois(plot_df)[: int(max_rois)]
    stim_order = sorted(plot_df["stimulus_type"].astype(str).unique().tolist())
    mat = (
        plot_df.groupby(["stimulus_type", "roi"], as_index=False)["r_obs"]
        .mean()
        .pivot(index="stimulus_type", columns="roi", values="r_obs")
        .reindex(index=stim_order, columns=roi_order)
    )
    vmax = float(np.nanmax(np.abs(mat.to_numpy(dtype=float)))) if np.isfinite(mat.to_numpy(dtype=float)).any() else 0.1
    vmax = max(vmax, 0.1)

    fig_w = max(8.0, 0.38 * len(roi_order) + 3.0)
    fig_h = max(4.5, 0.42 * len(stim_order) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "brain-behavior r_obs"},
    )
    ax.set_title(f"Brain-behavior ROI heatmap ({branch_tag}, {p_col} <= {alpha:g})")
    ax.set_xlabel("ROI")
    ax.set_ylabel("stimulus_type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    sign_tag = "all"
    if bool(positive_only):
        sign_tag = "pos"
    elif bool(negative_only):
        sign_tag = "neg"
    fig_path = out_dir / f"brain_behavior_heatmap_{branch_tag}_{p_col}_a{alpha:g}_{sign_tag}.png"
    csv_path = out_dir / f"brain_behavior_heatmap_{branch_tag}_{p_col}_a{alpha:g}_{sign_tag}.csv"
    fig.savefig(fig_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    mat.to_csv(csv_path)
    return {"figure": fig_path, "matrix_csv": csv_path}


def draw_top_bars(
    df_sig: pd.DataFrame,
    df_all: pd.DataFrame,
    out_dir: Path,
    branch_tag: str,
    p_col: str,
    alpha: float,
    top_k: int,
    dpi: int,
    show_all_if_no_sig: bool,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_meta: List[Dict[str, object]] = []
    all_stims = sorted(df_all["stimulus_type"].astype(str).unique().tolist())

    for stim in all_stims:
        sub_sig = df_sig[df_sig["stimulus_type"].astype(str) == str(stim)].copy()
        source = "significant"
        sub = sub_sig
        if sub.empty and bool(show_all_if_no_sig):
            sub = df_all[df_all["stimulus_type"].astype(str) == str(stim)].copy()
            source = "all"
        if sub.empty:
            continue

        sub = sub.reindex(sub["r_obs"].abs().sort_values(ascending=False).index).head(int(top_k)).copy()
        sub = sub.sort_values("r_obs", ascending=True)
        colors = ["#C23B22" if v > 0 else "#2C6DB2" for v in sub["r_obs"].astype(float).tolist()]

        fig_h = max(3.5, 0.42 * sub.shape[0] + 1.5)
        fig, ax = plt.subplots(figsize=(7.5, fig_h))
        ax.barh(sub["roi"].astype(str), sub["r_obs"].astype(float), color=colors, edgecolor="black", linewidth=0.4)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"{stim} | top {min(int(top_k), sub.shape[0])} ROI ({source})")
        ax.set_xlabel("brain-behavior r_obs")
        ax.set_ylabel("ROI")
        plt.tight_layout()

        out_png = out_dir / f"brain_behavior_top_roi_{branch_tag}_{stim}_{source}_{p_col}_a{alpha:g}.png"
        out_csv = out_dir / f"brain_behavior_top_roi_{branch_tag}_{stim}_{source}_{p_col}_a{alpha:g}.csv"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        sub.to_csv(out_csv, index=False)
        rows_meta.append(
            {
                "stimulus_type": str(stim),
                "source": str(source),
                "n_rows_plotted": int(sub.shape[0]),
                "figure": str(out_png),
                "csv": str(out_csv),
            }
        )

    meta_df = pd.DataFrame(rows_meta)
    if not meta_df.empty:
        meta_df.to_csv(out_dir / f"brain_behavior_top_roi_{branch_tag}_{p_col}_summary.csv", index=False)
    return meta_df


def main() -> None:
    args = parse_args()
    branch_tag = _branch_tag(str(args.stimulus_dir_name))
    result_dir = Path(args.matrix_dir) / "figures" / f"{str(args.stimulus_dir_name)}_brain_behavior"
    result_dir.mkdir(parents=True, exist_ok=True)

    df_all, df_sig = collect_results(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        result_file=str(args.result_file),
        p_col=str(args.p_col),
        alpha=float(args.alpha),
        min_abs_r=float(args.min_abs_r),
        positive_only=bool(args.positive_only),
        negative_only=bool(args.negative_only),
    )

    all_csv = result_dir / f"brain_behavior_all_results_{branch_tag}.csv"
    sig_csv = result_dir / f"brain_behavior_sig_results_{branch_tag}_{str(args.p_col)}_a{float(args.alpha):g}.csv"
    df_all.sort_values(["stimulus_type", str(args.p_col), "roi"]).to_csv(all_csv, index=False)
    df_sig.sort_values(["stimulus_type", str(args.p_col), "roi"]).to_csv(sig_csv, index=False)

    if not df_sig.empty:
        draw_heatmap(
            df=df_sig,
            out_dir=result_dir,
            branch_tag=branch_tag,
            p_col=str(args.p_col),
            alpha=float(args.alpha),
            max_rois=int(args.max_rois_heatmap),
            dpi=int(args.dpi),
            positive_only=bool(args.positive_only),
            negative_only=bool(args.negative_only),
        )

    draw_top_bars(
        df_sig=df_sig,
        df_all=df_all,
        out_dir=result_dir,
        branch_tag=branch_tag,
        p_col=str(args.p_col),
        alpha=float(args.alpha),
        top_k=int(args.top_k),
        dpi=int(args.dpi),
        show_all_if_no_sig=bool(args.show_all_if_no_sig),
    )


if __name__ == "__main__":
    main()
