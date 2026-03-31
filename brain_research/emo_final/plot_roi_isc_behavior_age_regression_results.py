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
DEFAULT_MODELS = ("M_nn", "M_conv", "M_div")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot brain-behavior age-regression results")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus", choices=("by_stimulus", "by_emotion"))
    p.add_argument("--result-file", type=str, default="roi_isc_behavior_age_regression.csv")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument(
        "--effect-col",
        type=str,
        default="beta_interaction",
        choices=("beta_interaction", "beta_behavior", "beta_age", "t_interaction", "t_behavior", "t_age"),
    )
    p.add_argument(
        "--p-col",
        type=str,
        default="p_fdr_perm_interaction_model_wise",
        choices=(
            "p_fdr_perm_interaction_model_wise",
            "p_fwer_interaction_model_wise",
            "p_perm_interaction",
            "p_fdr_perm_behavior_model_wise",
            "p_fwer_behavior_model_wise",
            "p_perm_behavior",
            "p_fdr_perm_age_model_wise",
            "p_fwer_age_model_wise",
            "p_perm_age",
        ),
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-rois-heatmap", type=int, default=40)
    p.add_argument("--min-abs-effect", type=float, default=0.0)
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--negative-only", action="store_true")
    p.add_argument("--no-show-all-if-no-sig", action="store_false", dest="show_all_if_no_sig", default=True)
    return p.parse_args()


def _branch_tag(stimulus_dir_name: str) -> str:
    return "trial" if str(stimulus_dir_name) == "by_stimulus" else "emotion"


def _filter_sign(df: pd.DataFrame, effect_col: str, positive_only: bool, negative_only: bool) -> pd.DataFrame:
    out = df.copy()
    if bool(positive_only) and bool(negative_only):
        raise ValueError("Cannot use both --positive-only and --negative-only")
    if bool(positive_only):
        out = out[out[str(effect_col)].astype(float) > 0].copy()
    if bool(negative_only):
        out = out[out[str(effect_col)].astype(float) < 0].copy()
    return out


def collect_results(
    matrix_dir: Path,
    stimulus_dir_name: str,
    result_file: str,
    models: List[str],
    effect_col: str,
    p_col: str,
    alpha: float,
    min_abs_effect: float,
    positive_only: bool,
    negative_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    keep_models = {str(m) for m in models}
    rows_all: List[pd.DataFrame] = []
    rows_sig: List[pd.DataFrame] = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        csv_path = stim_dir / str(result_file)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        need = {"roi", "model", str(effect_col), str(p_col)}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{csv_path} missing columns: {sorted(miss)}")

        sub = df[
            np.isfinite(pd.to_numeric(df[str(effect_col)], errors="coerce").to_numpy(dtype=float))
            & np.isfinite(pd.to_numeric(df[str(p_col)], errors="coerce").to_numpy(dtype=float))
        ].copy()
        if sub.empty:
            continue
        sub = sub[sub["model"].astype(str).isin(keep_models)].copy()
        if sub.empty:
            continue
        sub["stimulus_type"] = stim_dir.name
        sub = _filter_sign(sub, effect_col=str(effect_col), positive_only=bool(positive_only), negative_only=bool(negative_only))
        sub = sub[np.abs(pd.to_numeric(sub[str(effect_col)], errors="coerce").to_numpy(dtype=float)) >= float(min_abs_effect)].copy()
        if sub.empty:
            continue
        rows_all.append(sub.copy())
        sig = sub[pd.to_numeric(sub[str(p_col)], errors="coerce") <= float(alpha)].copy()
        if not sig.empty:
            rows_sig.append(sig)

    if not rows_all:
        raise ValueError("No usable age-regression result files found")
    df_all = pd.concat(rows_all, axis=0, ignore_index=True)
    df_sig = pd.concat(rows_sig, axis=0, ignore_index=True) if rows_sig else pd.DataFrame(columns=df_all.columns)
    return df_all, df_sig


def _rank_rois(df: pd.DataFrame, effect_col: str, p_col: str) -> List[str]:
    rank_df = (
        df.groupby("roi", as_index=False)
        .agg(
            p_best=(str(p_col), "min"),
            effect_abs_max=(str(effect_col), lambda x: float(np.nanmax(np.abs(pd.to_numeric(x, errors="coerce"))))),
        )
        .sort_values(["p_best", "effect_abs_max", "roi"], ascending=[True, False, True])
    )
    return rank_df["roi"].astype(str).tolist()


def draw_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    branch_tag: str,
    models: List[str],
    effect_col: str,
    p_col: str,
    alpha: float,
    max_rois: int,
    dpi: int,
    positive_only: bool,
    negative_only: bool,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_meta: List[Dict[str, object]] = []
    sign_tag = "all"
    if bool(positive_only):
        sign_tag = "pos"
    elif bool(negative_only):
        sign_tag = "neg"

    for model_name in [str(m) for m in models]:
        sub = df[df["model"].astype(str) == str(model_name)].copy()
        if sub.empty:
            continue
        roi_order = _rank_rois(sub, effect_col=str(effect_col), p_col=str(p_col))[: int(max_rois)]
        stim_order = sorted(sub["stimulus_type"].astype(str).unique().tolist())
        mat = (
            sub.groupby(["stimulus_type", "roi"], as_index=False)[str(effect_col)]
            .mean()
            .pivot(index="stimulus_type", columns="roi", values=str(effect_col))
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
            cbar_kws={"label": str(effect_col)},
        )
        ax.set_title(f"Age-regression heatmap | {model_name} | {branch_tag} | {p_col} <= {alpha:g}")
        ax.set_xlabel("ROI")
        ax.set_ylabel("stimulus_type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()

        fig_path = out_dir / f"brain_behavior_age_regression_heatmap_{branch_tag}_{model_name}_{effect_col}_{p_col}_a{alpha:g}_{sign_tag}.png"
        csv_path = out_dir / f"brain_behavior_age_regression_heatmap_{branch_tag}_{model_name}_{effect_col}_{p_col}_a{alpha:g}_{sign_tag}.csv"
        fig.savefig(fig_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        mat.to_csv(csv_path)
        rows_meta.append(
            {
                "model": str(model_name),
                "n_stimulus_types": int(len(stim_order)),
                "n_rois": int(len(roi_order)),
                "figure": str(fig_path),
                "matrix_csv": str(csv_path),
            }
        )

    meta_df = pd.DataFrame(rows_meta)
    if not meta_df.empty:
        meta_df.to_csv(out_dir / f"brain_behavior_age_regression_heatmap_{branch_tag}_{effect_col}_{p_col}_summary.csv", index=False)
    return meta_df


def draw_top_bars(
    df_sig: pd.DataFrame,
    df_all: pd.DataFrame,
    out_dir: Path,
    branch_tag: str,
    models: List[str],
    effect_col: str,
    p_col: str,
    alpha: float,
    top_k: int,
    dpi: int,
    show_all_if_no_sig: bool,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_meta: List[Dict[str, object]] = []
    stimulus_types = sorted(df_all["stimulus_type"].astype(str).unique().tolist())

    for model_name in [str(m) for m in models]:
        for stim in stimulus_types:
            sub_sig = df_sig[(df_sig["model"].astype(str) == str(model_name)) & (df_sig["stimulus_type"].astype(str) == str(stim))].copy()
            source = "significant"
            sub = sub_sig
            if sub.empty and bool(show_all_if_no_sig):
                sub = df_all[(df_all["model"].astype(str) == str(model_name)) & (df_all["stimulus_type"].astype(str) == str(stim))].copy()
                source = "all"
            if sub.empty:
                continue

            sub = sub.reindex(pd.to_numeric(sub[str(effect_col)], errors="coerce").abs().sort_values(ascending=False).index).head(int(top_k)).copy()
            sub = sub.sort_values(str(effect_col), ascending=True)
            vals = pd.to_numeric(sub[str(effect_col)], errors="coerce").to_numpy(dtype=float)
            colors = ["#C23B22" if v > 0 else "#2C6DB2" for v in vals.tolist()]

            fig_h = max(3.5, 0.42 * sub.shape[0] + 1.5)
            fig, ax = plt.subplots(figsize=(7.8, fig_h))
            ax.barh(sub["roi"].astype(str), vals, color=colors, edgecolor="black", linewidth=0.4)
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_title(f"{stim} | {model_name} | top {min(int(top_k), sub.shape[0])} ROI ({source})")
            ax.set_xlabel(str(effect_col))
            ax.set_ylabel("ROI")
            plt.tight_layout()

            out_png = out_dir / f"brain_behavior_age_regression_top_roi_{branch_tag}_{stim}_{model_name}_{source}_{effect_col}_{p_col}_a{alpha:g}.png"
            out_csv = out_dir / f"brain_behavior_age_regression_top_roi_{branch_tag}_{stim}_{model_name}_{source}_{effect_col}_{p_col}_a{alpha:g}.csv"
            fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
            plt.close(fig)
            sub.to_csv(out_csv, index=False)
            rows_meta.append(
                {
                    "stimulus_type": str(stim),
                    "model": str(model_name),
                    "source": str(source),
                    "n_rows_plotted": int(sub.shape[0]),
                    "figure": str(out_png),
                    "csv": str(out_csv),
                }
            )

    meta_df = pd.DataFrame(rows_meta)
    if not meta_df.empty:
        meta_df.to_csv(out_dir / f"brain_behavior_age_regression_top_roi_{branch_tag}_{effect_col}_{p_col}_summary.csv", index=False)
    return meta_df


def main() -> None:
    args = parse_args()
    branch_tag = _branch_tag(str(args.stimulus_dir_name))
    out_dir = Path(args.matrix_dir) / "figures" / f"{str(args.stimulus_dir_name)}_brain_behavior_age_regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [str(m) for m in args.models]
    df_all, df_sig = collect_results(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        result_file=str(args.result_file),
        models=models,
        effect_col=str(args.effect_col),
        p_col=str(args.p_col),
        alpha=float(args.alpha),
        min_abs_effect=float(args.min_abs_effect),
        positive_only=bool(args.positive_only),
        negative_only=bool(args.negative_only),
    )

    all_csv = out_dir / f"brain_behavior_age_regression_all_results_{branch_tag}.csv"
    sig_csv = out_dir / f"brain_behavior_age_regression_sig_results_{branch_tag}_{args.effect_col}_{args.p_col}_a{float(args.alpha):g}.csv"
    df_all.sort_values(["stimulus_type", "model", str(args.p_col), "roi"]).to_csv(all_csv, index=False)
    df_sig.sort_values(["stimulus_type", "model", str(args.p_col), "roi"]).to_csv(sig_csv, index=False)

    if not df_sig.empty:
        draw_heatmaps(
            df=df_sig,
            out_dir=out_dir,
            branch_tag=branch_tag,
            models=models,
            effect_col=str(args.effect_col),
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
        out_dir=out_dir,
        branch_tag=branch_tag,
        models=models,
        effect_col=str(args.effect_col),
        p_col=str(args.p_col),
        alpha=float(args.alpha),
        top_k=int(args.top_k),
        dpi=int(args.dpi),
        show_all_if_no_sig=bool(args.show_all_if_no_sig),
    )


if __name__ == "__main__":
    main()
