#!/usr/bin/env python3
"""
plot_step5c_rsa_main.py

Create the main Step 5C RSA figure for a result directory.

Outputs
- fig_step5c_main.png
- table_step5c_condition_summary.tsv
- table_step5c_roi_delta.tsv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import (
    abbreviate_roi_name,
    add_panel_label,
    apply_publication_rcparams,
    default_base_dir,
    default_figure_dir,
    save_png_pdf,
)


PALETTE = {"Metaphor": "#d9485f", "Spatial": "#3b82f6", "Baseline": "#94a3b8"}


def _condition_order(frame: pd.DataFrame) -> list[str]:
    preferred = ["Metaphor", "Spatial", "Baseline"]
    present = [c for c in preferred if c in set(frame["condition"].astype(str))]
    return present


def _default_results_dir() -> Path:
    base = default_base_dir()
    roi_set = os.environ.get("METAPHOR_ROI_SET", "main_functional")
    return base / "paper_outputs" / "qc" / f"rsa_results_optimized_{roi_set}"


def _default_out_dir(results_dir: Path) -> Path:
    return default_figure_dir("figures_main")


def _load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"subject", "roi", "stage", "Sim_Metaphor", "Sim_Spatial"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RSA summary missing columns: {sorted(missing)}")
    frame = frame.copy()
    if "Sim_Baseline" not in frame.columns:
        frame["Sim_Baseline"] = pd.NA
    frame["stage"] = frame["stage"].astype(str).str.title()
    return frame


def _load_itemwise(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"subject", "roi", "stage", "condition", "similarity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Itemwise RSA file missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["stage"] = frame["stage"].astype(str).str.title()
    frame["condition"] = frame["condition"].astype(str).str.title()
    frame["similarity"] = pd.to_numeric(frame["similarity"], errors="coerce")
    return frame.dropna(subset=["similarity"]).reset_index(drop=True)


def _infer_roi_set(results_dir: Path) -> str:
    name = results_dir.name
    prefix = "rsa_results_optimized_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return os.environ.get("METAPHOR_ROI_SET", "")


def _load_lmm_fdr(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def _to_long(frame: pd.DataFrame) -> pd.DataFrame:
    long = frame.melt(
        id_vars=["subject", "roi", "stage"],
        value_vars=["Sim_Metaphor", "Sim_Spatial", "Sim_Baseline"],
        var_name="condition",
        value_name="similarity",
    )
    long["condition"] = long["condition"].str.replace("Sim_", "", regex=False)
    long["similarity"] = pd.to_numeric(long["similarity"], errors="coerce")
    long = long.dropna(subset=["similarity"]).reset_index(drop=True)
    return long


def _condition_summary(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["subject", "roi", "condition", "stage"], as_index=False)["similarity"]
        .mean()
        .sort_values(["subject", "roi", "condition", "stage"])
        .reset_index(drop=True)
    )


def _roi_delta_summary(long: pd.DataFrame) -> pd.DataFrame:
    subject_roi = (
        long.groupby(["subject", "roi", "condition", "stage"], as_index=False)["similarity"]
        .mean()
    )
    pivot = subject_roi.pivot_table(
        index=["subject", "roi", "condition"],
        columns="stage",
        values="similarity",
    ).reset_index()
    pivot["delta_post_minus_pre"] = pivot["Post"] - pivot["Pre"]
    pivot["roi_label"] = pivot["roi"].map(abbreviate_roi_name)
    return pivot


def _roi_condition_group_summary(roi_delta: pd.DataFrame, lmm_fdr: pd.DataFrame | None = None, roi_set: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (roi, condition), subset in roi_delta.groupby(["roi", "condition"], sort=False):
        work = subset[["subject", "Pre", "Post", "delta_post_minus_pre", "roi_label"]].dropna().copy()
        if work.empty:
            continue
        sem_delta = float(work["delta_post_minus_pre"].std(ddof=1) / np.sqrt(len(work))) if len(work) > 1 else np.nan
        rows.append(
            {
                "roi_set": roi_set or "",
                "roi": roi,
                "roi_label": work["roi_label"].iloc[0],
                "condition": condition,
                "n_subjects": int(len(work)),
                "mean_pre": float(work["Pre"].mean()),
                "mean_post": float(work["Post"].mean()),
                "mean_delta_post_minus_pre": float(work["delta_post_minus_pre"].mean()),
                "sem_delta_post_minus_pre": sem_delta,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["planned_interaction_q"] = np.nan
    if lmm_fdr is not None and not lmm_fdr.empty:
        term_map = {
            "Metaphor": "C(condition)[T.Metaphor]:C(time)[T.Pre]",
            "Spatial": "C(condition)[T.Spatial]:C(time)[T.Pre]",
        }
        lmm = lmm_fdr.copy()
        if roi_set:
            lmm = lmm[lmm["roi_set"].astype(str).eq(str(roi_set))]
        lmm = lmm[lmm["analysis_level"].astype(str).eq("roi")].copy()
        for idx, row in out.iterrows():
            term = term_map.get(str(row["condition"]))
            if term is None:
                continue
            match = lmm[
                lmm["analysis_name"].astype(str).eq(str(row["roi"]))
                & lmm["term"].astype(str).eq(term)
            ]
            if not match.empty:
                out.loc[idx, "planned_interaction_q"] = pd.to_numeric(
                    match.iloc[0]["q_bh_planned_interaction_within_level"],
                    errors="coerce",
                )
    return out.sort_values(["roi", "condition"]).reset_index(drop=True)


def _q_star(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    if q_value < 0.10:
        return "+"
    return ""


def _plot_roi_heatmap(ax: plt.Axes, roi_group: pd.DataFrame, cond_order: list[str]) -> None:
    if roi_group.empty:
        ax.text(0.5, 0.5, "No ROI summary", ha="center", va="center")
        ax.axis("off")
        return
    roi_order = (
        roi_group.groupby("roi_label")["mean_delta_post_minus_pre"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    matrix = roi_group.pivot(index="roi_label", columns="condition", values="mean_delta_post_minus_pre").reindex(roi_order)
    matrix = matrix[[condition for condition in cond_order if condition in matrix.columns]]
    vmax = np.nanmax(np.abs(matrix.to_numpy(dtype=float))) if not matrix.empty else 0.1
    vmax = max(float(vmax), 0.02)
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Post - Pre"},
        ax=ax,
    )
    q_lookup = {
        (row.roi_label, row.condition): _q_star(float(row.planned_interaction_q))
        for row in roi_group.itertuples(index=False)
        if "planned_interaction_q" in roi_group.columns
    }
    for y_idx, roi_label in enumerate(matrix.index):
        for x_idx, condition in enumerate(matrix.columns):
            value = matrix.loc[roi_label, condition]
            if pd.isna(value):
                continue
            star = q_lookup.get((roi_label, condition), "")
            text = f"{value:+.3f}{star}"
            ax.text(x_idx + 0.5, y_idx + 0.5, text, ha="center", va="center", fontsize=7.2, color="black")
    ax.set_title("ROI-wise Step 5C change", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot(condition_summary: pd.DataFrame, roi_delta: pd.DataFrame, roi_group: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    apply_publication_rcparams()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), gridspec_kw={"width_ratios": [1.05, 1.65]})
    cond_order = _condition_order(condition_summary)

    ax0 = axes[0]
    _plot_roi_heatmap(ax0, roi_group, cond_order)
    add_panel_label(ax0, "a")

    roi_order = (
        roi_delta.groupby("roi_label")["delta_post_minus_pre"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    ax1 = axes[1]
    sns.pointplot(
        data=roi_delta,
        y="roi_label",
        x="delta_post_minus_pre",
        hue="condition",
        order=roi_order,
        hue_order=cond_order,
        palette=PALETTE,
        dodge=0.45,
        linestyles="none",
        markers=["o", "s", "D"][: len(cond_order)],
        errorbar=("sd"),
        ax=ax1,
    )
    ax1.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax1.set_title("ROI-Level Delta Similarity (Post - Pre)", fontweight="bold")
    ax1.set_xlabel("Delta Similarity")
    ax1.set_ylabel("")
    ax1.legend(title="Condition", frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.16), ncol=len(cond_order))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    add_panel_label(ax1, "b")

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.90, wspace=0.32)
    save_png_pdf(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step 5C RSA main figure.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_default_results_dir(),
        help="RSA results directory containing rsa_summary_stats.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional figure output directory. Defaults to <results-dir>/figures",
    )
    parser.add_argument(
        "--lmm-fdr",
        type=Path,
        default=default_base_dir() / "paper_outputs" / "tables_si" / "table_rsa_lmm_fdr.tsv",
        help="Optional Step 5C LMM/FDR table used for ROI-wise significance labels.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir(results_dir)
    roi_set = _infer_roi_set(results_dir)

    itemwise_file = results_dir / "rsa_itemwise_details.csv"
    if itemwise_file.exists():
        long = _load_itemwise(itemwise_file)
    else:
        summary = _load_summary(results_dir / "rsa_summary_stats.csv")
        long = _to_long(summary)
    condition_summary = _condition_summary(long)
    roi_delta = _roi_delta_summary(long)
    lmm_fdr = _load_lmm_fdr(args.lmm_fdr)
    roi_group = _roi_condition_group_summary(roi_delta, lmm_fdr=lmm_fdr, roi_set=roi_set)

    out_dir.mkdir(parents=True, exist_ok=True)
    condition_summary.to_csv(out_dir / "table_step5c_condition_summary.tsv", sep="\t", index=False)
    roi_delta.to_csv(out_dir / "table_step5c_roi_delta.tsv", sep="\t", index=False)
    roi_group.to_csv(out_dir / "table_step5c_roi_condition_group.tsv", sep="\t", index=False)
    _plot(condition_summary, roi_delta, roi_group, out_dir / "fig_step5c_main.png")
    print(f"[step5c-plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
