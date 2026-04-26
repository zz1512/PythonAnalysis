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

from figure_utils import abbreviate_roi_name, add_panel_label, apply_publication_rcparams, save_png_pdf


PALETTE = {"Metaphor": "#d9485f", "Spatial": "#3b82f6", "Baseline": "#94a3b8"}


def _condition_order(frame: pd.DataFrame) -> list[str]:
    preferred = ["Metaphor", "Spatial", "Baseline"]
    present = [c for c in preferred if c in set(frame["condition"].astype(str))]
    return present


def _default_results_dir() -> Path:
    base = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
    roi_set = os.environ.get("METAPHOR_ROI_SET", "main_functional")
    return base / f"rsa_results_optimized_{roi_set}"


def _default_out_dir(results_dir: Path) -> Path:
    override = os.environ.get("METAPHOR_FIG_OUT_DIR", "").strip()
    if override:
        return Path(override)
    return results_dir / "figures"


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
        long.groupby(["subject", "condition", "stage"], as_index=False)["similarity"]
        .mean()
        .sort_values(["subject", "condition", "stage"])
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


def _plot(condition_summary: pd.DataFrame, roi_delta: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    apply_publication_rcparams()

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), gridspec_kw={"width_ratios": [1.0, 1.6]})
    cond_order = _condition_order(condition_summary)

    ax0 = axes[0]
    sns.violinplot(
        data=condition_summary,
        x="condition",
        y="similarity",
        hue="stage",
        order=cond_order,
        hue_order=["Pre", "Post"],
        palette={"Pre": "#9ca3af", "Post": "#111827"},
        split=True,
        inner=None,
        cut=0,
        ax=ax0,
    )
    sns.stripplot(
        data=condition_summary,
        x="condition",
        y="similarity",
        hue="stage",
        order=cond_order,
        hue_order=["Pre", "Post"],
        dodge=True,
        palette={"Pre": "#9ca3af", "Post": "#111827"},
        alpha=0.25,
        size=2.8,
        ax=ax0,
    )
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[:2], labels[:2], title="Stage", frameon=False, loc="upper right")
    ax0.set_title("Pooled Step 5C Similarity", fontweight="bold")
    ax0.set_xlabel("")
    ax0.set_ylabel("Mean Pair Similarity")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    y_min = float(condition_summary["similarity"].min())
    y_max = float(condition_summary["similarity"].max())
    y_pad = max(0.015, (y_max - y_min) * 0.10)
    ax0.set_ylim(y_min - y_pad, y_max + y_pad)
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
    ax1.legend(title="Condition", frameon=False, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    add_panel_label(ax1, "b")

    fig.tight_layout()
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
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir(results_dir)

    itemwise_file = results_dir / "rsa_itemwise_details.csv"
    if itemwise_file.exists():
        long = _load_itemwise(itemwise_file)
    else:
        summary = _load_summary(results_dir / "rsa_summary_stats.csv")
        long = _to_long(summary)
    condition_summary = _condition_summary(long)
    roi_delta = _roi_delta_summary(long)

    out_dir.mkdir(parents=True, exist_ok=True)
    condition_summary.to_csv(out_dir / "table_step5c_condition_summary.tsv", sep="\t", index=False)
    roi_delta.to_csv(out_dir / "table_step5c_roi_delta.tsv", sep="\t", index=False)
    _plot(condition_summary, roi_delta, out_dir / "fig_step5c_main.png")
    print(f"[step5c-plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
