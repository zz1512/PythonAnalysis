#!/usr/bin/env python3
"""
Create a compact four-ROI-set robustness figure for Step 5C.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import add_panel_label, apply_publication_rcparams, default_base_dir, default_figure_dir, save_png_pdf


PALETTE = {"Metaphor": "#d9485f", "Spatial": "#3b82f6", "Baseline": "#94a3b8"}


def _base_dir() -> Path:
    return default_base_dir()


def _default_out_dir() -> Path:
    return default_figure_dir("figures_main")


def _load_itemwise(result_dir: Path, roi_set: str) -> pd.DataFrame:
    frame = pd.read_csv(result_dir / "rsa_itemwise_details.csv")
    required = {"subject", "roi", "stage", "condition", "similarity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Itemwise RSA file missing columns: {sorted(missing)} ({result_dir})")
    frame = frame.copy()
    frame["roi_set"] = roi_set
    return frame


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    subject_level = frame.groupby(["roi_set", "subject", "condition", "stage"], as_index=False)["similarity"].mean()
    pivot = subject_level.pivot_table(
        index=["roi_set", "subject", "condition"],
        columns="stage",
        values="similarity",
    ).reset_index()
    pivot["delta_post_minus_pre"] = pivot["Post"] - pivot["Pre"]
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step 5C robustness across ROI sets.")
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir())
    args = parser.parse_args()

    base = _base_dir()
    sources = {
        "main_functional": base / "paper_outputs" / "qc" / "rsa_results_optimized_main_functional",
        "literature": base / "paper_outputs" / "qc" / "rsa_results_optimized_literature",
        "literature_spatial": base / "paper_outputs" / "qc" / "rsa_results_optimized_literature_spatial",
        "atlas_robustness": base / "paper_outputs" / "qc" / "rsa_results_optimized_atlas_robustness",
    }

    available_sources = {
        roi_set: path for roi_set, path in sources.items()
        if (path / "rsa_itemwise_details.csv").exists()
    }
    missing_sources = sorted(set(sources) - set(available_sources))
    if not available_sources:
        raise FileNotFoundError("No rsa_itemwise_details.csv files were found for ROI-set robustness plotting.")
    if missing_sources:
        print(f"[roi-set-robustness-plot] skipping missing ROI sets: {', '.join(missing_sources)}")

    frames = [_load_itemwise(path, roi_set) for roi_set, path in available_sources.items()]
    combined = pd.concat(frames, ignore_index=True)
    summary = _summarize(combined)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "table_roi_set_robustness.tsv", sep="\t", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    apply_publication_rcparams()
    order = [
        roi_set for roi_set in ["main_functional", "literature", "literature_spatial", "atlas_robustness"]
        if roi_set in available_sources
    ]
    label_map = {
        "main_functional": "Main",
        "literature": "Literature",
        "literature_spatial": "Space",
        "atlas_robustness": "Atlas",
    }
    conds = ["Metaphor", "Spatial", "Baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2), sharey=True)

    for idx, (ax, cond) in enumerate(zip(axes, conds)):
        subset = summary[summary["condition"] == cond].copy()
        subset["roi_label"] = subset["roi_set"].map(label_map)
        sns.violinplot(
            data=subset,
            x="roi_label",
            y="delta_post_minus_pre",
            order=[label_map[o] for o in order],
            color=PALETTE[cond],
            inner=None,
            cut=0,
            linewidth=1.0,
            ax=ax,
        )
        for collection in ax.collections[: len(order)]:
            collection.set_alpha(0.70)
        sns.boxplot(
            data=subset,
            x="roi_label",
            y="delta_post_minus_pre",
            order=[label_map[o] for o in order],
            width=0.18,
            showcaps=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0, "alpha": 0.82},
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            ax=ax,
            zorder=2,
        )
        x_positions = np.arange(len(order))
        for xpos, roi_name in zip(x_positions, [label_map[o] for o in order]):
            vals = subset.loc[subset["roi_label"] == roi_name, "delta_post_minus_pre"].to_numpy()
            jitter = np.random.default_rng(100 + xpos).uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=18,
                color=PALETTE[cond],
                alpha=0.34,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
        stats = (
            subset.groupby("roi_label", as_index=False)["delta_post_minus_pre"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.errorbar(
            list(x_positions),
            stats["mean"],
            yerr=stats["std"],
            fmt="o",
            color="black",
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            zorder=4,
        )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(cond, fontweight="bold", color=PALETTE[cond])
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, ["a", "b", "c"][idx])
        if idx == 0:
            ax.set_ylabel("Delta Similarity (Post - Pre)")
        else:
            ax.set_ylabel("")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([label_map[o] for o in order], rotation=0)

    fig.tight_layout()
    save_png_pdf(fig, out_dir / "fig_roi_set_robustness.png")
    plt.close(fig)
    print(f"[roi-set-robustness-plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
