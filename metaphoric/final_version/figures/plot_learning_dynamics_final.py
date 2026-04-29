#!/usr/bin/env python3
"""
Plot the finalized B1 learning-dynamics figure from paper_outputs tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import (
    abbreviate_roi_name,
    add_panel_label,
    apply_publication_rcparams,
    default_base_dir,
    default_figure_dir,
    save_png_pdf,
)


def _bootstrap_mean_ci(values: np.ndarray, *, rng_seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    vec = np.asarray(values, dtype=float)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return float("nan"), float("nan")
    if vec.size == 1:
        return float(vec[0]), float(vec[0])
    rng = np.random.default_rng(rng_seed)
    draws = rng.choice(vec, size=(n_boot, vec.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _default_subject_table(roi_set: str) -> Path:
    roi_tag = str(roi_set).strip()
    return (
        default_base_dir()
        / "paper_outputs"
        / "tables_si"
        / f"table_learning_repeated_exposure_subject_{roi_tag}.tsv"
    )


def _default_out_path(roi_set: str) -> Path:
    roi_tag = str(roi_set).strip()
    return default_figure_dir("figures_main") / f"fig_learning_dynamics_{roi_tag}.png"


def _plot(subject_frame: pd.DataFrame, out_path: Path, *, top_n: int) -> None:
    if subject_frame.empty:
        raise ValueError("Learning dynamics subject table is empty.")

    roi_order = (
        subject_frame.groupby("roi")["subject"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    plot_frame = subject_frame[subject_frame["roi"].isin(roi_order)].copy()
    plot_frame["roi_label"] = plot_frame["roi"].map(abbreviate_roi_name)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(4.6, 0.72 * len(roi_order) + 1.5)), sharey=True)
    colors = {"yy": "#b23a48", "kj": "#3574b7"}
    metrics = [
        ("pair_discrimination", "A2: same - different"),
        ("same_sentence_cross_run_similarity", "A1: same-sentence cross-run similarity"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        positions = np.arange(len(roi_order))
        for offset, condition in [(-0.12, "yy"), (0.12, "kj")]:
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for roi in roi_order:
                vec = plot_frame.loc[
                    (plot_frame["roi"] == roi) & (plot_frame["condition"] == condition),
                    metric,
                ].to_numpy(dtype=float)
                mean_val = float(np.nanmean(vec))
                ci_low, ci_high = _bootstrap_mean_ci(vec)
                means.append(mean_val)
                lows.append(mean_val - ci_low if np.isfinite(ci_low) else np.nan)
                highs.append(ci_high - mean_val if np.isfinite(ci_high) else np.nan)
            ax.errorbar(
                means,
                positions + offset,
                xerr=np.vstack([lows, highs]),
                fmt="o",
                color=colors[condition],
                ecolor=colors[condition],
                elinewidth=1.2,
                capsize=3,
                markersize=5,
                label="Metaphor" if condition == "yy" else "Spatial",
            )
        ax.axvline(0.0, color="#9ca3af", linestyle="--", linewidth=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean similarity")
        ax.set_yticks(positions)
        ax.set_yticklabels([abbreviate_roi_name(roi) for roi in roi_order], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_pdf(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot finalized B1 learning-dynamics figure.")
    parser.add_argument("--roi-set", default=os.environ.get("METAPHOR_ROI_SET", "literature"))
    parser.add_argument("--subject-table", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()

    subject_table = args.subject_table or _default_subject_table(args.roi_set)
    out_path = args.out_path or _default_out_path(args.roi_set)
    subject_frame = pd.read_csv(subject_table, sep="\t")

    apply_publication_rcparams()
    _plot(subject_frame, out_path, top_n=args.top_n)
    print(f"[learning-dynamics-plot] wrote {out_path}")


if __name__ == "__main__":
    main()
