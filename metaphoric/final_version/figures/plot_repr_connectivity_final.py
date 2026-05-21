#!/usr/bin/env python3
"""
Plot the finalized B2 representational connectivity figure from paper_outputs tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import add_panel_label, apply_publication_rcparams, default_base_dir, default_figure_dir, save_png_pdf


SCOPE_ORDER = [
    ("main_functional", "within_Metaphor_gt_Spatial"),
    ("main_functional", "within_Spatial_gt_Metaphor"),
    ("main_functional", "cross_main_functional_family"),
    ("independent", "within_literature"),
    ("independent", "within_literature_spatial"),
    ("independent", "cross_independent_sets"),
]

SCOPE_LABELS = {
    ("main_functional", "within_Metaphor_gt_Spatial"): "Main: within M>S",
    ("main_functional", "within_Spatial_gt_Metaphor"): "Main: within S>M",
    ("main_functional", "cross_main_functional_family"): "Main: cross family",
    ("independent", "within_literature"): "Independent: within literature",
    ("independent", "within_literature_spatial"): "Independent: within spatial lit",
    ("independent", "cross_independent_sets"): "Independent: cross sets",
}


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


def _default_subject_table() -> Path:
    return default_base_dir() / "paper_outputs" / "tables_si" / "table_repr_connectivity_subject.tsv"


def _default_out_path() -> Path:
    return default_figure_dir("figures_main") / "fig_repr_connectivity.png"


def _plot(summary_frame: pd.DataFrame, out_path: Path) -> None:
    if summary_frame.empty:
        raise ValueError("Representational connectivity subject summary is empty.")

    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    colors = {"yy": "#b23a48", "kj": "#3574b7"}
    positions = np.arange(len(SCOPE_ORDER))

    for offset, condition in [(-0.12, "yy"), (0.12, "kj")]:
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for scope_key in SCOPE_ORDER:
            roi_pool, pair_scope = scope_key
            vec = summary_frame.loc[
                (summary_frame["roi_pool"] == roi_pool)
                & (summary_frame["pair_scope"] == pair_scope)
                & (summary_frame["condition"] == condition),
                "repr_connectivity_z",
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
    ax.set_yticks(positions)
    ax.set_yticklabels([SCOPE_LABELS[key] for key in SCOPE_ORDER], fontsize=9)
    ax.set_xlabel("Representational connectivity (Fisher z)")
    ax.set_ylabel("")
    ax.set_title("B2 representational connectivity from item-wise post-pre neural change", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "a")

    fig.tight_layout()
    save_png_pdf(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot finalized B2 representational connectivity figure.")
    parser.add_argument("--subject-table", type=Path, default=_default_subject_table())
    parser.add_argument("--out-path", type=Path, default=_default_out_path())
    args = parser.parse_args()

    apply_publication_rcparams()
    summary_frame = pd.read_csv(args.subject_table, sep="\t")
    _plot(summary_frame, args.out_path)
    print(f"[repr-connectivity-plot] wrote {args.out_path}")


if __name__ == "__main__":
    main()
