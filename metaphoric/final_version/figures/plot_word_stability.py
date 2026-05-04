#!/usr/bin/env python3
"""Plot single-word stability and neighborhood-change summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import add_panel_label, apply_publication_rcparams, default_paper_output_root, save_png_pdf


ROI_SET_LABELS = {
    "main_functional": "Main functional",
    "literature": "Metaphor literature",
    "literature_spatial": "Spatial literature",
}
ROI_SET_COLORS = {
    "main_functional": "#4C78A8",
    "literature": "#F58518",
    "literature_spatial": "#54A24B",
}
CONDITION_LABELS = {"yy": "YY", "kj": "KJ", "baseline": "Baseline"}


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _star(q_value: float) -> str:
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


def _prepare_top(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frame = summary[
        summary["contrast"].eq("same_word_stability_gt_zero")
        & summary["is_primary_contrast"].astype(bool)
    ].copy()
    frame["q_plot"] = pd.to_numeric(frame["q_bh_primary_family"], errors="coerce")
    frame["mean_effect"] = pd.to_numeric(frame["mean_effect"], errors="coerce")
    frame["ci_low"] = pd.to_numeric(frame["ci_low"], errors="coerce")
    frame["ci_high"] = pd.to_numeric(frame["ci_high"], errors="coerce")
    frame["roi_set_label"] = frame["roi_set"].map(ROI_SET_LABELS).fillna(frame["roi_set"])
    frame["condition_label"] = frame["condition"].map(CONDITION_LABELS).fillna(frame["condition"])
    frame["plot_label"] = frame["roi_set_label"] + " | " + frame["roi"].astype(str) + " | " + frame["condition_label"]
    return frame.sort_values(["q_plot", "p", "roi_set", "roi"], na_position="last").head(top_n).reset_index(drop=True)


def _plot_stability_forest(ax: plt.Axes, top: pd.DataFrame) -> None:
    if top.empty:
        ax.text(0.5, 0.5, "No stability rows", ha="center", va="center")
        ax.axis("off")
        return
    frame = top.iloc[::-1].copy()
    y = np.arange(len(frame))
    colors = frame["roi_set"].map(ROI_SET_COLORS).fillna("#777777")
    for idx, row in enumerate(frame.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [idx, idx], color=colors.iloc[idx], lw=1.8)
        label = _star(float(row.q_plot))
        if label:
            ax.text(row.ci_high + 0.004, idx, label, va="center", ha="left", fontsize=9, fontweight="bold")
    ax.scatter(frame["mean_effect"], y, color=colors, s=44, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(0, color="#444444", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["plot_label"], fontsize=7.5)
    ax.set_xlabel("Same-word pre/post correlation")
    ax.set_title("Single-word identity stability", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _plot_neighborhood(ax: plt.Axes, subject: pd.DataFrame) -> None:
    frame = subject.copy()
    frame["condition_label"] = frame["condition"].map(CONDITION_LABELS).fillna(frame["condition"])
    order = ["YY", "KJ", "Baseline"]
    sns.pointplot(
        data=frame,
        x="condition_label",
        y="mean_neighborhood_delta_post_minus_pre",
        order=[item for item in order if item in set(frame["condition_label"])],
        errorbar=("ci", 95),
        color="#333333",
        markers="o",
        linestyles="",
        ax=ax,
    )
    ax.axhline(0, color="#444444", ls="--", lw=0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Neighborhood change (Post - Pre)")
    ax.set_title("Word-to-neighbor change", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot word stability results.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--top-n", type=int, default=18)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    summary = _read_table(paper_root / "tables_main" / "table_word_stability.tsv")
    subject = _read_table(paper_root / "qc" / "word_stability" / "word_stability_subject.tsv")
    top = _prepare_top(summary, args.top_n)

    figures_dir = paper_root / "figures_main"
    figures_dir.mkdir(parents=True, exist_ok=True)
    top_path = figures_dir / "table_word_stability_top.tsv"
    top.to_csv(top_path, sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, max(5.4, len(top) * 0.32)), gridspec_kw={"width_ratios": [2.1, 1.0]})
    _plot_stability_forest(axes[0], top)
    _plot_neighborhood(axes[1], subject)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.text(0.05, 0.02, "* q < .05, ** q < .01, *** q < .001; + q < .10 within primary family", fontsize=8)
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.18, top=0.90, wspace=0.35)
    save_png_pdf(fig, figures_dir / "fig_word_stability.png")
    plt.close(fig)
    print(f"[word-stability-plot] wrote {figures_dir / 'fig_word_stability.png'}")
    print(f"[word-stability-plot] wrote {top_path}")


if __name__ == "__main__":
    main()
