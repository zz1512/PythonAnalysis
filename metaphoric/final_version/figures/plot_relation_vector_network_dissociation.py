#!/usr/bin/env python3
"""Plot network-level relation-vector dissociation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import add_panel_label, apply_publication_rcparams, default_paper_output_root, save_png_pdf


NETWORK_ORDER = [
    "semantic_metaphor",
    "spatial_context",
    "hippocampus_core",
    "main_metaphor_functional",
    "main_spatial_functional",
]
NETWORK_LABELS = {
    "semantic_metaphor": "Semantic/metaphor",
    "spatial_context": "Spatial/context",
    "hippocampus_core": "Hippocampus",
    "main_metaphor_functional": "Main M>S",
    "main_spatial_functional": "Main S>M",
}
MODEL_LABELS = {
    "M9_relation_vector_direct": "direct vector",
    "M9_relation_vector_abs": "absolute vector",
}


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


def _prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[summary["analysis_type"].eq("network_condition_contrast")].copy()
    frame["network_label"] = frame["network"].map(NETWORK_LABELS).fillna(frame["network"])
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    for col in ["mean_effect", "ci_low", "ci_high", "q_bh_within_model"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _prepare_planned(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frame = summary[summary["analysis_type"].eq("network_difference_of_condition_contrast")].copy()
    if frame.empty:
        return frame
    for col in ["mean_effect", "ci_low", "ci_high", "q_bh_within_model", "p"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["left_label"] = frame["left_network"].map(NETWORK_LABELS).fillna(frame["left_network"])
    frame["right_label"] = frame["right_network"].map(NETWORK_LABELS).fillna(frame["right_network"])
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame["plot_label"] = frame["left_label"] + " - " + frame["right_label"] + " | " + frame["model_label"]
    return frame.sort_values(["q_bh_within_model", "p"], na_position="last").head(top_n)


def _plot_network_summary(ax: plt.Axes, network_summary: pd.DataFrame) -> None:
    if network_summary.empty:
        ax.text(0.5, 0.5, "No network summaries", ha="center", va="center")
        ax.axis("off")
        return
    order = [network for network in NETWORK_ORDER if network in set(network_summary["network"])]
    sns.pointplot(
        data=network_summary,
        x="network",
        y="mean_effect",
        hue="model_label",
        order=order,
        dodge=0.35,
        errorbar=None,
        markers="o",
        linestyles="",
        ax=ax,
    )
    for xpos, network in enumerate(order):
        subset = network_summary[network_summary["network"].eq(network)].sort_values("model")
        offsets = np.linspace(-0.18, 0.18, len(subset)) if len(subset) > 1 else [0.0]
        for offset, row in zip(offsets, subset.itertuples(index=False)):
            ax.plot([xpos + offset, xpos + offset], [row.ci_low, row.ci_high], color="#333333", lw=1.3)
            label = _star(float(row.q_bh_within_model))
            if label:
                ax.text(xpos + offset, row.ci_high + 0.006, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="#444444", ls="--", lw=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([NETWORK_LABELS.get(item, item) for item in order], rotation=28, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("YY - KJ relation decoupling")
    ax.set_title("Network-level condition contrast", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title="")


def _plot_planned(ax: plt.Axes, planned: pd.DataFrame) -> None:
    if planned.empty:
        ax.text(0.5, 0.5, "No planned network contrasts", ha="center", va="center")
        ax.axis("off")
        return
    frame = planned.iloc[::-1].copy()
    y = np.arange(len(frame))
    colors = np.where(frame["mean_effect"] >= 0, "#4C78A8", "#B1443C")
    for idx, row in enumerate(frame.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [idx, idx], color="#444444", lw=1.4)
        label = _star(float(row.q_bh_within_model))
        if label:
            ax.text(row.ci_high + 0.004, idx, label, va="center", ha="left", fontsize=9, fontweight="bold")
    ax.scatter(frame["mean_effect"], y, color=colors, s=48, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(0, color="#444444", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["plot_label"], fontsize=8)
    ax.set_xlabel("Network difference of YY-KJ contrast")
    ax.set_title("Planned condition x network contrasts", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot relation-vector network dissociation.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    summary = _read_table(paper_root / "tables_main" / "table_relation_vector_network_dissociation.tsv")
    network_summary = _prepare_summary(summary)
    planned = _prepare_planned(summary, args.top_n)
    figures_dir = paper_root / "figures_main"
    figures_dir.mkdir(parents=True, exist_ok=True)
    top_path = figures_dir / "table_relation_vector_network_dissociation_top.tsv"
    planned.to_csv(top_path, sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.1, 1.4]})
    _plot_network_summary(axes[0], network_summary)
    _plot_planned(axes[1], planned)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.text(0.05, 0.02, "Positive values indicate stronger YY than KJ relation-vector decoupling; * q < .05, + q < .10.", fontsize=8)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.24, top=0.90, wspace=0.38)
    save_png_pdf(fig, figures_dir / "fig_relation_vector_network_dissociation.png")
    plt.close(fig)
    print(f"[relation-vector-network-plot] wrote {figures_dir / 'fig_relation_vector_network_dissociation.png'}")
    print(f"[relation-vector-network-plot] wrote {top_path}")


if __name__ == "__main__":
    main()
