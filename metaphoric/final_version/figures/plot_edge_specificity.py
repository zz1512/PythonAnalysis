#!/usr/bin/env python3
"""Plot learned-edge specificity contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import abbreviate_roi_name, add_panel_label, apply_publication_rcparams, default_paper_output_root, save_png_pdf


CONTRAST_LABELS = {
    "yy_trained_drop_minus_baseline_pseudo_drop": "YY trained > baseline",
    "yy_trained_drop_minus_untrained_nonedge_drop": "YY trained > YY non-edge",
    "yy_specificity_minus_kj_specificity": "YY specificity > KJ",
    "yy_trained_drop_minus_kj_trained_drop": "YY trained > KJ trained",
}
ROI_SET_LABELS = {
    "main_functional": "Main",
    "literature": "Lit",
    "literature_spatial": "Spatial-lit",
}
ROI_SET_COLORS = {
    "main_functional": "#4C78A8",
    "literature": "#F58518",
    "literature_spatial": "#54A24B",
}
COMPONENT_COLORS = {
    "YY trained edge": "#d9485f",
    "YY untrained non-edge": "#f2a0a8",
    "KJ trained edge": "#3b82f6",
    "KJ untrained non-edge": "#9ec5fe",
    "Baseline pseudo-edge": "#64748b",
    "Baseline untrained non-edge": "#cbd5e1",
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


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


def _prepare(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frame = summary[summary["is_primary_contrast"].astype(bool)].copy()
    frame["q_plot"] = pd.to_numeric(frame["q_bh_primary_family"], errors="coerce")
    frame["p"] = pd.to_numeric(frame["p"], errors="coerce")
    frame["mean_effect"] = pd.to_numeric(frame["mean_effect"], errors="coerce")
    frame["ci_low"] = pd.to_numeric(frame["ci_low"], errors="coerce")
    frame["ci_high"] = pd.to_numeric(frame["ci_high"], errors="coerce")
    frame["contrast_label"] = frame["contrast"].map(CONTRAST_LABELS).fillna(frame["contrast"])
    frame["roi_set_label"] = frame["roi_set"].map(ROI_SET_LABELS).fillna(frame["roi_set"])
    frame["roi_short"] = frame["roi"].map(abbreviate_roi_name)
    frame["plot_label"] = frame["roi_set_label"] + " | " + frame["roi_short"] + " | " + frame["contrast_label"]
    return frame.sort_values(["q_plot", "p", "roi_set", "roi"], na_position="last").head(top_n).reset_index(drop=True)


def _component_label(condition: str, edge_status: str) -> str:
    condition_label = {"yy": "YY", "kj": "KJ", "baseline": "Baseline"}.get(str(condition), str(condition))
    edge_label = {
        "trained_edge": "trained edge",
        "untrained_nonedge": "untrained non-edge",
        "baseline_pseudo_edge": "pseudo-edge",
    }.get(str(edge_status), str(edge_status).replace("_", " "))
    return f"{condition_label} {edge_label}"


def _plot_forest(ax: plt.Axes, plot_df: pd.DataFrame) -> None:
    if plot_df.empty:
        ax.text(0.5, 0.5, "No primary contrasts", ha="center", va="center")
        ax.axis("off")
        return
    frame = plot_df.iloc[::-1].copy()
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
    ax.set_xlabel("Drop contrast (Pre - Post)")
    ax.set_title("Primary ROI-wise learned-edge contrasts", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _plot_edge_process(ax: plt.Axes, delta: pd.DataFrame, plot_df: pd.DataFrame, top_roi_n: int = 6) -> None:
    if plot_df.empty:
        ax.text(0.5, 0.5, "No primary ROI contrasts", ha="center", va="center")
        ax.axis("off")
        return
    roi_keys = (
        plot_df[["roi_set", "roi", "q_plot", "p"]]
        .drop_duplicates(["roi_set", "roi"])
        .sort_values(["q_plot", "p"], na_position="last")
        .head(top_roi_n)
    )
    keep = delta.merge(roi_keys[["roi_set", "roi"]], on=["roi_set", "roi"], how="inner")
    keep = keep[
        keep["condition"].isin(["yy", "kj", "baseline"])
        & keep["edge_status"].isin(["trained_edge", "untrained_nonedge", "baseline_pseudo_edge"])
    ].copy()
    if keep.empty:
        ax.text(0.5, 0.5, "No process rows", ha="center", va="center")
        ax.axis("off")
        return
    keep["component"] = [
        _component_label(condition, edge_status)
        for condition, edge_status in zip(keep["condition"], keep["edge_status"])
    ]
    keep["roi_label"] = keep["roi_set"].map(ROI_SET_LABELS).fillna(keep["roi_set"]) + " | " + keep["roi"].map(abbreviate_roi_name)
    keep["roi_label"] = pd.Categorical(
        keep["roi_label"],
        categories=(
            roi_keys.assign(
                roi_label=roi_keys["roi_set"].map(ROI_SET_LABELS).fillna(roi_keys["roi_set"])
                + " | "
                + roi_keys["roi"].map(abbreviate_roi_name)
            )["roi_label"].tolist()
        )[::-1],
        ordered=True,
    )
    component_order = [label for label in COMPONENT_COLORS if label in set(keep["component"])]
    sns.pointplot(
        data=keep,
        y="roi_label",
        x="drop_pre_minus_post",
        hue="component",
        hue_order=component_order,
        palette=COMPONENT_COLORS,
        errorbar=("ci", 95),
        dodge=0.55,
        markers=["o", "s", "D", "^", "v", "P"][: len(component_order)],
        linestyles="none",
        ax=ax,
    )
    ax.axvline(0, color="#444444", ls="--", lw=0.9)
    ax.set_xlabel("Similarity drop (Pre - Post)")
    ax.set_ylabel("")
    ax.set_title("Process values in the same top ROIs", fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.legend(title="", frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.16), ncol=2, fontsize=7.4)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learned-edge specificity.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    summary_path = paper_root / "tables_main" / "table_edge_specificity.tsv"
    delta_path = paper_root / "qc" / "edge_specificity" / "edge_delta_subject.tsv"
    summary = _read_table(summary_path)
    delta = _read_table(delta_path)
    plot_df = _prepare(summary, args.top_n)

    figures_dir = paper_root / "figures_main"
    figures_dir.mkdir(parents=True, exist_ok=True)
    table_path = figures_dir / "table_edge_specificity_top.tsv"
    plot_df.to_csv(table_path, sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, max(5.4, len(plot_df) * 0.34)), gridspec_kw={"width_ratios": [1.55, 1.35]})
    _plot_forest(axes[0], plot_df)
    _plot_edge_process(axes[1], delta, plot_df)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.text(0.05, 0.02, "* q < .05, ** q < .01, *** q < .001; + q < .10 within primary contrast family", fontsize=8)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.26, top=0.90, wspace=0.35)
    save_png_pdf(fig, figures_dir / "fig_edge_specificity.png")
    plt.close(fig)
    print(f"[edge-specificity-plot] wrote {figures_dir / 'fig_edge_specificity.png'}")
    print(f"[edge-specificity-plot] wrote {table_path}")


if __name__ == "__main__":
    main()
