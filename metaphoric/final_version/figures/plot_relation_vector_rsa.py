#!/usr/bin/env python3
"""
Plot relation-vector RSA summaries.

Outputs a compact mechanism figure plus main/SI tables for the relation-vector
Model-RSA branch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from figure_utils import (
    abbreviate_roi_name,
    add_panel_label,
    apply_publication_rcparams,
    default_paper_output_root,
    save_png_pdf,
)


ROI_SETS = ["main_functional", "literature", "literature_spatial"]
PRIMARY_MODELS = ["M9_relation_vector_direct", "M9_relation_vector_abs"]
MODEL_LABELS = {
    "M9_relation_vector_direct": "direct vector",
    "M9_relation_vector_abs": "absolute vector",
    "M9_relation_vector_length": "length control",
    "M3_embedding_pair_distance": "pair-distance control",
    "M3_embedding_pair_centroid": "centroid control",
}
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
SET_LABELS = {
    "main_functional": "Main functional",
    "literature": "Metaphor literature",
    "literature_spatial": "Spatial literature",
}
SET_COLORS = {
    "main_functional": "#4C78A8",
    "literature": "#F58518",
    "literature_spatial": "#54A24B",
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _load_summaries(paper_output_root: Path, roi_sets: list[str]) -> pd.DataFrame:
    frames = []
    for roi_set in roi_sets:
        path = paper_output_root / "qc" / f"relation_vector_rsa_{roi_set}" / "relation_vector_group_summary_fdr.tsv"
        frame = _read_table(path)
        if "roi_set" not in frame.columns:
            frame.insert(0, "roi_set", roi_set)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["roi_label"] = combined["roi"].map(abbreviate_roi_name)
    combined["model_label"] = combined["model"].map(MODEL_LABELS).fillna(combined["model"])
    combined["condition_label"] = combined["condition"].map(CONDITION_LABELS).fillna(combined["condition"].str.upper())
    combined["roi_set_label"] = combined["roi_set"].map(SET_LABELS).fillna(combined["roi_set"])
    return combined


def _prepare_primary(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[
        summary["model"].isin(PRIMARY_MODELS)
        & summary["neural_rdm_type"].eq("relation_vector")
    ].copy()
    primary["q_plot"] = pd.to_numeric(primary["q_bh_primary_family"], errors="coerce")
    primary["p"] = pd.to_numeric(primary["p"], errors="coerce")
    primary["mean_delta"] = pd.to_numeric(primary["mean_delta"], errors="coerce")
    primary["t"] = pd.to_numeric(primary["t"], errors="coerce")
    primary["n_subjects"] = pd.to_numeric(primary["n_subjects"], errors="coerce")
    primary["cohens_dz"] = pd.to_numeric(primary["cohens_dz"], errors="coerce")
    df = primary["n_subjects"] - 1
    se = (primary["mean_delta"].abs() / primary["t"].abs()).replace([np.inf, -np.inf], np.nan)
    tcrit = df.map(lambda value: stats.t.ppf(0.975, value) if np.isfinite(value) and value > 0 else np.nan)
    primary["ci_low"] = primary["mean_delta"] - tcrit * se
    primary["ci_high"] = primary["mean_delta"] + tcrit * se
    primary["plot_label"] = (
        primary["roi_set_label"]
        + " | "
        + primary["roi_label"]
        + " | "
        + primary["condition_label"]
        + " | "
        + primary["model_label"]
    )
    return primary.sort_values(["q_plot", "p", "roi_set", "roi"]).reset_index(drop=True)


def _load_subject_metrics(paper_output_root: Path) -> pd.DataFrame:
    path = paper_output_root / "tables_si" / "table_relation_vector_subject_metrics.tsv"
    frame = _read_table(path)
    frame["roi_label"] = frame["roi"].map(abbreviate_roi_name)
    frame["condition_label"] = frame["condition"].map(CONDITION_LABELS).fillna(frame["condition"].str.upper())
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame["roi_set_label"] = frame["roi_set"].map(SET_LABELS).fillna(frame["roi_set"])
    frame["rho"] = pd.to_numeric(frame["rho"], errors="coerce")
    return frame


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


def _plot_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cue = np.array([0.22, 0.33])
    target = np.array([0.73, 0.70])
    centroid = (cue + target) / 2
    ax.scatter([cue[0], target[0]], [cue[1], target[1]], s=360, color=["#4C78A8", "#E45756"], edgecolor="white", linewidth=1.2)
    ax.annotate("", xy=target, xytext=cue, arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#333333"})
    ax.scatter([centroid[0]], [centroid[1]], s=95, color="#72B7B2", edgecolor="white", zorder=3)
    ax.text(cue[0], cue[1] - 0.13, "cue", ha="center", va="top", fontsize=10)
    ax.text(target[0], target[1] + 0.11, "target", ha="center", va="bottom", fontsize=10)
    ax.text(centroid[0] + 0.04, centroid[1] - 0.02, "centroid", ha="left", va="center", fontsize=9, color="#2B6F6A")
    ax.text(0.5, 0.94, "Relation-vector model", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.08, "model RDM = dissimilarity between target-cue vectors", ha="center", va="bottom", fontsize=9)


def _plot_forest(ax: plt.Axes, primary: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    plot_df = primary.head(max_rows).iloc[::-1].copy()
    y = np.arange(len(plot_df))
    colors = plot_df["roi_set"].map(SET_COLORS).fillna("#777777")
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [idx, idx], color=colors.iloc[idx], lw=1.8, alpha=0.75)
    ax.scatter(plot_df["mean_delta"], y, s=44, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        label = _star(float(row.q_plot))
        if label:
            ax.text(row.ci_high + 0.004, idx, label, ha="left", va="center", fontsize=9, fontweight="bold")
    ax.axvline(0, color="#444444", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"], fontsize=8)
    ax.set_xlabel("Delta rho (Post - Pre)")
    ax.set_title("Primary relation-vector effects", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=7, label=label)
        for roi_set, label in SET_LABELS.items()
        for color in [SET_COLORS[roi_set]]
    ]
    ax.legend(handles=handles, title="ROI set", frameon=False, loc="lower right", fontsize=8, title_fontsize=8)
    return plot_df.iloc[::-1].copy()


def _plot_pre_post(ax: plt.Axes, subject_metrics: pd.DataFrame, top_rows: pd.DataFrame, *, max_panels: int) -> None:
    records = []
    for rank, row in enumerate(top_rows.head(max_panels).itertuples(index=False), start=1):
        subset = subject_metrics[
            subject_metrics["roi_set"].eq(row.roi_set)
            & subject_metrics["roi"].eq(row.roi)
            & subject_metrics["condition"].eq(row.condition)
            & subject_metrics["neural_rdm_type"].eq("relation_vector")
            & subject_metrics["model"].eq(row.model)
        ].copy()
        subset["effect"] = f"{rank}. {row.roi_label}\n{row.condition_label}, {row.model_label}"
        records.append(subset)
    if not records:
        ax.text(0.5, 0.5, "No subject metrics", ha="center", va="center")
        ax.axis("off")
        return
    plot_df = pd.concat(records, ignore_index=True)
    order = list(dict.fromkeys(plot_df["effect"]))
    sns.pointplot(
        data=plot_df,
        x="effect",
        y="rho",
        hue="time",
        order=order,
        hue_order=["pre", "post"],
        errorbar=("ci", 95),
        dodge=0.28,
        markers=["o", "s"],
        linestyles="",
        palette={"pre": "#8E8E8E", "post": "#2F5597"},
        ax=ax,
    )
    ax.axhline(0, color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("Subject rho")
    ax.set_title("Top effects: pre/post alignment", fontweight="bold")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="", frameon=False, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot relation-vector RSA mechanism figure.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--roi-sets", nargs="+", default=ROI_SETS)
    parser.add_argument("--top-n", type=int, default=16)
    parser.add_argument("--prepost-n", type=int, default=4)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    figures_dir = paper_root / "figures_main"
    tables_main = paper_root / "tables_main"
    tables_si = paper_root / "tables_si"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_main.mkdir(parents=True, exist_ok=True)
    tables_si.mkdir(parents=True, exist_ok=True)

    summary = _load_summaries(paper_root, list(args.roi_sets))
    primary = _prepare_primary(summary)
    subject_metrics = _load_subject_metrics(paper_root)

    full_path = tables_si / "table_relation_vector_rsa_full.tsv"
    top_path = figures_dir / "table_relation_vector_rsa_top.tsv"
    main_path = tables_main / "table_relation_vector_rsa.tsv"
    summary.to_csv(full_path, sep="\t", index=False)
    primary.to_csv(main_path, sep="\t", index=False)
    primary.head(max(args.top_n, 30)).to_csv(top_path, sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6.4),
        gridspec_kw={"width_ratios": [1.05, 1.95, 1.55]},
    )
    ax_a, ax_b, ax_c = axes

    _plot_schematic(ax_a)
    plotted = _plot_forest(ax_b, primary, max_rows=args.top_n)
    _plot_pre_post(ax_c, subject_metrics, primary, max_panels=args.prepost_n)
    fig.text(0.36, 0.01, "* q < .05, ** q < .01, *** q < .001; + q < .10 within primary family", fontsize=8)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.18, top=0.90, wspace=0.55)
    save_png_pdf(fig, figures_dir / "fig_relation_vector_rsa.png")
    plt.close(fig)

    print(f"[relation-vector-plot] wrote {figures_dir / 'fig_relation_vector_rsa.png'}")
    print(f"[relation-vector-plot] wrote {top_path}")
    print(f"[relation-vector-plot] wrote {full_path}")
    print(f"[relation-vector-plot] plotted_top_rows={len(plotted)}")


if __name__ == "__main__":
    main()
