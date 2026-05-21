#!/usr/bin/env python3
"""Plot learning-stage relation-vector trajectory checks."""

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


MODEL_LABELS = {
    "M9_relation_vector_direct": "direct",
    "M9_relation_vector_abs": "absolute",
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
    return pd.read_csv(path, sep="\t")


def _prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary[summary["is_primary_model"].astype(str).str.lower().isin(["true", "1"])].copy()
    numeric_cols = ["mean_run3", "mean_run4", "mean_delta", "t", "p", "cohens_dz", "q_bh_primary_family", "n_subjects"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    df = frame["n_subjects"] - 1
    se = (frame["mean_delta"].abs() / frame["t"].abs()).replace([np.inf, -np.inf], np.nan)
    tcrit = df.map(lambda value: stats.t.ppf(0.975, value) if np.isfinite(value) and value > 0 else np.nan)
    frame["ci_low"] = frame["mean_delta"] - tcrit * se
    frame["ci_high"] = frame["mean_delta"] + tcrit * se
    frame["roi_label"] = frame["roi"].map(abbreviate_roi_name)
    frame["condition_label"] = frame["condition"].map(CONDITION_LABELS).fillna(frame["condition"].str.upper())
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame["roi_set_label"] = frame["roi_set"].map(SET_LABELS).fillna(frame["roi_set"])
    frame["plot_label"] = frame["roi_set_label"] + " | " + frame["roi_label"] + " | " + frame["condition_label"] + " | " + frame["model_label"]
    return frame.sort_values(["q_bh_primary_family", "p"], na_position="last").reset_index(drop=True)


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


def _plot_forest(ax: plt.Axes, top: pd.DataFrame) -> None:
    plot_df = top.iloc[::-1].copy()
    y = np.arange(len(plot_df))
    colors = plot_df["roi_set"].map(SET_COLORS).fillna("#777777")
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [idx, idx], lw=1.8, color=colors.iloc[idx], alpha=0.8)
    ax.scatter(plot_df["mean_delta"], y, s=44, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        label = _star(float(row.q_bh_primary_family))
        if label:
            ax.text(row.ci_high + 0.003, idx, label, va="center", fontsize=9, fontweight="bold")
    ax.axvline(0, color="#444444", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"], fontsize=8)
    ax.set_xlabel("Delta rho (Run4 - Run3)")
    ax.set_title("Learning-stage trajectory: top primary effects", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SET_COLORS[roi_set], markeredgecolor="white", markersize=7, label=label)
        for roi_set, label in SET_LABELS.items()
    ]
    ax.legend(handles=handles, title="ROI set", frameon=False, fontsize=8, title_fontsize=8, loc="best")


def _plot_run_lines(ax: plt.Axes, subject: pd.DataFrame, top: pd.DataFrame, max_panels: int) -> None:
    records = []
    for rank, row in enumerate(top.head(max_panels).itertuples(index=False), start=1):
        subset = subject[
            subject["roi_set"].eq(row.roi_set)
            & subject["roi"].eq(row.roi)
            & subject["condition"].eq(row.condition)
            & subject["model"].eq(row.model)
        ].copy()
        subset["effect"] = f"{rank}. {row.roi_label}\n{row.condition_label}, {row.model_label}"
        records.append(subset)
    if not records:
        ax.axis("off")
        return
    plot_df = pd.concat(records, ignore_index=True)
    plot_df["run"] = pd.to_numeric(plot_df["run"], errors="coerce").map({3: "Run3", 4: "Run4"})
    plot_df["rho"] = pd.to_numeric(plot_df["rho"], errors="coerce")
    order = list(dict.fromkeys(plot_df["effect"]))
    sns.pointplot(
        data=plot_df,
        x="effect",
        y="rho",
        hue="run",
        order=order,
        hue_order=["Run3", "Run4"],
        errorbar=("ci", 95),
        dodge=0.25,
        markers=["o", "s"],
        linestyles="",
        palette={"Run3": "#8E8E8E", "Run4": "#2F5597"},
        ax=ax,
    )
    ax.axhline(0, color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("Subject rho")
    ax.set_title("Run-wise pair-pattern relation alignment", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="", frameon=False, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot relation-vector learning trajectory.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--run-panel-n", type=int, default=4)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    summary_path = paper_root / "tables_main" / "table_relation_learning_dynamics.tsv"
    subject_path = paper_root / "tables_si" / "table_relation_learning_dynamics_subject.tsv"
    out_dir = paper_root / "figures_main"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _prepare_summary(_read_table(summary_path))
    subject = _read_table(subject_path)
    top = summary.head(args.top_n).copy()
    top.to_csv(out_dir / "table_relation_learning_top.tsv", sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.65, 1.35]})
    _plot_forest(axes[0], top)
    _plot_run_lines(axes[1], subject, top, args.run_panel_n)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.text(0.40, 0.01, "Learning-stage patterns are pair/item patterns; trajectory tests run4 - run3 pair-pattern RDM alignment.", fontsize=8)
    fig.subplots_adjust(left=0.30, right=0.99, bottom=0.20, top=0.88, wspace=0.38)
    save_png_pdf(fig, out_dir / "fig_relation_learning_dynamics.png")
    plt.close(fig)
    print(f"[relation-learning-plot] wrote {out_dir / 'fig_relation_learning_dynamics.png'}")
    print(f"[relation-learning-plot] wrote {out_dir / 'table_relation_learning_top.tsv'}")


if __name__ == "__main__":
    main()
