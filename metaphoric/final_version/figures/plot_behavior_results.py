#!/usr/bin/env python3
"""
plot_behavior_results.py

Create the primary behavior figure from refined behavior outputs.

Outputs
- fig_behavior_accuracy_rt.png
- table_behavior_plot_summary.tsv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import add_panel_label, apply_publication_rcparams, save_png_pdf


PALETTE = {"KJ": "#3b82f6", "YY": "#d9485f"}


def _default_behavior_dir() -> Path:
    base = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
    return base / "behavior_results" / "refined"


def _default_out_dir() -> Path:
    base = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
    override = os.environ.get("METAPHOR_FIG_OUT_DIR", "").strip()
    if override:
        return Path(override)
    return base / "figures_main_story"


def _load_subject_condition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"subject", "condition", "accuracy", "mean_rt_correct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Behavior subject-condition summary missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["condition"] = frame["condition"].astype(str).str.upper()
    frame = frame[frame["condition"].isin(["KJ", "YY"])].copy()
    return frame


def _build_long_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_map = {
        "accuracy": "Memory Accuracy",
        "mean_rt_correct": "Correct-Trial RT (ms)",
    }
    for source_col, metric_label in metric_map.items():
        for _, row in frame.iterrows():
            rows.append(
                {
                    "subject": row["subject"],
                    "condition": row["condition"],
                    "metric": metric_label,
                    "value": row[source_col],
                }
            )
    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    return out


def _write_summary(long_frame: pd.DataFrame, out_path: Path) -> None:
    summary = (
        long_frame.groupby(["metric", "condition"], as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, sep="\t", index=False)


def _plot(long_frame: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    apply_publication_rcparams()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    metric_order = ["Memory Accuracy", "Correct-Trial RT (ms)"]
    ylabels = {
        "Memory Accuracy": "Proportion Correct",
        "Correct-Trial RT (ms)": "RT (ms)",
    }

    for ax, metric in zip(axes, metric_order):
        subset = long_frame[long_frame["metric"] == metric].copy()
        pivot = subset.pivot(index="subject", columns="condition", values="value").reset_index()
        for _, row in pivot.iterrows():
            ax.plot(["KJ", "YY"], [row["KJ"], row["YY"]], color="#cbd5e1", linewidth=1.0, alpha=0.8, zorder=1)

        sns.violinplot(
            data=subset,
            x="condition",
            y="value",
            hue="condition",
            order=["KJ", "YY"],
            palette=PALETTE,
            dodge=False,
            inner=None,
            cut=0,
            linewidth=1.0,
            ax=ax,
            zorder=2,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for collection in ax.collections[:2]:
            collection.set_alpha(0.70)

        sns.boxplot(
            data=subset,
            x="condition",
            y="value",
            order=["KJ", "YY"],
            width=0.18,
            showcaps=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1, "alpha": 0.85},
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": "black", "linewidth": 1.1},
            capprops={"color": "black", "linewidth": 1.1},
            ax=ax,
            zorder=3,
        )
        for xpos, cond in enumerate(["KJ", "YY"]):
            vals = subset.loc[subset["condition"] == cond, "value"].to_numpy()
            jitter = np.random.default_rng(42 + xpos).uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=22,
                color=PALETTE[cond],
                alpha=0.38,
                edgecolor="white",
                linewidth=0.3,
                zorder=4,
            )
        stats = subset.groupby("condition", as_index=False)["value"].agg(["mean", "std"]).reset_index()
        for xpos, (_, row) in enumerate(stats.iterrows()):
            ax.errorbar(
                xpos,
                row["mean"],
                yerr=row["std"],
                fmt="o",
                color="black",
                elinewidth=1.2,
                capsize=3,
                markersize=4.5,
                zorder=5,
            )
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabels[metric])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, "a" if metric == "Memory Accuracy" else "b")

    fig.tight_layout()
    save_png_pdf(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot refined behavior results.")
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        default=_default_behavior_dir(),
        help="Directory containing refined behavior outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Directory where figure outputs will be written.",
    )
    args = parser.parse_args()

    behavior_dir = args.behavior_dir
    out_dir = args.out_dir

    frame = _load_subject_condition(behavior_dir / "subject_condition_summary.tsv")
    long_frame = _build_long_plot_frame(frame)
    _write_summary(long_frame, out_dir / "table_behavior_plot_summary.tsv")
    _plot(long_frame, out_dir / "fig_behavior_accuracy_rt.png")
    print(f"[behavior-plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
