#!/usr/bin/env python3
"""
Create a compact Model-RSA mechanism figure from main_functional and literature outputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from figure_utils import (
    abbreviate_roi_name,
    add_panel_label,
    apply_publication_rcparams,
    default_base_dir,
    default_figure_dir,
    save_png_pdf,
)


MODEL_ORDER = [
    "M2_pair",
    "M8_reverse_pair",
    "M3_embedding",
    "M1_condition",
    "M7_binary",
    "M7_continuous_confidence",
]
MODEL_LABELS = {
    "M2_pair": "M2 pair",
    "M8_reverse_pair": "M8 reverse pair",
    "M3_embedding": "M3 embedding",
    "M1_condition": "M1 condition",
    "M7_binary": "M7 binary",
    "M7_continuous_confidence": "M7 cont. confidence",
}


def _base_dir() -> Path:
    return default_base_dir()


def _default_out_dir() -> Path:
    return default_figure_dir("figures_main")


def _load_group_summary(path: Path, roi_set: str, *, condition_group: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"roi", "model", "mean_a", "mean_b", "p"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Model-RSA summary missing columns: {sorted(missing)} ({path})")
    frame = frame.copy()
    if "condition_group" in frame.columns:
        frame["condition_group"] = frame["condition_group"].fillna("all").astype(str)
        frame = frame[frame["condition_group"] == condition_group].copy()
    else:
        frame["condition_group"] = "all"
    frame["model"] = frame["model"].replace({"M7_memory": "M7_binary"})
    frame = frame[frame["model"].isin(MODEL_ORDER)].copy()
    frame["roi_set"] = roi_set
    frame["delta_rho"] = frame["mean_a"] - frame["mean_b"]
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame["roi_label"] = frame["roi"].map(abbreviate_roi_name)
    frame["sig_label"] = ""
    frame.loc[frame["p"] < 0.05, "sig_label"] = "*"
    frame.loc[frame["p"] < 0.01, "sig_label"] = "**"
    frame.loc[frame["p"] < 0.001, "sig_label"] = "***"
    return frame


def _plot_panel(ax: plt.Axes, subset: pd.DataFrame, title: str, vlim: float, show_cbar: bool) -> None:
    work = subset.copy()
    roi_order = work.groupby("roi_label")["delta_rho"].min().sort_values().index.tolist()
    heat = (
        work.pivot(index="model_label", columns="roi_label", values="delta_rho")
        .reindex([MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in work["model_label"].unique()])
        [roi_order]
    )
    annot = (
        work.pivot(index="model_label", columns="roi_label", values="sig_label")
        .reindex([MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in work["model_label"].unique()])
        [roi_order]
    )
    sns.heatmap(
        heat,
        cmap="RdBu_r",
        center=0,
        vmin=-vlim,
        vmax=vlim,
        linewidths=0.5,
        linecolor="white",
        cbar=show_cbar,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        cbar_kws={"label": "Delta rho (Post - Pre)", "shrink": 0.7} if show_cbar else None,
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact Model-RSA mechanism heatmaps.")
    parser.add_argument(
        "--main-file",
        type=Path,
        default=_base_dir() / "paper_outputs" / "qc" / "model_rdm_results_main_functional" / "model_rdm_group_summary.tsv",
    )
    parser.add_argument(
        "--literature-file",
        type=Path,
        default=_base_dir() / "paper_outputs" / "qc" / "model_rdm_results_literature" / "model_rdm_group_summary.tsv",
    )
    parser.add_argument("--condition-group", default="all", help="Condition group to plot, default: pooled all.")
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir())
    args = parser.parse_args()

    main_df = _load_group_summary(args.main_file, "main_functional", condition_group=args.condition_group)
    lit_df = _load_group_summary(args.literature_file, "literature", condition_group=args.condition_group)
    combined = pd.concat([main_df, lit_df], ignore_index=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "table_model_rsa_mechanism.tsv", sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), constrained_layout=True)
    vlim = float(combined["delta_rho"].abs().max())

    _plot_panel(axes[0], main_df, f"Model-RSA Mechanism: main_functional ({args.condition_group})", vlim=vlim, show_cbar=True)
    _plot_panel(axes[1], lit_df, f"Model-RSA Mechanism: literature ({args.condition_group})", vlim=vlim, show_cbar=False)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")

    save_png_pdf(fig, out_dir / "fig_model_rsa_mechanism.png")
    plt.close(fig)
    print(f"[model-rsa-plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
