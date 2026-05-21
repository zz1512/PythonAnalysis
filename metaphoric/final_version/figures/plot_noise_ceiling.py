#!/usr/bin/env python3
"""
Plot B5 noise ceiling summaries from paper_outputs tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import abbreviate_roi_name, add_panel_label, apply_publication_rcparams, default_base_dir, default_figure_dir, save_png_pdf


def _default_table() -> Path:
    return default_base_dir() / "paper_outputs" / "tables_si" / "table_noise_ceiling.tsv"


def _default_out_path() -> Path:
    return default_figure_dir("figures_si") / "fig_noise_ceiling.png"


def _label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "analysis" not in out.columns:
        out["analysis"] = "unknown"
    out["roi_label"] = out["roi_scope"].map(abbreviate_roi_name)
    out["label"] = out["roi_label"]
    step_mask = out["analysis"].eq("step5c_itemwise_delta")
    out.loc[step_mask, "label"] = (
        out.loc[step_mask, "roi_label"].astype(str)
        + " | "
        + out.loc[step_mask, "condition"].astype(str).str.upper()
    )
    model_mask = out["analysis"].eq("model_rsa_neural_rdm")
    out.loc[model_mask, "label"] = (
        out.loc[model_mask, "roi_label"].astype(str)
        + " | "
        + out.loc[model_mask, "time"].astype(str)
        + " | "
        + out.loc[model_mask, "condition"].astype(str)
    )
    return out


def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, *, title: str) -> None:
    work = frame.sort_values("lower_mean_r", ascending=True).reset_index(drop=True)
    y = np.arange(len(work))
    ax.hlines(y, work["lower_mean_r"], work["upper_mean_r"], color="#94a3b8", linewidth=2.0, zorder=1)
    ax.scatter(work["lower_mean_r"], y, color="#0f766e", s=34, label="Lower", zorder=3)
    ax.scatter(work["upper_mean_r"], y, color="#b45309", s=34, label="Upper", zorder=3)
    ax.axvline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(work["label"], fontsize=8.5)
    ax.set_xlabel("Noise ceiling (r)")
    ax.set_ylabel("")
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B5 noise ceiling summary figure.")
    parser.add_argument("--table", type=Path, default=_default_table())
    parser.add_argument("--out-path", type=Path, default=_default_out_path())
    parser.add_argument(
        "--model-condition",
        default="all",
        help="Model-RSA condition group to display (default: all).",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.table, sep="\t")
    frame = _label_rows(frame)
    step = frame[frame["analysis"] == "step5c_itemwise_delta"].copy()
    model = frame[
        (frame["analysis"] == "model_rsa_neural_rdm")
        & (frame["condition"].astype(str) == str(args.model_condition))
    ].copy()

    if step.empty and model.empty:
        raise ValueError("No noise ceiling rows matched the requested filters.")

    apply_publication_rcparams()
    n_rows = max(len(step), len(model), 1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, max(4.8, 0.32 * n_rows + 2.0)),
        constrained_layout=True,
    )
    if step.empty:
        axes[0].axis("off")
    else:
        _plot_panel(axes[0], step, title="Step 5C item-wise delta noise ceiling")
        add_panel_label(axes[0], "a")
    if model.empty:
        axes[1].axis("off")
    else:
        _plot_panel(axes[1], model, title=f"Model-RSA neural RDM noise ceiling ({args.model_condition})")
        add_panel_label(axes[1], "b")

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        unique = {}
        for h, l in zip(handles, labels):
            unique.setdefault(l, h)
        fig.legend(unique.values(), unique.keys(), frameon=False, loc="upper center", ncol=2)

    save_png_pdf(fig, args.out_path)
    plt.close(fig)
    print(f"[noise-ceiling-plot] wrote {args.out_path}")


if __name__ == "__main__":
    main()
