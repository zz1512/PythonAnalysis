#!/usr/bin/env python3
"""
Plot C7 ROI-level RD and GPS summaries from paper_outputs tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import abbreviate_roi_name, add_panel_label, apply_publication_rcparams, default_base_dir, default_figure_dir, save_png_pdf


def _default_table(metric: str, roi_set: str) -> Path:
    return default_base_dir() / "paper_outputs" / "tables_si" / f"table_{metric}_{roi_set}.tsv"


def _default_out_path(roi_set: str) -> Path:
    return default_figure_dir("figures_si") / f"fig_rd_gps_{roi_set}.png"


def _prep_metric(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    work = frame.copy()
    work["metric"] = metric_name
    work["roi_label"] = work["roi"].map(abbreviate_roi_name)
    work["yy_minus_kj_delta"] = (
        pd.to_numeric(work["mean_yy_delta"], errors="coerce")
        - pd.to_numeric(work["mean_kj_delta"], errors="coerce")
    )
    return work


def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, *, title: str) -> None:
    work = frame.sort_values("yy_minus_kj_delta", ascending=True).reset_index(drop=True)
    y = np.arange(len(work))
    ax.hlines(y, work["mean_kj_delta"], work["mean_yy_delta"], color="#cbd5e1", linewidth=2.0, zorder=1)
    ax.scatter(work["mean_kj_delta"], y, color="#3574b7", s=36, label="Spatial delta", zorder=3)
    ax.scatter(work["mean_yy_delta"], y, color="#b23a48", s=36, label="Metaphor delta", zorder=4)
    ax.axvline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(work["roi_label"], fontsize=8.8)
    ax.set_xlabel("Post - pre change")
    ax.set_ylabel("")
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, row in work.iterrows():
        pval = pd.to_numeric(row.get("p_yy_vs_kj_delta"), errors="coerce")
        if np.isfinite(pval) and pval < 0.05:
            xpos = max(float(row["mean_yy_delta"]), float(row["mean_kj_delta"]))
            ax.text(xpos, yi + 0.16, "*", fontsize=10, fontweight="bold", color="black")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROI-level RD and GPS summary figure.")
    parser.add_argument("--roi-set", default=os.environ.get("METAPHOR_ROI_SET", "main_functional"))
    parser.add_argument("--rd-table", type=Path, default=None)
    parser.add_argument("--gps-table", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    rd_table = args.rd_table or _default_table("rd", args.roi_set)
    gps_table = args.gps_table or _default_table("gps", args.roi_set)
    out_path = args.out_path or _default_out_path(args.roi_set)

    rd = _prep_metric(pd.read_csv(rd_table, sep="\t"), "RD")
    gps = _prep_metric(pd.read_csv(gps_table, sep="\t"), "GPS")

    apply_publication_rcparams()
    n_rows = max(len(rd), len(gps), 1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.6, max(4.6, 0.34 * n_rows + 2.0)),
        constrained_layout=True,
        sharey=False,
    )
    _plot_panel(axes[0], rd, title="RD summary")
    _plot_panel(axes[1], gps, title="GPS summary")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:2], labels[:2], frameon=False, loc="upper center", ncol=2)

    save_png_pdf(fig, out_path)
    plt.close(fig)
    print(f"[rd-gps-plot] wrote {out_path}")


if __name__ == "__main__":
    main()
