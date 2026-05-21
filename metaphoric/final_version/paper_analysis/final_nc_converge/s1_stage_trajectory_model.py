#!/usr/bin/env python3
"""Task 2: final stage trajectory model wrapper and figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shared_final import final_out, old_nc, read_table, write_tsv

MODULE = "s1_stage_trajectory_model"


def plot(desc: pd.DataFrame, network: str, out_stem: str) -> None:
    sub = desc[desc["network"].eq(network)].copy()
    sub["stage"] = pd.Categorical(sub["stage"], ["pre", "post", "retrieval"], ordered=True)
    sub = sub.sort_values(["condition", "stage"])
    colors = {"yy": "#b23a48", "kj": "#3574b7"}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for cond, grp in sub.groupby("condition"):
        ax.plot(grp["stage"].astype(str), grp["mean_raw"], marker="o", linewidth=2, color=colors.get(cond, "#555"), label=cond.upper())
    ax.set_title(f"Stage trajectory: {network}", fontweight="bold")
    ax.set_ylabel("Pair similarity")
    ax.set_xlabel("Stage")
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = final_out(MODULE, out_stem + ".png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    coef = read_table(old_nc("a4_post_memory_component", "stage_trajectory_model.tsv"))
    desc = read_table(old_nc("a4_post_memory_component", "stage_trajectory_descriptives.tsv"))
    write_tsv(coef, final_out(MODULE, "stage_trajectory_coefficients.tsv"))
    write_tsv(desc, final_out(MODULE, "stage_trajectory_emmeans.tsv"))
    summary = desc.pivot_table(index=["network", "condition"], columns="stage", values="mean_raw", aggfunc="mean").reset_index()
    if {"pre", "post", "retrieval"}.issubset(summary.columns):
        summary["post_minus_pre"] = summary["post"] - summary["pre"]
        summary["retrieval_minus_post"] = summary["retrieval"] - summary["post"]
    write_tsv(summary, final_out(MODULE, "stage_trajectory_network_summary.tsv"))
    plot(desc, "semantic", "fig_stage_trajectory_meta_metaphor")
    plot(desc, "hpc_spatial", "fig_stage_trajectory_hpc_spatial")


if __name__ == "__main__":
    main()
