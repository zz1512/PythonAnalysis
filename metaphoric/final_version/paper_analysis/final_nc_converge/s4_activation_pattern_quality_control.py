#!/usr/bin/env python3
"""Task 5: mean activation / pattern quality sanity checks."""

from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_tsv

MODULE = "s4_activation_pattern_quality_control"


def main() -> None:
    controlled = read_table(DEFAULT_PAPER_OUTPUTS / "qc" / "reviewer_supp" / "m2_univariate_sanity" / "step5c_univariate_controlled.tsv")
    stage = read_table(old_nc("a4_post_memory_component", "stage_trajectory_model.tsv"))
    edge = controlled[controlled["response"].astype(str).isin(["post_edge_specificity", "trained_edge_drop"])].copy()
    summary = []
    for response, sub in edge.groupby("response"):
        cond = sub[sub["term"].astype(str).str.contains("condition", regex=False)]
        summary.append({
            "response": response,
            "n_condition_terms": len(cond),
            "n_condition_fdr": int((cond["q_bh"] < 0.05).sum()),
            "direction_reversed_any": bool((cond["estimate"] < 0).any()),
            "interpretation": "robust_direction_retained" if not (cond["estimate"] < 0).any() else "review_direction",
        })
    write_tsv(pd.DataFrame(summary), final_out(MODULE, "activation_pattern_qc_summary.tsv"))
    write_tsv(stage, final_out(MODULE, "controlled_stage_model.tsv"))
    write_tsv(edge, final_out(MODULE, "controlled_edge_specificity_model.tsv"))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    plot_df = pd.DataFrame(summary)
    ax.bar(plot_df["response"], plot_df["n_condition_fdr"], color="#4f7f61")
    ax.set_ylabel("FDR-significant YY terms")
    ax.set_title("Activation / pattern-quality sanity check", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png = final_out(MODULE, "fig_activation_quality_control.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
