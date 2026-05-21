#!/usr/bin/env python3
"""Task 9: learning-update proxy gate.

Current data do not contain clean run3/run4 behavioral RT/fluency columns, so
this script writes an explicit boundary report rather than inventing a proxy.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, write_text, write_tsv

MODULE = "s8_learning_update_proxy"


def main() -> None:
    candidate = DEFAULT_PAPER_OUTPUTS / "qc" / "behavior_results" / "refined" / "behavior_trials.tsv"
    status = "missing_behavior_trials" if not candidate.exists() else "run3_run4_update_columns_unverified"
    metrics = pd.DataFrame([{"metric": "learning_update_rt", "status": status, "path": str(candidate)}])
    models = pd.DataFrame([{"model": "post_edge_differentiation ~ condition * learning_update_z", "status": "not_run_no_clean_run3_run4_behavior_proxy"}])
    summary = pd.DataFrame([{"interpretation": "learning-stage behavior did not explain post edge separation because no clean proxy was available", "status": "boundary_not_run"}])
    write_tsv(metrics, final_out(MODULE, "learning_update_metrics.tsv"))
    write_tsv(models, final_out(MODULE, "learning_update_models.tsv"))
    write_tsv(summary, final_out(MODULE, "learning_update_boundary_summary.tsv"))
    write_text("No clean run3/run4 behavioral RT or fluency proxy was available; do not use this as main evidence.\n", final_out(MODULE, "fig_learning_update_proxy.md"))
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.axis("off")
    ax.text(0.02, 0.65, "Learning-update proxy not run", fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.38, "No clean run3/run4 behavioral RT or fluency proxy was available.", fontsize=9, transform=ax.transAxes)
    ax.text(0.02, 0.18, "Interpretation: boundary check only; main storyline unchanged.", fontsize=9, transform=ax.transAxes)
    png = final_out(MODULE, "fig_learning_update_proxy.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
