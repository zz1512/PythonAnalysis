#!/usr/bin/env python3
"""Task 8: novelty / familiarity moderation final wrapper."""

from __future__ import annotations

import shutil
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_tsv

MODULE = "s7_material_moderation"


def main() -> None:
    coef = read_table(old_nc("c3_novelty_familiarity_moderation", "novelty_familiarity_moderation.tsv"))
    coverage = read_table(old_nc("c3_novelty_familiarity_moderation", "novelty_familiarity_input_coverage.tsv"))
    rows = []
    for network, sub in coef.groupby("network"):
        cond = sub[sub["term"].eq("condition[T.yy]")]
        nov = sub[sub["term"].eq("condition[T.yy]:novelty_z")]
        rows.append({
            "network": network,
            "condition_stable": bool(((cond["q_bh"] < 0.05) & cond["status"].eq("ok")).any()),
            "novelty_interaction_stable": bool(((nov["q_bh"] < 0.05) & nov["status"].eq("ok")).any()),
            "interpretation": "novelty_familiarity_boundary_check",
        })
    write_tsv(coef, final_out(MODULE, "material_moderation_coefficients.tsv"))
    write_tsv(coverage, final_out(MODULE, "material_covariate_balance.tsv"))
    write_tsv(pd.DataFrame(rows), final_out(MODULE, "novelty_familiarity_boundary_summary.tsv"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_c1_c2_c3_boundaries.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_material_moderation_forest.png"))
        if src.with_suffix(".pdf").exists():
            shutil.copy2(src.with_suffix(".pdf"), final_out(MODULE, "fig_material_moderation_forest.pdf"))


if __name__ == "__main__":
    main()
