#!/usr/bin/env python3
"""Task 6: input-output transformation final wrapper."""

from __future__ import annotations

import shutil
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_tsv

MODULE = "s5_input_output_transformation"


def main() -> None:
    coef = read_table(old_nc("c1_input_output_transformation", "input_output_transformation.tsv"))
    desc = read_table(old_nc("c1_input_output_transformation", "input_output_descriptives.tsv"))
    boundary = []
    for network, sub in coef.groupby("network"):
        cond = sub[sub["term"].eq("condition[T.yy]")]
        slope = sub[sub["term"].astype(str).str.contains("pre_pair_similarity_z:condition|condition\\[T.yy\\]:pre_pair", regex=True)]
        boundary.append({
            "network": network,
            "condition_main_fdr": bool(((cond["q_bh"] < 0.05) & (cond["estimate"] < 0)).any()),
            "slope_interaction_fdr": bool((slope["q_bh"] < 0.05).any()),
            "interpretation": "lower_post_output_similarity" if ((cond["q_bh"] < 0.05) & (cond["estimate"] < 0)).any() else "boundary",
        })
    write_tsv(coef, final_out(MODULE, "input_output_coefficients.tsv"))
    write_tsv(desc, final_out(MODULE, "input_output_slope_by_network.tsv"))
    write_tsv(pd.DataFrame(boundary), final_out(MODULE, "input_output_boundary_summary.tsv"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_c1_c2_c3_boundaries.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_input_output_partial_effect.png"))
        if src.with_suffix(".pdf").exists():
            shutil.copy2(src.with_suffix(".pdf"), final_out(MODULE, "fig_input_output_partial_effect.pdf"))


if __name__ == "__main__":
    main()

