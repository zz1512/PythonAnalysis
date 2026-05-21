#!/usr/bin/env python3
"""Task 3: final memory component model wrapper."""

from __future__ import annotations

import shutil
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_tsv

MODULE = "s2_memory_component_model"


def main() -> None:
    coef = read_table(old_nc("a4_post_memory_component", "memory_component_model.tsv"))
    write_tsv(coef, final_out(MODULE, "memory_component_coefficients.tsv"))
    sens = coef[coef["outcome"].astype(str).isin(["memory_strict", "memory_lenient", "memory_prop"])].copy()
    write_tsv(sens, final_out(MODULE, "memory_component_sensitivity.tsv"))
    flags = []
    for network in sorted(coef["network"].dropna().unique()):
        sub = coef[(coef["network"].eq(network)) & coef["term"].astype(str).str.contains("condition\\[T.yy\\]:", regex=True)]
        post = sub[sub["term"].astype(str).str.contains("post_pair_similarity")]
        ret = sub[sub["term"].astype(str).str.contains("retrieval_pair_similarity")]
        flags.append({
            "network": network,
            "post_negative_any_fdr": bool(((post["estimate"] < 0) & (post["q_bh"] < 0.05)).any()),
            "retrieval_positive_any_fdr": bool(((ret["estimate"] > 0) & (ret["q_bh"] < 0.05)).any()),
            "interpretation": "prior_post_stage_separation" if ((post["estimate"] < 0) & (post["q_bh"] < 0.05)).any() else "boundary",
        })
    write_tsv(pd.DataFrame(flags), final_out(MODULE, "memory_component_interpretation_flags.tsv"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_a4_memory_components.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_memory_component_forest.png"))
        pdf = src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, final_out(MODULE, "fig_memory_component_forest.pdf"))
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_memory_post_vs_retrieval_effect.png"))
        pdf = src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, final_out(MODULE, "fig_memory_post_vs_retrieval_effect.pdf"))


if __name__ == "__main__":
    main()
