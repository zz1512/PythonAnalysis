#!/usr/bin/env python3
"""Task 10: predefined network representational coupling final wrapper."""

from __future__ import annotations

import shutil
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, final_out, old_nc, read_table, write_tsv

MODULE = "s9_network_representational_coupling"


def main() -> None:
    models = read_table(old_nc("b3_repr_coupling", "repr_coupling.tsv"))
    item = models.copy()
    subject = pd.DataFrame([{"model": "rdm_coupling_z ~ condition * stage + (1|subject)", "status": "not_run_no_subject_rdm_coupling_input"}])
    summary_rows = []
    for coupling, sub in models.groupby("coupling"):
        yy_inter = sub[sub["term"].astype(str).str.contains(":condition\\[T.yy\\]", regex=True)]
        main = sub[~sub["term"].astype(str).str.contains(":condition\\[T.yy\\]", regex=True)]
        summary_rows.append({
            "coupling": coupling,
            "n_terms": len(sub),
            "any_main_fdr": bool(((main["q_bh"] < 0.05) & main["status"].eq("ok")).any()),
            "yy_specific_interaction_fdr": bool(((yy_inter["q_bh"] < 0.05) & yy_inter["status"].eq("ok")).any()),
            "interpretation": "retrieval-stage semantic-hpc coordination" if "retrieval" in coupling else "YY condition-level post hpc-spatial differentiation",
        })
    write_tsv(item, final_out(MODULE, "network_coupling_item_models.tsv"))
    write_tsv(subject, final_out(MODULE, "network_coupling_subject_models.tsv"))
    write_tsv(pd.DataFrame(summary_rows), final_out(MODULE, "network_coupling_summary.tsv"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_b3_network_coupling.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_network_coupling.png"))
        if src.with_suffix(".pdf").exists():
            shutil.copy2(src.with_suffix(".pdf"), final_out(MODULE, "fig_network_coupling.pdf"))


if __name__ == "__main__":
    main()

