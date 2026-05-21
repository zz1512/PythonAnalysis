#!/usr/bin/env python3
"""Task 11: HPC long-axis exploratory gate and summary."""

from __future__ import annotations

import shutil
import pandas as pd

from shared_final import DEFAULT_PAPER_OUTPUTS, HPC_QC_ROOT, final_out, read_table, write_text, write_tsv

MODULE = "s10_hpc_long_axis_gate_summary"


def main() -> None:
    manifest = read_table(HPC_QC_ROOT / "s1_segmentation" / "subfield_manifest.tsv")
    s3 = read_table(HPC_QC_ROOT / "s3_step5c_subfield" / "subfield_step5c.tsv")
    s4 = read_table(HPC_QC_ROOT / "s4_edge_specificity_subfield" / "subfield_edge_specificity.tsv")
    s5 = read_table(HPC_QC_ROOT / "s5_anterior_posterior_contrast" / "anterior_posterior_contrast.tsv")
    source_text = manifest.astype(str).agg(" ".join, axis=1)
    status_col = manifest["status"].astype(str) if "status" in manifest.columns else pd.Series("", index=manifest.index)
    gate = pd.DataFrame([
        {"check": "FS60_available", "value": bool(source_text.str.contains("FS60", case=False, na=False).any()), "status": "exploratory_mni_fallback"},
        {"check": "mni_fallback_used", "value": bool((manifest.astype(str).apply(lambda c: c.str.contains("mni", case=False, na=False))).any().any()), "status": "ok"},
        {"check": "qc_warn_count", "value": int(status_col.str.contains("warn", case=False, na=False).sum()), "status": "report"},
        {"check": "s3_singular_count", "value": int(s3["status"].astype(str).str.contains("Singular", case=False, na=False).sum()), "status": "boundary"},
        {"check": "s4_edge_specificity_fdr_count", "value": int((s4.get("q", pd.Series(dtype=float)) < 0.05).sum()), "status": "supportive"},
        {"check": "s5_ap_interaction_fdr", "value": bool(((s5["term"].astype(str).str.contains("axis_position")) & (s5["q"] < 0.05)).any()), "status": "no_stable_gradient"},
    ])
    write_tsv(gate, final_out(MODULE, "hpc_long_axis_gate_report.tsv"))
    write_tsv(pd.concat([s4.assign(source_table="s4_edge_specificity"), s5.assign(source_table="s5_ap_gradient")], ignore_index=True, sort=False), final_out(MODULE, "hpc_long_axis_summary.tsv"))
    text = (
        "HPC long-axis is exploratory anatomical robustness only. MNI fallback was used; "
        "S4 supports broad trained-edge differentiation, but S5 does not support a stable anterior-posterior gradient.\n"
    )
    write_text(text, final_out(MODULE, "hpc_long_axis_interpretation.md"))
    src = DEFAULT_PAPER_OUTPUTS / "figures_result_final" / "fig_result_final_hpc_long_axis_summary.png"
    if src.exists():
        shutil.copy2(src, final_out(MODULE, "fig_hpc_long_axis_summary.png"))
        if src.with_suffix(".pdf").exists():
            shutil.copy2(src.with_suffix(".pdf"), final_out(MODULE, "fig_hpc_long_axis_summary.pdf"))


if __name__ == "__main__":
    main()
