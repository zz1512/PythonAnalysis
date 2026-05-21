#!/usr/bin/env python3
"""Task 12: final FDR tiering and evidence table."""

from __future__ import annotations

import pandas as pd

from shared_final import final_out, final_qc_root, read_table, write_text, write_tsv

MODULE = "s11_fdr_tiering_evidence_table"


def add_rows(rows: list[dict], path, module: str, tier: str, claim: str) -> None:
    if not path.exists():
        rows.append({"module": module, "tier": tier, "claim": claim, "status": "missing_output"})
        return
    df = read_table(path)
    q_col = "q_bh" if "q_bh" in df.columns else "q" if "q" in df.columns else None
    n_sig = int((pd.to_numeric(df[q_col], errors="coerce") < 0.05).sum()) if q_col else 0
    rows.append({"module": module, "tier": tier, "claim": claim, "status": "ok", "n_rows": len(df), "n_fdr": n_sig, "source": str(path)})


def main() -> None:
    root = final_qc_root()
    rows: list[dict] = []
    add_rows(rows, root / "s1_stage_trajectory_model" / "stage_trajectory_coefficients.tsv", "stage_trajectory", "Tier A", "stagewise post differentiation and retrieval rebound")
    add_rows(rows, root / "s2_memory_component_model" / "memory_component_coefficients.tsv", "memory_component", "Tier A", "memory bridge driven by prior post-stage separation")
    add_rows(rows, root / "s3_mahalanobis_robustness_proxy" / "mahalanobis_vs_correlation_comparison.tsv", "mahalanobis_proxy", "Tier B", "directional robustness proxy, not strict crossnobis")
    add_rows(rows, root / "s5_input_output_transformation" / "input_output_coefficients.tsv", "input_output", "Tier C", "lower post output similarity; slope boundary")
    add_rows(rows, root / "s6_global_similarity_centrality" / "global_centrality_models.tsv", "global_centrality", "Tier C", "global geometry refinement")
    add_rows(rows, root / "s7_material_moderation" / "material_moderation_coefficients.tsv", "material_moderation", "Tier C", "novelty/familiarity boundary")
    add_rows(rows, root / "s9_network_representational_coupling" / "network_coupling_item_models.tsv", "network_coupling", "Tier A", "retrieval-stage semantic-hpc coordination")
    add_rows(rows, root / "s10_hpc_long_axis_gate_summary" / "hpc_long_axis_summary.tsv", "hpc_long_axis", "Tier B", "exploratory anatomical robustness")
    evidence = pd.DataFrame(rows)
    write_tsv(evidence, final_out(MODULE, "final_evidence_tier_table.tsv"))
    write_tsv(evidence[evidence["tier"].isin(["Tier A"])], final_out(MODULE, "main_text_evidence_table.tsv"))
    write_tsv(evidence[evidence["tier"].isin(["Tier B", "Tier C"])], final_out(MODULE, "supplementary_evidence_table.tsv"))
    go = "GO" if (evidence["status"].eq("ok") & evidence["tier"].eq("Tier A")).sum() >= 2 else "NO-GO"
    write_text(f"# Go/No-Go\n\nDecision: **{go}**\n\nPrimary Tier A evidence is available; keep boundary analyses in SI.\n", final_out(MODULE, "go_no_go_report.md"))


if __name__ == "__main__":
    main()

