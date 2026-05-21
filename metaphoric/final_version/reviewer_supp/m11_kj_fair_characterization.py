#!/usr/bin/env python3
"""M11: collect KJ-specific significant signatures from predefined upstream tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared import add_common_args, default_config, read_table, write_standard_outputs

MODULE = "m11_kj_fair_characterization"
FDR_COLUMNS = ("q", "q_bh", "q_bh_within_family", "q_fdr", "p_fdr")
UNCORRECTED_COLUMNS = ("p",)
TARGET_FILES = {
    "relation_vector": [
        "relation_vector_rsa_meta_metaphor/relation_vector_group_summary_fdr.tsv",
        "relation_vector_rsa_meta_spatial/relation_vector_group_summary_fdr.tsv",
    ],
    "semantic_edge_model": [
        "stagewise_mechanism/stage3_semantic_edge_condition_models.tsv",
        "stagewise_mechanism/stage3_semantic_edge_shift_models.tsv",
    ],
    "retrieval_rebinding": [
        "stagewise_mechanism/stage1_post_to_retrieval_models.tsv",
        "stagewise_mechanism/stage1_post_to_retrieval_network_models.tsv",
    ],
    "existing_ers": [
        "encoding_retrieval_similarity_meta_metaphor/ers_group_one_sample.tsv",
        "encoding_retrieval_similarity_meta_spatial/ers_group_one_sample.tsv",
        "encoding_retrieval_similarity_meta_metaphor/ers_itemlevel_gee.tsv",
        "encoding_retrieval_similarity_meta_spatial/ers_itemlevel_gee.tsv",
    ],
}


def annotate_evidence_tier(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["evidence_tier"] = ""
    out["significance_column"] = ""
    out["significance_value"] = np.nan
    for col in FDR_COLUMNS:
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce")
            hit = out["evidence_tier"].eq("") & vals.le(0.05)
            out.loc[hit, "evidence_tier"] = "FDR-corrected"
            out.loc[hit, "significance_column"] = col
            out.loc[hit, "significance_value"] = vals.loc[hit]
    for col in UNCORRECTED_COLUMNS:
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce")
            hit = out["evidence_tier"].eq("") & vals.le(0.05)
            out.loc[hit, "evidence_tier"] = "uncorrected-only"
            out.loc[hit, "significance_column"] = col
            out.loc[hit, "significance_value"] = vals.loc[hit]
    return out


def kj_like_rows(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for col in frame.columns:
        values = frame[col].astype(str).str.lower()
        if col.lower() in {
            "condition",
            "condition_label",
            "contrast",
            "term",
            "comparison",
            "effect",
            "model",
            "model_name",
        }:
            mask = mask | values.str.contains("kj", regex=False, na=False)
    return mask


def collect_from_predefined_sources(qc_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_name, relative_paths in TARGET_FILES.items():
        for rel_path in relative_paths:
            path = qc_root / rel_path
            if not path.exists():
                rows.append(
                    {
                        "source_name": source_name,
                        "source_file": str(path),
                        "status": "missing_source",
                    }
                )
                continue
            try:
                frame = read_table(path)
            except Exception as exc:
                rows.append(
                    {
                        "source_name": source_name,
                        "source_file": str(path),
                        "status": f"read_failed: {exc}",
                    }
                )
                continue
            if frame.empty:
                rows.append(
                    {
                        "source_name": source_name,
                        "source_file": str(path),
                        "status": "empty_source",
                    }
                )
                continue
            annotated = annotate_evidence_tier(frame)
            mask = annotated["evidence_tier"].ne("") & kj_like_rows(annotated)
            subset = annotated.loc[mask].copy()
            if subset.empty:
                rows.append(
                    {
                        "source_name": source_name,
                        "source_file": str(path),
                        "status": "no_kj_hits",
                    }
                )
                continue
            for _, row in subset.iterrows():
                rec = {
                    "source_name": source_name,
                    "source_file": str(path),
                    "status": "candidate",
                    "evidence_tier": row.get("evidence_tier"),
                    "significance_column": row.get("significance_column"),
                    "significance_value": row.get("significance_value"),
                    "downgraded_due_to_shared_post": bool(source_name == "retrieval_rebinding"),
                }
                protected = set(rec)
                for col in frame.columns[:40]:
                    value = row.get(col)
                    if np.isscalar(value):
                        out_col = f"source_row_{col}" if col in protected else col
                        rec[out_col] = value
                rows.append(rec)
    return pd.DataFrame(rows)


def build_markdown(summary: pd.DataFrame) -> str:
    n = int((summary["status"] == "candidate").sum()) if (not summary.empty and "status" in summary.columns) else 0
    n_fdr = int(((summary["status"] == "candidate") & (summary.get("evidence_tier") == "FDR-corrected")).sum()) if (not summary.empty and "evidence_tier" in summary.columns) else 0
    n_unc = int(((summary["status"] == "candidate") & (summary.get("evidence_tier") == "uncorrected-only")).sum()) if (not summary.empty and "evidence_tier" in summary.columns) else 0
    n_down = int(((summary["status"] == "candidate") & (summary.get("downgraded_due_to_shared_post") == True)).sum()) if (not summary.empty and "downgraded_due_to_shared_post" in summary.columns) else 0
    lines = [
        "# M11 KJ-Specific Signature",
        "",
        f"Predefined-source scan found {n} candidate KJ/spatial significant rows from upstream QC tables.",
        f"- FDR-corrected candidates: {n_fdr}",
        f"- Uncorrected-only candidates: {n_unc}",
        f"- Downgraded shared-post retrieval candidates: {n_down}",
        "",
        "Interpretation draft: KJ should be treated as an active spatial-relational learning condition rather than a pure null control. FDR-corrected KJ signatures can be discussed as stronger supplementary evidence; uncorrected-only rows should be described only as descriptive leads.",
        "",
        "Retrieval-rebinding rows are flagged as downgraded when they come from raw Stage1 sources, because earlier audits showed shared-post coupling risk. The summary is constrained to predefined upstream sources and only counts rows with explicit KJ labels.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--qc-root", type=Path, default=None)
    args = parser.parse_args()
    cfg = default_config(args)
    qc_root = args.qc_root or cfg.paper_output_root / "qc"
    summary = collect_from_predefined_sources(qc_root)
    outputs = {"kj_significant_results.tsv": summary, "kj_specific_signature.md": build_markdown(summary)}
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
