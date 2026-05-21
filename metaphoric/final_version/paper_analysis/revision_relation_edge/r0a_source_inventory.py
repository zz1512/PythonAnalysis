#!/usr/bin/env python3
"""Audit source-level inputs for relation-edge revision analyses.

This script does not compute statistics. It records which variables can be
rebuilt from the original source tree and which current revision metrics still
depend on locked upstream neural summary tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from shared_revision import DEFAULT_BASE_DIR, DEFAULT_PAPER_OUTPUTS, out_path, read_table, write_json, write_text, write_tsv

MODULE = "r0a_source_inventory"


def _safe_cols(path: Path, *, sep: str | None = None) -> tuple[int, list[str], str]:
    try:
        if path.suffix.lower() in {".tsv", ".txt"}:
            df = pd.read_csv(path, sep="\t", nrows=100, low_memory=False)
            n = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=100, low_memory=False)
            n = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        else:
            return 0, [], "unsupported"
        return max(n, 0), list(df.columns), "ok"
    except Exception as exc:
        return 0, [], f"failed: {exc}"


def audit_events() -> pd.DataFrame:
    rows = []
    for path in sorted((DEFAULT_BASE_DIR / "data_events").glob("sub-*/sub-*_run-*_*events.*")):
        match = re.search(r"(sub-\d+)_run-(\d+)_events\.(csv|tsv)$", path.name)
        if not match:
            continue
        subject, run, suffix = match.group(1), int(match.group(2)), match.group(3)
        n, cols, status = _safe_cols(path)
        rows.append(
            {
                "source_family": "raw_events",
                "subject": subject,
                "run": run,
                "format": suffix,
                "path": str(path),
                "n_rows": n,
                "n_columns": len(cols),
                "columns": "|".join(cols),
                "has_sentence3_rt": "sentence3_RT" in cols,
                "has_sentence3_resp": "sentence3_RESP" in cols,
                "has_sentence4_rt": "sentence4_RT" in cols,
                "has_sentence4_resp": "sentence4_RESP" in cols,
                "has_memory": "memory" in cols,
                "has_action_time": "action_time" in cols or "eword7_RT" in cols or "word7_RT" in cols,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def audit_pattern_root() -> pd.DataFrame:
    rows = []
    root = DEFAULT_BASE_DIR / "pattern_root"
    for path in sorted(root.glob("sub-*/*_metadata.tsv")):
        subject = path.parent.name
        n, cols, status = _safe_cols(path)
        rows.append(
            {
                "source_family": "pattern_metadata",
                "subject": subject,
                "file": path.name,
                "path": str(path),
                "n_rows": n,
                "n_columns": len(cols),
                "columns": "|".join(cols),
                "has_pair_id": "pair_id" in cols,
                "has_beta_path": "beta_path" in cols,
                "has_memory": "memory" in cols,
                "status": status,
            }
        )
    for path in sorted(root.glob("sub-*/*.nii.gz")):
        rows.append(
            {
                "source_family": "pattern_nifti",
                "subject": path.parent.name,
                "file": path.name,
                "path": str(path),
                "n_rows": pd.NA,
                "n_columns": pd.NA,
                "columns": "",
                "has_pair_id": pd.NA,
                "has_beta_path": pd.NA,
                "has_memory": pd.NA,
                "status": "present",
            }
        )
    return pd.DataFrame(rows)


def audit_materials() -> pd.DataFrame:
    rows = []
    root = DEFAULT_BASE_DIR / "materials_detail"
    for path in sorted(root.glob("*.xlsx")):
        try:
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                try:
                    df = pd.read_excel(path, sheet_name=sheet, nrows=20)
                    rows.append(
                        {
                            "source_family": "materials_excel",
                            "file": path.name,
                            "sheet": sheet,
                            "path": str(path),
                            "n_preview_rows": len(df),
                            "n_columns": len(df.columns),
                            "columns": "|".join(map(str, df.columns)),
                            "has_novelty": any("新颖" in str(c) or "novel" in str(c).lower() for c in df.columns),
                            "has_familiarity": any("熟悉" in str(c) or "familiar" in str(c).lower() for c in df.columns),
                            "has_comprehensibility": any("可理解" in str(c) or "compreh" in str(c).lower() for c in df.columns),
                            "has_difficulty": any("难度" in str(c) or "difficult" in str(c).lower() for c in df.columns),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    rows.append({"source_family": "materials_excel", "file": path.name, "sheet": sheet, "path": str(path), "status": f"failed: {exc}"})
        except Exception as exc:
            rows.append({"source_family": "materials_excel", "file": path.name, "path": str(path), "status": f"failed: {exc}"})
    return pd.DataFrame(rows)


def audit_roi_and_embeddings() -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_path = DEFAULT_BASE_DIR / "roi_library" / "manifest.tsv"
    if roi_path.exists():
        roi = read_table(roi_path)
        roi_summary = roi.groupby(["roi_set"], dropna=False).agg(
            n_roi=("roi_name", "count"),
            n_include_rsa=("include_in_rsa", "sum"),
            n_include_mvpa=("include_in_mvpa", "sum"),
        ).reset_index()
        roi_summary["source_path"] = str(roi_path)
    else:
        roi_summary = pd.DataFrame([{"source_path": str(roi_path), "status": "missing"}])

    emb_path = DEFAULT_BASE_DIR / "stimulus_embeddings" / "stimulus_embeddings_bert.tsv"
    if emb_path.exists():
        n, cols, status = _safe_cols(emb_path)
        emb = pd.DataFrame([{"source_path": str(emb_path), "n_rows": n, "n_columns": len(cols), "columns": "|".join(cols), "status": status}])
    else:
        emb = pd.DataFrame([{"source_path": str(emb_path), "status": "missing"}])
    return roi_summary, emb


def variable_provenance() -> pd.DataFrame:
    rows = [
        ("memory_ord/memory_strict/memory_lenient", "source", "data_events run7 raw CSV/TSV and lss trial_info; currently also present in nc_converge master"),
        ("run3_understand/run3_rt_z", "source", "data_events/sub-XX/sub-XX_run-3_events.csv: sentence3_RESP/sentence3_RT"),
        ("run4_like/run4_rt_z", "source", "data_events/sub-XX/sub-XX_run-4_events.csv: sentence4_RESP/sentence4_RT"),
        ("novelty/familiarity/comprehensibility/difficulty", "source", "materials_detail Excel ratings; currently available through table_stimulus_control_from_materials.tsv"),
        ("word frequency/length/valence/arousal", "source_or_material_summary", "material/lexical summary columns in item_mechanism_table.tsv"),
        ("pre/post/retrieval pair similarity", "locked_upstream_neural_summary", "paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv, ultimately derived from pattern_root NIfTI and ROI masks"),
        ("post_separation/retrieval_rebound", "locked_upstream_neural_summary", "computed from upstream pair similarity fields in nc_converge master"),
        ("semantic/hpc-spatial network composites", "locked_upstream_neural_summary", "computed by nc_converge from ROI-level neural summaries"),
        ("MVPA stage-state summary", "locked_upstream_neural_summary", "paper_outputs MVPA summary tables"),
    ]
    return pd.DataFrame(rows, columns=["variable", "provenance_tier", "source_detail"])


def make_review(events: pd.DataFrame, pattern: pd.DataFrame, provenance: pd.DataFrame) -> str:
    run_csv = events[(events["format"].eq("csv")) & (events["run"].isin([3, 4]))]
    run3_ok = int(run_csv["has_sentence3_rt"].sum())
    run4_ok = int(run_csv["has_sentence4_rt"].sum())
    subjects = int(events["subject"].nunique()) if not events.empty else 0
    pattern_subjects = int(pattern.loc[pattern["source_family"].eq("pattern_metadata"), "subject"].nunique()) if not pattern.empty else 0
    locked = provenance[provenance["provenance_tier"].str.contains("locked", na=False)]["variable"].tolist()
    text = f"""# Source Inventory Review

## Summary

- Source root: `{DEFAULT_BASE_DIR}`
- Event subjects detected: {subjects}
- Pattern-root subjects detected: {pattern_subjects}
- Run3 raw CSV files with `sentence3_RT`: {run3_ok}
- Run4 raw CSV files with `sentence4_RT`: {run4_ok}

## Interpretation

Run3/run4 behavior is available from original E-Prime CSV files and should be merged into the revision master table. The current r0 script now reads the source-derived `learning_behavior_item.tsv`, rebuilding it from raw CSVs if missing.

The following neural variables are still read from locked upstream neural summary tables, not recomputed inside the revision scripts:

{chr(10).join(f'- `{v}`' for v in locked)}

This is acceptable for fast manuscript revision only if those upstream tables are treated as frozen outputs from source-level RSA/MVPA pipelines. For a strict source-only rerun, add a separate neural recomputation step from `pattern_root/*.nii.gz` plus `roi_library` masks before r1-r5.
"""
    return text


def main() -> None:
    events = audit_events()
    pattern = audit_pattern_root()
    materials = audit_materials()
    roi, emb = audit_roi_and_embeddings()
    provenance = variable_provenance()

    write_tsv(events, out_path(MODULE, "source_event_schema.tsv"))
    write_tsv(pattern, out_path(MODULE, "source_pattern_schema.tsv"))
    write_tsv(materials, out_path(MODULE, "source_material_schema.tsv"))
    write_tsv(roi, out_path(MODULE, "source_roi_schema.tsv"))
    write_tsv(emb, out_path(MODULE, "source_embedding_schema.tsv"))
    write_tsv(provenance, out_path(MODULE, "revision_variable_provenance.tsv"))
    write_text(make_review(events, pattern, provenance), out_path(MODULE, "source_inventory_review.md"))
    write_json(
        {
            "status": "ok",
            "source_root": str(DEFAULT_BASE_DIR),
            "n_event_files": int(len(events)),
            "n_pattern_entries": int(len(pattern)),
            "n_material_sheets": int(len(materials)),
        },
        out_path(MODULE, "source_inventory_manifest.json"),
    )
    print(f"Wrote source inventory to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
