"""
export_gppi_summary.py

Purpose
- Post-hoc exporter for S2 gPPI outputs.
- It reads the per-seed SI tables already written by `gPPI_analysis.py`,
  then consolidates them into:
  - `paper_outputs/tables_main/table_gppi_summary_main.tsv`
  - `paper_outputs/tables_si/table_gppi_summary_full.tsv`

Why a separate script
- Do not touch or restart the currently running gPPI job.
- Keep the heavy first-level/second-level computation unchanged.
- Only add a light-weight export layer that normalizes final paper tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402


def _default_paper_root() -> Path:
    try:
        from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

        return Path(BASE_DIR) / "paper_outputs"
    except Exception:
        return Path(os.environ.get("PYTHON_METAPHOR_ROOT", ".")).resolve() / "paper_outputs"


def _candidate_tables(tables_si: Path) -> list[Path]:
    paths = sorted(tables_si.glob("table_gppi_summary_*.tsv"))
    excluded = {
        "table_gppi_summary_main.tsv",
        "table_gppi_summary_full.tsv",
    }
    return [path for path in paths if path.name not in excluded]


def _read_seed_tables(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = read_table(path).copy()
        if frame.empty:
            continue
        frame["source_table"] = str(path)
        if "seed" not in frame.columns and "seed_roi" in frame.columns:
            frame["seed"] = frame["seed_roi"]
        if "roi_set" not in frame.columns:
            frame["roi_set"] = pd.NA
        if "deconvolution" not in frame.columns:
            frame["deconvolution"] = pd.NA
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    numeric_cols = ["n", "mean_post", "mean_pre", "t", "p", "cohens_dz", "mean_diff"]
    for col in numeric_cols:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")
    return full


def _build_main_table(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return pd.DataFrame()
    scope_targets = {"__scope_mean__", "__cross_sets__"}
    scope_order = {
        "within_literature": 0,
        "within_literature_spatial": 1,
        "cross_sets": 2,
    }
    main = full.copy()
    main = main[main["test_type"].astype(str).eq("post_vs_pre")]
    main = main[main["ppi_condition"].astype(str).eq("yy_minus_kj")]
    main = main[main["target_scope"].astype(str).isin(scope_order)]
    main = main[main["target_roi"].astype(str).isin(scope_targets)]
    if main.empty:
        return main
    main = main.assign(
        significant_p_uncorrected=main["p"].lt(0.05),
        scope_rank=main["target_scope"].map(scope_order).fillna(999).astype(int),
    )
    ordered = [
        "seed",
        "roi_set",
        "deconvolution",
        "target_scope",
        "ppi_condition",
        "test_type",
        "n",
        "mean_post",
        "mean_pre",
        "mean_diff",
        "t",
        "p",
        "cohens_dz",
        "significant_p_uncorrected",
        "source_table",
    ]
    present = [col for col in ordered if col in main.columns]
    main = main.sort_values(["seed", "scope_rank", "p"], ascending=[True, True, True]).reset_index(drop=True)
    return main[present]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export consolidated paper tables from finished gPPI outputs.")
    parser.add_argument("paper_output_root", nargs="?", type=Path, default=None)
    args = parser.parse_args()

    paper_root = ensure_dir(args.paper_output_root or _default_paper_root())
    tables_main = ensure_dir(paper_root / "tables_main")
    tables_si = ensure_dir(paper_root / "tables_si")
    qc_root = ensure_dir(paper_root / "qc")

    source_tables = _candidate_tables(tables_si)
    full_df = _read_seed_tables(source_tables)
    main_df = _build_main_table(full_df)

    write_table(full_df, tables_si / "table_gppi_summary_full.tsv")
    write_table(main_df, tables_main / "table_gppi_summary_main.tsv")
    save_json(
        {
            "paper_output_root": str(paper_root),
            "n_source_tables": int(len(source_tables)),
            "source_tables": [str(path) for path in source_tables],
            "n_full_rows": int(len(full_df)),
            "n_main_rows": int(len(main_df)),
            "full_table": str(tables_si / "table_gppi_summary_full.tsv"),
            "main_table": str(tables_main / "table_gppi_summary_main.tsv"),
        },
        qc_root / "gppi_summary_export_manifest.json",
    )


if __name__ == "__main__":
    main()
