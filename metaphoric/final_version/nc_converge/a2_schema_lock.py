#!/usr/bin/env python3
"""A2: lock condition_item_id schema and run7 subject intersection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, audit_condition_items, default_config, discover_tables, subject_intersection, write_outputs, RUN_MAP

MODULE = "a2_schema_lock"
DEFAULT_PATTERNS = [
    "stagewise_mechanism/**/*.tsv",
    "**/stagewise_mechanism/**/*.tsv",
    "*memory*/*.tsv",
    "*retrieval*/*.tsv",
    "*step5c*/*.tsv",
    "*edge*/*.tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None, help="Explicit upstream tables to audit.")
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS, help="Glob patterns under paper output root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.paper_output_root, args.patterns)
    if not paths and not args.allow_empty:
        raise FileNotFoundError(f"No upstream tables found under {cfg.paper_output_root}; use --input or --allow-empty.")
    audit = audit_condition_items(paths)
    intersection = subject_intersection(paths)
    conflicts = audit[audit.get("status", pd.Series(dtype=str)).astype(str).eq("duplicate_condition_item_id")] if not audit.empty else pd.DataFrame()
    if not conflicts.empty:
        audit["fatal"] = audit["path"].isin(conflicts["path"])
    log = {
        "module": MODULE,
        "run_map": RUN_MAP,
        "n_tables": len(paths),
        "n_conflict_tables": int(len(conflicts)),
        "status": "ok" if conflicts.empty else "duplicate_condition_item_id",
    }
    write_outputs(cfg, MODULE, {
        "id_audit.tsv": audit,
        "subject_intersection.tsv": intersection,
        "schema_lock.log": json.dumps(log, indent=2, ensure_ascii=False),
    })
    if not conflicts.empty:
        raise ValueError("condition_item_id duplicates found; downstream modules should not run.")


if __name__ == "__main__":
    main()
