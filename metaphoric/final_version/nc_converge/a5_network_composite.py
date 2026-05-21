#!/usr/bin/env python3
"""A5: build network composite summaries for main trajectory metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, default_config, discover_tables, make_network_composite, read_table, value_column, write_outputs

MODULE = "a5_network_composite"
PATTERNS = [
    "**/*step5c*.tsv", "**/*edge_specificity*.tsv", "**/*retrieval*rebind*.tsv", "**/*pair_similarity*.tsv",
    "**/stagewise_mechanism/*.tsv", "**/relation_step5c*.tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["stage", "phase", "condition", "network", "metric"] if c in comp.columns]
    if not group_cols:
        return comp
    summary = comp.groupby(group_cols, dropna=False, as_index=False).agg(
        n=("metric_value", "count"),
        mean=("metric_value", "mean"),
        sd=("metric_value", "std"),
        n_items=("condition_item_id", "nunique") if "condition_item_id" in comp.columns else ("metric_value", "count"),
    )
    return summary


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.paper_output_root, PATTERNS)
    comps = []
    manifest_rows = []
    for path in paths:
        try:
            frame = read_table(path)
        except Exception as exc:
            manifest_rows.append({"path": str(path), "status": f"read_failed: {exc}"})
            continue
        if frame.empty:
            manifest_rows.append({"path": str(path), "status": "empty"})
            continue
        metric_name = frame["metric"].iloc[0] if "metric" in frame.columns and frame["metric"].notna().any() else path.stem
        val = value_column(frame)
        comp = make_network_composite(frame.assign(metric=metric_name), value_col=val, metric_name=str(metric_name)) if val else pd.DataFrame()
        if comp.empty or "metric_value" not in comp.columns or comp["metric_value"].notna().sum() == 0:
            manifest_rows.append({"path": str(path), "status": "no_usable_network_composite", "value_column": val or ""})
            continue
        comp["source_path"] = str(path)
        comps.append(comp)
        manifest_rows.append({"path": str(path), "status": "ok", "value_column": val, "n_composite_rows": len(comp)})
    raw = pd.concat(comps, ignore_index=True, sort=False) if comps else pd.DataFrame()
    if raw.empty and not args.allow_empty:
        raise FileNotFoundError("No network composite candidate tables found; use --input or --allow-empty.")
    summary = summarize(raw)
    write_outputs(cfg, MODULE, {
        "network_composite_summary.tsv": summary,
        "network_trajectory_long.tsv": raw,
        "network_composite_input_manifest.tsv": pd.DataFrame(manifest_rows),
    })


if __name__ == "__main__":
    main()
