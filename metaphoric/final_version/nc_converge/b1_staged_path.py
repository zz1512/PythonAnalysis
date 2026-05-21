#!/usr/bin/env python3
"""B1: network-level staged trajectory characterization without mediation claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, build_condition_item_id, default_config, discover_tables, first_existing_column, fit_formula, markdown_table, read_table, write_outputs, zscore, q_for_ok_rows

MODULE = "b1_staged_path"
PATTERNS = ["**/network_trajectory_long.tsv", "**/network_composite_summary.tsv", "**/*memory*component*.tsv", "**/*rebind*.tsv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = read_table(path)
        except Exception:
            continue
        frame["source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def wide_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = build_condition_item_id(frame)
    value = first_existing_column(out, ["metric_value", "mean", "estimate", "value"])
    if value is None:
        return pd.DataFrame()
    out[value] = pd.to_numeric(out[value], errors="coerce")
    if "metric" not in out.columns:
        out["metric"] = out["source_path"].astype(str).str.extract(r"([^/]+)\.tsv$")[0]
    keys = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in out.columns]
    if not {"subject", "condition_item_id"}.issubset(keys):
        return out
    wide = out.pivot_table(index=keys, columns="metric", values=value, aggfunc="mean").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    for col in wide.columns:
        if col not in keys:
            wide[f"{col}_z"] = zscore(wide[col])
    return wide


def find_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        for low, original in lower.items():
            if cand.lower() in low:
                return original
    return None


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.output_root, PATTERNS) or discover_tables(cfg.paper_output_root, PATTERNS)
    raw = load(paths)
    if raw.empty and not args.allow_empty:
        raise FileNotFoundError("No staged trajectory inputs found; run A5/A4 first or pass --input.")
    wide = wide_metrics(raw)
    trajectories_to_fit = [
        ("learning_to_post", ["learning", "geometry"], ["post", "separation"]),
        ("post_to_retrieval", ["post", "separation"], ["retrieval", "rebound"]),
        ("retrieval_to_memory", ["retrieval", "rebound"], ["memory", "prop"]),
    ]
    rows = []
    for label, pred_keys, out_keys in trajectories_to_fit:
        pred = find_col(wide, pred_keys)
        outcome = find_col(wide, out_keys)
        if pred and outcome and {"subject", "condition_item_id"}.issubset(wide.columns):
            data = wide.dropna(subset=[pred, outcome]).copy()
            if "condition" not in data.columns:
                data["condition"] = "all"
            res = fit_formula(data, f"{outcome} ~ {pred} * condition", family="binomial" if "memory" in outcome.lower() else "gaussian")
            res.insert(0, "trajectory", label)
            res.insert(1, "predictor", pred)
            res.insert(2, "outcome", outcome)
            rows.append(res)
        else:
            rows.append(pd.DataFrame([{"trajectory": label, "term": "__model__", "status": f"missing_predictor_or_outcome: predictor={pred}, outcome={outcome}"}]))
    table = q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))
    md = "# Network-level Staged Trajectory Characterization\n\n"
    md += "These are direct stage-to-stage associations used to describe trajectory consistency; no mediation or causal-path inference is claimed.\n\n"
    md += markdown_table(table[[c for c in ["trajectory", "predictor", "outcome", "term", "estimate", "p", "q_bh", "status"] if c in table.columns]])
    write_outputs(cfg, MODULE, {"stage5_network_trajectory_table.tsv": table, "staged_trajectory.md": md})


if __name__ == "__main__":
    main()
