#!/usr/bin/env python3
"""B3: predefined semantic/metaphor and hpc-spatial network coupling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, add_network_column, build_condition_item_id, default_config, discover_tables, fit_formula, read_table, write_outputs, zscore, q_for_ok_rows

MODULE = "b3_repr_coupling"
PATTERNS = [
    "**/network_trajectory_long.tsv",
    "**/*item_mechanism*.tsv",
    "**/*rebound*.tsv",
    "**/*reactivation*.tsv",
    "**/*post*separation*.tsv",
]
ITEM_METRICS = [
    "learning_specificity",
    "learning_self_similarity",
    "post_pair_similarity",
    "trained_edge_drop",
    "retrieval_pair_similarity",
    "retrieval_minus_post_pair_similarity",
    "retrieval_minus_pre_pair_similarity",
]


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


def pivot_networks(frame: pd.DataFrame) -> pd.DataFrame:
    out = build_condition_item_id(frame)
    if "network" not in out.columns or "metric" not in out.columns:
        return pd.DataFrame()
    value = "metric_value" if "metric_value" in out.columns else ("mean" if "mean" in out.columns else None)
    if value is None:
        return pd.DataFrame()
    out[value] = pd.to_numeric(out[value], errors="coerce")
    keys = [c for c in ["subject", "condition_item_id", "condition", "metric"] if c in out.columns]
    wide = out.pivot_table(index=keys, columns="network", values=value, aggfunc="mean").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    for col in ["semantic", "hpc_spatial"]:
        if col in wide.columns:
            wide[f"{col}_z"] = zscore(wide[col])
    return wide


def item_metric_wide(frame: pd.DataFrame) -> pd.DataFrame:
    out = build_condition_item_id(frame)
    if "network" in out.columns:
        mapped = add_network_column(out.drop(columns=["network"], errors="ignore"))
        if "network" in mapped.columns:
            out["network"] = out["network"].where(out["network"].notna(), mapped["network"])
    else:
        out = add_network_column(out)
    if not {"subject", "condition_item_id", "condition", "network"}.issubset(out.columns):
        return pd.DataFrame()
    metrics = [col for col in ITEM_METRICS if col in out.columns]
    if not metrics:
        return pd.DataFrame()
    id_cols = ["subject", "condition_item_id", "condition", "network"]
    long = out[id_cols + metrics].melt(id_vars=id_cols, value_vars=metrics, var_name="metric", value_name="metric_value")
    long["metric_value"] = pd.to_numeric(long["metric_value"], errors="coerce")
    long = long.dropna(subset=["metric_value", "network"])
    if long.empty:
        return pd.DataFrame()
    long = long.groupby(id_cols + ["metric"], dropna=False, as_index=False).agg(metric_value=("metric_value", "mean"))
    wide = long.pivot_table(index=["subject", "condition_item_id", "condition"], columns=["network", "metric"], values="metric_value", aggfunc="mean").reset_index()
    flat_cols = []
    for col in wide.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(part) for part in col if str(part)))
        else:
            flat_cols.append(str(col))
    wide.columns = flat_cols
    for col in wide.columns:
        if col not in {"subject", "condition_item_id", "condition"}:
            wide[f"{col}_z"] = zscore(wide[col])
    return wide


def coupling_class(metric_label: object) -> str:
    text = str(metric_label).lower()
    if "learning" in text and ("post" in text or "separation" in text):
        return "learning_semantic_to_post_hpc_spatial"
    if "retrieval" in text and ("rebound" in text or "retrieval" in text):
        return "retrieval_semantic_to_retrieval_hpc_spatial"
    if "post" in text and "separation" in text:
        return "post_semantic_to_post_hpc_spatial"
    return "predefined_semantic_hpc_spatial"


def fit_planned_item_couplings(frame: pd.DataFrame) -> pd.DataFrame:
    wide = item_metric_wide(frame)
    if wide.empty:
        return pd.DataFrame()
    planned = [
        (
            "learning_semantic_to_post_hpc_spatial",
            "semantic_learning_specificity_z",
            "hpc_spatial_trained_edge_drop_z",
        ),
        (
            "retrieval_semantic_to_retrieval_hpc_spatial",
            "semantic_retrieval_minus_post_pair_similarity_z",
            "hpc_spatial_retrieval_minus_post_pair_similarity_z",
        ),
    ]
    rows = []
    for label, predictor, outcome in planned:
        needed = {predictor, outcome, "condition", "subject", "condition_item_id"}
        if not needed.issubset(wide.columns):
            rows.append(pd.DataFrame([{"coupling": label, "term": "__model__", "status": f"missing_columns: {sorted(needed - set(wide.columns))}"}]))
            continue
        sub = wide.dropna(subset=[predictor, outcome, "condition"]).copy()
        if len(sub) < 10:
            rows.append(pd.DataFrame([{"coupling": label, "term": "__model__", "status": "too_few_rows", "n_obs": len(sub)}]))
            continue
        res = fit_formula(sub, f"{outcome} ~ {predictor} * condition", family="gaussian")
        res.insert(0, "coupling", label)
        res.insert(1, "coupling_class", label)
        res.insert(2, "predictor_network", "semantic")
        res.insert(3, "outcome_network", "hpc_spatial")
        res.insert(4, "predictor_metric", predictor)
        res.insert(5, "outcome_metric", outcome)
        rows.append(res)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or sorted(set(discover_tables(cfg.output_root, PATTERNS) + discover_tables(cfg.paper_output_root, PATTERNS)))
    raw = load(paths)
    if raw.empty and not args.allow_empty:
        raise FileNotFoundError("No representational coupling inputs found; run A5 or pass --input.")
    rows = []
    planned = fit_planned_item_couplings(raw)
    if not planned.empty:
        rows.append(planned)
    wide = pivot_networks(raw)
    if {"semantic_z", "hpc_spatial_z", "subject", "condition_item_id"}.issubset(wide.columns):
        if "condition" not in wide.columns:
            wide["condition"] = "all"
        for label, sub in wide.groupby("metric", dropna=False) if "metric" in wide.columns else [("all_metrics", wide)]:
            if len(sub.dropna(subset=["semantic_z", "hpc_spatial_z"])) < 10:
                rows.append(pd.DataFrame([{"coupling": label, "term": "__model__", "status": "too_few_rows", "n_obs": len(sub)}]))
                continue
            res = fit_formula(sub.dropna(subset=["semantic_z", "hpc_spatial_z"]), "hpc_spatial_z ~ semantic_z * condition", family="gaussian")
            res.insert(0, "coupling", label)
            res.insert(1, "coupling_class", coupling_class(label))
            res.insert(2, "predictor_network", "semantic")
            res.insert(3, "outcome_network", "hpc_spatial")
            rows.append(res)
    else:
        if not rows:
            rows.append(pd.DataFrame([{"coupling": "all", "term": "__model__", "status": f"missing_columns: {sorted({'semantic_z','hpc_spatial_z','subject','condition_item_id'} - set(wide.columns))}"}]))
    table = q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))
    write_outputs(cfg, MODULE, {"repr_coupling.tsv": table, "repr_coupling_input_manifest.tsv": pd.DataFrame({"path": [str(p) for p in paths]})})


if __name__ == "__main__":
    main()
