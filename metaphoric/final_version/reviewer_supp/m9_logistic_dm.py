#!/usr/bin/env python3
"""M9: binarized memory logistic GLM sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared import add_common_args, bh_fdr, default_config, existing_table, fit_formula, write_standard_outputs, zscore, ensure_condition_item_id

MODULE = "m9_logistic_dm"
METRICS = [
    "post_edge_specificity",
    "retrieval_rebinding",
    "retrieval_pair_similarity",
    "trained_edge_drop",
    "hpc_spatial_rebinding",
    "hpc_spatial_post_separation",
    "hpc_spatial_retrieval_pair_similarity",
    "semantic_learning_edge_trace",
    "hpc_spatial_learning_edge_trace",
]


def pivot_long_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metric_col = None
    value_col = None
    for candidate_metric, candidate_value in [
        ("metric_name", "neural_metric"),
        ("component_name", "component_metric"),
    ]:
        if {candidate_metric, candidate_value}.issubset(frame.columns):
            metric_col = candidate_metric
            value_col = candidate_value
            break
    if metric_col is None or value_col is None:
        return frame
    id_cols = [
        col
        for col in frame.columns
        if col
        not in {
            metric_col,
            value_col,
            "neural_metric_z",
            "component_metric_z",
            "metric_family",
            "network",
            "remembered_strict",
            "memory_strict",
            "memory_lenient",
        }
    ]
    wide = (
        frame.pivot_table(
            index=id_cols,
            columns=metric_col,
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    aliases = {
        "hpc_spatial_rebinding": "retrieval_rebinding",
        "hpc_spatial_retrieval_pair_similarity": "retrieval_pair_similarity",
        "hpc_spatial_post_separation": "post_edge_specificity",
    }
    for source, target in aliases.items():
        if source in wide.columns and target not in wide.columns:
            wide[target] = wide[source]
    return wide


def load_item_table(path: Path) -> pd.DataFrame:
    frame = ensure_condition_item_id(pivot_long_metrics(existing_table(path)))
    if "memory" not in frame.columns:
        raise ValueError(f"{path} must contain memory column")
    frame = frame.copy()
    frame["memory_score"] = pd.to_numeric(frame["memory"], errors="coerce")
    frame["memory_strict"] = np.where(frame["memory_score"].eq(1.0), 1, np.where(frame["memory_score"].isin([0.0, 0.5]), 0, np.nan))
    frame["memory_lenient"] = np.where(frame["memory_score"].isin([1.0, 0.5]), 1, np.where(frame["memory_score"].eq(0.0), 0, np.nan))
    return frame


def fit_models(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows = []
    available = [m for m in METRICS if m in frame.columns]
    for metric in available:
        work = frame.dropna(subset=[outcome, metric, "condition"]).copy()
        if work.empty:
            continue
        work[f"{metric}_z"] = zscore(work[metric])
        group_cols = ["roi_set", "roi"] if {"roi_set", "roi"}.issubset(work.columns) else ["__all__"]
        if group_cols == ["__all__"]:
            work["__all__"] = "all"
        for key, sub in work.groupby(group_cols, dropna=False):
            model = fit_formula(
                sub,
                f"{outcome} ~ {metric}_z * C(condition, Treatment(reference='kj'))",
                family="binomial",
                group_col="subject" if "subject" in sub.columns else None,
                item_col="condition_item_id" if "condition_item_id" in sub.columns else None,
                prefer_mixed=True,
            )
            if not isinstance(key, tuple):
                key = (key,)
            for col, val in zip(group_cols, key):
                model.insert(0, col, val)
            model.insert(0, "metric", metric)
            model.insert(0, "outcome", outcome)
            rows.append(model)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty and "p" in out:
        ok = out["status"].eq("ok")
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--item-table", type=Path, default=None, help="Default: paper_outputs/qc/stagewise_mechanism/stage4_subsequent_memory_item.tsv")
    args = parser.parse_args()
    cfg = default_config(args)
    item_table = args.item_table or cfg.paper_output_root / "qc" / "stagewise_mechanism" / "stage4_subsequent_memory_item.tsv"
    item = load_item_table(item_table)
    strict = fit_models(item, "memory_strict")
    lenient = fit_models(item, "memory_lenient")
    outputs = {"logistic_dm_input_item.tsv": item, "logistic_dm_strict.tsv": strict, "logistic_dm_lenient.tsv": lenient}
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
