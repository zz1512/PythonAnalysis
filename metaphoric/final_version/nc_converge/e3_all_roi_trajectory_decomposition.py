#!/usr/bin/env python3
"""Append 518 E3: all-ROI trajectory decomposition.

This script reads the ROI-level voxel trajectory table produced by D2 and
decomposes the post-to-retrieval displacement into:

1. a signed return-to-pre component along the post->pre axis;
2. an orthogonal/new-state component that cannot be explained by simple
   return-to-pre.

The goal is to test whether retrieval rebound is better described as return,
continued differentiation, or task-driven new-state reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    add_network_column,
    build_condition_item_id,
    default_config,
    fit_formula,
    q_for_ok_rows,
    read_table,
    write_outputs,
)

MODULE = "e3_all_roi_trajectory_decomposition"
DEFAULT_TRAJECTORY_ITEM = Path("qc/nc_converge/d2_voxel_trajectory_alignment/trajectory_cosines_item.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
BASE_OUTCOMES = [
    "return_to_pre_cos",
    "continued_diff_cos",
    "pre_retrieval_axis_cos",
    "differentiation_axis_cos",
]
DECOMP_OUTCOMES = [
    "return_component_signed",
    "return_component_abs",
    "return_fraction_signed",
    "return_fraction_abs",
    "new_state_component_norm",
    "new_state_fraction",
    "return_new_log_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--trajectory-item", type=Path, default=None)
    return parser.parse_args()


def _usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10
    ]


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_network_column(build_condition_item_id(frame))
    if "status" in out.columns:
        out = out[out["status"].astype(str).eq("ok")].copy()
    for col in [
        *BASE_OUTCOMES,
        "pre_recovery_projection",
        "v_retrieval_norm",
        "v_post_norm",
        "memory_strict",
        *COVARIATES,
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if not {"pre_recovery_projection", "v_retrieval_norm"}.issubset(out.columns):
        return out
    projection = out["pre_recovery_projection"].astype(float)
    retrieval_norm = out["v_retrieval_norm"].astype(float)
    safe_norm = retrieval_norm.where(retrieval_norm > 0, np.nan)
    out["return_component_signed"] = projection
    out["return_component_abs"] = projection.abs()
    out["return_fraction_signed"] = projection / safe_norm
    out["return_fraction_abs"] = projection.abs() / safe_norm
    orth_sq = np.maximum(np.square(retrieval_norm) - np.square(projection), 0.0)
    out["new_state_component_norm"] = np.sqrt(orth_sq)
    out["new_state_fraction"] = out["new_state_component_norm"] / safe_norm
    # Raw ratios are useful for QC but unstable when the orthogonal component
    # is near zero. Use the log ratio for modeling and interpretation.
    out["return_new_ratio"] = out["return_component_abs"] / (out["new_state_component_norm"] + 1e-8)
    out["return_new_log_ratio"] = np.log(out["return_component_abs"] + 1e-8) - np.log(
        out["new_state_component_norm"] + 1e-8
    )
    return out


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame([{"status": "empty_input"}])
    rows = []
    for keys, sub in data.groupby(["network", "roi", "condition"], dropna=False, sort=False):
        rec = dict(zip(["network", "roi", "condition"], keys))
        rec["n"] = int(len(sub))
        rec["n_subjects"] = int(sub["subject"].nunique()) if "subject" in sub.columns else np.nan
        rec["n_items"] = int(sub["condition_item_id"].nunique()) if "condition_item_id" in sub.columns else np.nan
        for col in BASE_OUTCOMES + DECOMP_OUTCOMES:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                rec[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                rec[f"{col}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_roi_models(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    outcomes = [c for c in BASE_OUTCOMES + DECOMP_OUTCOMES if c in data.columns]
    for (network, roi), sub in data.groupby(["network", "roi"], dropna=False, sort=False):
        covs = _usable_columns(sub, COVARIATES)
        cov_expr = " + " + " + ".join(covs) if covs else ""
        has_memory = "memory_strict" in sub.columns and sub["memory_strict"].notna().any()
        for outcome in outcomes:
            keep = [outcome, *covs]
            if has_memory:
                keep.append("memory_strict")
            model_data = sub.dropna(subset=keep).copy()
            if len(model_data) < 30:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "network": network,
                                "roi": roi,
                                "outcome": outcome,
                                "mechanism_model": "roi_trajectory_decomposition",
                                "term": "__model__",
                                "status": "too_few_rows",
                                "n_obs": len(model_data),
                            }
                        ]
                    )
                )
                continue
            interaction = " * memory_strict" if has_memory else ""
            formula = f"{outcome} ~ C(condition, Treatment('kj')){interaction}{cov_expr}"
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "roi", roi)
            res.insert(2, "outcome", outcome)
            res.insert(3, "mechanism_model", "roi_trajectory_decomposition")
            rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def focal_terms(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty or "term" not in models.columns:
        return pd.DataFrame()
    terms = {
        "C(condition, Treatment('kj'))[T.yy]",
        "memory_strict",
        "C(condition, Treatment('kj'))[T.yy]:memory_strict",
    }
    return models[models["term"].isin(terms)].copy()


def aggregate_network(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in ["subject", "network", "condition", "condition_item_id"] if c in data.columns]
    cols = [c for c in ["memory_strict", *COVARIATES, *BASE_OUTCOMES, *DECOMP_OUTCOMES] if c in data.columns]
    return data[keys + cols].groupby(keys, dropna=False, as_index=False).mean()


def fit_network_models(network_data: pd.DataFrame) -> pd.DataFrame:
    if network_data.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    outcomes = [c for c in BASE_OUTCOMES + DECOMP_OUTCOMES if c in network_data.columns]
    for network, sub in network_data.groupby("network", dropna=False, sort=False):
        covs = _usable_columns(sub, COVARIATES)
        cov_expr = " + " + " + ".join(covs) if covs else ""
        has_memory = "memory_strict" in sub.columns and sub["memory_strict"].notna().any()
        for outcome in outcomes:
            keep = [outcome, *covs]
            if has_memory:
                keep.append("memory_strict")
            model_data = sub.dropna(subset=keep).copy()
            if len(model_data) < 30:
                continue
            interaction = " * memory_strict" if has_memory else ""
            formula = f"{outcome} ~ C(condition, Treatment('kj')){interaction}{cov_expr}"
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "outcome", outcome)
            res.insert(2, "mechanism_model", "network_trajectory_decomposition")
            rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    path = Path(args.trajectory_item or cfg.paper_output_root / DEFAULT_TRAJECTORY_ITEM)
    raw = read_table(path)
    data = prepare(raw)
    roi_models = fit_roi_models(data)
    network_data = aggregate_network(data)
    network_models = fit_network_models(network_data)
    outputs = {
        "trajectory_decomposition_item.tsv": data,
        "trajectory_decomposition_descriptives.tsv": descriptives(data),
        "trajectory_decomposition_roi_models.tsv": roi_models,
        "trajectory_decomposition_roi_focal_terms.tsv": focal_terms(roi_models),
        "trajectory_decomposition_network_models.tsv": network_models,
        "trajectory_decomposition_network_focal_terms.tsv": focal_terms(network_models),
        "manifest.tsv": pd.DataFrame(
            [
                {
                    "trajectory_item_path": str(path),
                    "input_rows": int(len(raw)),
                    "prepared_rows": int(len(data)),
                    "n_rois": int(data["roi"].nunique()) if "roi" in data.columns and not data.empty else 0,
                    "n_networks": int(data["network"].nunique()) if "network" in data.columns and not data.empty else 0,
                    "status": "ok" if not data.empty else "empty",
                }
            ]
        ),
    }
    write_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
