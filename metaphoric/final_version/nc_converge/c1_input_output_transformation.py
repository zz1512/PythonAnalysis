#!/usr/bin/env python3
"""C1: input-output transformation from pre to post pair similarity.

Unit policy
-----------
The earlier implementation chained ``make_network_composite`` (which already
z-scored ROI-level values within ``subject x roi x metric``) with a second
``groupby('network').transform(zscore)`` and then fed the resulting
``*_network_z`` columns into the regression. Because YY/KJ samples are nearly
balanced within each network, that double z-score forced ``mean(KJ_post_z)
~ -mean(YY_post_z)`` mathematically, which is the same family of problems that
caused the §28.4 mirror artefact.

This version aggregates raw ROI-level ``pre_pair_similarity`` /
``post_pair_similarity`` to ``subject x network x condition_item_id``, then
applies a single global z-score to feed the regression. Descriptives are
reported in both raw and z units so readers can tell which scale a number is
on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    zscore,
)

MODULE = "c1_input_output_transformation"
COVARIATES = ["sentence_char_len_z", "word_frequency_mean_z", "stroke_count_mean_z", "valence_mean_z", "arousal_mean_z"]
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    return parser.parse_args()


def usable_covariates(frame: pd.DataFrame) -> list[str]:
    out = []
    for col in COVARIATES:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10:
            out.append(col)
    return out


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw ROI-level similarity to network mean, then global z-score."""
    out = add_network_column(build_condition_item_id(frame))
    for col in ["pre_pair_similarity", "post_pair_similarity", *COVARIATES]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "network" not in out.columns or "condition_item_id" not in out.columns:
        return out
    out = out[out["network"].notna() & out["condition_item_id"].astype(str).ne("")].copy()
    id_keys = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in out.columns]
    raw_metrics = [c for c in ["pre_pair_similarity", "post_pair_similarity"] if c in out.columns]
    if not raw_metrics or not id_keys:
        return out
    network_raw = out.groupby(id_keys, dropna=False, as_index=False)[raw_metrics].mean()
    cov_keep = [c for c in COVARIATES if c in out.columns]
    if cov_keep:
        cov_table = out.groupby(id_keys, dropna=False, as_index=False)[cov_keep].mean()
        network_raw = network_raw.merge(cov_table, on=id_keys, how="left")
    network_raw = network_raw.rename(
        columns={
            "pre_pair_similarity": "pre_pair_similarity_network_raw",
            "post_pair_similarity": "post_pair_similarity_network_raw",
        }
    )
    # Single global z-score (one mean / one SD across the whole network-level
    # table) keeps condition contrasts in their natural scale rather than
    # forcing the within-network mean to zero.
    for raw_col, z_col in [
        ("pre_pair_similarity_network_raw", "pre_pair_similarity_z"),
        ("post_pair_similarity_network_raw", "post_pair_similarity_z"),
    ]:
        if raw_col in network_raw.columns:
            network_raw[z_col] = zscore(network_raw[raw_col])
    return network_raw


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    needed = {"condition", "network", "pre_pair_similarity_network_raw", "post_pair_similarity_network_raw"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    desc = data.groupby(["network", "condition"], dropna=False, as_index=False).agg(
        n=("post_pair_similarity_network_raw", "count"),
        n_subjects=("subject", "nunique") if "subject" in data.columns else ("post_pair_similarity_network_raw", "size"),
        n_items=("condition_item_id", "nunique") if "condition_item_id" in data.columns else ("post_pair_similarity_network_raw", "size"),
        pre_mean_raw=("pre_pair_similarity_network_raw", "mean"),
        post_mean_raw=("post_pair_similarity_network_raw", "mean"),
        pre_sd_raw=("pre_pair_similarity_network_raw", "std"),
        post_sd_raw=("post_pair_similarity_network_raw", "std"),
        pre_mean_z=("pre_pair_similarity_z", "mean") if "pre_pair_similarity_z" in data.columns else ("pre_pair_similarity_network_raw", "size"),
        post_mean_z=("post_pair_similarity_z", "mean") if "post_pair_similarity_z" in data.columns else ("post_pair_similarity_network_raw", "size"),
    )
    desc["post_minus_pre_raw"] = desc["post_mean_raw"] - desc["pre_mean_raw"]
    if {"pre_mean_z", "post_mean_z"}.issubset(desc.columns):
        desc["post_minus_pre_z"] = desc["post_mean_z"] - desc["pre_mean_z"]
    return desc


def fit_models(data: pd.DataFrame) -> pd.DataFrame:
    needed = {"subject", "condition_item_id", "condition", "network", "pre_pair_similarity_z", "post_pair_similarity_z"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"term": "__model__", "status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    rows = []
    for network, sub in data.groupby("network", dropna=False):
        covs = usable_covariates(sub)
        model_data = sub.dropna(subset=["pre_pair_similarity_z", "post_pair_similarity_z", *covs]).copy()
        if len(model_data) < 30:
            rows.append(pd.DataFrame([{"network": network, "term": "__model__", "status": "too_few_rows", "n_obs": len(model_data)}]))
            continue
        formula = "post_pair_similarity_z ~ pre_pair_similarity_z * condition"
        if covs:
            formula += " + " + " + ".join(covs)
        res = fit_formula(model_data, formula, family="gaussian")
        res.insert(0, "network", network)
        res.insert(1, "mechanism_model", "input_output_transformation")
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    raw = read_table(path)
    data = prepare(raw)
    desc = descriptives(data)
    models = fit_models(data)
    write_outputs(cfg, MODULE, {
        "input_output_transformation.tsv": models,
        "input_output_descriptives.tsv": desc,
        "input_output_manifest.tsv": pd.DataFrame([{"path": str(path), "status": "ok", "n_rows": len(raw)}]),
    })


if __name__ == "__main__":
    main()
