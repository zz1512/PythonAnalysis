#!/usr/bin/env python3
"""Append 518 E1: all-ROI sensitivity for pre-similarity tertile effects.

This script mirrors C1b but keeps each ROI separate instead of collapsing
ROIs into the semantic / hpc-spatial network composites. It is intended as
supplementary sensitivity evidence, not as a replacement for the predefined
network-level analysis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    add_network_column,
    bh_fdr,
    build_condition_item_id,
    default_config,
    fit_formula,
    q_for_ok_rows,
    read_table,
    write_outputs,
    zscore,
)

MODULE = "e1_all_roi_pre_similarity_tertile"
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--low-quantile", type=float, default=0.30)
    parser.add_argument("--high-quantile", type=float, default=0.70)
    return parser.parse_args()


def _usable_covariates(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in COVARIATES
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10
    ]


def _assign_tertile(frame: pd.DataFrame, low_q: float, high_q: float) -> pd.Series:
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    if not {"roi", "condition", "pre_pair_similarity_roi_raw"}.issubset(frame.columns):
        return labels
    for (_roi, _condition), idx in frame.groupby(["roi", "condition"], dropna=False).groups.items():
        values = pd.to_numeric(frame.loc[idx, "pre_pair_similarity_roi_raw"], errors="coerce")
        if values.notna().sum() < 6:
            continue
        low_cut = values.quantile(low_q)
        high_cut = values.quantile(high_q)
        if not (np.isfinite(low_cut) and np.isfinite(high_cut)) or high_cut <= low_cut:
            continue
        labels.loc[idx[values <= low_cut]] = "low"
        labels.loc[idx[(values > low_cut) & (values < high_cut)]] = "mid"
        labels.loc[idx[values >= high_cut]] = "high"
    return labels


def prepare(frame: pd.DataFrame, low_q: float, high_q: float) -> pd.DataFrame:
    out = add_network_column(build_condition_item_id(frame))
    metric_col = (
        "post_edge_specificity"
        if "post_edge_specificity" in out.columns
        else ("trained_edge_drop" if "trained_edge_drop" in out.columns else None)
    )
    if metric_col is None or "pre_pair_similarity" not in out.columns or "post_pair_similarity" not in out.columns:
        return pd.DataFrame()
    needed = [
        "pre_pair_similarity",
        "post_pair_similarity",
        metric_col,
        *COVARIATES,
    ]
    for col in needed:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[
        out.get("roi", pd.Series(index=out.index, dtype=object)).notna()
        & out.get("condition_item_id", pd.Series(index=out.index, dtype=object)).astype(str).ne("")
    ].copy()
    keys = [c for c in ["subject", "condition_item_id", "condition", "network", "roi"] if c in out.columns]
    agg_cols = [c for c in needed if c in out.columns]
    data = out.groupby(keys, dropna=False, as_index=False)[agg_cols].mean()
    data = data.rename(
        columns={
            "pre_pair_similarity": "pre_pair_similarity_roi_raw",
            "post_pair_similarity": "post_pair_similarity_roi_raw",
            metric_col: "post_edge_differentiation_raw",
        }
    )
    data["pre_pair_similarity_z"] = zscore(data["pre_pair_similarity_roi_raw"])
    data["post_pair_similarity_z"] = zscore(data["post_pair_similarity_roi_raw"])
    data["post_edge_differentiation_z"] = zscore(data["post_edge_differentiation_raw"])
    data["pre_tertile"] = _assign_tertile(data, low_q, high_q)
    return data


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame([{"status": "empty_input"}])
    rows = []
    for keys, sub in data.groupby(["network", "roi", "condition", "pre_tertile"], dropna=False, sort=False):
        record = dict(zip(["network", "roi", "condition", "pre_tertile"], keys))
        record["n"] = int(len(sub))
        record["n_subjects"] = int(sub["subject"].nunique()) if "subject" in sub.columns else np.nan
        record["n_items"] = int(sub["condition_item_id"].nunique()) if "condition_item_id" in sub.columns else np.nan
        for col in [
            "pre_pair_similarity_roi_raw",
            "post_pair_similarity_roi_raw",
            "post_edge_differentiation_raw",
            "post_edge_differentiation_z",
        ]:
            values = pd.to_numeric(sub[col], errors="coerce")
            record[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def fit_models(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    outcomes = [
        ("post_edge_differentiation_z", "roi_edge_separation_by_pre_tertile"),
        ("post_pair_similarity_z", "roi_post_similarity_by_pre_tertile"),
    ]
    for (network, roi), sub in data.groupby(["network", "roi"], dropna=False, sort=False):
        covs = _usable_covariates(sub)
        cov_expr = " + " + " + ".join(covs) if covs else ""
        for outcome, label in outcomes:
            model_data = sub.dropna(subset=[outcome, "pre_tertile", *covs]).copy()
            model_data = model_data[model_data["pre_tertile"].isin(["low", "mid", "high"])]
            if len(model_data) < 30:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "network": network,
                                "roi": roi,
                                "outcome": outcome,
                                "mechanism_model": label,
                                "term": "__model__",
                                "status": "too_few_rows",
                                "n_obs": len(model_data),
                            }
                        ]
                    )
                )
                continue
            formula = (
                f"{outcome} ~ C(condition, Treatment(reference='kj')) "
                f"* C(pre_tertile, Treatment(reference='low')){cov_expr}"
            )
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "roi", roi)
            res.insert(2, "outcome", outcome)
            res.insert(3, "mechanism_model", label)
            rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def focal_contrasts(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    rows = []
    for (network, roi), sub in data.groupby(["network", "roi"], dropna=False, sort=False):
        cond_stats = {}
        for condition, cond_data in sub.groupby("condition", dropna=False):
            high = pd.to_numeric(cond_data.loc[cond_data["pre_tertile"].eq("high"), "post_edge_differentiation_z"], errors="coerce")
            low = pd.to_numeric(cond_data.loc[cond_data["pre_tertile"].eq("low"), "post_edge_differentiation_z"], errors="coerce")
            high = high[np.isfinite(high)]
            low = low[np.isfinite(low)]
            if len(high) < 5 or len(low) < 5:
                continue
            est = float(high.mean() - low.mean())
            se = float(np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low)))
            stat = est / se if se > 0 else np.nan
            p = float(2 * (1 - 0.5 * (1 + math.erf(abs(stat) / math.sqrt(2))))) if np.isfinite(stat) else np.nan
            cond_key = str(condition).lower()
            cond_stats[cond_key] = {"estimate": est, "se": se}
            rows.append(
                {
                    "network": network,
                    "roi": roi,
                    "contrast": f"{cond_key}_high_minus_low",
                    "estimate": est,
                    "se": se,
                    "stat": stat,
                    "p": p,
                    "n_high": int(len(high)),
                    "n_low": int(len(low)),
                }
            )
        if "yy" in cond_stats and "kj" in cond_stats:
            est = cond_stats["yy"]["estimate"] - cond_stats["kj"]["estimate"]
            se = float(np.sqrt(cond_stats["yy"]["se"] ** 2 + cond_stats["kj"]["se"] ** 2))
            stat = est / se if se > 0 else np.nan
            p = float(2 * (1 - 0.5 * (1 + math.erf(abs(stat) / math.sqrt(2))))) if np.isfinite(stat) else np.nan
            rows.append(
                {
                    "network": network,
                    "roi": roi,
                    "contrast": "did_yy_minus_kj_high_low",
                    "estimate": est,
                    "se": se,
                    "stat": stat,
                    "p": p,
                    "n_high": np.nan,
                    "n_low": np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and "p" in out.columns:
        out["q_bh"] = bh_fdr(out["p"])
    return out


def focal_terms(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty or "term" not in models.columns:
        return pd.DataFrame()
    terms = [
        "C(condition, Treatment(reference='kj'))[T.yy]:C(pre_tertile, Treatment(reference='low'))[T.high]",
        "C(condition, Treatment(reference='kj'))[T.yy]:C(pre_tertile, Treatment(reference='low'))[T.mid]",
        "C(condition, Treatment(reference='kj'))[T.yy]",
    ]
    return models[models["term"].isin(terms)].copy()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    raw = read_table(path)
    data = prepare(raw, float(args.low_quantile), float(args.high_quantile))
    models = fit_models(data)
    outputs = {
        "roi_pre_tertile_descriptives.tsv": descriptives(data),
        "roi_pre_tertile_model_coefficients.tsv": models,
        "roi_pre_tertile_focal_terms.tsv": focal_terms(models),
        "roi_pre_tertile_focal_contrasts.tsv": focal_contrasts(data),
        "manifest.tsv": pd.DataFrame(
            [
                {
                    "input_path": str(path),
                    "input_rows": int(len(raw)),
                    "prepared_rows": int(len(data)),
                    "n_rois": int(data["roi"].nunique()) if "roi" in data.columns and not data.empty else 0,
                    "low_quantile": float(args.low_quantile),
                    "high_quantile": float(args.high_quantile),
                    "status": "ok" if not data.empty else "empty",
                }
            ]
        ),
    }
    write_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
