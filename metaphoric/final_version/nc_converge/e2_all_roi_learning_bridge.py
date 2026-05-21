#!/usr/bin/env python3
"""Append 518 E2: all-ROI sensitivity for the learning-behavior bridge.

This script mirrors D1 at the single-ROI level. It tests whether run3
understand / run4 like predict post separation, post pair similarity, and
subsequent memory within each ROI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from d1_learning_understand_like_bridge import _coerce_binary, merge_learning_behavior
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

MODULE = "e2_all_roi_learning_bridge"
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_LEARNING_BEHAVIOR = Path("qc/learning_reactivation_mapping/learning_behavior_item.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
RT_COVARS = [
    "run3_rt_z_subject_condition",
    "run4_rt_z_subject_condition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--learning-behavior", type=Path, default=None)
    return parser.parse_args()


def _usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10
    ]


def _profile_label(understand: float, like: float) -> str:
    if not (np.isfinite(understand) and np.isfinite(like)):
        return ""
    return ("understand" if understand >= 0.5 else "no_understand") + "+" + ("like" if like >= 0.5 else "no_like")


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_network_column(build_condition_item_id(frame))
    metric_col = (
        "post_edge_specificity"
        if "post_edge_specificity" in out.columns
        else ("trained_edge_drop" if "trained_edge_drop" in out.columns else None)
    )
    if metric_col is None or "pre_pair_similarity" not in out.columns or "post_pair_similarity" not in out.columns:
        return pd.DataFrame()
    numeric_cols = list(
        dict.fromkeys(
            [
                "pre_pair_similarity",
                "post_pair_similarity",
                metric_col,
                *COVARIATES,
                *RT_COVARS,
                "learning_fluency_shift",
                "run3_understand_yes",
                "run4_like_yes",
                "memory",
                "memory_score",
                "memory_successes",
                "remembered_strict",
            ]
        )
    )
    for col in numeric_cols:
        if col in out.columns:
            if col in {"run3_understand_yes", "run4_like_yes"}:
                out[col] = _coerce_binary(out[col])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[
        out.get("roi", pd.Series(index=out.index, dtype=object)).notna()
        & out.get("condition_item_id", pd.Series(index=out.index, dtype=object)).astype(str).ne("")
    ].copy()
    keys = [c for c in ["subject", "condition_item_id", "condition", "network", "roi"] if c in out.columns]
    agg_cols = [c for c in numeric_cols if c in out.columns]
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
    if "learning_fluency_shift" in data.columns:
        data["learning_fluency_shift_z"] = zscore(data["learning_fluency_shift"])
    if {"run3_understand_yes", "run4_like_yes"}.issubset(data.columns):
        data["learning_response_profile"] = [
            _profile_label(u, l) for u, l in zip(data["run3_understand_yes"], data["run4_like_yes"])
        ]
    if "remembered_strict" in data.columns:
        data["memory_strict"] = pd.to_numeric(data["remembered_strict"], errors="coerce")
    elif "memory_score" in data.columns:
        ms = pd.to_numeric(data["memory_score"], errors="coerce")
        data["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    elif "memory" in data.columns:
        ms = pd.to_numeric(data["memory"], errors="coerce")
        data["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    elif "memory_successes" in data.columns:
        ms = pd.to_numeric(data["memory_successes"], errors="coerce")
        data["memory_strict"] = np.where(ms.ge(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    return data


def _covariate_expr(frame: pd.DataFrame) -> tuple[str, list[str]]:
    covs = _usable_columns(frame, COVARIATES + RT_COVARS + ["learning_fluency_shift_z"])
    return (" + " + " + ".join(covs)) if covs else "", covs


def _fit_by_roi(data: pd.DataFrame, *, outcome: str, label: str, family: str, with_neural: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (network, roi), sub in data.groupby(["network", "roi"], dropna=False, sort=False):
        cov_expr, covs = _covariate_expr(sub)
        base_keep = [
            outcome,
            "run3_understand_yes",
            "run4_like_yes",
            "pre_pair_similarity_z",
            *covs,
        ]
        if with_neural:
            base_keep.append("post_edge_differentiation_z")
        base_keep = [c for c in base_keep if c in sub.columns]
        model_data = sub.dropna(subset=base_keep).copy()
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
        if family == "binomial" and model_data[outcome].nunique(dropna=True) < 2:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "network": network,
                            "roi": roi,
                            "outcome": outcome,
                            "mechanism_model": label,
                            "term": "__model__",
                            "status": "single_class_outcome",
                            "n_obs": len(model_data),
                        }
                    ]
                )
            )
            continue
        if outcome == "post_pair_similarity_z":
            formula = (
                "post_pair_similarity_z ~ pre_pair_similarity_z "
                "+ C(condition, Treatment('kj')) * (run3_understand_yes + run4_like_yes)"
                + cov_expr
            )
        else:
            formula = (
                f"{outcome} ~ C(condition, Treatment('kj')) "
                "* (run3_understand_yes + run4_like_yes) + pre_pair_similarity_z"
                + cov_expr
            )
        if with_neural:
            formula += " + post_edge_differentiation_z"
        res = fit_formula(model_data, formula, family=family)
        res.insert(0, "network", network)
        res.insert(1, "roi", roi)
        res.insert(2, "outcome", outcome)
        res.insert(3, "mechanism_model", label)
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def fit_models(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "roi_understand_like_to_post_separation.tsv": _fit_by_roi(
            data,
            outcome="post_edge_differentiation_z",
            label="roi_understand_like_to_post_separation",
            family="gaussian",
        ),
        "roi_understand_like_to_post_pair_similarity.tsv": _fit_by_roi(
            data,
            outcome="post_pair_similarity_z",
            label="roi_understand_like_to_post_pair_similarity",
            family="gaussian",
        ),
        "roi_understand_like_to_memory_direct.tsv": _fit_by_roi(
            data,
            outcome="memory_strict",
            label="roi_understand_like_to_memory_direct",
            family="binomial",
        ),
        "roi_understand_like_to_memory_with_neural.tsv": _fit_by_roi(
            data,
            outcome="memory_strict",
            label="roi_understand_like_to_memory_with_neural",
            family="binomial",
            with_neural=True,
        ),
    }


def focal_terms(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    terms = {
        "C(condition, Treatment('kj'))[T.yy]",
        "run3_understand_yes",
        "run4_like_yes",
        "C(condition, Treatment('kj'))[T.yy]:run3_understand_yes",
        "C(condition, Treatment('kj'))[T.yy]:run4_like_yes",
        "post_edge_differentiation_z",
    }
    frames = []
    for name, table in models.items():
        if table.empty or "term" not in table.columns:
            continue
        sub = table[table["term"].isin(terms)].copy()
        sub.insert(0, "source_file", name)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame([{"status": "empty_input"}])
    rows = []
    for keys, sub in data.groupby(["network", "roi", "condition", "learning_response_profile"], dropna=False, sort=False):
        rec = dict(zip(["network", "roi", "condition", "learning_response_profile"], keys))
        rec["n"] = int(len(sub))
        rec["n_subjects"] = int(sub["subject"].nunique()) if "subject" in sub.columns else np.nan
        rec["n_items"] = int(sub["condition_item_id"].nunique()) if "condition_item_id" in sub.columns else np.nan
        for col in ["post_edge_differentiation_z", "post_pair_similarity_z", "memory_strict"]:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                rec[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    behavior_path = Path(args.learning_behavior or cfg.paper_output_root / DEFAULT_LEARNING_BEHAVIOR)
    raw = read_table(path)
    raw, behavior_audit = merge_learning_behavior(raw, behavior_path, allow_missing=args.allow_empty)
    data = prepare(raw)
    models = fit_models(data)
    outputs = {
        **models,
        "roi_learning_bridge_focal_terms.tsv": focal_terms(models),
        "roi_learning_response_profile_descriptives.tsv": descriptives(data),
        "manifest.tsv": pd.DataFrame(
            [
                {
                    "input_path": str(path),
                    "learning_behavior_path": str(behavior_path),
                    "input_rows": int(len(raw)),
                    "prepared_rows": int(len(data)),
                    "n_rois": int(data["roi"].nunique()) if "roi" in data.columns and not data.empty else 0,
                    "behavior_merge_status": behavior_audit.iloc[0].get("learning_behavior_status", "") if not behavior_audit.empty else "",
                    "run3_understand_nonnull": int(data["run3_understand_yes"].notna().sum()) if "run3_understand_yes" in data.columns else 0,
                    "run4_like_nonnull": int(data["run4_like_yes"].notna().sum()) if "run4_like_yes" in data.columns else 0,
                    "memory_nonnull": int(data["memory_strict"].notna().sum()) if "memory_strict" in data.columns else 0,
                    "status": "ok" if not data.empty else "empty",
                }
            ]
        ),
    }
    write_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
