#!/usr/bin/env python3
"""Append 518 E6: focused learning-to-post bridge in memory-linked ROIs.

Restrict analyses to the four ROIs where E2 found a FDR-significant
post_edge_differentiation -> subsequent memory relationship:

meta_L_PPC_SPL, meta_R_PPC_SPL, meta_R_precuneus, meta_R_pMTG_pSTS.

This avoids another all-ROI fishing pass and directly tests whether run3
understanding or learning activation explains the post-stage edge
differentiation bridge.
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

MODULE = "e6_focused_learning_to_post_memory_rois"
DEFAULT_ITEM = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_BEHAVIOR = Path("qc/learning_reactivation_mapping/learning_behavior_item.tsv")
DEFAULT_ACTIVATION = Path("qc/nc_converge/e4_learning_activation_sme/learning_activation_roi_item.tsv")
MEMORY_ROIS = ["meta_L_PPC_SPL", "meta_R_PPC_SPL", "meta_R_precuneus", "meta_R_pMTG_pSTS"]
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
    parser.add_argument("--learning-behavior", type=Path, default=None)
    parser.add_argument("--activation", type=Path, default=None)
    return parser.parse_args()


def _usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().sum() >= 10]


def _memory_strict(frame: pd.DataFrame) -> pd.Series:
    if "remembered_strict" in frame.columns:
        return pd.to_numeric(frame["remembered_strict"], errors="coerce")
    if "memory_score" in frame.columns:
        ms = pd.to_numeric(frame["memory_score"], errors="coerce")
        return pd.Series(np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan)), index=frame.index)
    if "memory" in frame.columns:
        ms = pd.to_numeric(frame["memory"], errors="coerce")
        return pd.Series(np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan)), index=frame.index)
    if "memory_successes" in frame.columns:
        ms = pd.to_numeric(frame["memory_successes"], errors="coerce")
        return pd.Series(np.where(ms.ge(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan)), index=frame.index)
    return pd.Series(np.nan, index=frame.index)


def _activation_wide(path: Path) -> pd.DataFrame:
    act = read_table(path)
    act = act[act["roi"].isin(MEMORY_ROIS)].copy()
    keep = ["subject", "condition", "condition_item_id", "roi", "run", "activation_z_subject_roi_run"]
    act = act[keep].dropna(subset=["activation_z_subject_roi_run"])
    wide = act.pivot_table(
        index=["subject", "condition", "condition_item_id", "roi"],
        columns="run",
        values="activation_z_subject_roi_run",
        aggfunc="mean",
    ).reset_index()
    wide = wide.rename(columns={3: "run3_activation_z", 4: "run4_activation_z"})
    return wide


def prepare(raw: pd.DataFrame, behavior_path: Path, activation_path: Path) -> pd.DataFrame:
    raw, _audit = merge_learning_behavior(raw, behavior_path, allow_missing=False)
    out = add_network_column(build_condition_item_id(raw))
    out = out[out["roi"].isin(MEMORY_ROIS)].copy()
    metric_col = "post_edge_specificity" if "post_edge_specificity" in out.columns else "trained_edge_drop"
    numeric = [
        "pre_pair_similarity",
        "post_pair_similarity",
        metric_col,
        "run3_understand_yes",
        "run4_like_yes",
        "run3_rt_z_subject_condition",
        "run4_rt_z_subject_condition",
        "learning_fluency_shift",
        "memory",
        "memory_score",
        "memory_successes",
        "remembered_strict",
        *COVARIATES,
    ]
    for col in numeric:
        if col in out.columns:
            if col in {"run3_understand_yes", "run4_like_yes"}:
                out[col] = _coerce_binary(out[col])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    keys = ["subject", "condition", "condition_item_id", "network", "roi"]
    cols = [c for c in numeric if c in out.columns]
    data = out[keys + cols].groupby(keys, dropna=False, as_index=False).mean()
    data = data.rename(
        columns={
            "pre_pair_similarity": "pre_pair_similarity_raw",
            "post_pair_similarity": "post_pair_similarity_raw",
            metric_col: "post_edge_differentiation_raw",
        }
    )
    data["post_edge_differentiation_z"] = zscore(data["post_edge_differentiation_raw"])
    data["post_pair_similarity_z"] = zscore(data["post_pair_similarity_raw"])
    data["pre_pair_similarity_z"] = zscore(data["pre_pair_similarity_raw"])
    data["learning_fluency_shift_z"] = zscore(data["learning_fluency_shift"]) if "learning_fluency_shift" in data.columns else np.nan
    data["memory_strict"] = _memory_strict(data)
    data = data.merge(_activation_wide(activation_path), on=["subject", "condition", "condition_item_id", "roi"], how="left")
    return data


def fit_models(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows_post: list[pd.DataFrame] = []
    rows_memory: list[pd.DataFrame] = []
    for roi, sub in data.groupby("roi", dropna=False, sort=False):
        network = sub["network"].dropna().iloc[0] if sub["network"].notna().any() else ""
        covs = _usable_columns(sub, COVARIATES + ["run3_rt_z_subject_condition", "run4_rt_z_subject_condition", "learning_fluency_shift_z"])
        cov_expr = " + " + " + ".join(covs) if covs else ""

        # Learning -> post differentiation.
        predictors = [
            "run3_understand_yes",
            "run4_like_yes",
            "run3_activation_z",
            "run4_activation_z",
            "pre_pair_similarity_z",
            *covs,
        ]
        model_data = sub.dropna(subset=[c for c in ["post_edge_differentiation_z", *predictors] if c in sub.columns]).copy()
        if len(model_data) >= 30:
            formula = (
                "post_edge_differentiation_z ~ C(condition, Treatment('kj')) "
                "* (run3_understand_yes + run4_like_yes + run3_activation_z + run4_activation_z)"
                " + pre_pair_similarity_z"
                + cov_expr
            )
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "roi", roi)
            res.insert(2, "mechanism_model", "focused_learning_to_post_edge")
            rows_post.append(res)

        # Memory model, checking whether activation explains the post-edge bridge.
        predictors = [
            "post_edge_differentiation_z",
            "run3_understand_yes",
            "run4_like_yes",
            "run3_activation_z",
            "run4_activation_z",
            "pre_pair_similarity_z",
            *covs,
        ]
        model_data = sub.dropna(subset=[c for c in ["memory_strict", *predictors] if c in sub.columns]).copy()
        if len(model_data) >= 30 and model_data["memory_strict"].nunique(dropna=True) >= 2:
            formula = (
                "memory_strict ~ C(condition, Treatment('kj')) "
                "* (run3_understand_yes + run4_like_yes + run3_activation_z + run4_activation_z)"
                " + post_edge_differentiation_z + pre_pair_similarity_z"
                + cov_expr
            )
            res = fit_formula(model_data, formula, family="binomial")
            res.insert(0, "network", network)
            res.insert(1, "roi", roi)
            res.insert(2, "mechanism_model", "focused_memory_with_learning_and_post")
            rows_memory.append(res)

    return {
        "focused_learning_to_post_edge_models.tsv": q_for_ok_rows(pd.concat(rows_post, ignore_index=True, sort=False)) if rows_post else pd.DataFrame(),
        "focused_memory_with_learning_and_post_models.tsv": q_for_ok_rows(pd.concat(rows_memory, ignore_index=True, sort=False)) if rows_memory else pd.DataFrame(),
    }


def focal_terms(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    terms = {
        "run3_understand_yes",
        "run4_like_yes",
        "run3_activation_z",
        "run4_activation_z",
        "C(condition, Treatment('kj'))[T.yy]:run3_understand_yes",
        "C(condition, Treatment('kj'))[T.yy]:run4_like_yes",
        "C(condition, Treatment('kj'))[T.yy]:run3_activation_z",
        "C(condition, Treatment('kj'))[T.yy]:run4_activation_z",
        "post_edge_differentiation_z",
        "pre_pair_similarity_z",
    }
    frames = []
    for name, table in models.items():
        if table.empty:
            continue
        sub = table[table["term"].isin(terms)].copy()
        sub.insert(0, "source_file", name)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["network", "roi", "condition"], dropna=False, as_index=False)
        .agg(
            n=("post_edge_differentiation_z", "count"),
            n_subjects=("subject", "nunique"),
            n_items=("condition_item_id", "nunique"),
            post_edge_mean=("post_edge_differentiation_z", "mean"),
            memory_mean=("memory_strict", "mean"),
            run3_understand_mean=("run3_understand_yes", "mean"),
            run3_activation_mean=("run3_activation_z", "mean"),
        )
    )


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    input_path = Path(args.input or cfg.paper_output_root / DEFAULT_ITEM)
    behavior_path = Path(args.learning_behavior or cfg.paper_output_root / DEFAULT_BEHAVIOR)
    activation_path = Path(args.activation or cfg.paper_output_root / DEFAULT_ACTIVATION)
    raw = read_table(input_path)
    data = prepare(raw, behavior_path, activation_path)
    models = fit_models(data)
    write_outputs(
        cfg,
        MODULE,
        {
            "focused_learning_post_table.tsv": data,
            "focused_learning_post_descriptives.tsv": descriptives(data),
            **models,
            "focused_learning_post_focal_terms.tsv": focal_terms(models),
            "manifest.tsv": pd.DataFrame(
                [
                    {
                        "input_path": str(input_path),
                        "learning_behavior": str(behavior_path),
                        "activation": str(activation_path),
                        "memory_rois": ",".join(MEMORY_ROIS),
                        "rows": int(len(data)),
                        "n_rois": int(data["roi"].nunique()) if not data.empty else 0,
                        "status": "ok" if not data.empty else "empty",
                    }
                ]
            ),
        },
    )


if __name__ == "__main__":
    main()
