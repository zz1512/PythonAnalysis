#!/usr/bin/env python3
"""Model learning reactivation / behavior update predicting post edge differentiation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [FINAL_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lr_utils import available_covariates, fit_gee_by_roi, write_tsv  # noqa: E402


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _z_by_roi(frame: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby(["roi_set", "roi"], sort=False).groups.items():
        vals = pd.to_numeric(frame.loc[idx, col], errors="coerce")
        sd = vals.std(ddof=1)
        if not np.isfinite(sd) or np.isclose(sd, 0.0):
            out.loc[idx] = np.nan
        else:
            out.loc[idx] = (vals - vals.mean()) / sd
    return out


def build_analysis_table(base_dir: Path, out_dir: Path) -> pd.DataFrame:
    old_item = pd.read_csv(
        base_dir / "paper_outputs" / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv",
        sep="\t",
    )
    react = pd.read_csv(out_dir / "learning_reactivation_item.tsv", sep="\t")
    behavior = pd.read_csv(out_dir / "learning_behavior_item.tsv", sep="\t")
    mapping_path = out_dir / "directional_mapping_item.tsv"
    mapping = pd.read_csv(mapping_path, sep="\t") if mapping_path.exists() else pd.DataFrame()

    key_roi = ["subject", "roi_set", "roi", "condition", "original_pair_id", "template_pair_id", "condition_item_id"]
    keep_old = key_roi + [
        col for col in [
            "post_edge_specificity",
            "trained_edge_drop",
            "pseudo_edge_drop",
            "pre_pair_similarity",
            "post_pair_similarity",
            "retrieval_minus_post_pair_similarity",
            "retrieval_minus_pre_pair_similarity",
            "retrieval_pair_similarity",
            "memory",
            "log_rt_correct",
            "sentence_char_len_z",
            "word_frequency_mean_z",
            "stroke_count_mean_z",
            "valence_mean_z",
            "arousal_mean_z",
        ]
        if col in old_item.columns
    ]
    table = old_item[keep_old].copy()
    react_keep = key_roi + [
        "react_role_a_avg",
        "react_role_b_avg",
        "react_pair_mean_avg",
        "react_asymmetry_b_minus_a_avg",
        "react_pair_mean_delta_run4_minus_run3",
    ]
    table = table.merge(react[[col for col in react_keep if col in react.columns]], on=key_roi, how="left")
    behavior_key = ["subject", "condition", "original_pair_id", "template_pair_id", "condition_item_id"]
    behavior_keep = behavior_key + [
        "run3_rt",
        "run3_rt_z_subject_condition",
        "run3_resp",
        "run3_resp_missing",
        "run3_understand_yes",
        "run4_rt",
        "run4_rt_z_subject_condition",
        "run4_resp",
        "run4_resp_missing",
        "run4_like_yes",
        "learning_fluency_shift",
        "learning_response_profile",
    ]
    table = table.merge(behavior[[col for col in behavior_keep if col in behavior.columns]], on=behavior_key, how="left")
    if not mapping.empty:
        mapping_keep = key_roi + [
            "target_toward_source",
            "source_toward_target",
            "mapping_asymmetry",
            "post_role_a_pre_role_a",
            "post_role_b_pre_role_b",
            "target_source_over_self",
            "source_target_over_self",
            "mapping_asymmetry_self_corrected",
        ]
        table = table.merge(mapping[[col for col in mapping_keep if col in mapping.columns]], on=key_roi, how="left")

    for col in [
        "pre_pair_similarity",
        "post_pair_similarity",
        "post_edge_specificity",
        "trained_edge_drop",
        "pseudo_edge_drop",
        "retrieval_pair_similarity",
        "retrieval_minus_post_pair_similarity",
        "retrieval_minus_pre_pair_similarity",
        "react_role_a_avg",
        "react_role_b_avg",
        "react_pair_mean_avg",
        "react_asymmetry_b_minus_a_avg",
        "react_pair_mean_delta_run4_minus_run3",
        "learning_fluency_shift",
        "target_toward_source",
        "source_toward_target",
        "mapping_asymmetry",
        "post_role_a_pre_role_a",
        "post_role_b_pre_role_b",
        "target_source_over_self",
        "source_target_over_self",
        "mapping_asymmetry_self_corrected",
    ]:
        if col in table.columns:
            table[f"{col}_z"] = _z_by_roi(table, col)
    write_tsv(table, out_dir / "reactivation_mapping_analysis_table.tsv")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping")
    parser.add_argument("--table-out-dir", type=Path, default=base_dir / "paper_outputs" / "tables_exploratory" / "learning_reactivation_mapping")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.table_out_dir.mkdir(parents=True, exist_ok=True)
    table = build_analysis_table(args.base_dir, args.out_dir)
    covs = available_covariates(table)
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    specs = [
        (
            "reactivation_to_post_primary",
            "post_edge_specificity ~ react_pair_mean_avg_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z" + cov_expr,
            "post_edge_specificity",
        ),
        (
            "behavior_fluency_to_post",
            "post_edge_specificity ~ learning_fluency_shift_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z" + cov_expr,
            "post_edge_specificity",
        ),
        (
            "reactivation_x_fluency_to_post",
            "post_edge_specificity ~ react_pair_mean_avg_z * learning_fluency_shift_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z" + cov_expr,
            "post_edge_specificity",
        ),
        (
            "mapping_to_post",
            "post_edge_specificity ~ mapping_asymmetry_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z" + cov_expr,
            "post_edge_specificity",
        ),
        (
            "mapping_self_corrected_to_post",
            "post_edge_specificity ~ mapping_asymmetry_self_corrected_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z" + cov_expr,
            "post_edge_specificity",
        ),
        (
            "target_source_over_self_to_post",
            "post_edge_specificity ~ target_source_over_self_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z" + cov_expr,
            "post_edge_specificity",
        ),
        (
            "reactivation_to_pre_negative",
            "pre_pair_similarity ~ react_pair_mean_avg_z * C(condition, Treatment(reference='kj'))" + cov_expr,
            "pre_pair_similarity",
        ),
        (
            "reactivation_to_pseudo_negative",
            "pseudo_edge_drop ~ react_pair_mean_avg_z * C(condition, Treatment(reference='kj'))" + cov_expr,
            "pseudo_edge_drop",
        ),
    ]
    results = []
    for model_name, formula, outcome in specs:
        results.append(fit_gee_by_roi(table, formula=formula, outcome=outcome, model_name=model_name))
    models = pd.concat(results, ignore_index=True)
    write_tsv(models, args.out_dir / "reactivation_to_post_models.tsv")
    write_tsv(models, args.table_out_dir / "table_reactivation_to_post.tsv")
    summary = {
        "script": str(Path(__file__).resolve()),
        "n_analysis_rows": int(len(table)),
        "n_subjects": int(table["subject"].nunique()),
        "n_rois": int(table[["roi_set", "roi"]].drop_duplicates().shape[0]),
        "n_model_rows": int(len(models)),
        "covariates": covs,
    }
    (args.out_dir / "reactivation_to_post_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
