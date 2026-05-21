#!/usr/bin/env python3
"""E7: pooled run3-understanding to post-edge differentiation check.

This sensitivity analysis answers a narrow question: if YY/KJ condition is
removed from the fixed effects, does run3 understanding correlate with
post-stage edge differentiation?

Outputs are written under:
paper_outputs/qc/nc_converge/e7_pooled_run3_understand_post_edge/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from d1_learning_understand_like_bridge import merge_learning_behavior, prepare as prepare_network
from e2_all_roi_learning_bridge import prepare as prepare_roi
from shared_nc import add_common_args, default_config, fit_formula, q_for_ok_rows, read_table

MODULE = "e7_pooled_run3_understand_post_edge"
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_LEARNING_BEHAVIOR = Path("qc/learning_reactivation_mapping/learning_behavior_item.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
    "run3_rt_z_subject_condition",
    "run4_rt_z_subject_condition",
    "learning_fluency_shift_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--learning-behavior", type=Path, default=None)
    return parser.parse_args()


def usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10
    ]


def add_q(frame: pd.DataFrame) -> pd.DataFrame:
    if "p" not in frame.columns:
        return frame
    return q_for_ok_rows(frame, p_col="p")


def fit_by_group(data: pd.DataFrame, group_cols: list[str], level: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, sub in data.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        keydict = dict(zip(group_cols, key))
        model_specs = [
            ("simple_no_condition", ["run3_understand_yes"]),
            (
                "adjusted_no_condition",
                [
                    "run3_understand_yes",
                    "run4_like_yes",
                    "pre_pair_similarity_z",
                    *usable_columns(sub, COVARIATES),
                ],
            ),
        ]
        for label, rhs_base in model_specs:
            rhs = list(dict.fromkeys(rhs_base))
            needed = ["post_edge_differentiation_z", *rhs, "subject", "condition_item_id"]
            model_data = sub.dropna(subset=[c for c in needed if c in sub.columns]).copy()
            if len(model_data) < 30 or model_data["run3_understand_yes"].nunique(dropna=True) < 2:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                **keydict,
                                "level": level,
                                "model": label,
                                "term": "__model__",
                                "status": "too_few_rows_or_no_variation",
                                "n_obs": len(model_data),
                                "n_subjects_input": model_data["subject"].nunique()
                                if "subject" in model_data.columns
                                else np.nan,
                                "n_items_input": model_data["condition_item_id"].nunique()
                                if "condition_item_id" in model_data.columns
                                else np.nan,
                            }
                        ]
                    )
                )
                continue
            formula = "post_edge_differentiation_z ~ " + " + ".join(rhs)
            result = fit_formula(model_data, formula, family="gaussian").copy()
            for col, value in keydict.items():
                result[col] = value
            result["level"] = level
            result["model"] = label
            result["formula"] = formula
            result["n_obs_input"] = len(model_data)
            result["n_subjects_input"] = model_data["subject"].nunique()
            result["n_items_input"] = model_data["condition_item_id"].nunique()
            rows.append(result)
    if not rows:
        return pd.DataFrame()
    return add_q(pd.concat(rows, ignore_index=True, sort=False))


def describe(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        data.dropna(subset=["run3_understand_yes", "post_edge_differentiation_z"])
        .groupby([*group_cols, "run3_understand_yes"], as_index=False)
        .agg(
            n=("post_edge_differentiation_z", "size"),
            n_subjects=("subject", "nunique"),
            n_items=("condition_item_id", "nunique"),
            post_edge_mean=("post_edge_differentiation_z", "mean"),
            post_edge_sd=("post_edge_differentiation_z", "std"),
        )
    )


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    input_path = cfg.paper_output_root / (args.input or DEFAULT_INPUT)
    behavior_path = cfg.paper_output_root / (args.learning_behavior or DEFAULT_LEARNING_BEHAVIOR)
    output_dir = cfg.output_root / MODULE
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(input_path)
    merged, audit = merge_learning_behavior(raw, behavior_path, allow_missing=False)
    network_data = prepare_network(merged)
    roi_data = prepare_roi(merged)

    outputs = {
        "pooled_run3_understand_post_edge_network.tsv": fit_by_group(network_data, ["network"], "network"),
        "pooled_run3_understand_post_edge_roi.tsv": fit_by_group(roi_data, ["network", "roi"], "roi"),
        "pooled_run3_understand_post_edge_network_descriptives.tsv": describe(network_data, ["network"]),
        "pooled_run3_understand_post_edge_roi_descriptives.tsv": describe(roi_data, ["network", "roi"]),
        "merge_audit.tsv": audit,
        "manifest.tsv": pd.DataFrame(
            [
                {
                    "input_path": str(input_path),
                    "learning_behavior_path": str(behavior_path),
                    "network_rows": len(network_data),
                    "roi_rows": len(roi_data),
                    "n_subjects_network": network_data["subject"].nunique()
                    if "subject" in network_data.columns
                    else np.nan,
                    "n_items_network": network_data["condition_item_id"].nunique()
                    if "condition_item_id" in network_data.columns
                    else np.nan,
                }
            ]
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, sep="\t", index=False)
    print(f"[OK] wrote {len(outputs)} files to {output_dir}")


if __name__ == "__main__":
    main()
