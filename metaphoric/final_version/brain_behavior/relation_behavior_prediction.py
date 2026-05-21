#!/usr/bin/env python3
"""
Subject-level relation-vector brain-behavior analysis.

This script bridges the relation-vector RSA branch to run-7 behavior. It uses
subject-level ROI metrics from A1 and tests several interpretable predictors
instead of focusing only on post-pre decreases:
- relation_pre
- relation_post
- relation_delta = post - pre
- relation_decoupling = pre - post
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402


DEFAULT_ROI_SETS = ["main_functional", "literature", "literature_spatial"]
PRIMARY_MODELS = {"M9_relation_vector_direct", "M9_relation_vector_abs"}
PRIMARY_BEHAVIOR = {"run7_memory_accuracy", "run7_correct_retrieval_efficiency"}
CONDITION_MAP = {"YY": "yy", "KJ": "kj", "yy": "yy", "kj": "kj"}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _infer_roi_set(path: Path) -> str:
    name = path.name
    prefix = "relation_vector_rsa_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _safe_corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 5 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return {
            "n_subjects": int(len(pair)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
    pearson_r, pearson_p = stats.pearsonr(pair["x"], pair["y"])
    spearman_r, spearman_p = stats.spearmanr(pair["x"], pair["y"])
    return {
        "n_subjects": int(len(pair)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def _load_relation_subject_predictors(relation_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for relation_dir in relation_dirs:
        path = relation_dir / "relation_vector_subject_metrics.tsv"
        frame = _read_table(path)
        required = {"subject", "roi", "condition", "time", "neural_rdm_type", "model", "rho"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        roi_set = frame["roi_set"].iloc[0] if "roi_set" in frame.columns else _infer_roi_set(relation_dir)
        frame = frame.copy()
        frame["roi_set"] = str(roi_set)
        frame["condition"] = frame["condition"].astype(str).str.strip().str.lower()
        frame["time"] = frame["time"].astype(str).str.strip().str.lower()
        frame["rho"] = pd.to_numeric(frame["rho"], errors="coerce")
        frame = frame[frame["neural_rdm_type"].eq("relation_vector") & frame["condition"].isin(["yy", "kj"])].copy()
        pivot = (
            frame.pivot_table(
                index=["subject", "roi_set", "roi", "condition", "neural_rdm_type", "model", "model_role"],
                columns="time",
                values="rho",
                aggfunc="mean",
            )
            .reset_index()
        )
        if "pre" not in pivot.columns or "post" not in pivot.columns:
            raise ValueError(f"{path} does not contain matched pre/post rows.")
        pivot["relation_pre"] = pivot["pre"]
        pivot["relation_post"] = pivot["post"]
        pivot["relation_delta"] = pivot["post"] - pivot["pre"]
        pivot["relation_decoupling"] = pivot["pre"] - pivot["post"]
        rows.append(pivot)
    relation = pd.concat(rows, ignore_index=True)
    predictor_cols = ["relation_pre", "relation_post", "relation_delta", "relation_decoupling"]
    return relation.melt(
        id_vars=["subject", "roi_set", "roi", "condition", "neural_rdm_type", "model", "model_role"],
        value_vars=predictor_cols,
        var_name="predictor_name",
        value_name="predictor_value",
    )


def _append_behavior_rows(rows: list[dict[str, object]], frame: pd.DataFrame, *, metric: str, prefix: str, transform) -> None:
    for raw_condition in ["YY", "KJ"]:
        col = f"{prefix}_{raw_condition}"
        if col not in frame.columns:
            continue
        condition = CONDITION_MAP[raw_condition]
        values = pd.to_numeric(frame[col], errors="coerce")
        for subject, raw_value in zip(frame["subject"], values):
            if np.isfinite(raw_value):
                rows.append(
                    {
                        "subject": str(subject),
                        "condition": condition,
                        "condition_group": condition,
                        "behavior_metric": metric,
                        "behavior_value_raw": float(raw_value),
                        "behavior_value": float(transform(float(raw_value))),
                    }
                )


def _prepare_behavior(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    if "subject" not in frame.columns:
        raise ValueError(f"{path} must contain subject column.")
    frame = frame.copy()
    frame["subject"] = frame["subject"].astype(str)
    rows: list[dict[str, object]] = []
    _append_behavior_rows(rows, frame, metric="run7_memory_accuracy", prefix="accuracy", transform=lambda x: x)
    _append_behavior_rows(rows, frame, metric="run7_correct_retrieval_efficiency", prefix="inverse_efficiency", transform=lambda x: -x)
    _append_behavior_rows(rows, frame, metric="run7_correct_retrieval_speed", prefix="mean_rt_correct", transform=lambda x: -x)
    return pd.DataFrame(rows)


def _build_condition_specific(relation: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    merged = relation.merge(
        behavior,
        on=["subject", "condition"],
        how="inner",
        validate="many_to_many",
    )
    merged["analysis_type"] = "condition_specific"
    return merged


def _build_condition_contrast(relation: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    relation_wide = relation.pivot_table(
        index=["subject", "roi_set", "roi", "neural_rdm_type", "model", "model_role", "predictor_name"],
        columns="condition",
        values="predictor_value",
        aggfunc="mean",
    ).reset_index()
    if "yy" not in relation_wide.columns or "kj" not in relation_wide.columns:
        return pd.DataFrame()
    relation_wide["predictor_value"] = relation_wide["yy"] - relation_wide["kj"]
    relation_wide["condition"] = "yy_minus_kj"
    behavior_wide = behavior.pivot_table(
        index=["subject", "behavior_metric"],
        columns="condition",
        values="behavior_value",
        aggfunc="mean",
    ).reset_index()
    if "yy" not in behavior_wide.columns or "kj" not in behavior_wide.columns:
        return pd.DataFrame()
    behavior_wide["behavior_value"] = behavior_wide["yy"] - behavior_wide["kj"]
    behavior_wide["behavior_value_raw"] = behavior_wide["behavior_value"]
    behavior_wide["condition"] = "yy_minus_kj"
    merged = relation_wide.merge(
        behavior_wide[["subject", "behavior_metric", "condition", "behavior_value", "behavior_value_raw"]],
        on=["subject", "condition"],
        how="inner",
        validate="many_to_many",
    )
    merged["condition_group"] = "yy_minus_kj"
    merged["analysis_type"] = "condition_contrast"
    return merged


def _summarize_associations(merged: pd.DataFrame, *, min_subjects: int) -> pd.DataFrame:
    rows = []
    group_cols = [
        "analysis_type",
        "roi_set",
        "roi",
        "condition_group",
        "neural_rdm_type",
        "model",
        "model_role",
        "predictor_name",
        "behavior_metric",
    ]
    for keys, subset in merged.groupby(group_cols, sort=False):
        (
            analysis_type,
            roi_set,
            roi,
            condition_group,
            neural_rdm_type,
            model,
            model_role,
            predictor_name,
            behavior_metric,
        ) = keys
        pair = subset[["subject", "predictor_value", "behavior_value"]].dropna().drop_duplicates("subject")
        effect = _safe_corr(pair["predictor_value"], pair["behavior_value"])
        if effect["n_subjects"] < min_subjects:
            continue
        rows.append(
            {
                "analysis_type": analysis_type,
                "roi_set": roi_set,
                "roi": roi,
                "condition_group": condition_group,
                "neural_rdm_type": neural_rdm_type,
                "model": model,
                "model_role": model_role,
                "predictor_name": predictor_name,
                "behavior_metric": behavior_metric,
                "n_subjects": effect["n_subjects"],
                "pearson_r": effect["pearson_r"],
                "pearson_p": effect["pearson_p"],
                "spearman_r": effect["spearman_r"],
                "spearman_p": effect["spearman_p"],
                "mean_predictor": float(pd.to_numeric(pair["predictor_value"], errors="coerce").mean()),
                "mean_behavior": float(pd.to_numeric(pair["behavior_value"], errors="coerce").mean()),
                "is_primary_model": str(model) in PRIMARY_MODELS,
                "is_primary_behavior": str(behavior_metric) in PRIMARY_BEHAVIOR,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_analysis_family"] = np.nan
    out["q_bh_primary_family"] = np.nan
    for _, idx in out.groupby(["analysis_type", "roi_set", "condition_group", "behavior_metric"], dropna=False).groups.items():
        out.loc[list(idx), "q_bh_analysis_family"] = _bh_fdr(out.loc[list(idx), "spearman_p"])
    primary = out["is_primary_model"] & out["is_primary_behavior"]
    for _, idx in out[primary].groupby(["analysis_type", "roi_set", "condition_group"], dropna=False).groups.items():
        out.loc[list(idx), "q_bh_primary_family"] = _bh_fdr(out.loc[list(idx), "spearman_p"])
    return out.sort_values(["q_bh_primary_family", "spearman_p"], na_position="last").reset_index(drop=True)


def _qc_rows(relation: pd.DataFrame, behavior: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"table": "relation_subject_predictors", "n_rows": len(relation), "n_subjects": relation["subject"].nunique()},
        {"table": "behavior_subject_long", "n_rows": len(behavior), "n_subjects": behavior["subject"].nunique()},
        {"table": "merged_subject_long", "n_rows": len(merged), "n_subjects": merged["subject"].nunique()},
    ]
    for condition, frame in merged.groupby("condition_group", dropna=False):
        rows.append(
            {
                "table": f"merged_condition_{condition}",
                "n_rows": len(frame),
                "n_subjects": frame["subject"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Subject-level relation-vector brain-behavior prediction.")
    parser.add_argument(
        "--relation-dirs",
        nargs="+",
        type=Path,
        default=[base_dir / "paper_outputs" / "qc" / f"relation_vector_rsa_{roi_set}" for roi_set in DEFAULT_ROI_SETS],
    )
    parser.add_argument(
        "--behavior-summary",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "behavior_results" / "refined" / "subject_summary_wide.tsv",
    )
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--min-subjects", type=int, default=26)
    args = parser.parse_args()

    qc_dir = ensure_dir(args.paper_output_root / "qc" / "relation_behavior")
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    relation = _load_relation_subject_predictors(args.relation_dirs)
    behavior = _prepare_behavior(args.behavior_summary)
    condition_specific = _build_condition_specific(relation, behavior)
    condition_contrast = _build_condition_contrast(relation, behavior)
    merged = pd.concat([condition_specific, condition_contrast], ignore_index=True)
    summary = _summarize_associations(merged, min_subjects=args.min_subjects)

    write_table(merged, qc_dir / "relation_behavior_subject_long.tsv")
    write_table(_qc_rows(relation, behavior, merged), qc_dir / "relation_behavior_join_qc.tsv")
    write_table(pd.DataFrame(columns=["note"]), tables_si / "table_relation_behavior_itemlevel.tsv")
    write_table(summary, tables_si / "table_relation_behavior_prediction_fdr.tsv")
    main_summary = summary[
        summary["is_primary_model"].astype(bool)
        & summary["is_primary_behavior"].astype(bool)
        & summary["predictor_name"].isin(["relation_pre", "relation_post", "relation_delta", "relation_decoupling"])
    ].copy()
    write_table(main_summary, tables_main / "table_relation_behavior_prediction.tsv")
    save_json(
        {
            "relation_dirs": [str(path) for path in args.relation_dirs],
            "behavior_summary": str(args.behavior_summary),
            "paper_output_root": str(args.paper_output_root),
            "n_relation_rows": int(len(relation)),
            "n_behavior_rows": int(len(behavior)),
            "n_merged_rows": int(len(merged)),
            "n_summary_rows": int(len(summary)),
            "min_subjects": int(args.min_subjects),
            "item_level": "not_run; A1 currently exports subject-level relation metrics only",
        },
        qc_dir / "relation_behavior_manifest.json",
    )
    print(f"[relation-behavior] wrote {len(summary)} association rows to {tables_si / 'table_relation_behavior_prediction_fdr.tsv'}")


if __name__ == "__main__":
    main()
