#!/usr/bin/env python3
"""Predict run-7 retrieval success from retrieval-stage geometry.

This script uses outputs from `rsa_analysis/retrieval_pair_similarity.py`.
It is intentionally intersection-based: only run-7 trials with neural item
metrics and non-missing behavior enter the models.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE


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


PRIMARY_METRICS = {
    "post_to_retrieval_word_reinstatement",
    "post_to_retrieval_pair_reinstatement",
    "retrieval_pair_similarity",
}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
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


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def _merge_pair_metrics(item_metrics: pd.DataFrame, pair_metrics_path: Path | None) -> pd.DataFrame:
    if pair_metrics_path is None or not pair_metrics_path.exists():
        return item_metrics
    pair = _read_table(pair_metrics_path)
    cols = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "pair_id",
        "post_to_retrieval_pair_reinstatement",
        "pre_to_retrieval_pair_reinstatement",
        "post_to_retrieval_pair_same",
        "pre_to_retrieval_pair_same",
    ]
    cols = [col for col in cols if col in pair.columns]
    if len(cols) <= 5:
        return item_metrics
    pair = pair[cols].drop_duplicates(["subject", "roi_set", "roi", "condition", "pair_id"])
    out = item_metrics.merge(
        pair,
        on=["subject", "roi_set", "roi", "condition", "pair_id"],
        how="left",
        validate="many_to_one",
    )
    return out


def _prepare_long(item_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = item_metrics.copy()
    frame["subject"] = frame["subject"].astype(str)
    frame["memory"] = pd.to_numeric(frame["memory"], errors="coerce")
    frame["action_time"] = pd.to_numeric(frame["action_time"], errors="coerce")
    frame["log_rt_correct"] = np.where(
        frame["memory"].eq(1) & frame["action_time"].gt(0),
        np.log(frame["action_time"]),
        np.nan,
    )

    metric_cols = [
        "retrieval_pair_similarity",
        "post_pair_similarity",
        "pre_pair_similarity",
        "post_to_retrieval_word_reinstatement",
        "pre_to_retrieval_word_reinstatement",
        "post_to_retrieval_pair_reinstatement",
        "pre_to_retrieval_pair_reinstatement",
        "post_to_retrieval_same_word",
        "pre_to_retrieval_same_word",
        "post_to_retrieval_pair_same",
        "pre_to_retrieval_pair_same",
    ]
    available = [col for col in metric_cols if col in frame.columns]
    long = frame.melt(
        id_vars=["subject", "roi_set", "roi", "condition", "condition_label", "pair_id", "word_label", "memory", "log_rt_correct"],
        value_vars=available,
        var_name="metric",
        value_name="metric_value",
    )
    long["metric_value"] = pd.to_numeric(long["metric_value"], errors="coerce")
    return long.dropna(subset=["metric_value", "memory"])


def _fit_gee(subset: pd.DataFrame, response: str) -> dict[str, object] | None:
    frame = subset.copy()
    frame = frame.dropna(subset=["metric_value", response, "subject"])
    if frame["subject"].nunique() < 8 or frame[response].nunique() < 2:
        return None
    frame["metric_between_subject"] = frame.groupby("subject")["metric_value"].transform("mean")
    frame["metric_within_subject"] = frame["metric_value"] - frame["metric_between_subject"]
    frame["metric_within_z"] = _zscore(frame["metric_within_subject"])
    frame["metric_between_z"] = _zscore(frame["metric_between_subject"])
    frame = frame.dropna(subset=["metric_within_z", "metric_between_z", response])
    if len(frame) < 30 or frame["subject"].nunique() < 8:
        return None
    family = Binomial() if response == "memory" else Gaussian()
    formula = f"{response} ~ metric_within_z + metric_between_z"
    try:
        model = GEE.from_formula(
            formula,
            groups="subject",
            data=frame,
            family=family,
            cov_struct=Exchangeable(),
        )
        fit = model.fit()
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "n_trials": int(len(frame)),
            "n_subjects": int(frame["subject"].nunique()),
        }
    params = fit.params
    pvalues = fit.pvalues
    bse = fit.bse
    return {
        "status": "ok",
        "n_trials": int(len(frame)),
        "n_subjects": int(frame["subject"].nunique()),
        "within_beta": float(params.get("metric_within_z", np.nan)),
        "within_se": float(bse.get("metric_within_z", np.nan)),
        "within_p": float(pvalues.get("metric_within_z", np.nan)),
        "between_beta": float(params.get("metric_between_z", np.nan)),
        "between_se": float(bse.get("metric_between_z", np.nan)),
        "between_p": float(pvalues.get("metric_between_z", np.nan)),
        "family": "binomial" if response == "memory" else "gaussian",
    }


def summarize_models(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "condition", "condition_label", "metric"]
    for keys, subset in long.groupby(group_cols, sort=False):
        roi_set, roi, condition, condition_label, metric = keys
        for response in ["memory", "log_rt_correct"]:
            fit = _fit_gee(subset, response)
            if fit is None:
                continue
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "condition_label": condition_label,
                    "metric": metric,
                    "metric_role": "primary" if metric in PRIMARY_METRICS else "control",
                    "response": response,
                    **fit,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_roi_set_response"] = np.nan
    out["q_bh_primary_family"] = np.nan
    ok = out["status"].eq("ok")
    for _, idx in out[ok].groupby(["roi_set", "response"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set_response"] = _bh_fdr(out.loc[idx, "within_p"])
    primary = ok & out["metric_role"].eq("primary")
    for _, idx in out[primary].groupby(["roi_set", "response"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_primary_family"] = _bh_fdr(out.loc[idx, "within_p"])
    return out.sort_values(["q_bh_primary_family", "q_bh_within_roi_set_response", "within_p"], na_position="last")


def build_qc(long: pd.DataFrame, behavior_subjects_path: Path | None = None) -> pd.DataFrame:
    neural_subjects = sorted(long["subject"].astype(str).unique().tolist()) if not long.empty else []
    rows: list[dict[str, object]] = [
        {
            "scope": "retrieval_neural_item_metrics",
            "n_subjects": len(neural_subjects),
            "subjects": ",".join(neural_subjects),
            "n_rows": int(len(long)),
        }
    ]
    if behavior_subjects_path is not None and behavior_subjects_path.exists():
        behavior = _read_table(behavior_subjects_path)
        if "subject" in behavior.columns:
            behavior_subjects = sorted(behavior["subject"].astype(str).unique().tolist())
            inter = sorted(set(neural_subjects) & set(behavior_subjects))
            rows.extend(
                [
                    {
                        "scope": "behavior_summary",
                        "n_subjects": len(behavior_subjects),
                        "subjects": ",".join(behavior_subjects),
                        "n_rows": int(len(behavior)),
                    },
                    {
                        "scope": "neural_behavior_intersection",
                        "n_subjects": len(inter),
                        "subjects": ",".join(inter),
                        "n_rows": int(long["subject"].isin(inter).sum()),
                    },
                    {
                        "scope": "behavior_only_missing_neural",
                        "n_subjects": len(sorted(set(behavior_subjects) - set(neural_subjects))),
                        "subjects": ",".join(sorted(set(behavior_subjects) - set(neural_subjects))),
                        "n_rows": 0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Run-7 retrieval success prediction from retrieval geometry.")
    parser.add_argument(
        "--retrieval-item-metrics",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "retrieval_geometry" / "retrieval_item_metrics.tsv",
    )
    parser.add_argument(
        "--retrieval-pair-metrics",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "retrieval_geometry" / "retrieval_pair_metrics.tsv",
    )
    parser.add_argument(
        "--behavior-summary",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "behavior_results" / "refined" / "subject_summary_wide.tsv",
    )
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    args = parser.parse_args()

    output_dir = ensure_dir(args.paper_output_root / "qc" / "retrieval_success_prediction")
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    item_metrics = _read_table(args.retrieval_item_metrics)
    item_metrics = _merge_pair_metrics(item_metrics, args.retrieval_pair_metrics)
    long = _prepare_long(item_metrics)
    qc = build_qc(long, args.behavior_summary)
    summary = summarize_models(long)

    write_table(long, output_dir / "retrieval_success_item_long.tsv")
    write_table(qc, output_dir / "retrieval_success_join_qc.tsv")
    write_table(summary, output_dir / "retrieval_success_group_fdr.tsv")
    write_table(summary, tables_main / "table_retrieval_success_prediction.tsv")
    write_table(summary, tables_si / "table_retrieval_success_prediction_fdr.tsv")
    save_json(
        {
            "n_item_metric_rows": int(len(item_metrics)),
            "n_long_rows": int(len(long)),
            "n_model_rows": int(len(summary)),
            "primary_metrics": sorted(PRIMARY_METRICS),
        },
        output_dir / "retrieval_success_manifest.json",
    )


if __name__ == "__main__":
    main()
