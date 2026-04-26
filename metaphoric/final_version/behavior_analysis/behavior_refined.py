#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_refined.py

Purpose
- Refine run-7 behavior analysis around the current neural story.
- Keep memory accuracy as the primary behavior endpoint.
- Treat RT as a supplementary endpoint, focusing on correct-trial RT.
- Export subject-level YY-KJ difference tables for downstream brain-behavior analysis.

Default inputs
- `${METAPHOR_DATA_EVENTS:-${PYTHON_METAPHOR_ROOT}/data_events}/sub-xx/sub-xx_run-7_events.tsv`

Default outputs
- `${METAPHOR_BEHAVIOR_RESULTS:-${PYTHON_METAPHOR_ROOT}/behavior_results}/refined/`
  - `behavior_trials.tsv`
  - `subject_condition_summary.tsv`
  - `subject_summary_wide.tsv`
  - `item_condition_summary.tsv`
  - `analysis_manifest.json`
  - optional `lmm/run7_memory/`
  - optional `lmm/run7_log_rt_correct/`
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd


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
from behavior_analysis.behavior_lmm import (  # noqa: E402
    DATA_DIR,
    FILE_PATTERN,
    OUT_DIR,
    collect_paths,
    fit_mixed_model,
    prepare_trials,
)


def _safe_log(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric <= 0 or math.isnan(numeric):
        return None
    return math.log(numeric)


def build_refined_trials(trials: pd.DataFrame) -> pd.DataFrame:
    frame = trials.copy()
    if "memory" in frame.columns:
        frame["memory"] = pd.to_numeric(frame["memory"], errors="coerce")
    if "action_time" in frame.columns:
        frame["action_time"] = pd.to_numeric(frame["action_time"], errors="coerce")
    frame["is_correct"] = frame["memory"].eq(1) if "memory" in frame.columns else False
    frame["rt_correct"] = frame["action_time"].where(frame["is_correct"]) if "action_time" in frame.columns else pd.NA
    frame["log_action_time"] = frame["action_time"].map(_safe_log) if "action_time" in frame.columns else pd.NA
    frame["log_rt_correct"] = frame["rt_correct"].map(_safe_log) if "action_time" in frame.columns else pd.NA
    return frame


def summarize_subject_condition(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["subject", "condition"]
    for (subject, condition), group in frame.groupby(group_cols):
        memory_series = pd.to_numeric(group["memory"], errors="coerce")
        rt_series = pd.to_numeric(group["action_time"], errors="coerce")
        rt_correct = pd.to_numeric(group["rt_correct"], errors="coerce")
        accuracy = float(memory_series.mean()) if memory_series.notna().any() else float("nan")
        mean_rt_all = float(rt_series.mean()) if rt_series.notna().any() else float("nan")
        mean_rt_correct = float(rt_correct.mean()) if rt_correct.notna().any() else float("nan")
        median_rt_correct = float(rt_correct.median()) if rt_correct.notna().any() else float("nan")
        inverse_efficiency = float(mean_rt_correct / accuracy) if accuracy and not math.isnan(mean_rt_correct) else float("nan")
        rows.append(
            {
                "subject": subject,
                "condition": condition,
                "n_trials": int(len(group)),
                "n_correct": int(group["is_correct"].sum()),
                "accuracy": accuracy,
                "mean_rt_all": mean_rt_all,
                "mean_rt_correct": mean_rt_correct,
                "median_rt_correct": median_rt_correct,
                "inverse_efficiency": inverse_efficiency,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_subject_wide(subject_condition: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_trials",
        "n_correct",
        "accuracy",
        "mean_rt_all",
        "mean_rt_correct",
        "median_rt_correct",
        "inverse_efficiency",
    ]
    wide = subject_condition.pivot(index="subject", columns="condition", values=metrics)
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()

    diff_pairs = [
        ("accuracy", "behavior_primary_diff_yy_minus_kj"),
        ("mean_rt_correct", "rt_correct_diff_yy_minus_kj"),
        ("inverse_efficiency", "ies_diff_yy_minus_kj"),
    ]
    for metric, diff_name in diff_pairs:
        yy_col = f"{metric}_YY"
        kj_col = f"{metric}_KJ"
        if yy_col in wide.columns and kj_col in wide.columns:
            wide[diff_name] = wide[yy_col] - wide[kj_col]
    return wide.sort_values("subject").reset_index(drop=True)


def summarize_item_condition(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition, item), group in frame.groupby(["condition", "item"]):
        memory_series = pd.to_numeric(group["memory"], errors="coerce")
        rt_correct = pd.to_numeric(group["rt_correct"], errors="coerce")
        rows.append(
            {
                "condition": condition,
                "item": item,
                "n_trials": int(len(group)),
                "n_subjects": int(group["subject"].nunique()),
                "accuracy": float(memory_series.mean()) if memory_series.notna().any() else float("nan"),
                "mean_rt_correct": float(rt_correct.mean()) if rt_correct.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "item"]).reset_index(drop=True)


def export_lmm(frame: pd.DataFrame, response_col: str, output_dir: Path) -> dict[str, object]:
    from behavior_analysis.behavior_lmm import export_outputs  # noqa: E402

    model_frame = frame[["subject", "condition", "item", response_col]].copy()
    model_frame[response_col] = pd.to_numeric(model_frame[response_col], errors="coerce")
    model_frame = model_frame.dropna(subset=[response_col])
    if model_frame.empty:
        return {"ran": False, "reason": f"no_non_missing_{response_col}"}

    fit, fit_strategy, formula = fit_mixed_model(model_frame, response_col)
    export_outputs(
        fit,
        output_dir,
        formula=formula,
        response_label=response_col,
        response_col=response_col,
        fit_strategy=fit_strategy,
        model_frame=model_frame,
    )
    return {
        "ran": True,
        "response": response_col,
        "formula": formula,
        "fit_strategy": fit_strategy,
        "n_obs": int(fit.nobs),
        "n_subjects": int(model_frame["subject"].nunique()),
        "n_items": int(model_frame["item"].nunique()),
        "converged": bool(getattr(fit, "converged", True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refined run-7 behavior analysis for metaphor project.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing run-7 events tables.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR / "refined", help="Output directory.")
    parser.add_argument("--pattern", default=FILE_PATTERN, help="Glob pattern used to find run-7 event files.")
    parser.add_argument(
        "--skip-lmm",
        action="store_true",
        help="Only export summary tables without fitting refined LMMs.",
    )
    args = parser.parse_args()

    paths = collect_paths(args.data_dir, args.pattern)
    if not paths:
        raise FileNotFoundError(f"No files matching '{args.pattern}' were found under {args.data_dir}.")

    output_dir = ensure_dir(args.output_dir)
    trials = prepare_trials(paths)
    refined = build_refined_trials(trials)
    write_table(refined, output_dir / "behavior_trials.tsv")

    subject_condition = summarize_subject_condition(refined)
    write_table(subject_condition, output_dir / "subject_condition_summary.tsv")

    subject_wide = summarize_subject_wide(subject_condition)
    write_table(subject_wide, output_dir / "subject_summary_wide.tsv")

    item_condition = summarize_item_condition(refined)
    write_table(item_condition, output_dir / "item_condition_summary.tsv")

    lmm_outputs: dict[str, object] = {}
    if not args.skip_lmm:
        lmm_outputs["memory"] = export_lmm(refined, "memory", output_dir / "lmm" / "run7_memory")
        lmm_outputs["log_rt_correct"] = export_lmm(refined, "log_rt_correct", output_dir / "lmm" / "run7_log_rt_correct")

    save_json(
        {
            "data_dir": str(args.data_dir),
            "pattern": args.pattern,
            "n_subjects": int(refined["subject"].nunique()),
            "n_trials": int(len(refined)),
            "primary_behavior_endpoint": "memory",
            "supplementary_behavior_endpoint": "log_rt_correct",
            "brain_behavior_primary_subject_column": "behavior_primary_diff_yy_minus_kj",
            "brain_behavior_supplementary_subject_column": "rt_correct_diff_yy_minus_kj",
            "lmm_outputs": lmm_outputs,
        },
        output_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
