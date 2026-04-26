#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_lmm.py

Purpose
- Run a trial-level mixed-effects model on run-7 behavior data.
- Match the same default data layout used by getTimeActionScore.py:
  `${METAPHOR_DATA_EVENTS:-${PYTHON_METAPHOR_ROOT}/data_events}/sub-xx/sub-xx_run-7_events.tsv`
- When launched without arguments, automatically scan run-7 files and fit
  all available default responses (`memory`, `action_time`).

Outputs
- `${METAPHOR_BEHAVIOR_RESULTS:-${PYTHON_METAPHOR_ROOT}/behavior_results}/lmm/run7_<response>/`
  - `behavior_lmm_summary.txt`
  - `behavior_lmm_params.tsv`
  - `behavior_lmm_model.json`
"""

from __future__ import annotations

import argparse
import glob
import os
import re
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

from common.final_utils import ensure_dir, save_json, write_table


_ROOT = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
DATA_DIR = Path(os.environ.get("METAPHOR_DATA_EVENTS", str(_ROOT / "data_events")))
OUT_DIR = Path(os.environ.get("METAPHOR_BEHAVIOR_RESULTS", str(_ROOT / "behavior_results")))
FILE_PATTERN = "*run-7_events.tsv"

YY_SET = {"yyw", "yyew"}
KJ_SET = {"kjw", "kjew"}
DEFAULT_RESPONSES = ("memory", "action_time")


def smart_read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def get_sub_id_from_name(path: str | Path) -> str:
    base = Path(path).name
    match = re.search(r"sub[-_]?(\d+)", base, flags=re.IGNORECASE)
    if match:
        return f"sub-{match.group(1).zfill(2)}"
    match = re.search(r"(\d+)", base)
    if match:
        return f"sub-{match.group(1).zfill(2)}"
    return Path(path).stem


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    match = re.fullmatch(r"sub[-_]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"sub-{match.group(1).zfill(2)}"
    if text.isdigit():
        return f"sub-{text.zfill(2)}"
    fallback = get_sub_id_from_name(text)
    return fallback or text


def collect_paths(data_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(data_dir / "**" / pattern), recursive=True))
    if not paths:
        paths = sorted(Path(p) for p in glob.glob(str(data_dir / pattern)))
    return paths


def derive_item_column(frame: pd.DataFrame) -> pd.Series:
    if "pic_out" in frame.columns:
        return frame["pic_out"].astype(str)
    if {"trial_type", "pic_num"}.issubset(frame.columns):
        return frame["trial_type"].astype(str) + "_" + frame["pic_num"].astype(str)
    if "pic_num" in frame.columns:
        return frame["pic_num"].astype(str)
    return pd.Series([f"trial_{idx:04d}" for idx in range(len(frame))], index=frame.index)


def prepare_trials(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, path in enumerate(paths, start=1):
        frame = smart_read(path)
        if "trial_type" not in frame.columns:
            raise ValueError(f"{path} is missing required column 'trial_type'.")

        frame = frame.copy()
        frame["condition"] = frame["trial_type"].map(
            lambda value: "YY"
            if value in YY_SET
            else ("KJ" if value in KJ_SET else "OTHER")
        )
        frame = frame.loc[frame["condition"].isin({"YY", "KJ"})].copy()
        if frame.empty:
            print(f"[{idx:2d}/{len(paths)}] {path.name}: skipped (no YY/KJ trials)")
            continue

        sub_id = (
            normalize_subject_id(frame["subject"].iloc[0])
            if "subject" in frame.columns and frame["subject"].notna().any()
            else get_sub_id_from_name(path)
        )
        if not sub_id:
            sub_id = get_sub_id_from_name(path)
        frame["subject"] = sub_id
        frame["item"] = derive_item_column(frame)
        frame["source_file"] = str(path)
        frames.append(frame)
        print(f"[{idx:2d}/{len(paths)}] {sub_id}: loaded {len(frame)} YY/KJ trials")

    if not frames:
        raise RuntimeError("No valid run-7 YY/KJ trials were found.")
    return pd.concat(frames, ignore_index=True)


def resolve_response_column(frame: pd.DataFrame, requested: str) -> tuple[str, str]:
    aliases = {
        "auto": [],
        "memory": ["memory", "accuracy"],
        "accuracy": ["accuracy", "memory"],
        "action_time": ["action_time"],
    }
    if requested not in aliases:
        return requested, requested
    for candidate in aliases[requested]:
        if candidate in frame.columns:
            return requested, candidate
    raise ValueError(
        f"Requested response '{requested}' was not found. "
        f"Available columns include: {', '.join(frame.columns)}"
    )


def choose_response_specs(frame: pd.DataFrame, requested: str) -> list[tuple[str, str]]:
    if requested != "auto":
        return [resolve_response_column(frame, requested)]

    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in DEFAULT_RESPONSES:
        try:
            label, actual = resolve_response_column(frame, name)
        except ValueError:
            continue
        if actual not in seen:
            specs.append((label, actual))
            seen.add(actual)
    if not specs:
        raise ValueError(
            "No default response columns were found. "
            "Expected one of: memory, action_time, accuracy."
        )
    return specs


def build_model_frame(trials: pd.DataFrame, response_col: str) -> pd.DataFrame:
    model_frame = trials[["subject", "condition", "item", response_col]].copy()
    model_frame[response_col] = pd.to_numeric(model_frame[response_col], errors="coerce")
    model_frame = model_frame.dropna(subset=[response_col])
    if model_frame.empty:
        raise RuntimeError(f"No non-missing data found for response '{response_col}'.")
    return model_frame


def fit_mixed_model(model_frame: pd.DataFrame, response_col: str):
    import statsmodels.formula.api as smf

    formula = f"{response_col} ~ C(condition)"
    fit_errors: list[str] = []
    attempts = []

    if model_frame["item"].nunique() > 1:
        attempts.append(
            {
                "label": "subject_intercept_plus_item_vc",
                "kwargs": {"vc_formula": {"item": "0 + C(item)"}, "re_formula": "1"},
            }
        )

    attempts.append({"label": "subject_intercept_only", "kwargs": {"re_formula": "1"}})

    for attempt in attempts:
        try:
            model = smf.mixedlm(
                formula,
                data=model_frame,
                groups=model_frame["subject"],
                **attempt["kwargs"],
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            return fit, attempt["label"], formula
        except Exception as exc:
            fit_errors.append(f"{attempt['label']}: {exc}")

    raise RuntimeError(" ; ".join(fit_errors))


def export_outputs(
    fit,
    output_dir: Path,
    *,
    formula: str,
    response_label: str,
    response_col: str,
    fit_strategy: str,
    model_frame: pd.DataFrame,
) -> None:
    output_dir = ensure_dir(output_dir)
    (output_dir / "behavior_lmm_summary.txt").write_text(str(fit.summary()), encoding="utf-8")

    params = pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.values,
            "std_error": fit.bse.reindex(fit.params.index).values,
            "z_value": fit.tvalues.reindex(fit.params.index).values,
            "p_value": fit.pvalues.reindex(fit.params.index).values,
        }
    )
    write_table(params, output_dir / "behavior_lmm_params.tsv")

    save_json(
        {
            "formula": formula,
            "response_label": response_label,
            "response_column": response_col,
            "fit_strategy": fit_strategy,
            "aic": None if pd.isna(fit.aic) else float(fit.aic),
            "bic": None if pd.isna(fit.bic) else float(fit.bic),
            "log_likelihood": float(fit.llf),
            "n_obs": int(fit.nobs),
            "n_subjects": int(model_frame["subject"].nunique()),
            "n_items": int(model_frame["item"].nunique()),
        },
        output_dir / "behavior_lmm_model.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-effects modeling for run-7 behavior data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing run-7 events tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "lmm",
        help="Base directory for model outputs.",
    )
    parser.add_argument(
        "--pattern",
        default=FILE_PATTERN,
        help="Glob pattern used to find run-7 events files.",
    )
    parser.add_argument(
        "--response",
        default="auto",
        choices=["auto", "memory", "accuracy", "action_time"],
        help="Response column to analyze. Default runs all available standard responses.",
    )
    args = parser.parse_args()

    paths = collect_paths(args.data_dir, args.pattern)
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{args.pattern}' were found under {args.data_dir}."
        )

    print(f"Found {len(paths)} run-7 event files under {args.data_dir}")
    trials = prepare_trials(paths)
    response_specs = choose_response_specs(trials, args.response)

    for response_label, response_col in response_specs:
        print(f"\n=== Fitting LMM for {response_label} (column: {response_col}) ===")
        model_frame = build_model_frame(trials, response_col)
        fit, fit_strategy, formula = fit_mixed_model(model_frame, response_col)
        output_dir = args.output_dir / f"run7_{response_label}"
        export_outputs(
            fit,
            output_dir,
            formula=formula,
            response_label=response_label,
            response_col=response_col,
            fit_strategy=fit_strategy,
            model_frame=model_frame,
        )
        print(f"Saved outputs to {output_dir}")
        print(
            f"n_obs={int(fit.nobs)}, n_subjects={model_frame['subject'].nunique()}, "
            f"n_items={model_frame['item'].nunique()}, strategy={fit_strategy}"
        )


if __name__ == "__main__":
    main()
