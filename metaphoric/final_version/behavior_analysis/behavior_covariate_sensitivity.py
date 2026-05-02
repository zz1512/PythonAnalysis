#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_covariate_sensitivity.py

Add a lightweight stimulus-covariate sensitivity analysis for the refined
run-7 behavior result. This script does not replace the original behavior
outputs. It exports adjusted tables that can be reported alongside the
unadjusted primary behavior model.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
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
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402


ROOT = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
DEFAULT_BEHAVIOR_TRIALS = (
    ROOT / "paper_outputs" / "qc" / "behavior_results" / "refined" / "behavior_trials.tsv"
)
DEFAULT_STIMULUS_CONTROL = ROOT / "paper_outputs" / "tables_si" / "table_stimulus_control_s0.tsv"
DEFAULT_TABLES_SI = ROOT / "paper_outputs" / "tables_si"
DEFAULT_QC_DIR = ROOT / "paper_outputs" / "qc" / "behavior_results" / "refined" / "covariate_sensitivity"

PRIMARY_COVARIATES = ("sentence_char_count", "word_frequency", "word_arousal")
PRIMARY_PLUS_VALENCE = (*PRIMARY_COVARIATES, "word_valence")


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _trial_word_label(row: pd.Series) -> str | None:
    trial_type = str(row.get("trial_type", "")).strip().lower()
    pic_num = row.get("pic_num")
    if not trial_type or pd.isna(pic_num):
        return None
    try:
        pic_int = int(float(pic_num))
    except Exception:
        text = str(pic_num).strip()
        match = re.search(r"\d+", text)
        if not match:
            return None
        pic_int = int(match.group(0))
    return f"{trial_type}_{pic_int}"


def build_word_covariate_table(stimulus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    role_specs = [
        ("first", "first_word_label", "first_word"),
        ("second", "second_word_label", "second_word"),
    ]
    for _, row in stimulus.iterrows():
        for role, label_col, word_col in role_specs:
            label = row.get(label_col)
            if pd.isna(label) or not str(label).strip():
                continue
            rows.append(
                {
                    "word_label": str(label).strip(),
                    "s0_condition": row.get("condition"),
                    "source_id": row.get("source_id"),
                    "pair_id_s0": row.get("pair_id_s0"),
                    "word_role": role,
                    "word": row.get(word_col),
                    "sentence_char_count": row.get("sentence_char_count"),
                    "embedding_cosine_distance": row.get("embedding_cosine_distance"),
                    "pair_word_frequency_mean": row.get("pair_word_frequency_mean"),
                    "pair_arousal_mean": row.get("pair_arousal_mean"),
                    "pair_valence_mean": row.get("pair_valence_mean"),
                    "word_frequency": row.get(f"{role}_word_frequency"),
                    "word_stroke_count": row.get(f"{role}_stroke_count"),
                    "word_valence": row.get(f"{role}_valence"),
                    "word_arousal": row.get(f"{role}_arousal"),
                }
            )
    covariates = pd.DataFrame(rows)
    if covariates.empty:
        raise RuntimeError("No word-level covariates could be derived from the S0 stimulus table.")
    duplicate_labels = covariates.loc[covariates["word_label"].duplicated(keep=False), "word_label"].unique()
    if len(duplicate_labels):
        examples = ", ".join(sorted(str(label) for label in duplicate_labels[:10]))
        raise ValueError(f"S0 stimulus table contains duplicate word_label values: {examples}")
    return covariates.reset_index(drop=True)


def load_analysis_frame(behavior_trials: Path, stimulus_control: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    behavior = pd.read_csv(behavior_trials, sep="\t")
    stimulus = pd.read_csv(stimulus_control, sep="\t")
    covariates = build_word_covariate_table(stimulus)

    frame = behavior.copy()
    frame["word_label"] = frame.apply(_trial_word_label, axis=1)
    merged = frame.merge(covariates, on="word_label", how="left", validate="many_to_one")

    numeric_cols = [
        "memory",
        "log_rt_correct",
        "sentence_char_count",
        "embedding_cosine_distance",
        "pair_word_frequency_mean",
        "pair_arousal_mean",
        "pair_valence_mean",
        "word_frequency",
        "word_stroke_count",
        "word_valence",
        "word_arousal",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    coverage = pd.DataFrame(
        [
            {
                "n_behavior_trials": int(len(merged)),
                "n_trials_with_word_label": int(merged["word_label"].notna().sum()),
                "n_trials_with_s0_match": int(merged["s0_condition"].notna().sum()),
                "match_rate": float(merged["s0_condition"].notna().mean()),
                "n_unique_behavior_words": int(merged["word_label"].nunique()),
                "n_unique_matched_words": int(merged.loc[merged["s0_condition"].notna(), "word_label"].nunique()),
            }
        ]
    )
    return merged, coverage


def standardize_columns(frame: pd.DataFrame, covariates: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for covariate in covariates:
        values = pd.to_numeric(result[covariate], errors="coerce")
        mean = values.mean()
        std = values.std(ddof=1)
        z_col = f"z_{covariate}"
        if pd.isna(std) or math.isclose(float(std), 0.0):
            result[z_col] = 0.0
        else:
            result[z_col] = (values - mean) / std
    return result


def fit_gee(frame: pd.DataFrame, response: str, covariates: tuple[str, ...], *, adjusted: bool):
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial, Gaussian
    from statsmodels.genmod.generalized_estimating_equations import GEE

    family = Binomial() if response == "memory" else Gaussian()
    condition_term = 'C(condition, Treatment(reference="KJ"))'
    terms = [condition_term]
    if adjusted:
        terms.extend([f"z_{covariate}" for covariate in covariates])
    formula = f"{response} ~ " + " + ".join(terms)
    model = GEE.from_formula(
        formula,
        groups="subject",
        cov_struct=Exchangeable(),
        family=family,
        data=frame,
    )
    fit = model.fit()
    return fit, formula, family.__class__.__name__


def _extract_params(fit, *, response: str, model_name: str, formula: str, family: str, frame: pd.DataFrame) -> pd.DataFrame:
    conf_int = fit.conf_int()
    rows: list[dict[str, object]] = []
    for term in fit.params.index:
        estimate = _safe_float(fit.params.get(term))
        low = _safe_float(conf_int.loc[term, 0]) if term in conf_int.index else float("nan")
        high = _safe_float(conf_int.loc[term, 1]) if term in conf_int.index else float("nan")
        row = {
            "response": response,
            "model": model_name,
            "formula": formula,
            "family": family,
            "term": term,
            "estimate": estimate,
            "std_error": _safe_float(getattr(fit, "bse", pd.Series()).get(term)),
            "z_value": _safe_float(getattr(fit, "tvalues", pd.Series()).get(term)),
            "p_value": _safe_float(getattr(fit, "pvalues", pd.Series()).get(term)),
            "ci_low": low,
            "ci_high": high,
            "n_obs": int(getattr(fit, "nobs", len(frame))),
            "n_subjects": int(frame["subject"].nunique()),
            "n_items": int(frame["item"].nunique()),
        }
        if response == "memory":
            row["odds_ratio"] = float(np.exp(estimate)) if math.isfinite(estimate) else float("nan")
            row["odds_ratio_ci_low"] = float(np.exp(low)) if math.isfinite(low) else float("nan")
            row["odds_ratio_ci_high"] = float(np.exp(high)) if math.isfinite(high) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run_model_set(
    frame: pd.DataFrame,
    *,
    response: str,
    covariates: tuple[str, ...],
    set_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["subject", "condition", "item", response, *covariates]
    model_frame = frame.loc[:, required].copy()
    model_frame[response] = pd.to_numeric(model_frame[response], errors="coerce")
    model_frame = model_frame.dropna().copy()
    model_frame = model_frame.loc[model_frame["condition"].isin(["YY", "KJ"])].copy()
    model_frame = standardize_columns(model_frame, covariates)

    param_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for adjusted, model_name in [
        (False, f"unadjusted_matched_{set_name}"),
        (True, f"adjusted_{set_name}"),
    ]:
        fit, formula, family = fit_gee(model_frame, response, covariates, adjusted=adjusted)
        params = _extract_params(
            fit,
            response=response,
            model_name=model_name,
            formula=formula,
            family=family,
            frame=model_frame,
        )
        param_tables.append(params)
        condition_rows = params[params["term"].str.contains(r"\[T\.YY\]", regex=True, na=False)]
        if condition_rows.empty:
            condition_row = params.iloc[0].copy()
        else:
            condition_row = condition_rows.iloc[0].copy()
        summary = condition_row.to_dict()
        summary["covariate_set"] = set_name
        summary["adjusted"] = bool(adjusted)
        summary["covariates"] = " + ".join(covariates) if adjusted else ""
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows), pd.concat(param_tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stimulus-covariate sensitivity models for refined run-7 behavior."
    )
    parser.add_argument("--behavior-trials", type=Path, default=DEFAULT_BEHAVIOR_TRIALS)
    parser.add_argument("--stimulus-control", type=Path, default=DEFAULT_STIMULUS_CONTROL)
    parser.add_argument("--tables-si-dir", type=Path, default=DEFAULT_TABLES_SI)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    args = parser.parse_args()

    frame, coverage = load_analysis_frame(args.behavior_trials, args.stimulus_control)
    qc_dir = ensure_dir(args.qc_dir)
    tables_si_dir = ensure_dir(args.tables_si_dir)

    write_table(frame, qc_dir / "behavior_trials_with_s0_covariates.tsv")
    write_table(coverage, qc_dir / "behavior_covariate_match_coverage.tsv")

    summary_tables: list[pd.DataFrame] = []
    param_tables: list[pd.DataFrame] = []
    for response in ["memory", "log_rt_correct"]:
        for covariates, set_name in [
            (PRIMARY_COVARIATES, "primary_s0"),
            (PRIMARY_PLUS_VALENCE, "primary_s0_plus_valence"),
        ]:
            summary, params = run_model_set(
                frame,
                response=response,
                covariates=covariates,
                set_name=set_name,
            )
            summary_tables.append(summary)
            param_tables.append(params)

    summary = pd.concat(summary_tables, ignore_index=True)
    params = pd.concat(param_tables, ignore_index=True)
    write_table(summary, tables_si_dir / "table_behavior_covariate_sensitivity.tsv")
    write_table(params, tables_si_dir / "table_behavior_covariate_sensitivity_params.tsv")
    save_json(
        {
            "behavior_trials": str(args.behavior_trials),
            "stimulus_control": str(args.stimulus_control),
            "outputs": {
                "summary": str(tables_si_dir / "table_behavior_covariate_sensitivity.tsv"),
                "params": str(tables_si_dir / "table_behavior_covariate_sensitivity_params.tsv"),
                "merged_trials": str(qc_dir / "behavior_trials_with_s0_covariates.tsv"),
                "coverage": str(qc_dir / "behavior_covariate_match_coverage.tsv"),
            },
            "primary_covariates": list(PRIMARY_COVARIATES),
            "primary_plus_valence_covariates": list(PRIMARY_PLUS_VALENCE),
            "model_note": "GEE sensitivity models clustered by subject; primary behavior output remains unchanged.",
        },
        qc_dir / "behavior_covariate_sensitivity_manifest.json",
    )


if __name__ == "__main__":
    main()
