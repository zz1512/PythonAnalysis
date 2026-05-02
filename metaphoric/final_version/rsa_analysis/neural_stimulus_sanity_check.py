#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_stimulus_sanity_check.py

Lightweight neural sanity check for S0 stimulus covariates.

This script does not recompute neural patterns or replace the primary RSA LMM.
It reads existing ROI-level itemwise RSA outputs, collapses each subject/ROI/pair
to post-minus-pre similarity change, and asks whether the Metaphor-vs-Spatial
delta-similarity contrast keeps the same direction after adding pair-level
stimulus covariates.
"""

from __future__ import annotations

import argparse
import math
import os
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
DEFAULT_STIMULUS_CONTROL = ROOT / "paper_outputs" / "tables_si" / "table_stimulus_control_s0.tsv"
DEFAULT_TABLES_SI = ROOT / "paper_outputs" / "tables_si"
DEFAULT_QC_DIR = ROOT / "paper_outputs" / "qc" / "neural_stimulus_sanity_check"
ROI_SETS = ("main_functional", "literature", "literature_spatial")

PRIMARY_COVARIATES = (
    "sentence_char_count",
    "pair_word_frequency_mean",
    "pair_arousal_mean",
    "embedding_cosine_distance",
)
PRIMARY_PLUS_VALENCE = (*PRIMARY_COVARIATES, "pair_valence_mean")


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _itemwise_path(roi_set: str) -> Path:
    return ROOT / "paper_outputs" / "qc" / f"rsa_results_optimized_{roi_set}" / "rsa_itemwise_details.csv"


def _inference_family(roi_set: str, roi: str) -> str:
    if roi_set != "main_functional":
        return roi_set
    if roi.startswith("func_Metaphor_gt_Spatial"):
        return "main_functional_metaphor_gt_spatial"
    if roi.startswith("func_Spatial_gt_Metaphor"):
        return "main_functional_spatial_gt_metaphor"
    return "main_functional_other"


def _pair_key(first_label: object, second_label: object) -> str:
    labels = [str(first_label).strip(), str(second_label).strip()]
    return "||".join(sorted(labels))


def load_pair_covariates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame = frame.loc[frame["condition"].isin(["Metaphor", "Spatial"])].copy()
    required = ["condition", "source_id", "first_word_label", "second_word_label", *PRIMARY_PLUS_VALENCE]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    pair_cov = frame.loc[:, required].copy()
    pair_cov["pair_key"] = [
        _pair_key(first, second)
        for first, second in zip(pair_cov["first_word_label"], pair_cov["second_word_label"])
    ]
    duplicates = pair_cov.duplicated(subset=["condition", "pair_key"], keep=False)
    if duplicates.any():
        examples = pair_cov.loc[duplicates, ["condition", "pair_key"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate S0 condition/pair_key rows: {examples}")
    return pair_cov


def load_roi_delta(roi_set: str, pair_covariates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = _itemwise_path(roi_set)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame = frame.loc[frame["condition"].isin(["Metaphor", "Spatial"])].copy()
    frame["similarity"] = pd.to_numeric(frame["similarity"], errors="coerce")
    frame["stage"] = frame["stage"].astype(str).str.lower()
    pivot = frame.pivot_table(
        index=["subject", "roi", "condition", "pair_id", "word_label", "partner_label"],
        columns="stage",
        values="similarity",
        aggfunc="mean",
    ).reset_index()
    if "pre" not in pivot.columns or "post" not in pivot.columns:
        raise ValueError(f"{path} must contain both Pre and Post stage rows.")
    pivot["delta_similarity_post_minus_pre"] = pivot["post"] - pivot["pre"]
    pivot["roi_set"] = roi_set
    pivot["inference_family"] = [
        _inference_family(roi_set, str(roi)) for roi in pivot["roi"]
    ]
    pivot["pair_id"] = pd.to_numeric(pivot["pair_id"], errors="coerce").astype("Int64")
    pivot["pair_key"] = [
        _pair_key(first, second)
        for first, second in zip(pivot["word_label"], pivot["partner_label"])
    ]
    merged = pivot.merge(pair_covariates, on=["condition", "pair_key"], how="left", validate="many_to_one")
    coverage = pd.DataFrame(
        [
            {
                "roi_set": roi_set,
                "source_file": str(path),
                "n_delta_rows": int(len(merged)),
                "n_subjects": int(merged["subject"].nunique()),
                "n_rois": int(merged["roi"].nunique()),
                "n_pairs": int(merged[["condition", "pair_id"]].drop_duplicates().shape[0]),
                "n_rows_with_s0_match": int(merged["sentence_char_count"].notna().sum()),
                "match_rate": float(merged["sentence_char_count"].notna().mean()),
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


def fit_gee(frame: pd.DataFrame, covariates: tuple[str, ...], *, adjusted: bool):
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Gaussian
    from statsmodels.genmod.generalized_estimating_equations import GEE

    condition_term = 'C(condition, Treatment(reference="Spatial"))'
    terms = [condition_term]
    if adjusted:
        terms.extend([f"z_{covariate}" for covariate in covariates])
    formula = "delta_similarity_post_minus_pre ~ " + " + ".join(terms)
    model = GEE.from_formula(
        formula,
        groups="subject",
        cov_struct=Exchangeable(),
        family=Gaussian(),
        data=frame,
    )
    fit = model.fit()
    return fit, formula


def extract_condition_row(fit, *, formula: str, model_name: str, frame: pd.DataFrame) -> dict[str, object]:
    term = 'C(condition, Treatment(reference="Spatial"))[T.Metaphor]'
    conf_int = fit.conf_int()
    return {
        "model": model_name,
        "formula": formula,
        "term": term,
        "estimate": _safe_float(fit.params.get(term)),
        "std_error": _safe_float(getattr(fit, "bse", pd.Series()).get(term)),
        "z_value": _safe_float(getattr(fit, "tvalues", pd.Series()).get(term)),
        "p_value": _safe_float(getattr(fit, "pvalues", pd.Series()).get(term)),
        "ci_low": _safe_float(conf_int.loc[term, 0]) if term in conf_int.index else float("nan"),
        "ci_high": _safe_float(conf_int.loc[term, 1]) if term in conf_int.index else float("nan"),
        "n_obs": int(getattr(fit, "nobs", len(frame))),
        "n_subjects": int(frame["subject"].nunique()),
        "n_pairs": int(frame[["condition", "pair_id"]].drop_duplicates().shape[0]),
    }


def run_roi_models(frame: pd.DataFrame, covariates: tuple[str, ...], set_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = ["subject", "condition", "pair_id", "delta_similarity_post_minus_pre", *covariates]
    for (roi_set, roi), group in frame.groupby(["roi_set", "roi"], sort=True):
        model_frame = group.loc[:, required].copy()
        model_frame["delta_similarity_post_minus_pre"] = pd.to_numeric(
            model_frame["delta_similarity_post_minus_pre"],
            errors="coerce",
        )
        model_frame = model_frame.dropna().copy()
        if model_frame.empty:
            continue
        model_frame = standardize_columns(model_frame, covariates)
        for adjusted, model_name in [
            (False, f"unadjusted_matched_{set_name}"),
            (True, f"adjusted_{set_name}"),
        ]:
            fit, formula = fit_gee(model_frame, covariates, adjusted=adjusted)
            row = extract_condition_row(fit, formula=formula, model_name=model_name, frame=model_frame)
            row.update(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "inference_family": _inference_family(str(roi_set), str(roi)),
                    "covariate_set": set_name,
                    "adjusted": bool(adjusted),
                    "covariates": " + ".join(covariates) if adjusted else "",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_pairwise_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "inference_family", "covariate_set"]
    for keys, group in summary.groupby(group_cols, dropna=False):
        unadjusted = group.loc[~group["adjusted"]]
        adjusted = group.loc[group["adjusted"]]
        if unadjusted.empty or adjusted.empty:
            continue
        unadj = unadjusted.iloc[0]
        adj = adjusted.iloc[0]
        unadj_est = _safe_float(unadj["estimate"])
        adj_est = _safe_float(adj["estimate"])
        rows.append(
            {
                "roi_set": keys[0],
                "roi": keys[1],
                "inference_family": keys[2],
                "covariate_set": keys[3],
                "unadjusted_estimate": unadj_est,
                "unadjusted_p": _safe_float(unadj["p_value"]),
                "adjusted_estimate": adj_est,
                "adjusted_p": _safe_float(adj["p_value"]),
                "direction_preserved": bool(np.sign(unadj_est) == np.sign(adj_est)),
                "absolute_estimate_change": abs(adj_est - unadj_est),
                "relative_estimate_change": (
                    abs(adj_est - unadj_est) / abs(unadj_est)
                    if math.isfinite(unadj_est) and not math.isclose(unadj_est, 0.0)
                    else float("nan")
                ),
                "n_obs": int(adj["n_obs"]),
                "n_subjects": int(adj["n_subjects"]),
                "n_pairs": int(adj["n_pairs"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight neural S0 covariate sanity check.")
    parser.add_argument("--stimulus-control", type=Path, default=DEFAULT_STIMULUS_CONTROL)
    parser.add_argument("--tables-si-dir", type=Path, default=DEFAULT_TABLES_SI)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--roi-sets", nargs="+", default=list(ROI_SETS), choices=list(ROI_SETS))
    args = parser.parse_args()

    pair_covariates = load_pair_covariates(args.stimulus_control)
    delta_frames: list[pd.DataFrame] = []
    coverage_tables: list[pd.DataFrame] = []
    for roi_set in args.roi_sets:
        delta, coverage = load_roi_delta(roi_set, pair_covariates)
        delta_frames.append(delta)
        coverage_tables.append(coverage)

    qc_dir = ensure_dir(args.qc_dir)
    tables_si_dir = ensure_dir(args.tables_si_dir)
    delta_all = pd.concat(delta_frames, ignore_index=True)
    coverage_all = pd.concat(coverage_tables, ignore_index=True)
    write_table(delta_all, qc_dir / "neural_delta_with_s0_covariates.tsv")
    write_table(coverage_all, qc_dir / "neural_s0_match_coverage.tsv")

    summaries = [
        run_roi_models(delta_all, PRIMARY_COVARIATES, "primary_pair_s0"),
        run_roi_models(delta_all, PRIMARY_PLUS_VALENCE, "primary_pair_s0_plus_valence"),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    comparison = add_pairwise_comparison(summary)

    write_table(summary, tables_si_dir / "table_neural_stimulus_sanity_check.tsv")
    write_table(comparison, tables_si_dir / "table_neural_stimulus_sanity_check_comparison.tsv")
    save_json(
        {
            "stimulus_control": str(args.stimulus_control),
            "roi_sets": list(args.roi_sets),
            "outputs": {
                "summary": str(tables_si_dir / "table_neural_stimulus_sanity_check.tsv"),
                "comparison": str(tables_si_dir / "table_neural_stimulus_sanity_check_comparison.tsv"),
                "merged_delta": str(qc_dir / "neural_delta_with_s0_covariates.tsv"),
                "coverage": str(qc_dir / "neural_s0_match_coverage.tsv"),
            },
            "primary_covariates": list(PRIMARY_COVARIATES),
            "primary_plus_valence_covariates": list(PRIMARY_PLUS_VALENCE),
            "model_note": "GEE sanity check clustered by subject on ROI pair-level post-minus-pre similarity; primary RSA LMM remains unchanged.",
        },
        qc_dir / "neural_stimulus_sanity_check_manifest.json",
    )


if __name__ == "__main__":
    main()
