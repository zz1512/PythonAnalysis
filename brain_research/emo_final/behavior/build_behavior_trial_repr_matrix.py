#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from behavior_data import DEFAULT_BEH_DATA_DIR, DEFAULT_FMRI_DATA_DIR, collect_behavior_feature_table  # noqa: E402
    from behavior_matrix_utils import compute_difference_matrix, save_npz_named  # noqa: E402
    from ..calc_roi_isc_by_age import distance_to_similarity  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_FEATURE_COLS = ("emot_rating",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build trial-level behavior matrices aligned to emo_final stimulus order")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--brain-repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument("--beh-data-dir", type=Path, default=DEFAULT_BEH_DATA_DIR)
    p.add_argument("--fmri-data-dir", type=Path, default=DEFAULT_FMRI_DATA_DIR)
    p.add_argument("--participants-file", type=Path, default=None)
    p.add_argument("--valid-er-file", type=Path, default=None)
    p.add_argument("--valid-tg-file", type=Path, default=None)
    p.add_argument("--feature-cols", nargs="+", default=list(DEFAULT_FEATURE_COLS))
    p.add_argument("--agg-func", type=str, default="mean", choices=("mean", "median"))
    p.add_argument("--diff-method", type=str, default="euclidean", choices=("euclidean", "cityblock", "manhattan", "mahalanobis"))
    p.add_argument("--score-file-name", type=str, default="behavior_subject_stimulus_scores.csv")
    p.add_argument("--pattern-prefix", type=str, default="behavior_patterns_trial")
    p.add_argument("--diff-prefix", type=str, default="behavior_diff_matrix_trial")
    p.add_argument("--repr-prefix", type=str, default="behavior_repr_matrix_trial")
    return p.parse_args()


def _mode_or_first(series: pd.Series) -> str:
    valid = series.dropna().astype(str)
    if valid.empty:
        return ""
    mode = valid.mode()
    return str(mode.iloc[0]) if not mode.empty else str(valid.iloc[0])


def _detect_required_validity(feature_cols: Sequence[str]) -> Dict[str, bool]:
    cols = {str(c) for c in feature_cols}
    return {
        "need_er": any(c.startswith("emot_") for c in cols),
        "need_tg": any(c.startswith("choice_") or c == "return_score" for c in cols),
    }


def _check_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    missing = [str(c) for c in feature_cols if str(c) not in df.columns]
    if missing:
        raise ValueError(f"Behavior feature table missing columns: {missing}")
    return [str(c) for c in feature_cols]


def _agg_name(agg_func: str):
    if str(agg_func).strip().lower() == "mean":
        return np.nanmean
    if str(agg_func).strip().lower() == "median":
        return np.nanmedian
    raise ValueError(f"Unsupported agg func: {agg_func}")


def _load_branch_plan(by_stim: Path, brain_repr_prefix: str) -> List[Tuple[Path, List[str], List[str]]]:
    plans = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        subjects_path = stim_dir / f"{brain_repr_prefix}_subjects.csv"
        order_path = stim_dir / "stimulus_order.csv"
        if (not subjects_path.exists()) or (not order_path.exists()):
            continue
        subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
        stim_order = pd.read_csv(order_path)["stimulus_order"].astype(str).tolist()
        if subjects and stim_order:
            plans.append((stim_dir, subjects, stim_order))
    return plans


def _build_one_stimulus(
    feature_df: pd.DataFrame,
    brain_subjects: Sequence[str],
    stim_order: Sequence[str],
    feature_cols: Sequence[str],
    agg_func: str,
    diff_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    feature_cols = [str(c) for c in feature_cols]
    agg = _agg_name(agg_func)
    brain_subjects = [str(s) for s in brain_subjects]
    stim_order = [str(s) for s in stim_order]

    df = feature_df[
        feature_df["subject"].astype(str).isin(brain_subjects)
        & feature_df["stimulus_content"].astype(str).isin(stim_order)
    ].copy()
    if df.empty:
        empty_cov = pd.DataFrame({"subject": brain_subjects, "n_stimuli_present": 0, "n_stimuli_expected": len(stim_order), "complete_for_behavior": False})
        return pd.DataFrame(), empty_cov, {}, {}, {}, []

    meta = (
        df.groupby(["subject", "stimulus_content"], as_index=False)
        .agg(
            task=("task", _mode_or_first),
            raw_condition=("raw_condition", _mode_or_first),
            raw_emotion=("raw_emotion", _mode_or_first),
            n_trials_agg=("subject", "size"),
        )
    )

    agg_df = df.groupby(["subject", "stimulus_content"], as_index=False)[feature_cols].agg(agg).reset_index(drop=True)
    long_scores = meta.merge(agg_df, on=["subject", "stimulus_content"], how="outer")
    long_scores["all_features_finite"] = long_scores[feature_cols].apply(lambda row: bool(np.isfinite(row.to_numpy(dtype=float)).all()), axis=1)

    coverage_rows = []
    complete_subjects: List[str] = []
    for sub in brain_subjects:
        sub_df = long_scores[long_scores["subject"].astype(str) == str(sub)].copy()
        present = sub_df.loc[sub_df["all_features_finite"], "stimulus_content"].astype(str).nunique()
        complete = int(present) == int(len(stim_order))
        coverage_rows.append(
            {
                "subject": str(sub),
                "n_stimuli_present": int(present),
                "n_stimuli_expected": int(len(stim_order)),
                "complete_for_behavior": bool(complete),
            }
        )
        if complete:
            complete_subjects.append(str(sub))

    coverage_df = pd.DataFrame(coverage_rows)
    if not complete_subjects:
        return long_scores, coverage_df, {}, {}, {}, []

    pattern_by_sub: Dict[str, np.ndarray] = {}
    diff_by_sub: Dict[str, np.ndarray] = {}
    repr_by_sub: Dict[str, np.ndarray] = {}
    keep_long_rows = []

    for sub in complete_subjects:
        sub_df = long_scores[long_scores["subject"].astype(str) == str(sub)].copy()
        sub_df = sub_df.set_index("stimulus_content").loc[stim_order].reset_index()
        keep_long_rows.append(sub_df)
        X = sub_df[feature_cols].to_numpy(dtype=np.float32)
        D = compute_difference_matrix(X, method=str(diff_method)).astype(np.float32)
        S = distance_to_similarity(D).astype(np.float32)
        pattern_by_sub[str(sub)] = X
        diff_by_sub[str(sub)] = D
        repr_by_sub[str(sub)] = S

    long_scores_keep = pd.concat(keep_long_rows, ignore_index=True) if keep_long_rows else pd.DataFrame()
    return long_scores_keep, coverage_df, pattern_by_sub, diff_by_sub, repr_by_sub, complete_subjects


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    brain_repr_prefix: str,
    beh_data_dir: Path,
    fmri_data_dir: Path,
    participants_file: Optional[Path],
    valid_er_file: Optional[Path],
    valid_tg_file: Optional[Path],
    feature_cols: Sequence[str],
    agg_func: str,
    diff_method: str,
    score_file_name: str,
    pattern_prefix: str,
    diff_prefix: str,
    repr_prefix: str,
) -> None:
    matrix_dir = Path(matrix_dir)
    by_stim = matrix_dir / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    plans = _load_branch_plan(by_stim, brain_repr_prefix=str(brain_repr_prefix))
    if not plans:
        raise FileNotFoundError(f"No stimulus directories with {brain_repr_prefix}_subjects.csv found under {by_stim}")

    union_subjects = sorted({sub for _, subjects, _ in plans for sub in subjects})
    feature_df = collect_behavior_feature_table(
        subjects=union_subjects,
        dir_beh_data=beh_data_dir,
        dir_fmri_data=fmri_data_dir,
        participants_file=participants_file,
        valid_er_file=valid_er_file,
        valid_tg_file=valid_tg_file,
    )
    feature_cols = _check_feature_columns(feature_df, feature_cols)
    validity_policy = _detect_required_validity(feature_cols)

    if validity_policy["need_er"]:
        feature_df = feature_df[feature_df["valid_subject_er"].astype(bool)].copy()
    if validity_policy["need_tg"]:
        feature_df = feature_df[feature_df["valid_subject_tg"].astype(bool)].copy()

    summary_rows = []
    for stim_dir, brain_subjects, stim_order in plans:
        long_scores, coverage_df, pattern_by_sub, diff_by_sub, repr_by_sub, beh_subjects = _build_one_stimulus(
            feature_df=feature_df,
            brain_subjects=brain_subjects,
            stim_order=stim_order,
            feature_cols=feature_cols,
            agg_func=agg_func,
            diff_method=diff_method,
        )

        coverage_df.to_csv(stim_dir / "behavior_subject_coverage.csv", index=False)
        if long_scores.empty or not beh_subjects:
            summary_rows.append(
                {
                    "stimulus_type": stim_dir.name,
                    "n_subjects_brain": int(len(brain_subjects)),
                    "n_subjects_behavior": 0,
                    "n_stimuli": int(len(stim_order)),
                    "feature_cols": "|".join(feature_cols),
                    "agg_func": str(agg_func),
                    "diff_method": str(diff_method),
                }
            )
            continue

        long_scores.to_csv(stim_dir / str(score_file_name), index=False)
        save_npz_named(stim_dir / f"{pattern_prefix}.npz", pattern_by_sub)
        save_npz_named(stim_dir / f"{diff_prefix}.npz", diff_by_sub)
        save_npz_named(stim_dir / f"{repr_prefix}.npz", repr_by_sub)
        pd.DataFrame({"subject": beh_subjects}).to_csv(stim_dir / f"{pattern_prefix}_subjects.csv", index=False)
        pd.DataFrame({"subject": beh_subjects}).to_csv(stim_dir / f"{diff_prefix}_subjects.csv", index=False)
        pd.DataFrame({"subject": beh_subjects}).to_csv(stim_dir / f"{repr_prefix}_subjects.csv", index=False)
        pd.DataFrame({"feature": feature_cols}).to_csv(stim_dir / f"{pattern_prefix}_features.csv", index=False)
        pd.DataFrame({"feature": feature_cols}).to_csv(stim_dir / f"{diff_prefix}_features.csv", index=False)
        pd.DataFrame({"feature": feature_cols}).to_csv(stim_dir / f"{repr_prefix}_features.csv", index=False)
        meta_df = pd.DataFrame(
            {
                "repr_level": ["trial"],
                "n_subjects": [int(len(beh_subjects))],
                "n_stimuli": [int(len(stim_order))],
                "n_features": [int(len(feature_cols))],
                "feature_cols": ["|".join(feature_cols)],
                "agg_func": [str(agg_func)],
                "diff_method": [str(diff_method)],
                "brain_repr_prefix": [str(brain_repr_prefix)],
            }
        )
        meta_df.to_csv(stim_dir / f"{pattern_prefix}_meta.csv", index=False)
        meta_df.to_csv(stim_dir / f"{diff_prefix}_meta.csv", index=False)
        meta_df.to_csv(stim_dir / f"{repr_prefix}_meta.csv", index=False)

        summary_rows.append(
            {
                "stimulus_type": stim_dir.name,
                "n_subjects_brain": int(len(brain_subjects)),
                "n_subjects_behavior": int(len(beh_subjects)),
                "n_stimuli": int(len(stim_order)),
                "feature_cols": "|".join(feature_cols),
                "agg_func": str(agg_func),
                "diff_method": str(diff_method),
            }
        )

    pd.DataFrame(summary_rows).sort_values("stimulus_type").to_csv(matrix_dir / f"{repr_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=args.matrix_dir,
        stimulus_dir_name=str(args.stimulus_dir_name),
        brain_repr_prefix=str(args.brain_repr_prefix),
        beh_data_dir=args.beh_data_dir,
        fmri_data_dir=args.fmri_data_dir,
        participants_file=args.participants_file,
        valid_er_file=args.valid_er_file,
        valid_tg_file=args.valid_tg_file,
        feature_cols=[str(x) for x in args.feature_cols],
        agg_func=str(args.agg_func),
        diff_method=str(args.diff_method),
        score_file_name=str(args.score_file_name),
        pattern_prefix=str(args.pattern_prefix),
        diff_prefix=str(args.diff_prefix),
        repr_prefix=str(args.repr_prefix),
    )


if __name__ == "__main__":
    main()
