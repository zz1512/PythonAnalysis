#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check behavior branch coverage, ISC inclusion, and brain-behavior joint results")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--behavior-data-dir", type=Path, default=None)
    p.add_argument("--trial-dir-name", type=str, default="by_stimulus")
    p.add_argument("--emotion-dir-name", type=str, default="by_emotion")
    p.add_argument("--trial-brain-repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument("--emotion-brain-repr-prefix", type=str, default="roi_repr_matrix_232_emotion4")
    p.add_argument("--trial-beh-repr-prefix", type=str, default="behavior_repr_matrix_trial")
    p.add_argument("--emotion-beh-repr-prefix", type=str, default="behavior_repr_matrix_emotion4")
    p.add_argument("--trial-beh-isc-prefix", type=str, default="behavior_isc_mahalanobis_by_age")
    p.add_argument("--emotion-beh-isc-prefix", type=str, default="behavior_isc_mahalanobis_by_age")
    p.add_argument("--trial-score-file", type=str, default="behavior_subject_stimulus_scores.csv")
    p.add_argument("--emotion-score-file", type=str, default="behavior_subject_emotion_scores.csv")
    p.add_argument("--coverage-file", type=str, default="behavior_subject_coverage.csv")
    p.add_argument("--joint-file", type=str, default="roi_isc_behavior_perm_fwer.csv")
    return p.parse_args()


def _safe_read_csv(path: Path, col: str) -> List[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"{path} missing column: {col}")
    return df[col].astype(str).tolist()


def _safe_read_subject_set(path: Path, col: str = "subject") -> Set[str]:
    if not path.exists():
        return set()
    return set(_safe_read_csv(path, col))


def _age_sorted_ok(ages: np.ndarray) -> bool:
    a = np.asarray(ages, dtype=float).reshape(-1)
    finite = np.isfinite(a)
    if int(finite.sum()) <= 1:
        return True
    af = a[finite]
    return bool(np.all(np.diff(af) >= -1e-8))


def _summarize_coverage(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, object] = {"coverage_file_ok": True}
    if "subject" in df.columns:
        out["coverage_n_subjects"] = int(df["subject"].astype(str).nunique())
    if "include_for_behavior" in df.columns:
        inc = df["include_for_behavior"].astype(bool)
        out["coverage_n_include"] = int(inc.sum())
        out["coverage_n_excluded"] = int((~inc).sum())
    if "complete_for_behavior" in df.columns:
        out["coverage_n_complete"] = int(df["complete_for_behavior"].astype(bool).sum())
    if "n_stimuli_present" in df.columns:
        vals = df["n_stimuli_present"].to_numpy(dtype=float)
        out["coverage_present_min"] = float(np.nanmin(vals)) if vals.size else float("nan")
        out["coverage_present_median"] = float(np.nanmedian(vals)) if vals.size else float("nan")
        out["coverage_present_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
        out["coverage_present_max"] = float(np.nanmax(vals)) if vals.size else float("nan")
        out["coverage_n_zero_present"] = int(np.sum(vals <= 0))
    return out


def _summarize_score_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, object] = {"score_file_ok": True, "score_n_rows": int(df.shape[0])}
    if "subject" in df.columns:
        out["score_n_subjects"] = int(df["subject"].astype(str).nunique())
    if "stimulus_content" in df.columns:
        out["score_n_conditions"] = int(df["stimulus_content"].astype(str).nunique())
    elif "emotion" in df.columns:
        out["score_n_conditions"] = int(df["emotion"].astype(str).nunique())

    value_cols = [c for c in df.columns if c not in {"subject", "stimulus_content", "emotion", "task", "raw_condition", "raw_emotion", "n_trials_agg", "n_stimuli_agg", "all_features_finite", "all_features_missing"}]
    if "all_features_finite" in df.columns:
        out["score_n_all_features_finite"] = int(df["all_features_finite"].astype(bool).sum())
    if "all_features_missing" in df.columns:
        out["score_n_all_features_missing"] = int(df["all_features_missing"].astype(bool).sum())
    elif value_cols:
        arr = df[value_cols].to_numpy(dtype=float)
        out["score_n_all_features_missing"] = int(np.sum(np.all(~np.isfinite(arr), axis=1)))
    return out


def _summarize_repr_subjects(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    subs = _safe_read_csv(path, "subject")
    return {"beh_repr_file_ok": True, "n_subjects_behavior_repr": int(len(subs))}


def _summarize_behavior_isc(stim_dir: Path, isc_prefix: str) -> Dict[str, object]:
    npy_path = stim_dir / f"{isc_prefix}.npy"
    subs_path = stim_dir / f"{isc_prefix}_subjects_sorted.csv"
    meta_path = stim_dir / f"{isc_prefix}_meta.csv"
    if not (npy_path.exists() and subs_path.exists()):
        return {}
    isc = np.load(npy_path)
    subs = pd.read_csv(subs_path)
    out: Dict[str, object] = {
        "beh_isc_files_ok": True,
        "beh_isc_shape": str(tuple(isc.shape)),
        "n_subjects_behavior_isc": int(subs.shape[0]),
        "beh_isc_nan_ratio": float(np.mean(~np.isfinite(isc))),
    }
    if "age" in subs.columns:
        ages = subs["age"].to_numpy(dtype=float)
        out["beh_isc_age_sorted_ok"] = bool(_age_sorted_ok(ages))
        out["beh_isc_age_min"] = float(np.nanmin(ages)) if ages.size else float("nan")
        out["beh_isc_age_max"] = float(np.nanmax(ages)) if ages.size else float("nan")
    if "missing_fraction_repr" in subs.columns:
        miss = subs["missing_fraction_repr"].to_numpy(dtype=float)
        out["beh_repr_missing_frac_mean"] = float(np.nanmean(miss)) if miss.size else float("nan")
        out["beh_repr_missing_frac_median"] = float(np.nanmedian(miss)) if miss.size else float("nan")
        out["beh_repr_missing_frac_max"] = float(np.nanmax(miss)) if miss.size else float("nan")
        out["beh_repr_n_subjects_missing_gt0"] = int(np.sum(miss > 0))
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "missing_fill" in meta.columns:
            out["beh_isc_missing_fill"] = str(meta.loc[0, "missing_fill"])
    return out


def _summarize_joint_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, object] = {"joint_file_ok": True, "joint_n_rows": int(df.shape[0])}
    if "roi" in df.columns:
        out["joint_n_rois"] = int(df["roi"].astype(str).nunique())
    if {"r_obs", "p_fwer_model_wise"}.issubset(df.columns):
        out["joint_n_sig_pos_fwer"] = int(((df["r_obs"] > 0) & (df["p_fwer_model_wise"] <= 0.05)).sum())
    if {"r_obs", "p_fdr_bh_global"}.issubset(df.columns):
        out["joint_n_sig_pos_fdr"] = int(((df["r_obs"] > 0) & (df["p_fdr_bh_global"] <= 0.05)).sum())
    return out


def _load_coverage_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).copy()
    if "subject" not in df.columns:
        return pd.DataFrame()
    df["subject"] = df["subject"].astype(str)
    return df


def _load_behavior_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).copy()
    if "subject" not in df.columns:
        return pd.DataFrame()
    df["subject"] = df["subject"].astype(str)
    return df


def _collect_brain_subject_union(
    stim_root: Path,
    brain_repr_prefix: str,
) -> Set[str]:
    out: Set[str] = set()
    if not stim_root.exists():
        return out
    for stim_dir in sorted([p for p in stim_root.iterdir() if p.is_dir()]):
        out |= _safe_read_subject_set(stim_dir / f"{brain_repr_prefix}_subjects.csv")
    return out


def _build_branch_detail_rows(
    branch: str,
    stim_dir: Path,
    brain_subs: Sequence[str],
    beh_subs: Sequence[str],
    coverage_df: Optional[pd.DataFrame] = None,
    isc_subs_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    coverage_df = coverage_df if coverage_df is not None else pd.DataFrame()
    coverage_map = coverage_df.set_index("subject") if (not coverage_df.empty and "subject" in coverage_df.columns) else None
    beh_set = set(beh_subs)
    isc_set = _safe_read_subject_set(isc_subs_path) if isc_subs_path is not None else set()

    rows: List[Dict[str, object]] = []
    for subject in sorted(set(brain_subs)):
        row: Dict[str, object] = {
            "branch": branch,
            "stimulus_type": stim_dir.name,
            "subject": subject,
            "in_brain_subjects": True,
            "in_behavior_repr_subjects": bool(subject in beh_set),
            "in_behavior_isc_subjects": bool(subject in isc_set),
            "brain_missing_in_behavior_repr": bool(subject not in beh_set),
        }
        if coverage_map is not None and subject in coverage_map.index:
            cov = coverage_map.loc[subject]
            if isinstance(cov, pd.DataFrame):
                cov = cov.iloc[0]
            for col in ["include_for_behavior", "complete_for_behavior", "n_stimuli_total", "n_stimuli_present"]:
                if col in coverage_map.columns:
                    row[col] = cov[col]
        rows.append(row)
    return rows


def _build_er_vs_brain_rows(
    brain_trial_subjects: Set[str],
    brain_emotion_subjects: Set[str],
    audit_df: pd.DataFrame,
    er_df: Optional[pd.DataFrame] = None,
    feature_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    all_subjects = sorted(brain_trial_subjects | brain_emotion_subjects)
    if not all_subjects:
        return pd.DataFrame()

    audit_map = audit_df.set_index("subject") if (not audit_df.empty and "subject" in audit_df.columns) else None
    er_subs = set(er_df["participant_id"].astype(str).unique()) if (er_df is not None and not er_df.empty and "participant_id" in er_df.columns) else set()
    feat_subs = set(feature_df["subject"].astype(str).unique()) if (feature_df is not None and not feature_df.empty and "subject" in feature_df.columns) else set()

    rows: List[Dict[str, object]] = []
    for subject in all_subjects:
        row: Dict[str, object] = {
            "subject": subject,
            "in_any_brain_trial": bool(subject in brain_trial_subjects),
            "in_any_brain_emotion": bool(subject in brain_emotion_subjects),
            "in_any_brain": True,
            "in_data_4_hddm_ER": bool(subject in er_subs),
            "in_behavior_feature_table": bool(subject in feat_subs),
        }
        if audit_map is not None and subject in audit_map.index:
            audit_row = audit_map.loc[subject]
            if isinstance(audit_row, pd.DataFrame):
                audit_row = audit_row.iloc[0]
            for col in audit_map.columns:
                row[col] = audit_row[col]
        else:
            row["listed_in_participants"] = False
            row["included_in_data_4_hddm_ER"] = False
            row["er_exclusion_reason"] = "not_in_behavior_participants_list"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    behavior_data_dir = args.behavior_data_dir or (args.matrix_dir / "behavior")

    branches = [
        {
            "branch": "trial",
            "stim_root": args.matrix_dir / str(args.trial_dir_name),
            "brain_repr_prefix": str(args.trial_brain_repr_prefix),
            "beh_repr_prefix": str(args.trial_beh_repr_prefix),
            "beh_isc_prefix": str(args.trial_beh_isc_prefix),
            "score_file": str(args.trial_score_file),
            "coverage_file": str(args.coverage_file),
        },
        {
            "branch": "emotion",
            "stim_root": args.matrix_dir / str(args.emotion_dir_name),
            "brain_repr_prefix": str(args.emotion_brain_repr_prefix),
            "beh_repr_prefix": str(args.emotion_beh_repr_prefix),
            "beh_isc_prefix": str(args.emotion_beh_isc_prefix),
            "score_file": str(args.emotion_score_file),
            "coverage_file": "",
        },
    ]

    brain_subject_unions: Dict[str, Set[str]] = {}
    for cfg in branches:
        stim_root = Path(cfg["stim_root"])
        if not stim_root.exists():
            continue
        brain_subject_unions[str(cfg["branch"])] = _collect_brain_subject_union(
            stim_root=stim_root,
            brain_repr_prefix=str(cfg["brain_repr_prefix"]),
        )
        for stim_dir in sorted([p for p in stim_root.iterdir() if p.is_dir()]):
            row: Dict[str, object] = {"branch": str(cfg["branch"]), "stimulus_type": stim_dir.name}
            try:
                brain_subs_path = stim_dir / f"{cfg['brain_repr_prefix']}_subjects.csv"
                beh_subs_path = stim_dir / f"{cfg['beh_repr_prefix']}_subjects.csv"
                row["brain_subjects_file_ok"] = bool(brain_subs_path.exists())
                if brain_subs_path.exists():
                    brain_subs = _safe_read_csv(brain_subs_path, "subject")
                    row["n_subjects_brain"] = int(len(brain_subs))
                else:
                    brain_subs = []

                row.update(_summarize_repr_subjects(beh_subs_path))
                beh_subs = _safe_read_csv(beh_subs_path, "subject") if beh_subs_path.exists() else []
                coverage_df = _load_coverage_map(stim_dir / str(cfg["coverage_file"])) if str(cfg["coverage_file"]) else pd.DataFrame()
                if brain_subs:
                    row["subjects_behavior_subset_brain"] = bool(set(beh_subs).issubset(set(brain_subs)))
                    row["n_subjects_overlap_brain_behavior"] = int(len(set(beh_subs) & set(brain_subs)))
                    row["n_subjects_brain_not_in_behavior"] = int(len(set(brain_subs) - set(beh_subs)))
                    row["subjects_missing_in_behavior_repr"] = "|".join(sorted(set(brain_subs) - set(beh_subs)))

                if str(cfg["coverage_file"]):
                    row.update(_summarize_coverage(stim_dir / str(cfg["coverage_file"])))
                row.update(_summarize_score_file(stim_dir / str(cfg["score_file"])))
                row.update(_summarize_behavior_isc(stim_dir, isc_prefix=str(cfg["beh_isc_prefix"])))
                row.update(_summarize_joint_file(stim_dir / str(args.joint_file)))
                detail_rows.extend(
                    _build_branch_detail_rows(
                        branch=str(cfg["branch"]),
                        stim_dir=stim_dir,
                        brain_subs=brain_subs,
                        beh_subs=beh_subs,
                        coverage_df=coverage_df,
                        isc_subs_path=stim_dir / f"{cfg['beh_isc_prefix']}_subjects_sorted.csv",
                    )
                )
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["branch", "stimulus_type"])
    out_path = args.matrix_dir / "data_check_behavior_branch.csv"
    out.to_csv(out_path, index=False)
    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["branch", "stimulus_type", "subject"])
    detail_path = args.matrix_dir / "data_check_behavior_branch_subjects.csv"
    detail_df.to_csv(detail_path, index=False)

    audit_df = _load_behavior_audit(behavior_data_dir / "data_4_hddm_ER_subject_audit.csv")
    er_df = pd.read_csv(behavior_data_dir / "data_4_hddm_ER.csv") if (behavior_data_dir / "data_4_hddm_ER.csv").exists() else pd.DataFrame()
    feature_df = pd.read_csv(behavior_data_dir / "data_4_behavior_feature_table.csv") if (behavior_data_dir / "data_4_behavior_feature_table.csv").exists() else pd.DataFrame()
    er_vs_brain_df = _build_er_vs_brain_rows(
        brain_trial_subjects=brain_subject_unions.get("trial", set()),
        brain_emotion_subjects=brain_subject_unions.get("emotion", set()),
        audit_df=audit_df,
        er_df=er_df,
        feature_df=feature_df,
    )
    er_vs_brain_path = args.matrix_dir / "data_check_behavior_er_vs_brain_subjects.csv"
    er_vs_brain_df.to_csv(er_vs_brain_path, index=False)

    print(out.to_string(index=False))
    if not er_vs_brain_df.empty:
        missing_df = er_vs_brain_df.loc[~er_vs_brain_df["in_data_4_hddm_ER"].astype(bool)].copy()
        print("\nSubjects in brain data but missing from data_4_hddm_ER:")
        if missing_df.empty:
            print("None")
        else:
            cols = [c for c in ["subject", "er_exclusion_reason", "valid_subject_er", "has_task_txt", "has_task_tsv", "n_trials_valid_emot"] if c in missing_df.columns]
            print(missing_df.loc[:, cols].to_string(index=False))


if __name__ == "__main__":
    main()
