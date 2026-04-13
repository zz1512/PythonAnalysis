#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_maturity_index.py

Step 4:
计算 Representational Maturity Index，将每位被试的 RSM 模式与
成熟模板（17–18 岁）比较，并关联行为评分。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from calc_roi_isc_by_age import flatten_upper, load_subject_ages_map, sort_subjects_by_age  # noqa: E402
from joint_analysis_roi_isc_dev_models import bh_fdr  # noqa: E402
from utils_network_assignment import get_roi_network_map, get_scan_cortical_rois, get_schaefer200_network_labels, get_tian_s2_labels  # noqa: E402

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: Representational Maturity Index")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--condition", type=str, default="Reappraisal")
    p.add_argument("--repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument(
        "--subject-info",
        type=Path,
        default=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
    )
    p.add_argument("--behavior-file", type=Path, default=None)
    p.add_argument("--behavior-cols", nargs="+", default=["emot_rating"])
    p.add_argument(
        "--behavior-join",
        type=str,
        default="left",
        choices=["left", "inner"],
        help="行为表与 MI 表合并方式。left: 不丢弃被试（推荐）；inner: 仅保留有行为数据的被试。",
    )
    p.add_argument("--mature-age-min", type=float, default=17.0)
    # Optional: run maturity-threshold sensitivity analysis (does not change the main --mature-age-min run)
    p.add_argument(
        "--mature-age-grid",
        nargs="+",
        type=float,
        default=None,
        help="Sensitivity grid for mature-age threshold, e.g. 16.5 17 17.5 18.0",
    )
    # Optional: bootstrap the mature template to quantify MI stability
    p.add_argument(
        "--bootstrap-template",
        type=int,
        default=0,
        help="Bootstrap times B for mature template stability (0 disables).",
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Random seed for template bootstrap (defaults to --seed).",
    )
    # Optional: leave-one-out (LOO) influence of each mature subject on the template
    p.add_argument(
        "--loo-template",
        action="store_true",
        default=False,
        help="Run a minimal leave-one-out influence analysis for mature template.",
    )
    p.add_argument("--target-networks", nargs="+", default=["Default", "DorsAttn"])
    p.add_argument("--include-subcortical", nargs="+", default=["Hippocampus"])
    p.add_argument("--include-scan", action="store_true", default=False)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _warn(msg: str) -> None:
    print(f"[maturity_index][warn] {msg}", file=sys.stderr)


def _find_subject_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("subject", "sub_id", "被试编号"):
        if c in df.columns:
            return c
    return None


def _normalize_subject_ids(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    return s.apply(lambda x: x if x.startswith("sub-") else f"sub-{x}")


def _pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_condition_col(df: pd.DataFrame) -> Optional[str]:
    # Common naming across behavior exports
    candidates = [
        "condition",
        "cond",
        "trial_type",
        "trialtype",
        "task_condition",
        "task",
        "regulation",
    ]
    return _pick_first_existing_col(df, candidates)


def _infer_stimulus_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "stimulus",
        "stim",
        "stim_id",
        "stimulus_id",
        "item",
        "trial",
        "trial_id",
        "image",
        "picture",
        "movie",
    ]
    return _pick_first_existing_col(df, candidates)


def _infer_rating_col(df: pd.DataFrame) -> Optional[str]:
    # Default behavior pipeline uses emot_rating; keep that as highest priority.
    candidates = [
        "emot_rating",
        "rating",
        "emotion_rating",
        "emo_rating",
        "valence_rating",
        "score",
    ]
    return _pick_first_existing_col(df, candidates)


def _is_likely_rating_like_col(col_name: str) -> bool:
    name = str(col_name).strip().lower()
    keywords = ("rating", "emot", "emotion", "score", "valence", "affect")
    return any(k in name for k in keywords)


def _map_condition_to_two_levels(raw: pd.Series) -> pd.Series:
    # Map raw condition names to {passive, reappraisal}; keep unknown as NaN.
    def _norm(x: object) -> str:
        return str(x).strip().lower()

    vals = raw.apply(_norm)
    out = pd.Series(np.full(len(vals), np.nan, dtype=object), index=vals.index)

    passive_keys = ("passive", "passive_emo", "watch", "view", "natural")
    reapp_keys = ("reappraisal", "reappraise", "regulate", "reapp")

    for k in passive_keys:
        out[vals.str.contains(k, na=False)] = "passive"
    for k in reapp_keys:
        out[vals.str.contains(k, na=False)] = "reappraisal"
    return out


def _try_compute_reappraisal_success(
    beh_df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    """
    Try to compute subject-level reappraisal_success from the *loaded* behavior table.

    Preferred definition:
      reappraisal_success = rating_passive - rating_reappraisal

    Returns:
      (df_success, meta)
    where df_success has columns: subject, reappraisal_success, reappraisal_success_n_pairs (optional)
    and meta contains status/method/reason for logging/output.
    """
    meta: Dict[str, object] = {
        "status": "not_computed",
        "method": "",
        "reason": "",
        "rating_col": "",
        "condition_col": "",
        "stimulus_col": "",
    }

    sub_col = _find_subject_col(beh_df)
    if sub_col is None:
        meta["reason"] = "missing_subject_col (need one of: subject/sub_id/被试编号)"
        return None, meta

    df = beh_df.copy()
    df[sub_col] = _normalize_subject_ids(df[sub_col])

    # 1) Wide format: detect *passive / *reappraisal paired columns (case-insensitive, suffix-based).
    cols_lower = {c.lower(): c for c in df.columns}
    # Common explicit pairs
    explicit_pairs = [
        ("rating_passive", "rating_reappraisal"),
        ("emot_rating_passive", "emot_rating_reappraisal"),
        ("passive", "reappraisal"),
    ]
    for a_l, b_l in explicit_pairs:
        if a_l in cols_lower and b_l in cols_lower:
            col_a = cols_lower[a_l]
            col_b = cols_lower[b_l]
            a = pd.to_numeric(df[col_a], errors="coerce")
            b = pd.to_numeric(df[col_b], errors="coerce")
            df_s = pd.DataFrame({
                "subject": df[sub_col],
                "reappraisal_success": (a - b).astype(np.float64),
            })
            df_s = df_s.groupby("subject", as_index=False)["reappraisal_success"].mean()
            meta.update({
                "status": "ok",
                "method": "wide_columns_mean",
                "reason": "",
                "rating_col": f"{col_a}|{col_b}",
                "condition_col": "",
                "stimulus_col": "",
            })
            return df_s, meta

    # Suffix pattern: <base>_passive and <base>_reappraisal
    lower_names = [c.lower() for c in df.columns]
    for c in lower_names:
        if c.endswith("_passive"):
            base = c[: -len("_passive")]
            c2 = base + "_reappraisal"
            if c2 in cols_lower:
                col_a = cols_lower[c]
                col_b = cols_lower[c2]
                if not (_is_likely_rating_like_col(col_a) or _is_likely_rating_like_col(col_b)):
                    continue
                a = pd.to_numeric(df[col_a], errors="coerce")
                b = pd.to_numeric(df[col_b], errors="coerce")
                df_s = pd.DataFrame({
                    "subject": df[sub_col],
                    "reappraisal_success": (a - b).astype(np.float64),
                })
                df_s = df_s.groupby("subject", as_index=False)["reappraisal_success"].mean()
                meta.update({
                    "status": "ok",
                    "method": "wide_suffix_mean",
                    "reason": "",
                    "rating_col": f"{col_a}|{col_b}",
                    "condition_col": "",
                    "stimulus_col": "",
                })
                return df_s, meta

    # 2) Long format: use condition column + rating column.
    cond_col = _infer_condition_col(df)
    rating_col = _infer_rating_col(df)
    if cond_col is None:
        meta["reason"] = "missing_condition_col (need e.g. condition/cond/trial_type)"
        return None, meta
    if rating_col is None:
        meta["reason"] = "missing_rating_col (need e.g. emot_rating/rating)"
        meta["condition_col"] = str(cond_col)
        return None, meta

    df["_cond2"] = _map_condition_to_two_levels(df[cond_col])
    df["_rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    has_passive = bool((df["_cond2"] == "passive").any())
    has_reapp = bool((df["_cond2"] == "reappraisal").any())
    if not (has_passive and has_reapp):
        meta.update({
            "reason": "condition_levels_incomplete (need both passive and reappraisal in same table)",
            "rating_col": str(rating_col),
            "condition_col": str(cond_col),
        })
        return None, meta

    stim_col = _infer_stimulus_col(df)
    meta.update({
        "rating_col": str(rating_col),
        "condition_col": str(cond_col),
        "stimulus_col": str(stim_col) if stim_col is not None else "",
    })

    # Preferred: pair by (subject, stimulus) if stimulus is available.
    if stim_col is not None:
        pivot = (
            df.dropna(subset=["_cond2"])
            .pivot_table(
                index=[sub_col, stim_col],
                columns="_cond2",
                values="_rating",
                aggfunc="mean",
            )
        )
        if "passive" in pivot.columns and "reappraisal" in pivot.columns:
            diff = pivot["passive"] - pivot["reappraisal"]
            # Count paired stimuli per subject (both finite)
            ok = np.isfinite(diff.to_numpy(dtype=np.float64))
            tmp = pd.DataFrame({
                "subject": _normalize_subject_ids(pd.Series(pivot.index.get_level_values(0), index=pivot.index)),
                "diff": diff.to_numpy(dtype=np.float64),
                "ok": ok.astype(int),
            })
            df_s = tmp.groupby("subject", as_index=False).agg(
                reappraisal_success=("diff", "mean"),
                reappraisal_success_n_pairs=("ok", "sum"),
            )
            meta.update({
                "status": "ok",
                "method": "paired_by_stimulus_mean",
                "reason": "",
            })
            return df_s, meta

    # Fallback: subject-level condition means (unpaired).
    grp = (
        df.dropna(subset=["_cond2"])
        .groupby([sub_col, "_cond2"], as_index=False)["_rating"]
        .mean()
    )
    wide = grp.pivot(index=sub_col, columns="_cond2", values="_rating")
    if "passive" in wide.columns and "reappraisal" in wide.columns:
        df_s = wide.reset_index().rename(columns={sub_col: "subject"})
        df_s["subject"] = _normalize_subject_ids(df_s["subject"])
        df_s["reappraisal_success"] = (df_s["passive"] - df_s["reappraisal"]).astype(np.float64)
        df_s = df_s[["subject", "reappraisal_success"]]
        meta.update({
            "status": "ok_degraded",
            "method": "condition_mean_diff_unpaired",
            "reason": "no_stimulus_col_or_cannot_pivot_pairs",
        })
        return df_s, meta

    meta["reason"] = "cannot_compute_success (unexpected condition pivot failure)"
    return None, meta


def _make_subject_level_behavior_df(
    beh_df: pd.DataFrame,
    behavior_cols: List[str],
    success_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one-row-per-subject table for MI~behavior analysis while keeping compatibility:
    - Numeric behavior columns are aggregated by mean if multiple rows per subject exist.
    - Sex is kept (first non-null) if present.
    - reappraisal_success (if computed) is merged in.
    """
    df = beh_df.copy()
    sub_col = _find_subject_col(df)
    if sub_col is None:
        return df
    df[sub_col] = _normalize_subject_ids(df[sub_col])

    agg: Dict[str, object] = {}
    # Keep requested behavior columns if present; aggregate numeric mean.
    for c in behavior_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg[c] = "mean"
    if "sex" in df.columns:
        agg["sex"] = lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan

    if len(agg) > 0:
        df_out = df.groupby(sub_col, as_index=False).agg(agg)
    else:
        # At least deduplicate by subject to avoid trial-level overweighting downstream.
        df_out = df.drop_duplicates(subset=[sub_col]).copy()

    # Ensure the subject identifier column name is preserved for _merge_behavior()
    if sub_col not in df_out.columns:
        df_out[sub_col] = df.groupby(sub_col).size().index.astype(str)

    if success_df is not None and "subject" in success_df.columns:
        df_out = df_out.merge(success_df, left_on=sub_col, right_on="subject", how="left")
        # Remove duplicate subject column created by merge
        if "subject" in df_out.columns and sub_col != "subject":
            df_out = df_out.drop(columns=["subject"])
    return df_out


def _try_lowess(
    x: np.ndarray, y: np.ndarray, frac: float = 0.6
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

        result = sm_lowess(y, x, frac=frac, return_sorted=True)
        return result[:, 0], result[:, 1]
    except ImportError:
        return None


def _safe_spearmanr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a_c = a[mask]
    b_c = b[mask]
    if a_c.size < 5:
        return float("nan"), float("nan")
    res = spearmanr(a_c, b_c)
    return float(res.statistic), float(res.pvalue)


def _residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    for col_i in range(covariates.shape[1]):
        mask &= np.isfinite(covariates[:, col_i])
    out = np.full_like(y, np.nan, dtype=np.float64)
    if int(mask.sum()) < 5:
        return out
    y_c = y[mask]
    X = np.column_stack([np.ones(int(mask.sum()), dtype=np.float64), covariates[mask]])
    beta, _, _, _ = np.linalg.lstsq(X, y_c, rcond=None)
    out[mask] = y_c - X @ beta
    return out


def _permutation_partial(
    mi_resid: np.ndarray,
    beh_resid: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, float]:
    mask = np.isfinite(mi_resid) & np.isfinite(beh_resid)
    mr = mi_resid[mask]
    br = beh_resid[mask]
    if mr.size < 5:
        return float("nan"), float("nan"), float("nan")
    r_obs = float(spearmanr(mr, br).statistic)
    p_obs = float(spearmanr(mr, br).pvalue)
    rng = np.random.default_rng(int(seed))
    count = 0
    for _ in range(int(n_perm)):
        mr_perm = rng.permutation(mr)
        r_perm = float(spearmanr(mr_perm, br).statistic)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_perm = (count + 1.0) / (float(n_perm) + 1.0)
    return r_obs, p_obs, p_perm


def _permutation_test_age(
    ages: np.ndarray,
    scores: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, float]:
    mask = np.isfinite(ages) & np.isfinite(scores)
    a = ages[mask]
    s = scores[mask]
    if a.size < 5:
        return float("nan"), float("nan"), float("nan")
    r_obs = float(spearmanr(a, s).statistic)
    rng = np.random.default_rng(int(seed))
    count = 0
    for _ in range(int(n_perm)):
        a_perm = rng.permutation(a)
        r_perm = float(spearmanr(a_perm, s).statistic)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_perm = (count + 1.0) / (float(n_perm) + 1.0)
    return r_obs, float(spearmanr(a, s).pvalue), p_perm


def _build_mega_vectors(
    npz: np.lib.npyio.NpzFile,
    target_rois: List[str],
    order: List[int],
) -> np.ndarray:
    """
    Build concatenated upper-triangle vectors for each subject (age-sorted order).

    Output shape: (n_sub, n_features)
    """
    n_sub = len(order)
    roi_mats: List[np.ndarray] = []
    for roi in target_rois:
        roi_data = np.asarray(npz[roi])
        roi_vecs: List[np.ndarray] = []
        for si in range(n_sub):
            roi_vecs.append(flatten_upper(roi_data[order[si]]).astype(np.float64))
        roi_mats.append(np.stack(roi_vecs, axis=0))
    return np.concatenate(roi_mats, axis=1).astype(np.float64)


def _compute_template(mega_vecs: np.ndarray, mature_mask: np.ndarray) -> Optional[np.ndarray]:
    if int(mature_mask.sum()) < 1:
        return None
    # Keep backward-compatibility with the original implementation:
    # if template features contain NaNs, they will be excluded via mask_valid in MI computation.
    return np.mean(mega_vecs[mature_mask], axis=0).astype(np.float64)


def _compute_mi_scores(mega_vecs: np.ndarray, template: np.ndarray) -> np.ndarray:
    n_sub = int(mega_vecs.shape[0])
    mi = np.full(n_sub, np.nan, dtype=np.float64)
    for si in range(n_sub):
        v = mega_vecs[si]
        mask_valid = np.isfinite(v) & np.isfinite(template)
        if int(mask_valid.sum()) < 5:
            mi[si] = float("nan")
        else:
            mi[si] = float(spearmanr(v[mask_valid], template[mask_valid]).statistic)
    return mi


def _load_behavior_df(
    behavior_file: Optional[Path],
    cond_dir: Path,
) -> Optional[pd.DataFrame]:
    beh_df: Optional[pd.DataFrame] = None
    if behavior_file is not None:
        if Path(behavior_file).exists():
            try:
                beh_df = pd.read_csv(behavior_file)
            except Exception as e:
                _warn(f"读取行为文件失败：{behavior_file} ({type(e).__name__}: {e})，将跳过 MI~behavior 分析。")
                beh_df = None
        else:
            _warn(f"指定的 --behavior-file 不存在：{behavior_file}，将尝试使用默认回退文件。")
    if beh_df is None:
        fallback_path = cond_dir / "behavior_subject_stimulus_scores.csv"
        if fallback_path.exists():
            try:
                beh_df = pd.read_csv(fallback_path)
            except Exception as e:
                _warn(f"读取回退行为文件失败：{fallback_path} ({type(e).__name__}: {e})，将跳过 MI~behavior 分析。")
                beh_df = None
        else:
            _warn(f"未找到行为文件：既没有 --behavior-file，也没有 {fallback_path}。将跳过 MI~behavior 分析。")
    return beh_df


def _merge_behavior(
    beh_df: pd.DataFrame,
    subjects_sorted: List[str],
    ages_arr: np.ndarray,
    mi_scores: np.ndarray,
    behavior_join: str = "left",
) -> Optional[pd.DataFrame]:
    sub_col = _find_subject_col(beh_df)
    if sub_col is None:
        return None
    beh_df = beh_df.copy()
    beh_df[sub_col] = _normalize_subject_ids(beh_df[sub_col])
    subj_df = pd.DataFrame({
        "subject": subjects_sorted,
        "age": ages_arr,
        "maturity_index": mi_scores,
    })
    how = str(behavior_join).strip().lower()
    if how not in {"left", "inner"}:
        raise ValueError(f"Unsupported behavior_join: {behavior_join}")
    merged = subj_df.merge(beh_df, left_on="subject", right_on=sub_col, how=how)
    n_total = int(subj_df.shape[0])
    # If a subject has no behavior row matched, the right-side subject column will be NaN (left-join).
    n_with_beh = int(merged[sub_col].notna().sum()) if sub_col in merged.columns else 0
    n_lost = int(n_total - n_with_beh)
    if n_lost > 0:
        _warn(
            f"行为数据中缺少 {n_lost}/{n_total} 个被试；MI~behavior 分析最多基于 {n_with_beh} 个被试（join={how}）。"
        )
    merged["_behavior_join"] = how
    merged["_behavior_subject_col"] = str(sub_col)
    merged["_behavior_n_total_subjects"] = n_total
    merged["_behavior_n_with_behavior"] = n_with_beh
    merged["_behavior_n_lost_behavior"] = n_lost
    return merged


def _compute_stats_rows(
    ages_arr: np.ndarray,
    mi_scores: np.ndarray,
    n_perm: int,
    seed: int,
    beh_merged: Optional[pd.DataFrame],
    behavior_cols: List[str],
) -> List[Dict[str, object]]:
    r_age, p_age, p_age_perm = _permutation_test_age(ages_arr, mi_scores, n_perm=n_perm, seed=seed)
    stat_rows: List[Dict[str, object]] = [
        {
            "variable": "age",
            "r": r_age,
            "p": p_age,
            "p_perm": p_age_perm,
            "method": "age_correlation",
        },
    ]

    has_sex = bool(beh_merged is not None and "sex" in beh_merged.columns)
    if beh_merged is None:
        return stat_rows

    for bcol in behavior_cols:
        if bcol not in beh_merged.columns:
            continue
        beh_vals = beh_merged[bcol].to_numpy(dtype=np.float64)
        mi_vals = beh_merged["maturity_index"].to_numpy(dtype=np.float64)
        age_vals = beh_merged["age"].to_numpy(dtype=np.float64)

        if has_sex:
            sex_vals = pd.Categorical(beh_merged["sex"]).codes.astype(np.float64)
            covariates = np.column_stack([age_vals, sex_vals])
        else:
            covariates = age_vals.reshape(-1, 1)

        mi_resid = _residualize(mi_vals, covariates)
        beh_resid = _residualize(beh_vals, covariates)

        r_partial, p_partial, p_partial_perm = _permutation_partial(
            mi_resid, beh_resid, n_perm=n_perm, seed=seed
        )
        stat_rows.append({
            "variable": str(bcol),
            "r": r_partial,
            "p": p_partial,
            "p_perm": p_partial_perm,
            "method": "partial_correlation",
        })
    return stat_rows


def _determine_target_rois(
    target_networks: List[str],
    include_subcortical: List[str],
    include_scan: bool = False,
) -> List[str]:
    cortical_df = get_schaefer200_network_labels()
    cortical_rois = cortical_df[cortical_df["network"].isin(target_networks)]["roi"].tolist()

    subcortical_df = get_tian_s2_labels()
    subcortical_rois = subcortical_df[subcortical_df["structure_group"].isin(include_subcortical)]["roi"].tolist()

    target_rois = sorted(set(cortical_rois + subcortical_rois))

    if include_scan:
        try:
            scan_rois = get_scan_cortical_rois()
            target_rois = sorted(set(target_rois) | set(scan_rois))
        except Exception:
            pass

    return target_rois


def run_maturity_index(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition: str,
    repr_prefix: str,
    subject_info: Path,
    behavior_file: Optional[Path],
    behavior_cols: List[str],
    behavior_join: str,
    mature_age_min: float,
    target_networks: List[str],
    include_subcortical: List[str],
    include_scan: bool,
    n_perm: int,
    seed: int,
    alpha: float,
    dpi: int,
    out_dir: Optional[Path],
    mature_age_grid: Optional[List[float]] = None,
    bootstrap_template: int = 0,
    bootstrap_seed: Optional[int] = None,
    loo_template: bool = False,
) -> None:
    cond_dir = matrix_dir / str(stimulus_dir_name) / str(condition)
    npz_path = cond_dir / f"{repr_prefix}.npz"
    subjects_path = cond_dir / f"{repr_prefix}_subjects.csv"

    npz = np.load(npz_path, allow_pickle=False)
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    idx_map = {s: i for i, s in enumerate(subjects)}
    order = [idx_map[s] for s in subjects_sorted]

    n_sub = len(subjects_sorted)
    ages_arr = np.asarray(ages_sorted, dtype=np.float64)

    if out_dir is None:
        out_dir = matrix_dir / "gradient_analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_rois = _determine_target_rois(target_networks, include_subcortical, include_scan)
    available_rois = set(npz.files)
    missing = [r for r in target_rois if r not in available_rois]
    if missing:
        raise ValueError(f"以下目标 ROI 不在数据中: {missing}")

    mature_mask = ages_arr >= mature_age_min
    young_mask = ages_arr < mature_age_min
    n_mature = int(mature_mask.sum())
    n_young = int(young_mask.sum())
    if n_mature < 5:
        _warn(
            f"成熟被试数量偏小（n_mature={n_mature} < 5）。主 MI 仍会继续计算，但模板/稳定性结果需谨慎解释；"
            f"可考虑降低 --mature-age-min 或用 --mature-age-grid 做敏感性。"
        )

    mega_vecs = _build_mega_vectors(npz, target_rois=target_rois, order=order)
    mega_template = _compute_template(mega_vecs, mature_mask=mature_mask)
    if mega_template is None:
        raise ValueError("无法构建成熟模板：成熟被试数量为 0。请降低 --mature-age-min。")
    mi_scores = _compute_mi_scores(mega_vecs, template=mega_template)

    beh_df = _load_behavior_df(behavior_file, cond_dir=cond_dir)
    beh_merged: Optional[pd.DataFrame] = None
    success_meta: Dict[str, object] = {
        "status": "not_attempted",
        "method": "",
        "reason": "",
        "rating_col": "",
        "condition_col": "",
        "stimulus_col": "",
    }
    if beh_df is not None:
        # Task 5: try to compute subject-level reappraisal_success (non-fatal).
        success_df, success_meta = _try_compute_reappraisal_success(beh_df)
        if success_df is None:
            if success_meta.get("reason"):
                _warn(f"未能生成 reappraisal_success：{success_meta.get('reason')}")
        else:
            tag = str(success_meta.get("status", "ok"))
            method = str(success_meta.get("method", ""))
            if tag == "ok_degraded":
                _warn(f"reappraisal_success 使用退化定义：{method}（{success_meta.get('reason','')}）")
            else:
                print(f"[maturity_index] 已生成 reappraisal_success（{method}）")

        # Build subject-level behavior table to avoid trial-level overweighting.
        beh_subject = _make_subject_level_behavior_df(
            beh_df,
            behavior_cols=behavior_cols,
            success_df=success_df,
        )
        beh_merged = _merge_behavior(
            beh_subject,
            subjects_sorted,
            ages_arr,
            mi_scores,
            behavior_join=str(behavior_join),
        )
        if beh_merged is None:
            _warn("行为文件存在但无法识别被试列（支持 subject/sub_id/被试编号），将跳过 MI~behavior 分析。")
        else:
            missing_req = [c for c in behavior_cols if c not in beh_merged.columns]
            # Only warn about requested columns if behavior analysis is enabled but columns are absent.
            if len(missing_req) > 0:
                extra = ""
                if "reappraisal_success" in missing_req and str(success_meta.get("reason", "")):
                    extra = f"（reappraisal_success 原因: {success_meta.get('reason')}）"
                _warn(f"以下 --behavior-cols 在行为表中不存在，将跳过相应分析: {missing_req}{extra}")

    stat_rows = _compute_stats_rows(
        ages_arr=ages_arr,
        mi_scores=mi_scores,
        n_perm=n_perm,
        seed=seed,
        beh_merged=beh_merged,
        behavior_cols=behavior_cols,
    )

    score_rows: List[Dict[str, object]] = []
    for si in range(n_sub):
        row: Dict[str, object] = {
            "subject": subjects_sorted[si],
            "age": float(ages_arr[si]),
            "maturity_index": float(mi_scores[si]),
        }
        if beh_merged is not None:
            match = beh_merged[beh_merged["subject"] == subjects_sorted[si]]
            if not match.empty:
                for bcol in behavior_cols:
                    if bcol in match.columns:
                        row[bcol] = float(match.iloc[0][bcol]) if pd.notna(match.iloc[0][bcol]) else float("nan")
        score_rows.append(row)

    pd.DataFrame(score_rows).to_csv(out_dir / "maturity_index_subject_scores.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out_dir / "maturity_index_stats.csv", index=False)

    config_rows = [{
        "condition": str(condition),
        "mature_age_min": float(mature_age_min),
        "n_mature": int(n_mature),
        "n_young": int(n_young),
        "target_rois": "|".join(target_rois),
        "include_scan": include_scan,
        "behavior_join": str(behavior_join),
        "reappraisal_success_status": str(success_meta.get("status", "")),
        "reappraisal_success_method": str(success_meta.get("method", "")),
        "reappraisal_success_reason": str(success_meta.get("reason", "")),
        "reappraisal_success_rating_col": str(success_meta.get("rating_col", "")),
        "reappraisal_success_condition_col": str(success_meta.get("condition_col", "")),
        "reappraisal_success_stimulus_col": str(success_meta.get("stimulus_col", "")),
    }]
    if beh_merged is not None:
        for k in (
            "_behavior_subject_col",
            "_behavior_n_total_subjects",
            "_behavior_n_with_behavior",
            "_behavior_n_lost_behavior",
            "_behavior_join",
        ):
            if k in beh_merged.columns:
                config_rows[0][k.lstrip("_")] = beh_merged[k].iloc[0]
    pd.DataFrame(config_rows).to_csv(out_dir / "maturity_index_config.csv", index=False)

    # Extract the main MI~age summary from stats for plotting/printing
    r_age_plot = float("nan")
    p_age_perm_plot = float("nan")
    for sr in stat_rows:
        if sr.get("variable") == "age":
            r_age_plot = float(sr.get("r")) if sr.get("r") is not None else float("nan")
            p_age_perm_plot = float(sr.get("p_perm")) if sr.get("p_perm") is not None else float("nan")
            break

    has_behavior_panel = False
    beh_col_for_plot: Optional[str] = None
    mi_resid_plot: Optional[np.ndarray] = None
    beh_resid_plot: Optional[np.ndarray] = None
    partial_r_plot: Optional[float] = None
    partial_p_plot: Optional[float] = None
    has_sex = bool(beh_merged is not None and "sex" in beh_merged.columns)
    if beh_merged is not None:
        for bcol in behavior_cols:
            if bcol not in beh_merged.columns:
                continue
            beh_vals = beh_merged[bcol].to_numpy(dtype=np.float64)
            mi_vals = beh_merged["maturity_index"].to_numpy(dtype=np.float64)
            age_vals = beh_merged["age"].to_numpy(dtype=np.float64)
            if has_sex:
                sex_vals = pd.Categorical(beh_merged["sex"]).codes.astype(np.float64)
                covariates = np.column_stack([age_vals, sex_vals])
            else:
                covariates = age_vals.reshape(-1, 1)
            mi_r = _residualize(mi_vals, covariates)
            beh_r = _residualize(beh_vals, covariates)
            mask_valid = np.isfinite(mi_r) & np.isfinite(beh_r)
            if int(mask_valid.sum()) >= 5:
                has_behavior_panel = True
                beh_col_for_plot = bcol
                mi_resid_plot = mi_r
                beh_resid_plot = beh_r
                for sr in stat_rows:
                    if sr["variable"] == bcol:
                        partial_r_plot = sr["r"]
                        partial_p_plot = sr["p_perm"]
                break

    n_panels = 2 if has_behavior_panel else 1
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), squeeze=False)
    axes_flat = axes.flatten()

    ax0 = axes_flat[0]
    mask0 = np.isfinite(ages_arr) & np.isfinite(mi_scores)
    x0 = ages_arr[mask0]
    y0 = mi_scores[mask0]
    norm = plt.Normalize(vmin=float(x0.min()), vmax=float(x0.max()))
    cmap = plt.cm.viridis
    ax0.scatter(x0, y0, c=x0, cmap=cmap, norm=norm, s=22, alpha=0.7, linewidths=0, zorder=2)

    lowess_result = _try_lowess(x0, y0)
    if lowess_result is not None:
        lx, ly = lowess_result
        ax0.plot(lx, ly, color="#d62728", lw=2.0, zorder=3)
    else:
        if x0.size >= 3:
            lr = linregress(x0, y0)
            x_line = np.linspace(float(x0.min()), float(x0.max()), 200)
            ax0.plot(x_line, lr.intercept + lr.slope * x_line, color="#d62728", lw=2.0, zorder=3)

    ax0.annotate(
        f"r={r_age_plot:.3f}, p_perm={p_age_perm_plot:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="top",
    )
    ax0.set_title(f"Maturity Index ~ Age ({condition})")
    ax0.set_xlabel("Age")
    ax0.set_ylabel("Maturity Index")

    if has_behavior_panel:
        ax1 = axes_flat[1]
        mask1 = np.isfinite(mi_resid_plot) & np.isfinite(beh_resid_plot)
        xr = mi_resid_plot[mask1]
        yr = beh_resid_plot[mask1]
        ax1.scatter(xr, yr, s=22, alpha=0.7, linewidths=0, color="#4C72B0", zorder=2)

        lowess_r = _try_lowess(xr, yr)
        if lowess_r is not None:
            lx_r, ly_r = lowess_r
            ax1.plot(lx_r, ly_r, color="#d62728", lw=2.0, zorder=3)
        else:
            if xr.size >= 3:
                lr_r = linregress(xr, yr)
                x_line_r = np.linspace(float(xr.min()), float(xr.max()), 200)
                ax1.plot(x_line_r, lr_r.intercept + lr_r.slope * x_line_r, color="#d62728", lw=2.0, zorder=3)

        r_str = f"{partial_r_plot:.3f}" if partial_r_plot is not None else "NaN"
        p_str = f"{partial_p_plot:.4f}" if partial_p_plot is not None else "NaN"
        ax1.annotate(
            f"partial r={r_str}, p_perm={p_str}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            verticalalignment="top",
        )
        ax1.set_title(f"MI ~ {beh_col_for_plot} (age-partialled, {condition})")
        ax1.set_xlabel("Maturity Index (residualized)")
        ax1.set_ylabel(f"{beh_col_for_plot} (residualized)")

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_maturity_index.png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"[maturity_index] MI ~ age: r={r_age_plot:.4f}, p_perm={p_age_perm_plot:.4f}")
    if has_behavior_panel and partial_r_plot is not None:
        print(f"[maturity_index] MI ~ {beh_col_for_plot} (partial): r={partial_r_plot:.4f}, p_perm={partial_p_plot:.4f}")
    print(f"[maturity_index] 结果已保存至 {out_dir}")

    # -------------------------
    # Optional analyses (Task 4)
    # -------------------------

    # 1) Mature threshold sensitivity grid
    if mature_age_grid is not None and len(mature_age_grid) > 0:
        grid = [float(x) for x in mature_age_grid]
        grid = sorted(set(grid))
        sens_rows: List[Dict[str, object]] = []
        for thr in grid:
            mm = ages_arr >= float(thr)
            ym = ages_arr < float(thr)
            nm = int(mm.sum())
            ny = int(ym.sum())

            row_prefix = {
                "mature_age_min": float(thr),
                "n_mature": int(nm),
                "n_young": int(ny),
            }
            if nm < 1:
                sens_rows.append({
                    **row_prefix,
                    "status": "skipped_no_mature",
                    "variable": "age",
                    "r": float("nan"),
                    "p": float("nan"),
                    "p_perm": float("nan"),
                    "method": "age_correlation",
                })
                continue
            if nm < 5:
                _warn(f"--mature-age-grid 阈值 {thr:.3f} 下成熟组过小（n_mature={nm} < 5），敏感性统计将输出 NaN/不稳定。")

            tmpl = _compute_template(mega_vecs, mature_mask=mm)
            if tmpl is None:
                continue
            mi_t = _compute_mi_scores(mega_vecs, template=tmpl)

            beh_merged_t: Optional[pd.DataFrame] = None
            if beh_df is not None:
                beh_merged_t = _merge_behavior(beh_df, subjects_sorted, ages_arr, mi_t)

            stats_t = _compute_stats_rows(
                ages_arr=ages_arr,
                mi_scores=mi_t,
                n_perm=n_perm,
                seed=seed,
                beh_merged=beh_merged_t,
                behavior_cols=behavior_cols,
            )
            for sr in stats_t:
                sens_rows.append({**row_prefix, "status": "ok", **sr})
        pd.DataFrame(sens_rows).to_csv(out_dir / "maturity_index_sensitivity.csv", index=False)

    # 2) Bootstrap template stability
    B = int(bootstrap_template)
    if B > 0:
        b_seed = int(seed) if bootstrap_seed is None else int(bootstrap_seed)
        rng = np.random.default_rng(b_seed)
        mature_idx = np.where(mature_mask)[0]
        if mature_idx.size < 1:
            _warn("bootstrap 模板稳定性跳过：成熟被试数量为 0。")
        else:
            if mature_idx.size < 5:
                _warn(f"bootstrap 模板稳定性：成熟组偏小（n_mature={mature_idx.size} < 5），CI/SD 可能不稳定。")
            mi_boot = np.full((B, n_sub), np.nan, dtype=np.float32)
            for bi in range(B):
                sample_idx = rng.choice(mature_idx, size=int(mature_idx.size), replace=True)
                tmpl_b = np.mean(mega_vecs[sample_idx], axis=0).astype(np.float64)
                mi_boot[bi] = _compute_mi_scores(mega_vecs, template=tmpl_b).astype(np.float32)
                if (bi + 1) % max(1, (B // 5)) == 0:
                    print(f"[maturity_index][bootstrap] {bi+1}/{B}")

            mi_mean = np.nanmean(mi_boot, axis=0).astype(np.float64)
            mi_sd = np.nanstd(mi_boot, axis=0, ddof=1).astype(np.float64)
            ci_low = np.nanpercentile(mi_boot, 2.5, axis=0).astype(np.float64)
            ci_high = np.nanpercentile(mi_boot, 97.5, axis=0).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                mi_cv = mi_sd / np.abs(mi_mean)

            df_sub = pd.DataFrame({
                "row_type": "subject",
                "subject": subjects_sorted,
                "age": ages_arr.astype(np.float64),
                "maturity_index_mean": mi_mean,
                "maturity_index_sd": mi_sd,
                "maturity_index_ci_low": ci_low,
                "maturity_index_ci_high": ci_high,
                "maturity_index_cv": mi_cv,
                "global_mean_sd": np.nan,
                "global_median_sd": np.nan,
                "global_mean_cv": np.nan,
                "global_median_cv": np.nan,
                "B": int(B),
                "bootstrap_seed": int(b_seed),
                "mature_age_min": float(mature_age_min),
                "n_mature": int(mature_idx.size),
                "n_sub": int(n_sub),
            })

            global_row = {
                "row_type": "global",
                "subject": "__GLOBAL__",
                "age": float("nan"),
                "maturity_index_mean": float("nan"),
                "maturity_index_sd": float("nan"),
                "maturity_index_ci_low": float("nan"),
                "maturity_index_ci_high": float("nan"),
                "maturity_index_cv": float("nan"),
                "global_mean_sd": float(np.nanmean(mi_sd)),
                "global_median_sd": float(np.nanmedian(mi_sd)),
                "global_mean_cv": float(np.nanmean(mi_cv)),
                "global_median_cv": float(np.nanmedian(mi_cv)),
                "B": int(B),
                "bootstrap_seed": int(b_seed),
                "mature_age_min": float(mature_age_min),
                "n_mature": int(mature_idx.size),
                "n_sub": int(n_sub),
            }
            df_out = pd.concat([df_sub, pd.DataFrame([global_row])], ignore_index=True)
            df_out.to_csv(out_dir / "maturity_index_stability.csv", index=False)

    # 3) Minimal LOO template influence
    if bool(loo_template):
        mature_idx = np.where(mature_mask)[0]
        if mature_idx.size < 2:
            _warn(f"LOO 模板影响分析跳过：成熟被试数量不足（n_mature={mature_idx.size} < 2）。")
            pd.DataFrame([{
                "status": "skipped_small_mature",
                "reason": "n_mature < 2",
                "mature_age_min": float(mature_age_min),
                "n_mature": int(mature_idx.size),
            }]).to_csv(out_dir / "maturity_index_loo.csv", index=False)
        else:
            loo_rows: List[Dict[str, object]] = []
            for ii, left_i in enumerate(mature_idx.tolist()):
                keep = mature_idx[mature_idx != left_i]
                tmpl_loo = np.mean(mega_vecs[keep], axis=0).astype(np.float64)
                mi_loo = _compute_mi_scores(mega_vecs, template=tmpl_loo)

                # Template similarity (full vs LOO) as a quick influence proxy
                mask_t = np.isfinite(mega_template) & np.isfinite(tmpl_loo)
                tmpl_sim = float("nan")
                if int(mask_t.sum()) >= 5:
                    tmpl_sim = float(spearmanr(mega_template[mask_t], tmpl_loo[mask_t]).statistic)

                mask_mi = np.isfinite(mi_scores) & np.isfinite(mi_loo)
                corr_full_loo = float("nan")
                mean_abs_delta = float("nan")
                max_abs_delta = float("nan")
                if int(mask_mi.sum()) >= 5:
                    corr_full_loo = float(spearmanr(mi_scores[mask_mi], mi_loo[mask_mi]).statistic)
                    diffs = np.abs(mi_scores[mask_mi] - mi_loo[mask_mi])
                    mean_abs_delta = float(np.mean(diffs))
                    max_abs_delta = float(np.max(diffs))

                # Minimal age association (no permutation to keep LOO cheap)
                r_loo, p_loo = _safe_spearmanr(ages_arr, mi_loo)

                loo_rows.append({
                    "left_out_subject": subjects_sorted[left_i],
                    "left_out_age": float(ages_arr[left_i]),
                    "mature_age_min": float(mature_age_min),
                    "n_mature_full": int(mature_idx.size),
                    "n_mature_loo": int(keep.size),
                    "template_similarity_to_full": tmpl_sim,
                    "corr_mi_full_vs_loo": corr_full_loo,
                    "mean_abs_delta_mi": mean_abs_delta,
                    "max_abs_delta_mi": max_abs_delta,
                    "r_age_loo": float(r_loo),
                    "p_age_loo": float(p_loo),
                })
                if (ii + 1) % max(1, (len(mature_idx) // 5)) == 0:
                    print(f"[maturity_index][loo] {ii+1}/{len(mature_idx)}")
            pd.DataFrame(loo_rows).to_csv(out_dir / "maturity_index_loo.csv", index=False)


def main() -> None:
    args = parse_args()
    run_maturity_index(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        condition=str(args.condition),
        repr_prefix=str(args.repr_prefix),
        subject_info=Path(args.subject_info),
        behavior_file=Path(args.behavior_file) if args.behavior_file is not None else None,
        behavior_cols=[str(x) for x in args.behavior_cols],
        behavior_join=str(args.behavior_join),
        mature_age_min=float(args.mature_age_min),
        target_networks=[str(x) for x in args.target_networks],
        include_subcortical=[str(x) for x in args.include_subcortical],
        include_scan=bool(args.include_scan),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        alpha=float(args.alpha),
        dpi=int(args.dpi),
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
        mature_age_grid=[float(x) for x in args.mature_age_grid] if args.mature_age_grid is not None else None,
        bootstrap_template=int(args.bootstrap_template),
        bootstrap_seed=int(args.bootstrap_seed) if args.bootstrap_seed is not None else None,
        loo_template=bool(args.loo_template),
    )


if __name__ == "__main__":
    main()
