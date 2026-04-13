#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_representational_connectivity.py

Step 3:
计算被试内 ROI 对之间的表征连通性（Representational Connectivity），
并检验其与年龄的关系（回归 + 置换检验），生成 Figure 3。
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
from utils_network_assignment import get_roi_network_map, get_scan_cortical_rois  # noqa: E402

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: 被试内 ROI 对表征连通性及年龄效应")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--condition", type=str, default="Reappraisal")
    p.add_argument("--repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument(
        "--subject-info",
        type=Path,
        default=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
    )
    p.add_argument(
        "--roi-pairs",
        nargs="+",
        default=["V_1:R_73", "V_2:R_73", "V_1:R_78", "V_3:L_14", "V_3:R_14"],
    )
    p.add_argument(
        "--network-pairs",
        nargs="+",
        default=["Hippocampus:Default", "Amygdala:SomMot", "SCAN:Default"],
    )
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--condition-contrast", type=str, default=None)
    # Supplementary analysis (Task 7): non-linear developmental modeling (optional; default off).
    p.add_argument(
        "--dev-modeling",
        action="store_true",
        default=False,
        help="启用发育曲线模型对比（线性 vs 二次），输出模型对比摘要 CSV；默认关闭，不影响既有输出",
    )
    p.add_argument(
        "--dev-model-criterion",
        type=str,
        default="AIC",
        choices=["AIC", "R2_adj"],
        help="模型选择准则：AIC(越小越好) / R2_adj(越大越好)",
    )
    p.add_argument(
        "--dev-model-min-n",
        type=int,
        default=8,
        help="每个 pair_label 进行模型对比所需的最小有效样本数（默认 8）",
    )
    # Supplementary analysis (Task 6): representational complexity / distinctiveness.
    p.add_argument(
        "--skip-repr-complexity",
        action="store_true",
        default=False,
        help="跳过表征复杂性/区分度补充分析（默认会输出 repr_complexity_*.csv，不影响既有 RC 输出）",
    )
    p.add_argument(
        "--repr-complexity-level",
        type=str,
        default="roi",
        choices=["roi", "network", "both"],
        help="表征复杂性输出层级：roi(逐 ROI) / network(按网络聚合) / both(两者都输出)",
    )
    p.add_argument(
        "--rc-aggregate",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="网络级 RC 的聚合方式：mean（默认）或 median。",
    )
    return p.parse_args()


def _parse_pair(pair_str: str) -> Tuple[str, str]:
    parts = str(pair_str).split(":")
    if len(parts) != 2:
        raise ValueError(f"无效的 pair 格式（应为 A:B）: {pair_str}")
    return parts[0].strip(), parts[1].strip()


def _compute_rc_single(
    npz: dict[str, np.ndarray],
    roi_a: str,
    roi_b: str,
    subject_idx: int,
) -> float:
    rsm_a = np.asarray(npz[roi_a][subject_idx], dtype=np.float64)
    rsm_b = np.asarray(npz[roi_b][subject_idx], dtype=np.float64)
    vec_a = flatten_upper(rsm_a)
    vec_b = flatten_upper(rsm_b)
    if vec_a.size < 3:
        return float("nan")
    result = spearmanr(vec_a, vec_b)
    return float(result.statistic)


def _get_cross_group_pairs(
    group_a: str,
    group_b: str,
    roi_network_map: Dict[str, str],
    available_rois: set[str],
) -> List[Tuple[str, str]]:
    rois_a = sorted([r for r in _resolve_group_rois(group_a, roi_network_map) if r in available_rois])
    rois_b = sorted([r for r in _resolve_group_rois(group_b, roi_network_map) if r in available_rois])
    pairs: List[Tuple[str, str]] = []
    for ra in rois_a:
        for rb in rois_b:
            pairs.append((ra, rb))
    return pairs


def _try_lowess(
    x: np.ndarray, y: np.ndarray, frac: float = 0.6
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

        result = sm_lowess(y, x, frac=frac, return_sorted=True)
        return result[:, 0], result[:, 1]
    except ImportError:
        return None


def _ols_fit_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Lightweight OLS (no hard dependency on statsmodels).

    Metrics are suitable for comparing models fit on the same y (same n):
    - AIC: Gaussian with unknown variance, constant term dropped: n*log(RSS/n) + 2k
    - R2_adj: adjusted R^2
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    k = int(X.shape[1])
    if n == 0 or k == 0:
        return {
            "n": float(n),
            "k": float(k),
            "rss": float("nan"),
            "aic": float("nan"),
            "r2": float("nan"),
            "r2_adj": float("nan"),
        }
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.sum(resid ** 2))

    # AIC (drop additive constants shared across models with same n).
    eps = 1e-12
    sigma2 = max(rss / float(n), eps)
    aic = float(float(n) * np.log(sigma2) + 2.0 * float(k))

    # R^2 and adjusted R^2.
    y_mean = float(np.mean(y))
    tss = float(np.sum((y - y_mean) ** 2))
    if tss <= 0.0:
        r2 = float("nan")
        r2_adj = float("nan")
    else:
        r2 = float(1.0 - (rss / tss))
        if n > k:
            r2_adj = float(1.0 - (1.0 - r2) * (float(n - 1) / float(n - k)))
        else:
            r2_adj = float("nan")

    out: Dict[str, float] = {
        "n": float(n),
        "k": float(k),
        "rss": float(rss),
        "aic": float(aic),
        "r2": float(r2),
        "r2_adj": float(r2_adj),
    }
    # Coefficients for downstream reporting (beta0, beta1, beta2...).
    for i, b in enumerate(np.asarray(beta, dtype=np.float64).tolist()):
        out[f"beta_{i}"] = float(b)
    return out


def _compare_dev_models_linear_vs_quadratic(
    df_scores: pd.DataFrame,
    condition: str,
    criterion: str,
    min_n: int,
) -> pd.DataFrame:
    """
    For each pair_label, compare:
    - linear: y ~ 1 + age
    - quadratic: y ~ 1 + age + age^2
    """
    if df_scores.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for pair_label, sub in df_scores.groupby("pair_label", sort=True):
        ages = np.asarray(sub["age"].to_numpy(dtype=np.float64), dtype=np.float64)
        scores = np.asarray(sub["RC_score"].to_numpy(dtype=np.float64), dtype=np.float64)
        mask = np.isfinite(ages) & np.isfinite(scores)
        x = ages[mask]
        y = scores[mask]
        n = int(y.size)

        base_row: Dict[str, object] = {
            "condition": str(condition),
            "pair_label": str(pair_label),
            "n": int(n),
            "criterion": str(criterion),
        }
        if n < int(min_n):
            base_row.update(
                {
                    "best_model": "skip",
                    "skip_reason": f"n<{int(min_n)}",
                    "aic_linear": float("nan"),
                    "aic_quadratic": float("nan"),
                    "delta_aic_quadratic_minus_linear": float("nan"),
                    "r2_adj_linear": float("nan"),
                    "r2_adj_quadratic": float("nan"),
                    "quad_vertex_age": float("nan"),
                    "quad_vertex_in_age_range": False,
                }
            )
            rows.append(base_row)
            continue

        X_lin = np.column_stack([np.ones_like(x), x])
        X_quad = np.column_stack([np.ones_like(x), x, x ** 2])
        m_lin = _ols_fit_metrics(X_lin, y)
        m_quad = _ols_fit_metrics(X_quad, y)

        aic_lin = float(m_lin.get("aic", float("nan")))
        aic_quad = float(m_quad.get("aic", float("nan")))
        r2a_lin = float(m_lin.get("r2_adj", float("nan")))
        r2a_quad = float(m_quad.get("r2_adj", float("nan")))

        # Quadratic turning point: -b1 / (2*b2)
        b1 = float(m_quad.get("beta_1", float("nan")))
        b2 = float(m_quad.get("beta_2", float("nan")))
        vertex_age = float("nan")
        vertex_in_range = False
        if np.isfinite(b1) and np.isfinite(b2) and abs(b2) > 1e-12:
            vertex_age = float(-b1 / (2.0 * b2))
            vertex_in_range = bool((vertex_age >= float(np.nanmin(x))) and (vertex_age <= float(np.nanmax(x))))

        # Select best model.
        best_model = "linear"
        if str(criterion).upper() == "AIC":
            if np.isfinite(aic_quad) and np.isfinite(aic_lin) and (aic_quad < aic_lin):
                best_model = "quadratic"
        else:  # R2_adj
            if np.isfinite(r2a_quad) and np.isfinite(r2a_lin) and (r2a_quad > r2a_lin):
                best_model = "quadratic"

        base_row.update(
            {
                "best_model": best_model,
                "skip_reason": "",
                "aic_linear": aic_lin,
                "aic_quadratic": aic_quad,
                "delta_aic_quadratic_minus_linear": float(aic_quad - aic_lin)
                if (np.isfinite(aic_quad) and np.isfinite(aic_lin))
                else float("nan"),
                "r2_adj_linear": r2a_lin,
                "r2_adj_quadratic": r2a_quad,
                "quad_beta_age": float(m_quad.get("beta_1", float("nan"))),
                "quad_beta_age2": float(m_quad.get("beta_2", float("nan"))),
                "quad_vertex_age": vertex_age,
                "quad_vertex_in_age_range": bool(vertex_in_range),
            }
        )
        rows.append(base_row)
    return pd.DataFrame(rows)


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


def _resolve_group_rois(group_name: str, roi_network_map: Dict[str, str]) -> List[str]:
    if group_name == "SCAN":
        try:
            return get_scan_cortical_rois()
        except Exception:
            return []
    return [roi for roi, net in roi_network_map.items() if net == group_name]


def _offdiag_upper_std(rsm: np.ndarray, iu: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Minimal viable "representational complexity/distinctiveness" metric:
    std of the strict upper-triangle off-diagonal elements of an ROI RSM.
    """
    arr = np.asarray(rsm, dtype=np.float64)
    vals = arr[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def _pattern_distinctiveness(rsm: np.ndarray) -> float:
    """
    A more interpretable distinctiveness metric:
    for each stimulus, compute its mean similarity to all *other* stimuli, then
    quantify the standard deviation across stimuli. Larger values imply that some
    stimuli are systematically more/less confusable than others.
    """
    arr = np.asarray(rsm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return float("nan")
    n = arr.shape[0]
    if n < 3:
        return float("nan")
    arr = arr.copy()
    np.fill_diagonal(arr, np.nan)
    row_means = np.nanmean(arr, axis=1)
    row_means = row_means[np.isfinite(row_means)]
    if row_means.size < 2:
        return float("nan")
    return float(np.std(row_means, ddof=1))


def _compute_condition_repr_complexity(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition: str,
    repr_prefix: str,
    subject_info: Path,
    level: str,
    n_perm: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Compute supplementary representational complexity scores and age association stats.

    Output rows are compatible with gradient_analysis/*.csv conventions in this repo.
    """
    cond_dir = matrix_dir / str(stimulus_dir_name) / str(condition)
    npz_path = cond_dir / f"{repr_prefix}.npz"
    subjects_path = cond_dir / f"{repr_prefix}_subjects.csv"
    rois_path = cond_dir / f"{repr_prefix}_rois.csv"

    npz = np.load(npz_path, allow_pickle=False)
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
    rois = pd.read_csv(rois_path)["roi"].astype(str).tolist()

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    idx_map = {s: i for i, s in enumerate(subjects)}
    order = [idx_map[s] for s in subjects_sorted]

    n_sub = len(subjects_sorted)
    ages_arr = np.asarray(ages_sorted, dtype=np.float64)

    # Determine RSM size from the first ROI entry.
    if len(rois) == 0:
        return [], []
    sample = np.asarray(npz[rois[0]][order[0]], dtype=np.float64)
    iu = np.triu_indices(sample.shape[0], k=1)

    score_rows: List[Dict[str, object]] = []
    stat_rows: List[Dict[str, object]] = []

    want_roi = str(level) in ("roi", "both")
    want_net = str(level) in ("network", "both")

    metric_funcs = {
        "offdiag_std_upper": lambda rsm: _offdiag_upper_std(rsm, iu),
        "pattern_distinctiveness": _pattern_distinctiveness,
    }
    roi_scores_by_metric: Dict[str, np.ndarray] = {
        metric: np.full((n_sub, len(rois)), np.nan, dtype=np.float64)
        for metric in metric_funcs
    }
    if want_roi or want_net:
        for ri, roi in enumerate(rois):
            arr = np.asarray(npz[roi], dtype=np.float64)
            arr = arr[order, :, :]
            for si in range(n_sub):
                for metric_name, func in metric_funcs.items():
                    roi_scores_by_metric[metric_name][si, ri] = func(arr[si])

    if want_roi:
        for metric_name, roi_scores in roi_scores_by_metric.items():
            for ri, roi in enumerate(rois):
                scores = roi_scores[:, ri]
                for si in range(n_sub):
                    score_rows.append(
                        {
                            "subject": subjects_sorted[si],
                            "age": float(ages_arr[si]),
                            "level": "roi",
                            "label": str(roi),
                            "metric": metric_name,
                            "repr_complexity": float(scores[si]) if np.isfinite(scores[si]) else float("nan"),
                        }
                    )
                r_age, p_age, p_perm = _permutation_test_age(ages_arr, scores, n_perm=n_perm, seed=seed)
                stat_rows.append(
                    {
                        "condition": str(condition),
                        "level": "roi",
                        "label": str(roi),
                        "metric": metric_name,
                        "r_age": r_age,
                        "p_age": p_age,
                        "p_perm": p_perm,
                        "n_subjects": int(n_sub),
                    }
                )

    # 2) Network-level aggregation (mean across ROIs within network_or_structure)
    if want_net:
        roi_network_map = get_roi_network_map(roi_set=232)
        roi_to_group = {roi: roi_network_map.get(roi, "Unknown") for roi in rois}
        groups = sorted(set(roi_to_group.values()))
        for metric_name, roi_scores in roi_scores_by_metric.items():
            for g in groups:
                idx = [ri for ri, roi in enumerate(rois) if roi_to_group.get(roi) == g]
                if not idx:
                    continue
                scores = np.nanmean(roi_scores[:, idx], axis=1)
                for si in range(n_sub):
                    score_rows.append(
                        {
                            "subject": subjects_sorted[si],
                            "age": float(ages_arr[si]),
                            "level": "network",
                            "label": str(g),
                            "metric": metric_name,
                            "repr_complexity": float(scores[si]) if np.isfinite(scores[si]) else float("nan"),
                        }
                    )
                r_age, p_age, p_perm = _permutation_test_age(ages_arr, scores, n_perm=n_perm, seed=seed)
                stat_rows.append(
                    {
                        "condition": str(condition),
                        "level": "network",
                        "label": str(g),
                        "metric": metric_name,
                        "r_age": r_age,
                        "p_age": p_age,
                        "p_perm": p_perm,
                        "n_subjects": int(n_sub),
                        "n_rois_in_group": int(len(idx)),
                    }
                )

    return score_rows, stat_rows


def _compute_condition_rc(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition: str,
    repr_prefix: str,
    subject_info: Path,
    roi_pairs: List[Tuple[str, str]],
    network_pairs: List[Tuple[str, str]],
    n_perm: int,
    seed: int,
    rc_aggregate: str = "mean",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, np.ndarray], np.ndarray]:
    cond_dir = matrix_dir / str(stimulus_dir_name) / str(condition)
    npz_path = cond_dir / f"{repr_prefix}.npz"
    subjects_path = cond_dir / f"{repr_prefix}_subjects.csv"
    rois_path = cond_dir / f"{repr_prefix}_rois.csv"

    npz = np.load(npz_path, allow_pickle=False)
    subjects = pd.read_csv(subjects_path)["subject"].astype(str).tolist()
    rois = pd.read_csv(rois_path)["roi"].astype(str).tolist()
    available_rois = set(rois)

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted, ages_sorted = sort_subjects_by_age(subjects, age_map)
    idx_map = {s: i for i, s in enumerate(subjects)}
    order = [idx_map[s] for s in subjects_sorted]

    n_sub = len(subjects_sorted)
    ages_arr = np.asarray(ages_sorted, dtype=np.float64)

    score_rows: List[Dict[str, object]] = []
    stat_rows: List[Dict[str, object]] = []

    for roi_a, roi_b in roi_pairs:
        if roi_a not in available_rois:
            raise ValueError(f"ROI 不在数据中: {roi_a}")
        if roi_b not in available_rois:
            raise ValueError(f"ROI 不在数据中: {roi_b}")
        pair_label = f"{roi_a}-{roi_b}"
        rc_scores = np.empty(n_sub, dtype=np.float64)
        for si in range(n_sub):
            rc_scores[si] = _compute_rc_single(npz, roi_a, roi_b, order[si])
        for si in range(n_sub):
            score_rows.append({
                "subject": subjects_sorted[si],
                "age": float(ages_arr[si]),
                "pair_label": pair_label,
                "RC_score": float(rc_scores[si]),
            })
        r_age, p_age, p_perm = _permutation_test_age(ages_arr, rc_scores, n_perm=n_perm, seed=seed)
        stat_rows.append({
            "pair_label": pair_label,
            "condition": str(condition),
            "r_age": r_age,
            "p_age": p_age,
            "p_perm": p_perm,
            "n_subjects": int(n_sub),
        })

    roi_network_map = get_roi_network_map(roi_set=232)

    agg = str(rc_aggregate).strip().lower()
    if agg not in {"mean", "median"}:
        raise ValueError(f"Unsupported rc_aggregate: {rc_aggregate}")

    network_rc: Dict[str, np.ndarray] = {}
    for group_a, group_b in network_pairs:
        # Keep group ROI lists to report n_roi_pairs and group sizes for interpretability.
        rois_a = sorted([r for r in _resolve_group_rois(group_a, roi_network_map) if r in available_rois])
        rois_b = sorted([r for r in _resolve_group_rois(group_b, roi_network_map) if r in available_rois])
        cross_pairs: List[Tuple[str, str]] = []
        for ra in rois_a:
            for rb in rois_b:
                cross_pairs.append((ra, rb))
        if len(cross_pairs) == 0:
            raise ValueError(f"网络对 {group_a}:{group_b} 在当前 ROI 集合中无交叉 ROI 对")
        net_label = f"{group_a}-{group_b}"
        rc_mat = np.empty((n_sub, len(cross_pairs)), dtype=np.float64)
        for pi, (ra, rb) in enumerate(cross_pairs):
            for si in range(n_sub):
                rc_mat[si, pi] = _compute_rc_single(npz, ra, rb, order[si])
        agg_rc = np.nanmedian(rc_mat, axis=1) if agg == "median" else np.nanmean(rc_mat, axis=1)
        network_rc[net_label] = agg_rc

        for si in range(n_sub):
            score_rows.append({
                "subject": subjects_sorted[si],
                "age": float(ages_arr[si]),
                "pair_label": net_label,
                "RC_score": float(agg_rc[si]),
            })
        r_age, p_age, p_perm = _permutation_test_age(ages_arr, agg_rc, n_perm=n_perm, seed=seed)
        stat_rows.append({
            "pair_label": net_label,
            "condition": str(condition),
            "r_age": r_age,
            "p_age": p_age,
            "p_perm": p_perm,
            "n_subjects": int(n_sub),
            "rc_aggregate": str(agg),
            "n_roi_pairs": int(len(cross_pairs)),
            "n_rois_group_a": int(len(rois_a)),
            "n_rois_group_b": int(len(rois_b)),
        })
        print(f"[repr_connectivity] {net_label}: n_roi_pairs={len(cross_pairs)} (agg={agg})")

    return score_rows, stat_rows, network_rc, ages_arr


def run_representational_connectivity(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition: str,
    repr_prefix: str,
    subject_info: Path,
    roi_pairs: List[str],
    network_pairs: List[str],
    n_perm: int,
    seed: int,
    dpi: int,
    out_dir: Optional[Path],
    condition_contrast: Optional[str],
    dev_modeling: bool = False,
    dev_model_criterion: str = "AIC",
    dev_model_min_n: int = 8,
    skip_repr_complexity: bool = False,
    repr_complexity_level: str = "roi",
    rc_aggregate: str = "mean",
) -> None:
    if out_dir is None:
        out_dir = matrix_dir / "gradient_analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_roi_pairs = [_parse_pair(rp) for rp in roi_pairs]
    parsed_network_pairs = [_parse_pair(np_str) for np_str in network_pairs]

    score_rows, stat_rows, network_rc, ages_arr = _compute_condition_rc(
        matrix_dir=matrix_dir,
        stimulus_dir_name=stimulus_dir_name,
        condition=condition,
        repr_prefix=repr_prefix,
        subject_info=subject_info,
        roi_pairs=parsed_roi_pairs,
        network_pairs=parsed_network_pairs,
        n_perm=n_perm,
        seed=seed,
        rc_aggregate=str(rc_aggregate),
    )

    pd.DataFrame(score_rows).to_csv(out_dir / "repr_connectivity_subject_scores.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out_dir / "repr_connectivity_stats.csv", index=False)

    # Task 7: (optional) developmental model comparison (linear vs quadratic).
    if bool(dev_modeling):
        dev_df = _compare_dev_models_linear_vs_quadratic(
            df_scores=pd.DataFrame(score_rows),
            condition=str(condition),
            criterion=str(dev_model_criterion),
            min_n=int(dev_model_min_n),
        )
        if not dev_df.empty:
            dev_df.to_csv(out_dir / "repr_connectivity_dev_model_comparison.csv", index=False)

    # Task 6: representational complexity/distinctiveness (supplementary; does not change RC outputs)
    if not bool(skip_repr_complexity):
        comp_score_rows, comp_stat_rows = _compute_condition_repr_complexity(
            matrix_dir=matrix_dir,
            stimulus_dir_name=stimulus_dir_name,
            condition=condition,
            repr_prefix=repr_prefix,
            subject_info=subject_info,
            level=str(repr_complexity_level),
            n_perm=n_perm,
            seed=seed,
        )
        pd.DataFrame(comp_score_rows).to_csv(out_dir / "repr_complexity_subject_scores.csv", index=False)
        pd.DataFrame(comp_stat_rows).to_csv(out_dir / "repr_complexity_stats.csv", index=False)

    if parsed_network_pairs:
        _plot_network_rc(
            network_rc=network_rc,
            ages=ages_arr,
            condition=str(condition),
            stat_rows=[r for r in stat_rows if r["pair_label"] in network_rc],
            out_dir=out_dir,
            dpi=dpi,
        )

    if len(parsed_network_pairs) == 2:
        _plot_contrast(
            network_rc=network_rc,
            ages=ages_arr,
            condition=str(condition),
            stat_rows=[r for r in stat_rows if r["pair_label"] in network_rc],
            out_dir=out_dir,
            dpi=dpi,
        )

    if condition_contrast:
        contrast_score_rows, contrast_stat_rows, contrast_network_rc, contrast_ages = _compute_condition_rc(
            matrix_dir=matrix_dir,
            stimulus_dir_name=stimulus_dir_name,
            condition=condition_contrast,
            repr_prefix=repr_prefix,
            subject_info=subject_info,
            roi_pairs=parsed_roi_pairs,
            network_pairs=parsed_network_pairs,
            n_perm=n_perm,
            seed=seed,
            rc_aggregate=str(rc_aggregate),
        )

        safe_name = str(condition_contrast).replace("/", "_")
        pd.DataFrame(contrast_score_rows).to_csv(
            out_dir / f"repr_connectivity_subject_scores_{safe_name}.csv", index=False
        )
        pd.DataFrame(contrast_stat_rows).to_csv(
            out_dir / f"repr_connectivity_stats_{safe_name}.csv", index=False
        )

        if bool(dev_modeling):
            contrast_dev_df = _compare_dev_models_linear_vs_quadratic(
                df_scores=pd.DataFrame(contrast_score_rows),
                condition=str(condition_contrast),
                criterion=str(dev_model_criterion),
                min_n=int(dev_model_min_n),
            )
            if not contrast_dev_df.empty:
                contrast_dev_df.to_csv(
                    out_dir / f"repr_connectivity_dev_model_comparison_{safe_name}.csv", index=False
                )

        _plot_condition_overlay(
            primary_network_rc=network_rc,
            primary_ages=ages_arr,
            primary_condition=str(condition),
            contrast_network_rc=contrast_network_rc,
            contrast_ages=contrast_ages,
            contrast_condition=str(condition_contrast),
            out_dir=out_dir,
            dpi=dpi,
        )


def _plot_network_rc(
    network_rc: Dict[str, np.ndarray],
    ages: np.ndarray,
    condition: str,
    stat_rows: List[Dict[str, object]],
    out_dir: Path,
    dpi: int,
) -> None:
    n_panels = len(network_rc)
    if n_panels == 0:
        return
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), squeeze=False)
    axes_flat = axes.flatten()

    stat_map = {str(r["pair_label"]): r for r in stat_rows}
    for idx, (net_label, rc_scores) in enumerate(network_rc.items()):
        ax = axes_flat[idx]
        mask = np.isfinite(ages) & np.isfinite(rc_scores)
        x = ages[mask]
        y = rc_scores[mask]

        ax.scatter(x, y, s=18, alpha=0.55, linewidths=0, color="#4C72B0", zorder=2)

        lowess_result = _try_lowess(x, y)
        if lowess_result is not None:
            lx, ly = lowess_result
            ax.plot(lx, ly, color="#d62728", lw=2.0, zorder=3)
        else:
            if x.size >= 3:
                lr = linregress(x, y)
                x_line = np.linspace(float(x.min()), float(x.max()), 200)
                ax.plot(x_line, lr.intercept + lr.slope * x_line, color="#d62728", lw=2.0, zorder=3)

        st = stat_map.get(net_label, {})
        r_val = st.get("r_age", float("nan"))
        p_val = st.get("p_perm", float("nan"))
        ax.annotate(
            f"r={r_val:.3f}, p_perm={p_val:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            verticalalignment="top",
        )
        ax.set_title(f"{net_label} Representational Connectivity ({condition})")
        ax.set_xlabel("Age")
        ax.set_ylabel("RC (Spearman)")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_representational_connectivity.png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_contrast(
    network_rc: Dict[str, np.ndarray],
    ages: np.ndarray,
    condition: str,
    stat_rows: List[Dict[str, object]],
    out_dir: Path,
    dpi: int,
) -> None:
    labels = list(network_rc.keys())
    if len(labels) != 2:
        return
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    stat_map = {str(r["pair_label"]): r for r in stat_rows}
    for idx, net_label in enumerate(labels):
        ax = axes[idx]
        rc_scores = network_rc[net_label]
        mask = np.isfinite(ages) & np.isfinite(rc_scores)
        x = ages[mask]
        y = rc_scores[mask]

        ax.scatter(x, y, s=18, alpha=0.55, linewidths=0, color="#4C72B0", zorder=2)

        lowess_result = _try_lowess(x, y)
        if lowess_result is not None:
            lx, ly = lowess_result
            ax.plot(lx, ly, color="#d62728", lw=2.0, zorder=3)
        else:
            if x.size >= 3:
                lr = linregress(x, y)
                x_line = np.linspace(float(x.min()), float(x.max()), 200)
                ax.plot(x_line, lr.intercept + lr.slope * x_line, color="#d62728", lw=2.0, zorder=3)

        st = stat_map.get(net_label, {})
        r_val = st.get("r_age", float("nan"))
        p_val = st.get("p_perm", float("nan"))
        ax.annotate(
            f"r={r_val:.3f}, p_perm={p_val:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            verticalalignment="top",
        )
        ax.set_title(f"{net_label} ({condition})")
        ax.set_xlabel("Age")
        ax.set_ylabel("RC (Spearman)")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_representational_connectivity_contrast.png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_condition_overlay(
    primary_network_rc: Dict[str, np.ndarray],
    primary_ages: np.ndarray,
    primary_condition: str,
    contrast_network_rc: Dict[str, np.ndarray],
    contrast_ages: np.ndarray,
    contrast_condition: str,
    out_dir: Path,
    dpi: int,
) -> None:
    common_labels = [
        label for label in primary_network_rc.keys() if label in contrast_network_rc
    ]
    if not common_labels:
        return

    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, len(common_labels), figsize=(5.5 * len(common_labels), 4.5), squeeze=False)
    axes_flat = axes.flatten()

    for idx, net_label in enumerate(common_labels):
        ax = axes_flat[idx]
        primary_scores = primary_network_rc[net_label]
        contrast_scores = contrast_network_rc[net_label]

        mask_p = np.isfinite(primary_ages) & np.isfinite(primary_scores)
        x_p = primary_ages[mask_p]
        y_p = primary_scores[mask_p]
        ax.scatter(x_p, y_p, s=18, alpha=0.45, linewidths=0, color="#4C72B0", zorder=2, label=primary_condition)
        lowess_p = _try_lowess(x_p, y_p)
        if lowess_p is not None:
            lx_p, ly_p = lowess_p
            ax.plot(lx_p, ly_p, color="#4C72B0", lw=2.0, zorder=3)

        mask_c = np.isfinite(contrast_ages) & np.isfinite(contrast_scores)
        x_c = contrast_ages[mask_c]
        y_c = contrast_scores[mask_c]
        ax.scatter(x_c, y_c, s=18, alpha=0.45, linewidths=0, color="#d62728", zorder=2, label=contrast_condition)
        lowess_c = _try_lowess(x_c, y_c)
        if lowess_c is not None:
            lx_c, ly_c = lowess_c
            ax.plot(lx_c, ly_c, color="#d62728", lw=2.0, zorder=3)

        ax.set_title(f"{net_label}: {primary_condition} vs {contrast_condition}")
        ax.set_xlabel("Age")
        ax.set_ylabel("RC (Spearman)")
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    fig.savefig(
        out_dir / "fig3_representational_connectivity_condition_overlay.png",
        dpi=int(dpi),
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_representational_connectivity(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        condition=str(args.condition),
        repr_prefix=str(args.repr_prefix),
        subject_info=Path(args.subject_info),
        roi_pairs=[str(x) for x in args.roi_pairs],
        network_pairs=[str(x) for x in args.network_pairs],
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        dpi=int(args.dpi),
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
        condition_contrast=str(args.condition_contrast) if args.condition_contrast is not None else None,
        dev_modeling=bool(args.dev_modeling),
        dev_model_criterion=str(args.dev_model_criterion),
        dev_model_min_n=int(args.dev_model_min_n),
        skip_repr_complexity=bool(args.skip_repr_complexity),
        repr_complexity_level=str(args.repr_complexity_level),
        rc_aggregate=str(args.rc_aggregate),
    )


if __name__ == "__main__":
    main()
