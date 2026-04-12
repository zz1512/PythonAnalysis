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

    network_rc: Dict[str, np.ndarray] = {}
    for group_a, group_b in network_pairs:
        cross_pairs = _get_cross_group_pairs(group_a, group_b, roi_network_map, available_rois)
        if len(cross_pairs) == 0:
            raise ValueError(f"网络对 {group_a}:{group_b} 在当前 ROI 集合中无交叉 ROI 对")
        net_label = f"{group_a}-{group_b}"
        rc_mat = np.empty((n_sub, len(cross_pairs)), dtype=np.float64)
        for pi, (ra, rb) in enumerate(cross_pairs):
            for si in range(n_sub):
                rc_mat[si, pi] = _compute_rc_single(npz, ra, rb, order[si])
        mean_rc = np.nanmean(rc_mat, axis=1)
        network_rc[net_label] = mean_rc

        for si in range(n_sub):
            score_rows.append({
                "subject": subjects_sorted[si],
                "age": float(ages_arr[si]),
                "pair_label": net_label,
                "RC_score": float(mean_rc[si]),
            })
        r_age, p_age, p_perm = _permutation_test_age(ages_arr, mean_rc, n_perm=n_perm, seed=seed)
        stat_rows.append({
            "pair_label": net_label,
            "condition": str(condition),
            "r_age": r_age,
            "p_age": p_age,
            "p_perm": p_perm,
            "n_subjects": int(n_sub),
        })

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
    )

    pd.DataFrame(score_rows).to_csv(out_dir / "repr_connectivity_subject_scores.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out_dir / "repr_connectivity_stats.csv", index=False)

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
        )

        safe_name = str(condition_contrast).replace("/", "_")
        pd.DataFrame(contrast_score_rows).to_csv(
            out_dir / f"repr_connectivity_subject_scores_{safe_name}.csv", index=False
        )
        pd.DataFrame(contrast_stat_rows).to_csv(
            out_dir / f"repr_connectivity_stats_{safe_name}.csv", index=False
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
    )


if __name__ == "__main__":
    main()
