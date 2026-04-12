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
    p.add_argument("--mature-age-min", type=float, default=17.0)
    p.add_argument("--target-networks", nargs="+", default=["Default", "DorsAttn"])
    p.add_argument("--include-subcortical", nargs="+", default=["Hippocampus"])
    p.add_argument("--include-scan", action="store_true", default=False)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


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
    mature_age_min: float,
    target_networks: List[str],
    include_subcortical: List[str],
    include_scan: bool,
    n_perm: int,
    seed: int,
    alpha: float,
    dpi: int,
    out_dir: Optional[Path],
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
        raise ValueError(f"成熟被试数量不足（{n_mature} < 5），请降低 --mature-age-min")

    template_parts: List[np.ndarray] = []
    for roi in target_rois:
        roi_data = np.asarray(npz[roi])
        vecs_mature: List[np.ndarray] = []
        for si in range(n_sub):
            if mature_mask[si]:
                vecs_mature.append(flatten_upper(roi_data[order[si]]))
        template_parts.append(np.mean(np.stack(vecs_mature, axis=0), axis=0))
    mega_template = np.concatenate(template_parts).astype(np.float64)

    mi_scores = np.empty(n_sub, dtype=np.float64)
    for si in range(n_sub):
        subj_parts: List[np.ndarray] = []
        for roi in target_rois:
            roi_data = np.asarray(npz[roi])
            subj_parts.append(flatten_upper(roi_data[order[si]]))
        mega_vec = np.concatenate(subj_parts).astype(np.float64)
        mask_valid = np.isfinite(mega_vec) & np.isfinite(mega_template)
        if int(mask_valid.sum()) < 5:
            mi_scores[si] = float("nan")
        else:
            mi_scores[si] = float(spearmanr(mega_vec[mask_valid], mega_template[mask_valid]).statistic)

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

    beh_df: Optional[pd.DataFrame] = None
    if behavior_file is not None and Path(behavior_file).exists():
        beh_df = pd.read_csv(behavior_file)
    else:
        fallback_path = cond_dir / "behavior_subject_stimulus_scores.csv"
        if fallback_path.exists():
            beh_df = pd.read_csv(fallback_path)

    beh_merged: Optional[pd.DataFrame] = None
    if beh_df is not None:
        sub_col: Optional[str] = None
        for c in ("subject", "sub_id", "被试编号"):
            if c in beh_df.columns:
                sub_col = c
                break
        if sub_col is not None:
            beh_df[sub_col] = beh_df[sub_col].astype(str)
            beh_df[sub_col] = beh_df[sub_col].apply(
                lambda x: x if x.startswith("sub-") else f"sub-{x}"
            )
            subj_df = pd.DataFrame({
                "subject": subjects_sorted,
                "age": ages_arr,
                "maturity_index": mi_scores,
            })
            beh_merged = subj_df.merge(beh_df, left_on="subject", right_on=sub_col, how="inner")

    has_sex = False
    if beh_merged is not None and "sex" in beh_merged.columns:
        has_sex = True

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
    }]
    pd.DataFrame(config_rows).to_csv(out_dir / "maturity_index_config.csv", index=False)

    has_behavior_panel = False
    beh_col_for_plot: Optional[str] = None
    mi_resid_plot: Optional[np.ndarray] = None
    beh_resid_plot: Optional[np.ndarray] = None
    partial_r_plot: Optional[float] = None
    partial_p_plot: Optional[float] = None
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
        f"r={r_age:.3f}, p_perm={p_age_perm:.4f}",
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

    print(f"[maturity_index] MI ~ age: r={r_age:.4f}, p_perm={p_age_perm:.4f}")
    if has_behavior_panel and partial_r_plot is not None:
        print(f"[maturity_index] MI ~ {beh_col_for_plot} (partial): r={partial_r_plot:.4f}, p_perm={partial_p_plot:.4f}")
    print(f"[maturity_index] 结果已保存至 {out_dir}")


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
        mature_age_min=float(args.mature_age_min),
        target_networks=[str(x) for x in args.target_networks],
        include_subcortical=[str(x) for x in args.include_subcortical],
        include_scan=bool(args.include_scan),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        alpha=float(args.alpha),
        dpi=int(args.dpi),
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
    )


if __name__ == "__main__":
    main()
