#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_task_dissociation.py

Task Dissociation 分析：ΔISC 交互检验（condition_a vs condition_b）。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from calc_roi_isc_by_age import flatten_upper, load_subject_ages_map, sort_subjects_by_age  # noqa: E402
from joint_analysis_roi_isc_dev_models import assoc, bh_fdr, build_models, fisher_z, zscore_1d  # noqa: E402
from utils_network_assignment import DEFAULT_DISTANCE_THRESHOLD_MM, get_roi_network_map, get_scan_roi_mapping  # noqa: E402

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ΔISC Task Dissociation 置换检验")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--condition-a", type=str, default="Passive_Emo")
    p.add_argument("--condition-b", type=str, default="Reappraisal")
    p.add_argument("--model", type=str, default="M_conv", choices=("M_nn", "M_conv", "M_div"))
    p.add_argument("--isc-prefix", type=str, default="roi_isc_mahalanobis_by_age")
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--scan-distance-threshold",
        type=float,
        default=float(DEFAULT_DISTANCE_THRESHOLD_MM),
        help="SCAN 映射阈值（mm）。仅影响 SCAN 标注列，不影响主分析流程。",
    )
    p.add_argument(
        "--direction-check",
        action="store_true",
        default=False,
        help="输出 ΔISC 方向/网络分布摘要，用于快速核对叙事一致性（提示型，不影响主分析）。",
    )
    return p.parse_args()


def _load_isc_cube(
    stim_dir: Path, isc_prefix: str
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    isc = np.load(stim_dir / f"{isc_prefix}.npy")
    sub_df = pd.read_csv(stim_dir / f"{isc_prefix}_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / f"{isc_prefix}_rois.csv")
    subjects = sub_df["subject"].astype(str).tolist()
    ages = sub_df["age"].astype(float).to_numpy()
    rois = roi_df["roi"].astype(str).tolist()
    return isc, subjects, rois, ages


def _align_subjects(
    isc_a: np.ndarray,
    subjects_a: List[str],
    ages_a: np.ndarray,
    isc_b: np.ndarray,
    subjects_b: List[str],
    ages_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    set_a = set(subjects_a)
    set_b = set(subjects_b)
    common = sorted(set_a & set_b)
    if len(common) == 0:
        raise ValueError("两个条件之间没有共同被试")

    age_map = {s: float(a) for s, a in zip(subjects_a, ages_a)}
    age_map.update({s: float(a) for s, a in zip(subjects_b, ages_b)})

    sorted_subs, sorted_ages = sort_subjects_by_age(common, age_map)

    idx_a = {s: i for i, s in enumerate(subjects_a)}
    idx_b = {s: i for i, s in enumerate(subjects_b)}
    order_a = [idx_a[s] for s in sorted_subs]
    order_b = [idx_b[s] for s in sorted_subs]

    isc_a_aligned = isc_a[:, order_a, :][:, :, order_a]
    isc_b_aligned = isc_b[:, order_b, :][:, :, order_b]

    return isc_a_aligned, isc_b_aligned, sorted_subs, np.asarray(sorted_ages, dtype=np.float64)


def _select_model_vec(
    model_name: str, ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool = True
) -> np.ndarray:
    m_nn, m_conv, m_div = build_models(ages, iu, ju, normalize=normalize)
    mapping = {"M_nn": m_nn, "M_conv": m_conv, "M_div": m_div}
    return mapping[model_name]


def run_dissociation(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition_a: str,
    condition_b: str,
    model: str,
    isc_prefix: str,
    assoc_method: str,
    n_perm: int,
    seed: int,
    alpha: float,
    scan_distance_threshold: float = float(DEFAULT_DISTANCE_THRESHOLD_MM),
    direction_check: bool = False,
) -> pd.DataFrame:
    by_stim = matrix_dir / str(stimulus_dir_name)
    dir_a = by_stim / str(condition_a)
    dir_b = by_stim / str(condition_b)

    isc_a, subjects_a, rois_a, ages_a = _load_isc_cube(dir_a, isc_prefix)
    isc_b, subjects_b, rois_b, ages_b = _load_isc_cube(dir_b, isc_prefix)

    if rois_a != rois_b:
        raise ValueError("两个条件的 ROI 列表不一致")
    rois = rois_a

    isc_a, isc_b, subjects, ages = _align_subjects(
        isc_a, subjects_a, ages_a, isc_b, subjects_b, ages_b
    )

    n_sub = len(subjects)
    n_rois = len(rois)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)

    assoc_m = str(assoc_method).strip().lower()
    S_z = np.empty((n_rois, n_pairs), dtype=np.float32)
    for ri in range(n_rois):
        vec_a = isc_a[ri][iu, ju]
        vec_b = isc_b[ri][iu, ju]
        delta = vec_a - vec_b
        if assoc_m == "spearman":
            mask_v = np.isfinite(delta)
            vr = np.full_like(delta, np.nan, dtype=np.float32)
            vr[mask_v] = rankdata(delta[mask_v]).astype(np.float32, copy=False)
            S_z[ri] = zscore_1d(vr)
        else:
            S_z[ri] = zscore_1d(delta.astype(np.float32))

    m_obs = _select_model_vec(model, ages, iu, ju, normalize=True)
    r_obs = assoc(S_z, m_obs, method=str(assoc_method))

    c_raw = np.zeros(n_rois, dtype=np.int64)
    c_fwer = np.zeros(n_rois, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    for _ in range(int(n_perm)):
        ages_p = ages[rng.permutation(n_sub)]
        m_perm = _select_model_vec(model, ages_p, iu, ju, normalize=True)
        r_perm = assoc(S_z, m_perm, method=str(assoc_method))

        valid = np.isfinite(r_perm) & np.isfinite(r_obs)
        c_raw[valid] += (np.abs(r_perm[valid]) >= np.abs(r_obs[valid]))

        valid_any = np.isfinite(r_perm)
        if bool(valid_any.any()):
            max_abs = float(np.nanmax(np.abs(r_perm[valid_any])))
            idx_obs_ok = np.where(np.isfinite(r_obs))[0]
            c_fwer[idx_obs_ok] += (max_abs >= np.abs(r_obs[idx_obs_ok]))

    ok = np.isfinite(r_obs)
    p_raw = np.full(n_rois, np.nan, dtype=np.float64)
    p_fwer = np.full(n_rois, np.nan, dtype=np.float64)
    p_raw[ok] = (c_raw[ok] + 1.0) / (float(n_perm) + 1.0)
    p_fwer[ok] = (c_fwer[ok] + 1.0) / (float(n_perm) + 1.0)
    p_fdr = bh_fdr(p_raw)

    rows = []
    for ri, roi in enumerate(rois):
        r_val = float(r_obs[ri])
        p_two = float(p_raw[ri])
        p_fw = float(p_fwer[ri])
        p_fd = float(p_fdr[ri])
        sig = p_fd <= float(alpha)
        if sig and r_val > 0:
            label = "condition_a_dominant"
        elif sig and r_val < 0:
            label = "condition_b_dominant"
        else:
            label = "ns"
        rows.append({
            "roi": roi,
            "condition_a": str(condition_a),
            "condition_b": str(condition_b),
            "model": str(model),
            "r_obs_delta": r_val,
            "p_perm_two_tailed": p_two,
            "p_fwer": p_fw,
            "p_fdr_bh": p_fd,
            "n_subjects": int(n_sub),
            "n_pairs": int(n_pairs),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "dissociation_label": label,
        })

    out_df = pd.DataFrame(rows).sort_values("roi")

    if bool(direction_check):
        roi_net_map = get_roi_network_map(roi_set=232)
        out_df["network_or_structure"] = out_df["roi"].map(lambda x: roi_net_map.get(str(x), "Unknown"))
        sig_df = out_df[out_df["p_fdr_bh"].astype(float) <= float(alpha)].copy()
        if sig_df.empty:
            print("[dissociation][direction_check] No FDR-significant ROIs (skip network summary).")
        else:
            tab = (
                sig_df.groupby(["dissociation_label", "network_or_structure"], as_index=False)
                .size()
                .sort_values(["dissociation_label", "size"], ascending=[True, False])
            )
            print("[dissociation][direction_check] Significant ROI count by label x network:")
            for _, rr in tab.iterrows():
                print(f"  {rr['dissociation_label']} | {rr['network_or_structure']}: {int(rr['size'])}")

            if "passive" in str(condition_a).lower():
                n_sommot = int(sig_df[(sig_df["dissociation_label"] == "condition_a_dominant") & (sig_df["network_or_structure"] == "SomMot")].shape[0])
                n_amy = int(sig_df[(sig_df["dissociation_label"] == "condition_a_dominant") & (sig_df["network_or_structure"] == "Amygdala")].shape[0])
                if (n_sommot + n_amy) > 0:
                    print(
                        "[dissociation][direction_check][note] "
                        "condition_a=Passive 但出现 Passive-dominant 的 SomMot/Amygdala 显著 ROI；"
                        "这表示这些 ROI 的 ΔISC 与发育模型正相关，是否与“早期成熟/稳定”叙事一致需核对。"
                    )

    try:
        thr = float(scan_distance_threshold)
        scan_map = get_scan_roi_mapping(distance_threshold=thr)
        scan_lookup: dict[str, tuple[str, float]] = {}
        for _, row in scan_map.iterrows():
            r = str(row["nearest_roi"])
            if r not in scan_lookup or float(row["distance_mm"]) < scan_lookup[r][1]:
                scan_lookup[r] = (str(row["scan_region"]), float(row["distance_mm"]))
        out_df["is_scan_overlap"] = out_df["roi"].apply(
            lambda x: x in scan_lookup and scan_lookup[x][1] <= thr
        )
        out_df["nearest_scan_region"] = out_df["roi"].apply(
            lambda x: scan_lookup.get(x, ("", 999.0))[0]
        )
        out_df["scan_distance_mm"] = out_df["roi"].apply(
            lambda x: scan_lookup.get(x, ("", 999.0))[1]
        )
    except Exception as e:
        warnings.warn(f"SCAN 标注失败（已跳过，不影响主分析输出）：{e}")
        out_df["is_scan_overlap"] = False
        out_df["nearest_scan_region"] = ""
        out_df["scan_distance_mm"] = float("nan")

    out_dir = matrix_dir / "gradient_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "roi_isc_task_dissociation.csv", index=False)

    n_scan_a = int(out_df[(out_df["dissociation_label"] == "condition_a_dominant") & out_df["is_scan_overlap"]].shape[0])
    n_scan_b = int(out_df[(out_df["dissociation_label"] == "condition_b_dominant") & out_df["is_scan_overlap"]].shape[0])
    n_a = int((out_df["dissociation_label"] == "condition_a_dominant").sum())
    n_b = int((out_df["dissociation_label"] == "condition_b_dominant").sum())
    print(f"[dissociation] SCAN overlap: {condition_a}-dominant {n_scan_a}/{n_a}, {condition_b}-dominant {n_scan_b}/{n_b}")

    return out_df


def main() -> None:
    args = parse_args()
    run_dissociation(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        condition_a=str(args.condition_a),
        condition_b=str(args.condition_b),
        model=str(args.model),
        isc_prefix=str(args.isc_prefix),
        assoc_method=str(args.assoc_method),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        alpha=float(args.alpha),
        scan_distance_threshold=float(args.scan_distance_threshold),
        direction_check=bool(args.direction_check),
    )


if __name__ == "__main__":
    main()
