#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils_network_assignment import (  # noqa: E402
    DEFAULT_DISTANCE_THRESHOLD_MM,
    get_schaefer200_network_labels,
    get_sensorimotor_association_order,
    get_scan_cortical_rois,
    get_scan_roi_mapping,
    SCAN_FUNCTIONAL_GROUPS,
    SCAN_REGIONS,
)

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")

CORTICAL_ROIS: List[str] = [f"L_{i}" for i in range(1, 101)] + [f"R_{i}" for i in range(1, 101)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S-A Gradient Shift 分析：将发育效应映射到 Margulies (2016) 主梯度")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--condition-a", type=str, default="Passive_Emo")
    p.add_argument("--condition-b", type=str, default="Reappraisal")
    p.add_argument("--model", type=str, default="M_conv")
    p.add_argument("--perm-result-file", type=str, default="roi_isc_dev_models_perm_fwer.csv")
    p.add_argument("--gradient-source", type=str, default="neuromaps", choices=["neuromaps", "manual"])
    p.add_argument("--gradient-file", type=Path, default=None)
    p.add_argument("--atlas-lh", type=Path, default=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"))
    p.add_argument("--atlas-rh", type=Path, default=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"))
    p.add_argument("--n-spin", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=300)

    # SCAN robustness / negative control (Task 2)
    p.add_argument(
        "--scan-atlas-source",
        type=str,
        default="nilearn",
        choices=["nilearn", "nifti", "centroids", "auto"],
        help="SCAN 映射用 Schaefer200 centroid 来源：nilearn(在线)/nifti(本地)/centroids(CSV/NPY)/auto(优先离线)",
    )
    p.add_argument("--scan-atlas-nifti", type=Path, default=None, help="SCAN 映射用本地 Schaefer200 atlas NIfTI（.nii/.nii.gz）")
    p.add_argument("--scan-centroids-file", type=Path, default=None, help="SCAN 映射用本地 centroid 缓存（.csv/.npy）")
    p.add_argument(
        "--scan-distance-threshold",
        type=float,
        default=float(DEFAULT_DISTANCE_THRESHOLD_MM),
        help="SCAN 主分析阈值（用于 fig/scan_overlap/负对照的观测 ROI 集合）",
    )
    p.add_argument(
        "--scan-threshold-grid",
        type=float,
        nargs="+",
        default=[10.0, 15.0, 20.0, 25.0],
        help="SCAN 阈值敏感性阈值网格（mm），例如：--scan-threshold-grid 10 15 20 25",
    )
    p.add_argument(
        "--scan-negctl-n",
        type=int,
        default=1000,
        help="负对照重复次数 N（从指定网络随机抽样与 SCAN ROI 数量相同的 parcels）",
    )
    p.add_argument(
        "--scan-negctl-networks",
        type=str,
        nargs="+",
        default=["SomMot"],
        help="负对照抽样网络集合，例如：--scan-negctl-networks SomMot Vis Limbic",
    )
    return p.parse_args()


def _load_r_obs(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition: str,
    model: str,
    perm_result_file: str,
) -> Tuple[np.ndarray, List[str]]:
    csv_path = matrix_dir / str(stimulus_dir_name) / str(condition) / str(perm_result_file)
    df = pd.read_csv(csv_path)
    df = df[df["model"] == str(model)].copy()
    df = df[df["roi"].isin(CORTICAL_ROIS)].copy()
    df = df.set_index("roi").loc[CORTICAL_ROIS].reset_index()
    r_obs = df["r_obs"].to_numpy(dtype=np.float64)
    rois = df["roi"].astype(str).tolist()
    return r_obs, rois


def _parcellate_gradient_neuromaps(
    atlas_lh: Path, atlas_rh: Path
) -> np.ndarray:
    try:
        from neuromaps.datasets import fetch_annotation
        from neuromaps.images import parcellate
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "neuromaps 未安装。请运行: pip install neuromaps\n"
            "同时确保已安装 nibabel: pip install nibabel"
        )

    g1_lh, g1_rh = fetch_annotation(
        source="margulies2016", desc="fcgradient01", space="fsLR", den="32k"
    )

    atlas_lh_img = nib.load(str(atlas_lh))
    atlas_rh_img = nib.load(str(atlas_rh))

    g1_lh_img = nib.load(str(g1_lh)) if isinstance(g1_lh, (str, Path)) else g1_lh
    g1_rh_img = nib.load(str(g1_rh)) if isinstance(g1_rh, (str, Path)) else g1_rh

    lh_labels = atlas_lh_img.darrays[0].data.astype(int)
    rh_labels = atlas_rh_img.darrays[0].data.astype(int)

    g1_lh_data = g1_lh_img.darrays[0].data.astype(np.float64)
    g1_rh_data = g1_rh_img.darrays[0].data.astype(np.float64)

    g1_scores = np.zeros(200, dtype=np.float64)

    for i in range(1, 101):
        mask = lh_labels == i
        if int(mask.sum()) > 0:
            g1_scores[i - 1] = float(np.nanmean(g1_lh_data[mask]))
        else:
            g1_scores[i - 1] = np.nan

    for i in range(1, 101):
        mask = rh_labels == i
        if int(mask.sum()) > 0:
            g1_scores[99 + i] = float(np.nanmean(g1_rh_data[mask]))
        else:
            g1_scores[99 + i] = np.nan

    return g1_scores


def _load_gradient_manual(gradient_file: Path) -> np.ndarray:
    df = pd.read_csv(gradient_file)
    df = df.set_index("roi").loc[CORTICAL_ROIS].reset_index()
    return df["g1_score"].to_numpy(dtype=np.float64)


def _spin_test(
    g1_scores: np.ndarray,
    r_obs_vec: np.ndarray,
    atlas_lh: Path,
    atlas_rh: Path,
    n_spin: int,
    seed: int,
) -> Tuple[float, float, Dict[str, object]]:
    observed_r, _ = spearmanr(g1_scores, r_obs_vec, nan_policy="omit")

    try:
        from neuromaps.nulls import alexander_bloch

        g1_lh_arr = g1_scores[:100]
        g1_rh_arr = g1_scores[100:]

        nulls = alexander_bloch(
            g1_lh_arr,
            g1_rh_arr,
            atlas="fsLR",
            density="32k",
            n_perm=int(n_spin),
            seed=int(seed),
            parcellation=(str(atlas_lh), str(atlas_rh)),
        )

        nulls_arr = np.asarray(nulls)
        # neuromaps currently returns (n_parcels, n_perm) for parcellated data. We harden against
        # format changes to avoid silent misalignment.
        if nulls_arr.ndim != 2:
            raise ValueError(f"Unexpected nulls ndim={nulls_arr.ndim} (expected 2)")
        if nulls_arr.shape[0] == 200:
            nulls_parcel_perm = nulls_arr
        elif nulls_arr.shape[1] == 200:
            nulls_parcel_perm = nulls_arr.T
        else:
            raise ValueError(
                f"Unexpected nulls shape={nulls_arr.shape} (expected 200 parcels on one axis)"
            )
        if int(nulls_parcel_perm.shape[0]) != 200:
            raise AssertionError(
                f"Expected 200 parcels, got {int(nulls_parcel_perm.shape[0])}"
            )

        null_rs = np.zeros(int(n_spin), dtype=np.float64)
        for pi in range(int(n_spin)):
            rotated = nulls_parcel_perm[:, pi]
            r_null, _ = spearmanr(rotated, r_obs_vec, nan_policy="omit")
            null_rs[pi] = r_null

        p_spin = float((np.sum(np.abs(null_rs) >= np.abs(observed_r)) + 1) / (int(n_spin) + 1))
        meta: Dict[str, object] = {
            "spin_test_available": True,
            "spin_test_method": "alexander_bloch",
            "nulls_shape": f"{int(nulls_parcel_perm.shape[0])}x{int(nulls_parcel_perm.shape[1])}",
            "spin_test_note": "",
        }
        return float(observed_r), p_spin, meta

    except Exception as e:
        warnings.warn(
            f"neuromaps spin test 失败 ({e})，退回随机置换近似。"
        )
        rng = np.random.default_rng(int(seed))
        null_rs = np.zeros(int(n_spin), dtype=np.float64)
        for pi in range(int(n_spin)):
            perm_idx = rng.permutation(len(g1_scores))
            r_null, _ = spearmanr(g1_scores[perm_idx], r_obs_vec, nan_policy="omit")
            null_rs[pi] = r_null

        p_spin = float((np.sum(np.abs(null_rs) >= np.abs(observed_r)) + 1) / (int(n_spin) + 1))
        meta = {
            "spin_test_available": False,
            "spin_test_method": "random_permutation",
            "nulls_shape": "",
            "spin_test_note": "exploratory_fallback_no_spatial_autocorr",
        }
        return float(observed_r), p_spin, meta


def _make_figure(
    g1_scores: np.ndarray,
    r_obs_a: np.ndarray,
    r_obs_b: np.ndarray,
    condition_a: str,
    condition_b: str,
    r_gradient_a: float,
    p_spin_a: float,
    r_gradient_b: float,
    p_spin_b: float,
    network_labels: pd.DataFrame,
    sa_order: List[str],
    out_path: Path,
    dpi: int,
    scan_rois: Optional[List[str]] = None,
    scan_map: Optional[pd.DataFrame] = None,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax0 = axes[0, 0]
    ax0.scatter(g1_scores, r_obs_a, c="tab:blue", alpha=0.5, s=15, label=str(condition_a), zorder=2)
    ax0.scatter(g1_scores, r_obs_b, c="tab:red", alpha=0.5, s=15, label=str(condition_b), zorder=2)

    mask_a = np.isfinite(g1_scores) & np.isfinite(r_obs_a)
    if int(mask_a.sum()) > 1:
        z_a = np.polyfit(g1_scores[mask_a], r_obs_a[mask_a], 1)
        x_line = np.linspace(float(np.nanmin(g1_scores)), float(np.nanmax(g1_scores)), 100)
        ax0.plot(x_line, np.polyval(z_a, x_line), c="tab:blue", lw=1.5, zorder=3)

    mask_b = np.isfinite(g1_scores) & np.isfinite(r_obs_b)
    if int(mask_b.sum()) > 1:
        z_b = np.polyfit(g1_scores[mask_b], r_obs_b[mask_b], 1)
        x_line = np.linspace(float(np.nanmin(g1_scores)), float(np.nanmax(g1_scores)), 100)
        ax0.plot(x_line, np.polyval(z_b, x_line), c="tab:red", lw=1.5, zorder=3)

    ax0.set_xlabel("Sensorimotor \u2190 G1 Gradient \u2192 Association")
    ax0.set_ylabel("Developmental Effect (M_conv r)")
    p_str_a = f"p<0.001" if p_spin_a < 0.001 else f"p={p_spin_a:.3f}"
    p_str_b = f"p<0.001" if p_spin_b < 0.001 else f"p={p_spin_b:.3f}"
    ax0.set_title(
        f"{condition_a}: r={r_gradient_a:.3f}, {p_str_a}\n"
        f"{condition_b}: r={r_gradient_b:.3f}, {p_str_b}",
        fontsize=9,
    )
    ax0.legend(fontsize=8, loc="best")

    ax1 = axes[0, 1]
    roi_to_net: Dict[str, str] = dict(zip(network_labels["roi"], network_labels["network"]))
    net_means_a: Dict[str, float] = {}
    net_means_b: Dict[str, float] = {}
    for net in sa_order:
        idx = [i for i, roi in enumerate(CORTICAL_ROIS) if roi_to_net.get(roi) == net]
        if idx:
            net_means_a[net] = float(np.nanmean(r_obs_a[idx]))
            net_means_b[net] = float(np.nanmean(r_obs_b[idx]))

    x_pos = np.arange(len(sa_order))
    width = 0.35
    vals_a = [net_means_a.get(n, 0.0) for n in sa_order]
    vals_b = [net_means_b.get(n, 0.0) for n in sa_order]
    ax1.bar(x_pos - width / 2, vals_a, width, label=str(condition_a), color="tab:blue", alpha=0.8)
    ax1.bar(x_pos + width / 2, vals_b, width, label=str(condition_b), color="tab:red", alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sa_order, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Network (S \u2192 A)")
    ax1.set_ylabel("Mean Developmental Effect (r)")
    ax1.legend(fontsize=8, loc="best")
    ax1.set_title("Network-level Gradient Shift")

    scan_rois_set = set(scan_rois) if scan_rois else set()

    ax2 = axes[1, 0]
    if scan_rois_set:
        scan_roi_to_region: Dict[str, str] = {}
        if scan_map is not None and not scan_map.empty:
            cortical_map = scan_map[scan_map["is_cortical"] & scan_map["within_threshold"]]
            for _, row in cortical_map.iterrows():
                scan_roi_to_region[row["nearest_roi"]] = row["scan_region"]

        for i, roi in enumerate(CORTICAL_ROIS):
            if roi not in scan_rois_set:
                ax2.scatter(
                    g1_scores[i], r_obs_a[i], s=8, c="lightgrey", alpha=0.3, zorder=1,
                )
                ax2.scatter(
                    g1_scores[i], r_obs_b[i], s=8, c="lightgrey", alpha=0.3, zorder=1,
                )

        for i, roi in enumerate(CORTICAL_ROIS):
            if roi in scan_rois_set:
                ax2.scatter(
                    g1_scores[i], r_obs_a[i],
                    s=60, marker="*", c="tab:blue", edgecolors="black",
                    linewidths=0.5, zorder=5, label=str(condition_a) if i == 0 or roi == sorted(scan_rois_set)[0] else "",
                )
                ax2.scatter(
                    g1_scores[i], r_obs_b[i],
                    s=60, marker="*", c="tab:red", edgecolors="black",
                    linewidths=0.5, zorder=5, label=str(condition_b) if i == 0 or roi == sorted(scan_rois_set)[0] else "",
                )
                region_name = scan_roi_to_region.get(roi, roi)
                ax2.annotate(
                    region_name, (g1_scores[i], r_obs_a[i]),
                    fontsize=5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points",
                )

        handles, labels = ax2.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax2.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    else:
        ax2.scatter(g1_scores, r_obs_a, c="tab:blue", alpha=0.5, s=15, label=str(condition_a))
        ax2.scatter(g1_scores, r_obs_b, c="tab:red", alpha=0.5, s=15, label=str(condition_b))
        ax2.legend(fontsize=8, loc="best")

    ax2.set_xlabel("Sensorimotor \u2190 G1 Gradient \u2192 Association")
    ax2.set_ylabel("Developmental Effect (M_conv r)")
    ax2.set_title("SCAN ROIs on S-A Gradient", fontsize=9)

    ax3 = axes[1, 1]
    target_groups = ["inter_effector", "medial_motor"]
    group_labels_plot: List[str] = []
    group_vals_a: List[float] = []
    group_vals_b: List[float] = []

    for grp in target_groups:
        grp_regions = SCAN_FUNCTIONAL_GROUPS.get(grp, [])
        grp_cortical_rois: List[str] = []
        if scan_map is not None and not scan_map.empty:
            grp_map = scan_map[
                (scan_map["scan_region"].isin(grp_regions))
                & (scan_map["is_cortical"])
                & (scan_map["within_threshold"])
            ]
            grp_cortical_rois = sorted(set(grp_map["nearest_roi"].tolist()))

        grp_idx = [i for i, roi in enumerate(CORTICAL_ROIS) if roi in grp_cortical_rois]
        mean_a = float(np.nanmean(r_obs_a[grp_idx])) if grp_idx else 0.0
        mean_b = float(np.nanmean(r_obs_b[grp_idx])) if grp_idx else 0.0
        group_labels_plot.append(grp.replace("_", " ").title())
        group_vals_a.append(mean_a)
        group_vals_b.append(mean_b)

    group_labels_plot.append("All Cortex")
    group_vals_a.append(float(np.nanmean(r_obs_a)))
    group_vals_b.append(float(np.nanmean(r_obs_b)))

    x_grp = np.arange(len(group_labels_plot))
    bar_w = 0.35
    ax3.bar(x_grp - bar_w / 2, group_vals_a, bar_w, label=str(condition_a), color="tab:blue", alpha=0.8)
    ax3.bar(x_grp + bar_w / 2, group_vals_b, bar_w, label=str(condition_b), color="tab:red", alpha=0.8)
    ax3.set_xticks(x_grp)
    ax3.set_xticklabels(group_labels_plot, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Mean Developmental Effect (r)")
    ax3.legend(fontsize=8, loc="best")
    ax3.set_title("SCAN Functional Group Comparison")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path


def _summarize_r(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
    }


def _write_scan_mapping_sensitivity(
    *,
    out_dir: Path,
    threshold_grid: List[float],
    atlas_source: str,
    atlas_nifti: Optional[Path],
    centroids_file: Optional[Path],
    r_obs_a: np.ndarray,
    r_obs_b: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> Optional[Path]:
    """
    Task 2: SCAN threshold sensitivity analysis.
    Output: scan_mapping_sensitivity.csv
    """
    roi_index_map: Dict[str, int] = {roi: i for i, roi in enumerate(CORTICAL_ROIS)}

    rows: List[dict] = []
    for thr in threshold_grid:
        try:
            scan_map = get_scan_roi_mapping(
                atlas_source=str(atlas_source),
                distance_threshold=float(thr),
                atlas_nifti=atlas_nifti,
                centroids_file=centroids_file,
            )
        except Exception as e:
            warnings.warn(f"SCAN 映射失败，已跳过阈值敏感性分析（threshold={thr}mm）：{e}")
            return None

        scan_map_c = scan_map[scan_map["is_cortical"]].copy()
        within = scan_map_c[scan_map_c["within_threshold"]].copy()
        within_rois = sorted(set(within["nearest_roi"].astype(str).tolist()))
        within_idx = [roi_index_map[r] for r in within_rois if r in roi_index_map]

        dists = within["distance_mm"].to_numpy(dtype=np.float64) if not within.empty else np.array([], dtype=np.float64)
        dists_all_c = scan_map_c["distance_mm"].to_numpy(dtype=np.float64) if not scan_map_c.empty else np.array([], dtype=np.float64)

        r_a = r_obs_a[within_idx] if within_idx else np.array([], dtype=np.float64)
        r_b = r_obs_b[within_idx] if within_idx else np.array([], dtype=np.float64)
        r_delta = (r_obs_a - r_obs_b)[within_idx] if within_idx else np.array([], dtype=np.float64)

        row = {
            "distance_threshold_mm": float(thr),
            "atlas_source": str(atlas_source),
            "atlas_nifti": str(atlas_nifti) if atlas_nifti is not None else "",
            "centroids_file": str(centroids_file) if centroids_file is not None else "",
            "n_scan_regions_total": int(len(SCAN_REGIONS)),
            "n_scan_cortical_rows": int(scan_map_c.shape[0]),
            "n_scan_cortical_within_threshold_rows": int(within.shape[0]),
            "n_scan_cortical_unique_rois_within_threshold": int(len(within_rois)),
            "distance_cortical_mean_mm": float(np.nanmean(dists_all_c)) if dists_all_c.size else float("nan"),
            "distance_cortical_median_mm": float(np.nanmedian(dists_all_c)) if dists_all_c.size else float("nan"),
            "distance_cortical_p95_mm": float(np.nanpercentile(dists_all_c, 95)) if dists_all_c.size else float("nan"),
            "distance_cortical_max_mm": float(np.nanmax(dists_all_c)) if dists_all_c.size else float("nan"),
            "distance_within_mean_mm": float(np.nanmean(dists)) if dists.size else float("nan"),
            "distance_within_median_mm": float(np.nanmedian(dists)) if dists.size else float("nan"),
            "condition_a": str(condition_a),
            "condition_b": str(condition_b),
        }

        s_a = _summarize_r(r_a)
        s_b = _summarize_r(r_b)
        s_d = _summarize_r(r_delta)
        row.update(
            {
                "r_obs_a_mean": s_a["mean"],
                "r_obs_a_median": s_a["median"],
                "r_obs_a_std": s_a["std"],
                "r_obs_b_mean": s_b["mean"],
                "r_obs_b_median": s_b["median"],
                "r_obs_b_std": s_b["std"],
                "r_obs_delta_mean": s_d["mean"],
                "r_obs_delta_median": s_d["median"],
                "r_obs_delta_std": s_d["std"],
            }
        )
        rows.append(row)

    out_path = out_dir / "scan_mapping_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _write_scan_negative_control(
    *,
    out_dir: Path,
    scan_rois: List[str],
    scan_distance_threshold: float,
    atlas_source: str,
    atlas_nifti: Optional[Path],
    centroids_file: Optional[Path],
    r_obs_a: np.ndarray,
    r_obs_b: np.ndarray,
    seed: int,
    n_repeat: int,
    network_populations: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Task 2: Negative control baseline.
    Randomly sample parcels from one or more comparison networks with same count as SCAN cortical ROI count.
    Output: scan_negative_control.csv (includes empirical p-values on observed row).
    """
    if int(n_repeat) <= 0:
        return None

    net_df = get_schaefer200_network_labels()
    target_networks = [str(x) for x in (network_populations or ["SomMot"])]
    target_networks = [x for x in target_networks if x]
    if not target_networks:
        target_networks = ["SomMot"]

    roi_index_map: Dict[str, int] = {roi: i for i, roi in enumerate(CORTICAL_ROIS)}
    scan_idx = [roi_index_map[r] for r in sorted(set(scan_rois)) if r in roi_index_map]
    n_scan = int(len(scan_idx))

    rows: List[dict] = []

    obs_mean_a = float(np.nanmean(r_obs_a[scan_idx])) if scan_idx else float("nan")
    obs_mean_b = float(np.nanmean(r_obs_b[scan_idx])) if scan_idx else float("nan")
    obs_mean_delta = float(np.nanmean((r_obs_a - r_obs_b)[scan_idx])) if scan_idx else float("nan")

    # Store observed as row 0; p-values filled after null distribution is generated.
    rows.append(
        {
            "kind": "observed",
            "iter": 0,
            "seed": int(seed),
            "distance_threshold_mm": float(scan_distance_threshold),
            "atlas_source": str(atlas_source),
            "atlas_nifti": str(atlas_nifti) if atlas_nifti is not None else "",
            "centroids_file": str(centroids_file) if centroids_file is not None else "",
            "network_population": "SCAN_observed",
            "network_population_n": int(n_scan),
            "n_rois": int(n_scan),
            "mean_r_obs_a": obs_mean_a,
            "mean_r_obs_b": obs_mean_b,
            "mean_r_obs_delta": obs_mean_delta,
            "empirical_p_two_sided": float("nan"),
            "empirical_p_greater": float("nan"),
            "empirical_p_less": float("nan"),
        }
    )

    if n_scan == 0:
        warnings.warn("SCAN cortical ROI 数为 0，负对照无法计算（已仅输出 observed 行）。")
        out_path0 = out_dir / "scan_negative_control.csv"
        pd.DataFrame(rows).to_csv(out_path0, index=False)
        return out_path0

    rng = np.random.default_rng(int(seed))
    all_null_delta: List[float] = []
    for net_name in target_networks:
        pool = net_df[net_df["network"] == net_name]["roi"].astype(str).tolist()
        pool = [r for r in pool if r in set(CORTICAL_ROIS)]
        pool = sorted(set(pool))
        if n_scan > int(len(pool)):
            warnings.warn(
                f"{net_name} 网络可用 parcel 数({len(pool)}) < SCAN ROI 数({n_scan})，该网络负对照已跳过。"
            )
            continue

        null_delta = np.zeros(int(n_repeat), dtype=np.float64)
        pool_idx = np.array([roi_index_map[r] for r in pool if r in roi_index_map], dtype=int)
        for i in range(int(n_repeat)):
            pick = rng.choice(pool_idx, size=int(n_scan), replace=False)
            mean_a = float(np.nanmean(r_obs_a[pick]))
            mean_b = float(np.nanmean(r_obs_b[pick]))
            mean_d = float(np.nanmean((r_obs_a - r_obs_b)[pick]))
            null_delta[i] = mean_d
            rows.append(
                {
                    "kind": "null",
                    "iter": int(i + 1),
                    "seed": int(seed),
                    "distance_threshold_mm": float(scan_distance_threshold),
                    "atlas_source": str(atlas_source),
                    "atlas_nifti": str(atlas_nifti) if atlas_nifti is not None else "",
                    "centroids_file": str(centroids_file) if centroids_file is not None else "",
                    "network_population": str(net_name),
                    "network_population_n": int(len(pool)),
                    "n_rois": int(n_scan),
                    "mean_r_obs_a": mean_a,
                    "mean_r_obs_b": mean_b,
                    "mean_r_obs_delta": mean_d,
                    "empirical_p_two_sided": float("nan"),
                    "empirical_p_greater": float("nan"),
                    "empirical_p_less": float("nan"),
                }
            )
        all_null_delta.extend([float(x) for x in null_delta if np.isfinite(x)])

    # Empirical p-values for observed mean_delta against the combined null distribution.
    obs = float(obs_mean_delta)
    null_ok = np.asarray(all_null_delta, dtype=np.float64)
    if np.isfinite(obs) and null_ok.size > 0:
        p_two = float((np.sum(np.abs(null_ok) >= np.abs(obs)) + 1.0) / (null_ok.size + 1.0))
        p_g = float((np.sum(null_ok >= obs) + 1.0) / (null_ok.size + 1.0))
        p_l = float((np.sum(null_ok <= obs) + 1.0) / (null_ok.size + 1.0))
    else:
        p_two, p_g, p_l = float("nan"), float("nan"), float("nan")

    rows[0]["network_population"] = "|".join(target_networks)
    rows[0]["network_population_n"] = int(sum((net_df["network"] == x).sum() for x in target_networks))
    rows[0]["empirical_p_two_sided"] = p_two
    rows[0]["empirical_p_greater"] = p_g
    rows[0]["empirical_p_less"] = p_l

    out_path = out_dir / "scan_negative_control.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def run_gradient_shift(
    matrix_dir: Path,
    stimulus_dir_name: str,
    condition_a: str,
    condition_b: str,
    model: str,
    perm_result_file: str,
    gradient_source: str,
    gradient_file: Optional[Path],
    atlas_lh: Path,
    atlas_rh: Path,
    n_spin: int,
    seed: int,
    alpha: float,
    out_dir: Optional[Path],
    dpi: int,
    scan_atlas_source: str = "nilearn",
    scan_atlas_nifti: Optional[Path] = None,
    scan_centroids_file: Optional[Path] = None,
    scan_distance_threshold: float = float(DEFAULT_DISTANCE_THRESHOLD_MM),
    scan_threshold_grid: Optional[List[float]] = None,
    scan_negctl_n: int = 1000,
    scan_negctl_networks: Optional[List[str]] = None,
) -> None:
    if out_dir is None:
        out_dir = Path(matrix_dir) / "gradient_analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r_obs_a, rois_a = _load_r_obs(matrix_dir, stimulus_dir_name, condition_a, model, perm_result_file)
    r_obs_b, rois_b = _load_r_obs(matrix_dir, stimulus_dir_name, condition_b, model, perm_result_file)

    if rois_a != rois_b:
        raise ValueError("两个条件的皮层 ROI 列表不一致")

    if str(gradient_source) == "neuromaps":
        g1_scores = _parcellate_gradient_neuromaps(atlas_lh, atlas_rh)
    elif str(gradient_source) == "manual":
        if gradient_file is None:
            raise ValueError("gradient_source='manual' 时必须指定 --gradient-file")
        g1_scores = _load_gradient_manual(Path(gradient_file))
    else:
        raise ValueError(f"未知 gradient_source: {gradient_source}")

    r_gradient_a, _ = spearmanr(g1_scores, r_obs_a, nan_policy="omit")
    r_gradient_b, _ = spearmanr(g1_scores, r_obs_b, nan_policy="omit")

    r_gradient_a_spin, p_spin_a, spin_meta_a = _spin_test(g1_scores, r_obs_a, atlas_lh, atlas_rh, n_spin, seed)
    r_gradient_b_spin, p_spin_b, spin_meta_b = _spin_test(g1_scores, r_obs_b, atlas_lh, atlas_rh, n_spin, seed)

    network_labels = get_schaefer200_network_labels()
    sa_order = get_sensorimotor_association_order()

    dissociation_path = out_dir / "roi_isc_task_dissociation.csv"
    r_gradient_delta: Optional[float] = None
    p_spin_delta: Optional[float] = None
    spin_meta_delta: Optional[Dict[str, object]] = None
    if dissociation_path.exists():
        dissoc_df = pd.read_csv(dissociation_path)
        dissoc_df = dissoc_df[dissoc_df["roi"].isin(CORTICAL_ROIS)].copy()
        dissoc_df = dissoc_df.set_index("roi").loc[CORTICAL_ROIS].reset_index()
        r_obs_delta = dissoc_df["r_obs_delta"].to_numpy(dtype=np.float64)
        r_gradient_delta, p_spin_delta, spin_meta_delta = _spin_test(
            g1_scores, r_obs_delta, atlas_lh, atlas_rh, n_spin, seed
        )

    try:
        scan_rois = get_scan_cortical_rois(
            distance_threshold=float(scan_distance_threshold),
            atlas_source=str(scan_atlas_source),
            atlas_nifti=scan_atlas_nifti,
            centroids_file=scan_centroids_file,
        )
        scan_map_df = get_scan_roi_mapping(
            atlas_source=str(scan_atlas_source),
            distance_threshold=float(scan_distance_threshold),
            atlas_nifti=scan_atlas_nifti,
            centroids_file=scan_centroids_file,
        )
    except Exception as e:
        warnings.warn(f"SCAN 映射不可用（将跳过 SCAN 标注/稳健性输出）：{e}")
        scan_rois = []
        scan_map_df = pd.DataFrame()

    _make_figure(
        g1_scores=g1_scores,
        r_obs_a=r_obs_a,
        r_obs_b=r_obs_b,
        condition_a=str(condition_a),
        condition_b=str(condition_b),
        r_gradient_a=float(r_gradient_a_spin),
        p_spin_a=float(p_spin_a),
        r_gradient_b=float(r_gradient_b_spin),
        p_spin_b=float(p_spin_b),
        network_labels=network_labels,
        sa_order=sa_order,
        out_path=out_dir / "fig2_gradient_shift.png",
        dpi=int(dpi),
        scan_rois=scan_rois,
        scan_map=scan_map_df,
    )

    results_rows = [
        {
            "condition": str(condition_a),
            "model": str(model),
            "r_gradient": float(r_gradient_a_spin),
            "p_spin": float(p_spin_a),
            "spin_test_available": bool(spin_meta_a.get("spin_test_available", False)),
            "spin_test_method": str(spin_meta_a.get("spin_test_method", "")),
            "nulls_shape": str(spin_meta_a.get("nulls_shape", "")),
            "spin_test_note": str(spin_meta_a.get("spin_test_note", "")),
            "n_cortical_rois": int(len(rois_a)),
            "n_spin": int(n_spin),
        },
        {
            "condition": str(condition_b),
            "model": str(model),
            "r_gradient": float(r_gradient_b_spin),
            "p_spin": float(p_spin_b),
            "spin_test_available": bool(spin_meta_b.get("spin_test_available", False)),
            "spin_test_method": str(spin_meta_b.get("spin_test_method", "")),
            "nulls_shape": str(spin_meta_b.get("nulls_shape", "")),
            "spin_test_note": str(spin_meta_b.get("spin_test_note", "")),
            "n_cortical_rois": int(len(rois_b)),
            "n_spin": int(n_spin),
        },
    ]
    if r_gradient_delta is not None and p_spin_delta is not None:
        smd = spin_meta_delta or {}
        results_rows.append({
            "condition": f"delta_{condition_a}_minus_{condition_b}",
            "model": str(model),
            "r_gradient": float(r_gradient_delta),
            "p_spin": float(p_spin_delta),
            "spin_test_available": bool(smd.get("spin_test_available", False)),
            "spin_test_method": str(smd.get("spin_test_method", "")),
            "nulls_shape": str(smd.get("nulls_shape", "")),
            "spin_test_note": str(smd.get("spin_test_note", "")),
            "n_cortical_rois": 200,
            "n_spin": int(n_spin),
        })
    pd.DataFrame(results_rows).to_csv(out_dir / "gradient_shift_results.csv", index=False)

    roi_to_net: Dict[str, str] = dict(zip(network_labels["roi"], network_labels["network"]))
    roi_gradient_rows = []
    for i, roi in enumerate(CORTICAL_ROIS):
        roi_gradient_rows.append({
            "roi": roi,
            "g1_score": float(g1_scores[i]),
            "r_obs_passive": float(r_obs_a[i]),
            "r_obs_reappraisal": float(r_obs_b[i]),
            "network": roi_to_net.get(roi, ""),
        })
    pd.DataFrame(roi_gradient_rows).to_csv(out_dir / "roi_gradient_scores.csv", index=False)

    roi_index_map: Dict[str, int] = {roi: i for i, roi in enumerate(CORTICAL_ROIS)}
    scan_overlap_rows: list[dict] = []
    if scan_map_df is not None and not scan_map_df.empty:
        for _, srow in scan_map_df.iterrows():
            nearest = srow["nearest_roi"]
            idx = roi_index_map.get(nearest)
            r_a = float(r_obs_a[idx]) if idx is not None else float("nan")
            r_b = float(r_obs_b[idx]) if idx is not None else float("nan")
            scan_overlap_rows.append({
                "scan_region": srow["scan_region"],
                "nearest_roi": nearest,
                "distance_mm": float(srow["distance_mm"]),
                "r_obs_cond_a": r_a,
                "r_obs_cond_b": r_b,
                "roi_network": srow.get("roi_network", ""),
            })
    if scan_overlap_rows:
        pd.DataFrame(scan_overlap_rows).to_csv(out_dir / "scan_overlap_analysis.csv", index=False)

    # Task 2 outputs: sensitivity + negative control baseline (new files only).
    thr_grid = scan_threshold_grid if scan_threshold_grid is not None else [10.0, 15.0, 20.0, 25.0]
    thr_grid = [float(x) for x in thr_grid]
    thr_grid = sorted(set(thr_grid))

    sens_path = _write_scan_mapping_sensitivity(
        out_dir=out_dir,
        threshold_grid=thr_grid,
        atlas_source=str(scan_atlas_source),
        atlas_nifti=scan_atlas_nifti,
        centroids_file=scan_centroids_file,
        r_obs_a=r_obs_a,
        r_obs_b=r_obs_b,
        condition_a=str(condition_a),
        condition_b=str(condition_b),
    )

    negctl_path = _write_scan_negative_control(
        out_dir=out_dir,
        scan_rois=scan_rois,
        scan_distance_threshold=float(scan_distance_threshold),
        atlas_source=str(scan_atlas_source),
        atlas_nifti=scan_atlas_nifti,
        centroids_file=scan_centroids_file,
        r_obs_a=r_obs_a,
        r_obs_b=r_obs_b,
        seed=int(seed),
        n_repeat=int(scan_negctl_n),
        network_populations=[str(x) for x in (scan_negctl_networks or ["SomMot"])],
    )

    print(f"[gradient_shift] {condition_a}: r={r_gradient_a_spin:.4f}, p_spin={p_spin_a:.4f}")
    print(f"[gradient_shift] {condition_b}: r={r_gradient_b_spin:.4f}, p_spin={p_spin_b:.4f}")
    if r_gradient_delta is not None:
        print(f"[gradient_shift] ΔISC: r={r_gradient_delta:.4f}, p_spin={p_spin_delta:.4f}")
    if scan_rois:
        print(f"[gradient_shift] SCAN cortical ROIs identified: {len(scan_rois)}")
        for sr in scan_rois:
            idx = roi_index_map.get(sr)
            if idx is not None:
                print(f"  {sr}: {condition_a} r={r_obs_a[idx]:.4f}, {condition_b} r={r_obs_b[idx]:.4f}")
    if scan_overlap_rows:
        print(f"[gradient_shift] SCAN overlap analysis saved ({len(scan_overlap_rows)} entries)")
    if sens_path is not None:
        print(f"[gradient_shift] SCAN mapping sensitivity saved: {sens_path}")
    if negctl_path is not None:
        try:
            nc = pd.read_csv(negctl_path)
            obs_row = nc[nc["kind"] == "observed"].head(1)
            if not obs_row.empty:
                p_two = float(obs_row.iloc[0].get("empirical_p_two_sided"))
                n_rois = int(obs_row.iloc[0].get("n_rois"))
                print(f"[gradient_shift] SCAN negative control saved: {negctl_path} (n_rois={n_rois}, p_two={p_two:.4g})")
            else:
                print(f"[gradient_shift] SCAN negative control saved: {negctl_path}")
        except Exception as e:
            warnings.warn(f"SCAN 负对照摘要读取失败（已忽略，仅提示）：{e}")
            print(f"[gradient_shift] SCAN negative control saved: {negctl_path}")
    print(f"[gradient_shift] 结果已保存至 {out_dir}")


def main() -> None:
    args = parse_args()
    run_gradient_shift(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        condition_a=str(args.condition_a),
        condition_b=str(args.condition_b),
        model=str(args.model),
        perm_result_file=str(args.perm_result_file),
        gradient_source=str(args.gradient_source),
        gradient_file=args.gradient_file,
        atlas_lh=Path(args.atlas_lh),
        atlas_rh=Path(args.atlas_rh),
        n_spin=int(args.n_spin),
        seed=int(args.seed),
        alpha=float(args.alpha),
        out_dir=args.out_dir,
        dpi=int(args.dpi),
        scan_atlas_source=str(args.scan_atlas_source),
        scan_atlas_nifti=args.scan_atlas_nifti,
        scan_centroids_file=args.scan_centroids_file,
        scan_distance_threshold=float(args.scan_distance_threshold),
        scan_threshold_grid=[float(x) for x in (args.scan_threshold_grid or [])],
        scan_negctl_n=int(args.scan_negctl_n),
        scan_negctl_networks=[str(x) for x in (args.scan_negctl_networks or ["SomMot"])],
    )


if __name__ == "__main__":
    main()
