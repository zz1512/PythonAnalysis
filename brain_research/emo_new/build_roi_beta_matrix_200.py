#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_roi_beta_matrix_200.py

执行顺序（新链路 Step 1/3）：
1) calc_mean_beta_by_stimulus_type.py 先生成 avg_beta_index_surface_L/R.csv（如果已生成可跳过）
2) 本脚本：build_roi_beta_matrix_200.py
3) calc_roi_isc_by_age.py
4) joint_analysis_roi_isc_dev_models_perm_fwer.py

基于 avg_beta_index_surface_L/R.csv（stimulus_type 级“全脑 beta 平均图”），构建 ROI×被试×beta 的矩阵文件。

与 emo/build_roi_beta_matrix_232.py 一致的点：
- 80% 覆盖率筛选 stimulus_type，再筛出拥有全部这些 stimulus_type 的被试
- 输出 subjects/rois/stimuli 侧表

不同点（按 emo_new 设定）：
- ROI=200（Schaefer surface L/R）
- 每个 ROI 的“beta”不是均值标量，而是 ROI 内顶点向量（保留空间信息）
- 每个被试的 ROI 向量为 concat(stimulus_type1 ROI vertices, stimulus_type2 ROI vertices, ...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import surface

DEFAULT_AVG_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/averaged_betas_by_stimulus")
DEFAULT_OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")

ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 roi_beta_matrix_200（ROI 顶点向量特征，stimulus_type 级平均）")
    p.add_argument("--avg-dir", type=Path, default=DEFAULT_AVG_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold-ratio", type=float, default=0.80)
    return p.parse_args()


def load_atlas(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(int)


def prepare_roi_indices(atlas_labels: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
    labels = np.asarray(atlas_labels, dtype=int).reshape(-1)
    roi_ids = np.unique(labels)
    roi_ids = roi_ids[roi_ids > 0].astype(int)
    roi_ids = sorted(roi_ids.tolist())
    roi_indices = [np.where(labels == rid)[0] for rid in roi_ids]
    if not roi_indices:
        raise ValueError("Atlas 中未找到 ROI")
    return roi_ids, roi_indices


def intelligent_filter_pairs(df: pd.DataFrame, threshold_ratio: float) -> Tuple[List[str], List[str]]:
    df = df.copy()
    if "subject" not in df.columns or "stimulus_type" not in df.columns:
        raise ValueError("索引文件缺少 subject 或 stimulus_type 列")

    df["subject"] = df["subject"].astype(str).str.strip()
    df["stimulus_type"] = df["stimulus_type"].astype(str).str.strip()
    df = df[(df["subject"].str.len() > 0) & (df["stimulus_type"].str.len() > 0)].copy()
    unique_pairs = df[["subject", "stimulus_type"]].drop_duplicates()
    n_total_subjects = int(unique_pairs["subject"].nunique())
    if n_total_subjects <= 0:
        raise ValueError("索引中无有效被试")

    stim_counts = unique_pairs["stimulus_type"].value_counts()
    threshold_count = float(n_total_subjects) * float(threshold_ratio)
    valid_stimuli = stim_counts[stim_counts >= threshold_count].index.tolist()
    valid_stimuli = sorted([str(x) for x in valid_stimuli])
    if not valid_stimuli:
        raise ValueError("没有 stimulus_type 满足覆盖率阈值")

    df_valid_stim = unique_pairs[unique_pairs["stimulus_type"].isin(valid_stimuli)]
    sub_counts = df_valid_stim.groupby("subject")["stimulus_type"].nunique()
    valid_subjects = sub_counts[sub_counts == len(valid_stimuli)].index.tolist()
    valid_subjects = sorted([str(x) for x in valid_subjects])
    if not valid_subjects:
        raise ValueError("筛选后无剩余被试")

    return valid_subjects, valid_stimuli


def build_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        s = str(row["subject"]).strip()
        stim = str(row["stimulus_type"]).strip()
        p = Path(str(row["file"]))
        key = (s, stim)
        if key not in out:
            out[key] = p
    return out


def run(avg_dir: Path, out_dir: Path, threshold_ratio: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_l = avg_dir / "avg_beta_index_surface_L.csv"
    idx_r = avg_dir / "avg_beta_index_surface_R.csv"
    if not idx_l.exists() or not idx_r.exists():
        raise FileNotFoundError(f"缺少索引文件: {idx_l} 或 {idx_r}")

    if not ATLAS_SURF_L.exists() or not ATLAS_SURF_R.exists():
        raise FileNotFoundError(f"缺少 Atlas: {ATLAS_SURF_L} 或 {ATLAS_SURF_R}")

    df_l = pd.read_csv(idx_l)
    df_r = pd.read_csv(idx_r)
    if df_l.empty or df_r.empty:
        raise ValueError("avg_beta_index 为空")

    valid_subjects, valid_stimuli = intelligent_filter_pairs(df_l, threshold_ratio=float(threshold_ratio))
    stimuli_sorted = sorted(valid_stimuli)

    df_l = df_l[df_l["subject"].astype(str).isin(valid_subjects) & df_l["stimulus_type"].astype(str).isin(stimuli_sorted)].copy()
    df_r = df_r[df_r["subject"].astype(str).isin(valid_subjects) & df_r["stimulus_type"].astype(str).isin(stimuli_sorted)].copy()
    lookup_l = build_lookup(df_l)
    lookup_r = build_lookup(df_r)

    atlas_l = load_atlas(ATLAS_SURF_L)
    atlas_r = load_atlas(ATLAS_SURF_R)
    roi_ids_l, roi_idx_l = prepare_roi_indices(atlas_l)
    roi_ids_r, roi_idx_r = prepare_roi_indices(atlas_r)
    if len(roi_ids_l) != 100 or len(roi_ids_r) != 100:
        raise ValueError(f"ROI 数量异常: L={len(roi_ids_l)} R={len(roi_ids_r)}")

    rois = [f"L_{rid}" for rid in roi_ids_l] + [f"R_{rid}" for rid in roi_ids_r]
    out_prefix = "roi_beta_matrix_200"
    written = []

    for stim in stimuli_sorted:
        subjects_this = sorted(
            {
                s
                for s in valid_subjects
                if (s, stim) in lookup_l
                and (s, stim) in lookup_r
                and lookup_l[(s, stim)].exists()
                and lookup_r[(s, stim)].exists()
            }
        )
        n_sub = len(subjects_this)
        if n_sub < 3:
            continue

        mats_l = [np.zeros((n_sub, int(idx.size)), dtype=np.float32) for idx in roi_idx_l]
        mats_r = [np.zeros((n_sub, int(idx.size)), dtype=np.float32) for idx in roi_idx_r]

        for si, sub in enumerate(subjects_this):
            p_l = lookup_l[(sub, stim)]
            p_r = lookup_r[(sub, stim)]

            vec_l = surface.load_surf_data(str(p_l)).astype(np.float32).reshape(-1)
            vec_r = surface.load_surf_data(str(p_r)).astype(np.float32).reshape(-1)
            if vec_l.shape[0] != atlas_l.shape[0]:
                raise ValueError(f"surface_L 顶点数不匹配: {sub} {stim} beta={vec_l.shape[0]} atlas={atlas_l.shape[0]}")
            if vec_r.shape[0] != atlas_r.shape[0]:
                raise ValueError(f"surface_R 顶点数不匹配: {sub} {stim} beta={vec_r.shape[0]} atlas={atlas_r.shape[0]}")

            for ri, idx in enumerate(roi_idx_l):
                mats_l[ri][si, :] = vec_l[idx]
            for ri, idx in enumerate(roi_idx_r):
                mats_r[ri][si, :] = vec_r[idx]

        roi_mats: Dict[str, np.ndarray] = {}
        for i, rid in enumerate(roi_ids_l):
            roi_mats[f"L_{rid}"] = mats_l[i]
        for i, rid in enumerate(roi_ids_r):
            roi_mats[f"R_{rid}"] = mats_r[i]

        stim_dir = out_dir / "by_stimulus" / stim
        stim_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stim_dir / f"{out_prefix}.npz", **roi_mats)
        pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
        pd.DataFrame({"subject": subjects_this}).to_csv(stim_dir / f"{out_prefix}_subjects.csv", index=False)
        pd.DataFrame({"stimulus_type": [stim]}).to_csv(stim_dir / f"{out_prefix}_stimuli.csv", index=False)
        written.append({"stimulus_type": stim, "n_subjects": n_sub})

    if not written:
        raise ValueError("没有任何 stimulus_type 满足条件（至少 3 名被试且 L/R 数据完整）")
    pd.DataFrame(written).sort_values("stimulus_type").to_csv(out_dir / "roi_beta_matrix_200_by_stimulus_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run(args.avg_dir, args.out_dir, float(args.threshold_ratio))


if __name__ == "__main__":
    main()
