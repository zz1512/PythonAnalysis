#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_roi_repr_matrix_232.py

Step 1:
按刺激类型(stimulus_type)分别构建 ROI 表征结构矩阵：
- 输入：LSS index + beta 文件
- 输出：每个 stimulus_type 一个 npz，键为 ROI 名，值为 [n_sub, n_stim, n_stim]

关键约束：
1) 被试必须同时具备 EMO 与 SOC 任务；
2) ROI = surface(L/R) + volume，共 232；
3) 每个 stimulus_type 内，所有被试使用完全一致的 stimulus_content 排序。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy.stats import rankdata

DEFAULT_LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
DEFAULT_OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")

ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_100Parcels_7Networks_order_lh.annot")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_100Parcels_7Networks_order_rh.annot")
ATLAS_VOL = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_32Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建每个刺激类型的 ROI 表征结构矩阵（232 ROI）")
    p.add_argument("--lss-root", type=Path, default=DEFAULT_LSS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stim-type-col", type=str, default="raw_emotion")
    p.add_argument("--stim-id-col", type=str, default="stimulus_content")
    p.add_argument("--threshold-ratio", type=float, default=0.80)
    return p.parse_args()


def load_lss_index(lss_root: Path, stim_type_col: str, stim_id_col: str) -> pd.DataFrame:
    files = sorted(lss_root.glob("lss_index_*_aligned.csv"))
    if not files:
        raise FileNotFoundError(f"未找到索引文件: {lss_root}/lss_index_*_aligned.csv")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        required = {"subject", "task", "file", stim_type_col, stim_id_col}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df["scenario"] = f.stem.replace("lss_index_", "").replace("_aligned", "")
        dfs.append(df)

    if not dfs:
        raise ValueError("没有可用的 aligned 索引文件（缺必要列）")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out["subject"] = out["subject"].astype(str).str.strip()
    out["task"] = out["task"].astype(str).str.upper().str.strip()
    out["stimulus_type"] = out[stim_type_col].astype(str).str.strip()
    out["stimulus_id"] = out[stim_id_col].astype(str).str.strip()
    out = out[(out["subject"].str.len() > 0) & (out["stimulus_type"].str.len() > 0) & (out["stimulus_id"].str.len() > 0)].copy()

    task_sets = out.groupby("subject")["task"].agg(lambda x: set(x.dropna().tolist()))
    eligible = task_sets[task_sets.apply(lambda s: {"EMO", "SOC"}.issubset(s))].index
    out = out[out["subject"].isin(eligible)].copy()
    if out.empty:
        raise ValueError("无同时具备 EMO+SOC 的被试")
    return out


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", ""))
    if f.startswith("/"):
        return Path(f)

    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))

    p1 = lss_root / task / sub / f
    if p1.exists():
        return p1

    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def load_surface_atlas(path: Path) -> Tuple[List[int], List[np.ndarray]]:
    labels, _, _ = nib.freesurfer.read_annot(str(path))
    labels = labels.reshape(-1)
    roi_ids = sorted(np.unique(labels[labels > 0]).astype(int).tolist())
    roi_indices = [np.where(labels == rid)[0] for rid in roi_ids]
    return roi_ids, roi_indices


def load_volume_atlas(path: Path) -> Tuple[object, List[int], List[np.ndarray]]:
    img = image.load_img(str(path))
    data = img.get_fdata().reshape(-1)
    roi_ids = sorted(np.unique(data[data > 0]).astype(int).tolist())
    roi_indices = [np.where(data == rid)[0] for rid in roi_ids]
    return img, roi_ids, roi_indices


def spearman_rsm(X: np.ndarray) -> np.ndarray:
    # X: [n_stim, n_feat]
    ranks = np.apply_along_axis(rankdata, 1, X)
    mean = ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True)
    std[std == 0] = np.nan
    Z = (ranks - mean) / std
    p = ranks.shape[1]
    C = (Z @ Z.T) / float(max(1, p - 1))
    return np.clip(C, -1.0, 1.0).astype(np.float32)


def pick_subjects_stimuli(df: pd.DataFrame, threshold_ratio: float) -> Tuple[List[str], Dict[str, List[str]]]:
    subjects = sorted(df["subject"].unique().tolist())
    n_sub = len(subjects)
    by_type: Dict[str, List[str]] = {}

    for stim_type, g in df.groupby("stimulus_type"):
        pairs = g[["subject", "stimulus_id"]].drop_duplicates()
        counts = pairs["stimulus_id"].value_counts()
        valid_ids = sorted(counts[counts >= n_sub * float(threshold_ratio)].index.tolist())
        if not valid_ids:
            continue
        sub_counts = pairs[pairs["stimulus_id"].isin(valid_ids)].groupby("subject")["stimulus_id"].nunique()
        full_sub = sorted(sub_counts[sub_counts == len(valid_ids)].index.tolist())
        if len(full_sub) >= 3:
            by_type[str(stim_type)] = valid_ids

    if not by_type:
        raise ValueError("没有任何 stimulus_type 通过覆盖率筛选")
    return subjects, by_type


def build_lookup(df: pd.DataFrame, lss_root: Path) -> Dict[Tuple[str, str, str, str], Path]:
    out: Dict[Tuple[str, str, str, str], Path] = {}
    for _, row in df.iterrows():
        key = (str(row["scenario"]), str(row["subject"]), str(row["stimulus_type"]), str(row["stimulus_id"]))
        if key not in out:
            out[key] = resolve_beta_path(lss_root, row)
    return out


def run(lss_root: Path, out_dir: Path, stim_type_col: str, stim_id_col: str, threshold_ratio: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_lss_index(lss_root, stim_type_col=stim_type_col, stim_id_col=stim_id_col)
    all_subjects, stim2ids = pick_subjects_stimuli(df, threshold_ratio=float(threshold_ratio))
    lookup = build_lookup(df, lss_root)

    roi_ids_l, roi_idx_l = load_surface_atlas(ATLAS_SURF_L)
    roi_ids_r, roi_idx_r = load_surface_atlas(ATLAS_SURF_R)
    vol_img, roi_ids_v, roi_idx_v = load_volume_atlas(ATLAS_VOL)
    rois = [f"L_{x}" for x in roi_ids_l] + [f"R_{x}" for x in roi_ids_r] + [f"V_{x}" for x in roi_ids_v]

    summary_rows = []
    by_stim_dir = out_dir / "by_stimulus"

    for stim_type in sorted(stim2ids.keys()):
        stim_ids = sorted(stim2ids[stim_type])
        pairs = df[df["stimulus_type"] == stim_type][["subject", "stimulus_id"]].drop_duplicates()
        sub_counts = pairs[pairs["stimulus_id"].isin(stim_ids)].groupby("subject")["stimulus_id"].nunique()
        subjects = sorted(sub_counts[sub_counts == len(stim_ids)].index.tolist())
        if len(subjects) < 3:
            continue

        n_sub, n_stim = len(subjects), len(stim_ids)
        out_roi: Dict[str, np.ndarray] = {r: np.zeros((n_sub, n_stim, n_stim), dtype=np.float32) for r in rois}

        for si, sub in enumerate(subjects):
            feats_l = [np.zeros((n_stim, int(idx.size)), dtype=np.float32) for idx in roi_idx_l]
            feats_r = [np.zeros((n_stim, int(idx.size)), dtype=np.float32) for idx in roi_idx_r]
            feats_v = [np.zeros((n_stim, int(idx.size)), dtype=np.float32) for idx in roi_idx_v]

            for ki, stim_id in enumerate(stim_ids):
                p_l = lookup.get(("surface_L", sub, stim_type, stim_id))
                p_r = lookup.get(("surface_R", sub, stim_type, stim_id))
                p_v = lookup.get(("volume", sub, stim_type, stim_id))
                if p_l is None or p_r is None or p_v is None:
                    raise FileNotFoundError(f"缺少 beta：{sub} {stim_type} {stim_id}")
                if (not p_l.exists()) or (not p_r.exists()) or (not p_v.exists()):
                    raise FileNotFoundError(f"beta 文件不存在：{sub} {stim_type} {stim_id}")

                vec_l = nib.load(str(p_l)).darrays[0].data.astype(np.float32).reshape(-1)
                vec_r = nib.load(str(p_r)).darrays[0].data.astype(np.float32).reshape(-1)
                img_v = image.load_img(str(p_v))
                if img_v.shape != vol_img.shape or not np.allclose(img_v.affine, vol_img.affine):
                    img_v = image.resample_to_img(img_v, vol_img, interpolation="nearest")
                vec_v = img_v.get_fdata().reshape(-1).astype(np.float32)

                for ri, idx in enumerate(roi_idx_l):
                    feats_l[ri][ki, :] = vec_l[idx]
                for ri, idx in enumerate(roi_idx_r):
                    feats_r[ri][ki, :] = vec_r[idx]
                for ri, idx in enumerate(roi_idx_v):
                    feats_v[ri][ki, :] = vec_v[idx]

            for ri, rid in enumerate(roi_ids_l):
                out_roi[f"L_{rid}"][si] = spearman_rsm(feats_l[ri])
            for ri, rid in enumerate(roi_ids_r):
                out_roi[f"R_{rid}"][si] = spearman_rsm(feats_r[ri])
            for ri, rid in enumerate(roi_ids_v):
                out_roi[f"V_{rid}"][si] = spearman_rsm(feats_v[ri])

        stim_dir = by_stim_dir / stim_type
        stim_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stim_dir / "roi_repr_matrix_232.npz", **out_roi)
        pd.DataFrame({"roi": rois}).to_csv(stim_dir / "roi_repr_matrix_232_rois.csv", index=False)
        pd.DataFrame({"subject": subjects}).to_csv(stim_dir / "roi_repr_matrix_232_subjects.csv", index=False)
        pd.DataFrame({"stimulus_type": [stim_type], "n_stimuli": [n_stim]}).to_csv(
            stim_dir / "roi_repr_matrix_232_meta.csv", index=False
        )
        pd.DataFrame({"stimulus_order": stim_ids}).to_csv(stim_dir / "stimulus_order.csv", index=False)
        pd.DataFrame(
            [{"subject": s, "order_id": "|".join(stim_ids)} for s in subjects]
        ).to_csv(stim_dir / "stimulus_order_by_subject.csv", index=False)

        summary_rows.append({"stimulus_type": stim_type, "n_subjects": n_sub, "n_stimuli": n_stim, "n_rois": len(rois)})

    if not summary_rows:
        raise ValueError("没有写出任何刺激类型结果")

    pd.DataFrame(summary_rows).sort_values("stimulus_type").to_csv(out_dir / "roi_repr_matrix_232_summary.csv", index=False)
    pd.DataFrame({"subject": all_subjects}).to_csv(out_dir / "eligible_subjects_emo_soc.csv", index=False)


def main() -> None:
    args = parse_args()
    run(args.lss_root, args.out_dir, args.stim_type_col, args.stim_id_col, float(args.threshold_ratio))


if __name__ == "__main__":
    main()
