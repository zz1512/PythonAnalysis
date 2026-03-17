#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用“按刺激类型平均后的 beta 图”计算相似性矩阵：
1) 每种刺激类型的全脑被试×被试矩阵（保留空间特征，最后展平做相关）
2) 每种刺激类型、每个 ROI 的被试×被试矩阵
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import surface

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

DEFAULT_AVG_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/averaged_betas_by_stimulus")
DEFAULT_OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_by_stimulus")
DEFAULT_SUBJECT_INFO_PATH = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")

ATLAS_SURF_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURF_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")

ROI_CONFIG = {
    "Visual": {"L": [1, 2, 3, 4, 5, 6, 7], "R": [101, 102, 103, 104, 105, 106, 107]},
    "Salience_Insula": {"L": [38, 39, 40, 41, 42], "R": [138, 139, 140, 141, 142]},
    "Control_PFC": {"L": [57, 58, 59, 60, 61, 62], "R": [157, 158, 159, 160, 161, 162]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按刺激类型计算全脑/ROI 相似性矩阵")
    p.add_argument("--avg-dir", type=Path, default=DEFAULT_AVG_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--subject-info", type=Path, default=DEFAULT_SUBJECT_INFO_PATH)
    return p.parse_args()


def parse_age(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y = re.search(r"(\d+)\s*岁", s)
    m = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d = re.search(r"(\d+)\s*天", s)
    yy = int(y.group(1)) if y else 0
    mm = int(m.group(1)) if m else 0
    dd = int(d.group(1)) if d else 0
    return yy + mm / 12.0 + dd / 365.0


def load_age_map(path: Path) -> Dict[str, float]:
    if str(path).endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t")
    elif str(path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if COL_SUB_ID in df.columns and COL_AGE in df.columns:
        sid_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in df.columns and LEGACY_COL_AGE in df.columns:
        sid_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError("年龄表列名不匹配")

    out = {}
    for _, r in df.iterrows():
        sid = str(r[sid_col]).strip()
        sid = sid if sid.startswith("sub-") else f"sub-{sid}"
        out[sid] = parse_age(r[age_col])
    return out


def load_atlas(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(int)


def save_matrix(mat: np.ndarray, subs: List[str], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mat, index=subs, columns=subs).to_csv(out_file)


def compute_similarity(vectors: List[np.ndarray]) -> np.ndarray:
    data = np.stack(vectors, axis=0)
    return np.corrcoef(data)


def process_space(
    df: pd.DataFrame,
    space: str,
    age_map: Dict[str, float],
    out_dir: Path,
    atlas_labels: np.ndarray | None,
    hemi: str | None,
) -> None:
    for stim, g in df.groupby("stimulus_type"):
        subjects = sorted(set(g["subject"].astype(str)))
        subjects = [s for s in subjects if np.isfinite(age_map.get(s, np.nan))]
        subjects.sort(key=lambda s: (age_map.get(s, np.nan), s))
        if len(subjects) < 3:
            continue

        lookup = {str(r["subject"]): Path(r["file"]) for _, r in g.iterrows()}

        whole_vectors = []
        final_subs = []
        for s in subjects:
            p = lookup.get(s)
            if not p or not p.exists():
                continue
            try:
                whole_vectors.append(surface.load_surf_data(str(p)).astype(np.float32).reshape(-1))
                final_subs.append(s)
            except Exception:
                continue

        if len(final_subs) < 3:
            continue

        mat = compute_similarity(whole_vectors)
        save_matrix(mat, final_subs, out_dir / f"similarity_{space}_{stim}_AgeSorted.csv")
        pd.DataFrame({"subject": final_subs, "age": [age_map[s] for s in final_subs]}).to_csv(
            out_dir / f"subject_order_{space}_{stim}.csv", index=False
        )

        if atlas_labels is None or hemi is None:
            continue

        for roi_name, side_ids in ROI_CONFIG.items():
            target_ids = side_ids.get(hemi, [])
            if not target_ids:
                continue
            mask = np.isin(atlas_labels, target_ids)
            if int(mask.sum()) < 10:
                continue

            roi_vectors = []
            roi_subs = []
            for s in final_subs:
                p = lookup.get(s)
                try:
                    vec = surface.load_surf_data(str(p)).astype(np.float32).reshape(-1)
                    roi_vectors.append(vec[mask])
                    roi_subs.append(s)
                except Exception:
                    continue

            if len(roi_subs) < 3:
                continue

            roi_mat = compute_similarity(roi_vectors)
            save_matrix(roi_mat, roi_subs, out_dir / f"similarity_{space}_{roi_name}_{stim}_AgeSorted.csv")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    age_map = load_age_map(args.subject_info)

    atlas_l = load_atlas(ATLAS_SURF_L) if ATLAS_SURF_L.exists() else None
    atlas_r = load_atlas(ATLAS_SURF_R) if ATLAS_SURF_R.exists() else None

    index_files = sorted(args.avg_dir.glob("avg_beta_index_*.csv"))
    if not index_files:
        raise FileNotFoundError(f"未找到: {args.avg_dir}/avg_beta_index_*.csv")

    for idx_file in index_files:
        df = pd.read_csv(idx_file)
        if df.empty:
            continue
        scenario = str(df.iloc[0]["scenario"])

        if "surface_L" in scenario:
            process_space(df, "surface_L", age_map, args.out_dir, atlas_l, "L")
        elif "surface_R" in scenario:
            process_space(df, "surface_R", age_map, args.out_dir, atlas_r, "R")
        else:
            print(f"[提示] 暂不处理 volume ROI: {scenario}")

    print(f"完成，输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
