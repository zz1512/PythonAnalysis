#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_roi_similarity_comprehensive.py
功能：全方位验证 Surface(L/R) 和 Volume 的 ROI 相似性。
升级：支持 ROI 定义为 ID 列表（合并多个小区为一个大网络），提高稳定性。
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# ================= 核心配置区 (支持列表) =================

# 1. 基础路径
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
CSV_DIR = LSS_ROOT / "similarity_matrices_final"
TASK = "EMO"

# 2. Atlas 路径
ATLAS_SURF_L = "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"
ATLAS_SURF_R = "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"
ATLAS_VOL = "/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T.nii.gz"

# 3. ROI 定义 (支持列表)
# Schaefer 200 (7 Networks) 的大致范围：
# Visual: L(1-14), R(101-114)
# SalVentAttn (Insula): L(38-49), R(138-149)
# Control (PFC): L(57-79), R(157-179)
# range(1, 15) 生成 1 到 14 (不含15)

ROIS_CONFIG = {
    "surface_L": {
        "atlas": ATLAS_SURF_L,
        "rois": {
            # 视觉网络 (大范围)
            "L_Visual_Net": list(range(1, 15)),
            # 突显网络 (包含脑岛)
            "L_SalVentAttn_Net": list(range(38, 50)),
            # 控制网络 (包含前额叶)
            "L_Control_PFC_Net": list(range(57, 80))
        }
    },
    "surface_R": {
        "atlas": ATLAS_SURF_R,
        "rois": {
            # 右脑对应网络 (ID + 100)
            "R_Visual_Net": list(range(101, 115)),
            "R_SalVentAttn_Net": list(range(138, 150)),
            "R_Control_PFC_Net": list(range(157, 180))
        }
    }
    # Volume 如果需要也可以配列表
    # "volume": { ... }
}


# ========================================================

def load_surf_data(path):
    try:
        g = nib.load(str(path))
        return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32)
    except:
        return None


def load_vol_data(path):
    try:
        img = nib.load(str(path))
        return img.get_fdata().astype(np.float32)
    except:
        return None


def load_atlas(path, space_type):
    try:
        if space_type == 'volume':
            img = nib.load(str(path))
            return img.get_fdata().astype(int)
        else:
            g = nib.load(str(path))
            return np.asarray(g.darrays[0].data).reshape(-1).astype(int)
    except Exception as e:
        print(f"❌ 加载图谱失败: {path}\nError: {e}")
        return None


def safe_corr(a, b):
    if len(a) < 5: return np.nan
    return np.corrcoef(a, b)[0, 1]


def resolve_beta_path(lss_root, task, subject, run, rel_file):
    base = Path(lss_root)
    candidates = [
        base / task / subject / rel_file,
        base / task / subject / f"run-{run}" / rel_file,
        base / task.lower() / subject / rel_file,
        base / task.lower() / subject / f"run-{run}" / rel_file
    ]
    for p in candidates:
        if p.exists(): return p
    return None


def process_space(space_name, cfg):
    print(f"\n{'=' * 20} 分析空间: {space_name} {'=' * 20}")

    csv_candidates = list(LSS_ROOT.rglob(f"lss_index_{space_name}_aligned.csv"))
    if not csv_candidates:
        csv_candidates = list(CSV_DIR.rglob(f"*{space_name}*aligned.csv"))

    if not csv_candidates:
        print(f"⚠️ 未找到 {space_name} 的对齐 CSV 文件，跳过。")
        return

    csv_path = csv_candidates[0]
    print(f"索引文件: {csv_path.name}")

    df = pd.read_csv(csv_path)
    df = df[df["task"] == TASK].copy()
    if df.empty:
        print("任务数据为空，跳过。")
        return

    atlas_path = cfg['atlas']
    print(f"加载图谱: {Path(atlas_path).name}")
    is_vol = 'volume' in space_name
    atlas_data = load_atlas(atlas_path, 'volume' if is_vol else 'surface')

    if atlas_data is None: return

    cnt = Counter(df["stimulus_content"].tolist())
    if not cnt:
        print("无刺激内容，跳过。")
        return
    target_stim = cnt.most_common(1)[0][0]
    print(f"基准刺激: {target_stim}")

    df_stim = df[df["stimulus_content"] == target_stim]
    sub_data_map = {}

    print(f"正在加载 Beta 图 ({len(df_stim)} files)...")
    for _, row in tqdm(df_stim.iterrows(), total=len(df_stim)):
        p = resolve_beta_path(LSS_ROOT, TASK, row["subject"], row["run"], row["file"])
        if p:
            if is_vol:
                data = load_vol_data(p)
            else:
                data = load_surf_data(p)

            if data is not None:
                if data.shape == atlas_data.shape:
                    sub_data_map[row["subject"]] = data

    valid_subs = list(sub_data_map.keys())
    print(f"有效被试: {len(valid_subs)}")
    if len(valid_subs) < 3: return

    results = []
    roi_dict = cfg['rois']

    for roi_name, roi_ids in roi_dict.items():
        # 【核心修改】支持 ID 列表
        # 如果 roi_ids 是单个数字，这就变成了 (atlas == id)
        # 如果 roi_ids 是列表，这就是 (atlas in [id1, id2...])
        if isinstance(roi_ids, list) or isinstance(roi_ids, range):
            mask = np.isin(atlas_data, roi_ids)
        else:
            mask = (atlas_data == roi_ids)

        n_voxels = np.sum(mask)

        if n_voxels < 5:
            print(f"  [Skip] {roi_name}: 顶点太少 ({n_voxels})")
            continue

        roi_matrix = []
        for sub in valid_subs:
            full_data = sub_data_map[sub]
            roi_matrix.append(full_data[mask])

        roi_matrix = np.stack(roi_matrix)

        sum_vec = roi_matrix.sum(axis=0)
        n = roi_matrix.shape[0]
        corrs = []

        for i in range(n):
            my_vec = roi_matrix[i]
            loo_mean = (sum_vec - my_vec) / (n - 1)
            r = safe_corr(my_vec, loo_mean)
            corrs.append(r)

        mean_r = np.nanmean(corrs)
        results.append({
            "ROI": roi_name,
            "Mean_r": mean_r,
            "N_Voxels": n_voxels
        })
        print(f"  ✅ {roi_name:<20} : r = {mean_r:.4f} (Vertices: {n_voxels})")

    return pd.DataFrame(results)


def main():
    all_res = []
    for space, cfg in ROIS_CONFIG.items():
        res = process_space(space, cfg)
        if res is not None:
            res['Space'] = space
            all_res.append(res)

    if all_res:
        final_df = pd.concat(all_res)
        print("\n\n" + "=" * 50)
        print("   全方位 ROI (Network Level) 相似性检查报告   ")
        print("=" * 50)
        cols = ['Space', 'ROI', 'Mean_r', 'N_Voxels']
        print(final_df[cols].to_string(index=False))

        final_df.to_csv("comprehensive_roi_check_results.csv", index=False)
    else:
        print("未生成任何结果。")


if __name__ == "__main__":
    main()