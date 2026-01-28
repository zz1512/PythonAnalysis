#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_isc_roi_aligned_age.py
功能：
1. 读取被试年龄信息并排序 (从小到大)。
2. 基于 Schaefer 图谱，计算特定脑区 (ROI) 的被试间相似性矩阵。
3. 输出结果按年龄排序，方便观察发育趋势。
"""

import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import gc
import re

# ================= 核心配置区 =================
# 1. 基础路径
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
OUTPUT_DIR = LSS_ROOT / "similarity_matrices_roi_age_sorted"

# 2. 被试信息表路径 (支持 .xlsx 或 .tsv)
# 请确认这是你最新的被试信息表路径
SUBJECT_INFO_PATH = "/public/home/dingrui/BIDS_DATA/emo_20250623/横断队列被试信息表.tsv"
COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

# 3. Atlas 路径 (Schaefer 200, 已拆分为左右脑)
ATLAS_SURF_L = "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"
ATLAS_SURF_R = "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"

# 4. ROI 定义 (Schaefer 7Networks)
ROI_CONFIG = {
    "Visual": {
        "L": [1, 2, 3, 4, 5, 6, 7],
        "R": [101, 102, 103, 104, 105, 106, 107]
    },
    "Salience_Insula": {
        "L": [38, 39, 40, 41, 42],
        "R": [138, 139, 140, 141, 142]
    },
    "Control_PFC": {
        "L": [57, 58, 59, 60, 61, 62],
        "R": [157, 158, 159, 160, 161, 162]
    }
}

# 任务定义
TASK_A = 'EMO'
TASK_B = 'SOC'


# ==========================================

def parse_chinese_age(age_str):
    """解析 '11岁11个月10天'"""
    if pd.isna(age_str) or str(age_str).strip() == "":
        return 999.0
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    age_str = str(age_str).strip()
    try:
        return float(age_str)
    except Exception:
        pass
    y_match = re.search(r'(\d+)\s*岁', age_str)
    m_match = re.search(r'(\d+)\s*个月', age_str) or re.search(r'(\d+)\s*月', age_str)
    d_match = re.search(r'(\d+)\s*天', age_str)

    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0
    return years + (months / 12.0) + (days / 365.0)


def load_subject_ages_map():
    """读取表格建立 ID->Age 映射"""
    print(f"正在读取被试信息: {SUBJECT_INFO_PATH}")
    path_str = str(SUBJECT_INFO_PATH)
    try:
        if path_str.endswith('.tsv') or path_str.endswith('.txt'):
            info_df = pd.read_csv(path_str, sep='\t')
        else:
            info_df = pd.read_excel(path_str)

        if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
            sub_col, age_col = COL_SUB_ID, COL_AGE
        elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
            sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
        else:
            print(f"❌ 年龄表缺少必要列。需要 {COL_SUB_ID}/{COL_AGE} 或 {LEGACY_COL_SUB_ID}/{LEGACY_COL_AGE}")
            print(f"   实际列名: {list(info_df.columns)}")
            return {}

        age_map = {}
        for _, row in info_df.iterrows():
            raw_id = str(row[sub_col]).strip()
            # 统一格式 sub-xxx
            sub_id = f"sub-{raw_id}" if not raw_id.startswith('sub-') else raw_id
            age_map[sub_id] = parse_chinese_age(row[age_col])

        print(f"已加载 {len(age_map)} 名被试年龄。")
        return age_map
    except Exception as e:
        print(f"❌ 读取年龄表失败: {e}")
        return {}


def load_atlas(path):
    try:
        g = nib.load(str(path))
        return np.asarray(g.darrays[0].data).reshape(-1).astype(int)
    except:
        return None


def load_gifti_data(path):
    try:
        g = nib.load(str(path))
        return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32)
    except:
        return None


def compute_roi_matrix_age_sorted(df_full, space_name, atlas_path, roi_dict, age_map):
    print(f"\n{'=' * 20} 处理空间: {space_name} {'=' * 20}")

    # 1. 加载 Atlas
    atlas_labels = load_atlas(atlas_path)
    if atlas_labels is None:
        print("❌ Atlas 加载失败，跳过。")
        return

    # 2. 筛选公共刺激 & 被试
    df = df_full.dropna(subset=['stimulus_content']).copy()
    subjects_emo = set(df[df['task'] == TASK_A]['subject'])
    subjects_soc = set(df[df['task'] == TASK_B]['subject'])
    valid_subjects_unsorted = list(subjects_emo & subjects_soc)

    if not valid_subjects_unsorted:
        print("无有效被试。")
        return

    # ================= 核心：按年龄排序 =================
    def sort_key(sub_id):
        return (age_map.get(sub_id, 999), sub_id)

    valid_subjects_sorted = sorted(valid_subjects_unsorted, key=sort_key)
    print(f"有效被试: {len(valid_subjects_sorted)} (已按年龄排序)")
    if len(valid_subjects_sorted) > 0:
        print(f"  Min Age: {age_map.get(valid_subjects_sorted[0], 'N/A')}")
        print(f"  Max Age: {age_map.get(valid_subjects_sorted[-1], 'N/A')}")
    # ==================================================

    df = df[df['subject'].isin(valid_subjects_sorted)].copy()

    # 筛选 Stimuli
    key_counts = df['stimulus_content'].value_counts()
    threshold = int(len(valid_subjects_sorted) * 0.9)
    common_keys = key_counts[key_counts >= threshold].index.tolist()
    common_keys.sort()

    if not common_keys:
        print("无公共刺激。")
        return

    # 3. 建立文件索引
    lookup_dict = {}
    for _, row in df.iterrows():
        task_folder = row['task'].lower() if 'task' in row else ('emo' if 'EMO' in row['stimulus_content'] else 'soc')
        fpath = LSS_ROOT / task_folder / row['subject'] / row['file']
        if not fpath.exists():
            fpath = LSS_ROOT / task_folder / row['subject'] / f"run-{row['run']}" / row['file']
        lookup_dict[(row['subject'], row['stimulus_content'])] = fpath

    # 4. 循环 ROI 计算
    hemi = "L" if "surface_L" in space_name else "R"

    for roi_name, id_map in roi_dict.items():
        target_ids = id_map.get(hemi, [])
        if not target_ids: continue

        print(f"\n--- 计算 ROI: {roi_name} ({hemi}) ---")

        mask = np.isin(atlas_labels, target_ids)
        n_vertices = np.sum(mask)
        print(f"包含顶点数: {n_vertices}")

        if n_vertices < 10:
            print("顶点太少，跳过。")
            continue

        # 加载数据 (按排序后的列表)
        subject_vectors = []
        final_subs = []
        final_ages = []  # 用于保存顺序文件

        for sub in tqdm(valid_subjects_sorted, desc=f"Loading {roi_name}"):
            sub_feats = []
            is_ok = True
            for key in common_keys:
                p = lookup_dict.get((sub, key))
                if p and p.exists():
                    data = load_gifti_data(p)
                    if data is not None:
                        sub_feats.append(data[mask])
                    else:
                        is_ok = False;
                        break
                else:
                    is_ok = False;
                    break

            if is_ok:
                subject_vectors.append(np.concatenate(sub_feats))
                final_subs.append(sub)
                final_ages.append(age_map.get(sub, 'N/A'))

        if final_subs:
            # 计算矩阵
            data_mat = np.stack(subject_vectors)
            sim_mat = np.corrcoef(data_mat)

            # 保存矩阵
            fname = f"similarity_{space_name}_{roi_name}_AgeSorted.csv"
            fpath = OUTPUT_DIR / fname
            pd.DataFrame(sim_mat, index=final_subs, columns=final_subs).to_csv(fpath)
            print(f"✅ 矩阵已保存: {fname}")

            # 保存排序参考表 (每个 ROI 可能有效被试略有不同，建议都存)
            pd.DataFrame({'subject': final_subs, 'age': final_ages}).to_csv(
                OUTPUT_DIR / f"order_{space_name}_{roi_name}.csv", index=False
            )

            # 打印统计 (去除对角线)
            vals = sim_mat[np.triu_indices_from(sim_mat, k=1)]
            if len(vals) > 0:
                print(f"  📊 统计: Mean={np.mean(vals):.4f} | Max={np.max(vals):.4f} | Min={np.min(vals):.4f}")

            del data_mat, sim_mat
            gc.collect()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载年龄
    age_map = load_subject_ages_map()
    if not age_map: return

    # 2. 查找文件
    files = list(LSS_ROOT.glob("*_aligned.csv"))

    for f in files:
        if "surface_L" in f.name:
            df = pd.read_csv(f)
            compute_roi_matrix_age_sorted(df, "surface_L", ATLAS_SURF_L, ROI_CONFIG, age_map)
        elif "surface_R" in f.name:
            df = pd.read_csv(f)
            compute_roi_matrix_age_sorted(df, "surface_R", ATLAS_SURF_R, ROI_CONFIG, age_map)


if __name__ == "__main__":
    main()
