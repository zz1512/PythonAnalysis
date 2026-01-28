#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_isc_aligned_by_age.py
功能：
1. 读取被试年龄表 (支持 .xlsx/.tsv/.csv)，优先使用数值年龄列。
2. 按照年龄（从小到大）对被试进行排序。
3. 基于排序后的被试列表，计算 ISC 矩阵。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import surface, image
from tqdm import tqdm
import gc
import os
import re

# ================= 配置区 =================
# 1. 结果根目录
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")

# 2. 输出目录
OUTPUT_DIR = LSS_ROOT / "similarity_matrices_final_age_sorted"

# 3. 被试信息表路径
# 环境变量 SUBJECT_AGE_TABLE 可覆盖该默认路径
SUBJECT_INFO_PATH = os.environ.get(
    "SUBJECT_AGE_TABLE",
    "/public/home/dingrui/BIDS_DATA/emo_20250623/横断队列被试信息表.tsv",
)

# 4. 表格中的列名配置
COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

# 任务定义
TASK_A = 'EMO'
TASK_B = 'SOC'

# Volume Mask
MNI_MASK_PATH = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")


# ==========================================

def parse_chinese_age(age_str):
    """
    解析年龄字段：
    - 若为数值或可直接转 float，则直接返回 float(age)
    - 否则尝试解析 '11岁11个月10天' 格式，返回四舍五入后的整数年龄
    """
    if pd.isna(age_str) or str(age_str).strip() == "":
        return 999.0  # 缺失值放到最后

    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)

    age_str = str(age_str).strip()
    try:
        return float(age_str)
    except Exception:
        pass

    # 使用正则表达式提取数字
    y_match = re.search(r'(\d+)\s*岁', age_str)
    m_match = re.search(r'(\d+)\s*个月', age_str)
    if not m_match:
        m_match = re.search(r'(\d+)\s*月', age_str)
    d_match = re.search(r'(\d+)\s*天', age_str)

    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0

    # 计算精确年龄
    exact_age = years + (months / 12.0) + (days / 365.0)

    return round(exact_age)


def load_subject_ages_map():
    """
    智能读取表格 (支持 xlsx 和 tsv)，建立 {sub-ID: age} 的映射字典
    """
    print(f"正在读取被试信息表: {SUBJECT_INFO_PATH}")
    path_str = str(SUBJECT_INFO_PATH)

    try:
        # === 核心修复：根据后缀选择读取方式 ===
        if path_str.endswith('.tsv') or path_str.endswith('.txt'):
            print("识别为 TSV 文本格式...")
            info_df = pd.read_csv(path_str, sep='\t')
        elif path_str.endswith('.xlsx') or path_str.endswith('.xls'):
            print("识别为 Excel 格式...")
            info_df = pd.read_excel(path_str)
        elif path_str.endswith('.csv'):
            print("识别为 CSV 格式...")
            info_df = pd.read_csv(path_str)
        else:
            raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")

        if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
            sub_col, age_col = COL_SUB_ID, COL_AGE
        elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
            sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
        else:
            print(f"当前表格列名: {info_df.columns.tolist()}")
            raise ValueError(
                "年龄表缺少必要列。需要以下任意一组列名：\n"
                f"- {COL_SUB_ID}, {COL_AGE}\n"
                f"- {LEGACY_COL_SUB_ID}, {LEGACY_COL_AGE}\n"
            )

        age_map = {}
        for _, row in info_df.iterrows():
            raw_id = str(row[sub_col]).strip()
            age_str = row[age_col]

            # 标准化 ID
            if not raw_id.startswith('sub-'):
                sub_id = f"sub-{raw_id}"
            else:
                sub_id = raw_id

            age_val = parse_chinese_age(age_str)
            age_map[sub_id] = age_val

        print(f"成功加载 {len(age_map)} 名被试的年龄信息。")
        return age_map

    except Exception as e:
        print(f"❌ 读取被试信息表失败: {e}")
        return {}


def get_data_flattener(file_type, mask_img=None):
    if file_type == 'surface':
        return lambda fpath: surface.load_surf_data(str(fpath)).astype(np.float32)
    elif file_type == 'volume':
        if mask_img is None: raise ValueError("Volume 模式需要 Mask")
        mask_data = mask_img.get_fdata() > 0

        def flatten_volume(fpath):
            img = image.load_img(str(fpath))
            return img.get_fdata(dtype=np.float32)[mask_data]

        return flatten_volume


def compute_matrix_with_age_sorting(df_full, context_name, flattener, age_map):
    print(f"\n>>> 正在处理场景: {context_name}")

    # 1. 基础清洗
    df = df_full.dropna(subset=['stimulus_content']).copy()

    # 2. 被试交集筛选
    subjects_emo = set(df[df['task'] == TASK_A]['subject'])
    subjects_soc = set(df[df['task'] == TASK_B]['subject'])
    valid_subjects_unsorted = list(subjects_emo & subjects_soc)

    if not valid_subjects_unsorted:
        print("  ❌ 错误：没有被试同时拥有两个任务的数据。")
        return

    # =======================================================
    # 根据年龄对被试进行排序
    # =======================================================
    def sort_key(sub_id):
        age = age_map.get(sub_id, np.nan)
        if not np.isfinite(age):
            return (999.0, sub_id)
        return (float(age), sub_id)

    valid_subjects_sorted = sorted(valid_subjects_unsorted, key=sort_key)

    print(f"  - 有效被试数: {len(valid_subjects_sorted)}")
    if len(valid_subjects_sorted) > 0:
        first = valid_subjects_sorted[0]
        last = valid_subjects_sorted[-1]
        print(f"  - 最小被试: {first} ({age_map.get(first, 'N/A')}岁)")
        print(f"  - 最大被试: {last} ({age_map.get(last, 'N/A')}岁)")

    # 只保留这些有效被试
    df = df[df['subject'].isin(valid_subjects_sorted)].copy()

    # 3. 筛选公共刺激
    key_counts = df['stimulus_content'].value_counts()
    threshold = int(len(valid_subjects_sorted) * 0.9)
    common_keys = key_counts[key_counts >= threshold].index.tolist()
    common_keys.sort()

    if not common_keys:
        print(f"  ⚠️ 没有公共刺激 (阈值: {threshold})")
        return

    print(f"  - 用于计算的特征数 (Stimuli): {len(common_keys)}")

    # 4. 建立索引
    lookup_dict = {}
    for _, row in df.iterrows():
        if str(row['file']).startswith('/'):
            fpath = Path(row['file'])
        else:
            task_folder = row['task'].lower() if 'task' in row else (
                'emo' if 'EMO' in row['stimulus_content'] else 'soc')
            fpath = LSS_ROOT / task_folder / row['subject'] / row['file']
            if not fpath.exists():
                fpath = LSS_ROOT / task_folder / row['subject'] / f"run-{row['run']}" / row['file']
        lookup_dict[(row['subject'], row['stimulus_content'])] = fpath

    # 5. 加载数据
    subject_vectors = []
    final_subjects = []
    final_ages = []

    for sub in tqdm(valid_subjects_sorted, desc="  - Loading Data (Age Sorted)"):
        sub_features = []
        is_complete = True

        for content_key in common_keys:
            fpath = lookup_dict.get((sub, content_key))
            if fpath and fpath.exists():
                try:
                    data = flattener(fpath)
                    sub_features.append(data)
                except:
                    is_complete = False;
                    break
            else:
                is_complete = False;
                break

        if is_complete:
            flat_vec = np.concatenate(sub_features)
            subject_vectors.append(flat_vec)
            final_subjects.append(sub)
            final_ages.append(age_map.get(sub, 'N/A'))

    # 6. 计算与保存
    if final_subjects:
        print(f"  - 最终计算矩阵规模: {len(final_subjects)} Subjects")
        data_matrix = np.stack(subject_vectors)
        del subject_vectors
        gc.collect()

        print("  - 正在计算相关系数...")
        sim_matrix = np.corrcoef(data_matrix)

        out_name = f"similarity_{context_name}_AgeSorted.csv"
        out_path = OUTPUT_DIR / out_name
        pd.DataFrame(sim_matrix, index=final_subjects, columns=final_subjects).to_csv(out_path)
        print(f"  ✅ 矩阵已保存: {out_path}")

        order_info = pd.DataFrame({'subject': final_subjects, 'age': final_ages})
        order_info.to_csv(OUTPUT_DIR / f"subject_order_{context_name}.csv", index=False)

        del data_matrix, sim_matrix
        gc.collect()
    else:
        print("  ❌ 加载后无有效被试。")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    age_map = load_subject_ages_map()
    if not age_map:
        print("无法加载年龄信息，程序终止。")
        return

    aligned_files = list(LSS_ROOT.glob("*_aligned.csv"))
    if not aligned_files:
        print(f"未找到 *_aligned.csv 文件！")
        return

    vol_mask_img = None
    if any("volume" in f.name.lower() for f in aligned_files):
        if MNI_MASK_PATH.exists():
            print("加载 Volume Mask...")
            vol_mask_img = image.load_img(str(MNI_MASK_PATH))

    for idx_file in aligned_files:
        scenario_name = idx_file.stem.replace("lss_index_", "").replace("_aligned", "")

        is_volume = "volume" in scenario_name.lower()
        file_type = 'volume' if is_volume else 'surface'

        try:
            flattener = get_data_flattener(file_type, vol_mask_img if is_volume else None)
        except ValueError:
            continue

        df = pd.read_csv(idx_file)
        compute_matrix_with_age_sorting(df, scenario_name, flattener, age_map)

        gc.collect()


if __name__ == "__main__":
    main()
