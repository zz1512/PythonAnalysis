#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_isc_aligned.py
功能：基于 stimulus_content (内容对齐列) 计算被试间相似性矩阵。
适用：已解决了试次随机化问题的 _aligned.csv 文件。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import surface, image
from tqdm import tqdm
import gc
import os

# ================= 配置区 =================
# 结果根目录
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")

# 输出目录
OUTPUT_DIR = LSS_ROOT / "similarity_matrices_final"

# 任务定义 (用于筛选被试交集)
TASK_A = 'EMO'
TASK_B = 'SOC'

# Volume Mask 路径 (如果算Volume需要)
MNI_MASK_PATH = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")


# ==========================================

def get_data_flattener(file_type, mask_img=None):
    """工厂函数：返回读取并展平数据的函数"""
    if file_type == 'surface':
        return lambda fpath: surface.load_surf_data(str(fpath)).astype(np.float32)
    elif file_type == 'volume':
        if mask_img is None: raise ValueError("Volume 模式需要 Mask")
        mask_data = mask_img.get_fdata() > 0

        def flatten_volume(fpath):
            img = image.load_img(str(fpath))
            return img.get_fdata(dtype=np.float32)[mask_data]

        return flatten_volume


def compute_matrix_with_content_alignment(df_full, context_name, flattener):
    """
    核心逻辑：使用 stimulus_content 列进行精确对齐
    """
    print(f"\n>>> 正在处理场景: {context_name}")

    # 1. 基础清洗
    # 去除 stimulus_content 为空的行 (可能是解析失败或无用Trial)
    df = df_full.dropna(subset=['stimulus_content']).copy()

    # 2. 被试交集筛选 (Subject Intersection)
    # 找出既有 EMO 又有 SOC 数据的被试
    subjects_emo = set(df[df['task'] == TASK_A]['subject'])
    subjects_soc = set(df[df['task'] == TASK_B]['subject'])

    # 取交集
    valid_subjects_set = sorted(list(subjects_emo & subjects_soc))

    print(f"  - EMO 被试数: {len(subjects_emo)}")
    print(f"  - SOC 被试数: {len(subjects_soc)}")
    print(f"  - [筛选后] 双任务均有的被试数: {len(valid_subjects_set)}")

    if len(valid_subjects_set) == 0:
        print("  ❌ 错误：没有被试同时拥有两个任务的数据。")
        return

    # 只保留这些有效被试
    df = df[df['subject'].isin(valid_subjects_set)].copy()

    # 3. 筛选公共刺激 (Stimulus Intersection)
    # 直接统计 stimulus_content 的出现频率
    # 这里的 stimulus_content 已经是唯一的了 (如 EMO_Passive_Negative_001)
    key_counts = df['stimulus_content'].value_counts()

    # 阈值：90% 的有效被试都有该图片
    threshold = int(len(valid_subjects_set) * 0.9)
    common_keys = key_counts[key_counts >= threshold].index.tolist()

    # 排序：保证矩阵的列顺序一致
    common_keys.sort()

    if not common_keys:
        print(f"  ⚠️ 没有公共刺激 (阈值: {threshold})")
        print("  Top 5 刺激出现次数:", key_counts.head(5).to_dict())
        return

    print(f"  - 用于计算的特征数 (Stimuli): {len(common_keys)}")
    print(f"  - 特征示例: {common_keys[:3]} ...")

    # 4. 建立快速查找索引
    # Key: (subject, stimulus_content) -> Value: file_path
    # 注意：file 列如果是相对路径，需要拼全
    lookup_dict = {}
    for _, row in df.iterrows():
        # 假设 csv 里的 file 列是相对路径 (beta_xxx.gii)
        # 或者是绝对路径，这里做个兼容
        if str(row['file']).startswith('/'):
            fpath = Path(row['file'])
        else:
            # 根据目录结构拼接: root / task_folder / subject / run / file
            # 注意：这里需要根据你的实际目录结构微调
            # 你的 lss_main 生成的目录结构是: OUTPUT_ROOT / task_type / sub / file
            task_folder = row['task'].lower() if 'task' in row else (
                'emo' if 'EMO' in row['stimulus_content'] else 'soc')
            # 你的lss_main生成的文件直接在 task/sub 下，还是 task/sub/run 下？
            # 之前的脚本是 task/sub/file，如果是 task/sub/run/file 请加上 .parent
            # 这里尝试直接查找
            fpath = LSS_ROOT / task_folder / row['subject'] / row['file']

            # 如果找不到，尝试加 run 文件夹 (容错)
            if not fpath.exists():
                fpath = LSS_ROOT / task_folder / row['subject'] / f"run-{row['run']}" / row['file']

        lookup_dict[(row['subject'], row['stimulus_content'])] = fpath

    # 5. 加载数据 (核心耗时步)
    subject_vectors = []
    final_subjects = []

    for sub in tqdm(valid_subjects_set, desc="  - Loading Data"):
        sub_features = []
        is_complete = True

        for content_key in common_keys:
            fpath = lookup_dict.get((sub, content_key))

            if fpath and fpath.exists():
                try:
                    data = flattener(fpath)
                    sub_features.append(data)
                except Exception as e:
                    # print(f"Error loading {fpath}: {e}")
                    is_complete = False
                    break
            else:
                # print(f"Missing: {sub} {content_key}")
                is_complete = False
                break

        if is_complete:
            # 拼接该被试所有刺激的 Beta 图
            # 形状: (n_stimuli * n_vertices, )
            flat_vec = np.concatenate(sub_features)
            subject_vectors.append(flat_vec)
            final_subjects.append(sub)

    # 6. 计算矩阵
    if final_subjects:
        print(f"  - 最终计算矩阵规模: {len(final_subjects)} Subjects")

        # 堆叠成大矩阵 (N_subs, N_features)
        # 注意内存：如果这里报错 MemoryError，需要优化
        data_matrix = np.stack(subject_vectors)

        # 主动释放列表内存
        del subject_vectors
        gc.collect()

        print("  - 正在计算相关系数...")
        sim_matrix = np.corrcoef(data_matrix)

        # 保存
        out_name = f"similarity_{context_name}_ContentAligned.csv"
        out_path = OUTPUT_DIR / out_name
        pd.DataFrame(sim_matrix, index=final_subjects, columns=final_subjects).to_csv(out_path)
        print(f"  ✅ 结果已保存: {out_path}")

        del data_matrix, sim_matrix
        gc.collect()
    else:
        print("  ❌ 加载后无有效被试 (可能文件路径不对或文件缺失)。")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 搜索所有 _aligned.csv 文件
    aligned_files = list(LSS_ROOT.glob("*_aligned.csv"))

    if not aligned_files:
        print(f"未找到 *_aligned.csv 文件！请先运行 generate_content_aligned_index.py")
        return

    # 2. 准备 Mask (如果算 Volume)
    vol_mask_img = None
    if any("volume" in f.name.lower() for f in aligned_files):
        if MNI_MASK_PATH.exists():
            print("加载 Volume Mask...")
            vol_mask_img = image.load_img(str(MNI_MASK_PATH))

    for idx_file in aligned_files:
        # 从文件名提取场景名 (去除 lss_index_ 和 _aligned)
        scenario_name = idx_file.stem.replace("lss_index_", "").replace("_aligned", "")

        is_volume = "volume" in scenario_name.lower()
        file_type = 'volume' if is_volume else 'surface'

        try:
            flattener = get_data_flattener(file_type, vol_mask_img if is_volume else None)
        except ValueError:
            continue

        df = pd.read_csv(idx_file)
        compute_matrix_with_content_alignment(df, scenario_name, flattener)

        gc.collect()


if __name__ == "__main__":
    main()