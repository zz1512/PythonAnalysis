#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_isc_combined_strict.py
功能：
1. 严格筛选：只保留同时拥有 EMO 和 SOC 数据的被试 (Intersection)。
2. 数据拼接：将 EMO 和 SOC 的 Beta图拼接成长向量。
3. 矩阵计算：计算被试间相似性。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from nilearn import surface, image
from tqdm import tqdm
import gc

# ================= 配置区 =================
# 结果根目录
LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")

# 输出目录
OUTPUT_DIR = LSS_ROOT / "similarity_matrices_combined"

# Volume Mask 路径
MNI_MASK_PATH = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz")

# 任务定义
TASK_A = 'EMO'
TASK_B = 'SOC'
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

def compute_strict_combined_matrix(df_full, context_name, flattener):
    """
    核心逻辑：严格匹配被试 -> 拼接数据 -> 计算矩阵
    """
    print(f"\n>>> 正在处理场景: {context_name}")
    
    # 1. 基础清洗：去除 Other
    df = df_full[~df_full['label'].str.contains('Other', case=False)].copy()
    
    # ---------------------------------------------------------
    # 【核心修改步骤】: 被试交集筛选 (Subject Intersection)
    # ---------------------------------------------------------
    # 确保 task 列存在且大写
    if 'task' not in df.columns:
        # 如果 csv 没 task 列，尝试从 file 路径或 run 推断 (根据你的逻辑 EMO=Run1, SOC=Run2)
        # 这是一个补救措施，假设 Run1=EMO, Run2=SOC
        df['task'] = df['run'].apply(lambda x: TASK_A if x == 1 else TASK_B)
    
    subjects_emo = set(df[df['task'] == TASK_A]['subject'])
    subjects_soc = set(df[df['task'] == TASK_B]['subject'])
    
    # 取交集
    valid_subjects_set = sorted(list(subjects_emo & subjects_soc))
    
    print(f"  - EMO 被试数: {len(subjects_emo)}")
    print(f"  - SOC 被试数: {len(subjects_soc)}")
    print(f"  - [筛选后] 双任务均有的被试数: {len(valid_subjects_set)}")
    
    if len(valid_subjects_set) == 0:
        print("  ❌ 错误：没有被试同时拥有两个任务的数据，停止计算。")
        return

    # 只保留这些被试的记录
    df = df[df['subject'].isin(valid_subjects_set)].copy()
    # ---------------------------------------------------------

    # 2. 构造 Global Key (Task + Label)
    df['global_key'] = df.apply(lambda row: f"{row['task']}_{row['label']}", axis=1)
    
    # 3. 筛选公共刺激 (Stimulus Intersection)
    # 要求：在剩下的这些有效被试中，90%以上的人都有这个刺激
    key_counts = df['global_key'].value_counts()
    threshold = int(len(valid_subjects_set) * 0.9)
    common_keys = key_counts[key_counts >= threshold].index.tolist()
    
    # 排序非常重要：确保前半部分是 EMO，后半部分是 SOC (或者字母序混合，只要一致即可)
    common_keys.sort() 
    
    if not common_keys:
        print("  ⚠️ 没有公共刺激，跳过。")
        return

    print(f"  - 用于计算的特征数 (Stimuli): {len(common_keys)}")

    # 4. 建立路径索引 (Lookup Dict)
    # key = (subject, global_key)
    lookup_dict = {}
    for _, row in df.iterrows():
        # 路径拼接
        task_folder = row['task'].lower() if 'task' in row else ('emo' if row['run'] == 1 else 'soc')
        run_folder = f"run-{row['run']}"
        fpath = LSS_ROOT / task_folder / row['subject'] / run_folder / row['file']
        lookup_dict[(row['subject'], row['global_key'])] = fpath

    # 5. 加载数据 (Loading)
    subject_vectors = []
    final_subjects = [] # 再次确认加载成功的被试
    
    for sub in tqdm(valid_subjects_set, desc="  - Loading Data"):
        sub_features = []
        is_complete = True
        
        for gkey in common_keys:
            fpath = lookup_dict.get((sub, gkey))
            
            if fpath and fpath.exists():
                try:
                    data = flattener(fpath)
                    sub_features.append(data)
                except Exception:
                    is_complete = False
                    break
            else:
                is_complete = False
                break
        
        if is_complete:
            # 拼接 EMO 和 SOC 的特征
            flat_vec = np.concatenate(sub_features)
            subject_vectors.append(flat_vec)
            final_subjects.append(sub)
    
    # 6. 计算与保存
    if final_subjects:
        print(f"  - 最终计算矩阵规模: {len(final_subjects)} Subjects")
        data_matrix = np.stack(subject_vectors)
        del subject_vectors
        gc.collect()
        
        # Pearson Correlation
        sim_matrix = np.corrcoef(data_matrix)
        
        # 保存
        out_name = f"similarity_{context_name}_Paired.csv"
        out_path = OUTPUT_DIR / out_name
        pd.DataFrame(sim_matrix, index=final_subjects, columns=final_subjects).to_csv(out_path)
        print(f"  ✅ 结果已保存: {out_path}")
        
        del data_matrix, sim_matrix
        gc.collect()
    else:
        print("  ❌ 加载数据后无有效被试（可能是文件缺失）。")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_files = list(LSS_ROOT.glob("lss_index_*.csv"))
    
    # 预加载 Volume Mask
    vol_mask_img = None
    if any("volume" in f.name for f in index_files):
        if MNI_MASK_PATH.exists():
            print("加载 Volume Mask...")
            vol_mask_img = image.load_img(str(MNI_MASK_PATH))
        else:
            print("⚠️ 未找到 Volume Mask，跳过 Volume 计算")

    for idx_file in index_files:
        scenario_name = idx_file.stem.replace("lss_index_", "")
        
        is_volume = "volume" in scenario_name
        file_type = 'volume' if is_volume else 'surface'
        
        try:
            flattener = get_data_flattener(file_type, vol_mask_img if is_volume else None)
        except ValueError:
            continue
            
        # 读取 CSV
        df = pd.read_csv(idx_file)
        
        # 直接传入完整 df，在函数内部做 intersection 筛选
        compute_strict_combined_matrix(df, scenario_name, flattener)
        
        gc.collect()

if __name__ == "__main__":
    main()