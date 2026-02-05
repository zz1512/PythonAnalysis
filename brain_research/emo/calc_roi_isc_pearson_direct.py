#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calc_roi_isc_pearson_direct.py

功能整合：
1. [来自 build_roi_beta_matrix_232]: 读取 Surface L/R 和 Volume 的 Atlas，定义 232 个 ROI。
2. [来自 build_roi_beta_matrix_232]: 智能筛选：保留覆盖率 >80% 的刺激，并筛选拥有这些刺激的被试。
3. [来自 calc_isc_combined_strict]: 读取年龄表，按年龄从小到大对被试进行严格排序。
4. [核心修复]: 逐个 ROI 提取数据，**保留空间维度** (Flatten & Concatenate)，不求平均。
5. [来自 calc_isc_combined_strict]: 计算 Pearson 相关系数。
6. 输出: 
   - roi_isc_pearson_by_age.npy (232 x N_sub x N_sub)
   - 对应的 subjects 和 rois 列表文件。

"""

import os
import re
import gc
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import image
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================= 配置区 =================

LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
OUTPUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")
OUTPUT_PREFIX = "roi_isc_pearson_by_age" # 输出文件前缀

# 索引文件
INDEX_SURFACE_L = LSS_ROOT / "lss_index_surface_L_aligned.csv"
INDEX_SURFACE_R = LSS_ROOT / "lss_index_surface_R_aligned.csv"
INDEX_VOLUME = LSS_ROOT / "lss_index_volume_aligned.csv"

# Atlas 路径
ATLAS_SURFACE_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
ATLAS_SURFACE_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
ATLAS_VOLUME = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")

# 年龄表
SUBJECT_INFO_PATH = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")

# ================= 辅助函数 =================

def parse_chinese_age(age_str: object) -> float:
    """解析年龄字符串为浮点数"""
    if pd.isna(age_str) or str(age_str).strip() == "":
        return 999.0
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    
    s = str(age_str).strip()
    try:
        return float(s)
    except:
        pass
        
    y_match = re.search(r"(\d+)\s*岁", s)
    m_match = re.search(r"(\d+)\s个月", s) or re.search(r"(\d+)\s月", s)
    d_match = re.search(r"(\d+)\s*天", s)
    
    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0
    
    return years + (months / 12.0) + (days / 365.0)

def load_subject_ages_map(info_path: Path) -> dict:
    """加载被试年龄映射"""
    if str(info_path).endswith('.tsv'):
        df = pd.read_csv(info_path, sep='\t')
    else:
        df = pd.read_csv(info_path) if str(info_path).endswith('.csv') else pd.read_excel(info_path)
    
    # 兼容不同列名
    sub_col = "sub_id" if "sub_id" in df.columns else "被试编号"
    age_col = "age" if "age" in df.columns else "采集年龄"
    
    age_map = {}
    for _, row in df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        age_map[sub_id] = parse_chinese_age(row[age_col])
    return age_map

def load_atlas_data(path: Path, space: str):
    """加载 Atlas 数据 (Volume NIfTI 或 Surface GIfTI)"""
    if space == 'volume':
        img = image.load_img(str(path))
        return img # 返回 NIfTI 对象以便重采样
    else:
        g = nib.load(str(path))
        return g.darrays[0].data.astype(int).reshape(-1) # 返回 Label 数组

def get_roi_definitions(atlas_data, space: str, prefix: str):
    """
    解析 Atlas，返回 ROI 列表。
    格式: [{'id': 1, 'name': 'L_1', 'indices': [...], 'space': 'surface_L'}, ...]
    """
    rois = []
    
    if space == 'volume':
        # Volume 需要先拿 data 数组来找 ID
        data_arr = atlas_data.get_fdata().reshape(-1)
    else:
        data_arr = atlas_data
        
    unique_ids = np.unique(data_arr)
    unique_ids = unique_ids[unique_ids > 0] # 去除背景 0
    unique_ids.sort()
    
    for uid in unique_ids:
        indices = np.where(data_arr == uid)[0]
        rois.append({
            'id': uid,
            'name': f"{prefix}_{int(uid)}",
            'indices': indices,
            'space': space,
            'atlas_obj': atlas_data # 存入对象，Volume 需要用到 affine
        })
    return rois

def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    """解析文件路径"""
    f = str(row.get("file", ""))
    if f.startswith("/"): return Path(f)
    
    task = str(row.get("task", "")).lower()
    if not task and str(row.get("stimulus_content", "")).startswith("EMO"): task = "emo"
    if not task and str(row.get("stimulus_content", "")).startswith("SOC"): task = "soc"
    
    sub = str(row.get("subject", ""))
    # Try direct path
    p = lss_root / task / sub / f
    if p.exists(): return p
    # Try run path
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f

# ================= 核心处理逻辑 =================

def intelligent_filter(df_list, threshold_ratio=0.80):
    """
    合并多个空间的 DataFrame，进行智能筛选：
    1. 筛选出出现在 >80% 被试中的刺激。
    2. 筛选出拥有所有这些“共性刺激”的被试。
    """
    # 合并只为了统计 stimulus_content
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['clean_id'] = full_df['stimulus_content'].astype(str).str.strip()
    full_df = full_df[full_df['clean_id'].str.len() > 1]
    
    # 统计刺激覆盖率
    unique_pairs = full_df[['subject', 'clean_id']].drop_duplicates()
    n_total_subs = unique_pairs['subject'].nunique()
    stim_counts = unique_pairs['clean_id'].value_counts()
    
    threshold = n_total_subs * threshold_ratio
    valid_stimuli = stim_counts[stim_counts >= threshold].index.tolist()
    valid_stimuli.sort()
    
    logger.info(f"刺激筛选: 总被试 {n_total_subs}, 阈值 {threshold:.1f}, 选中刺激数 {len(valid_stimuli)}")
    
    if not valid_stimuli:
        raise ValueError("没有刺激满足覆盖率要求！")

    # 反向筛选被试
    df_valid = unique_pairs[unique_pairs['clean_id'].isin(valid_stimuli)]
    sub_counts = df_valid.groupby('subject')['clean_id'].nunique()
    # 必须拥有所有选中的刺激
    valid_subjects = sub_counts[sub_counts == len(valid_stimuli)].index.tolist()
    
    logger.info(f"被试筛选: 拥有全部 {len(valid_stimuli)} 个刺激的被试数: {len(valid_subjects)}")
    
    return valid_subjects, valid_stimuli

def get_subject_roi_vector(
    subject: str, 
    roi_def: dict, 
    stimuli: list, 
    lookup: dict
) -> np.ndarray:
    """
    【核心修复】获取单个被试、单个ROI的特征向量。
    逻辑：
    1. 遍历所有公共刺激 (stimuli)。
    2. 加载每个刺激对应的 beta 图。
    3. 提取 ROI 掩膜内的所有体素/顶点值。
    4. 将所有刺激的向量 **拼接 (Concatenate)** 起来。
    
    结果维度: (N_stimuli * N_vertices_in_ROI, )
    """
    feats = []
    space = roi_def['space']
    indices = roi_def['indices']
    atlas_obj = roi_def['atlas_obj']
    
    for stim in stimuli:
        path = lookup.get((subject, stim))
        if path is None or not path.exists():
            # 理论上经过智能筛选不应该缺文件，但为了稳健性处理
            return None 
            
        try:
            if space == 'volume':
                img = image.load_img(str(path))
                # 检查维度匹配，必要时重采样 (Nearest Neighbor 以保持 label 边界)
                if img.shape != atlas_obj.shape or not np.allclose(img.affine, atlas_obj.affine):
                    img = image.resample_to_img(img, atlas_obj, interpolation='nearest')
                data = img.get_fdata().reshape(-1)
            else:
                # Surface
                g = nib.load(str(path))
                data = g.darrays[0].data.astype(np.float32).reshape(-1)
            
            # 提取 ROI 内数据
            roi_data = data[indices]
            
            # 简单的 NaN 处理
            if np.isnan(roi_data).any():
                roi_data = np.nan_to_num(roi_data)
                
            feats.append(roi_data)
            
        except Exception as e:
            logger.error(f"加载失败 {subject} {stim}: {e}")
            return None

    if not feats:
        return None
        
    # 拼接成 1D 长向量 [Stim1_Vox1, Stim1_Vox2, ..., Stim2_Vox1, ...]
    return np.concatenate(feats)

# ================= 主程序 =================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载年龄并构建映射
    age_map = load_subject_ages_map(SUBJECT_INFO_PATH)
    logger.info(f"已加载年龄信息: {len(age_map)} 条")

    # 2. 读取索引并清理
    logger.info("读取索引文件...")
    dfs = {}
    for name, path in [('surface_L', INDEX_SURFACE_L), ('surface_R', INDEX_SURFACE_R), ('volume', INDEX_VOLUME)]:
        df = pd.read_csv(path)
        df['clean_id'] = df['stimulus_content'].astype(str).str.strip()
        dfs[name] = df

    # 3. 智能筛选 (基于所有空间的合集)
    all_dfs = list(dfs.values())
    valid_subjects, valid_stimuli = intelligent_filter(all_dfs, threshold_ratio=0.80)
    
    # 4. **按年龄排序被试** (关键步骤)
    # 排序键: (年龄, ID) 以确保确定性
    valid_subjects.sort(key=lambda s: (age_map.get(s, 999.0), s))
    
    sorted_ages = [age_map.get(s, np.nan) for s in valid_subjects]
    logger.info(f"排序后被试 (Min Age: {min(sorted_ages):.1f}, Max Age: {max(sorted_ages):.1f})")
    
    # 5. 构建查找表 (Subject, Stimulus) -> Path
    # 需要分空间构建，因为同一个 stim 在 L/R/Vol 对应的文件不同
    lookups = {}
    for space, df in dfs.items():
        lk = {}
        # 只保留有效被试的数据加速查找
        sub_df = df[df['subject'].isin(valid_subjects)]
        for _, row in sub_df.iterrows():
            lk[(str(row['subject']), str(row['clean_id']))] = resolve_beta_path(LSS_ROOT, row)
        lookups[space] = lk

    # 6. 定义所有 ROI
    logger.info("定义 ROI...")
    all_roi_defs = []
    
    # Surface L
    atl_L = load_atlas_data(ATLAS_SURFACE_L, 'surface')
    all_roi_defs.extend(get_roi_definitions(atl_L, 'surface_L', 'L'))
    
    # Surface R
    atl_R = load_atlas_data(ATLAS_SURFACE_R, 'surface')
    all_roi_defs.extend(get_roi_definitions(atl_R, 'surface_R', 'R'))
    
    # Volume
    atl_V = load_atlas_data(ATLAS_VOLUME, 'volume')
    all_roi_defs.extend(get_roi_definitions(atl_V, 'volume', 'V'))
    
    n_rois = len(all_roi_defs)
    n_subs = len(valid_subjects)
    logger.info(f"总计 ROI: {n_rois} (预期 232)")
    
    # 7. 主计算循环
    # 结果矩阵: (ROI, Subject, Subject)
    final_isc_matrix = np.zeros((n_rois, n_subs, n_subs), dtype=np.float32)
    roi_names_list = []
    
    logger.info("开始逐 ROI 计算 ISC (Pearson, Spatial+Temporal)...")
    
    for i, roi_def in enumerate(tqdm(all_roi_defs, desc="Processing ROIs")):
        roi_name = roi_def['name']
        roi_names_list.append(roi_name)
        space = roi_def['space']
        current_lookup = lookups[space]
        
        # 收集该 ROI 下所有被试的长向量
        subject_vectors = []
        valid_for_this_roi = True
        
        for sub in valid_subjects:
            vec = get_subject_roi_vector(sub, roi_def, valid_stimuli, current_lookup)
            if vec is None:
                logger.warning(f"ROI {roi_name}: 被试 {sub} 数据提取失败，该 ROI 将被跳过或全填NaN")
                valid_for_this_roi = False
                break
            subject_vectors.append(vec)
            
        if not valid_for_this_roi:
            final_isc_matrix[i] = np.nan
            continue
            
        # 堆叠 -> (N_subs, N_features)
        # N_features = N_stimuli * N_vertices_in_ROI
        data_mat = np.stack(subject_vectors)
        
        # 计算 Pearson 相关矩阵
        # np.corrcoef 对行进行相关计算
        sim_mat = np.corrcoef(data_mat)
        
        final_isc_matrix[i] = sim_mat
        
        # 内存回收
        del data_mat, subject_vectors
        # 仅在必要时强制回收，避免拖慢速度
        if i % 20 == 0: gc.collect()

    # 8. 保存结果
    logger.info("保存结果...")
    
    # 保存大矩阵 .npy
    out_npy = OUTPUT_DIR / f"{OUTPUT_PREFIX}.npy"
    np.save(out_npy, final_isc_matrix)
    
    # 保存被试列表 (带年龄)
    out_subs = OUTPUT_DIR / f"{OUTPUT_PREFIX}_subjects_sorted.csv"
    pd.DataFrame({"subject": valid_subjects, "age": sorted_ages}).to_csv(out_subs, index=False)
    
    # 保存 ROI 列表
    out_rois = OUTPUT_DIR / f"{OUTPUT_PREFIX}_rois.csv"
    pd.DataFrame({"roi": roi_names_list}).to_csv(out_rois, index=False)
    
    # 保存刺激列表 (Reference)
    out_stim = OUTPUT_DIR / f"{OUTPUT_PREFIX}_stimuli.csv"
    pd.DataFrame({"stimulus_content": valid_stimuli}).to_csv(out_stim, index=False)
    
    logger.info(f"✅ 完成! 结果保存在: {OUTPUT_DIR}")
    logger.info(f"矩阵形状: {final_isc_matrix.shape}")

if __name__ == "__main__":
    main()