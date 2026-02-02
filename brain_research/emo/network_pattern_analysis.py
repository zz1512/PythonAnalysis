#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
network_pattern_analysis_v4.py [Yeo 7 Networks - Multivariate Pattern版]

核心改进：
不再对 Network 内的 Parcels 取平均！
而是将 Network 内的所有 Parcel Beta 值拼接成一个长向量（Spatial Pattern）。
这保留了网络内部的拓扑结构信息（Topography），避免了平均化导致的信号对冲。

预期：
这将大幅提高 r_obs，结果应该会接近全脑分析的显著性，但具有网络特异性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm


# ================= 配置区 =================

@dataclass(frozen=True)
class NetworkAnalysisPreset:
    name: str
    aligned_csv: Path
    atlas_label: Path
    atlas_info_txt: Path
    subject_info: Path
    out_dir: Path
    space: str
    hemi_tag: str
    task: str
    condition_group: str
    lss_root: Optional[Path] = None
    min_coverage: float = 0.9
    n_perm: int = 5000
    seed: int = 42
    normalize_models: bool = True


PRESETS: Dict[str, NetworkAnalysisPreset] = {
    "surface_L_EMO_Pattern": NetworkAnalysisPreset(
        name="surface_L_EMO_Pattern",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
        atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
        atlas_info_txt=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_order_info.txt"),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/network_pattern_out"),
        space="surface_L",
        hemi_tag="LH",
        task="",
        condition_group="all",
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
    )
}

ACTIVE_PRESET = "surface_L_EMO_Pattern"


# ================= 辅助函数 (复用 v3) =================

def standardize_sub_id(raw_id: str) -> str:
    s = str(raw_id).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def get_subjects_intersection(df: pd.DataFrame, subject_info_path: Path) -> Tuple[List[str], Dict[str, float]]:
    age_df = pd.read_csv(subject_info_path)
    sub_col = next((c for c in ['sub_id', '被试编号', 'subject'] if c in age_df.columns), None)
    age_col = next((c for c in ['age', '采集年龄'] if c in age_df.columns), None)

    if not sub_col or not age_col: raise ValueError("年龄表列名无法识别")

    age_map = {}
    for _, row in age_df.iterrows():
        sid = standardize_sub_id(row[sub_col])
        try:
            val = str(row[age_col]).split('岁')[0]
            age_map[sid] = float(val)
        except:
            pass

    df['std_subject'] = df['subject'].apply(standardize_sub_id)
    subs_emo = set(df[df['task'] == 'EMO']['std_subject'].unique())
    subs_soc = set(df[df['task'] == 'SOC']['std_subject'].unique())

    valid_tasks_subs = subs_emo & subs_soc if len(subs_soc) > 0 else subs_emo
    final_subs = sorted(list(valid_tasks_subs & set(age_map.keys())))
    final_subs.sort(key=lambda s: age_map[s])
    return final_subs, age_map


def parse_schaefer_lut(txt_path: Path, hemi_tag: str) -> Dict[int, str]:
    mapping = {}
    if not txt_path.exists(): raise FileNotFoundError(f"Missing info.txt: {txt_path}")
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    target_networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    hemi_lines = [line for line in lines if f"_{hemi_tag}_" in line]
    print(f"  Parsed {len(hemi_lines)} parcels for {hemi_tag}.")
    for idx, line in enumerate(hemi_lines):
        parcel_id = idx + 1
        label_name = line.strip().split()[0]
        matched_net = "Other"
        for net in target_networks:
            if f"_{net}_" in label_name:
                matched_net = net
                break
        mapping[parcel_id] = matched_net
    return mapping


def load_label_gii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).darrays[0].data).reshape(-1).astype(np.int32)


def load_beta_gii(path: Path) -> Optional[np.ndarray]:
    try:
        return np.asarray(nib.load(str(path)).darrays[0].data).reshape(-1).astype(np.float32)
    except:
        return None


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", ""))
    if f.startswith("/"): return Path(f)
    task = str(row.get("task", "")).lower()
    sub = standardize_sub_id(row.get("subject", ""))
    p1 = lss_root / task / sub / f
    if p1.exists(): return p1
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def zscore_1d(x):
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if int(mask.sum()) < 2: return x
    mu = x[mask].mean()
    sd = x[mask].std(ddof=0)
    if sd <= 1e-9: return x
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out


def build_model_vec(ages, n_sub, model, normalize):
    iu, ju = np.triu_indices(n_sub, k=1)
    a = ages.reshape(-1)
    ai, aj = a[iu], a[ju]
    amax = np.nanmax(a)
    if model == "M_nn":
        v = amax - np.abs(ai - aj)
    elif model == "M_conv":
        v = np.minimum(ai, aj)
    elif model == "M_div":
        v = amax - 0.5 * (ai + aj)
    return zscore_1d(v) if normalize else v


# ================= 核心修改：Pattern 计算 =================

def compute_network_pattern_similarity(parcel_feats, parcel_ids, lut_mapping, n_sub, n_key):
    """
    计算 Network Pattern Similarity
    input: parcel_feats (n_sub, n_parcel, n_key)
    logic: 对于每个 Network，取出属于它的所有 parcels，展平成一个大向量 (Spatial Pattern)
    output: 7个网络的相似性矩阵向量
    """
    unique_networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    iu, ju = np.triu_indices(n_sub, k=1)

    sim_vectors = []
    valid_net_names = []

    print("\n[Pattern Analysis] Calculating Spatial Pattern Similarity per Network...")

    for net_name in unique_networks:
        # 1. 找到该网络下的 Parcel Indices
        target_pids = [pid for pid in parcel_ids if lut_mapping.get(pid) == net_name]
        if not target_pids: continue

        try:
            indices = [np.where(parcel_ids == pid)[0][0] for pid in target_pids]
        except IndexError:
            continue

        # 2. 提取数据 (n_sub, n_subset_parcels, n_key)
        # 这是一个 3D 矩阵，包含了网络内的空间信息(维度1)和时间信息(维度2)
        net_data = parcel_feats[:, indices, :]

        # 3. 展平 (Flatten) -> (n_sub, n_features)
        # 将空间维度和 Stimulus 维度合并，构成该被试在该网络下的“反应指纹”
        # n_features = n_subset_parcels * n_key
        # 例如 Vis 有 30 个 parcels, 100 个 keys -> 特征长度 3000
        n_feat = net_data.shape[1] * net_data.shape[2]

        # Reshape: (n_sub, n_feat)
        # 注意: 这种 Flatten 保留了 Parcel 1 的 Key1..N, Parcel 2 的 Key1..N...
        # 实际上这就是一种 Pattern
        Y_pattern = net_data.reshape(n_sub, -1)

        # 4. 检查 NaN
        if np.isnan(Y_pattern).all():
            print(f"  Skipping {net_name}: All NaN")
            continue

        # 填补局部 NaN (Parcel 级缺失填 0)
        Y_pattern = np.nan_to_num(Y_pattern, nan=0.0)

        # 5. 计算相关 (Subject x Subject)
        # 先对 Feature 维度做 Z-score (Standardize pattern)
        mu = Y_pattern.mean(axis=1, keepdims=True)
        sd = Y_pattern.std(axis=1, keepdims=True, ddof=0)
        Z = (Y_pattern - mu) / (sd + 1e-9)

        # Correlation Matrix
        S = (Z @ Z.T) / n_feat

        # 取上三角
        v = S[iu, ju]
        sim_vectors.append(zscore_1d(v))
        valid_net_names.append(net_name)
        print(f"  - {net_name}: Pattern Vector Length = {n_feat}, Obs Corr Mean = {v.mean():.4f}")

    if not sim_vectors: raise ValueError("No valid networks found.")

    return np.stack(sim_vectors), valid_net_names, iu, ju


# ================= 主流程 =================

def run_pattern_analysis(preset: NetworkAnalysisPreset):
    print(f"\n{'=' * 60}")
    print(f"开始 Network Pattern Analysis (v4 - MVPA): {preset.name}")
    print(f"{'=' * 60}")

    preset.out_dir.mkdir(parents=True, exist_ok=True)
    lut_mapping = parse_schaefer_lut(preset.atlas_info_txt, preset.hemi_tag)

    # 1. 读取 & 筛选
    df = pd.read_csv(preset.aligned_csv)
    subjects, age_map = get_subjects_intersection(df, preset.subject_info)
    ages = np.array([age_map[s] for s in subjects])
    n_sub = len(subjects)

    # 2. Stimuli
    df_task = df[df['task'] == preset.task]
    std_subs_series = df_task['subject'].apply(standardize_sub_id)
    counts = df_task[std_subs_series.isin(subjects)]["stimulus_content"].value_counts()
    threshold = int(n_sub * preset.min_coverage)
    keys = counts[counts >= threshold].index.tolist()
    keys.sort()
    print(f"Subjects: {n_sub}, Keys: {len(keys)}")

    # 3. 加载 Parcel Beta (复用 v3 逻辑)
    labels = load_label_gii(preset.atlas_label)
    parcel_ids = np.unique(labels)
    parcel_ids = parcel_ids[parcel_ids > 0]

    # 构建 Lookup
    key_set = set(keys)
    df_load = df_task[df_task["stimulus_content"].isin(key_set)].copy()
    lookup = {}
    for _, row in df_load.iterrows():
        sid = standardize_sub_id(row["subject"])
        content = str(row["stimulus_content"])
        lookup[(sid, content)] = resolve_beta_path(preset.lss_root, row)

    # 加载数据 (耗时步骤)
    n_parcel = int(parcel_ids.size)
    max_label = int(labels.max())
    label_counts = np.bincount(labels, minlength=max_label + 1).astype(np.float32)
    label_counts[label_counts == 0] = np.nan

    parcel_feats = np.full((n_sub, n_parcel, len(keys)), np.nan, dtype=np.float32)

    loaded_cnt = 0
    for si, sub in enumerate(tqdm(subjects, desc="Loading Data")):
        has_data = False
        for ki, key in enumerate(keys):
            p = lookup.get((sub, key))
            if p and p.exists():
                beta = load_beta_gii(p)
                if beta is not None:
                    sums = np.bincount(labels, weights=beta, minlength=max_label + 1).astype(np.float32)
                    means = sums[parcel_ids] / label_counts[parcel_ids]
                    parcel_feats[si, :, ki] = means
                    has_data = True
        if has_data: loaded_cnt += 1

    if loaded_cnt == 0: raise ValueError("No data loaded.")

    # 4. [核心] 计算 Network Pattern Similarity
    S_z, net_names, iu, ju = compute_network_pattern_similarity(parcel_feats, parcel_ids, lut_mapping, n_sub, len(keys))

    # 5. 统计
    models = ["M_nn", "M_conv", "M_div"]
    model_vecs = {m: build_model_vec(ages, n_sub, m, preset.normalize_models) for m in models}
    n_pairs = len(iu)

    r_obs_dict = {m: (S_z @ model_vecs[m]) / n_pairs for m in models}
    r_obs_flat = np.concatenate([r_obs_dict[m] for m in models])

    print(f"Running Permutation ({preset.n_perm})...")
    count_raw = np.zeros(len(r_obs_flat))
    count_fwer = np.zeros(len(r_obs_flat))
    rng = np.random.default_rng(preset.seed)

    for _ in range(preset.n_perm):
        perm_idx = rng.permutation(n_sub)
        ages_p = ages[perm_idx]
        flat_list = []
        for m in models:
            mv_p = build_model_vec(ages_p, n_sub, m, preset.normalize_models)
            r_p = (S_z @ mv_p) / n_pairs
            flat_list.append(r_p)
        perm_flat = np.concatenate(flat_list)
        count_raw += (perm_flat >= r_obs_flat)
        count_fwer += (perm_flat.max() >= r_obs_flat)

    p_raw = (count_raw + 1) / (preset.n_perm + 1)
    p_fwer = (count_fwer + 1) / (preset.n_perm + 1)

    # 6. 保存
    rows = []
    idx = 0
    for m in models:
        rs = r_obs_dict[m]
        for i, net in enumerate(net_names):
            rows.append({
                "network": net,
                "model": m,
                "r_obs": rs[i],
                "p_perm": p_raw[idx],
                "p_fwer": p_fwer[idx],
                "n_subjects": n_sub
            })
            idx += 1

    df_res = pd.DataFrame(rows)
    out_path = preset.out_dir / f"network_pattern_analysis_{preset.space}_{preset.task}.csv"
    df_res.to_csv(out_path, index=False)

    print("\n🎉 Significant Networks (FWER < 0.05):")
    print(df_res[df_res['p_fwer'] < 0.05])
    print("\n🔍 Exploratory (Uncorrected < 0.05):")
    print(df_res[df_res['p_perm'] < 0.05])


if __name__ == "__main__":
    run_pattern_analysis(PRESETS[ACTIVE_PRESET])