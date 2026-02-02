#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_roi_joint_analysis_dev_models_perm_fwer.py [终极整合版]

功能：
1. 读取 aligned CSV (beta files)，构建被试间相似性矩阵 (ISC)。
2. 支持 Surface (全脑/ROI) 和 Volume (全脑/Mask) 数据读取。
3. 与发育模型 (NN, Conv, Div) 进行联合分析。
4. 统计检验：5000次置换 + 单侧检验(正效应) + FWER (Max-Statistic) 校正。

核心逻辑：
- 严格对齐被试：自动寻找 `subject_order_surface_L.csv` 确保与全脑分析被试一致。
- 数据清洗：使用 "Ensure Complete Keys" 策略，优先剔除缺失的刺激以保留完整被试。
- 兼容性：同时支持 .gii (Surface) and .nii.gz (Volume)。
"""

from __future__ import annotations

import re
import gc
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from tqdm import tqdm

# ================= 配置常量 =================

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

# ROI 定义示例 (Schaefer 200 7Networks)
ROI_CONFIG = {
    "Visual": {
        "L": [1, 2, 3, 4, 5, 6, 7],
        "R": [101, 102, 103, 104, 105, 106, 107],
    },
    "Salience_Insula": {
        "L": [38, 39, 40, 41, 42],
        "R": [138, 139, 140, 141, 142],
    },
    "Control_PFC": {
        "L": [57, 58, 59, 60, 61, 62],
        "R": [157, 158, 159, 160, 161, 162],
    },
}


# ================= 数据结构定义 =================

@dataclass(frozen=True)
class SimilaritySpec:
    name: str
    aligned_csv: Path
    space: str  # "surface_L", "surface_R", "volume"
    kind: str  # "combined", "roi"
    atlas_label: Optional[Path] = None  # ROI 模式必填 (Surface)
    roi_labels: Optional[Sequence[int]] = None  # ROI 模式必填
    volume_mask: Optional[Path] = None  # Volume 模式必填


@dataclass(frozen=True)
class JointPreset:
    name: str
    specs: Sequence[SimilaritySpec]
    subject_info: Path
    out_dir: Path
    lss_root: Optional[Path]
    task: str  # "EMO", "SOC"
    condition_group: str  # "all", "emotion_regulation" etc.
    min_coverage: float = 0.9
    ensure_complete_keys: bool = True  # 核心策略：丢刺激，保被试
    max_missing_stimuli: int = 0  # 允许每个被试缺失多少个刺激 (建议为0)
    n_perm: int = 5000  # 默认 5000 次
    seed: int = 42
    normalize_models: bool = True
    save_similarity: bool = True
    subject_order_path: Optional[Path] = None  # 可选：手动指定被试列表文件


# ================= 预设配置区 (请在此修改) =================

ACTIVE_PRESET = "EXAMPLE_EMO_joint_all"

PRESETS: Dict[str, JointPreset] = {
    "EXAMPLE_EMO_joint_all": JointPreset(
        name="EXAMPLE_EMO_joint_all",
        specs=(
            # --- 1. Surface 全脑 (Combined) ---
            SimilaritySpec(
                name="surface_L",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="combined",
            ),
            SimilaritySpec(
                name="surface_R",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="combined",
            ),

            # --- 2. Volume 全脑 (需提供 Mask) ---
            SimilaritySpec(
                name="volume",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"),
                space="volume",
                kind="combined",
                volume_mask=Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz"),
            ),

            # --- 3. Surface ROI 示例 ---
            SimilaritySpec(
                name="surface_L_Visual",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="roi",
                atlas_label=Path(
                    "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                roi_labels=ROI_CONFIG["Visual"]["L"],
            ),
            SimilaritySpec(
                name="surface_R_Visual",
                aligned_csv=Path(
                    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="roi",
                atlas_label=Path(
                    "/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                roi_labels=ROI_CONFIG["Visual"]["R"],
            ),
        ),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/joint_perm_fwer_out_roi"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="EMO",
        condition_group="all",
        n_perm=5000,
    )
}


# ================= 辅助函数：年龄与被试解析 =================

def parse_chinese_age_exact(age_str: object) -> float:
    """健壮的年龄解析函数"""
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan

    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)

    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass

    y_match = re.search(r'(\d+)\s*岁', s)
    m_match = re.search(r'(\d+)\s*个月', s) or re.search(r'(\d+)\s*月', s)
    d_match = re.search(r'(\d+)\s*天', s)

    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0

    return years + (months / 12.0) + (days / 365.0)


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    path_str = str(subject_info_path)
    if path_str.endswith(".tsv") or path_str.endswith(".txt"):
        info_df = pd.read_csv(path_str, sep="\t")
    elif path_str.endswith(".xlsx") or path_str.endswith(".xls"):
        info_df = pd.read_excel(path_str)
    elif path_str.endswith(".csv"):
        info_df = pd.read_csv(path_str)
    else:
        raise ValueError("不支持的文件格式，请使用 .tsv, .csv 或 .xlsx")

    # 兼容列名
    if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError(f"年龄表缺少必要列: {info_df.columns}")

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = f"sub-{raw_id}" if not raw_id.startswith("sub-") else raw_id
        age_map[sub_id] = parse_chinese_age_exact(row[age_col])
    return age_map


def load_subject_order_list(order_path: Path) -> List[str]:
    """读取 CSV 中的 subject 列"""
    if not order_path.exists():
        raise FileNotFoundError(f"未找到顺序文件: {order_path}")
    df = pd.read_csv(order_path)
    # 尝试常见的列名
    for col in ["subject", "sub_id", "被试编号"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    # 保底：取第一列
    return df.iloc[:, 0].astype(str).tolist()


# ================= 辅助函数：文件加载与数据清洗 =================

def infer_condition_group(row: pd.Series) -> str:
    cg = str(row.get("condition_group", "")).strip()
    if cg and cg.lower() != "nan": return cg
    raw_cond = str(row.get("raw_condition", "")).strip()
    raw_emo = str(row.get("raw_emotion", "")).strip()
    if raw_cond == "Reappraisal": return "emotion_regulation"
    if raw_cond == "PassiveLook":
        return "neutral_response" if raw_emo.lower().startswith("neutral") else "emotion_generation"
    return "other"


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", ""))
    if f.startswith("/"): return Path(f)
    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))
    # 优先查找: root/task/sub/file
    p1 = lss_root / task / sub / f
    if p1.exists(): return p1
    # 其次查找: root/task/sub/run-x/file
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def filter_dataframe(df: pd.DataFrame, task: str, condition_group: str) -> pd.DataFrame:
    """基础过滤：筛选任务和条件组"""
    # 必须包含这些列
    df = df.dropna(subset=["subject", "stimulus_content", "file", "task"]).copy()
    df["subject"] = df["subject"].astype(str)
    df["task"] = df["task"].astype(str)
    df["condition_group"] = df.apply(infer_condition_group, axis=1)

    if task:
        df = df[df["task"] == task].copy()
    if condition_group and condition_group != "all":
        df = df[df["condition_group"] == condition_group].copy()
    return df


def determine_subject_list(
        preset: JointPreset,
        age_map: Dict[str, float],
        dfs: List[pd.DataFrame]
) -> List[str]:
    """
    核心逻辑：确定最终被试列表。
    1. 优先使用 preset 指定的 subject_order_path。
    2. 其次自动查找 out_dir 下的 subject_order_surface_L.csv (或默认final目录)。
    3. 最后计算所有 dfs 的被试交集并按年龄排序。
    """
    target_order_path = preset.subject_order_path

    # --- 自动查找逻辑 ---
    if target_order_path is None:
        # 1. 检查当前输出目录
        auto_path = preset.out_dir / "subject_order_surface_L.csv"

        # 2. 检查默认的 calc_isc_combined_strict 输出目录 (Common Fallback)
        default_final_dir = Path(
            "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
        alt_path = default_final_dir / "subject_order_surface_L.csv"

        if auto_path.exists():
            print(f"  [Info] 自动检测到被试列表(Current Dir): {auto_path}")
            target_order_path = auto_path
        elif alt_path.exists():
            print(f"  [Info] 自动检测到被试列表(Default Final Dir): {alt_path}")
            target_order_path = alt_path
        else:
            print("  [Info] 未检测到现有的 subject_order 文件，将通过交集计算。")

    subjects_final = []

    if target_order_path and target_order_path.exists():
        print(f"  [Process] 使用外部文件限制被试范围: {target_order_path}")
        ordered_subs = load_subject_order_list(target_order_path)

        # 验证这些被试是否在所有 aligned CSV 中都存在
        available_sets = [set(d["subject"].unique()) for d in dfs]
        common_available = set.intersection(*available_sets)

        valid_subs = [s for s in ordered_subs if s in common_available]

        # 过滤无年龄数据的
        valid_subs = [s for s in valid_subs if np.isfinite(age_map.get(s, np.nan))]

        diff = len(ordered_subs) - len(valid_subs)
        if diff > 0:
            print(f"  [Warn] 外部列表中的 {diff} 个被试在当前数据中缺失或无年龄，已剔除。")
        subjects_final = valid_subs

    else:
        print("  [Process] 计算所有输入数据的被试交集...")
        sets = [set(d["subject"].unique()) for d in dfs]
        common = set.intersection(*sets)

        # 按年龄排序
        sorted_common = sorted(list(common), key=lambda s: (age_map.get(s, np.nan), s))
        # 过滤无年龄
        subjects_final = [s for s in sorted_common if np.isfinite(age_map.get(s, np.nan))]

    print(f"  [Result] 最终确定的被试数量: {len(subjects_final)}")
    if subjects_final:
        print(f"           First: {subjects_final[0]}, Last: {subjects_final[-1]}")

    return subjects_final


def load_beta_data(
        spec: SimilaritySpec,
        df: pd.DataFrame,
        subjects: List[str],
        lss_root: Path,
        preset: JointPreset,
) -> Optional[np.ndarray]:
    """
    加载数据并展平，返回 (n_sub, n_features)。
    包含 Volume/Surface 处理及 ROI 提取逻辑。
    包含 'Ensure Complete Keys' 逻辑。
    """

    # --- 1. 确定公共刺激 ---
    df_sub = df[df["subject"].isin(subjects)].copy()
    counts = df_sub["stimulus_content"].value_counts()

    # 刺激覆盖率阈值
    threshold = int(len(subjects) * preset.min_coverage)
    keys = counts[counts >= threshold].index.tolist()
    keys.sort()

    if not keys:
        print(f"  [Skip] {spec.name}: 无公共刺激 (Threshold={threshold})。")
        return None

    # --- 2. 建立路径查找表 ---
    lookup = {}
    for _, row in df_sub.iterrows():
        lookup[(str(row["subject"]), str(row["stimulus_content"]))] = resolve_beta_path(lss_root, row)

    # --- 3. 过滤完整性 (Ensure Complete Keys) ---
    # 策略：如果一个刺激在任何一个被试中缺失，则剔除该刺激，以保证被试数量不减少。
    if preset.ensure_complete_keys:
        valid_keys = []
        for k in keys:
            # 检查是否所有 subjects 都有这个 key 的文件
            is_complete = True
            for s in subjects:
                fpath = lookup.get((s, k))
                if fpath is None or not fpath.exists():
                    is_complete = False
                    break
            if is_complete:
                valid_keys.append(k)

        dropped = len(keys) - len(valid_keys)
        if dropped > 0:
            print(f"  [Data] {spec.name}: 为保被试完整，剔除 {dropped} 个有缺失的刺激。剩余刺激数: {len(valid_keys)}")
        keys = valid_keys

    if not keys:
        print(f"  [Skip] {spec.name}: 剔除缺失刺激后，无剩余刺激。")
        return None

    # --- 4. 准备 Mask (Volume 或 ROI) ---
    roi_mask = None  # for Surface ROI
    vol_mask_idx = None  # for Volume
    mask_img_vol = None  # for Volume Resampling

    # A. Surface ROI
    if spec.space.startswith("surface") and spec.kind == "roi":
        if not spec.atlas_label or not spec.roi_labels:
            raise ValueError(f"{spec.name}: ROI 模式需要 atlas_label 和 roi_labels")
        g = nib.load(str(spec.atlas_label))
        # 假设 label 是第一个 darray
        labels = np.asarray(g.darrays[0].data).reshape(-1)
        roi_mask = np.isin(labels, spec.roi_labels)
        print(f"  [Mask] Surface ROI mask created. Voxels/Vertices: {np.sum(roi_mask)}")

    # B. Volume
    elif spec.space == "volume":
        if not spec.volume_mask:
            raise ValueError(f"{spec.name}: Volume 模式需要 volume_mask")
        mask_img_vol = image.load_img(str(spec.volume_mask))
        mask_data = mask_img_vol.get_fdata() > 0
        vol_mask_idx = np.where(mask_data.reshape(-1))[0]
        print(f"  [Mask] Volume mask loaded. Voxels: {len(vol_mask_idx)}")

    # --- 5. 加载数据 ---
    matrix_list = []
    print(f"  [Load] Loading data for {spec.name} ({len(keys)} stimuli x {len(subjects)} subjects)...")

    # 检查最大允许缺失 (虽然 ensure_complete_keys 应该已经解决了这个问题，但作为双重保障)
    for s in subjects:
        for k in keys:
            if not (lookup.get((s, k)) and lookup[(s, k)].exists()):
                # 如果代码走到这里，说明 ensure_complete_keys=False 或者逻辑有漏网之鱼
                msg = f"被试 {s} 缺失刺激 {k}。当前配置不允许缺失(或已通过keys过滤)。"
                if preset.max_missing_stimuli == 0:
                    raise FileNotFoundError(msg)

    for s in tqdm(subjects, leave=False, desc=f"Reading {spec.name}"):
        feats = []
        for k in keys:
            p = lookup.get((s, k))
            # 这里 assume 文件存在，因为上面已经过滤了 keys
            try:
                if spec.space.startswith("surface"):
                    # Surface Load
                    g = nib.load(str(p))
                    data = g.darrays[0].data.astype(np.float32).reshape(-1)
                    if roi_mask is not None:
                        data = data[roi_mask]
                else:
                    # Volume Load
                    img = image.load_img(str(p))
                    # Resample if needed
                    if img.shape != mask_img_vol.shape or not np.allclose(img.affine, mask_img_vol.affine):
                        img = image.resample_to_img(img, mask_img_vol, interpolation="nearest")

                    data = img.get_fdata(dtype=np.float32).reshape(-1)
                    if vol_mask_idx is not None:
                        data = data[vol_mask_idx]

                feats.append(data)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                raise e

        if feats:
            # Concat features for one subject -> 1D array
            matrix_list.append(np.concatenate(feats))

    # Stack -> (n_sub, n_features)
    return np.stack(matrix_list)


# ================= 核心分析流程 =================

def zscore_1d(x: np.ndarray) -> np.ndarray:
    """对向量进行 Z-score 标准化，处理常数/NaN情况"""
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 2: return x
    mu = x[mask].mean()
    sd = x[mask].std(ddof=0)
    if sd <= 1e-9: return x  # 避免除零
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out


def build_dev_models(ages: np.ndarray, n: int, normalize: bool) -> Dict[str, np.ndarray]:
    """构建三种发育模型的上三角向量"""
    iu, ju = np.triu_indices(n, k=1)
    a = ages.reshape(-1)
    amax = np.nanmax(a)

    ai, aj = a[iu], a[ju]

    models = {}
    # 1. NN (Nearest Neighbor): 相似度随年龄差增大而减小
    raw = amax - np.abs(ai - aj)
    models["M_nn"] = zscore_1d(raw) if normalize else raw

    # 2. Conv (Convergence): 相似度随最小年龄增大而增大 (都大才像)
    raw = np.minimum(ai, aj)
    models["M_conv"] = zscore_1d(raw) if normalize else raw

    # 3. Div (Divergence): 相似度随平均年龄增大而减小 (越大约不一样)
    raw = amax - 0.5 * (ai + aj)
    models["M_div"] = zscore_1d(raw) if normalize else raw

    return models


def run_analysis(preset: JointPreset):
    print(f"\n{'=' * 60}")
    print(f"开始分析 Preset: {preset.name}")
    print(f"Task: {preset.task}, Condition: {preset.condition_group}")
    print(f"{'=' * 60}\n")

    preset.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载年龄表
    print("Loading subject info...")
    age_map = load_subject_ages_map(preset.subject_info)

    # 2. 读取并预过滤所有 aligned CSV
    dfs = []
    print("Loading aligned CSVs...")
    for spec in preset.specs:
        if not spec.aligned_csv.exists():
            raise FileNotFoundError(f"Aligned CSV not found: {spec.aligned_csv}")
        df = pd.read_csv(spec.aligned_csv)
        df = filter_dataframe(df, preset.task, preset.condition_group)
        if df.empty:
            raise ValueError(f"{spec.name} data is empty after filtering!")
        dfs.append(df)

    # 3. 确定统一被试列表 (关键步骤)
    subjects = determine_subject_list(preset, age_map, dfs)

    if len(subjects) < 10:
        print("❌ 有效被试过少 (<10)，终止分析。")
        return

    # 保存本次分析使用的被试列表
    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(preset.out_dir / "subjects_common_age_sorted.csv",
                                                            index=False)

    # 4. 构建相似性矩阵向量 (Z-scored)
    sim_vectors = []
    sim_names = []

    # 确定 lss_root (如果未指定，推测为 aligned_csv 的上两级)
    lss_root = preset.lss_root if preset.lss_root else preset.specs[0].aligned_csv.parent.parent

    # 三角索引
    n_subs = len(subjects)
    iu, ju = np.triu_indices(n_subs, k=1)

    print("\n>>> 开始构建相似性矩阵...")
    for spec, df in zip(preset.specs, dfs):
        print(f"\n--- Processing {spec.name} [{spec.space}] ---")
        try:
            data_mat = load_beta_data(spec, df, subjects, lss_root, preset)
        except Exception as e:
            print(f"⚠️ {spec.name} 数据加载失败: {e}")
            continue

        if data_mat is None:
            continue

        # 计算相关矩阵 (Subjects x Subjects)
        print(f"  [Corr] Calculating correlation matrix ({n_subs}x{n_subs})...")
        corr_mat = np.corrcoef(data_mat)

        # 保存矩阵 (可选)
        if preset.save_similarity:
            out_p = preset.out_dir / f"similarity_{spec.name}_AgeSorted.csv"
            pd.DataFrame(corr_mat, index=subjects, columns=subjects).to_csv(out_p)
            print(f"  [Save] Matrix saved to {out_p.name}")

        # 提取上三角并 Z-score
        vec = corr_mat[iu, ju]
        vec_z = zscore_1d(vec)

        if np.isnan(vec_z).any():
            print(f"⚠️ {spec.name} 矩阵包含 NaN，跳过。")
        else:
            sim_vectors.append(vec_z)
            sim_names.append(spec.name)

        # 内存回收
        del data_mat, corr_mat, vec, vec_z
        gc.collect()

    if not sim_vectors:
        print("❌ 无有效相似性矩阵生成，终止。")
        return

    S_z = np.stack(sim_vectors)  # shape: (n_mats, n_pairs)
    n_mats, n_pairs = S_z.shape

    # 5. 构建发育模型向量
    dev_models = build_dev_models(ages, n_subs, preset.normalize_models)
    model_keys = ["M_nn", "M_conv", "M_div"]

    # 6. 置换检验 (单侧, FWER)
    print(f"\n>>> 开始置换检验 (N={preset.n_perm}, 单侧-正效应)...")

    # --- 计算观测值 r_obs ---
    # r_obs_dict[model] -> array of r for each matrix
    r_obs_dict = {}
    all_r_obs_flat = []

    for m in model_keys:
        m_vec = dev_models[m]
        # Pearson r approx: dot(z1, z2) / n
        r_vals = (S_z @ m_vec) / n_pairs
        r_obs_dict[m] = r_vals
        all_r_obs_flat.extend(r_vals)

    all_r_obs_flat = np.array(all_r_obs_flat, dtype=np.float32)
    n_total_tests = len(all_r_obs_flat)

    count_raw = np.zeros(n_total_tests, dtype=np.int64)
    count_fwer = np.zeros(n_total_tests, dtype=np.int64)
    rng = np.random.default_rng(preset.seed)

    # --- Permutation Loop ---
    for i in range(preset.n_perm):
        if (i + 1) % 500 == 0:
            print(f"  Permutation {i + 1} / {preset.n_perm} ...")

        # 1. 置换年龄
        perm_idx = rng.permutation(n_subs)
        ages_perm = ages[perm_idx]

        # 2. 重建模型向量
        models_perm = build_dev_models(ages_perm, n_subs, preset.normalize_models)

        # 3. 计算所有 matrix * model 组合的 r
        current_perm_rs = []
        for m in model_keys:
            m_vec_p = models_perm[m]
            r_p = (S_z @ m_vec_p) / n_pairs
            current_perm_rs.append(r_p)

        # 展平为 1D 数组，顺序必须与 all_r_obs_flat 一致
        flat_perm_rs = np.concatenate(current_perm_rs)

        # 4. Raw P (One-sided: Positive Effect)
        # 计数规则: r_perm >= r_obs
        count_raw += (flat_perm_rs >= all_r_obs_flat)

        # 5. FWER (Max Statistic for One-sided)
        # 找出当前置换中所有检验里的最大值 (algebraic max, not abs)
        max_stat = flat_perm_rs.max()
        # 如果随机数据的最大值 >= 观测值，则该观测值的 FWER 计数 +1
        count_fwer += (max_stat >= all_r_obs_flat)

    # 计算 P 值
    p_raw = (count_raw + 1.0) / (preset.n_perm + 1.0)
    p_fwer = (count_fwer + 1.0) / (preset.n_perm + 1.0)

    # 7. 整理结果
    results = []
    idx = 0
    # 顺序必须与 all_r_obs_flat 构建顺序一致 (model -> matrix)
    for m in model_keys:
        rs = r_obs_dict[m]
        for mat_i, mat_name in enumerate(sim_names):
            results.append({
                "matrix": mat_name,
                "model": m,
                "r_obs": float(rs[mat_i]),
                "p_perm_one_tailed": float(p_raw[idx]),
                "p_fwer_one_tailed": float(p_fwer[idx]),
                "n_subjects": n_subs,
                "n_pairs": n_pairs,
                "n_perm": preset.n_perm,
                "threshold": preset.min_coverage
            })
            idx += 1

    res_df = pd.DataFrame(results)
    out_csv = preset.out_dir / "joint_analysis_dev_models_perm_fwer_onesided.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ 分析完成。结果已保存: {out_csv}")


def choose_preset(name: str, presets: Dict[str, JointPreset]) -> JointPreset:
    if name in presets: return presets[name]
    print(f"Error: Preset '{name}' not found.")
    print("Available presets:")
    for k in sorted(presets.keys()): print(f" - {k}")
    raise ValueError(f"Preset {name} not found.")


def main():
    preset = choose_preset(ACTIVE_PRESET, PRESETS)
    run_analysis(preset)


if __name__ == "__main__":
    main()