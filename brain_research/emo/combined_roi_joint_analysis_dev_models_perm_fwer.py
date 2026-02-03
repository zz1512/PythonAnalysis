#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_roi_joint_analysis_dev_models_perm_fwer.py [被试列表构建修复版]

功能：
1. 读取 aligned CSV (beta files)，构建被试间相似性矩阵 (ISC)。
2. 支持 Surface（左右半球）和 Volume（皮下/Mask）数据读取。
3. 与发育模型 (NN, Conv, Div) 进行联合分析。
4. 统计检验：5000次置换 + 单侧检验(正效应) + Model-wise FWER 校正。
5. 可选：置换多元回归（MRM/MRQAP 风格）。

核心修复 (Referencing joint_analysis_similarity_dev_models_perm_fwer):
- **强制对齐**: 不再动态计算交集，而是强制读取 `subject_order_surface_L.csv` 作为被试基准。
  这确保了本脚本生成的矩阵与 `calc_isc_combined_strict.py` 生成的矩阵在被试顺序上严格一致。
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

# 默认寻找被试列表的路径 (与 calc_isc 脚本输出一致)
DEFAULT_SUBJECT_ORDER_DIR = Path(
    "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")


# ================= 数据结构定义 =================

@dataclass(frozen=True)
class SimilaritySpec:
    name: str
    aligned_csv: Path
    space: str  # "surface_L", "surface_R", "volume"
    kind: str  # 仅支持 "combined"
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
    # 强制指定被试列表文件 (如果为None，则自动去 DEFAULT_SUBJECT_ORDER_DIR 找 subject_order_surface_L.csv)
    subject_order_path: Optional[Path] = None
    analysis_mode: str = "corr"  # "corr" | "mrm" | "both"


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
        ),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/joint_perm_fwer_out"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="",
        condition_group="all",
        n_perm=5000,
        analysis_mode="both",
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
    """读取 CSV 中的 subject 列 (兼容多种列名)"""
    if not order_path.exists():
        raise FileNotFoundError(f"未找到顺序文件: {order_path}")

    # 允许空格/不同分隔符
    try:
        df = pd.read_csv(order_path)
    except:
        df = pd.read_csv(order_path, sep=None, engine='python')

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


def determine_subject_list_strict(
        preset: JointPreset,
        age_map: Dict[str, float]
) -> List[str]:
    """
    【修复核心】严格按照参考文件确定被试列表。
    不再计算交集，而是读取参考文件，并验证年龄是否存在。
    """
    # 1. 确定参考文件路径
    target_path = preset.subject_order_path

    # 如果没手动指定，去默认目录找 subject_order_surface_L.csv
    if not target_path:
        default_file = DEFAULT_SUBJECT_ORDER_DIR / "subject_order_surface_L.csv"
        if default_file.exists():
            target_path = default_file
        else:
            # 尝试找 R
            default_file_r = DEFAULT_SUBJECT_ORDER_DIR / "subject_order_surface_R.csv"
            if default_file_r.exists():
                target_path = default_file_r

    if not target_path or not target_path.exists():
        raise FileNotFoundError(
            f"❌ 无法确定被试顺序！\n"
            f"请在 Preset 中指定 subject_order_path，或者确保以下文件存在：\n"
            f"{DEFAULT_SUBJECT_ORDER_DIR}/subject_order_surface_L.csv"
        )

    print(f"  [Info] 使用参考被试列表: {target_path}")

    # 2. 读取并验证
    raw_subjects = load_subject_order_list(target_path)

    valid_subjects = []
    for s in raw_subjects:
        if s in age_map and np.isfinite(age_map[s]):
            valid_subjects.append(s)
        else:
            print(f"  [Warn] 被试 {s} 在年龄表中缺失或无效，已跳过。")

    # 注意：不再根据年龄再次排序，而是信任 reference file 的顺序
    # (calc_isc 已经排过序了，再次排序可能因精度问题导致微小差异)

    print(f"  [Result] 最终被试数量: {len(valid_subjects)}")
    if valid_subjects:
        print(f"           First: {valid_subjects[0]}, Last: {valid_subjects[-1]}")

    return valid_subjects


def load_beta_data(
        spec: SimilaritySpec,
        df: pd.DataFrame,
        subjects: List[str],  # 这是严格的参考顺序
        lss_root: Path,
        preset: JointPreset,
) -> Optional[np.ndarray]:
    """
    加载数据并展平，返回 (n_sub, n_features)。

    【修改点】：
    - 严格依赖传入的 `subjects` 列表顺序。
    - 如果某个被试的全部/关键数据缺失，根据逻辑报错或填充（但 ISC 计算通常不允许填充）。
    - 这里的逻辑是：只保留所有被试都存在的 Stimulus (Ensure Complete Keys)，然后提取。
    """

    # --- 1. 确定公共刺激 ---
    # 先只看在列表里的被试
    df_sub = df[df["subject"].isin(subjects)].copy()

    if df_sub.empty:
        print(f"  [Skip] {spec.name}: 指定被试在数据表中无记录。")
        return None

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
    # 核心策略：丢刺激，保被试
    if preset.ensure_complete_keys:
        valid_keys = []
        for k in keys:
            # 检查是否 *所有* 指定的 subjects 都有这个 key
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
            print(f"  [Data] {spec.name}: 为保被试完整，剔除 {dropped} 个有缺失的刺激。剩余: {len(valid_keys)}")
        keys = valid_keys

    if not keys:
        print(f"  [Skip] {spec.name}: 剔除缺失刺激后，无剩余刺激。")
        return None

    # --- 4. 准备 Mask (Volume 或 ROI) ---
    vol_mask_idx = None  # for Volume
    mask_img_vol = None  # for Volume Resampling

    if spec.kind != "combined":
        raise ValueError(f"{spec.name}: 当前脚本只支持 kind='combined'（ROI/parcel 请使用 parcel_joint_analysis_* 脚本）")

    if spec.space == "volume":
        if not spec.volume_mask:
            raise ValueError(f"{spec.name}: Volume 模式需要 volume_mask")
        mask_img_vol = image.load_img(str(spec.volume_mask))
        mask_data = mask_img_vol.get_fdata() > 0
        vol_mask_idx = np.where(mask_data.reshape(-1))[0]
        print(f"  [Mask] Volume mask loaded. Voxels: {len(vol_mask_idx)}")

    # --- 5. 加载数据 ---
    matrix_list = []
    print(f"  [Load] Loading data for {spec.name} ({len(keys)} stimuli x {len(subjects)} subjects)...")

    # 双重检查：确保没有漏网之鱼
    for s in subjects:
        for k in keys:
            if not (lookup.get((s, k)) and lookup[(s, k)].exists()):
                raise FileNotFoundError(f"严重错误：被试 {s} 缺失刺激 {k}，这不应该发生(已通过keys过滤)。")

    for s in tqdm(subjects, leave=False, desc=f"Reading {spec.name}"):
        feats = []
        for k in keys:
            p = lookup.get((s, k))
            try:
                if spec.space.startswith("surface"):
                    # Surface Load
                    g = nib.load(str(p))
                    data = g.darrays[0].data.astype(np.float32).reshape(-1)
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


def fit_mrm_betas(
        S_z: np.ndarray,
        model_vecs: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(S_z, dtype=np.float64).T
    X = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in model_vecs], axis=1)
    n = int(Y.shape[0])
    Xd = np.concatenate([np.ones((n, 1), dtype=np.float64), X], axis=1)
    XtX = Xd.T @ Xd
    XtY = Xd.T @ Y
    XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
    coef = XtX_inv @ XtY
    betas = coef[1:, :].astype(np.float32, copy=False)

    y_sum = Y.sum(axis=0)
    y_ty = np.sum(Y * Y, axis=0)
    sst = y_ty - (y_sum * y_sum) / float(n)
    ssr = np.sum(coef * XtY, axis=0)
    sse = y_ty - ssr
    r2 = np.full((Y.shape[1],), np.nan, dtype=np.float32)
    ok = sst > 0
    r2[ok] = (1.0 - (sse[ok] / sst[ok])).astype(np.float32, copy=False)
    return betas, r2


def run_analysis(preset: JointPreset):
    print(f"\n{'=' * 60}")
    print(f"开始分析 Preset: {preset.name}")
    print(f"Task: {preset.task}, Condition: {preset.condition_group}")
    print(f"{'=' * 60}\n")

    preset.out_dir.mkdir(parents=True, exist_ok=True)
    ctx_parts = []
    if preset.task:
        ctx_parts.append(str(preset.task).strip())
    if preset.condition_group:
        ctx_parts.append(str(preset.condition_group).strip())
    ctx = "__".join([p for p in ctx_parts if p]) or "all"

    # 1. 加载年龄表
    print("Loading subject info...")
    age_map = load_subject_ages_map(preset.subject_info)

    # 2. 读取 aligned CSV
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

    # 3. 确定统一被试列表 (【修复】使用严格对齐逻辑)
    # 不再计算交集，而是直接加载标准列表，如果数据对不上就报错或剔除刺激
    subjects = determine_subject_list_strict(preset, age_map)

    if len(subjects) < 10:
        print("❌ 有效被试过少 (<10)，终止分析。")
        return

    # 保存本次分析使用的被试列表 (这应该与 subject_order_surface_L.csv 一致)
    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(preset.out_dir / f"subjects_common_age_sorted__{ctx}.csv",
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
            # 这里的 subjects 是已经排好序的标准列表
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
            out_p = preset.out_dir / f"similarity_{spec.name}_AgeSorted__{ctx}.csv"
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

    analysis_mode = str(preset.analysis_mode or "corr").strip().lower()

    # ... (后续统计部分逻辑保持一致，重点是上面数据的对齐) ...
    # 以下为统计检验部分，直接复用原逻辑，但加上 Model-wise FWER 注释

    run_corr = analysis_mode in {"corr", "both"}
    run_mrm = analysis_mode in {"mrm", "both"}

    r_obs_dict = {}
    r_obs_flat = None
    count_raw_corr = None
    count_fwer_corr = None

    beta_obs = None
    r2_obs = None
    beta_obs_flat = None
    count_raw_mrm = None
    count_fwer_mrm = None

    if run_corr:
        r_list = []
        for m in model_keys:
            r_vals = (S_z @ dev_models[m]) / n_pairs
            r_obs_dict[m] = r_vals.astype(np.float32, copy=False)
            r_list.append(r_obs_dict[m])
        # Flatten: [M_nn_Mat1, M_nn_Mat2, ..., M_conv_Mat1, ...]
        r_obs_flat = np.concatenate(r_list).astype(np.float32, copy=False)
        count_raw_corr = np.zeros(int(r_obs_flat.size), dtype=np.int64)
        count_fwer_corr = np.zeros(int(r_obs_flat.size), dtype=np.int64)

    if run_mrm:
        model_vecs_obs = [dev_models[m] for m in model_keys]
        beta_obs, r2_obs = fit_mrm_betas(S_z, model_vecs_obs)
        beta_obs_flat = np.concatenate([beta_obs[i, :] for i in range(beta_obs.shape[0])]).astype(np.float32,
                                                                                                  copy=False)
        count_raw_mrm = np.zeros(int(beta_obs_flat.size), dtype=np.int64)
        count_fwer_mrm = np.zeros(int(beta_obs_flat.size), dtype=np.int64)

    print(f"\n>>> 开始置换检验 (N={preset.n_perm}, 单侧-正效应, Model-wise FWER)...")
    rng = np.random.default_rng(preset.seed)

    for i in range(int(preset.n_perm)):
        if (i + 1) % 500 == 0:
            print(f"  Permutation {i + 1} / {preset.n_perm} ...")

        perm_idx = rng.permutation(n_subs)
        ages_perm = ages[perm_idx]
        models_perm = build_dev_models(ages_perm, n_subs, preset.normalize_models)

        if run_corr:
            # 1. Calculate Permuted R for all Models x Matrices
            current_perm_rs = []
            for m in model_keys:
                r_p = (S_z @ models_perm[m]) / n_pairs
                current_perm_rs.append(r_p.astype(np.float32, copy=False))
            flat_perm_rs = np.concatenate(current_perm_rs)

            # 2. Raw Count
            count_raw_corr += (flat_perm_rs >= r_obs_flat)

            # 3. Model-wise Max Statistic
            # flat_perm_rs 结构: [Model1(8), Model2(8), Model3(8)]
            # 我们需要对每个模型内部取 Max
            offset = 0
            for _ in model_keys:
                start = offset
                end = offset + n_mats

                # 获取该模型下的所有矩阵的 permuted r
                subset_r = flat_perm_rs[start:end]
                # 获取该模型下的所有矩阵的 observed r
                subset_obs = r_obs_flat[start:end]

                # 计算该模型内部的最大统计量
                max_stat = float(subset_r.max(initial=-1.0))

                # 更新 FWER 计数
                count_fwer_corr[start:end] += (max_stat >= subset_obs)

                offset += n_mats

        if run_mrm:
            # MRM 逻辑同理
            model_vecs_p = [models_perm[m] for m in model_keys]
            beta_p, _ = fit_mrm_betas(S_z, model_vecs_p)
            # Flatten: [Pred1_Mat1... Pred1_MatN, Pred2...]
            flat_beta_p = np.concatenate([beta_p[i, :] for i in range(beta_p.shape[0])]).astype(np.float32, copy=False)

            count_raw_mrm += (flat_beta_p >= beta_obs_flat)

            offset = 0
            for _ in model_keys:
                start = offset
                end = offset + n_mats
                subset_beta = flat_beta_p[start:end]
                subset_obs = beta_obs_flat[start:end]
                max_stat = float(subset_beta.max(initial=-1.0))
                count_fwer_mrm[start:end] += (max_stat >= subset_obs)
                offset += n_mats

    if run_corr:
        p_raw = (count_raw_corr + 1.0) / (preset.n_perm + 1.0)
        p_fwer = (count_fwer_corr + 1.0) / (preset.n_perm + 1.0)
        results = []
        idx = 0
        for m in model_keys:
            rs = r_obs_dict[m]
            for mat_i, mat_name in enumerate(sim_names):
                results.append(
                    {
                        "matrix": mat_name,
                        "model": m,
                        "r_obs": float(rs[mat_i]),
                        "p_perm_one_tailed": float(p_raw[idx]),
                        "p_fwer_model_wise": float(p_fwer[idx]),
                        "n_subjects": int(n_subs),
                        "n_pairs": int(n_pairs),
                        "n_perm": int(preset.n_perm),
                        "seed": int(preset.seed),
                        "threshold": float(preset.min_coverage),
                        "normalize_models": bool(preset.normalize_models),
                    }
                )
                idx += 1
        res_df = pd.DataFrame(results)
        out_csv = preset.out_dir / f"joint_analysis_dev_models_perm_fwer_onesided__{ctx}.csv"
        res_df.to_csv(out_csv, index=False)
        print(f"\n✅ 相关分析结果已保存: {out_csv}")

    if run_mrm:
        p_raw_b = (count_raw_mrm + 1.0) / (preset.n_perm + 1.0)
        p_fwer_b = (count_fwer_mrm + 1.0) / (preset.n_perm + 1.0)
        rows = []
        idx = 0
        for pred_i, pred_name in enumerate(model_keys):
            b = beta_obs[pred_i, :].astype(np.float32, copy=False)
            for mat_i, mat_name in enumerate(sim_names):
                rows.append(
                    {
                        "matrix": mat_name,
                        "predictor": pred_name,
                        "beta_obs": float(b[mat_i]),
                        "r2_obs": float(r2_obs[mat_i]),
                        "p_perm_one_tailed": float(p_raw_b[idx]),
                        "p_fwer_model_wise": float(p_fwer_b[idx]),
                        "tail": "greater",
                        "n_subjects": int(n_subs),
                        "n_pairs": int(n_pairs),
                        "n_perm": int(preset.n_perm),
                        "seed": int(preset.seed),
                        "threshold": float(preset.min_coverage),
                        "normalize_models": bool(preset.normalize_models),
                    }
                )
                idx += 1
        out_mrm = preset.out_dir / f"joint_analysis_dev_models_mrm_perm_fwer_onesided__{ctx}.csv"
        pd.DataFrame(rows).to_csv(out_mrm, index=False)
        print(f"\n✅ 多元回归(MRM/MRQAP)结果已保存: {out_mrm}")


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