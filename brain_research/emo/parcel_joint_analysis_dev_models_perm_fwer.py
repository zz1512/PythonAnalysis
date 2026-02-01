#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parcel_joint_analysis_dev_models_perm_fwer.py

目的
-
对 Schaefer200（编号 1-200）的“每个 parcel（脑区编号）”分别做：
1) 被试×被试相似性矩阵（基于 beta-series 的 parcel 平均特征）
2) 与 3 个发育模型（M_nn/M_conv/M_div）的相关
3) 置换检验（按年龄向量置换）与 max-stat FWER 校正

输入
-
- LSS 输出的 aligned 索引：lss_index_*_aligned.csv
  需要列：subject, task, run, file, stimulus_content
  若索引里有 condition_group，则直接用；否则用 raw_condition/raw_emotion 推断三条件。
- Atlas：
  - surface：Schaefer label.gii（单半球）
  - volume：Schaefer 体素图谱 label.nii / label.nii.gz（与 beta 同一 MNI 空间；必要时脚本会做 nearest resample）
- 年龄表：含 sub_id/age（或 被试编号/采集年龄）

输出
-
- parcel 级结果长表（每个 parcel_id × 每个 model 一行）：
  parcel_joint_analysis_dev_models_perm_fwer__{space}__{task}__{condition_group}.csv
  字段：parcel_id, model, r_obs, p_perm, p_fwer, n_subjects, n_pairs, ...
- subjects 与 stimuli 的审计文件（用于复现实验纳入集合）

使用方式（覆盖“1-200 编号”）
-
Schaefer200 常见做法是 L.label.gii + R.label.gii 分开存：
你需要分别对左/右半球各跑一次，合起来就是 1-200 的全部编号级别结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"


@dataclass(frozen=True)
class ParcelAnalysisPreset:
    name: str
    aligned_csv: Path
    atlas_label: Path
    subject_info: Path
    out_dir: Path
    space: str
    task: str
    condition_group: str
    lss_root: Optional[Path] = None
    subject_allowlist: Optional[Path] = None
    min_coverage: float = 0.9
    ensure_complete_keys: bool = True
    require_both_tasks: bool = False
    n_perm: int = 5000
    seed: int = 42
    normalize_models: bool = True


ACTIVE_ANALYSIS_PRESET = "EXAMPLE_surface_L_EMO_all"


PRESETS_ANALYSIS: Dict[str, ParcelAnalysisPreset] = {
    "EXAMPLE_surface_L_EMO_all": ParcelAnalysisPreset(
        name="EXAMPLE_surface_L_EMO_all",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
        atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out"),
        space="surface_L",
        task="EMO",
        condition_group="all",
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        min_coverage=0.9,
        require_both_tasks=False,
        n_perm=5000,
        seed=42,
        normalize_models=True,
    ),
    "EXAMPLE_surface_R_EMO_all": ParcelAnalysisPreset(
        name="EXAMPLE_surface_R_EMO_all",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
        atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out"),
        space="surface_R",
        task="EMO",
        condition_group="all",
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        min_coverage=0.9,
        require_both_tasks=False,
        n_perm=5000,
        seed=42,
        normalize_models=True,
    ),
    "EXAMPLE_volume_EMO_all": ParcelAnalysisPreset(
        name="EXAMPLE_volume_EMO_all",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"),
        atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_MNI152_label.nii.gz"),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/parcel_perm_fwer_out"),
        space="volume",
        task="EMO",
        condition_group="all",
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        min_coverage=0.9,
        require_both_tasks=False,
        n_perm=5000,
        seed=42,
        normalize_models=True,
    ),
}


def choose_preset(name: str, presets: Dict[str, ParcelAnalysisPreset]) -> ParcelAnalysisPreset:
    if name in presets:
        return presets[name]
    keys = sorted(presets.keys())
    if not keys:
        raise ValueError("未配置任何 preset。")
    print("可用 presets：")
    for i, k in enumerate(keys, start=1):
        print(f"  [{i}] {k}")
    s = input(f"请输入编号 (1-{len(keys)}): ").strip()
    idx = int(s) - 1
    if idx < 0 or idx >= len(keys):
        raise ValueError("编号超出范围。")
    return presets[keys[idx]]


@dataclass(frozen=True)
class Context:
    space: str
    task: str
    condition_group: str

    @property
    def name(self) -> str:
        parts = [self.space]
        if self.task:
            parts.append(self.task)
        if self.condition_group:
            parts.append(self.condition_group)
        return "__".join([p for p in parts if p])


def _parse_age_exact(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return float("nan")
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
    years = int(y.group(1)) if y else 0
    months = int(m.group(1)) if m else 0
    days = int(d.group(1)) if d else 0
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

    if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError(
            "年龄表缺少必要列。需要以下任意一组列名：\n"
            f"- {COL_SUB_ID}, {COL_AGE}\n"
            f"- {LEGACY_COL_SUB_ID}, {LEGACY_COL_AGE}\n"
            f"实际列名: {list(info_df.columns)}"
        )

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        age_map[sub_id] = _parse_age_exact(row[age_col])
    return age_map


def infer_condition_group(row: pd.Series) -> str:
    # 兼容策略：
    # 1) aligned.csv 若已有 condition_group 列，直接使用
    # 2) 否则根据 raw_condition/raw_emotion 推断三条件
    cg = str(row.get("condition_group", "")).strip()
    if cg:
        return cg
    raw_cond = str(row.get("raw_condition", "")).strip()
    raw_emo = str(row.get("raw_emotion", "")).strip()
    if raw_cond == "Reappraisal":
        return "emotion_regulation"
    if raw_cond == "PassiveLook":
        if raw_emo.lower().startswith("neutral"):
            return "neutral_response"
        return "emotion_generation"
    return "other"


def load_subject_allowlist(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"subject_allowlist 不存在: {path}")
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep="\t")
        if "subject" in df.columns:
            subjects = df["subject"].astype(str).tolist()
        else:
            subjects = df.iloc[:, 0].astype(str).tolist()
    else:
        subjects = path.read_text(encoding="utf-8").splitlines()
    cleaned = set()
    for s in subjects:
        sid = str(s).strip()
        if not sid:
            continue
        cleaned.add(sid if sid.startswith("sub-") else f"sub-{sid}")
    return cleaned


def load_label_gii(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.int32)


def _is_nifti(path: Path) -> bool:
    suf = "".join(path.suffixes).lower()
    return suf.endswith(".nii") or suf.endswith(".nii.gz")


def _is_gifti(path: Path) -> bool:
    return path.suffix.lower() == ".gii"


def _resample_label_to_ref_if_needed(atlas_path: Path, ref_img: nib.spatialimages.SpatialImage) -> Tuple[Path, nib.spatialimages.SpatialImage]:
    atlas_img = image.load_img(str(atlas_path))
    if atlas_img.shape == ref_img.shape[:3] and np.allclose(atlas_img.affine, ref_img.affine):
        return atlas_path, atlas_img
    atlas_rs = image.resample_to_img(atlas_img, ref_img, interpolation="nearest")
    return atlas_path, atlas_rs


def _find_first_existing_beta(df: pd.DataFrame, lss_root: Path, subjects_sorted: Sequence[str], common_keys: Sequence[str]) -> Path:
    sub_set = set(subjects_sorted)
    key_set = set(common_keys)
    df = df[df["subject"].astype(str).isin(sub_set) & df["stimulus_content"].astype(str).isin(key_set)].copy()
    for _, row in df.iterrows():
        p = resolve_beta_path(lss_root, row)
        if p.exists():
            return p
    raise FileNotFoundError("未找到任何可用 beta 文件用于 volume atlas resample。")


def load_beta_gii(path: Path) -> Optional[np.ndarray]:
    try:
        g = nib.load(str(path))
        return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32)
    except Exception:
        return None


def load_beta_nii_flat(path: Path, voxel_idx: np.ndarray) -> Optional[np.ndarray]:
    try:
        img = image.load_img(str(path))
        data = img.get_fdata(dtype=np.float32).reshape(-1)
        return data[voxel_idx].astype(np.float32, copy=False)
    except Exception:
        return None


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    # 兼容两种路径写法：
    # - file 本身是绝对路径
    # - file 是相对文件名，需要拼出 LSS_ROOT/task/sub/(run-*)/file
    f = str(row.get("file", ""))
    if f.startswith("/"):
        return Path(f)
    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))
    p = lss_root / task / sub / f
    if p.exists():
        return p
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def zscore_1d(x: np.ndarray) -> np.ndarray:
    # 对一维向量做 zscore，NaN 保留（并要求有效样本>=10）
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if int(mask.sum()) < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[mask].mean())
    sd = float(x[mask].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out.astype(np.float32, copy=False)


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(n), k=1)


def build_model_vec(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, model: str, normalize: bool) -> np.ndarray:
    # 三个发育模型定义（对上三角向量进行建模）：
    # - M_nn: 最近邻（年龄差越小越大）
    # - M_conv: 收敛（越小年龄越大）
    # - M_div: 发散（越大年龄越大）
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    amax = float(np.nanmax(a))
    ai = a[iu]
    aj = a[ju]
    if model == "M_nn":
        v = (amax - np.abs(ai - aj)).astype(np.float32, copy=False)
    elif model == "M_conv":
        v = np.minimum(ai, aj).astype(np.float32, copy=False)
    elif model == "M_div":
        v = (amax - 0.5 * (ai + aj)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"未知模型: {model}")
    return zscore_1d(v) if normalize else v


def compute_parcel_features(
    df: pd.DataFrame,
    lss_root: Path,
    labels: np.ndarray,
    parcel_ids: np.ndarray,
    subjects_sorted: Sequence[str],
    common_keys: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    # 把每个 stimulus 的 beta map 先聚合为 parcel 平均值：
    # - labels: 每个顶点/体素所属 parcel 的编号
    # - 对每个 beta：用 bincount 做 parcel 加权求和，再除以顶点数得到均值
    # 最终得到 feats[sub, parcel, stimulus]，后续再在 stimulus 维度做相关。
    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1).astype(np.float32)
    counts[counts <= 0] = np.nan

    key_set = set(common_keys)
    df = df[df["stimulus_content"].isin(key_set)].copy()

    lookup: Dict[Tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        s = str(row.get("subject", ""))
        k = str(row.get("stimulus_content", ""))
        lookup[(s, k)] = resolve_beta_path(lss_root, row)

    n_sub = len(subjects_sorted)
    n_parcel = int(parcel_ids.size)
    n_key = len(common_keys)
    feats = np.full((n_sub, n_parcel, n_key), np.nan, dtype=np.float32)

    keep_subjects: List[str] = []
    keep_rows: List[int] = []

    for si, sub in enumerate(subjects_sorted):
        ok = True
        for ki, key in enumerate(common_keys):
            p = lookup.get((sub, key))
            if p is None or not p.exists():
                ok = False
                break
            beta = load_beta_gii(p)
            if beta is None or beta.shape[0] != labels.shape[0]:
                ok = False
                break
            sums = np.bincount(labels, weights=beta, minlength=max_label + 1).astype(np.float32)
            means = sums[parcel_ids] / counts[parcel_ids]
            if not np.isfinite(means).all():
                ok = False
                break
            feats[si, :, ki] = means
        if ok:
            keep_subjects.append(sub)
            keep_rows.append(si)

    if not keep_subjects:
        return np.empty((0, n_parcel, n_key), dtype=np.float32), []

    feats = feats[np.array(keep_rows, dtype=int), :, :]
    return feats, keep_subjects


def compute_parcel_features_volume(
    df: pd.DataFrame,
    lss_root: Path,
    labels_vec: np.ndarray,
    voxel_idx: np.ndarray,
    parcel_ids: np.ndarray,
    subjects_sorted: Sequence[str],
    common_keys: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    max_label = int(labels_vec.max())
    counts = np.bincount(labels_vec, minlength=max_label + 1).astype(np.float32)
    counts[counts <= 0] = np.nan

    key_set = set(common_keys)
    df = df[df["stimulus_content"].isin(key_set)].copy()

    lookup: Dict[Tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        s = str(row.get("subject", ""))
        k = str(row.get("stimulus_content", ""))
        lookup[(s, k)] = resolve_beta_path(lss_root, row)

    n_sub = len(subjects_sorted)
    n_parcel = int(parcel_ids.size)
    n_key = len(common_keys)
    feats = np.full((n_sub, n_parcel, n_key), np.nan, dtype=np.float32)

    keep_subjects: List[str] = []
    keep_rows: List[int] = []

    for si, sub in enumerate(subjects_sorted):
        ok = True
        for ki, key in enumerate(common_keys):
            p = lookup.get((sub, key))
            if p is None or not p.exists():
                ok = False
                break
            beta_vec = load_beta_nii_flat(p, voxel_idx)
            if beta_vec is None or beta_vec.shape[0] != labels_vec.shape[0]:
                ok = False
                break
            sums = np.bincount(labels_vec, weights=beta_vec, minlength=max_label + 1).astype(np.float32)
            means = sums[parcel_ids] / counts[parcel_ids]
            if not np.isfinite(means).all():
                ok = False
                break
            feats[si, :, ki] = means
        if ok:
            keep_subjects.append(sub)
            keep_rows.append(si)

    if not keep_subjects:
        return np.empty((0, n_parcel, n_key), dtype=np.float32), []

    feats = feats[np.array(keep_rows, dtype=int), :, :]
    return feats, keep_subjects


def parcel_similarity_vectors_z(feats: np.ndarray) -> np.ndarray:
    # 对每个 parcel，计算被试×被试相似性并取上三角向量：
    # - 先对每个被试的 stimulus 向量做 zscore（避免均值/尺度差异影响相关）
    # - 相似性矩阵 S = (Z @ Z.T) / n_key
    # - 取上三角向量后再做一次 zscore，供置换时点积计算
    n_sub, n_parcel, n_key = feats.shape
    iu, ju = triu_indices(n_sub)
    n_pairs = int(iu.size)
    sim_z = np.empty((n_parcel, n_pairs), dtype=np.float32)

    for pi in range(n_parcel):
        Y = feats[:, pi, :].astype(np.float32, copy=False)
        mu = Y.mean(axis=1, keepdims=True)
        sd = Y.std(axis=1, keepdims=True, ddof=0)
        sd[sd <= 0] = np.nan
        Z = (Y - mu) / sd
        if not np.isfinite(Z).all():
            sim_z[pi, :] = np.nan
            continue
        S = (Z @ Z.T) / float(n_key)
        v = S[iu, ju].astype(np.float32, copy=False)
        sim_z[pi, :] = zscore_1d(v)
    return sim_z


def filter_dataframe(
    df: pd.DataFrame,
    task: str,
    condition_group: str,
    require_both_tasks: bool,
    subject_allowlist: Optional[set[str]] = None,
) -> pd.DataFrame:
    # 对齐索引过滤：
    # - task: EMO 或 SOC（为空表示不按 task 过滤）
    # - condition_group: 三条件（all 表示不过滤）
    # - require_both_tasks: 是否强制 EMO∩SOC 交集（做联合任务分析时用）
    df = df.dropna(subset=["subject", "stimulus_content", "file", "task"]).copy()
    df["subject"] = df["subject"].astype(str)
    df["task"] = df["task"].astype(str)
    df["stimulus_content"] = df["stimulus_content"].astype(str)
    df["condition_group"] = df.apply(infer_condition_group, axis=1)

    if task:
        df = df[df["task"] == task].copy()

    if condition_group and condition_group != "all":
        df = df[df["condition_group"] == condition_group].copy()

    if require_both_tasks:
        subs_a = set(df[df["task"] == "EMO"]["subject"].unique().tolist())
        subs_b = set(df[df["task"] == "SOC"]["subject"].unique().tolist())
        keep = subs_a & subs_b
        df = df[df["subject"].isin(sorted(keep))].copy()

    if subject_allowlist is not None:
        df = df[df["subject"].isin(subject_allowlist)].copy()

    return df


def common_subjects_age_sorted(df: pd.DataFrame, age_map: Dict[str, float]) -> List[str]:
    # 按年龄从小到大排序，缺失年龄放最后（但 run() 会拒绝进入最终样本）
    subs = sorted(df["subject"].unique().astype(str).tolist())

    def sort_key(sid: str) -> Tuple[float, str]:
        a = age_map.get(sid, float("nan"))
        if not np.isfinite(a):
            return (999.0, sid)
        return (float(a), sid)

    return sorted(subs, key=sort_key)


def common_stimulus_keys(df: pd.DataFrame, subjects: Sequence[str], min_coverage: float) -> List[str]:
    # “公共刺激”定义：stimulus_content 在被试中出现次数 >= min_coverage * N
    df = df[df["subject"].isin(set(subjects))].copy()
    counts = df["stimulus_content"].value_counts()
    thr = int(np.floor(float(min_coverage) * len(subjects)))
    thr = max(thr, 1)
    keys = counts[counts >= thr].index.astype(str).tolist()
    keys.sort()
    return keys


def filter_keys_complete(
    df: pd.DataFrame,
    lss_root: Path,
    subjects: Sequence[str],
    keys: Sequence[str],
) -> List[str]:
    key_set = set(keys)
    sub_set = set(subjects)
    df = df[df["subject"].astype(str).isin(sub_set) & df["stimulus_content"].astype(str).isin(key_set)].copy()
    lookup: Dict[Tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        s = str(row.get("subject", ""))
        k = str(row.get("stimulus_content", ""))
        lookup[(s, k)] = resolve_beta_path(lss_root, row)
    keep_keys = []
    for key in keys:
        ok = True
        for sub in subjects:
            fpath = lookup.get((sub, key))
            if fpath is None or not fpath.exists():
                ok = False
                break
        if ok:
            keep_keys.append(key)
    return keep_keys


def run(
    aligned_csv: Path,
    atlas_label: Path,
    subject_info: Path,
    out_dir: Path,
    space: str,
    task: str,
    condition_group: str,
    lss_root: Optional[Path],
    subject_allowlist: Optional[Path],
    min_coverage: float,
    ensure_complete_keys: bool,
    require_both_tasks: bool,
    n_perm: int,
    seed: int,
    normalize_models: bool,
) -> Path:
    # 主流程：
    # 1) 读 aligned 索引并按 task/condition 过滤
    # 2) 按年龄排序被试
    # 3) 选覆盖率足够的公共 stimulus_content
    # 4) 逐 stimulus 加载 beta 并聚合为 parcel 均值特征
    # 5) 对每个 parcel 计算被试×被试相似性上三角向量（zscore）
    # 6) 与 3 个模型向量点积得到 r_obs
    # 7) 置换：打乱年龄顺序生成模型向量，得到 p_perm（单侧，rp>=r_obs）
    # 8) max-stat：每次置换取所有 parcel 的最大 rp，得到 p_fwer（FWER 单侧）
    df_full = pd.read_csv(aligned_csv)
    allowlist = load_subject_allowlist(subject_allowlist)
    df = filter_dataframe(
        df_full,
        task=task,
        condition_group=condition_group,
        require_both_tasks=require_both_tasks,
        subject_allowlist=allowlist,
    )
    if df.empty:
        raise ValueError("过滤后 aligned 数据为空，请检查 task/condition_group 设置。")

    age_map = load_subject_ages_map(subject_info)
    subjects_sorted = common_subjects_age_sorted(df, age_map)
    if len(subjects_sorted) < 10:
        raise ValueError(f"有效被试数过少: {len(subjects_sorted)}")

    root = lss_root if lss_root is not None else aligned_csv.parent
    keys = common_stimulus_keys(df, subjects_sorted, min_coverage=min_coverage)
    if not keys:
        raise ValueError("没有满足覆盖率阈值的公共 stimulus_content。")
    if ensure_complete_keys:
        keys = filter_keys_complete(df, root, subjects_sorted, keys)
        if not keys:
            raise ValueError("剔除缺失刺激后无可用公共 stimulus_content。")
    atlas_label_used = atlas_label

    if _is_gifti(atlas_label):
        labels = load_label_gii(atlas_label)
        parcel_ids = np.unique(labels)
        parcel_ids = parcel_ids[(parcel_ids > 0)]
        parcel_ids = np.sort(parcel_ids.astype(np.int32))
        feats, subjects_kept = compute_parcel_features(
            df=df,
            lss_root=root,
            labels=labels,
            parcel_ids=parcel_ids,
            subjects_sorted=subjects_sorted,
            common_keys=keys,
        )
    elif _is_nifti(atlas_label):
        ref_beta = _find_first_existing_beta(df, root, subjects_sorted, keys)
        ref_img = image.load_img(str(ref_beta))
        _, atlas_img_rs = _resample_label_to_ref_if_needed(atlas_label, ref_img)

        ctx_tmp = Context(space=space, task=task, condition_group=condition_group).name
        out_dir.mkdir(parents=True, exist_ok=True)
        atlas_label_used = out_dir / f"atlas_resampled__{ctx_tmp}.nii.gz"
        atlas_img_rs.to_filename(str(atlas_label_used))

        lab = atlas_img_rs.get_fdata(dtype=np.float32)
        lab = np.asarray(lab, dtype=np.int32)
        lab_flat = lab.reshape(-1)
        mask = lab_flat > 0
        voxel_idx = np.where(mask)[0].astype(np.int64)
        labels_vec = lab_flat[mask].astype(np.int32, copy=False)
        parcel_ids = np.unique(labels_vec)
        parcel_ids = np.sort(parcel_ids.astype(np.int32))

        feats, subjects_kept = compute_parcel_features_volume(
            df=df,
            lss_root=root,
            labels_vec=labels_vec,
            voxel_idx=voxel_idx,
            parcel_ids=parcel_ids,
            subjects_sorted=subjects_sorted,
            common_keys=keys,
        )
    else:
        raise ValueError(f"不支持的 atlas 格式: {atlas_label}")

    if feats.shape[0] < 10:
        raise ValueError(f"成功加载数据的被试过少: {feats.shape[0]}")

    subjects = subjects_kept
    ages = np.array([age_map.get(s, np.nan) for s in subjects], dtype=np.float32)
    if not np.isfinite(ages).all():
        raise ValueError("存在缺失年龄的被试（进入了最终 subjects 列表），请先修正年龄表或过滤。")

    sim_z = parcel_similarity_vectors_z(feats)
    if not np.isfinite(sim_z).all():
        raise ValueError("相似性向量包含 NaN，通常意味着某些 parcel/刺激存在空顶点或数值异常。")

    n_sub = int(len(subjects))
    iu, ju = triu_indices(n_sub)
    n_pairs = int(iu.size)

    models = ["M_nn", "M_conv", "M_div"]
    mvec_obs = {m: build_model_vec(ages, iu, ju, model=m, normalize=normalize_models) for m in models}
    for m in models:
        if not np.isfinite(mvec_obs[m]).all():
            raise ValueError(f"模型向量 {m} 包含 NaN（可能年龄常数/过少）。")

    r_obs = {m: (sim_z @ mvec_obs[m]) / float(n_pairs) for m in models}

    rng = np.random.default_rng(int(seed))
    ge_count = {m: np.zeros(sim_z.shape[0], dtype=np.int32) for m in models}
    max_ge_count = {m: np.zeros(sim_z.shape[0], dtype=np.int32) for m in models}

    for _ in range(int(n_perm)):
        perm = rng.permutation(n_sub)
        ages_p = ages[perm]
        for m in models:
            mv = build_model_vec(ages_p, iu, ju, model=m, normalize=normalize_models)
            rp = (sim_z @ mv) / float(n_pairs)
            # 单侧检验：只关心“显著正相关”（rp >= r_obs）
            ge_count[m] += (rp >= r_obs[m]).astype(np.int32)
            # max-stat：同一置换下，所有 parcel 的最大统计量用于 FWER
            max_rp = float(np.max(rp))
            max_ge_count[m] += (max_rp >= r_obs[m]).astype(np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(space=space, task=task, condition_group=condition_group).name

    rows = []
    for m in models:
        p_perm = (ge_count[m].astype(np.float64) + 1.0) / (float(n_perm) + 1.0)
        p_fwer = (max_ge_count[m].astype(np.float64) + 1.0) / (float(n_perm) + 1.0)
        for pid, r, pp, pf in zip(parcel_ids.tolist(), r_obs[m].tolist(), p_perm.tolist(), p_fwer.tolist()):
            rows.append(
                {
                    "parcel_id": int(pid),
                    "model": m,
                    "r_obs": float(r),
                    "p_perm": float(pp),
                    "p_fwer": float(pf),
                    "tail": "greater",
                    "n_subjects": int(n_sub),
                    "n_pairs": int(n_pairs),
                    "n_perm": int(n_perm),
                    "space": space,
                    "task": task,
                    "condition_group": condition_group,
                    "aligned_csv": str(aligned_csv),
                    "atlas_label": str(atlas_label),
                    "atlas_label_used": str(atlas_label_used),
                }
            )

    result_path = out_dir / f"parcel_joint_analysis_dev_models_perm_fwer__{ctx}.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)

    pd.DataFrame({"subject": subjects, "age": ages.astype(float)}).to_csv(out_dir / f"subjects__{ctx}.csv", index=False)
    pd.DataFrame({"stimulus_content": list(keys)}).to_csv(out_dir / f"stimuli__{ctx}.csv", index=False)
    return result_path


def main() -> None:
    preset = choose_preset(ACTIVE_ANALYSIS_PRESET, PRESETS_ANALYSIS)
    out = run(
        aligned_csv=Path(preset.aligned_csv),
        atlas_label=Path(preset.atlas_label),
        subject_info=Path(preset.subject_info),
        out_dir=Path(preset.out_dir),
        space=str(preset.space),
        task=str(preset.task).strip(),
        condition_group=str(preset.condition_group).strip(),
        lss_root=Path(preset.lss_root) if preset.lss_root is not None else None,
        subject_allowlist=Path(preset.subject_allowlist) if preset.subject_allowlist is not None else None,
        min_coverage=float(preset.min_coverage),
        ensure_complete_keys=bool(preset.ensure_complete_keys),
        require_both_tasks=bool(preset.require_both_tasks),
        n_perm=int(preset.n_perm),
        seed=int(preset.seed),
        normalize_models=bool(preset.normalize_models),
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
