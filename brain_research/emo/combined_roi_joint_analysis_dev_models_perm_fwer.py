#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_roi_joint_analysis_dev_models_perm_fwer.py

目的
-
把“整块脑区 combined”与“ROI 相似性矩阵”的构建、以及 joint 的置换检验整合到一个脚本中。
对 EMO/SOC 或单任务条件下：
1) 从 aligned 索引读取 beta，构建相似性矩阵（combined + ROI）
2) 基于共同被试做 joint 分析（3 个发育模型 + permutation + FWER）

说明
-
- 支持 surface_L / surface_R / volume 的 combined
- ROI 默认基于 Schaefer200 的 7Networks parcel id（surface）
- 自动取多个矩阵的共同被试，保证被试一致

使用方法
-
1) 根据你的数据路径修改 PRESETS（建议复制一个新 preset）：
   - aligned_csv: 对应 space 的 lss_index_*_aligned.csv
   - atlas_label/roi_labels: ROI 模式必填（surface）
   - volume_mask: volume combined 必填
   - task/condition_group/min_coverage/require_both_tasks: 过滤规则
   - ensure_complete_keys=True: 自动剔除缺失刺激以保留被试
   - max_missing_stimuli=0: 允许每个被试最多缺失多少刺激（为 None 则不剔除被试）
2) 修改 ACTIVE_PRESET 为你的 preset 名称。
3) 直接运行本脚本：
   python combined_roi_joint_analysis_dev_models_perm_fwer.py

输出
-
- similarity_{name}_AgeSorted.csv（可选，save_similarity=True 时）
- subjects_common_age_sorted.csv
- joint_analysis_dev_models_perm_fwer.csv（最终统计结果）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"


@dataclass(frozen=True)
class SimilaritySpec:
    name: str
    aligned_csv: Path
    space: str
    kind: str  # "combined" | "roi"
    atlas_label: Optional[Path] = None
    roi_labels: Optional[Sequence[int]] = None
    volume_mask: Optional[Path] = None


@dataclass(frozen=True)
class JointPreset:
    name: str
    specs: Sequence[SimilaritySpec]
    subject_info: Path
    out_dir: Path
    lss_root: Optional[Path]
    task: str
    condition_group: str
    require_both_tasks: bool = False
    min_coverage: float = 0.9
    ensure_complete_keys: bool = True
    max_missing_stimuli: Optional[int] = 0
    n_perm: int = 10000
    seed: int = 42
    normalize_models: bool = True
    save_similarity: bool = True


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


ACTIVE_PRESET = "EXAMPLE_EMO_joint_all"


PRESETS: Dict[str, JointPreset] = {
    "EXAMPLE_EMO_joint_all": JointPreset(
        name="EXAMPLE_EMO_joint_all",
        specs=(
            SimilaritySpec(
                name="surface_L",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="combined",
            ),
            SimilaritySpec(
                name="surface_R",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="combined",
            ),
            SimilaritySpec(
                name="volume",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"),
                space="volume",
                kind="combined",
                volume_mask=Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_Binary_Mask.nii.gz"),
            ),
            SimilaritySpec(
                name="surface_L_Visual",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="roi",
                atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                roi_labels=ROI_CONFIG["Visual"]["L"],
            ),
            SimilaritySpec(
                name="surface_R_Visual",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="roi",
                atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                roi_labels=ROI_CONFIG["Visual"]["R"],
            ),
            SimilaritySpec(
                name="surface_L_Salience_Insula",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="roi",
                atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                roi_labels=ROI_CONFIG["Salience_Insula"]["L"],
            ),
            SimilaritySpec(
                name="surface_R_Salience_Insula",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="roi",
                atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                roi_labels=ROI_CONFIG["Salience_Insula"]["R"],
            ),
            SimilaritySpec(
                name="surface_L_Control_PFC",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
                space="surface_L",
                kind="roi",
                atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii"),
                roi_labels=ROI_CONFIG["Control_PFC"]["L"],
            ),
            SimilaritySpec(
                name="surface_R_Control_PFC",
                aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
                space="surface_R",
                kind="roi",
                atlas_label=Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii"),
                roi_labels=ROI_CONFIG["Control_PFC"]["R"],
            ),
        ),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/joint_perm_fwer_out"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="EMO",
        condition_group="all",
        require_both_tasks=False,
        min_coverage=0.9,
        ensure_complete_keys=True,
        max_missing_stimuli=0,
        n_perm=10000,
        seed=42,
        normalize_models=True,
        save_similarity=True,
    )
}


def choose_preset(name: str, presets: Dict[str, JointPreset]) -> JointPreset:
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


def filter_dataframe(
    df: pd.DataFrame,
    task: str,
    condition_group: str,
    require_both_tasks: bool,
) -> pd.DataFrame:
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

    return df


def common_stimulus_keys(df: pd.DataFrame, subjects: Sequence[str], min_coverage: float) -> List[str]:
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
    lookup = {}
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


def filter_subjects_by_missing(
    df: pd.DataFrame,
    lss_root: Path,
    subjects: Sequence[str],
    keys: Sequence[str],
    max_missing: int,
) -> List[str]:
    key_set = set(keys)
    sub_set = set(subjects)
    df = df[df["subject"].astype(str).isin(sub_set) & df["stimulus_content"].astype(str).isin(key_set)].copy()
    lookup = {}
    for _, row in df.iterrows():
        s = str(row.get("subject", ""))
        k = str(row.get("stimulus_content", ""))
        lookup[(s, k)] = resolve_beta_path(lss_root, row)
    keep_subjects = []
    for sub in subjects:
        missing = 0
        for key in keys:
            fpath = lookup.get((sub, key))
            if fpath is None or not fpath.exists():
                missing += 1
                if missing > max_missing:
                    break
        if missing <= max_missing:
            keep_subjects.append(sub)
    return keep_subjects


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
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


def load_label_gii(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.int32)


def load_beta_gii(path: Path) -> Optional[np.ndarray]:
    try:
        g = nib.load(str(path))
        return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32)
    except Exception:
        return None


def load_beta_nii_flat(
    path: Path,
    voxel_idx: np.ndarray,
    mask_img: nib.spatialimages.SpatialImage,
) -> Optional[np.ndarray]:
    try:
        img = image.load_img(str(path))
        if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine):
            img = image.resample_to_img(img, mask_img, interpolation="continuous")
        data = img.get_fdata(dtype=np.float32).reshape(-1)
        return data[voxel_idx].astype(np.float32, copy=False)
    except Exception:
        return None


def build_roi_mask(atlas_label: Path, roi_labels: Sequence[int]) -> np.ndarray:
    labels = load_label_gii(atlas_label)
    mask = np.isin(labels, np.asarray(roi_labels, dtype=np.int32))
    return mask


def build_lookup(df: pd.DataFrame, lss_root: Path, subjects: Sequence[str], keys: Sequence[str]) -> Dict[Tuple[str, str], Path]:
    sub_set = set(subjects)
    key_set = set(keys)
    df = df[df["subject"].astype(str).isin(sub_set) & df["stimulus_content"].astype(str).isin(key_set)].copy()
    lookup: Dict[Tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        s = str(row.get("subject", ""))
        k = str(row.get("stimulus_content", ""))
        lookup[(s, k)] = resolve_beta_path(lss_root, row)
    return lookup


def build_subject_feature_matrix(
    df: pd.DataFrame,
    spec: SimilaritySpec,
    lss_root: Path,
    subjects: Sequence[str],
    keys: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    lookup = build_lookup(df, lss_root, subjects, keys)
    subject_vectors: List[np.ndarray] = []
    keep_subjects: List[str] = []

    if spec.space.startswith("surface"):
        roi_mask = None
        if spec.kind == "roi":
            if spec.atlas_label is None or not spec.roi_labels:
                raise ValueError(f"{spec.name} 缺少 ROI atlas/labels。")
            roi_mask = build_roi_mask(spec.atlas_label, spec.roi_labels)
        for sub in subjects:
            feats: List[np.ndarray] = []
            ok = True
            for key in keys:
                p = lookup.get((sub, key))
                if p is None or not p.exists():
                    ok = False
                    break
                beta = load_beta_gii(p)
                if beta is None:
                    ok = False
                    break
                if roi_mask is not None:
                    beta = beta[roi_mask]
                feats.append(beta)
            if ok and feats:
                subject_vectors.append(np.concatenate(feats).astype(np.float32, copy=False))
                keep_subjects.append(sub)
    elif spec.space == "volume":
        if spec.kind != "combined":
            raise ValueError("volume 仅支持 combined。")
        if spec.volume_mask is None:
            raise ValueError("volume 模式需要提供 volume_mask。")
        mask_img = image.load_img(str(spec.volume_mask))
        mask = mask_img.get_fdata() > 0
        voxel_idx = np.where(mask.reshape(-1))[0].astype(np.int64)
        for sub in subjects:
            feats: List[np.ndarray] = []
            ok = True
            for key in keys:
                p = lookup.get((sub, key))
                if p is None or not p.exists():
                    ok = False
                    break
                beta = load_beta_nii_flat(p, voxel_idx, mask_img)
                if beta is None:
                    ok = False
                    break
                feats.append(beta)
            if ok and feats:
                subject_vectors.append(np.concatenate(feats).astype(np.float32, copy=False))
                keep_subjects.append(sub)
    else:
        raise ValueError(f"未知空间类型: {spec.space}")

    if not keep_subjects:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(subject_vectors, axis=0), keep_subjects


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[mask].mean())
    sd = float(x[mask].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out.astype(np.float32, copy=False)


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def build_model_vec(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, model: str, normalize: bool) -> np.ndarray:
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


def prepare_similarity_vectors(
    matrices: Dict[str, np.ndarray],
    subjects: Sequence[str],
) -> Tuple[np.ndarray, List[str], int]:
    n = len(subjects)
    iu, ju = triu_indices(n)
    vecs = []
    names = []
    for name, mat in matrices.items():
        v = mat[iu, ju]
        v_z = zscore_1d(v)
        if not np.isfinite(v_z).any():
            raise ValueError(f"相似性矩阵向量无有效值: {name}")
        vecs.append(v_z)
        names.append(name)
    S = np.stack(vecs, axis=0).astype(np.float32, copy=False)
    return S, names, int(iu.size)


def run_joint_analysis(preset: JointPreset) -> Path:
    age_map = load_subject_ages_map(preset.subject_info)
    if not age_map:
        raise ValueError("无法加载年龄信息。")

    dfs = []
    for spec in preset.specs:
        df_full = pd.read_csv(spec.aligned_csv)
        df = filter_dataframe(df_full, preset.task, preset.condition_group, preset.require_both_tasks)
        if df.empty:
            raise ValueError(f"{spec.name} 过滤后 aligned 数据为空。")
        dfs.append(df)

    subject_sets = []
    for df in dfs:
        subs = set(df["subject"].astype(str).tolist())
        subs = {s if s.startswith("sub-") else f"sub-{s}" for s in subs}
        subs = {s for s in subs if np.isfinite(age_map.get(s, np.nan))}
        subject_sets.append(subs)

    common_subjects = set.intersection(*subject_sets) if subject_sets else set()
    if not common_subjects:
        raise ValueError("不同空间之间没有共同被试。")

    subjects_sorted = sorted(common_subjects, key=lambda s: (age_map.get(s, np.nan), s))
    if len(subjects_sorted) < 10:
        raise ValueError(f"共同被试数量过少: {len(subjects_sorted)}")

    lss_root = preset.lss_root if preset.lss_root is not None else preset.specs[0].aligned_csv.parent

    features: Dict[str, np.ndarray] = {}
    subjects_by_spec: Dict[str, List[str]] = {}
    keys_by_spec: Dict[str, List[str]] = {}

    for spec, df in zip(preset.specs, dfs):
        keys = common_stimulus_keys(df, subjects_sorted, preset.min_coverage)
        if not keys:
            raise ValueError(f"{spec.name} 无公共 stimulus_content。")
        if preset.ensure_complete_keys:
            keys = filter_keys_complete(df, lss_root, subjects_sorted, keys)
            if not keys:
                raise ValueError(f"{spec.name} 剔除缺失刺激后无可用 stimulus_content。")
        if preset.max_missing_stimuli is not None:
            max_missing = int(preset.max_missing_stimuli)
            if max_missing >= 0:
                before = len(subjects_sorted)
                subjects_sorted = filter_subjects_by_missing(
                    df,
                    lss_root,
                    subjects_sorted,
                    keys,
                    max_missing,
                )
                dropped = before - len(subjects_sorted)
                if dropped > 0:
                    print(f"  - {spec.name}: 已剔除 {dropped} 个缺失过多的被试 (max_missing={max_missing})")
                if not subjects_sorted:
                    raise ValueError(f"{spec.name} 剔除缺失被试后无可用被试。")
        feats, subjects_kept = build_subject_feature_matrix(
            df=df,
            spec=spec,
            lss_root=lss_root,
            subjects=subjects_sorted,
            keys=keys,
        )
        if feats.size == 0:
            raise ValueError(f"{spec.name} 无有效被试（可能缺失 beta）。")
        features[spec.name] = feats
        subjects_by_spec[spec.name] = subjects_kept
        keys_by_spec[spec.name] = keys

    common_subjects_final = set(subjects_sorted)
    for subs in subjects_by_spec.values():
        common_subjects_final &= set(subs)
    if not common_subjects_final:
        raise ValueError("加载 beta 后无共同被试。")

    subjects_final = sorted(common_subjects_final, key=lambda s: (age_map.get(s, np.nan), s))
    if len(subjects_final) < 10:
        raise ValueError(f"最终共同被试数量过少: {len(subjects_final)}")

    matrices: Dict[str, np.ndarray] = {}
    for spec_name, feats in features.items():
        subs = subjects_by_spec[spec_name]
        idx = [subs.index(s) for s in subjects_final]
        feats_use = feats[np.array(idx, dtype=int), :]
        matrices[spec_name] = np.corrcoef(feats_use)

    out_dir = preset.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if preset.save_similarity:
        for name, mat in matrices.items():
            out_path = out_dir / f"similarity_{name}_AgeSorted.csv"
            pd.DataFrame(mat, index=subjects_final, columns=subjects_final).to_csv(out_path)

    ages = np.array([age_map[s] for s in subjects_final], dtype=np.float32)
    pd.DataFrame({"subject": subjects_final, "age": ages}).to_csv(out_dir / "subjects_common_age_sorted.csv", index=False)

    S_z, matrix_names, n_pairs = prepare_similarity_vectors(matrices, subjects_final)
    iu, ju = triu_indices(len(subjects_final))
    models = ["M_nn", "M_conv", "M_div"]
    model_vecs = {m: build_model_vec(ages, iu, ju, model=m, normalize=preset.normalize_models) for m in models}
    for m in models:
        if not np.isfinite(model_vecs[m]).all():
            raise ValueError(f"模型向量 {m} 包含 NaN。")

    r_obs = {m: (S_z @ model_vecs[m]) / float(n_pairs) for m in models}
    n_mats = S_z.shape[0]
    n_tests = n_mats * len(models)

    count_raw = np.zeros(n_tests, dtype=np.int64)
    count_fwer = np.zeros(n_tests, dtype=np.int64)
    rng = np.random.default_rng(preset.seed)
    abs_r_obs = np.abs(np.concatenate([r_obs[m] for m in models]))

    for _ in range(int(preset.n_perm)):
        perm = rng.permutation(len(subjects_final))
        ages_p = ages[perm]
        r_perm_all = np.empty(n_tests, dtype=np.float32)
        offset = 0
        for m in models:
            mv = build_model_vec(ages_p, iu, ju, model=m, normalize=preset.normalize_models)
            r_mats = (S_z @ mv) / float(n_pairs)
            r_perm_all[offset:offset + n_mats] = r_mats.astype(np.float32, copy=False)
            offset += n_mats
        abs_r_perm = np.abs(r_perm_all)
        count_raw += (abs_r_perm >= abs_r_obs)
        max_abs = float(abs_r_perm.max(initial=0.0))
        count_fwer += (max_abs >= abs_r_obs)

    p_perm = (count_raw + 1.0) / (float(preset.n_perm) + 1.0)
    p_fwer = (count_fwer + 1.0) / (float(preset.n_perm) + 1.0)

    rows = []
    idx = 0
    for m in models:
        r_vals = r_obs[m]
        for mi, name in enumerate(matrix_names):
            rows.append(
                {
                    "matrix": name,
                    "model": m,
                    "r_obs": float(r_vals[mi]),
                    "p_perm": float(p_perm[idx]),
                    "p_fwer": float(p_fwer[idx]),
                    "n_subjects": int(len(subjects_final)),
                    "n_pairs": int(n_pairs),
                    "n_perm": int(preset.n_perm),
                    "seed": int(preset.seed),
                    "normalize_models": bool(preset.normalize_models),
                }
            )
            idx += 1

    res_df = pd.DataFrame(rows).sort_values(["matrix", "model"])
    out_path = out_dir / "joint_analysis_dev_models_perm_fwer.csv"
    res_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    preset = choose_preset(ACTIVE_PRESET, PRESETS)
    out = run_joint_analysis(preset)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
