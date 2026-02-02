#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_roi_joint_analysis_dev_models_perm_fwer.py

目标：
1) 参考 combined_roi_joint_analysis_dev_models_perm_fwer 的整体流程（相关 + 置换 + FWER）
2) 参考 calc_isc_roi_aligned_age 的 ROI 配置方式（多个核心 ROI）
3) 在 surface_L / surface_R / volume 上分别计算多个核心脑区的相似性向量，
   并与 3 个发育模型 (M_nn/M_conv/M_div) 做相关，输出多 ROI 结果表。

说明：
- Surface ROI 使用 Schaefer 7-networks 的核心 ROI 列表（可按需增删）。
- Volume ROI 请根据 Tian 图谱 label ID 修改 VOLUME_ROI_CONFIG（目前给出示例分组）。
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
from tqdm import tqdm

# ================= 配置常量 =================

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

# ================= ROI 配置（核心脑区） =================

# Surface ROI 参考 calc_isc_roi_aligned_age 的定义方式
SURFACE_ROI_CONFIG = {
    "Visual": {"L": [1, 2, 3, 4, 5, 6, 7], "R": [101, 102, 103, 104, 105, 106, 107]},
    "Salience_Insula": {"L": [38, 39, 40, 41, 42], "R": [138, 139, 140, 141, 142]},
    "Control_PFC": {"L": [57, 58, 59, 60, 61, 62], "R": [157, 158, 159, 160, 161, 162]},
}

# Volume ROI 示例（请根据 Tian 图谱 label ID 进行校对/调整）
VOLUME_ROI_CONFIG = {
    "Thalamus": [1, 2],
    "Striatum": [3, 4, 5, 6],
    "Limbic_MTL": [9, 10, 11, 12],
}


@dataclass(frozen=True)
class RoiSpec:
    name: str
    aligned_csv: Path
    atlas_label: Path
    space: str  # "surface_L" | "surface_R" | "volume"
    roi_ids: Sequence[int]


@dataclass(frozen=True)
class MultiRoiPreset:
    name: str
    specs: Sequence[RoiSpec]
    subject_info: Path
    out_dir: Path
    lss_root: Optional[Path]
    task: str
    condition_group: str
    min_coverage: float = 0.9
    ensure_complete_keys: bool = True
    max_missing_stimuli: int = 0
    n_perm: int = 5000
    seed: int = 42
    normalize_models: bool = True
    subject_order_path: Optional[Path] = None


# ================= 预设配置区 =================

ACTIVE_PRESET = "EXAMPLE_MULTI_ROI"

PRESETS: Dict[str, MultiRoiPreset] = {
    "EXAMPLE_MULTI_ROI": MultiRoiPreset(
        name="EXAMPLE_MULTI_ROI",
        specs=(),
        subject_info=Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv"),
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/multi_roi_perm_fwer_out"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="",
        condition_group="all",
        n_perm=5000,
    )
}


# ================= 基础工具函数 =================

def parse_chinese_age_exact(age_str: object) -> float:
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


def infer_condition_group(row: pd.Series) -> str:
    cg = str(row.get("condition_group", "")).strip()
    if cg and cg.lower() != "nan":
        return cg
    raw_cond = str(row.get("raw_condition", "")).strip()
    raw_emo = str(row.get("raw_emotion", "")).strip()
    if raw_cond == "Reappraisal":
        return "emotion_regulation"
    if raw_cond == "PassiveLook":
        return "neutral_response" if raw_emo.lower().startswith("neutral") else "emotion_generation"
    return "other"


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", ""))
    if f.startswith("/"):
        return Path(f)
    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))
    p1 = lss_root / task / sub / f
    if p1.exists():
        return p1
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def filter_dataframe(df: pd.DataFrame, task: str, condition_group: str) -> pd.DataFrame:
    df = df.dropna(subset=["subject", "stimulus_content", "file", "task"]).copy()
    df["subject"] = df["subject"].astype(str)
    df["task"] = df["task"].astype(str)
    df["condition_group"] = df.apply(infer_condition_group, axis=1)
    if task:
        df = df[df["task"] == task].copy()
    if condition_group and condition_group != "all":
        df = df[df["condition_group"] == condition_group].copy()
    return df


def load_subject_order_list(order_path: Path) -> List[str]:
    if not order_path.exists():
        raise FileNotFoundError(f"未找到顺序文件: {order_path}")
    df = pd.read_csv(order_path)
    for col in ["subject", "sub_id", "被试编号"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def determine_subject_list(
    preset: MultiRoiPreset,
    age_map: Dict[str, float],
    dfs: Sequence[pd.DataFrame],
) -> List[str]:
    if preset.subject_order_path:
        ordered_subs = load_subject_order_list(preset.subject_order_path)
        available_sets = [set(d["subject"].unique()) for d in dfs]
        common_available = set.intersection(*available_sets)
        valid_subs = [s for s in ordered_subs if s in common_available]
        valid_subs = [s for s in valid_subs if np.isfinite(age_map.get(s, np.nan))]
        return valid_subs

    sets = [set(d["subject"].unique()) for d in dfs]
    common = set.intersection(*sets)
    sorted_common = sorted(list(common), key=lambda s: (age_map.get(s, np.nan), s))
    return [s for s in sorted_common if np.isfinite(age_map.get(s, np.nan))]


def load_surface_labels(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.int32)


def load_volume_labels(path: Path, ref_img: image.Nifti1Image) -> np.ndarray:
    atlas_img = image.load_img(str(path))
    if atlas_img.shape != ref_img.shape or not np.allclose(atlas_img.affine, ref_img.affine):
        atlas_img = image.resample_to_img(atlas_img, ref_img, interpolation="nearest")
    labels = atlas_img.get_fdata(dtype=np.float32)
    return np.asarray(labels, dtype=np.int32)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 2:
        return x
    mu = x[mask].mean()
    sd = x[mask].std(ddof=0)
    if sd <= 1e-9:
        return x
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out


def build_dev_models(ages: np.ndarray, n: int, normalize: bool) -> Dict[str, np.ndarray]:
    iu, ju = np.triu_indices(n, k=1)
    a = ages.reshape(-1)
    amax = np.nanmax(a)
    ai, aj = a[iu], a[ju]
    models = {}
    models["M_nn"] = zscore_1d(amax - np.abs(ai - aj)) if normalize else (amax - np.abs(ai - aj))
    models["M_conv"] = zscore_1d(np.minimum(ai, aj)) if normalize else np.minimum(ai, aj)
    models["M_div"] = zscore_1d(amax - 0.5 * (ai + aj)) if normalize else (amax - 0.5 * (ai + aj))
    return models


def load_roi_data(
    spec: RoiSpec,
    df: pd.DataFrame,
    subjects: Sequence[str],
    lss_root: Path,
    preset: MultiRoiPreset,
) -> Optional[np.ndarray]:
    df_sub = df[df["subject"].isin(subjects)].copy()
    counts = df_sub["stimulus_content"].value_counts()
    threshold = int(len(subjects) * preset.min_coverage)
    keys = counts[counts >= threshold].index.tolist()
    keys.sort()

    if not keys:
        print(f"  [Skip] {spec.name}: 无公共刺激 (Threshold={threshold})")
        return None

    lookup = {}
    for _, row in df_sub.iterrows():
        lookup[(str(row["subject"]), str(row["stimulus_content"]))] = resolve_beta_path(lss_root, row)

    if preset.ensure_complete_keys:
        valid_keys = []
        for k in keys:
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
            print(f"  [Data] {spec.name}: 剔除 {dropped} 个缺失刺激。剩余刺激数: {len(valid_keys)}")
        keys = valid_keys

    if not keys:
        print(f"  [Skip] {spec.name}: 剔除缺失刺激后无剩余刺激")
        return None

    matrix_list = []
    print(f"  [Load] {spec.name}: {len(keys)} stimuli x {len(subjects)} subjects")

    mask = None
    volume_labels = None

    if spec.space.startswith("surface"):
        labels = load_surface_labels(spec.atlas_label)
        mask = np.isin(labels, spec.roi_ids)
    else:
        sample_path = None
        for s in subjects:
            for k in keys:
                p = lookup.get((s, k))
                if p and p.exists():
                    sample_path = p
                    break
            if sample_path:
                break
        if sample_path is None:
            print(f"  [Skip] {spec.name}: 未找到可用的 beta 文件")
            return None
        ref_img = image.load_img(str(sample_path))
        volume_labels = load_volume_labels(spec.atlas_label, ref_img)
        mask = np.isin(volume_labels, spec.roi_ids)

    if mask is None or int(mask.sum()) < 5:
        print(f"  [Skip] {spec.name}: ROI 顶点/体素过少")
        return None

    for s in tqdm(subjects, leave=False, desc=f"Reading {spec.name}"):
        feats = []
        for k in keys:
            p = lookup.get((s, k))
            if p is None or not p.exists():
                if preset.max_missing_stimuli == 0:
                    raise FileNotFoundError(f"被试 {s} 缺失刺激 {k}")
                continue
            if spec.space.startswith("surface"):
                g = nib.load(str(p))
                data = g.darrays[0].data.astype(np.float32).reshape(-1)
                feats.append(data[mask])
            else:
                img = image.load_img(str(p))
                if img.shape != volume_labels.shape or not np.allclose(img.affine, ref_img.affine):
                    img = image.resample_to_img(img, ref_img, interpolation="nearest")
                data = img.get_fdata(dtype=np.float32)
                feats.append(data[mask])
        if feats:
            matrix_list.append(np.concatenate(feats))

    if not matrix_list:
        return None
    return np.stack(matrix_list)


def run_analysis(preset: MultiRoiPreset) -> None:
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

    age_map = load_subject_ages_map(preset.subject_info)

    dfs = []
    for spec in preset.specs:
        df = pd.read_csv(spec.aligned_csv)
        df = filter_dataframe(df, preset.task, preset.condition_group)
        if df.empty:
            raise ValueError(f"{spec.name} data is empty after filtering!")
        dfs.append(df)

    subjects = determine_subject_list(preset, age_map, dfs)
    if len(subjects) < 10:
        print("❌ 有效被试过少 (<10)，终止分析。")
        return

    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(
        preset.out_dir / f"subjects_common_age_sorted__{ctx}.csv",
        index=False,
    )

    sim_vectors = []
    sim_names = []

    for spec, df in zip(preset.specs, dfs):
        print(f"\n--- Processing {spec.name} ({spec.space}) ---")
        data_mat = load_roi_data(spec, df, subjects, preset.lss_root, preset)
        if data_mat is None:
            print(f"  [Skip] {spec.name}: 无有效数据")
            continue

        corr_mat = np.corrcoef(data_mat)
        iu, ju = np.triu_indices(len(subjects), k=1)
        vec = corr_mat[iu, ju]
        vec_z = zscore_1d(vec)
        if np.isnan(vec_z).any():
            print(f"  [Skip] {spec.name}: 向量含 NaN")
            continue
        sim_vectors.append(vec_z)
        sim_names.append(spec.name)

    if not sim_vectors:
        print("❌ 无有效相似性矩阵生成，终止。")
        return

    S_z = np.stack(sim_vectors)
    n_pairs = S_z.shape[1]
    model_keys = ["M_nn", "M_conv", "M_div"]
    dev_models = build_dev_models(ages, len(subjects), preset.normalize_models)

    r_obs_dict = {m: (S_z @ dev_models[m]) / n_pairs for m in model_keys}
    r_obs_flat = np.concatenate([r_obs_dict[m] for m in model_keys]).astype(np.float32, copy=False)

    count_raw = np.zeros(int(r_obs_flat.size), dtype=np.int64)
    count_fwer = np.zeros(int(r_obs_flat.size), dtype=np.int64)

    rng = np.random.default_rng(preset.seed)
    for _ in range(int(preset.n_perm)):
        perm_idx = rng.permutation(len(subjects))
        ages_perm = ages[perm_idx]
        models_perm = build_dev_models(ages_perm, len(subjects), preset.normalize_models)
        flat_perm_rs = np.concatenate(
            [(S_z @ models_perm[m]) / n_pairs for m in model_keys]
        ).astype(np.float32, copy=False)
        count_raw += (flat_perm_rs >= r_obs_flat)
        max_stat = float(flat_perm_rs.max(initial=-1.0))
        count_fwer += (max_stat >= r_obs_flat)

    p_raw = (count_raw + 1.0) / (preset.n_perm + 1.0)
    p_fwer = (count_fwer + 1.0) / (preset.n_perm + 1.0)

    rows = []
    idx = 0
    for m in model_keys:
        rs = r_obs_dict[m]
        for roi_name, r_val in zip(sim_names, rs.tolist()):
            rows.append(
                {
                    "roi": roi_name,
                    "model": m,
                    "r_obs": float(r_val),
                    "p_perm_one_tailed": float(p_raw[idx]),
                    "p_fwer_one_tailed": float(p_fwer[idx]),
                    "n_subjects": int(len(subjects)),
                    "n_pairs": int(n_pairs),
                    "n_perm": int(preset.n_perm),
                    "seed": int(preset.seed),
                    "threshold": float(preset.min_coverage),
                    "normalize_models": bool(preset.normalize_models),
                }
            )
            idx += 1

    out_csv = preset.out_dir / f"multi_roi_joint_analysis_dev_models_perm_fwer__{ctx}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ 结果已保存: {out_csv}")


def _build_default_specs() -> List[RoiSpec]:
    specs: List[RoiSpec] = []
    surface_l = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv")
    surface_r = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv")
    volume = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv")

    atlas_l = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
    atlas_r = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
    atlas_vol = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")

    for roi_name, hemi_map in SURFACE_ROI_CONFIG.items():
        specs.append(
            RoiSpec(
                name=f"surface_L__{roi_name}",
                aligned_csv=surface_l,
                atlas_label=atlas_l,
                space="surface_L",
                roi_ids=hemi_map.get("L", []),
            )
        )
        specs.append(
            RoiSpec(
                name=f"surface_R__{roi_name}",
                aligned_csv=surface_r,
                atlas_label=atlas_r,
                space="surface_R",
                roi_ids=hemi_map.get("R", []),
            )
        )

    for roi_name, roi_ids in VOLUME_ROI_CONFIG.items():
        specs.append(
            RoiSpec(
                name=f"volume__{roi_name}",
                aligned_csv=volume,
                atlas_label=atlas_vol,
                space="volume",
                roi_ids=roi_ids,
            )
        )

    return specs


def choose_preset(name: str, presets: Dict[str, MultiRoiPreset]) -> MultiRoiPreset:
    if name not in presets:
        raise ValueError(f"Preset '{name}' 未找到，可用 presets: {sorted(presets.keys())}")
    preset = presets[name]
    if not preset.specs:
        preset = MultiRoiPreset(
            name=preset.name,
            specs=tuple(_build_default_specs()),
            subject_info=preset.subject_info,
            out_dir=preset.out_dir,
            lss_root=preset.lss_root,
            task=preset.task,
            condition_group=preset.condition_group,
            min_coverage=preset.min_coverage,
            ensure_complete_keys=preset.ensure_complete_keys,
            max_missing_stimuli=preset.max_missing_stimuli,
            n_perm=preset.n_perm,
            seed=preset.seed,
            normalize_models=preset.normalize_models,
            subject_order_path=preset.subject_order_path,
        )
    return preset


def main() -> None:
    preset = choose_preset(ACTIVE_PRESET, PRESETS)
    run_analysis(preset)


if __name__ == "__main__":
    main()
