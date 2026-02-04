import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_atlas(atlas_img_path: Path, space: str) -> object:
    """加载 atlas 数据对象：volume 返回 NIfTI，surface 返回 label 向量。"""
    if not atlas_img_path.exists():
        raise FileNotFoundError(f"Atlas 图像不存在: {atlas_img_path}")
    if space == "volume":
        img = image.load_img(str(atlas_img_path))
        return img
    path_str = str(atlas_img_path)
    if path_str.endswith(".annot"):
        labels, _, _ = nib.freesurfer.read_annot(path_str)
        return labels
    if path_str.endswith(".gii"):
        gii = nib.load(path_str)
        data = gii.darrays[0].data.astype(int)
        return data
    raise ValueError(f"不支持的 surface atlas: {atlas_img_path}")


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    """从索引行解析 beta 文件路径，兼容扁平与 run 子目录两种落盘结构。"""
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


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一字段格式，保证 subject/stimulus_content/task 可用。"""
    df = df.dropna(subset=["subject", "stimulus_content", "file", "task"]).copy()
    df["subject"] = df["subject"].astype(str)
    df["stimulus_content"] = df["stimulus_content"].astype(str)
    df["task"] = df["task"].astype(str).str.upper()
    return df


def subjects_with_complete_stimuli(df: pd.DataFrame) -> List[str]:
    """筛选 EMO 与 SOC 均完整(各 21)且去重后无冲突的被试列表。"""
    dup_counts = df.groupby(["subject", "stimulus_content"]).size()
    dup_subjects = dup_counts[dup_counts > 1].index.get_level_values(0).unique().tolist()
    df = df[~df["subject"].isin(dup_subjects)].copy()
    counts = df.groupby(["subject", "task"])["stimulus_content"].nunique().unstack(fill_value=0)
    valid = counts[(counts.get("EMO", 0) == 21) & (counts.get("SOC", 0) == 21)].index.tolist()
    return sorted(valid)


def build_stimulus_order(df: pd.DataFrame, subjects: List[str]) -> List[str]:
    """构建全体被试一致的刺激顺序：EMO 在前、SOC 在后，按名称排序。"""
    df_sub = df[df["subject"].isin(subjects)].copy()
    df_sub = df_sub.drop_duplicates(["subject", "stimulus_content"])
    stim_to_task = df_sub.groupby("stimulus_content")["task"].first().to_dict()

    def key_func(stim: str) -> Tuple[int, str]:
        t = stim_to_task.get(stim, "")
        rank = 0 if t == "EMO" else 1 if t == "SOC" else 2
        return rank, stim
    stimuli = sorted(stim_to_task.keys(), key=key_func)
    if len(stimuli) != 42:
        raise ValueError(f"刺激数量不是 42，而是 {len(stimuli)}")
    for s in subjects:
        stim_set = set(df_sub[df_sub["subject"] == s]["stimulus_content"].tolist())
        if set(stimuli) != stim_set:
            raise ValueError(f"被试 {s} 的刺激集合不完整或不一致")
    return stimuli


def build_lookup(df: pd.DataFrame, subjects: List[str], stimuli: List[str], lss_root: Path) -> Dict[Tuple[str, str], Path]:
    """构建 (subject, stimulus_content) -> beta 文件路径 的查找表。"""
    df_sub = df[df["subject"].isin(subjects)].copy()
    lookup: Dict[Tuple[str, str], Path] = {}
    for _, row in df_sub.iterrows():
        key = (str(row["subject"]), str(row["stimulus_content"]))
        if key in lookup:
            continue
        lookup[key] = resolve_beta_path(lss_root, row)
    for s in subjects:
        for stim in stimuli:
            p = lookup.get((s, stim))
            if not p or not p.exists():
                raise FileNotFoundError(f"缺失 beta 文件: {s} {stim} {p}")
    return lookup


def prepare_roi_indices(atlas_obj: object, space: str) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    """从 atlas 中提取 ROI ID 与其体素/顶点索引。"""
    if space == "volume":
        atlas_data = atlas_obj.get_fdata().reshape(-1)
    else:
        if isinstance(atlas_obj, np.ndarray):
            atlas_data = atlas_obj.reshape(-1)
        else:
            atlas_data = np.asarray(atlas_obj).reshape(-1)
    roi_ids = np.unique(atlas_data)
    roi_ids = roi_ids[roi_ids > 0].astype(int)
    roi_ids = sorted(roi_ids)
    roi_indices = [np.where(atlas_data == rid)[0] for rid in roi_ids]
    if not roi_indices:
        raise ValueError("未匹配到 ROI")
    return atlas_data, roi_ids, roi_indices


def extract_roi_means(beta_data: np.ndarray, roi_indices: List[np.ndarray]) -> np.ndarray:
    """按 ROI 索引计算均值，得到每个 ROI 的 beta。"""
    return np.array([np.nanmean(beta_data[idx]) for idx in roi_indices], dtype=np.float32)


def extract_space_matrix(
    space: str,
    df: pd.DataFrame,
    subjects: List[str],
    stimuli: List[str],
    lss_root: Path,
    atlas_img: Path
) -> Tuple[np.ndarray, List[int]]:
    """读取某一空间的所有 beta，输出 ROI×被试×刺激 的矩阵与 ROI ID。"""
    atlas_obj = load_atlas(atlas_img, "volume" if space == "volume" else "surface")
    atlas_data, roi_ids, roi_indices = prepare_roi_indices(atlas_obj, "volume" if space == "volume" else "surface")
    lookup = build_lookup(df, subjects, stimuli, lss_root)
    n_roi = len(roi_ids)
    n_sub = len(subjects)
    n_stim = len(stimuli)
    mat = np.zeros((n_roi, n_sub, n_stim), dtype=np.float32)
    for si, s in enumerate(subjects):
        for ki, stim in enumerate(stimuli):
            p = lookup[(s, stim)]
            if space == "volume":
                img = image.load_img(str(p))
                if img.shape != atlas_obj.shape or not np.allclose(img.affine, atlas_obj.affine):
                    img = image.resample_to_img(img, atlas_obj, interpolation="nearest")
                data = img.get_fdata().reshape(-1)
            else:
                g = nib.load(str(p))
                data = g.darrays[0].data.astype(np.float32).reshape(-1)
            if data.shape[0] != atlas_data.shape[0]:
                raise ValueError(f"{space} 数据长度不匹配: {p}")
            mat[:, si, ki] = extract_roi_means(data, roi_indices)
    return mat, roi_ids


def validate_stimuli_consistency(df: pd.DataFrame, subjects: List[str], stimuli: List[str]) -> None:
    """验证每个被试的刺激集合与全体目标顺序一致。"""
    df_sub = df[df["subject"].isin(subjects)].copy()
    df_sub = df_sub.drop_duplicates(["subject", "stimulus_content"])
    target = set(stimuli)
    for s in subjects:
        stim_set = set(df_sub[df_sub["subject"] == s]["stimulus_content"].tolist())
        if stim_set != target:
            raise ValueError(f"被试 {s} 的刺激集合不一致")


def main() -> None:
    """主流程：读索引 -> 过滤被试 -> 读 beta -> ROI 均值 -> 拼接输出。"""
    LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
    INDEX_SURFACE_L = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv")
    INDEX_SURFACE_R = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv")
    INDEX_VOLUME = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv")
    ATLAS_SURFACE_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
    ATLAS_SURFACE_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
    ATLAS_VOLUME = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")
    OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")
    OUT_PREFIX = "roi_beta_matrix_232"

    df_l = normalize_df(pd.read_csv(INDEX_SURFACE_L))
    df_r = normalize_df(pd.read_csv(INDEX_SURFACE_R))
    df_v = normalize_df(pd.read_csv(INDEX_VOLUME))

    subjects_l = subjects_with_complete_stimuli(df_l)
    subjects_r = subjects_with_complete_stimuli(df_r)
    subjects_v = subjects_with_complete_stimuli(df_v)
    subjects = sorted(set(subjects_l) & set(subjects_r) & set(subjects_v))
    if not subjects:
        raise ValueError("没有满足 42 刺激且三空间均完整的被试")

    stimuli = build_stimulus_order(df_l, subjects)
    validate_stimuli_consistency(df_r, subjects, stimuli)
    validate_stimuli_consistency(df_v, subjects, stimuli)

    logger.info(f"被试数: {len(subjects)}")
    logger.info(f"刺激数: {len(stimuli)}")

    mat_l, names_l = extract_space_matrix(
        "surface_L", df_l, subjects, stimuli, LSS_ROOT, ATLAS_SURFACE_L
    )
    mat_r, names_r = extract_space_matrix(
        "surface_R", df_r, subjects, stimuli, LSS_ROOT, ATLAS_SURFACE_R
    )
    mat_v, names_v = extract_space_matrix(
        "volume", df_v, subjects, stimuli, LSS_ROOT, ATLAS_VOLUME
    )

    if (mat_l.shape[0] + mat_r.shape[0]) != 200 or mat_v.shape[0] != 32:
        raise ValueError(f"ROI 数量不匹配: L={mat_l.shape[0]} R={mat_r.shape[0]} V={mat_v.shape[0]}")

    matrix = np.concatenate([mat_l, mat_r, mat_v], axis=0)
    roi_names = [f"L_{n}" for n in names_l] + [f"R_{n}" for n in names_r] + [f"V_{n}" for n in names_v]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{OUT_PREFIX}.npy", matrix)
    pd.DataFrame({"roi": roi_names}).to_csv(OUT_DIR / f"{OUT_PREFIX}_rois.csv", index=False)
    pd.DataFrame({"subject": subjects}).to_csv(OUT_DIR / f"{OUT_PREFIX}_subjects.csv", index=False)
    pd.DataFrame({"stimulus_content": stimuli}).to_csv(OUT_DIR / f"{OUT_PREFIX}_stimuli.csv", index=False)
    logger.info(f"保存矩阵: {OUT_DIR / f'{OUT_PREFIX}.npy'}")


if __name__ == "__main__":
    main()
