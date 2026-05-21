import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. 基础工具函数
# =============================================================================

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
    """从索引行解析 beta 文件路径。"""
    f = str(row.get("file", ""))
    if f.startswith("/"):
        return Path(f)

    task = str(row.get("task", "")).lower()
    if not task and "stimulus_content" in row:
        content = str(row["stimulus_content"])
        if content.startswith("EMO"):
            task = "emo"
        elif content.startswith("SOC"):
            task = "soc"

    sub = str(row.get("subject", ""))

    # 策略 1: Task/Subject/File
    p1 = lss_root / task / sub / f
    if p1.exists():
        return p1

    # 策略 2: Task/Subject/run-x/File
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一字段格式。"""
    req_cols = ["subject", "file", "stimulus_content"]
    for c in req_cols:
        if c not in df.columns:
            df[c] = ""

    df = df.dropna(subset=["subject", "file", "stimulus_content"]).copy()
    df["subject"] = df["subject"].astype(str).str.strip()
    df["stimulus_content"] = df["stimulus_content"].astype(str).str.strip()
    return df


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
        raise ValueError("未匹配到 ROI (Atlas 为空?)")

    return atlas_data, roi_ids, roi_indices


def extract_roi_means(beta_data: np.ndarray, roi_indices: List[np.ndarray]) -> np.ndarray:
    """按 ROI 索引计算均值。"""
    return np.array([np.nanmean(beta_data[idx]) for idx in roi_indices], dtype=np.float32)


# =============================================================================
# 2. 核心逻辑：智能筛选 (直接使用 stimulus_content)
# =============================================================================

def intelligent_filter_data(df: pd.DataFrame, threshold_ratio: float = 0.80) -> Tuple[List[str], List[str]]:
    """
    智能过滤函数：
    1. 直接使用 stimulus_content 作为唯一标识 (不进行替换)。
    2. 找出出现在 >80% 被试中的刺激。
    3. 找出拥有所有这些热门刺激的被试。
    """
    df = df.copy()

    # 直接使用原始列，去除首尾空格
    df['clean_id'] = df['stimulus_content'].str.strip()

    # 排除无效行
    df = df[df['clean_id'].str.len() > 1]

    # --- 统计刺激覆盖率 ---
    unique_pairs = df[['subject', 'clean_id']].drop_duplicates()
    n_total_subjects = unique_pairs['subject'].nunique()
    stim_counts = unique_pairs['clean_id'].value_counts()

    # 阈值：80%
    threshold_count = n_total_subjects * threshold_ratio

    valid_stimuli = stim_counts[stim_counts >= threshold_count].index.tolist()
    valid_stimuli = sorted(valid_stimuli)

    logger.info("--- 刺激筛选报告 ---")
    logger.info(f"被试总数: {n_total_subjects}")
    logger.info(f"唯一刺激数 (Unique IDs): {len(stim_counts)}")
    logger.info(f"满足 80% 覆盖率的刺激数: {len(valid_stimuli)}")

    if len(valid_stimuli) == 0:
        raise ValueError(
            "没有刺激满足 80% 覆盖率！如果您的数据存在被试间平衡(Counterbalancing)，且 stimulus_content 包含条件名(Passive/Reappraisal)，请确保 CSV 已清洗，或降低阈值。")

    if len(valid_stimuli) != 42:
        logger.warning(f"注意：筛选出的刺激数量不是 42，而是 {len(valid_stimuli)}。")
    else:
        logger.info("成功锁定 42 个共性刺激。")

    # --- 反向筛选被试 ---
    df_valid_stim = unique_pairs[unique_pairs['clean_id'].isin(valid_stimuli)]
    sub_counts = df_valid_stim.groupby('subject')['clean_id'].nunique()

    # 只有拥有所有选中刺激的被试才保留
    valid_subjects = sub_counts[sub_counts == len(valid_stimuli)].index.tolist()
    valid_subjects = sorted(valid_subjects)

    logger.info(f"拥有全部 {len(valid_stimuli)} 个刺激的被试数: {len(valid_subjects)} (原始 {n_total_subjects})")

    return valid_subjects, valid_stimuli


def extract_space_matrix(
        space: str,
        df: pd.DataFrame,
        subjects: List[str],
        stimuli: List[str],
        lss_root: Path,
        atlas_img: Path
) -> Tuple[np.ndarray, List[int]]:
    """提取 beta 矩阵。"""

    atlas_obj = load_atlas(atlas_img, "volume" if space == "volume" else "surface")
    atlas_data, roi_ids, roi_indices = prepare_roi_indices(atlas_obj, "volume" if space == "volume" else "surface")

    df = df.copy()
    # 同样直接使用原始列
    df['clean_id'] = df['stimulus_content'].str.strip()

    # 构建查找表
    df_sub = df[df["subject"].isin(subjects)].copy()
    lookup: Dict[Tuple[str, str], Path] = {}

    for _, row in df_sub.iterrows():
        key = (str(row["subject"]), str(row["clean_id"]))
        if key not in lookup:
            lookup[key] = resolve_beta_path(lss_root, row)

    n_roi = len(roi_ids)
    n_sub = len(subjects)
    n_stim = len(stimuli)

    logger.info(f"提取 {space} ... 矩阵维度: ({n_roi}, {n_sub}, {n_stim})")

    mat = np.zeros((n_roi, n_sub, n_stim), dtype=np.float32)

    for si, s in enumerate(subjects):
        for ki, stim in enumerate(stimuli):
            p = lookup.get((s, stim))

            if p is None or not p.exists():
                raise FileNotFoundError(f"缺失文件: Subject={s}, Stimulus={stim}")

            if space == "volume":
                img = image.load_img(str(p))
                if img.shape != atlas_obj.shape or not np.allclose(img.affine, atlas_obj.affine):
                    img = image.resample_to_img(img, atlas_obj, interpolation="nearest")
                data = img.get_fdata().reshape(-1)
            else:
                g = nib.load(str(p))
                data = g.darrays[0].data.astype(np.float32).reshape(-1)

            mat[:, si, ki] = extract_roi_means(data, roi_indices)

    return mat, roi_ids


# =============================================================================
# 3. 主程序
# =============================================================================

def main() -> None:
    # --- 配置路径 ---
    LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")

    # 输入 CSV
    INDEX_SURFACE_L = LSS_ROOT / "lss_index_surface_L_aligned.csv"
    INDEX_SURFACE_R = LSS_ROOT / "lss_index_surface_R_aligned.csv"
    INDEX_VOLUME = LSS_ROOT / "lss_index_volume_aligned.csv"

    # Atlas
    ATLAS_SURFACE_L = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_L.label.gii")
    ATLAS_SURFACE_R = Path("/public/home/dingrui/tools/masks_atlas/Schaefer2018_200Parcels_7Networks_R.label.gii")
    ATLAS_VOLUME = Path("/public/home/dingrui/tools/masks_atlas/Tian_Subcortex_S2_3T_2009cAsym.nii.gz")

    # 输出
    OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")
    OUT_PREFIX = "roi_beta_matrix_232"

    # 1. 读取
    logger.info("读取索引文件...")
    df_l = normalize_df(pd.read_csv(INDEX_SURFACE_L))
    df_r = normalize_df(pd.read_csv(INDEX_SURFACE_R))
    df_v = normalize_df(pd.read_csv(INDEX_VOLUME))

    # 2. 智能筛选 (直接使用 stimulus_content, 80% 规则)
    logger.info("开始智能筛选 (Strict Match, 80% Threshold)...")
    valid_subjects, valid_stimuli = intelligent_filter_data(df_l, threshold_ratio=0.80)

    if not valid_subjects:
        raise ValueError("筛选后无剩余被试。")

    # 3. 提取矩阵
    mat_l, names_l = extract_space_matrix(
        "surface_L", df_l, valid_subjects, valid_stimuli, LSS_ROOT, ATLAS_SURFACE_L
    )
    mat_r, names_r = extract_space_matrix(
        "surface_R", df_r, valid_subjects, valid_stimuli, LSS_ROOT, ATLAS_SURFACE_R
    )
    mat_v, names_v = extract_space_matrix(
        "volume", df_v, valid_subjects, valid_stimuli, LSS_ROOT, ATLAS_VOLUME
    )

    # 4. 校验
    if (mat_l.shape[0] + mat_r.shape[0]) != 200 or mat_v.shape[0] != 32:
        logger.warning(f"ROI 数量异常: {mat_l.shape[0]}+{mat_r.shape[0]}+{mat_v.shape[0]}")
    else:
        logger.info("ROI 数量校验通过 (232).")

    matrix = np.concatenate([mat_l, mat_r, mat_v], axis=0)
    roi_names = [f"L_{n}" for n in names_l] + [f"R_{n}" for n in names_r] + [f"V_{n}" for n in names_v]

    # 5. 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{OUT_PREFIX}.npy", matrix)
    pd.DataFrame({"roi": roi_names}).to_csv(OUT_DIR / f"{OUT_PREFIX}_rois.csv", index=False)
    pd.DataFrame({"subject": valid_subjects}).to_csv(OUT_DIR / f"{OUT_PREFIX}_subjects.csv", index=False)
    # 这里的 stimulus_content 是原始未清洗的值
    pd.DataFrame({"stimulus_content": valid_stimuli}).to_csv(OUT_DIR / f"{OUT_PREFIX}_stimuli.csv", index=False)

    logger.info(f"处理完成! 矩阵: {matrix.shape}, 结果保存至: {OUT_DIR}")


if __name__ == "__main__":
    main()