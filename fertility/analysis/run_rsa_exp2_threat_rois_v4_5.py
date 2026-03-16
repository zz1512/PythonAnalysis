#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rsa_exp2_threat_rois_v4_3.py

实验2（威胁情境任务）ROI-RSA 增强版
-------------------------------------------------
在 v4.1 基础上的增强版：
1. 保留原有主假设检验 / 特异性检验 / 模型 RDM 检验
2. 继续使用“先 Fisher z，再平均，再反变换为 r”的 block 平均方式
3. 继续使用 cross-run only（run_i != run_j）构建类别相似性
4. 新增：按“类别 + item 对齐”排序的 group-level trial×trial similarity heatmap
5. 新增：导出每个 ROI、每个核心假设下每个被试的 left_r / right_r / left_z / right_z
6. 新增：基于被试层 left/right 数值绘制小提琴图
7. 新增：基于生育意愿连续得分（1–7）的调节分析代码（保持原有分析不变）
8. 新增：生育意愿低/中/高三组的探索性分组可视化
9. 中文字体自动配置，修复图片中文乱码
9. GPU 可选加速（优先 CuPy，其次 PyTorch；主要加速 trial×trial 相似性矩阵计算）

说明
-------------------------------------------------
- 输入：LSS 输出目录下各被试 run-4/5/6 的 beta 图与 trial_info.csv
- ROI：默认 ACC / Amygdala / Insula，可自行扩展
- trial-level pattern：在 ROI 内提取，每个 trial 做跨体素 z 标准化
- category matrix：先计算 trial×trial similarity，再做 block 平均得到 6×6 类别相似性矩阵
- block 平均：先对每个 pairwise r 做 Fisher z，再求平均；输出同时保存 mean_r 与 mean_z
- cross-run only：默认仅使用不同 run 之间的 trial pairs 参与 block 平均
- hypothesis tests：锚定式主假设检验
- specificity tests：检验目标类别对是否比同一 anchor 的其他类别对更高
- model-RSA：将观察到的 6×6 类别相似性矩阵与理论模型矩阵做 Spearman 相关，再做组层统计
- sorted trial heatmap：将 trial×trial similarity matrix 按“6 类条件 + item 对齐”后可视化，并在被试间求组平均

GPU 说明
-------------------------------------------------
- GPU 只能加速“相似性矩阵计算”这一步。
- NIfTI 读取、masker.transform、CSV 读取仍主要是 CPU / I/O 开销。
- 对 120 trial 这类规模，GPU 提升通常有限；如果 ROI 多、被试多，可以有一定收益。
"""

from __future__ import annotations

import re
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.maskers import NiftiMasker
from scipy.stats import ttest_rel, ttest_1samp, spearmanr, pearsonr, linregress
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 1. 配置区
# =========================
BASE_DIR = Path(r"C:/python_fertility")
LSS_ROOT = BASE_DIR / "lss_betas_exp2"
OUTPUT_DIR = BASE_DIR / "rsa_results_exp2"
RUNS = [4, 5, 6]
N_JOBS = 4
CROSS_RUN_ONLY = True  # 仅用跨 run 的 pairs 做类别相似性（推荐）
SAVE_GROUP_TRIAL_HEATMAPS = True
# 按沟通结果：不再默认生成每个被试的 trial 热图，正式结果只保留组平均 trial 热图
SAVE_SUBJECT_TRIAL_HEATMAPS = False

# GPU：主要加速 similarity matrix 计算；没有 CuPy / Torch CUDA 时自动回退 CPU
USE_GPU = True
GPU_BACKEND = "auto"   # auto | cupy | torch | cpu

# ROI 掩膜
ROI_MASKS = {
    "ACC_L": BASE_DIR / "roi_masks_refined/ACC_L.nii.gz",
    "ACC_R": BASE_DIR / "roi_masks_refined/ACC_R.nii.gz",
    "Amygdala_L": BASE_DIR / "roi_masks_refined/Amygdala_L.nii.gz",
    "Amygdala_R": BASE_DIR / "roi_masks_refined/Amygdala_R.nii.gz",
    "Insula_L_Anterior": BASE_DIR / "roi_masks_refined/Insula_L_Anterior.nii.gz",
    "Insula_L_Posterior": BASE_DIR / "roi_masks_refined/Insula_L_Posterior.nii.gz",
    "Insula_R_Anterior": BASE_DIR / "roi_masks_refined/Insula_R_Anterior.nii.gz",
    "Insula_R_Posterior": BASE_DIR / "roi_masks_refined/Insula_R_Posterior.nii.gz"
}

# 六类 threat code 固定顺序
COND_ORDER = ["fer", "soc", "val", "hed", "exi", "atf"]
COND_LABELS = {
    "fer": "生育威胁",
    "soc": "人际威胁",
    "val": "价值威胁",
    "hed": "享乐威胁",
    "exi": "生存威胁",
    "atf": "不生育威胁",
}

# 主假设（锚定式）
HYPOTHESIS_TESTS = [
    {
        "name": "H1_生育比不生育更接近生存",
        "left": ("fer", "exi"),
        "right": ("atf", "exi"),
        "alternative": "greater",
    },
    {
        "name": "H2_不生育比生育更接近享乐",
        "left": ("atf", "hed"),
        "right": ("fer", "hed"),
        "alternative": "greater",
    },
]

# 特异性检验：检验目标 pair 是否比同一“起点”下的其他 pair 更强
SPECIFICITY_TESTS = [
    # fer 是否特异性靠近 exi
    {"name": "S1_fer_exi_gt_fer_soc", "left": ("fer", "exi"), "right": ("fer", "soc"), "alternative": "greater"},
    {"name": "S2_fer_exi_gt_fer_val", "left": ("fer", "exi"), "right": ("fer", "val"), "alternative": "greater"},
    {"name": "S3_fer_exi_gt_fer_hed", "left": ("fer", "exi"), "right": ("fer", "hed"), "alternative": "greater"},
    {"name": "S4_fer_exi_gt_fer_atf", "left": ("fer", "exi"), "right": ("fer", "atf"), "alternative": "greater"},
    # atf 是否特异性靠近 hed
    {"name": "S5_atf_hed_gt_atf_soc", "left": ("atf", "hed"), "right": ("atf", "soc"), "alternative": "greater"},
    {"name": "S6_atf_hed_gt_atf_val", "left": ("atf", "hed"), "right": ("atf", "val"), "alternative": "greater"},
    {"name": "S7_atf_hed_gt_atf_exi", "left": ("atf", "hed"), "right": ("atf", "exi"), "alternative": "greater"},
    {"name": "S8_atf_hed_gt_atf_fer", "left": ("atf", "hed"), "right": ("atf", "fer"), "alternative": "greater"},
]

# 模型 RDM：这里用“模型相似性矩阵”，与观察矩阵做 Spearman 相关
# 值越大表示理论上越相似；对角线会被忽略
MODEL_SPECS = {
    "M1_fer_exi_only": [("fer", "exi")],
    "M2_atf_hed_only": [("atf", "hed")],
    "M3_combined": [("fer", "exi"), ("atf", "hed")],
}

# FDR 模式：global / by_test / by_roi
FDR_MODE_MAIN = "by_test"
FDR_MODE_SPECIFICITY = "by_test"
FDR_MODE_MODEL = "global"

# 图片风格
HEATMAP_CMAP = "coolwarm"
HEATMAP_VMIN = -0.2
HEATMAP_VMAX = 0.4
FIG_DPI = 200
TRIAL_HEATMAP_VMIN = -0.2
TRIAL_HEATMAP_VMAX = 0.4
TRIAL_HEATMAP_AUTO_RANGE = True
TRIAL_HEATMAP_QLOW = 0.05
TRIAL_HEATMAP_QHIGH = 0.95
TRIAL_HEATMAP_MASK_DIAGONAL = True
TRIAL_HEATMAP_MASK_SAME_RUN = True

PLOT_MAIN_PER_PAGE = 2
PLOT_SPECIFICITY_PER_PAGE = 4

# 最新热图范围要求
HEATMAP_VMIN = -0.2
HEATMAP_VMAX = 0.4

# 生育意愿连续得分分析（保持已有分析不变，仅新增）
RUN_FERTILITY_SCORE_ANALYSIS = True
FERTILITY_SCORE_FILE = BASE_DIR / "fertility_score.xlsx"
FERTILITY_SCORE_SHEET = 0
FERTILITY_SUBJECT_COL_CANDIDATES = ["参与者编号", "受试编号", "被试编号", "subject", "Subject", "ID", "id"]
FERTILITY_SCORE_COL_CANDIDATES = ["生育意愿", "生育意愿得分", "fertility_score", "score", "Score"]

# 生育意愿低/中/高三组探索性可视化
RUN_FERTILITY_GROUP_VIS = True
# 可选: "fixed"（1-2低,3-5中,6-7高） / "tertile"（按样本三分位）
FERTILITY_GROUP_MODE = "fixed"
FERTILITY_GROUP_ORDER = ["低生育意愿", "中生育意愿", "高生育意愿"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RSA_EXP2_V4_3")


# =========================
# 2. 字体 / GPU 初始化
# =========================
def configure_matplotlib_chinese():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "PingFang SC", "Heiti SC", "STHeiti",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break
    if chosen is None:
        for font_path in fm.findSystemFonts():
            low = font_path.lower()
            if any(k in low for k in ["yahei", "simhei", "simsun", "notosanscjk", "sourcehansans", "pingfang", "wqy"]):
                try:
                    prop = fm.FontProperties(fname=font_path)
                    chosen = prop.get_name()
                    break
                except Exception:
                    pass
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams.get("font.sans-serif", [])
        logger.info(f"Matplotlib 中文字体: {chosen}")
    else:
        logger.warning("未找到理想中文字体，图片可能仍有中文乱码；建议安装 Microsoft YaHei 或 Noto Sans CJK SC。")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


class GPUManager:
    def __init__(self, use_gpu: bool = True, backend: str = "auto"):
        self.use_gpu = use_gpu
        self.backend = "cpu"
        self.cp = None
        self.torch = None
        self.device = None

        if not use_gpu or backend == "cpu":
            logger.info("GPU 加速：关闭，使用 CPU。")
            return

        if backend in ("auto", "cupy"):
            try:
                import cupy as cp  # type: ignore
                _ = cp.cuda.runtime.getDeviceCount()
                self.cp = cp
                self.backend = "cupy"
                logger.info(f"GPU 加速：启用 CuPy，设备数量={cp.cuda.runtime.getDeviceCount()}")
                return
            except Exception:
                if backend == "cupy":
                    logger.warning("请求 CuPy GPU，但当前不可用，回退 CPU。")

        if backend in ("auto", "torch"):
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    self.torch = torch
                    self.backend = "torch"
                    self.device = torch.device("cuda")
                    logger.info(f"GPU 加速：启用 PyTorch CUDA，设备={torch.cuda.get_device_name(0)}")
                    return
                elif backend == "torch":
                    logger.warning("请求 PyTorch GPU，但 CUDA 不可用，回退 CPU。")
            except Exception:
                if backend == "torch":
                    logger.warning("请求 PyTorch GPU，但导入失败，回退 CPU。")

        logger.info("GPU 加速：不可用，使用 CPU。")

    def cleanup(self):
        if self.backend == "cupy" and self.cp is not None:
            try:
                self.cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        elif self.backend == "torch" and self.torch is not None:
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass


GPU = GPUManager(use_gpu=USE_GPU, backend=GPU_BACKEND)


# =========================
# 3. 工具函数
# =========================
def discover_subjects(lss_root: Path) -> List[str]:
    return sorted([p.name for p in lss_root.glob("sub-*") if p.is_dir()])


def parse_condition_from_row(row: pd.Series) -> str:
    for key in ["cond", "condition", "type", "trial_type"]:
        if key in row and pd.notna(row[key]):
            val = str(row[key]).strip().lower()
            if val in COND_ORDER:
                return val

    if "unique_label" in row and pd.notna(row["unique_label"]):
        val = str(row["unique_label"]).strip().lower()
        m = re.match(r"^([a-z]{3})[_-]", val)
        if m and m.group(1) in COND_ORDER:
            return m.group(1)

    if "beta_file" in row and pd.notna(row["beta_file"]):
        val = Path(str(row["beta_file"])).name.lower()
        m = re.search(r"_([a-z]{3})[_-]", val)
        if m and m.group(1) in COND_ORDER:
            return m.group(1)

    raise ValueError(f"无法从该行解析条件码: {row.to_dict()}")



def parse_item_key_from_row(row: pd.Series) -> str:
    """
    生成用于 trial 对齐与排序的 item key。
    优先顺序：
      1) unique_label，例如 fer_023
      2) stim，例如 023.png -> 023
      3) beta_file，例如 beta_trial-000_fer_023.nii.gz -> fer_023
      4) trial_index / onset 作为兜底
    返回格式尽量统一为: <cond>_<item>
    """
    cond = row["cond"] if "cond" in row else parse_condition_from_row(row)

    if "unique_label" in row and pd.notna(row["unique_label"]):
        s = str(row["unique_label"]).strip().lower()
        m = re.match(r"^([a-z]{3})[_-](.+)$", s)
        if m:
            return f"{m.group(1)}_{m.group(2)}"

    if "stim" in row and pd.notna(row["stim"]):
        s = Path(str(row["stim"])).stem.lower()
        return f"{cond}_{s}"

    if "beta_file" in row and pd.notna(row["beta_file"]):
        s = Path(str(row["beta_file"])).name.lower()
        m = re.search(r"_([a-z]{3})[_-]([^.]+)\.nii", s)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        m = re.search(r"_([a-z]{3})[_-]([^.]+)$", Path(s).stem)
        if m:
            return f"{m.group(1)}_{m.group(2)}"

    if "trial_index" in row and pd.notna(row["trial_index"]):
        return f"{cond}_trial{int(row['trial_index']):03d}"
    if "onset" in row and pd.notna(row["onset"]):
        return f"{cond}_onset{int(round(float(row['onset']))):06d}"
    return f"{cond}_unknown"


def item_sort_value(item_key: str):
    """
    返回一个稳定可比较的排序键：
    - 先按是否能提取数字排序
    - 再按数字或字符串本身排序
    """
    tail = item_key.split("_", 1)[1] if "_" in item_key else item_key
    nums = re.findall(r"\d+", tail)
    if nums:
        return (0, int(nums[-1]), "")
    return (1, 0, tail.lower())

def rowwise_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + eps)


def fisher_z_clip(r: np.ndarray, clip: float = 0.999999) -> np.ndarray:
    r = np.clip(r, -clip, clip)
    return np.arctanh(r)


def bh_fdr(pvals: List[float]) -> List[float]:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()


def apply_fdr(stats_df: pd.DataFrame, mode: str = "by_test", pcol_one: str = "p_one_sided", pcol_two: str = "p_two_sided") -> pd.DataFrame:
    stats_df = stats_df.copy()
    if len(stats_df) == 0:
        return stats_df

    if mode == "global":
        stats_df["q_fdr_one_sided"] = bh_fdr(stats_df[pcol_one].fillna(1.0).tolist())
        stats_df["q_fdr_two_sided"] = bh_fdr(stats_df[pcol_two].fillna(1.0).tolist())
    elif mode == "by_test":
        stats_df["q_fdr_one_sided"] = np.nan
        stats_df["q_fdr_two_sided"] = np.nan
        for _, subdf in stats_df.groupby("test"):
            idx = subdf.index
            stats_df.loc[idx, "q_fdr_one_sided"] = bh_fdr(subdf[pcol_one].fillna(1.0).tolist())
            stats_df.loc[idx, "q_fdr_two_sided"] = bh_fdr(subdf[pcol_two].fillna(1.0).tolist())
    elif mode == "by_roi":
        stats_df["q_fdr_one_sided"] = np.nan
        stats_df["q_fdr_two_sided"] = np.nan
        for _, subdf in stats_df.groupby("roi"):
            idx = subdf.index
            stats_df.loc[idx, "q_fdr_one_sided"] = bh_fdr(subdf[pcol_one].fillna(1.0).tolist())
            stats_df.loc[idx, "q_fdr_two_sided"] = bh_fdr(subdf[pcol_two].fillna(1.0).tolist())
    else:
        raise ValueError(f"未知 FDR 模式: {mode}")
    return stats_df


def one_sided_p_from_ttest(t_stat: float, p_two_sided: float, alternative: str = "greater") -> float:
    if np.isnan(t_stat) or np.isnan(p_two_sided):
        return np.nan
    if alternative == "greater":
        return p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    elif alternative == "less":
        return p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2
    else:
        return p_two_sided


def cohen_dz_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = np.std(d, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(d) / sd)


def cohen_d_onesample(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(x) / sd)


# =========================
# 4. 数据读取
# =========================
def load_subject_trials(sub: str) -> pd.DataFrame:
    dfs = []
    sub_dir = LSS_ROOT / sub

    for run in RUNS:
        run_dir = sub_dir / f"run-{run}"
        csv_path = run_dir / "trial_info.csv"
        if not csv_path.exists():
            logger.warning(f"{sub} run-{run}: 缺少 trial_info.csv，跳过")
            continue

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            logger.warning(f"{sub} run-{run}: trial_info.csv 为空，跳过")
            continue

        df = df.copy()
        df["subject"] = sub
        df["run"] = run
        df["cond"] = df.apply(parse_condition_from_row, axis=1)

        if "beta_file" not in df.columns:
            raise ValueError(f"{csv_path} 缺少 beta_file 列")

        df["beta_path"] = df["beta_file"].apply(lambda x: str((run_dir / Path(str(x)).name).resolve()))
        miss = df[~df["beta_path"].apply(lambda x: Path(x).exists())]
        if len(miss) > 0:
            examples = miss["beta_path"].tolist()[:5]
            raise FileNotFoundError(f"{sub} run-{run} 有 beta 文件缺失，例如: {examples}")

        dfs.append(df)

    if not dfs:
        raise ValueError(f"{sub} 没有可用 run 数据")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    sort_cols = [c for c in ["run", "trial_index", "onset"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    out["item_key"] = out.apply(parse_item_key_from_row, axis=1)

    cond_counts = out["cond"].value_counts().reindex(COND_ORDER).fillna(0).astype(int).to_dict()
    logger.info(f"{sub}: 各条件 trial 数 = {cond_counts}")
    return out


# =========================
# 5. RSA 核心
# =========================
def extract_patterns(beta_files: List[str], roi_mask_path: Path) -> np.ndarray:
    masker = NiftiMasker(mask_img=str(roi_mask_path), standardize=False)
    x = masker.fit_transform(beta_files)
    if x.ndim != 2:
        raise ValueError(f"提取到的 pattern 维度异常: {x.shape}")
    x = np.asarray(x, dtype=np.float32)
    x = rowwise_zscore(x).astype(np.float32, copy=False)
    return x


def compute_similarity_matrix(patterns: np.ndarray) -> np.ndarray:
    x = patterns

    if GPU.backend == "cupy" and GPU.cp is not None:
        cp = GPU.cp
        xg = cp.asarray(x, dtype=cp.float32)
        norms = cp.linalg.norm(xg, axis=1, keepdims=True)
        xg = xg / cp.maximum(norms, 1e-8)
        sim = xg @ xg.T
        sim = cp.clip(sim, -1.0, 1.0)
        out = cp.asnumpy(sim)
        del xg, norms, sim
        GPU.cleanup()
    elif GPU.backend == "torch" and GPU.torch is not None:
        torch = GPU.torch
        xg = torch.as_tensor(x, dtype=torch.float32, device=GPU.device)
        xg = torch.nn.functional.normalize(xg, p=2, dim=1)
        sim = xg @ xg.T
        sim = torch.clamp(sim, -1.0, 1.0)
        out = sim.detach().cpu().numpy()
        del xg, sim
        GPU.cleanup()
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-8)
        out = x @ x.T
        out = np.clip(out, -1.0, 1.0)

    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32, copy=False)


def block_mean_similarity(
    sim_mat: np.ndarray,
    labels: List[str],
    runs: List[int],
    cond_a: str,
    cond_b: str,
    cross_run_only: bool = True,
) -> Tuple[float, float, int]:
    idx_a = [i for i, x in enumerate(labels) if x == cond_a]
    idx_b = [i for i, x in enumerate(labels) if x == cond_b]

    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan, np.nan, 0

    vals = []
    for i in idx_a:
        for j in idx_b:
            if cond_a == cond_b and i == j:
                continue
            if cross_run_only and runs[i] == runs[j]:
                continue
            v = sim_mat[i, j]
            if np.isfinite(v):
                vals.append(v)

    if len(vals) == 0:
        return np.nan, np.nan, 0

    vals = np.asarray(vals, dtype=float)
    vals_z = fisher_z_clip(vals)
    mean_z = float(np.mean(vals_z))
    mean_r = float(np.tanh(mean_z))
    return mean_r, mean_z, int(len(vals))




def get_condition_sort_order(labels: List[str], item_keys: List[str] | None = None) -> np.ndarray:
    """
    按 COND_ORDER 排序；若提供 item_keys，则在类别内部再按 item 进行稳定对齐。
    用 Python 的 sorted 生成顺序，避免 np.lexsort 在 object/tuple 上的 shape 问题。
    """
    rank_map = {c: i for i, c in enumerate(COND_ORDER)}

    if item_keys is None:
        return np.array(
            sorted(range(len(labels)), key=lambda i: (rank_map.get(labels[i], 999), i)),
            dtype=int
        )

    if len(item_keys) != len(labels):
        raise ValueError(f"item_keys 长度 ({len(item_keys)}) 与 labels 长度 ({len(labels)}) 不一致")

    order = sorted(
        range(len(labels)),
        key=lambda i: (rank_map.get(labels[i], 999), item_sort_value(item_keys[i]), i)
    )
    return np.array(order, dtype=int)


def sort_trial_similarity_matrix(
    sim_mat: np.ndarray,
    labels: List[str],
    item_keys: List[str],
    runs: List[int] | None = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int] | None, np.ndarray]:
    """
    按条件 + item key 对 trial×trial 相似性矩阵排序，确保跨被试可对齐。
    返回：
        sorted_mat, sorted_labels, sorted_item_keys, sorted_runs, order
    """
    n = sim_mat.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels 长度 {len(labels)} 与 sim_mat 维度 {n} 不一致")
    if len(item_keys) != n:
        raise ValueError(f"item_keys 长度 {len(item_keys)} 与 sim_mat 维度 {n} 不一致")
    if runs is not None and len(runs) != n:
        raise ValueError(f"runs 长度 {len(runs)} 与 sim_mat 维度 {n} 不一致")

    order = get_condition_sort_order(labels, item_keys=item_keys)
    sorted_mat = sim_mat[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]
    sorted_item_keys = [item_keys[i] for i in order]
    sorted_runs = [runs[i] for i in order] if runs is not None else None
    return sorted_mat, sorted_labels, sorted_item_keys, sorted_runs, order


def get_condition_block_info(sorted_labels: List[str]) -> Tuple[List[int], List[float], List[str]]:
    """
    根据排序后的 labels 计算类别边界、刻度中心和刻度标签。
    """
    counts = [sum(1 for x in sorted_labels if x == cond) for cond in COND_ORDER]
    boundaries = np.cumsum(counts).tolist()
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    centers = (starts + np.array(counts) / 2.0 - 0.5).tolist()
    tick_labels = [COND_LABELS[c] for c in COND_ORDER]
    return boundaries, centers, tick_labels


def p_to_star(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def build_trial_heatmap_display_matrix(
    sorted_mat: np.ndarray,
    sorted_runs: List[int] | None,
    cross_run_only: bool = True,
    mask_diagonal: bool = True,
) -> np.ndarray:
    """
    让 trial 热图显示逻辑尽量与 6×6 block RSA 一致：
    - 去掉 self-pair（对角线）
    - 若 cross_run_only=True，则去掉同 run pair
    """
    disp = sorted_mat.astype(np.float64, copy=True)
    if mask_diagonal:
        np.fill_diagonal(disp, np.nan)
    if cross_run_only and (sorted_runs is not None):
        runs_arr = np.asarray(sorted_runs)
        same_run_mask = runs_arr[:, None] == runs_arr[None, :]
        disp[same_run_mask] = np.nan
    return disp


def compute_trial_heatmap_range(display_mat: np.ndarray) -> Tuple[float, float]:
    vals = display_mat[np.isfinite(display_mat)]
    if vals.size == 0:
        return TRIAL_HEATMAP_VMIN, TRIAL_HEATMAP_VMAX
    if TRIAL_HEATMAP_AUTO_RANGE:
        lo = float(np.quantile(vals, TRIAL_HEATMAP_QLOW))
        hi = float(np.quantile(vals, TRIAL_HEATMAP_QHIGH))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi):
            return TRIAL_HEATMAP_VMIN, TRIAL_HEATMAP_VMAX
        return lo, hi
    return TRIAL_HEATMAP_VMIN, TRIAL_HEATMAP_VMAX


def plot_sorted_trial_heatmap(
    sim_mat: np.ndarray,
    sorted_labels: List[str],
    roi_name: str,
    out_path: Path,
    title_prefix: str,
):
    """
    绘制按类别排序后的 trial×trial 相似性热图。
    sim_mat 可以包含 NaN（例如 diagonal / same-run 被 mask）。
    """
    vmin, vmax = compute_trial_heatmap_range(sim_mat)

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    disp = np.ma.masked_invalid(sim_mat)
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)

    boundaries, centers, tick_labels = get_condition_block_info(sorted_labels)
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1.3, alpha=0.95)
        ax.axvline(b - 0.5, color="white", linewidth=1.3, alpha=0.95)

    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_title(
        f"{title_prefix}：{roi_name} trial×trial 相似性（按类别排序）\n"
        f"display range = [{vmin:.2f}, {vmax:.2f}]",
        fontsize=12
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相似性 (r)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"已保存 trial 热图: {out_path}")


def collect_trial_heatmap_record(
    subject: str,
    roi_name: str,
    sorted_mat: np.ndarray,
    sorted_labels: List[str],
    sorted_item_keys: List[str],
    sorted_runs: List[int] | None,
) -> Dict:
    return {
        "subject": subject,
        "roi": roi_name,
        "matrix": sorted_mat.astype(np.float32, copy=False),
        "sorted_labels": list(sorted_labels),
        "sorted_item_keys": list(sorted_item_keys),
        "sorted_runs": list(sorted_runs) if sorted_runs is not None else None,
    }


def plot_group_sorted_trial_heatmaps(trial_records: List[Dict]):
    """
    组平均排序 trial 热图。
    为增强可读性并与 6×6 block 逻辑一致：
    - diagonal 置为 NaN
    - 若 CROSS_RUN_ONLY=True，则同 run pair 置为 NaN
    - 组平均时对每个 cell 先 Fisher z 再平均，再反变换回 r
    """
    if not trial_records:
        return

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    roi_names = sorted({x["roi"] for x in trial_records})
    for roi_name in roi_names:
        recs = [x for x in trial_records if x["roi"] == roi_name]
        if not recs:
            continue

        base_shape = recs[0]["matrix"].shape
        base_labels = recs[0]["sorted_labels"]
        base_items = recs[0]["sorted_item_keys"]
        base_runs = recs[0].get("sorted_runs", None)

        valid_disp_mats = []
        for rec in recs:
            if (
                rec["matrix"].shape == base_shape
                and rec["sorted_labels"] == base_labels
                and rec["sorted_item_keys"] == base_items
                and rec.get("sorted_runs", None) == base_runs
            ):
                disp_mat = build_trial_heatmap_display_matrix(
                    rec["matrix"],
                    rec.get("sorted_runs", None),
                    cross_run_only=(CROSS_RUN_ONLY and TRIAL_HEATMAP_MASK_SAME_RUN),
                    mask_diagonal=TRIAL_HEATMAP_MASK_DIAGONAL,
                )
                valid_disp_mats.append(disp_mat)

        if not valid_disp_mats:
            logger.warning(f"{roi_name}: 无法生成组平均 trial 热图（shape / labels / item_keys / runs 不一致）")
            continue

        mats = np.stack(valid_disp_mats, axis=0).astype(np.float64)
        mats_z = fisher_z_clip(mats)
        mean_z = np.nanmean(mats_z, axis=0)
        mean_mat = np.tanh(mean_z)

        logger.info(f"{roi_name}: group trial heatmap 纳入被试数 = {len(valid_disp_mats)}")

        out_path = fig_dir / f"heatmap_trials_group_{roi_name}.png"
        plot_sorted_trial_heatmap(mean_mat, base_labels, roi_name, out_path, title_prefix="组平均")


def extract_subject_pair_values(long_df: pd.DataFrame, tests: List[Dict]) -> pd.DataFrame:
    """
    提取每个 ROI、每个被试、每个检验下的 left/right 原始值。
    输出列：
        subject, roi, test, left_pair, right_pair, left_r, right_r, left_z, right_z, left_n_pairs, right_n_pairs, diff_r, diff_z
    """
    rows = []
    for roi_name in sorted(long_df["roi"].unique()):
        roi_df = long_df[long_df["roi"] == roi_name].copy()

        for test in tests:
            la, lb = test["left"]
            ra, rb = test["right"]

            left_df = roi_df[(roi_df["cond_a"] == la) & (roi_df["cond_b"] == lb)][
                ["subject", "similarity", "similarity_z", "n_pairs"]
            ].rename(columns={
                "similarity": "left_r",
                "similarity_z": "left_z",
                "n_pairs": "left_n_pairs",
            })

            right_df = roi_df[(roi_df["cond_a"] == ra) & (roi_df["cond_b"] == rb)][
                ["subject", "similarity", "similarity_z", "n_pairs"]
            ].rename(columns={
                "similarity": "right_r",
                "similarity_z": "right_z",
                "n_pairs": "right_n_pairs",
            })

            merged = left_df.merge(right_df, on="subject", how="inner")
            if len(merged) == 0:
                continue

            merged["roi"] = roi_name
            merged["test"] = test["name"]
            merged["left_pair"] = f"{la}-{lb}"
            merged["right_pair"] = f"{ra}-{rb}"
            merged["diff_r"] = merged["left_r"] - merged["right_r"]
            merged["diff_z"] = merged["left_z"] - merged["right_z"]
            rows.append(merged[
                ["subject", "roi", "test", "left_pair", "right_pair",
                 "left_r", "right_r", "left_z", "right_z",
                 "left_n_pairs", "right_n_pairs", "diff_r", "diff_z"]
            ])

    if not rows:
        return pd.DataFrame(columns=[
            "subject", "roi", "test", "left_pair", "right_pair",
            "left_r", "right_r", "left_z", "right_z",
            "left_n_pairs", "right_n_pairs", "diff_r", "diff_z"
        ])
    return pd.concat(rows, axis=0, ignore_index=True)

def build_category_similarity_tables(
    sim_mat: np.ndarray,
    labels: List[str],
    runs: List[int],
    subject: str,
    roi_name: str,
    cross_run_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    matrix_rows = []

    for ca in COND_ORDER:
        row = {"subject": subject, "roi": roi_name, "cond_row": ca}
        for cb in COND_ORDER:
            sim_r, sim_z, n_pairs = block_mean_similarity(
                sim_mat, labels, runs, ca, cb, cross_run_only=cross_run_only
            )
            row[cb] = sim_r
            long_rows.append({
                "subject": subject,
                "roi": roi_name,
                "cond_a": ca,
                "cond_b": cb,
                "similarity": sim_r,
                "similarity_z": sim_z,
                "n_pairs": n_pairs,
            })
        matrix_rows.append(row)

    return pd.DataFrame(long_rows), pd.DataFrame(matrix_rows)


def matrix_rows_to_square(matrix_df_subroi: pd.DataFrame) -> np.ndarray:
    tmp = matrix_df_subroi.set_index("cond_row").loc[COND_ORDER, COND_ORDER]
    return tmp.to_numpy(dtype=float)


def build_model_similarity_matrix(pairs_high: List[Tuple[str, str]]) -> np.ndarray:
    n = len(COND_ORDER)
    mat = np.zeros((n, n), dtype=float)
    np.fill_diagonal(mat, 1.0)
    idx = {c: i for i, c in enumerate(COND_ORDER)}
    for a, b in pairs_high:
        i, j = idx[a], idx[b]
        mat[i, j] = 1.0
        mat[j, i] = 1.0
    return mat


MODEL_MATRICES = {name: build_model_similarity_matrix(pairs) for name, pairs in MODEL_SPECS.items()}


# =========================
# 6. 单被试处理
# =========================
def process_subject(sub: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    trial_df = load_subject_trials(sub)
    beta_files = trial_df["beta_path"].tolist()
    labels = trial_df["cond"].tolist()
    item_keys = trial_df["item_key"].tolist()
    runs = trial_df["run"].tolist()

    long_tables = []
    matrix_tables = []
    trial_records = []

    for roi_name, roi_mask_path in ROI_MASKS.items():
        if not roi_mask_path.exists():
            raise FileNotFoundError(f"ROI mask 不存在: {roi_mask_path}")

        patterns = extract_patterns(beta_files, roi_mask_path)
        sim_mat = compute_similarity_matrix(patterns)

        # 1) 常规 6×6 类别矩阵
        long_df, matrix_df = build_category_similarity_tables(
            sim_mat, labels, runs, sub, roi_name, cross_run_only=CROSS_RUN_ONLY
        )
        long_tables.append(long_df)
        matrix_tables.append(matrix_df)

        # 2) 按类别排序的 trial×trial similarity matrix
        sorted_mat, sorted_labels, sorted_item_keys, sorted_runs, _ = sort_trial_similarity_matrix(
            sim_mat, labels, item_keys=item_keys, runs=runs
        )
        trial_records.append(
            collect_trial_heatmap_record(sub, roi_name, sorted_mat, sorted_labels, sorted_item_keys, sorted_runs)
        )

        if SAVE_SUBJECT_TRIAL_HEATMAPS:
            disp_mat = build_trial_heatmap_display_matrix(
                sorted_mat,
                sorted_runs,
                cross_run_only=(CROSS_RUN_ONLY and TRIAL_HEATMAP_MASK_SAME_RUN),
                mask_diagonal=TRIAL_HEATMAP_MASK_DIAGONAL,
            )
            out_path = OUTPUT_DIR / "figures" / "trial_heatmaps_subject" / roi_name / f"heatmap_trials_{sub}_{roi_name}.png"
            plot_sorted_trial_heatmap(disp_mat, sorted_labels, roi_name, out_path, title_prefix=sub)

        del patterns, sim_mat, long_df, matrix_df, sorted_mat
        gc.collect()
        GPU.cleanup()

    return (
        pd.concat(long_tables, axis=0, ignore_index=True),
        pd.concat(matrix_tables, axis=0, ignore_index=True),
        trial_records,
    )


# =========================
# 7. 统计检验
# =========================
def run_pairwise_tests(long_df: pd.DataFrame, tests: List[Dict], fdr_mode: str) -> pd.DataFrame:
    rows = []

    for roi_name in sorted(long_df["roi"].unique()):
        roi_df = long_df[long_df["roi"] == roi_name].copy()

        for test in tests:
            la, lb = test["left"]
            ra, rb = test["right"]

            left_df = roi_df[(roi_df["cond_a"] == la) & (roi_df["cond_b"] == lb)][["subject", "similarity", "similarity_z"]]
            right_df = roi_df[(roi_df["cond_a"] == ra) & (roi_df["cond_b"] == rb)][["subject", "similarity", "similarity_z"]]
            merged = left_df.merge(right_df, on="subject", suffixes=("_left", "_right"))

            if len(merged) < 2:
                rows.append({
                    "roi": roi_name,
                    "test": test["name"],
                    "left_pair": f"{la}-{lb}",
                    "right_pair": f"{ra}-{rb}",
                    "n": len(merged),
                    "mean_left_r": np.nan,
                    "mean_right_r": np.nan,
                    "mean_left_z": np.nan,
                    "mean_right_z": np.nan,
                    "dz": np.nan,
                    "t": np.nan,
                    "p_two_sided": np.nan,
                    "p_one_sided": np.nan,
                    "direction": test["alternative"],
                })
                continue

            x = merged["similarity_z_left"].to_numpy(dtype=float)
            y = merged["similarity_z_right"].to_numpy(dtype=float)
            t_stat, p_two = ttest_rel(x, y, nan_policy="omit")
            p_one = one_sided_p_from_ttest(t_stat, p_two, alternative=test["alternative"])
            dz = cohen_dz_paired(x, y)

            rows.append({
                "roi": roi_name,
                "test": test["name"],
                "left_pair": f"{la}-{lb}",
                "right_pair": f"{ra}-{rb}",
                "n": len(merged),
                "mean_left_r": merged["similarity_left"].mean(),
                "mean_right_r": merged["similarity_right"].mean(),
                "mean_left_z": merged["similarity_z_left"].mean(),
                "mean_right_z": merged["similarity_z_right"].mean(),
                "dz": dz,
                "t": t_stat,
                "p_two_sided": p_two,
                "p_one_sided": p_one,
                "direction": test["alternative"],
            })

    stats_df = pd.DataFrame(rows)
    if len(stats_df) > 0:
        stats_df = apply_fdr(stats_df, mode=fdr_mode)
    return stats_df


def run_model_rdm_tests(matrix_all: pd.DataFrame, fdr_mode: str = "global") -> Tuple[pd.DataFrame, pd.DataFrame]:
    subj_rows = []
    group_rows = []
    iu = np.triu_indices(len(COND_ORDER), k=1)

    for roi_name in sorted(matrix_all["roi"].unique()):
        roi_df = matrix_all[matrix_all["roi"] == roi_name].copy()

        for sub in sorted(roi_df["subject"].unique()):
            sub_df = roi_df[roi_df["subject"] == sub].copy()
            obs_mat = matrix_rows_to_square(sub_df)
            obs_vec = obs_mat[iu]

            for model_name, model_mat in MODEL_MATRICES.items():
                model_vec = model_mat[iu]
                rho, _ = spearmanr(obs_vec, model_vec)
                rho = float(rho) if np.isfinite(rho) else np.nan

                subj_rows.append({
                    "subject": sub,
                    "roi": roi_name,
                    "model": model_name,
                    "rho": rho,
                    "rho_z": fisher_z_clip(np.array([rho]))[0] if np.isfinite(rho) else np.nan,
                    "model_pairs": str(MODEL_SPECS[model_name]),
                })

    subj_df = pd.DataFrame(subj_rows)

    for roi_name in sorted(subj_df["roi"].unique()):
        roi_df = subj_df[subj_df["roi"] == roi_name]
        for model_name in sorted(subj_df["model"].unique()):
            tmp = roi_df[roi_df["model"] == model_name].copy()
            x = tmp["rho_z"].dropna().to_numpy(dtype=float)

            if len(x) < 2:
                group_rows.append({
                    "roi": roi_name,
                    "test": model_name,
                    "model": model_name,
                    "n": len(x),
                    "mean_rho": np.nan,
                    "mean_rho_z": np.nan,
                    "d": np.nan,
                    "t": np.nan,
                    "p_two_sided": np.nan,
                    "p_one_sided": np.nan,
                    "direction": "greater",
                })
                continue

            t_stat, p_two = ttest_1samp(x, popmean=0.0, nan_policy="omit")
            p_one = one_sided_p_from_ttest(t_stat, p_two, alternative="greater")
            d = cohen_d_onesample(x)

            group_rows.append({
                "roi": roi_name,
                "test": model_name,
                "model": model_name,
                "n": len(x),
                "mean_rho": tmp["rho"].mean(),
                "mean_rho_z": tmp["rho_z"].mean(),
                "d": d,
                "t": t_stat,
                "p_two_sided": p_two,
                "p_one_sided": p_one,
                "direction": "greater",
            })

    group_df = pd.DataFrame(group_rows)
    if len(group_df) > 0:
        group_df = apply_fdr(group_df, mode=fdr_mode)
    return subj_df, group_df



def normalize_subject_id(x) -> str:
    """统一被试编号格式，尽量转为 3 位数字字符串，如 4 -> 004, sub-004 -> 004, 101 -> 101。"""
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    num = int(m.group(1))
    return f"{num:03d}"


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(f"找不到候选列 {candidates}；现有列为: {list(df.columns)}")


def load_fertility_score_table(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"生育意愿得分文件不存在: {filepath}")
    if filepath.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, sheet_name=FERTILITY_SCORE_SHEET)
    else:
        df = pd.read_csv(filepath)

    sub_col = find_first_existing_column(df, FERTILITY_SUBJECT_COL_CANDIDATES)
    score_col = find_first_existing_column(df, FERTILITY_SCORE_COL_CANDIDATES)

    out = df[[sub_col, score_col]].copy()
    out.columns = ["subject_raw", "fertility_score"]
    out["subject_norm"] = out["subject_raw"].apply(normalize_subject_id)
    out["fertility_score"] = pd.to_numeric(out["fertility_score"], errors="coerce")
    out = out.dropna(subset=["fertility_score"]).drop_duplicates(subset=["subject_norm"], keep="last")
    return out[["subject_norm", "fertility_score"]]


def plot_hypothesis_violin(subject_hypo_df: pd.DataFrame, hypo_df: pd.DataFrame):
    """
    每个 ROI 输出一张图，每个核心假设一个子图。
    新增：
    - 显著性标记（优先用 q_fdr_one_sided，否则用 p_one_sided）
    - 图上直接显示 q/p/t
    """
    fig_dir = OUTPUT_DIR / "figures" / "violin"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in sorted(subject_hypo_df["roi"].unique()):
        roi_df = subject_hypo_df[subject_hypo_df["roi"] == roi_name].copy()
        tests = list(roi_df["test"].unique())
        if len(tests) == 0:
            continue

        fig, axes = plt.subplots(1, len(tests), figsize=(6.6 * len(tests), 5.6), sharey=True)
        if len(tests) == 1:
            axes = [axes]

        for ax, test_name in zip(axes, tests):
            sub_df = roi_df[roi_df["test"] == test_name].copy()
            if len(sub_df) == 0:
                continue

            left_vals = sub_df["left_r"].to_numpy(dtype=float)
            right_vals = sub_df["right_r"].to_numpy(dtype=float)
            positions = [0, 1]

            parts = ax.violinplot([left_vals, right_vals], positions=positions, widths=0.72, showmeans=False, showmedians=True)
            for i, body in enumerate(parts["bodies"]):
                body.set_alpha(0.45)
                body.set_facecolor(["#d62728", "#1f77b4"][i])
                body.set_edgecolor("black")
                body.set_linewidth(0.8)
            for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.8)

            for _, row in sub_df.iterrows():
                ax.plot(positions, [row["left_r"], row["right_r"]], color="gray", alpha=0.22, linewidth=0.9, zorder=1)
                ax.scatter(positions, [row["left_r"], row["right_r"]], color=["#d62728", "#1f77b4"], s=14, alpha=0.75, zorder=2)

            mean_left = np.nanmean(left_vals)
            mean_right = np.nanmean(right_vals)
            ax.plot(positions, [mean_left, mean_right], color="black", linewidth=2.0, zorder=3)
            ax.scatter(positions, [mean_left, mean_right], color="black", s=35, zorder=4)

            left_pair = sub_df["left_pair"].iloc[0]
            right_pair = sub_df["right_pair"].iloc[0]
            ax.set_xticks(positions)
            ax.set_xticklabels([left_pair, right_pair], rotation=20)
            ax.set_title(test_name, fontsize=11)
            ax.set_ylabel("相似性 (r)")
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

            stat_row = hypo_df[(hypo_df["roi"] == roi_name) & (hypo_df["test"] == test_name)]
            if len(stat_row) > 0:
                stat_row = stat_row.iloc[0]
                q = stat_row.get("q_fdr_one_sided", np.nan)
                p = stat_row.get("p_one_sided", np.nan)
                t = stat_row.get("t", np.nan)
                label_p = q if np.isfinite(q) else p
                stars = p_to_star(label_p)

                ymax = np.nanmax([np.nanmax(left_vals), np.nanmax(right_vals)])
                ymin = np.nanmin([np.nanmin(left_vals), np.nanmin(right_vals)])
                yspan = max(0.02, ymax - ymin)
                y = ymax + 0.12 * yspan
                h = 0.04 * yspan
                ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="black", linewidth=1.1)
                ax.text(0.5, y + h + 0.01 * yspan, stars, ha="center", va="bottom", fontsize=13)
                txt = f"t={t:.2f}\np={p:.3f}"
                if np.isfinite(q):
                    txt += f"\nq={q:.3f}"
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

        fig.suptitle(f"{roi_name}：核心假设 left/right 分布（小提琴图）", y=1.02, fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / f"violin_hypothesis_{roi_name}.png"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存小提琴图: {out_path}")


def run_fertility_score_modulation(subject_hypo_df: pd.DataFrame, fdr_mode: str = "by_test") -> pd.DataFrame:
    score_df = load_fertility_score_table(FERTILITY_SCORE_FILE)

    tmp = subject_hypo_df.copy()
    tmp["subject_norm"] = tmp["subject"].apply(normalize_subject_id)
    merged = tmp.merge(score_df, on="subject_norm", how="inner")
    if len(merged) == 0:
        raise RuntimeError("生育意愿表与 RSA 被试层结果没有成功匹配到任何被试。")

    rows = []
    for roi_name in sorted(merged["roi"].unique()):
        roi_df = merged[merged["roi"] == roi_name].copy()
        for test_name in sorted(merged["test"].unique()):
            sub_df = roi_df[roi_df["test"] == test_name].copy()
            if len(sub_df) < 4:
                continue

            x = sub_df["fertility_score"].to_numpy(dtype=float)
            y = sub_df["diff_z"].to_numpy(dtype=float)
            if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
                pear_r, pear_p = np.nan, np.nan
                spear_r, spear_p = np.nan, np.nan
                slope, intercept, r_val, p_lin, stderr = np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                pear_r, pear_p = pearsonr(x, y)
                spear_r, spear_p = spearmanr(x, y)
                slope, intercept, r_val, p_lin, stderr = linregress(x, y)

            rows.append({
                "roi": roi_name,
                "test": test_name,
                "n": len(sub_df),
                "left_pair": sub_df["left_pair"].iloc[0],
                "right_pair": sub_df["right_pair"].iloc[0],
                "mean_diff_r": sub_df["diff_r"].mean(),
                "mean_diff_z": sub_df["diff_z"].mean(),
                "pearson_r": pear_r,
                "pearson_p_two_sided": pear_p,
                "spearman_rho": spear_r,
                "spearman_p_two_sided": spear_p,
                "lin_slope": slope,
                "lin_intercept": intercept,
                "lin_r": r_val,
                "lin_p_two_sided": p_lin,
                "lin_stderr": stderr,
            })

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        out_df["p_one_sided"] = np.where(
            out_df["lin_slope"] > 0,
            out_df["lin_p_two_sided"] / 2.0,
            1 - out_df["lin_p_two_sided"] / 2.0,
        )
        out_df = apply_fdr(out_df, mode=fdr_mode, pcol_one="p_one_sided", pcol_two="lin_p_two_sided")
    return out_df


def plot_fertility_score_scatter(subject_hypo_df: pd.DataFrame):
    score_df = load_fertility_score_table(FERTILITY_SCORE_FILE)
    tmp = subject_hypo_df.copy()
    tmp["subject_norm"] = tmp["subject"].apply(normalize_subject_id)
    merged = tmp.merge(score_df, on="subject_norm", how="inner")

    fig_dir = OUTPUT_DIR / "figures" / "fertility_score"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in sorted(merged["roi"].unique()):
        roi_df = merged[merged["roi"] == roi_name].copy()
        tests = list(roi_df["test"].unique())
        if len(tests) == 0:
            continue

        fig, axes = plt.subplots(1, len(tests), figsize=(6.0 * len(tests), 5.0), sharey=True)
        if len(tests) == 1:
            axes = [axes]

        for ax, test_name in zip(axes, tests):
            sub_df = roi_df[roi_df["test"] == test_name].copy()
            x = sub_df["fertility_score"].to_numpy(dtype=float)
            y = sub_df["diff_z"].to_numpy(dtype=float)
            ax.scatter(x, y, s=28, alpha=0.85)
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

            if len(sub_df) >= 2 and np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0:
                slope, intercept, r_val, p_val, stderr = linregress(x, y)
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                yy = intercept + slope * xx
                ax.plot(xx, yy, linewidth=2.0)
                ax.text(0.03, 0.97, f"r={r_val:.2f}, p={p_val:.3f}", transform=ax.transAxes, ha="left", va="top")

            ax.set_xlabel("生育意愿得分")
            ax.set_ylabel("核心假设差值 (diff_z)")
            ax.set_title(test_name)

        fig.suptitle(f"{roi_name}：生育意愿连续得分与核心假设差值", y=1.02, fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / f"fertility_score_scatter_{roi_name}.png"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存生育意愿散点图: {out_path}")


# =========================
# 8. 可视化
# =========================
def plot_group_heatmaps(matrix_df: pd.DataFrame):
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in sorted(matrix_df["roi"].unique()):
        roi_df = matrix_df[matrix_df["roi"] == roi_name].copy()
        if len(roi_df) == 0:
            continue

        mean_mat = roi_df.groupby("cond_row")[COND_ORDER].mean().loc[COND_ORDER, COND_ORDER]

        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        im = ax.imshow(mean_mat.values, cmap=HEATMAP_CMAP, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABELS[x] for x in COND_ORDER], rotation=45, ha="right")
        ax.set_yticklabels([COND_LABELS[x] for x in COND_ORDER])
        ax.set_title(f"{roi_name}：组平均类别相似性矩阵", fontsize=13)

        for i in range(len(COND_ORDER)):
            for j in range(len(COND_ORDER)):
                val = mean_mat.values[i, j]
                color = "white" if val < 0.2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("相似性 (r)")
        plt.tight_layout()
        out_path = fig_dir / f"heatmap_{roi_name}.png"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存热图: {out_path}")


def plot_target_pairs(long_df: pd.DataFrame, tests: List[Dict], stats_df: pd.DataFrame, out_name_prefix: str, per_page: int = 4):
    """
    将原本拥挤的合图拆成分页图，并在每个子图上标注 t/p/q。
    """
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in sorted(long_df["roi"].unique()):
        roi_df = long_df[long_df["roi"] == roi_name].copy()

        data_rows = []
        for test in tests:
            la, lb = test["left"]
            ra, rb = test["right"]
            left = roi_df[(roi_df["cond_a"] == la) & (roi_df["cond_b"] == lb)][["subject", "similarity_z"]].rename(columns={"similarity_z": "left"})
            right = roi_df[(roi_df["cond_a"] == ra) & (roi_df["cond_b"] == rb)][["subject", "similarity_z"]].rename(columns={"similarity_z": "right"})
            merged = left.merge(right, on="subject")
            merged["test"] = test["name"]
            data_rows.append(merged)

        if not data_rows:
            continue
        plot_df = pd.concat(data_rows, axis=0, ignore_index=True)
        test_names = [t["name"] for t in tests]
        pages = [test_names[i:i + per_page] for i in range(0, len(test_names), per_page)]

        for page_idx, page_tests in enumerate(pages, start=1):
            n_panels = len(page_tests)
            fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8), sharey=True)
            if n_panels == 1:
                axes = [axes]

            for ax, test_name in zip(axes, page_tests):
                sub_df = plot_df[plot_df["test"] == test_name].copy()
                if len(sub_df) == 0:
                    continue
                x = np.array([0, 1])
                for _, row in sub_df.iterrows():
                    ax.plot(x, [row["left"], row["right"]], color="gray", alpha=0.28, linewidth=1)
                    ax.scatter(x, [row["left"], row["right"]], color=["#d62728", "#1f77b4"], s=18, zorder=2)

                mean_left = sub_df["left"].mean()
                mean_right = sub_df["right"].mean()
                ax.plot(x, [mean_left, mean_right], color="black", linewidth=2.4)
                ax.scatter(x, [mean_left, mean_right], color="black", s=38, zorder=3)

                stat_row = stats_df[(stats_df["roi"] == roi_name) & (stats_df["test"] == test_name)]
                left_pair = right_pair = ""
                if len(stat_row) > 0:
                    stat_row = stat_row.iloc[0]
                    left_pair = stat_row["left_pair"]
                    right_pair = stat_row["right_pair"]
                    q = stat_row.get("q_fdr_one_sided", np.nan)
                    p = stat_row.get("p_one_sided", np.nan)
                    t = stat_row.get("t", np.nan)
                    txt = f"t={t:.2f}\np={p:.3f}"
                    if np.isfinite(q):
                        txt += f"\nq={q:.3f}"
                    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

                ax.set_xticks([0, 1])
                ax.set_xticklabels([left_pair, right_pair], rotation=20)
                ax.set_title(test_name, fontsize=11)
                ax.set_ylabel("相似性 (Fisher z)")
                ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

            fig.suptitle(f"{roi_name}：{out_name_prefix}（第{page_idx}页）", y=1.02, fontsize=13)
            plt.tight_layout()
            out_path = fig_dir / f"{out_name_prefix}_{roi_name}_page{page_idx}.png"
            plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"已保存目标对图: {out_path}")


def plot_model_fit_bars(model_subj_df: pd.DataFrame, model_group_df: pd.DataFrame):
    """
    更可解释的 model fit 图：
    - 显示每个模型的被试层散点
    - 显示 mean rho 与 95% CI
    - 标注 q/p
    """
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_name_map = {
        "M1_fer_exi_only": "M1: 生育≈生存",
        "M2_atf_hed_only": "M2: 不生育≈享乐",
        "M3_combined": "M3: 组合模型",
    }

    for roi_name in sorted(model_group_df["roi"].unique()):
        roi_group = model_group_df[model_group_df["roi"] == roi_name].copy()
        roi_subj = model_subj_df[model_subj_df["roi"] == roi_name].copy()
        if len(roi_group) == 0:
            continue

        order = roi_group.sort_values("mean_rho", ascending=False)["model"].tolist()
        x = np.arange(len(order))

        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        for i, model_name in enumerate(order):
            subj_vals = roi_subj[roi_subj["model"] == model_name]["rho"].dropna().to_numpy(dtype=float)
            if len(subj_vals) > 0:
                jitter = np.random.uniform(-0.08, 0.08, size=len(subj_vals))
                ax.scatter(np.full(len(subj_vals), i) + jitter, subj_vals, s=22, alpha=0.35, color="gray", zorder=1)

            grp_row = roi_group[roi_group["model"] == model_name].iloc[0]
            mean_rho = grp_row["mean_rho"]
            if len(subj_vals) >= 2:
                se = np.nanstd(subj_vals, ddof=1) / np.sqrt(len(subj_vals))
                ci = 1.96 * se
            else:
                ci = np.nan

            ax.scatter([i], [mean_rho], color="black", s=55, zorder=3)
            if np.isfinite(ci):
                ax.plot([i, i], [mean_rho - ci, mean_rho + ci], color="black", linewidth=1.4, zorder=2)

            q = grp_row.get("q_fdr_one_sided", np.nan)
            p = grp_row.get("p_one_sided", np.nan)
            stars = p_to_star(q if np.isfinite(q) else p)
            label = f"{stars}\nq={q:.3f}" if np.isfinite(q) else f"{stars}\np={p:.3f}"
            ax.text(i, mean_rho + 0.03, label, ha="center", va="bottom", fontsize=9)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([model_name_map.get(m, m) for m in order], rotation=15)
        ax.set_ylabel("模型拟合（Spearman rho）")
        best_model = order[0] if order else "NA"
        ax.set_title(f"{roi_name}：模型拟合比较\n最佳模型 = {model_name_map.get(best_model, best_model)}", fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / f"model_fit_{roi_name}.png"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存模型拟合图: {out_path}")


# =========================
# 9. 主程序
# =========================
def main():
    configure_matplotlib_chinese()

    logger.info("=== 开始实验2 ROI-RSA v4.5 优化版分析 ===")
    logger.info(f"LSS_ROOT = {LSS_ROOT}")
    logger.info(f"OUTPUT_DIR = {OUTPUT_DIR}")
    logger.info(f"GPU backend = {GPU.backend}")
    logger.info(f"CROSS_RUN_ONLY = {CROSS_RUN_ONLY}")
    logger.info(f"SAVE_GROUP_TRIAL_HEATMAPS = {SAVE_GROUP_TRIAL_HEATMAPS}")
    logger.info(f"SAVE_SUBJECT_TRIAL_HEATMAPS = {SAVE_SUBJECT_TRIAL_HEATMAPS}")
    logger.info(f"HEATMAP range = [{HEATMAP_VMIN}, {HEATMAP_VMAX}]")
    logger.info(f"TRIAL_HEATMAP range = [{TRIAL_HEATMAP_VMIN}, {TRIAL_HEATMAP_VMAX}]")
    logger.info(f"RUN_FERTILITY_SCORE_ANALYSIS = {RUN_FERTILITY_SCORE_ANALYSIS}")

    subjects = discover_subjects(LSS_ROOT)
    if not subjects:
        raise RuntimeError(f"在 {LSS_ROOT} 下没有找到 sub-* 文件夹")
    logger.info(f"发现被试数: {len(subjects)} -> {subjects}")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject)(sub) for sub in subjects
    )

    long_tables = []
    matrix_tables = []
    trial_records = []
    for long_df, matrix_df, recs in results:
        long_tables.append(long_df)
        matrix_tables.append(matrix_df)
        trial_records.extend(recs)

    long_all = pd.concat(long_tables, axis=0, ignore_index=True)
    matrix_all = pd.concat(matrix_tables, axis=0, ignore_index=True)

    out_long = OUTPUT_DIR / "rsa_block_similarity_long.csv"
    out_matrix = OUTPUT_DIR / "rsa_category_similarity_matrix_rows.csv"
    long_all.to_csv(out_long, index=False)
    matrix_all.to_csv(out_matrix, index=False)
    logger.info(f"已保存长表: {out_long}")
    logger.info(f"已保存矩阵行表: {out_matrix}")

    hypo_df = run_pairwise_tests(long_all, HYPOTHESIS_TESTS, fdr_mode=FDR_MODE_MAIN)
    out_hypo = OUTPUT_DIR / "rsa_hypothesis_tests.csv"
    hypo_df.to_csv(out_hypo, index=False)
    logger.info(f"已保存主假设检验: {out_hypo}")

    subject_hypo_df = extract_subject_pair_values(long_all, HYPOTHESIS_TESTS)
    out_subject_hypo = OUTPUT_DIR / "rsa_subjectwise_hypothesis_lr.csv"
    subject_hypo_df.to_csv(out_subject_hypo, index=False)
    logger.info(f"已保存被试层主假设 left/right 数值: {out_subject_hypo}")

    spec_df = run_pairwise_tests(long_all, SPECIFICITY_TESTS, fdr_mode=FDR_MODE_SPECIFICITY)
    out_spec = OUTPUT_DIR / "rsa_specificity_tests.csv"
    spec_df.to_csv(out_spec, index=False)
    logger.info(f"已保存特异性检验: {out_spec}")

    model_subj_df, model_group_df = run_model_rdm_tests(matrix_all, fdr_mode=FDR_MODE_MODEL)
    out_model_subj = OUTPUT_DIR / "rsa_model_rdm_subjectwise.csv"
    out_model_group = OUTPUT_DIR / "rsa_model_rdm_group_tests.csv"
    model_subj_df.to_csv(out_model_subj, index=False)
    model_group_df.to_csv(out_model_group, index=False)
    logger.info(f"已保存模型RDM-被试层结果: {out_model_subj}")
    logger.info(f"已保存模型RDM-组层统计: {out_model_group}")

    plot_group_heatmaps(matrix_all)
    if SAVE_GROUP_TRIAL_HEATMAPS:
        plot_group_sorted_trial_heatmaps(trial_records)
    plot_target_pairs(long_all, HYPOTHESIS_TESTS, hypo_df, out_name_prefix="target_pairs_main", per_page=PLOT_MAIN_PER_PAGE)
    plot_target_pairs(long_all, SPECIFICITY_TESTS, spec_df, out_name_prefix="target_pairs_specificity", per_page=PLOT_SPECIFICITY_PER_PAGE)
    plot_model_fit_bars(model_subj_df, model_group_df)
    plot_hypothesis_violin(subject_hypo_df, hypo_df)

    if RUN_FERTILITY_SCORE_ANALYSIS:
        try:
            fert_df = run_fertility_score_modulation(subject_hypo_df, fdr_mode="by_test")
            out_fert = OUTPUT_DIR / "rsa_fertility_score_modulation.csv"
            fert_df.to_csv(out_fert, index=False)
            logger.info(f"已保存生育意愿连续得分调节分析: {out_fert}")
            plot_fertility_score_scatter(subject_hypo_df)
        except Exception as e:
            logger.warning(f"生育意愿连续得分分析未执行成功: {e}")

    if RUN_FERTILITY_GROUP_VIS:
        try:
            fert_group_subject_df, fert_group_summary_df = summarize_fertility_groups(subject_hypo_df)
            out_group_subj = OUTPUT_DIR / "rsa_fertility_score_grouped_subjectwise.csv"
            out_group_summary = OUTPUT_DIR / "rsa_fertility_score_grouped_summary.csv"
            fert_group_subject_df.to_csv(out_group_subj, index=False)
            fert_group_summary_df.to_csv(out_group_summary, index=False)
            logger.info(f"已保存生育意愿分组被试层数据: {out_group_subj}")
            logger.info(f"已保存生育意愿分组汇总数据: {out_group_summary}")
            plot_fertility_group_violin(subject_hypo_df)
        except Exception as e:
            logger.warning(f"生育意愿低/中/高三组探索性可视化未执行成功: {e}")

    logger.info("=== RSA 增强版分析完成 ===")

def prepare_fertility_group_subjectwise(subject_hypo_df: pd.DataFrame) -> pd.DataFrame:
    score_df = load_fertility_score_table(FERTILITY_SCORE_FILE).copy()
    tmp = subject_hypo_df.copy()
    tmp["subject_norm"] = tmp["subject"].apply(normalize_subject_id)
    merged = tmp.merge(score_df, on="subject_norm", how="inner")
    if len(merged) == 0:
        raise RuntimeError("生育意愿表与 RSA 被试层结果没有成功匹配到任何被试。")

    if FERTILITY_GROUP_MODE == "fixed":
        def _map_group(x):
            if pd.isna(x):
                return np.nan
            if x <= 2:
                return "低生育意愿"
            elif x <= 5:
                return "中生育意愿"
            else:
                return "高生育意愿"
        merged["fertility_group"] = merged["fertility_score"].apply(_map_group)

    elif FERTILITY_GROUP_MODE == "tertile":
        ranked = merged["fertility_score"].rank(method="average")
        merged["fertility_group"] = pd.qcut(
            ranked,
            q=3,
            labels=FERTILITY_GROUP_ORDER,
            duplicates="drop"
        ).astype(str)
    else:
        raise ValueError(f"未知 FERTILITY_GROUP_MODE: {FERTILITY_GROUP_MODE}")

    merged["fertility_group"] = pd.Categorical(
        merged["fertility_group"],
        categories=FERTILITY_GROUP_ORDER,
        ordered=True
    )
    merged = merged.dropna(subset=["fertility_group"]).copy()
    return merged


def summarize_fertility_groups(subject_hypo_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = prepare_fertility_group_subjectwise(subject_hypo_df)

    summary_rows = []
    for (roi_name, test_name, grp), sub_df in merged.groupby(["roi", "test", "fertility_group"], observed=True):
        if len(sub_df) == 0:
            continue
        summary_rows.append({
            "roi": roi_name,
            "test": test_name,
            "fertility_group": str(grp),
            "n": len(sub_df),
            "left_pair": sub_df["left_pair"].iloc[0],
            "right_pair": sub_df["right_pair"].iloc[0],
            "mean_score": sub_df["fertility_score"].mean(),
            "min_score": sub_df["fertility_score"].min(),
            "max_score": sub_df["fertility_score"].max(),
            "mean_left_r": sub_df["left_r"].mean(),
            "mean_right_r": sub_df["right_r"].mean(),
            "mean_left_z": sub_df["left_z"].mean(),
            "mean_right_z": sub_df["right_z"].mean(),
            "mean_diff_r": sub_df["diff_r"].mean(),
            "mean_diff_z": sub_df["diff_z"].mean(),
            "sd_diff_z": sub_df["diff_z"].std(ddof=1) if len(sub_df) > 1 else np.nan,
        })

    return merged, pd.DataFrame(summary_rows)


def plot_fertility_group_violin(subject_hypo_df: pd.DataFrame):
    merged = prepare_fertility_group_subjectwise(subject_hypo_df)

    fig_dir = OUTPUT_DIR / "figures" / "fertility_groups"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "低生育意愿": "#4c72b0",
        "中生育意愿": "#55a868",
        "高生育意愿": "#c44e52",
    }

    for roi_name in sorted(merged["roi"].unique()):
        roi_df = merged[merged["roi"] == roi_name].copy()
        tests = list(roi_df["test"].unique())
        if len(tests) == 0:
            continue

        fig, axes = plt.subplots(1, len(tests), figsize=(6.4 * len(tests), 5.4), sharey=True)
        if len(tests) == 1:
            axes = [axes]

        for ax, test_name in zip(axes, tests):
            sub_df = roi_df[roi_df["test"] == test_name].copy()
            if len(sub_df) == 0:
                continue

            group_vals = []
            positions = []
            mean_points = []
            xticklabels = []

            for idx, grp in enumerate(FERTILITY_GROUP_ORDER):
                gdf = sub_df[sub_df["fertility_group"] == grp].copy()
                vals = gdf["diff_z"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = len(vals)
                xticklabels.append(f"{grp}\n(n={n})")
                positions.append(idx)

                if n >= 2:
                    group_vals.append(vals)
                else:
                    group_vals.append(np.array([], dtype=float))

                # scatter with jitter
                if n > 0:
                    jitter = np.random.uniform(-0.08, 0.08, size=n)
                    ax.scatter(
                        np.full(n, idx) + jitter,
                        vals,
                        s=22,
                        alpha=0.85,
                        color=palette.get(grp, "gray"),
                        zorder=3
                    )
                    mean_points.append((idx, float(np.mean(vals))))
                else:
                    mean_points.append((idx, np.nan))

            # violin only for groups with >=2 values
            for idx, vals in enumerate(group_vals):
                if len(vals) >= 2:
                    parts = ax.violinplot([vals], positions=[idx], widths=0.75, showmeans=False, showmedians=True)
                    for body in parts["bodies"]:
                        body.set_alpha(0.35)
                        body.set_facecolor(palette.get(FERTILITY_GROUP_ORDER[idx], "gray"))
                        body.set_edgecolor("black")
                        body.set_linewidth(0.8)
                    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
                        if key in parts:
                            parts[key].set_color("black")
                            parts[key].set_linewidth(0.8)

            # connect group means
            xs = [x for x, y in mean_points if np.isfinite(y)]
            ys = [y for x, y in mean_points if np.isfinite(y)]
            if len(xs) >= 2:
                ax.plot(xs, ys, color="black", linewidth=1.8, zorder=4)
            for x, y in mean_points:
                if np.isfinite(y):
                    ax.scatter([x], [y], color="black", s=35, zorder=5)

            left_pair = sub_df["left_pair"].iloc[0]
            right_pair = sub_df["right_pair"].iloc[0]
            ax.set_title(f"{test_name}\n{left_pair} vs {right_pair}", fontsize=11)
            ax.set_xticks(positions)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("生育意愿分组")
            ax.set_ylabel("核心假设差值 (diff_z)")
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

        fig.suptitle(f"{roi_name}：生育意愿低/中/高三组探索性可视化", y=1.02, fontsize=13)
        plt.tight_layout()
        out_path = fig_dir / f"fertility_group_violin_{roi_name}.png"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存生育意愿分组小提琴图: {out_path}")


if __name__ == "__main__":
    main()