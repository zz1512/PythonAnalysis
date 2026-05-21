#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rsa_exp2_threat_rois.py

实验2（威胁情境任务）配套 RSA 脚本
-------------------------------------------------
功能：
1. 自动读取 LSS 输出目录下各被试 run-4/5/6 的 trial_info.csv 与 beta 图
2. 在目标 ROI（ACC / Amygdala / Insula）内提取每个 trial 的多体素模式
3. 计算 6 类威胁（fer / soc / val / hed / exi / atf）的 trial-by-trial 相似性矩阵
4. 汇总为类别 × 类别的 block-wise 平均相似性矩阵（真正的 RSA block summary）
5. 输出每个被试、每个 ROI 的目标类别对相似性
6. 自动完成核心假设检验：
   - H1: sim(fer, exi) > sim(atf, exi)
   - H2: sim(atf, hed) > sim(fer, hed)
7. 保存热图、长表、统计表

说明：
- 推荐输入为 LSS beta 图（单 trial beta），且 beta 图已在同一空间（如 MNI）
- RSA 默认使用 Pearson correlation similarity
- 每个 trial 模式先做“跨体素 z 标准化”再计算相似性
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.maskers import NiftiMasker
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# =========================
# 1. 配置区（按你的电脑改路径）
# =========================
BASE_DIR = Path(r"C:/python_fertility")
LSS_ROOT = BASE_DIR / "lss_betas_exp2"
OUTPUT_DIR = BASE_DIR / "rsa_results_exp2"
RUNS = [4, 5, 6]
N_JOBS = 4

# ROI 掩膜路径：请按你的实际路径修改
ROI_MASKS = {
    "ACC": BASE_DIR / "roi_masks/ACC_mask.nii.gz",
    "Amygdala": BASE_DIR / "roi_masks/Amygdala_mask.nii.gz",
    "Insula": BASE_DIR / "roi_masks/Insula_mask.nii.gz",
}

# 六类 threat code 的固定顺序（与你当前数据一致）
COND_ORDER = ["fer", "soc", "val", "hed", "exi", "atf"]
COND_LABELS = {
    "fer": "生育威胁",
    "soc": "人际威胁",
    "val": "价值威胁",
    "hed": "享乐威胁",
    "exi": "生存威胁",
    "atf": "不生育威胁",
}

# 核心假设对比
HYPOTHESIS_TESTS = [
    {
        "name": "H1_生育≈生存",
        "left": ("fer", "exi"),
        "right": ("atf", "exi"),
        "alternative": "greater",  # left > right
    },
    {
        "name": "H2_不生育≈安享",
        "left": ("atf", "hed"),
        "right": ("fer", "hed"),
        "alternative": "greater",  # left > right
    },
]

# 相似性度量：当前固定为 correlation similarity
METRIC = "correlation"

# 日志
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RSA_EXP2")


# =========================
# 2. 工具函数
# =========================
def discover_subjects(lss_root: Path) -> List[str]:
    subs = sorted([p.name for p in lss_root.glob("sub-*") if p.is_dir()])
    return subs


def parse_condition_from_row(row: pd.Series) -> str:
    """尽可能稳健地从 metadata 中解析条件码。"""
    # 优先 condition
    for key in ["condition", "type", "trial_type"]:
        if key in row and pd.notna(row[key]):
            val = str(row[key]).strip().lower()
            if val in COND_ORDER:
                return val

    # 其次从 unique_label 提取，如 fer_023
    if "unique_label" in row and pd.notna(row["unique_label"]):
        val = str(row["unique_label"]).strip().lower()
        m = re.match(r"^([a-z]{3})[_-]", val)
        if m and m.group(1) in COND_ORDER:
            return m.group(1)

    # 再从 beta_file 提取，如 beta_trial-000_fer_023.nii.gz
    if "beta_file" in row and pd.notna(row["beta_file"]):
        val = Path(str(row["beta_file"])).name.lower()
        m = re.search(r"_([a-z]{3})[_-]", val)
        if m and m.group(1) in COND_ORDER:
            return m.group(1)

    raise ValueError(f"无法从该行解析条件码: {row.to_dict()}")


def rowwise_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对每个 trial 的 voxel pattern 做跨体素 z 标准化。"""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + eps)


def fisher_z_clip(r: np.ndarray, clip: float = 0.999999) -> np.ndarray:
    r = np.clip(r, -clip, clip)
    return np.arctanh(r)


def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR 校正，返回 q 值。"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()


def one_sided_p_from_ttest(t_stat: float, p_two_sided: float, alternative: str = "greater") -> float:
    """把 scipy 双侧配对 t 检验结果转成单侧 p。"""
    if np.isnan(t_stat) or np.isnan(p_two_sided):
        return np.nan

    if alternative == "greater":
        return p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    elif alternative == "less":
        return p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2
    else:
        return p_two_sided


# =========================
# 3. 数据读取
# =========================
def load_subject_trials(sub: str) -> pd.DataFrame:
    """
    读取某个被试 run-4/5/6 的 trial_info.csv，并拼接为总表。
    结果至少包含：subject, run, beta_path, cond
    """
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

        # 拼接 beta 绝对路径
        if "beta_file" not in df.columns:
            raise ValueError(f"{csv_path} 缺少 beta_file 列")
        df["beta_path"] = df["beta_file"].apply(lambda x: str((run_dir / Path(str(x)).name).resolve()))

        # 检查文件存在
        miss = df[~df["beta_path"].apply(lambda x: Path(x).exists())]
        if len(miss) > 0:
            miss_examples = miss["beta_path"].tolist()[:5]
            raise FileNotFoundError(f"{sub} run-{run} 有 beta 文件缺失，例如: {miss_examples}")

        dfs.append(df)

    if not dfs:
        raise ValueError(f"{sub} 没有可用的 run 数据")

    out = pd.concat(dfs, axis=0, ignore_index=True)

    # 建议检查 6 类是否齐全
    cond_counts = out["cond"].value_counts().to_dict()
    logger.info(f"{sub}: 各条件 trial 数 = {cond_counts}")
    return out


# =========================
# 4. RSA 核心
# =========================
def extract_patterns(beta_files: List[str], roi_mask_path: Path) -> np.ndarray:
    masker = NiftiMasker(mask_img=str(roi_mask_path), standardize=False)
    x = masker.fit_transform(beta_files)  # [n_trials, n_voxels]
    if x.ndim != 2:
        raise ValueError(f"提取到的 pattern 维度异常: {x.shape}")
    x = rowwise_zscore(x)
    return x


def compute_similarity_matrix(patterns: np.ndarray) -> np.ndarray:
    if METRIC != "correlation":
        raise NotImplementedError("当前脚本只实现 correlation similarity")
    dist_vec = pdist(patterns, metric="correlation")
    sim_mat = 1.0 - squareform(dist_vec)
    np.fill_diagonal(sim_mat, 1.0)
    return sim_mat


def block_mean_similarity(sim_mat: np.ndarray, labels: List[str], cond_a: str, cond_b: str) -> Tuple[float, int]:
    idx_a = [i for i, x in enumerate(labels) if x == cond_a]
    idx_b = [i for i, x in enumerate(labels) if x == cond_b]

    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan, 0

    block = sim_mat[np.ix_(idx_a, idx_b)]
    if cond_a == cond_b:
        # 去对角线
        if block.shape[0] < 2:
            return np.nan, 0
        mask = ~np.eye(block.shape[0], dtype=bool)
        vals = block[mask]
    else:
        vals = block.ravel()

    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, 0
    return float(np.mean(vals)), int(len(vals))


def build_category_similarity_table(sim_mat: np.ndarray, labels: List[str], subject: str, roi_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    matrix_rows = []

    for i, ca in enumerate(COND_ORDER):
        row = {"subject": subject, "roi": roi_name, "cond_row": ca}
        for cb in COND_ORDER:
            sim, n_pairs = block_mean_similarity(sim_mat, labels, ca, cb)
            row[cb] = sim
            long_rows.append({
                "subject": subject,
                "roi": roi_name,
                "cond_a": ca,
                "cond_b": cb,
                "similarity": sim,
                "similarity_z": fisher_z_clip(np.array([sim]))[0] if np.isfinite(sim) else np.nan,
                "n_pairs": n_pairs,
            })
        matrix_rows.append(row)

    return pd.DataFrame(long_rows), pd.DataFrame(matrix_rows)


# =========================
# 5. 单被试单 ROI 处理
# =========================
def process_subject_roi(sub: str, roi_name: str, roi_mask_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not roi_mask_path.exists():
        raise FileNotFoundError(f"ROI mask 不存在: {roi_mask_path}")

    trial_df = load_subject_trials(sub)

    # 固定顺序：run, trial_index（若存在）
    sort_cols = [c for c in ["run", "trial_index", "onset"] if c in trial_df.columns]
    if sort_cols:
        trial_df = trial_df.sort_values(sort_cols).reset_index(drop=True)

    beta_files = trial_df["beta_path"].tolist()
    labels = trial_df["cond"].tolist()

    patterns = extract_patterns(beta_files, roi_mask_path)
    sim_mat = compute_similarity_matrix(patterns)

    long_df, matrix_df = build_category_similarity_table(sim_mat, labels, sub, roi_name)
    return long_df, matrix_df


# =========================
# 6. 统计检验
# =========================
def apply_fdr(stats_df: pd.DataFrame, mode: str = "by_test") -> pd.DataFrame:
    """
    mode:
        - global: 所有检验一起校正
        - by_test: 按 hypothesis 分开校正（推荐）
        - by_roi: 按 ROI 分开校正
    """
    stats_df = stats_df.copy()

    if mode == "global":
        stats_df["q_fdr_one_sided"] = bh_fdr(stats_df["p_one_sided"].fillna(1.0).tolist())
        stats_df["q_fdr_two_sided"] = bh_fdr(stats_df["p_two_sided"].fillna(1.0).tolist())

    elif mode == "by_test":
        stats_df["q_fdr_one_sided"] = np.nan
        stats_df["q_fdr_two_sided"] = np.nan
        for test_name, subdf in stats_df.groupby("test"):
            idx = subdf.index
            stats_df.loc[idx, "q_fdr_one_sided"] = bh_fdr(subdf["p_one_sided"].fillna(1.0).tolist())
            stats_df.loc[idx, "q_fdr_two_sided"] = bh_fdr(subdf["p_two_sided"].fillna(1.0).tolist())

    elif mode == "by_roi":
        stats_df["q_fdr_one_sided"] = np.nan
        stats_df["q_fdr_two_sided"] = np.nan
        for roi_name, subdf in stats_df.groupby("roi"):
            idx = subdf.index
            stats_df.loc[idx, "q_fdr_one_sided"] = bh_fdr(subdf["p_one_sided"].fillna(1.0).tolist())
            stats_df.loc[idx, "q_fdr_two_sided"] = bh_fdr(subdf["p_two_sided"].fillna(1.0).tolist())

    else:
        raise ValueError(f"未知 FDR 模式: {mode}")

    return stats_df

def run_hypothesis_tests(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for roi_name in sorted(long_df["roi"].unique()):
        roi_df = long_df[long_df["roi"] == roi_name].copy()

        for test in HYPOTHESIS_TESTS:
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
                    "t": np.nan,
                    "p_two_sided": np.nan,
                    "p_one_sided": np.nan,
                    "direction": test["alternative"],
                })
                continue

            # 建议在 Fisher z 上做 t 检验
            t_stat, p_two = ttest_rel(merged["similarity_z_left"], merged["similarity_z_right"], nan_policy="omit")
            p_one = one_sided_p_from_ttest(t_stat, p_two, alternative=test["alternative"])

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
                "t": t_stat,
                "p_two_sided": p_two,
                "p_one_sided": p_one,
                "direction": test["alternative"],
            })

    stats_df = pd.DataFrame(rows)
    if len(stats_df) > 0:
        stats_df = apply_fdr(stats_df, mode="by_test")  # 推荐
    return stats_df


# =========================
# 7. 可视化
# =========================
def plot_group_heatmaps(matrix_df: pd.DataFrame):
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in sorted(matrix_df["roi"].unique()):
        roi_df = matrix_df[matrix_df["roi"] == roi_name].copy()
        if len(roi_df) == 0:
            continue

        # group 平均
        mean_mat = roi_df.groupby("cond_row")[COND_ORDER].mean().loc[COND_ORDER, COND_ORDER]

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        im = ax.imshow(mean_mat.values, cmap="coolwarm", vmin=-0.2, vmax=0.8)
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_xticklabels([COND_LABELS[x] for x in COND_ORDER], rotation=45, ha="right")
        ax.set_yticklabels([COND_LABELS[x] for x in COND_ORDER])
        ax.set_title(f"{roi_name}：组平均类别相似性矩阵")

        for i in range(len(COND_ORDER)):
            for j in range(len(COND_ORDER)):
                val = mean_mat.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Similarity (r)")
        plt.tight_layout()
        out_path = fig_dir / f"heatmap_{roi_name}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存热图: {out_path}")


def plot_target_pairs(long_df: pd.DataFrame):
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in sorted(long_df["roi"].unique()):
        roi_df = long_df[long_df["roi"] == roi_name].copy()
        sub_list = sorted(roi_df["subject"].unique())

        data_rows = []
        for test in HYPOTHESIS_TESTS:
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
        if len(plot_df) == 0:
            continue

        fig, axes = plt.subplots(1, len(HYPOTHESIS_TESTS), figsize=(10.5, 4.6), sharey=True)
        if len(HYPOTHESIS_TESTS) == 1:
            axes = [axes]

        for ax, test in zip(axes, HYPOTHESIS_TESTS):
            sub_df = plot_df[plot_df["test"] == test["name"]].copy()
            x = np.array([0, 1])
            for _, row in sub_df.iterrows():
                ax.plot(x, [row["left"], row["right"]], color="gray", alpha=0.35, linewidth=1)
                ax.scatter(x, [row["left"], row["right"]], color=["#d62728", "#1f77b4"], s=18)

            mean_left = sub_df["left"].mean()
            mean_right = sub_df["right"].mean()
            ax.plot(x, [mean_left, mean_right], color="black", linewidth=2.4)
            ax.scatter(x, [mean_left, mean_right], color="black", s=36, zorder=3)

            la, lb = test["left"]
            ra, rb = test["right"]
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"{la}-{lb}", f"{ra}-{rb}"], rotation=20)
            ax.set_title(test["name"])
            ax.set_ylabel("Similarity (Fisher z)")
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)

        fig.suptitle(f"{roi_name}：目标类别对相似性", y=1.02)
        plt.tight_layout()
        out_path = fig_dir / f"target_pairs_{roi_name}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"已保存目标对图: {out_path}")


# =========================
# 8. 主程序
# =========================
def main():
    logger.info("=== 开始实验2 ROI-RSA 分析 ===")
    logger.info(f"LSS_ROOT = {LSS_ROOT}")
    logger.info(f"OUTPUT_DIR = {OUTPUT_DIR}")

    subjects = discover_subjects(LSS_ROOT)
    if not subjects:
        raise RuntimeError(f"在 {LSS_ROOT} 下没有找到 sub-* 文件夹")
    logger.info(f"发现被试数: {len(subjects)} -> {subjects}")

    tasks = []
    for sub in subjects:
        for roi_name, roi_path in ROI_MASKS.items():
            tasks.append((sub, roi_name, roi_path))

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject_roi)(sub, roi_name, roi_path)
        for sub, roi_name, roi_path in tasks
    )

    long_tables = []
    matrix_tables = []
    for long_df, matrix_df in results:
        long_tables.append(long_df)
        matrix_tables.append(matrix_df)

    long_all = pd.concat(long_tables, axis=0, ignore_index=True)
    matrix_all = pd.concat(matrix_tables, axis=0, ignore_index=True)

    out_long = OUTPUT_DIR / "rsa_block_similarity_long.csv"
    out_matrix = OUTPUT_DIR / "rsa_category_similarity_matrix_rows.csv"
    long_all.to_csv(out_long, index=False)
    matrix_all.to_csv(out_matrix, index=False)
    logger.info(f"已保存长表: {out_long}")
    logger.info(f"已保存矩阵行表: {out_matrix}")

    stats_df = run_hypothesis_tests(long_all)
    out_stats = OUTPUT_DIR / "rsa_hypothesis_tests.csv"
    stats_df.to_csv(out_stats, index=False)
    logger.info(f"已保存统计检验: {out_stats}")

    plot_group_heatmaps(matrix_all)
    plot_target_pairs(long_all)

    logger.info("=== RSA 分析完成 ===")


if __name__ == "__main__":
    main()
