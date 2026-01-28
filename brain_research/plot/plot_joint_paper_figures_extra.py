#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_joint_paper_figures_extra.py

生成论文图（F4扩展）：
1) 对最显著的若干（matrix, model）组合绘制置换零分布 + 观测值位置图
2) 对全部 8×3 个检验计算 bootstrap CI，并绘制 forest plot（按 model 分列）

默认零参数运行：
- 自动搜索最新 joint_analysis_dev_models_perm_fwer.csv
- 自动使用默认相似性矩阵目录与年龄表
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

DEFAULT_FINAL_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_final_age_sorted")
DEFAULT_ROI_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_roi_age_sorted")
DEFAULT_SUBJECT_INFO_PATH = Path(
    os.environ.get(
        "SUBJECT_AGE_TABLE",
        "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv",
    )
)


@dataclass(frozen=True)
class TestKey:
    matrix: str
    model: str


def _find_latest(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def auto_find_joint_csv() -> Path:
    roots = [
        Path.cwd(),
        Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        Path("/public/home/dingrui/fmri_analysis/zz_analysis"),
    ]
    candidates: List[Path] = []
    for r in roots:
        if r.exists():
            candidates.extend(list(r.rglob("joint_analysis_dev_models_perm_fwer.csv")))
    p = _find_latest(candidates)
    if p is None:
        raise FileNotFoundError("未找到 joint_analysis_dev_models_perm_fwer.csv")
    return p


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
    y_match = re.search(r"(\d+)\s*岁", s)
    m_match = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d_match = re.search(r"(\d+)\s*天", s)
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
        raise ValueError("年龄表缺少必要列 sub_id/age 或 被试编号/采集年龄")

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        age_map[sub_id] = parse_chinese_age_exact(row[age_col])
    return age_map


def sort_subjects_by_age(subjects: Iterable[str], age_map: Dict[str, float]) -> List[str]:
    subs = list(subjects)
    ages = np.array([age_map.get(s, np.nan) for s in subs], dtype=float)
    keep = np.isfinite(ages)
    subs = [s for s, k in zip(subs, keep) if k]
    subs.sort(key=lambda sid: (age_map.get(sid, np.nan), sid))
    return subs


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def zscore_1d(x: np.ndarray) -> np.ndarray:
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


def build_model_vec(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, model: str, normalize: bool) -> np.ndarray:
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    amax = float(a.max())
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


def matrix_name_to_path(name: str, final_dir: Path, roi_dir: Path) -> Path:
    if name == "surface_L":
        return final_dir / "similarity_surface_L_AgeSorted.csv"
    if name == "surface_R":
        return final_dir / "similarity_surface_R_AgeSorted.csv"
    if name.startswith("surface_L_"):
        roi = name.replace("surface_L_", "")
        return roi_dir / f"similarity_surface_L_{roi}_AgeSorted.csv"
    if name.startswith("surface_R_"):
        roi = name.replace("surface_R_", "")
        return roi_dir / f"similarity_surface_R_{roi}_AgeSorted.csv"
    raise ValueError(f"未知矩阵名: {name}")


def load_similarity_matrix(path: Path, subjects: Sequence[str]) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.loc[list(subjects), list(subjects)]
    return df.to_numpy(dtype=np.float32, copy=False)


def compute_similarity_vec_z(mat: np.ndarray, iu: np.ndarray, ju: np.ndarray) -> np.ndarray:
    v = mat[iu, ju]
    return zscore_1d(v)


def permutation_null(
    sim_vec_z: np.ndarray,
    ages: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    model: str,
    n_perm: int,
    seed: int,
    normalize_models: bool,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(n_perm), dtype=np.float32)
    n_pairs = int(iu.size)
    for i in range(int(n_perm)):
        perm = rng.permutation(len(ages))
        ages_p = ages[perm]
        mvec = build_model_vec(ages_p, iu, ju, model=model, normalize=normalize_models)
        out[i] = float(np.dot(sim_vec_z, mvec) / float(n_pairs))
    return out


def bootstrap_ci_all(
    mats: Dict[str, np.ndarray],
    ages: np.ndarray,
    normalize_models: bool,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    n = int(ages.size)
    iu, ju = triu_indices(n)
    n_pairs = int(iu.size)
    matrix_names = sorted(mats.keys())
    model_names = ["M_nn", "M_conv", "M_div"]

    rs = np.empty((int(n_boot), len(matrix_names), len(model_names)), dtype=np.float32)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        ages_b = ages[idx]
        mvecs = [build_model_vec(ages_b, iu, ju, m, normalize_models) for m in model_names]
        for mi, mname in enumerate(matrix_names):
            mat = mats[mname][np.ix_(idx, idx)]
            svec = compute_similarity_vec_z(mat, iu, ju)
            for mj in range(len(model_names)):
                rs[b, mi, mj] = float(np.dot(svec, mvecs[mj]) / float(n_pairs))

    rows = []
    for mi, mname in enumerate(matrix_names):
        for mj, model in enumerate(model_names):
            x = rs[:, mi, mj].astype(float)
            rows.append(
                {
                    "matrix": mname,
                    "model": model,
                    "ci_lo": float(np.nanpercentile(x, 2.5)),
                    "ci_hi": float(np.nanpercentile(x, 97.5)),
                    "boot_mean": float(np.nanmean(x)),
                    "n_boot": int(n_boot),
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def plot_null_distribution(null_rs: np.ndarray, r_obs: float, key: TestKey, out_path: Path) -> None:
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    plt.figure(figsize=(6.2, 4.2))
    sns.histplot(null_rs, bins=60, kde=True, color="#4c72b0", alpha=0.65)
    plt.axvline(r_obs, color="#c44e52", linewidth=2.2, label=f"Observed r={r_obs:.3f}")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.title(f"Permutation Null\n{key.matrix} × {key.model}")
    plt.xlabel("r (Similarity vs. Dev Model)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bootstrap_forest(ci_df: pd.DataFrame, res_df: pd.DataFrame, out_path: Path) -> None:
    sns.set_context("paper", font_scale=1.25)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    merged = res_df.merge(ci_df, on=["matrix", "model"], how="left")
    matrices = sorted(merged["matrix"].astype(str).unique().tolist(), key=lambda s: (0 if s.startswith("surface_") else 1, s))
    models = ["M_nn", "M_conv", "M_div"]

    fig_h = max(4.2, 0.38 * len(matrices) + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, fig_h), sharey=True)
    for j, model in enumerate(models):
        ax = axes[j]
        d = merged[merged["model"] == model].set_index("matrix").reindex(matrices).reset_index()
        y = np.arange(len(d))
        r = d["r_obs"].to_numpy(dtype=float)
        lo = d["ci_lo"].to_numpy(dtype=float)
        hi = d["ci_hi"].to_numpy(dtype=float)
        ax.errorbar(r, y, xerr=[r - lo, hi - r], fmt="o", color="#4c72b0", ecolor="#4c72b0", elinewidth=1.4, capsize=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(model)
        ax.set_xlabel("r (obs) with 95% bootstrap CI")
        ax.set_yticks(y)
        if j == 0:
            ax.set_yticklabels(d["matrix"].astype(str).tolist())
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()

    fig.suptitle("Joint Analysis Effect Sizes with Bootstrap CIs", y=0.995)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成 Joint Analysis 论文补充图（置换分布 + bootstrap CI）")
    p.add_argument("--joint-csv", type=Path, default=None, help="joint_analysis_dev_models_perm_fwer.csv（不填则自动找最新）")
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR, help="全脑相似性矩阵目录")
    p.add_argument("--roi-dir", type=Path, default=DEFAULT_ROI_DIR, help="ROI 相似性矩阵目录")
    p.add_argument("--subject-info", type=Path, default=DEFAULT_SUBJECT_INFO_PATH, help="被试信息表路径")
    p.add_argument("--out-dir", type=Path, default=Path("./figures_joint"), help="输出目录")
    p.add_argument("--top-k", type=int, default=4, help="置换分布图绘制的检验个数（按 p_fwer 最小排序）")
    p.add_argument("--n-perm-plot", type=int, default=5000, help="用于画零分布的置换次数（不必等于主分析 n_perm）")
    p.add_argument("--n-boot", type=int, default=300, help="bootstrap 次数（用于 CI）")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    joint_csv = Path(args.joint_csv) if args.joint_csv is not None else auto_find_joint_csv()
    res_df = pd.read_csv(joint_csv)
    required = {"matrix", "model", "r_obs", "p_fwer"}
    miss = required - set(res_df.columns)
    if miss:
        raise ValueError(f"结果CSV缺少列: {sorted(miss)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_age_sorted = joint_csv.parent / "subjects_common_age_sorted.csv"
    if subject_age_sorted.exists():
        sub_df = pd.read_csv(subject_age_sorted)
        subjects_all = sub_df["subject"].astype(str).tolist()
        ages_all = sub_df["age"].to_numpy(dtype=np.float32)
        keep = np.isfinite(ages_all)
        subjects = [s for s, k in zip(subjects_all, keep) if bool(k)]
        ages = ages_all[keep]
    else:
        age_map = load_subject_ages_map(Path(args.subject_info))
        mats = []
        for m in sorted(res_df["matrix"].astype(str).unique().tolist()):
            p = matrix_name_to_path(m, Path(args.final_dir), Path(args.roi_dir))
            df = pd.read_csv(p, index_col=0)
            df.index = df.index.astype(str)
            mats.append(set(df.index.tolist()))
        common = set.intersection(*mats) if mats else set()
        subjects = sort_subjects_by_age(sorted(common), age_map)
        ages = np.array([age_map[s] for s in subjects], dtype=np.float32)

    if len(subjects) < 10:
        raise ValueError(f"共同被试数量过少: {len(subjects)}")

    iu, ju = triu_indices(len(subjects))
    normalize_models = bool(res_df["normalize_models"].dropna().iloc[0]) if "normalize_models" in res_df.columns and res_df["normalize_models"].notna().any() else True
    seed = int(res_df["seed"].dropna().iloc[0]) if "seed" in res_df.columns and res_df["seed"].notna().any() else 42

    best = res_df.copy()
    best["p_fwer"] = pd.to_numeric(best["p_fwer"], errors="coerce")
    best = best[np.isfinite(best["p_fwer"].to_numpy(dtype=float))].sort_values(["p_fwer", "matrix", "model"])
    best = best.head(int(args.top_k))

    for _, row in best.iterrows():
        key = TestKey(matrix=str(row["matrix"]), model=str(row["model"]))
        r_obs = float(row["r_obs"])
        mat_path = matrix_name_to_path(key.matrix, Path(args.final_dir), Path(args.roi_dir))
        mat = load_similarity_matrix(mat_path, subjects)
        svec = compute_similarity_vec_z(mat, iu, ju)
        null_rs = permutation_null(
            sim_vec_z=svec,
            ages=ages,
            iu=iu,
            ju=ju,
            model=key.model,
            n_perm=int(args.n_perm_plot),
            seed=int(seed),
            normalize_models=normalize_models,
        )
        out_path = out_dir / f"perm_null_{key.matrix}_{key.model}.png"
        plot_null_distribution(null_rs, r_obs=r_obs, key=key, out_path=out_path)
        print(f"Saved: {out_path}")

    mats_dict: Dict[str, np.ndarray] = {}
    for m in sorted(res_df["matrix"].astype(str).unique().tolist()):
        p = matrix_name_to_path(m, Path(args.final_dir), Path(args.roi_dir))
        mats_dict[m] = load_similarity_matrix(p, subjects)

    ci_df = bootstrap_ci_all(
        mats=mats_dict,
        ages=ages.astype(np.float32, copy=False),
        normalize_models=normalize_models,
        n_boot=int(args.n_boot),
        seed=int(seed),
    )
    ci_csv = out_dir / "joint_bootstrap_ci.csv"
    ci_df.to_csv(ci_csv, index=False)
    print(f"Saved: {ci_csv}")

    out_forest = out_dir / "joint_bootstrap_forest.png"
    plot_bootstrap_forest(ci_df=ci_df, res_df=res_df, out_path=out_forest)
    print(f"Saved: {out_forest}")


if __name__ == "__main__":
    main()
