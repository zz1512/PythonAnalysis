#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_analysis_roi_isc_dev_models_perm_fwer.py

基于 ROI 级被试×被试相关矩阵，构造 3 个发育模型向量并进行置换检验，
在每个模型内部做 FWER 校正，仅检验正效应。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def zscore_1d(x: np.ndarray) -> np.ndarray:
    """对一维向量做 z-score，异常/缺失返回 NaN。"""
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


def build_dev_model_vectors(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据年龄构造三种发育模型上三角向量。"""
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("ages 包含 NaN/Inf，无法构建模型向量。")
    amax = float(a.max())
    ai = a[iu]
    aj = a[ju]
    nn = (amax - np.abs(ai - aj)).astype(np.float32, copy=False)
    conv = np.minimum(ai, aj).astype(np.float32, copy=False)
    div = (amax - 0.5 * (ai + aj)).astype(np.float32, copy=False)
    if normalize:
        return zscore_1d(nn), zscore_1d(conv), zscore_1d(div)
    return nn, conv, div


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回上三角索引（不含对角）。"""
    return np.triu_indices(n, k=1)


def load_subjects_sorted(path: Path) -> Tuple[List[str], np.ndarray]:
    """读取已按年龄排序的被试与年龄。"""
    df = pd.read_csv(path)
    if "subject" not in df.columns or "age" not in df.columns:
        raise ValueError(f"subjects_sorted 文件缺少 subject/age 列: {path}")
    subjects = df["subject"].astype(str).tolist()
    ages = df["age"].astype(float).to_numpy()
    return subjects, ages


def load_rois(path: Path) -> List[str]:
    """读取 ROI 名称列表。"""
    df = pd.read_csv(path)
    if "roi" in df.columns:
        return df["roi"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def main() -> None:
    """主流程：读入 ROI 相关矩阵，构造模型并置换检验，输出统计结果。"""
    ISC_MATRIX_PATH = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results/roi_isc_spearman_by_age.npy")
    SUBJECTS_SORTED_PATH = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results/roi_isc_spearman_by_age_subjects_sorted.csv")
    ROI_PATH = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results/roi_beta_matrix_232_rois.csv")
    OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")
    OUT_PREFIX = "roi_isc_dev_models_perm_fwer"
    N_PERM = 5000
    SEED = 42
    NORMALIZE_MODELS = True

    isc = np.load(ISC_MATRIX_PATH)
    subjects, ages = load_subjects_sorted(SUBJECTS_SORTED_PATH)
    rois = load_rois(ROI_PATH)

    if isc.shape[0] != len(rois):
        raise ValueError(f"ROI 数量不一致: isc={isc.shape[0]} rois={len(rois)}")
    if isc.shape[1] != len(subjects) or isc.shape[2] != len(subjects):
        raise ValueError(f"被试数量不一致: isc={isc.shape[1]} subjects={len(subjects)}")

    n_sub = len(subjects)
    iu, ju = triu_indices(n_sub)
    n_pairs = int(iu.size)

    S_vecs = []
    for i in range(len(rois)):
        v = isc[i][iu, ju]
        v_z = zscore_1d(v)
        if not np.isfinite(v_z).any():
            raise ValueError(f"ROI 向量无有效值: {rois[i]}")
        S_vecs.append(v_z)
    S_z = np.stack(S_vecs, axis=0).astype(np.float32, copy=False)

    m_nn, m_conv, m_div = build_dev_model_vectors(ages, iu, ju, normalize=NORMALIZE_MODELS)
    model_names = ("M_nn", "M_conv", "M_div")
    model_vecs_obs = (m_nn, m_conv, m_div)

    r_obs = []
    tests = []
    for mi, mname in enumerate(model_names):
        mvec = model_vecs_obs[mi]
        r_rois = (S_z @ mvec) / float(n_pairs)
        for ri in range(len(rois)):
            tests.append({"roi": rois[ri], "model": mname})
            r_obs.append(float(r_rois[ri]))
    r_obs_arr = np.asarray(r_obs, dtype=np.float32)
    n_rois = len(rois)
    n_tests = int(r_obs_arr.size)

    count_raw = np.zeros(n_tests, dtype=np.int64)
    count_fwer = np.zeros(n_tests, dtype=np.int64)
    rng = np.random.default_rng(SEED)

    for _ in range(int(N_PERM)):
        perm = rng.permutation(n_sub)
        ages_p = ages[perm]
        pm_nn, pm_conv, pm_div = build_dev_model_vectors(ages_p, iu, ju, normalize=NORMALIZE_MODELS)
        model_vecs_p = (pm_nn, pm_conv, pm_div)

        r_perm_all = np.empty(n_tests, dtype=np.float32)
        offset = 0
        for mvec in model_vecs_p:
            r_rois = (S_z @ mvec) / float(n_pairs)
            r_perm_all[offset:offset + n_rois] = r_rois.astype(np.float32, copy=False)
            offset += n_rois

        count_raw += (r_perm_all >= r_obs_arr)

        offset = 0
        for _ in range(len(model_names)):
            start_idx = offset
            end_idx = offset + n_rois
            r_perm_subset = r_perm_all[start_idx:end_idx]
            r_obs_subset = r_obs_arr[start_idx:end_idx]
            max_stat_model = float(r_perm_subset.max(initial=-1.0))
            count_fwer[start_idx:end_idx] += (max_stat_model >= r_obs_subset)
            offset += n_rois

    p_perm = (count_raw + 1.0) / (float(N_PERM) + 1.0)
    p_fwer = (count_fwer + 1.0) / (float(N_PERM) + 1.0)

    rows = []
    for i in range(n_tests):
        meta = tests[i]
        rows.append(
            {
                "roi": meta["roi"],
                "model": meta["model"],
                "r_obs": float(r_obs_arr[i]),
                "p_perm_one_tailed": float(p_perm[i]),
                "p_fwer_model_wise": float(p_fwer[i]),
                "n_subjects": int(n_sub),
                "n_pairs": int(n_pairs),
                "n_perm": int(N_PERM),
                "seed": int(SEED),
                "normalize_models": bool(NORMALIZE_MODELS),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["model", "roi"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_DIR / f"{OUT_PREFIX}.csv", index=False)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(
        OUT_DIR / f"{OUT_PREFIX}_subjects_sorted.csv", index=False
    )
    pd.DataFrame({"roi": rois}).to_csv(OUT_DIR / f"{OUT_PREFIX}_rois.csv", index=False)


if __name__ == "__main__":
    main()
