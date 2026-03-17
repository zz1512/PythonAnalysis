#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_analysis_roi_isc_dev_models_perm_fwer.py

执行顺序（新链路 Step 3/3）：
1) build_roi_beta_matrix_200.py
2) calc_roi_isc_by_age.py
3) 本脚本：joint_analysis_roi_isc_dev_models_perm_fwer.py

基于 ROI 级被试×被试相似性矩阵，构造 3 个发育模型向量并进行置换检验，
在每个模型内部做 FWER 校正，仅检验正效应。

输入默认来自 calc_roi_isc_by_age.py 输出：
- roi_isc_spearman_by_age.npy
- roi_isc_spearman_by_age_subjects_sorted.csv
- roi_isc_spearman_by_age_rois.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results")


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


def build_dev_model_vectors(
    ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def load_subjects_sorted(path: Path) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    if "subject" not in df.columns or "age" not in df.columns:
        raise ValueError(f"subjects_sorted 文件缺少 subject/age 列: {path}")
    subjects = df["subject"].astype(str).tolist()
    ages = df["age"].astype(float).to_numpy()
    return subjects, ages


def load_rois(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "roi" in df.columns:
        return df["roi"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI ISC vs 发育模型：置换检验 + model-wise FWER（正向单尾）")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--isc", type=Path, default=None)
    p.add_argument("--subjects-sorted", type=Path, default=None)
    p.add_argument("--rois", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    return p.parse_args()


def run(
    matrix_dir: Path,
    isc: Path = None,
    subjects_sorted: Path = None,
    rois: Path = None,
    out_dir: Path = None,
    n_perm: int = 5000,
    seed: int = 42,
    normalize_models: bool = True,
) -> None:
    isc_prefix = "roi_isc_spearman_by_age"
    isc_path = isc or (matrix_dir / f"{isc_prefix}.npy")
    subs_path = subjects_sorted or (matrix_dir / f"{isc_prefix}_subjects_sorted.csv")
    rois_path = rois or (matrix_dir / f"{isc_prefix}_rois.csv")
    out_dir2 = out_dir or matrix_dir

    isc_arr = np.load(isc_path)
    subjects, ages = load_subjects_sorted(subs_path)
    rois_list = load_rois(rois_path)

    if isc_arr.shape[0] != len(rois_list):
        raise ValueError(f"ROI 数量不一致: isc={isc_arr.shape[0]} rois={len(rois_list)}")
    if isc_arr.shape[1] != len(subjects) or isc_arr.shape[2] != len(subjects):
        raise ValueError(f"被试数量不一致: isc={isc_arr.shape[1]} subjects={len(subjects)}")

    n_sub = len(subjects)
    iu, ju = triu_indices(n_sub)
    n_pairs = int(iu.size)

    S_vecs = []
    for i in range(len(rois_list)):
        v = isc_arr[i][iu, ju]
        v_z = zscore_1d(v)
        if not np.isfinite(v_z).any():
            raise ValueError(f"ROI 向量无有效值: {rois_list[i]}")
        S_vecs.append(v_z)
    S_z = np.stack(S_vecs, axis=0).astype(np.float32, copy=False)

    m_nn, m_conv, m_div = build_dev_model_vectors(ages, iu, ju, normalize=bool(normalize_models))
    model_names = ("M_nn", "M_conv", "M_div")
    model_vecs_obs = (m_nn, m_conv, m_div)

    r_obs = []
    tests = []
    for mi, mname in enumerate(model_names):
        mvec = model_vecs_obs[mi]
        r_rois = (S_z @ mvec) / float(n_pairs)
        for ri in range(len(rois_list)):
            tests.append({"roi": rois_list[ri], "model": mname})
            r_obs.append(float(r_rois[ri]))
    r_obs_arr = np.asarray(r_obs, dtype=np.float32)
    n_rois = len(rois_list)
    n_tests = int(r_obs_arr.size)

    count_raw = np.zeros(n_tests, dtype=np.int64)
    count_fwer = np.zeros(n_tests, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    for _ in range(int(n_perm)):
        perm = rng.permutation(n_sub)
        ages_p = ages[perm]
        pm_nn, pm_conv, pm_div = build_dev_model_vectors(ages_p, iu, ju, normalize=bool(normalize_models))
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

    p_perm = (count_raw + 1.0) / (float(n_perm) + 1.0)
    p_fwer = (count_fwer + 1.0) / (float(n_perm) + 1.0)

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
                "n_perm": int(n_perm),
                "seed": int(seed),
                "normalize_models": bool(normalize_models),
            }
        )

    out_prefix = "roi_isc_dev_models_perm_fwer"
    out_dir2.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values(["model", "roi"])
    out_df.to_csv(out_dir2 / f"{out_prefix}.csv", index=False)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(out_dir2 / f"{out_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois_list}).to_csv(out_dir2 / f"{out_prefix}_rois.csv", index=False)


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=args.matrix_dir,
        isc=args.isc,
        subjects_sorted=args.subjects_sorted,
        rois=args.rois,
        out_dir=args.out_dir,
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        normalize_models=bool(args.normalize_models),
    )


if __name__ == "__main__":
    main()
