#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_analysis_roi_isc_dev_models_perm_fwer.py

Step 3:
对每个刺激类型，基于 ROI 级被试×被试相似性矩阵进行发育模型置换检验。
模型：M_nn, M_conv, M_div；仅正向单尾；模型内 FWER(max-T) 校正。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI ISC vs 发育模型：置换检验 + model-wise FWER")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    return p.parse_args()


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x)
    if int(m.sum()) < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[m].mean())
    sd = float(x[m].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~m] = np.nan
    return out.astype(np.float32)


def build_models(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("ages 包含 NaN/Inf")
    amax = float(a.max())
    ai = a[iu]
    aj = a[ju]
    nn = (amax - np.abs(ai - aj)).astype(np.float32)
    conv = np.minimum(ai, aj).astype(np.float32)
    div = (amax - 0.5 * (ai + aj)).astype(np.float32)
    if normalize:
        return zscore_1d(nn), zscore_1d(conv), zscore_1d(div)
    return nn, conv, div


def assoc(S_z: np.ndarray, m: np.ndarray) -> np.ndarray:
    mv = np.asarray(m, dtype=np.float32).reshape(-1)
    mask = np.isfinite(mv)
    n = int(mask.sum())
    if n <= 1:
        return np.full((S_z.shape[0],), np.nan, dtype=np.float32)
    return (S_z[:, mask] @ mv[mask]) / float(n)


def run_one(stim_dir: Path, n_perm: int, seed: int, normalize_models: bool) -> pd.DataFrame:
    isc = np.load(stim_dir / "roi_isc_spearman_by_age.npy")
    sub_df = pd.read_csv(stim_dir / "roi_isc_spearman_by_age_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / "roi_isc_spearman_by_age_rois.csv")
    ages = sub_df["age"].astype(float).to_numpy()
    subjects = sub_df["subject"].astype(str).tolist()
    rois = roi_df["roi"].astype(str).tolist()

    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)

    S_z = []
    for i in range(len(rois)):
        v = isc[i][iu, ju]
        S_z.append(zscore_1d(v))
    S_z = np.stack(S_z, axis=0).astype(np.float32)

    model_names = ("M_nn", "M_conv", "M_div")
    m_obs = build_models(ages, iu, ju, normalize=bool(normalize_models))

    obs = []
    tests = []
    for mi, mname in enumerate(model_names):
        rr = assoc(S_z, m_obs[mi])
        for ri, roi in enumerate(rois):
            tests.append((roi, mname))
            obs.append(float(rr[ri]))
    r_obs = np.asarray(obs, dtype=np.float32)

    n_rois = len(rois)
    n_tests = int(r_obs.size)
    c_raw = np.zeros(n_tests, dtype=np.int64)
    c_fwer = np.zeros(n_tests, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    for _ in range(int(n_perm)):
        ages_p = ages[rng.permutation(n_sub)]
        m_perm = build_models(ages_p, iu, ju, normalize=bool(normalize_models))
        r_all = np.empty(n_tests, dtype=np.float32)
        off = 0
        for mv in m_perm:
            rr = assoc(S_z, mv)
            r_all[off:off + n_rois] = rr
            off += n_rois

        valid = np.isfinite(r_all) & np.isfinite(r_obs)
        c_raw[valid] += (r_all[valid] >= r_obs[valid])

        off = 0
        for _m in model_names:
            s, e = off, off + n_rois
            perm_sub = r_all[s:e]
            obs_sub = r_obs[s:e]
            valid_sub = np.isfinite(perm_sub) & np.isfinite(obs_sub)
            if np.any(valid_sub):
                max_stat = float(np.max(perm_sub[valid_sub]))
                idx = np.where(np.isfinite(obs_sub))[0] + s
                c_fwer[idx] += (max_stat >= r_obs[idx])
            off += n_rois

    p_raw = np.full(n_tests, np.nan, dtype=np.float32)
    p_fwer = np.full(n_tests, np.nan, dtype=np.float32)
    ok = np.isfinite(r_obs)
    p_raw[ok] = (c_raw[ok] + 1.0) / (float(n_perm) + 1.0)
    p_fwer[ok] = (c_fwer[ok] + 1.0) / (float(n_perm) + 1.0)

    rows = []
    for i, (roi, model) in enumerate(tests):
        rows.append({
            "roi": roi,
            "model": model,
            "r_obs": float(r_obs[i]),
            "p_perm_one_tailed": float(p_raw[i]),
            "p_fwer_model_wise": float(p_fwer[i]),
            "n_subjects": int(n_sub),
            "n_pairs": int(n_pairs),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "normalize_models": bool(normalize_models),
        })

    out_df = pd.DataFrame(rows).sort_values(["model", "roi"])
    out_df.to_csv(stim_dir / "roi_isc_dev_models_perm_fwer.csv", index=False)
    return out_df


def run(matrix_dir: Path, n_perm: int, seed: int, normalize_models: bool) -> None:
    by_stim = matrix_dir / "by_stimulus"
    stim_dirs = sorted([p for p in by_stim.iterdir() if p.is_dir()])
    if not stim_dirs:
        raise FileNotFoundError(f"{by_stim} 下无条件目录")

    summary = []
    for d in stim_dirs:
        out = run_one(d, n_perm=int(n_perm), seed=int(seed), normalize_models=bool(normalize_models))
        sig = out[(out["r_obs"] > 0) & (out["p_fwer_model_wise"] <= 0.05)]
        summary.append({"stimulus_type": d.name, "n_rows": int(out.shape[0]), "n_sig_pos_fwer": int(sig.shape[0])})

    pd.DataFrame(summary).sort_values("stimulus_type").to_csv(matrix_dir / "roi_isc_dev_models_perm_fwer_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run(args.matrix_dir, int(args.n_perm), int(args.seed), bool(args.normalize_models))


if __name__ == "__main__":
    main()
