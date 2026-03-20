#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_analysis_roi_isc_dev_models.py

Step 3:
对每个刺激类型，基于 ROI 级被试×被试相似性矩阵进行发育模型置换检验。
模型：M_nn, M_conv, M_div；仅正向单尾；模型内 FWER(max-T) 校正。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI ISC vs 发育模型：置换检验 + model-wise FWER")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--isc-method", type=str, default="spearman", choices=("spearman", "pearson"))
    p.add_argument("--isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    p.add_argument("--no-fisher-z", action="store_false", dest="fisher_z", default=True)
    return p.parse_args()


def detect_isc_prefix(stim_dir: Path) -> List[str]:
    prefixes = []
    for npy in sorted(stim_dir.glob("roi_isc_*_by_age.npy")):
        prefix = npy.stem
        if (stim_dir / f"{prefix}_subjects_sorted.csv").exists() and (stim_dir / f"{prefix}_rois.csv").exists():
            prefixes.append(prefix)
    return prefixes


def resolve_isc_prefix(stim_dir: Path, isc_prefix: Optional[str], isc_method: Optional[str]) -> str:
    if isc_prefix is not None:
        return str(isc_prefix)
    found = detect_isc_prefix(stim_dir)
    if isc_method is not None:
        expected = f"roi_isc_{str(isc_method)}_by_age"
        if expected in found:
            return expected
        raise FileNotFoundError(f"未找到指定 isc-method 对应产物: {expected}（可用: {found}）")
    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        raise FileNotFoundError(f"{stim_dir} 下未找到 roi_isc_*_by_age.npy")
    raise ValueError(f"{stim_dir} 下存在多个 ISC 产物前缀: {found}，请指定 --isc-prefix 或 --isc-method")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    m = np.isfinite(p)
    if int(m.sum()) == 0:
        return q
    pv = p[m]
    n = int(pv.size)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * float(n) / np.arange(1, n + 1, dtype=np.float64)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_valid = np.empty_like(pv)
    q_valid[order] = q_ranked
    q[m] = q_valid
    return q


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


def fisher_z(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, dtype=np.float32)
    m = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=np.float32)
    if not bool(m.any()):
        return out
    xc = np.clip(x[m], -0.999999, 0.999999)
    out[m] = np.arctanh(xc).astype(np.float32, copy=False)
    return out


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


def assoc(S_z: np.ndarray, m: np.ndarray, method: str) -> np.ndarray:
    mv = np.asarray(m, dtype=np.float32).reshape(-1)
    mask_m = np.isfinite(mv)
    if int(mask_m.sum()) <= 1:
        return np.full((S_z.shape[0],), np.nan, dtype=np.float32)

    Sz = np.asarray(S_z, dtype=np.float32)
    mth = str(method).strip().lower()
    if mth == "pearson":
        joint = mask_m.reshape(1, -1) & np.isfinite(Sz)
        denom = joint.sum(axis=1).astype(np.float32, copy=False)
        prod = Sz * mv.reshape(1, -1)
        prod[~joint] = 0.0
        denom_safe = denom.copy()
        denom_safe[denom_safe == 0] = 1.0
        out = prod.sum(axis=1) / denom_safe
        out[denom <= 1] = np.nan
        return out.astype(np.float32, copy=False)

    if mth == "spearman":
        out = np.full((Sz.shape[0],), np.nan, dtype=np.float32)
        valid_rows = np.isfinite(Sz).all(axis=1)

        if bool(valid_rows.any()):
            mr = np.full_like(mv, np.nan, dtype=np.float32)
            mr[mask_m] = zscore_1d(rankdata(mv[mask_m]).astype(np.float32, copy=False))
            Sz_fast = Sz[valid_rows]
            joint = mask_m.reshape(1, -1) & np.isfinite(Sz_fast)
            denom = joint.sum(axis=1).astype(np.float32, copy=False)
            prod = Sz_fast * mr.reshape(1, -1)
            prod[~joint] = 0.0
            denom_safe = denom.copy()
            denom_safe[denom_safe == 0] = 1.0
            out_fast = prod.sum(axis=1) / denom_safe
            out_fast[denom <= 1] = np.nan
            out[valid_rows] = out_fast.astype(np.float32, copy=False)

        invalid_rows = ~valid_rows
        if bool(invalid_rows.any()):
            idx_invalid = np.where(invalid_rows)[0]
            for i in idx_invalid.tolist():
                x = Sz[i].reshape(-1)
                joint_1d = mask_m & np.isfinite(x)
                if int(joint_1d.sum()) < 3:
                    continue
                xr = rankdata(x[joint_1d]).astype(np.float32, copy=False)
                mr = rankdata(mv[joint_1d]).astype(np.float32, copy=False)
                xz = zscore_1d(xr)
                mz = zscore_1d(mr)
                ok = np.isfinite(xz) & np.isfinite(mz)
                if int(ok.sum()) < 3:
                    continue
                out[i] = float((xz[ok] * mz[ok]).mean())

        return out

    raise ValueError(f"未知 assoc-method: {method}")


def run_one(
    stim_dir: Path,
    n_perm: int,
    seed: int,
    isc_prefix: str,
    normalize_models: bool,
    fisher_z_enabled: bool,
    assoc_method: str,
) -> pd.DataFrame:
    isc = np.load(stim_dir / f"{isc_prefix}.npy")
    sub_df = pd.read_csv(stim_dir / f"{isc_prefix}_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / f"{isc_prefix}_rois.csv")
    ages = sub_df["age"].astype(float).to_numpy()
    subjects = sub_df["subject"].astype(str).tolist()
    rois = roi_df["roi"].astype(str).tolist()

    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)

    S_z = []
    assoc_m = str(assoc_method).strip().lower()
    for i in range(len(rois)):
        v = isc[i][iu, ju]
        if bool(fisher_z_enabled):
            v = fisher_z(v)
        if assoc_m == "spearman":
            mask_v = np.isfinite(v)
            vr = np.full_like(v, np.nan, dtype=np.float32)
            vr[mask_v] = rankdata(v[mask_v]).astype(np.float32, copy=False)
            S_z.append(zscore_1d(vr))
        else:
            S_z.append(zscore_1d(v))
    S_z = np.stack(S_z, axis=0).astype(np.float32)

    model_names = ("M_nn", "M_conv", "M_div")
    m_obs = build_models(ages, iu, ju, normalize=bool(normalize_models))

    obs = []
    tests = []
    for mi, mname in enumerate(model_names):
        rr = assoc(S_z, m_obs[mi], method=str(assoc_method))
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
            rr = assoc(S_z, mv, method=str(assoc_method))
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
            "isc_prefix": str(isc_prefix),
            "assoc_method": str(assoc_method),
            "fisher_z": bool(fisher_z_enabled),
        })

    out_df = pd.DataFrame(rows).sort_values(["model", "roi"])
    out_df["p_fdr_bh_model_wise"] = np.nan
    for _, idx in out_df.groupby("model").groups.items():
        out_df.loc[idx, "p_fdr_bh_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_one_tailed"].to_numpy(dtype=float))
    out_df["p_fdr_bh_global"] = bh_fdr(out_df["p_perm_one_tailed"].to_numpy(dtype=float))
    out_prefix = "roi_isc_dev_models_perm_fwer"
    out_df.to_csv(stim_dir / f"{out_prefix}.csv", index=False)
    sub_df[["subject", "age"]].to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
    return out_df


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    n_perm: int,
    seed: int,
    isc_prefix: Optional[str],
    isc_method: Optional[str],
    normalize_models: bool,
    fisher_z_enabled: bool,
    assoc_method: str,
) -> None:
    by_stim = matrix_dir / str(stimulus_dir_name)
    stim_dirs = sorted([p for p in by_stim.iterdir() if p.is_dir()])
    if not stim_dirs:
        raise FileNotFoundError(f"{by_stim} 下无条件目录")

    summary = []
    for d in stim_dirs:
        prefix = resolve_isc_prefix(d, isc_prefix=isc_prefix, isc_method=isc_method)
        out = run_one(
            d,
            n_perm=int(n_perm),
            seed=int(seed),
            isc_prefix=str(prefix),
            normalize_models=bool(normalize_models),
            fisher_z_enabled=bool(fisher_z_enabled),
            assoc_method=str(assoc_method),
        )
        sig = out[(out["r_obs"] > 0) & (out["p_fwer_model_wise"] <= 0.05)]
        summary.append({"stimulus_type": d.name, "n_rows": int(out.shape[0]), "n_sig_pos_fwer": int(sig.shape[0])})

    pd.DataFrame(summary).sort_values("stimulus_type").to_csv(matrix_dir / "roi_isc_dev_models_perm_fwer_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    by_stim = matrix_dir / str(args.stimulus_dir_name)
    stim_dirs = sorted([p for p in by_stim.iterdir() if p.is_dir()]) if by_stim.exists() else []
    if not stim_dirs:
        raise FileNotFoundError(f"{by_stim} 下无条件目录")
    run(
        matrix_dir,
        str(args.stimulus_dir_name),
        int(args.n_perm),
        int(args.seed),
        isc_prefix=args.isc_prefix,
        isc_method=args.isc_method,
        normalize_models=bool(args.normalize_models),
        fisher_z_enabled=bool(args.fisher_z),
        assoc_method=str(args.assoc_method),
    )


if __name__ == "__main__":
    main()
