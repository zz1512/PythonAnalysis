#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from joint_analysis_roi_isc_dev_models import bh_fdr, fisher_z, zscore_1d  # noqa: E402
else:
    from ..joint_analysis_roi_isc_dev_models import bh_fdr, fisher_z, zscore_1d  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiple Regression RSA: Brain_ISC = β1*Behavior_ISC + β2*Age_Model + β3*(Behavior_ISC*Age_Model)")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--brain-isc-method", type=str, default="mahalanobis")
    p.add_argument("--brain-isc-prefix", type=str, default=None)
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis")
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--fisher-z-brain", type=str, default=None)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rank-transform", action="store_true", dest="rank_transform", help="Rank transform matrices before regression (similar to Spearman)")
    g.add_argument("--no-rank-transform", action="store_false", dest="rank_transform", help="Disable rank transform and use standard z-scoring")
    p.set_defaults(rank_transform=True)
    p.add_argument("--tail", type=str, default="two_sided", choices=("two_sided", "positive"))
    p.add_argument("--correction-mode", type=str, default="perm_fwer_fdr", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def _policy_flag(method: str, override: Optional[str]) -> bool:
    if override is not None:
        return str(override).strip().lower() == "true"
    return str(method).strip().lower() in {"pearson", "spearman"}

def _resolve_prefix(prefix: Optional[str], method: str, fallback_template: str) -> str:
    return str(prefix) if prefix is not None else fallback_template.format(method=str(method).strip().lower())

def has_brain_files(stim_dir: Path, brain_isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{brain_isc_prefix}.npy",
        stim_dir / f"{brain_isc_prefix}_subjects_sorted.csv",
        stim_dir / f"{brain_isc_prefix}_rois.csv",
    )
    return all(p.exists() for p in required)

def has_behavior_files(stim_dir: Path, behavior_isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{behavior_isc_prefix}.npy",
        stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv",
    )
    return all(p.exists() for p in required)

def _align_subjects(
    brain_isc: np.ndarray,
    brain_subjects_df: pd.DataFrame,
    behavior_isc: np.ndarray,
    behavior_subjects_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    brain_subjects = brain_subjects_df["subject"].astype(str).tolist()
    behavior_subjects = behavior_subjects_df["subject"].astype(str).tolist()
    
    brain_has_age = "age" in brain_subjects_df.columns
    beh_has_age = "age" in behavior_subjects_df.columns
    if not (brain_has_age or beh_has_age):
        raise KeyError("Cannot find 'age' column in either brain or behavior subject CSV")

    age_map: dict[str, float] = {}
    if brain_has_age:
        age_map.update(dict(zip(brain_subjects_df["subject"].astype(str), brain_subjects_df["age"].astype(float))))
    if beh_has_age:
        beh_age_map = dict(zip(behavior_subjects_df["subject"].astype(str), behavior_subjects_df["age"].astype(float)))
        for s, a in beh_age_map.items():
            cur = age_map.get(s, np.nan)
            if not np.isfinite(cur):
                age_map[s] = a
    
    behavior_set = set(behavior_subjects)
    
    # Ensure subjects have valid age (not 999.0 and not NaN)
    overlap = []
    for s in brain_subjects:
        if s in behavior_set:
            age_val = age_map.get(s, np.nan)
            if np.isfinite(age_val) and age_val < 998.0:
                overlap.append(s)
                
    if len(overlap) < 3:
        raise ValueError(f"Fewer than 3 overlapping subjects with valid age between brain ISC and behavior ISC")

    brain_idx = {s: i for i, s in enumerate(brain_subjects)}
    behavior_idx = {s: i for i, s in enumerate(behavior_subjects)}
    
    ib = [brain_idx[s] for s in overlap]
    ih = [behavior_idx[s] for s in overlap]
    
    brain_isc = np.asarray(brain_isc, dtype=np.float32)[:, ib, :][:, :, ib]
    behavior_isc = np.asarray(behavior_isc, dtype=np.float32)[np.ix_(ih, ih)]
    ages = [float(age_map[s]) for s in overlap]
    
    return brain_isc, behavior_isc, overlap, ages

def build_models(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray, normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(ages, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("ages contains NaN/Inf")
    amax = float(a.max())
    ai = a[iu]
    aj = a[ju]
    nn = (amax - np.abs(ai - aj)).astype(np.float32)
    conv = np.minimum(ai, aj).astype(np.float32)
    div = (amax - 0.5 * (ai + aj)).astype(np.float32)
    if normalize:
        return zscore_1d(nn), zscore_1d(conv), zscore_1d(div)
    return nn, conv, div

def _ols_tvals(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xv = np.asarray(X, dtype=np.float64)
    Yv = np.asarray(Y, dtype=np.float64)
    if Yv.ndim != 2:
        raise ValueError("Y must be 2D (n_rois, n_obs)")
    if Xv.ndim != 2:
        raise ValueError("X must be 2D (n_obs, p)")
    n_obs = int(Xv.shape[0])
    p = int(Xv.shape[1])
    if int(Yv.shape[1]) != n_obs:
        raise ValueError("X and Y have incompatible shapes")
    dof = float(n_obs - p)
    if dof <= 0:
        raise ValueError("Too few observations for OLS")
    xtx_inv = np.linalg.pinv(Xv.T @ Xv)
    b = xtx_inv @ Xv.T
    beta = b @ Yv.T
    yhat = Xv @ beta
    resid = Yv.T - yhat
    mse = (resid * resid).sum(axis=0) / dof
    diag = np.diag(xtx_inv).reshape(-1, 1)
    se = np.sqrt(np.maximum(0.0, diag * mse.reshape(1, -1)))
    tvals = beta / se
    return beta.T, tvals.T

def run_one(
    stim_dir: Path,
    brain_isc_prefix: str,
    behavior_isc_prefix: str,
    fisher_z_brain: bool,
    fisher_z_behavior: bool,
    normalize_models: bool,
    rank_transform: bool,
    tail: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    brain_isc = np.load(stim_dir / f"{brain_isc_prefix}.npy")
    behavior_isc = np.load(stim_dir / f"{behavior_isc_prefix}.npy")
    brain_sub_df = pd.read_csv(stim_dir / f"{brain_isc_prefix}_subjects_sorted.csv")
    beh_sub_df = pd.read_csv(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / f"{brain_isc_prefix}_rois.csv")
    rois = roi_df["roi"].astype(str).tolist()

    brain_isc, behavior_isc, subjects, ages = _align_subjects(brain_isc, brain_sub_df, behavior_isc, beh_sub_df)
    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)

    # Prepare behavior vector (X1)
    beh_vec = behavior_isc[iu, ju].astype(np.float32)
    if bool(fisher_z_behavior):
        beh_vec = fisher_z(beh_vec)
        
    if rank_transform:
        mask_b = np.isfinite(beh_vec)
        br = np.full_like(beh_vec, np.nan, dtype=np.float32)
        br[mask_b] = rankdata(beh_vec[mask_b])
        beh_vec = zscore_1d(br)
    else:
        beh_vec = zscore_1d(beh_vec)

    # Prepare age models (X2)
    model_names = ("M_nn", "M_conv", "M_div")
    m_obs = build_models(np.array(ages), iu, ju, normalize=False)
    
    if rank_transform:
        m_obs_z = []
        for m in m_obs:
            mask_m = np.isfinite(m)
            mr = np.full_like(m, np.nan, dtype=np.float32)
            mr[mask_m] = rankdata(m[mask_m])
            m_obs_z.append(zscore_1d(mr))
    else:
        m_obs_z = [zscore_1d(m) if normalize_models else m for m in m_obs]

    # Prepare brain matrix (Y)
    S_z = []
    for ri in range(len(rois)):
        v = brain_isc[ri][iu, ju]
        if bool(fisher_z_brain):
            v = fisher_z(v)
        if rank_transform:
            mask_v = np.isfinite(v)
            vr = np.full_like(v, np.nan, dtype=np.float32)
            vr[mask_v] = rankdata(v[mask_v])
            S_z.append(zscore_1d(vr))
        else:
            S_z.append(zscore_1d(v))
    S_z = np.stack(S_z, axis=0).astype(np.float32)

    rows = []
    tail_mode = str(tail).strip().lower()
    mode = str(correction_mode).strip().lower()
    pair_pos = np.full((n_sub, n_sub), -1, dtype=np.int32)
    pair_pos[iu, ju] = np.arange(int(iu.size), dtype=np.int32)
    pair_pos[ju, iu] = pair_pos[iu, ju]
    rng = np.random.default_rng(int(seed)) if mode == "perm_fwer_fdr" else None

    for mi, mname in enumerate(model_names):
        age_vec = m_obs_z[mi]
        
        # Interaction term (X1 * X2)
        interaction_vec = beh_vec * age_vec
        if rank_transform or normalize_models:
            interaction_vec = zscore_1d(interaction_vec)
            
        # Design matrix X
        X_df = pd.DataFrame({
            'const': 1.0,
            'behavior': beh_vec,
            'age_model': age_vec,
            'interaction': interaction_vec
        })

        p_perm_behavior = np.full(len(rois), np.nan, dtype=np.float64)
        p_perm_age = np.full(len(rois), np.nan, dtype=np.float64)
        p_perm_interaction = np.full(len(rois), np.nan, dtype=np.float64)
        p_fwer_behavior = np.full(len(rois), np.nan, dtype=np.float64)
        p_fwer_age = np.full(len(rois), np.nan, dtype=np.float64)
        p_fwer_interaction = np.full(len(rois), np.nan, dtype=np.float64)
        if mode == "perm_fwer_fdr" and int(n_perm) > 0:
            X_mat = X_df[["const", "behavior", "age_model", "interaction"]].to_numpy(dtype=np.float64, copy=True)
            valid_pairs = np.isfinite(X_mat).all(axis=1)
            if int(valid_pairs.sum()) > int(X_mat.shape[1]) + 1:
                roi_ok = np.isfinite(S_z[:, valid_pairs]).all(axis=1)
                if bool(roi_ok.any()):
                    Xv = X_mat[valid_pairs]
                    Yv = S_z[roi_ok][:, valid_pairs].astype(np.float64, copy=False)
                    _, t_obs = _ols_tvals(Xv, Yv)
                    t_obs_b = t_obs[:, 1]
                    t_obs_a = t_obs[:, 2]
                    t_obs_i = t_obs[:, 3]

                    c_b = np.zeros(int(roi_ok.sum()), dtype=np.int64)
                    c_a = np.zeros(int(roi_ok.sum()), dtype=np.int64)
                    c_i = np.zeros(int(roi_ok.sum()), dtype=np.int64)
                    cfb = np.zeros(int(roi_ok.sum()), dtype=np.int64)
                    cfa = np.zeros(int(roi_ok.sum()), dtype=np.int64)
                    cfi = np.zeros(int(roi_ok.sum()), dtype=np.int64)

                    for _ in range(int(n_perm)):
                        perm = rng.permutation(n_sub)
                        idx_perm_full = pair_pos[perm[iu], perm[ju]]
                        idx_perm = idx_perm_full[valid_pairs]
                        Yp = S_z[roi_ok][:, idx_perm].astype(np.float64, copy=False)
                        _, t_perm = _ols_tvals(Xv, Yp)
                        tp_b = t_perm[:, 1]
                        tp_a = t_perm[:, 2]
                        tp_i = t_perm[:, 3]

                        vb = np.isfinite(t_obs_b) & np.isfinite(tp_b)
                        va = np.isfinite(t_obs_a) & np.isfinite(tp_a)
                        vi = np.isfinite(t_obs_i) & np.isfinite(tp_i)

                        if tail_mode == "two_sided":
                            c_b[vb] += (np.abs(tp_b[vb]) >= np.abs(t_obs_b[vb]))
                            c_a[va] += (np.abs(tp_a[va]) >= np.abs(t_obs_a[va]))
                            c_i[vi] += (np.abs(tp_i[vi]) >= np.abs(t_obs_i[vi]))
                        else:
                            c_b[vb] += (tp_b[vb] >= t_obs_b[vb])
                            c_a[va] += (tp_a[va] >= t_obs_a[va])
                            c_i[vi] += (tp_i[vi] >= t_obs_i[vi])

                        if tail_mode == "two_sided":
                            if bool(np.isfinite(tp_b).any()):
                                maxb = float(np.nanmax(np.abs(tp_b)))
                                cfb[vb] += (maxb >= np.abs(t_obs_b[vb]))
                            if bool(np.isfinite(tp_a).any()):
                                maxa = float(np.nanmax(np.abs(tp_a)))
                                cfa[va] += (maxa >= np.abs(t_obs_a[va]))
                            if bool(np.isfinite(tp_i).any()):
                                maxi = float(np.nanmax(np.abs(tp_i)))
                                cfi[vi] += (maxi >= np.abs(t_obs_i[vi]))
                        else:
                            if bool(np.isfinite(tp_b).any()):
                                maxb = float(np.nanmax(tp_b))
                                cfb[vb] += (maxb >= t_obs_b[vb])
                            if bool(np.isfinite(tp_a).any()):
                                maxa = float(np.nanmax(tp_a))
                                cfa[va] += (maxa >= t_obs_a[va])
                            if bool(np.isfinite(tp_i).any()):
                                maxi = float(np.nanmax(tp_i))
                                cfi[vi] += (maxi >= t_obs_i[vi])

                    ok_b = np.isfinite(t_obs_b)
                    ok_a = np.isfinite(t_obs_a)
                    ok_i = np.isfinite(t_obs_i)
                    denom = float(int(n_perm) + 1)
                    p_b = np.full(int(roi_ok.sum()), np.nan, dtype=np.float64)
                    p_a = np.full(int(roi_ok.sum()), np.nan, dtype=np.float64)
                    p_i = np.full(int(roi_ok.sum()), np.nan, dtype=np.float64)
                    pf_b = np.full(int(roi_ok.sum()), np.nan, dtype=np.float64)
                    pf_a = np.full(int(roi_ok.sum()), np.nan, dtype=np.float64)
                    pf_i = np.full(int(roi_ok.sum()), np.nan, dtype=np.float64)
                    p_b[ok_b] = (c_b[ok_b] + 1.0) / denom
                    p_a[ok_a] = (c_a[ok_a] + 1.0) / denom
                    p_i[ok_i] = (c_i[ok_i] + 1.0) / denom
                    pf_b[ok_b] = (cfb[ok_b] + 1.0) / denom
                    pf_a[ok_a] = (cfa[ok_a] + 1.0) / denom
                    pf_i[ok_i] = (cfi[ok_i] + 1.0) / denom

                    roi_ok_idx = np.where(roi_ok)[0]
                    p_perm_behavior[roi_ok_idx] = p_b
                    p_perm_age[roi_ok_idx] = p_a
                    p_perm_interaction[roi_ok_idx] = p_i
                    p_fwer_behavior[roi_ok_idx] = pf_b
                    p_fwer_age[roi_ok_idx] = pf_a
                    p_fwer_interaction[roi_ok_idx] = pf_i
        
        for ri, roi in enumerate(rois):
            y = S_z[ri]
            mask = (
                np.isfinite(y)
                & np.isfinite(X_df["behavior"])
                & np.isfinite(X_df["age_model"])
                & np.isfinite(X_df["interaction"])
            )
            
            if mask.sum() > 10:
                y_clean = y[mask]
                X_clean = X_df[mask]
                
                # OLS Regression
                model = sm.OLS(y_clean, X_clean)
                results = model.fit()
                
                rows.append({
                    "roi": roi,
                    "model": mname,
                    "n_subjects": n_sub,
                    "n_pairs": int(mask.sum()),
                    "beta_behavior": results.params['behavior'],
                    "p_behavior": results.pvalues['behavior'],
                    "t_behavior": results.tvalues['behavior'],
                    "beta_age": results.params['age_model'],
                    "p_age": results.pvalues['age_model'],
                    "t_age": results.tvalues['age_model'],
                    "beta_interaction": results.params['interaction'],
                    "p_interaction": results.pvalues['interaction'],
                    "t_interaction": results.tvalues['interaction'],
                    "r_squared": results.rsquared,
                    "r_squared_adj": results.rsquared_adj,
                    "f_pvalue": results.f_pvalue,
                    "p_perm_behavior": float(p_perm_behavior[ri]),
                    "p_perm_age": float(p_perm_age[ri]),
                    "p_perm_interaction": float(p_perm_interaction[ri]),
                    "p_fwer_behavior_model_wise": float(p_fwer_behavior[ri]),
                    "p_fwer_age_model_wise": float(p_fwer_age[ri]),
                    "p_fwer_interaction_model_wise": float(p_fwer_interaction[ri]),
                })
            else:
                rows.append({
                    "roi": roi, "model": mname, "n_subjects": n_sub, "n_pairs": int(mask.sum()),
                    "beta_behavior": np.nan, "p_behavior": np.nan, "t_behavior": np.nan,
                    "beta_age": np.nan, "p_age": np.nan, "t_age": np.nan,
                    "beta_interaction": np.nan, "p_interaction": np.nan, "t_interaction": np.nan,
                    "r_squared": np.nan, "r_squared_adj": np.nan, "f_pvalue": np.nan,
                    "p_perm_behavior": float(p_perm_behavior[ri]),
                    "p_perm_age": float(p_perm_age[ri]),
                    "p_perm_interaction": float(p_perm_interaction[ri]),
                    "p_fwer_behavior_model_wise": float(p_fwer_behavior[ri]),
                    "p_fwer_age_model_wise": float(p_fwer_age[ri]),
                    "p_fwer_interaction_model_wise": float(p_fwer_interaction[ri]),
                })

    out_df = pd.DataFrame(rows)
    out_df["tail"] = str(tail_mode)
    out_df["correction_mode"] = str(mode)
    out_df["n_perm"] = int(n_perm)
    out_df["seed"] = int(seed)
    
    # FDR Correction for interaction terms per model
    out_df["p_fdr_interaction_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"] == mname
        out_df.loc[idx, "p_fdr_interaction_model_wise"] = bh_fdr(out_df.loc[idx, "p_interaction"].to_numpy())
        
    out_df["p_fdr_behavior_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"] == mname
        out_df.loc[idx, "p_fdr_behavior_model_wise"] = bh_fdr(out_df.loc[idx, "p_behavior"].to_numpy())

    out_df["p_fdr_perm_interaction_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"] == mname
        out_df.loc[idx, "p_fdr_perm_interaction_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_interaction"].to_numpy())

    out_df["p_fdr_perm_behavior_model_wise"] = np.nan
    for mname in model_names:
        idx = out_df["model"] == mname
        out_df.loc[idx, "p_fdr_perm_behavior_model_wise"] = bh_fdr(out_df.loc[idx, "p_perm_behavior"].to_numpy())

    out_prefix = "roi_isc_behavior_age_regression"
    out_df.to_csv(stim_dir / f"{out_prefix}.csv", index=False)
    
    return out_df

def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    brain_isc_method: str,
    brain_isc_prefix: Optional[str],
    behavior_isc_method: str,
    behavior_isc_prefix: Optional[str],
    fisher_z_brain: Optional[str],
    fisher_z_behavior: Optional[str],
    normalize_models: bool,
    rank_transform: bool,
    tail: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    brain_isc_prefix = _resolve_prefix(brain_isc_prefix, method=str(brain_isc_method), fallback_template="roi_isc_{method}_by_age")
    behavior_isc_prefix = _resolve_prefix(behavior_isc_prefix, method=str(behavior_isc_method), fallback_template="behavior_isc_{method}_by_age")
    fisher_z_brain_eff = _policy_flag(method=str(brain_isc_method), override=fisher_z_brain)
    fisher_z_behavior_eff = _policy_flag(method=str(behavior_isc_method), override=fisher_z_behavior)

    summary = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_brain_files(stim_dir, brain_isc_prefix=str(brain_isc_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing brain ISC files")
            continue
        if not has_behavior_files(stim_dir, behavior_isc_prefix=str(behavior_isc_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing behavior ISC files")
            continue
            
        print(f"Running regression for {stim_dir.name}...")
        out = run_one(
            stim_dir=stim_dir,
            brain_isc_prefix=str(brain_isc_prefix),
            behavior_isc_prefix=str(behavior_isc_prefix),
            fisher_z_brain=bool(fisher_z_brain_eff),
            fisher_z_behavior=bool(fisher_z_behavior_eff),
            normalize_models=normalize_models,
            rank_transform=rank_transform,
            tail=str(tail),
            correction_mode=str(correction_mode),
            n_perm=int(n_perm),
            seed=int(seed),
        )
        
        # Count significant interactions
        for mname in ("M_nn", "M_conv", "M_div"):
            sub_out = out[out["model"] == mname]
            sig_inter_raw = sub_out[sub_out["p_interaction"] < 0.05]
            sig_inter_fdr = sub_out[sub_out["p_fdr_interaction_model_wise"] < 0.05]
            sig_inter_perm_fdr = sub_out[sub_out["p_fdr_perm_interaction_model_wise"] < 0.05]
            sig_inter_perm_fwer = sub_out[sub_out["p_fwer_interaction_model_wise"] < 0.05]
            
            summary.append({
                "stimulus_type": stim_dir.name,
                "model": mname,
                "n_rois": int(sub_out.shape[0]),
                "n_sig_inter_raw": int(sig_inter_raw.shape[0]),
                "n_sig_inter_fdr": int(sig_inter_fdr.shape[0]),
                "n_sig_inter_perm_fdr": int(sig_inter_perm_fdr.shape[0]),
                "n_sig_inter_perm_fwer": int(sig_inter_perm_fwer.shape[0]),
            })

    out_sum = pd.DataFrame(summary)
    if not out_sum.empty:
        out_sum = out_sum.sort_values(["stimulus_type", "model"])
    out_sum.to_csv(Path(matrix_dir) / "roi_isc_behavior_age_regression_summary.csv", index=False)
    print("All done!")

def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        brain_isc_method=str(args.brain_isc_method),
        brain_isc_prefix=args.brain_isc_prefix,
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        fisher_z_brain=args.fisher_z_brain,
        fisher_z_behavior=args.fisher_z_behavior,
        normalize_models=args.normalize_models,
        rank_transform=args.rank_transform,
        tail=str(args.tail),
        correction_mode=str(args.correction_mode),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
    )

if __name__ == "__main__":
    main()
