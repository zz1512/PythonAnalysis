#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from joint_analysis_roi_isc_dev_models import bh_fdr, build_models, fisher_z, zscore_1d  # noqa: E402
else:
    from ..joint_analysis_roi_isc_dev_models import bh_fdr, build_models, fisher_z, zscore_1d  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
MODEL_NAMES = ("M_nn", "M_conv", "M_div")
OUT_PREFIX = "behavior_isc_dev_models_perm_fwer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate behavior ISC with three age models")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus", choices=("by_stimulus", "by_emotion"))
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--tail", type=str, default="two_sided", choices=("positive", "two_sided"))
    p.add_argument("--correction-mode", type=str, default="perm_fwer_fdr", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
    p.add_argument("--no-normalize-models", action="store_false", dest="normalize_models", default=True)
    return p.parse_args()


def _policy_flag(method: str, override: Optional[str]) -> bool:
    if override is not None:
        return str(override).strip().lower() == "true"
    return str(method).strip().lower() in {"pearson", "spearman"}


def _resolve_prefix(prefix: Optional[str], method: str) -> str:
    return str(prefix) if prefix is not None else f"behavior_isc_{str(method).strip().lower()}_by_age"


def has_behavior_files(stim_dir: Path, behavior_isc_prefix: str) -> bool:
    required = (
        stim_dir / f"{behavior_isc_prefix}.npy",
        stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv",
    )
    return all(p.exists() for p in required)


def _assoc_one(x: np.ndarray, m: np.ndarray, method: str) -> float:
    xv = np.asarray(x, dtype=np.float32).reshape(-1)
    mv = np.asarray(m, dtype=np.float32).reshape(-1)
    ok = np.isfinite(xv) & np.isfinite(mv)
    if int(ok.sum()) < 3:
        return float("nan")

    if str(method).strip().lower() == "spearman":
        xr = rankdata(xv[ok]).astype(np.float32, copy=False)
        mr = rankdata(mv[ok]).astype(np.float32, copy=False)
        xz = zscore_1d(xr)
        mz = zscore_1d(mr)
    else:
        xz = zscore_1d(xv[ok])
        mz = zscore_1d(mv[ok])
    keep = np.isfinite(xz) & np.isfinite(mz)
    if int(keep.sum()) < 3:
        return float("nan")
    return float(np.mean(xz[keep] * mz[keep]))


def _p_from_counts(c: np.ndarray, n_perm: int, valid: np.ndarray) -> np.ndarray:
    p = np.full(c.shape, np.nan, dtype=np.float64)
    p[valid] = (c[valid].astype(float) + 1.0) / (float(n_perm) + 1.0)
    return p


def run_one(
    stim_dir: Path,
    behavior_isc_prefix: str,
    behavior_isc_method: str,
    fisher_z_behavior: bool,
    assoc_method: str,
    tail: str,
    correction_mode: str,
    normalize_models: bool,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    behavior_isc = np.load(stim_dir / f"{behavior_isc_prefix}.npy")
    sub_df = pd.read_csv(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv")
    subjects = sub_df["subject"].astype(str).tolist()
    ages = sub_df["age"].astype(float).to_numpy()

    n_sub = len(subjects)
    iu, ju = np.triu_indices(n_sub, k=1)
    n_pairs = int(iu.size)
    beh_vec = np.asarray(behavior_isc, dtype=np.float32)[iu, ju]
    if bool(fisher_z_behavior):
        beh_vec = fisher_z(beh_vec)

    models = build_models(ages, iu, ju, normalize=bool(normalize_models))
    r_obs = np.asarray([_assoc_one(beh_vec, m, method=str(assoc_method)) for m in models], dtype=np.float32)

    tail_mode = str(tail).strip().lower()
    mode = str(correction_mode).strip().lower()
    p_raw = np.full(len(MODEL_NAMES), np.nan, dtype=np.float64)
    p_fwer = np.full(len(MODEL_NAMES), np.nan, dtype=np.float64)
    valid_obs = np.isfinite(r_obs)

    if mode == "perm_fwer_fdr":
        c_raw = np.zeros(len(MODEL_NAMES), dtype=np.int64)
        c_fwer = np.zeros(len(MODEL_NAMES), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_perm)):
            ages_p = ages[rng.permutation(n_sub)]
            m_perm = build_models(ages_p, iu, ju, normalize=bool(normalize_models))
            r_perm = np.asarray([_assoc_one(beh_vec, m, method=str(assoc_method)) for m in m_perm], dtype=np.float32)
            valid = np.isfinite(r_perm) & valid_obs
            if tail_mode == "two_sided":
                c_raw[valid] += np.abs(r_perm[valid]) >= np.abs(r_obs[valid])
                if np.any(valid):
                    max_stat = float(np.max(np.abs(r_perm[valid])))
                    c_fwer[valid_obs] += max_stat >= np.abs(r_obs[valid_obs])
            else:
                c_raw[valid] += r_perm[valid] >= r_obs[valid]
                if np.any(valid):
                    max_stat = float(np.max(r_perm[valid]))
                    c_fwer[valid_obs] += max_stat >= r_obs[valid_obs]
        p_raw = _p_from_counts(c_raw, int(n_perm), valid_obs)
        p_fwer = _p_from_counts(c_fwer, int(n_perm), valid_obs)
    elif mode == "fdr_only":
        if n_pairs <= 2:
            raise ValueError("Too few subject pairs for parametric p-values")
        dfree = float(n_pairs - 2)
        r = np.clip(r_obs[valid_obs].astype(np.float64), -0.999999, 0.999999)
        t_stat = r * np.sqrt(dfree / np.maximum(1.0 - r * r, 1e-12))
        if tail_mode == "two_sided":
            p_raw[valid_obs] = 2.0 * t.sf(np.abs(t_stat), df=dfree)
        else:
            p_raw[valid_obs] = t.sf(t_stat, df=dfree)
    else:
        raise ValueError(f"Unsupported correction_mode: {correction_mode}")

    out = pd.DataFrame(
        {
            "model": list(MODEL_NAMES),
            "r_obs": r_obs.astype(float),
            "p_perm": p_raw.astype(float),
            "p_fwer_global": p_fwer.astype(float),
            "p_fdr_bh_global": bh_fdr(p_raw),
            "n_subjects": int(n_sub),
            "n_pairs": int(n_pairs),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "tail": str(tail_mode),
            "correction_mode": str(mode),
            "behavior_isc_prefix": str(behavior_isc_prefix),
            "behavior_isc_method": str(behavior_isc_method),
            "fisher_z_behavior": bool(fisher_z_behavior),
            "assoc_method": str(assoc_method),
            "normalize_models": bool(normalize_models),
        }
    ).sort_values("model")
    out.to_csv(stim_dir / f"{OUT_PREFIX}.csv", index=False)
    sub_df[["subject", "age"]].to_csv(stim_dir / f"{OUT_PREFIX}_subjects_sorted.csv", index=False)
    return out


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    behavior_isc_method: str,
    behavior_isc_prefix: Optional[str],
    assoc_method: str,
    tail: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    fisher_z_behavior: Optional[str],
    normalize_models: bool,
) -> None:
    by_stim = Path(matrix_dir) / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"Cannot find directory: {by_stim}")

    prefix = _resolve_prefix(behavior_isc_prefix, method=str(behavior_isc_method))
    fisher_z_eff = _policy_flag(method=str(behavior_isc_method), override=fisher_z_behavior)
    summary = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        if not has_behavior_files(stim_dir, behavior_isc_prefix=str(prefix)):
            print(f"[SKIP] {stim_dir.name}: missing behavior ISC files for {prefix}")
            continue
        out = run_one(
            stim_dir=stim_dir,
            behavior_isc_prefix=str(prefix),
            behavior_isc_method=str(behavior_isc_method),
            fisher_z_behavior=bool(fisher_z_eff),
            assoc_method=str(assoc_method),
            tail=str(tail),
            correction_mode=str(correction_mode),
            normalize_models=bool(normalize_models),
            n_perm=int(n_perm),
            seed=int(seed),
        )
        row = {"stimulus_type": stim_dir.name, "n_rows": int(out.shape[0])}
        for model in MODEL_NAMES:
            sub = out[out["model"].astype(str) == str(model)]
            if not sub.empty:
                row[f"{model}_r_obs"] = float(sub.iloc[0]["r_obs"])
                row[f"{model}_p_fdr_bh_global"] = float(sub.iloc[0]["p_fdr_bh_global"])
                row[f"{model}_p_fwer_global"] = float(sub.iloc[0]["p_fwer_global"])
        summary.append(row)

    summary_name = f"{OUT_PREFIX}_{str(stimulus_dir_name)}_summary.csv"
    pd.DataFrame(summary).sort_values("stimulus_type").to_csv(Path(matrix_dir) / summary_name, index=False)


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        assoc_method=str(args.assoc_method),
        tail=str(args.tail),
        correction_mode=str(args.correction_mode),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        fisher_z_behavior=args.fisher_z_behavior,
        normalize_models=bool(args.normalize_models),
    )


if __name__ == "__main__":
    main()
