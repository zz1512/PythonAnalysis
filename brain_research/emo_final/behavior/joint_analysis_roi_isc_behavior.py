#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

if __package__ in {None, "", "behavior"}:
    THIS_DIR = Path(__file__).resolve().parent
    EMO_DIR = THIS_DIR.parent
    if str(EMO_DIR) not in sys.path:
        sys.path.insert(0, str(EMO_DIR))

    from joint_analysis_roi_isc_dev_models import assoc, bh_fdr, fisher_z, zscore_1d  # noqa: E402
else:
    from ..joint_analysis_roi_isc_dev_models import assoc, bh_fdr, fisher_z, zscore_1d  # noqa: E402


DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Permutation test for ROI ISC vs behavior ISC")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--brain-repr-prefix", type=str, default=None)
    p.add_argument("--brain-isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--brain-isc-prefix", type=str, default=None)
    p.add_argument("--behavior-repr-prefix", type=str, default=None)
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis", choices=("spearman", "pearson", "euclidean", "mahalanobis"))
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--assoc-method", type=str, default="spearman", choices=("pearson", "spearman"))
    p.add_argument("--correction-mode", type=str, default="perm_fwer_fdr", choices=("fdr_only", "perm_fwer_fdr"))
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fisher-z-brain", type=str, default=None)
    p.add_argument("--fisher-z-behavior", type=str, default=None)
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
    behavior_set = set(behavior_subjects)
    overlap = [s for s in brain_subjects if s in behavior_set]
    if len(overlap) < 3:
        raise ValueError("Fewer than 3 overlapping subjects between brain ISC and behavior ISC")

    brain_idx = {s: i for i, s in enumerate(brain_subjects)}
    behavior_idx = {s: i for i, s in enumerate(behavior_subjects)}
    ib = [brain_idx[s] for s in overlap]
    ih = [behavior_idx[s] for s in overlap]
    brain_isc = np.asarray(brain_isc, dtype=np.float32)[:, ib, :][:, :, ib]
    behavior_isc = np.asarray(behavior_isc, dtype=np.float32)[np.ix_(ih, ih)]
    age_map = dict(zip(brain_subjects_df["subject"].astype(str), brain_subjects_df["age"].astype(float)))
    ages = [float(age_map[s]) for s in overlap]
    return brain_isc, behavior_isc, overlap, ages


def run_one(
    stim_dir: Path,
    brain_isc_prefix: str,
    behavior_isc_prefix: str,
    assoc_method: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    fisher_z_brain: bool,
    fisher_z_behavior: bool,
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
    n_pairs = int(iu.size)

    assoc_m = str(assoc_method).strip().lower()
    S_z = []
    for ri in range(len(rois)):
        v = brain_isc[ri][iu, ju]
        if bool(fisher_z_brain):
            v = fisher_z(v)
        if assoc_m == "spearman":
            mask_v = np.isfinite(v)
            vr = np.full_like(v, np.nan, dtype=np.float32)
            vr[mask_v] = rankdata(v[mask_v]).astype(np.float32, copy=False)
            S_z.append(zscore_1d(vr))
        else:
            S_z.append(zscore_1d(v))
    S_z = np.stack(S_z, axis=0).astype(np.float32)

    beh_vec = behavior_isc[iu, ju].astype(np.float32)
    if bool(fisher_z_behavior):
        beh_vec = fisher_z(beh_vec)
    r_obs = assoc(S_z, beh_vec, method=str(assoc_method)).astype(np.float32)

    p_raw = np.full(len(rois), np.nan, dtype=np.float64)
    p_fwer = np.full(len(rois), np.nan, dtype=np.float64)
    mode = str(correction_mode).strip().lower()

    if mode == "perm_fwer_fdr":
        rng = np.random.default_rng(int(seed))
        c_raw = np.zeros(len(rois), dtype=np.int64)
        c_fwer = np.zeros(len(rois), dtype=np.int64)
        finite_obs_idx = np.where(np.isfinite(r_obs))[0]
        for _ in range(int(n_perm)):
            perm = rng.permutation(n_sub)
            beh_perm = behavior_isc[np.ix_(perm, perm)]
            vec_perm = beh_perm[iu, ju].astype(np.float32)
            if bool(fisher_z_behavior):
                vec_perm = fisher_z(vec_perm)
            r_perm = assoc(S_z, vec_perm, method=str(assoc_method)).astype(np.float32)
            valid = np.isfinite(r_obs) & np.isfinite(r_perm)
            c_raw[valid] += (r_perm[valid] >= r_obs[valid])
            if finite_obs_idx.size > 0:
                valid_perm = np.isfinite(r_perm[finite_obs_idx])
                if np.any(valid_perm):
                    max_stat = float(np.max(r_perm[finite_obs_idx][valid_perm]))
                    c_fwer[finite_obs_idx] += (max_stat >= r_obs[finite_obs_idx])
        ok = np.isfinite(r_obs)
        p_raw[ok] = (c_raw[ok] + 1.0) / (float(n_perm) + 1.0)
        p_fwer[ok] = (c_fwer[ok] + 1.0) / (float(n_perm) + 1.0)
    elif mode == "fdr_only":
        ok = np.isfinite(r_obs)
        if n_pairs <= 2:
            raise ValueError("Too few subject pairs for parametric inference")
        dfree = float(n_pairs - 2)
        r = np.clip(r_obs[ok].astype(np.float64), -0.999999, 0.999999)
        t_stat = r * np.sqrt(dfree / np.maximum(1.0 - r * r, 1e-12))
        log_p = t.logsf(t_stat, df=dfree)
        min_log = float(np.log(np.finfo(np.float64).tiny))
        p_raw[ok] = np.exp(np.maximum(log_p, min_log))
    else:
        raise ValueError(f"Unsupported correction mode: {correction_mode}")

    out_df = pd.DataFrame(
        {
            "roi": rois,
            "r_obs": r_obs.astype(float),
            "p_perm_one_tailed": p_raw.astype(float),
            "p_fwer_model_wise": p_fwer.astype(float),
            "n_subjects": int(n_sub),
            "n_pairs": int(n_pairs),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "correction_mode": str(mode),
            "brain_isc_prefix": str(brain_isc_prefix),
            "behavior_isc_prefix": str(behavior_isc_prefix),
            "assoc_method": str(assoc_method),
            "fisher_z_brain": bool(fisher_z_brain),
            "fisher_z_behavior": bool(fisher_z_behavior),
        }
    ).sort_values("roi")
    out_df["p_fdr_bh_global"] = bh_fdr(out_df["p_perm_one_tailed"].to_numpy(dtype=float))

    out_prefix = "roi_isc_behavior_perm_fwer"
    out_df.to_csv(stim_dir / f"{out_prefix}.csv", index=False)
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(stim_dir / f"{out_prefix}_subjects_sorted.csv", index=False)
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
    return out_df


def run(
    matrix_dir: Path,
    stimulus_dir_name: str,
    brain_repr_prefix: Optional[str],
    brain_isc_method: str,
    brain_isc_prefix: Optional[str],
    behavior_repr_prefix: Optional[str],
    behavior_isc_method: str,
    behavior_isc_prefix: Optional[str],
    assoc_method: str,
    correction_mode: str,
    n_perm: int,
    seed: int,
    fisher_z_brain: Optional[str],
    fisher_z_behavior: Optional[str],
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
        if brain_repr_prefix is not None and not (stim_dir / f"{brain_repr_prefix}_subjects.csv").exists():
            print(f"[SKIP] {stim_dir.name}: missing brain repr files for {brain_repr_prefix}")
            continue
        if behavior_repr_prefix is not None and not (stim_dir / f"{behavior_repr_prefix}_subjects.csv").exists():
            print(f"[SKIP] {stim_dir.name}: missing behavior repr files for {behavior_repr_prefix}")
            continue
        if not has_brain_files(stim_dir, brain_isc_prefix=str(brain_isc_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing brain ISC files")
            continue
        if not has_behavior_files(stim_dir, behavior_isc_prefix=str(behavior_isc_prefix)):
            print(f"[SKIP] {stim_dir.name}: missing behavior ISC files")
            continue
        out = run_one(
            stim_dir=stim_dir,
            brain_isc_prefix=str(brain_isc_prefix),
            behavior_isc_prefix=str(behavior_isc_prefix),
            assoc_method=str(assoc_method),
            correction_mode=str(correction_mode),
            n_perm=int(n_perm),
            seed=int(seed),
            fisher_z_brain=bool(fisher_z_brain_eff),
            fisher_z_behavior=bool(fisher_z_behavior_eff),
        )
        sig_fwer = out[(out["r_obs"] > 0) & (out["p_fwer_model_wise"] <= 0.05)]
        sig_fdr = out[(out["r_obs"] > 0) & (out["p_fdr_bh_global"] <= 0.05)]
        summary.append(
            {
                "stimulus_type": stim_dir.name,
                "n_rows": int(out.shape[0]),
                "n_sig_pos_fwer": int(sig_fwer.shape[0]),
                "n_sig_pos_fdr": int(sig_fdr.shape[0]),
                "brain_isc_prefix": str(brain_isc_prefix),
                "behavior_isc_prefix": str(behavior_isc_prefix),
            }
        )

    out = pd.DataFrame(summary)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(Path(matrix_dir) / "roi_isc_behavior_perm_fwer_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        brain_repr_prefix=args.brain_repr_prefix,
        brain_isc_method=str(args.brain_isc_method),
        brain_isc_prefix=args.brain_isc_prefix,
        behavior_repr_prefix=args.behavior_repr_prefix,
        behavior_isc_method=str(args.behavior_isc_method),
        behavior_isc_prefix=args.behavior_isc_prefix,
        assoc_method=str(args.assoc_method),
        correction_mode=str(args.correction_mode),
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        fisher_z_brain=args.fisher_z_brain,
        fisher_z_behavior=args.fisher_z_behavior,
    )


if __name__ == "__main__":
    main()
