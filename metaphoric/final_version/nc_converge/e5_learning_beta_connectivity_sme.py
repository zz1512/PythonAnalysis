#!/usr/bin/env python3
"""Append 518 E5: learning-stage beta-series connectivity SME.

Build subject-level beta-series connectivity from E4 ROI activation rows.
For each subject x run x condition x memory group, ROI beta values across
items are correlated and Fisher-z transformed. This targets a plausible
encoding mechanism that may be missed by local RSA/activation analyses.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import add_common_args, default_config, fit_formula, q_for_ok_rows, read_table, write_outputs

MODULE = "e5_learning_beta_connectivity_sme"
DEFAULT_ACTIVATION = Path("qc/nc_converge/e4_learning_activation_sme/learning_activation_roi_item.tsv")
MEMORY_ROIS = {"meta_L_PPC_SPL", "meta_R_PPC_SPL", "meta_R_precuneus", "meta_R_pMTG_pSTS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--activation", type=Path, default=None)
    parser.add_argument("--min-items", type=int, default=5)
    return parser.parse_args()


def _fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def _pair_family(roi_a: str, roi_b: str, net_a: str, net_b: str) -> str:
    if {net_a, net_b} == {"semantic", "hpc_spatial"}:
        if roi_a in MEMORY_ROIS or roi_b in MEMORY_ROIS:
            return "cross_network_memory_locus"
        return "cross_network"
    if roi_a in MEMORY_ROIS or roi_b in MEMORY_ROIS:
        return "within_network_memory_locus"
    return "within_network_other"


def build_connectivity(frame: pd.DataFrame, *, min_items: int) -> pd.DataFrame:
    needed = {"subject", "condition", "run", "condition_item_id", "roi", "network", "activation_z_subject_roi_run", "memory_strict"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"activation table missing columns: {sorted(missing)}")
    data = frame.copy()
    data["memory_strict"] = pd.to_numeric(data["memory_strict"], errors="coerce")
    data = data[data["memory_strict"].isin([0.0, 1.0])].copy()
    roi_network = (
        data[["roi", "network"]]
        .dropna()
        .drop_duplicates("roi")
        .set_index("roi")["network"]
        .astype(str)
        .to_dict()
    )
    rows: list[dict] = []
    group_keys = ["subject", "run", "condition", "memory_strict"]
    for keys, sub in data.groupby(group_keys, dropna=False, sort=False):
        pivot = sub.pivot_table(
            index="condition_item_id",
            columns="roi",
            values="activation_z_subject_roi_run",
            aggfunc="mean",
        )
        if len(pivot) < min_items:
            continue
        rois = [r for r in pivot.columns if pivot[r].notna().sum() >= min_items]
        for roi_a, roi_b in itertools.combinations(sorted(rois), 2):
            pair_data = pivot[[roi_a, roi_b]].dropna()
            if len(pair_data) < min_items:
                continue
            r = pair_data[roi_a].corr(pair_data[roi_b])
            net_a = roi_network.get(roi_a, "")
            net_b = roi_network.get(roi_b, "")
            rows.append(
                {
                    "subject": keys[0],
                    "run": int(keys[1]),
                    "condition": str(keys[2]).lower(),
                    "memory_strict": float(keys[3]),
                    "roi_a": roi_a,
                    "roi_b": roi_b,
                    "network_a": net_a,
                    "network_b": net_b,
                    "roi_pair": f"{roi_a}__{roi_b}",
                    "network_pair": "__".join(sorted([net_a, net_b])),
                    "pair_family": _pair_family(roi_a, roi_b, net_a, net_b),
                    "n_items": int(len(pair_data)),
                    "connectivity_r": float(r) if np.isfinite(r) else np.nan,
                    "connectivity_z": _fisher_z(r),
                }
            )
    return pd.DataFrame(rows)


def fit_models(conn: pd.DataFrame) -> pd.DataFrame:
    if conn.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for (run, roi_pair), sub in conn.groupby(["run", "roi_pair"], dropna=False, sort=False):
        if len(sub) < 20 or sub["memory_strict"].nunique(dropna=True) < 2 or sub["condition"].nunique(dropna=True) < 2:
            continue
        formula = "connectivity_z ~ C(condition, Treatment('kj')) * memory_strict"
        res = fit_formula(sub.dropna(subset=["connectivity_z", "memory_strict"]), formula, family="gaussian", item_col=None)
        first = sub.iloc[0]
        res.insert(0, "run", int(run))
        res.insert(1, "roi_pair", roi_pair)
        res.insert(2, "roi_a", first["roi_a"])
        res.insert(3, "roi_b", first["roi_b"])
        res.insert(4, "network_pair", first["network_pair"])
        res.insert(5, "pair_family", first["pair_family"])
        res.insert(6, "mechanism_model", "learning_beta_connectivity_sme")
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def summarize_family(conn: pd.DataFrame) -> pd.DataFrame:
    if conn.empty:
        return pd.DataFrame()
    return (
        conn.groupby(["run", "condition", "memory_strict", "pair_family"], dropna=False, as_index=False)
        .agg(
            n=("connectivity_z", "count"),
            n_subjects=("subject", "nunique"),
            n_pairs=("roi_pair", "nunique"),
            mean_connectivity_z=("connectivity_z", "mean"),
            sd_connectivity_z=("connectivity_z", "std"),
            mean_n_items=("n_items", "mean"),
        )
    )


def focal_terms(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty or "term" not in models.columns:
        return pd.DataFrame()
    terms = {
        "C(condition, Treatment('kj'))[T.yy]",
        "memory_strict",
        "C(condition, Treatment('kj'))[T.yy]:memory_strict",
    }
    return models[models["term"].isin(terms)].copy()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    activation_path = Path(args.activation or cfg.paper_output_root / DEFAULT_ACTIVATION)
    activation = read_table(activation_path)
    conn = build_connectivity(activation, min_items=int(args.min_items))
    models = fit_models(conn)
    write_outputs(
        cfg,
        MODULE,
        {
            "learning_beta_connectivity_subject.tsv": conn,
            "learning_beta_connectivity_family_summary.tsv": summarize_family(conn),
            "learning_beta_connectivity_models.tsv": models,
            "learning_beta_connectivity_focal_terms.tsv": focal_terms(models),
            "manifest.tsv": pd.DataFrame(
                [
                    {
                        "activation_path": str(activation_path),
                        "activation_rows": int(len(activation)),
                        "connectivity_rows": int(len(conn)),
                        "n_roi_pairs": int(conn["roi_pair"].nunique()) if not conn.empty else 0,
                        "min_items": int(args.min_items),
                        "status": "ok" if not conn.empty else "empty",
                    }
                ]
            ),
        },
    )


if __name__ == "__main__":
    main()
