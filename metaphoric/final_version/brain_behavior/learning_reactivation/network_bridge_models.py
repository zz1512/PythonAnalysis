#!/usr/bin/env python3
"""Network-level semantic reactivation -> hpc/spatial separation bridge."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [FINAL_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lr_utils import (  # noqa: E402
    HPC_SPATIAL_ROIS,
    SEMANTIC_ROIS,
    bh_fdr,
    write_tsv,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _z(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    sd = vals.std(ddof=1)
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        return vals * np.nan
    return (vals - vals.mean()) / sd


def _build_network_item(table: pd.DataFrame) -> pd.DataFrame:
    key = ["subject", "condition", "original_pair_id", "template_pair_id", "condition_item_id"]
    semantic = table[table["roi"].isin(SEMANTIC_ROIS)].groupby(key, as_index=False).agg(
        semantic_reactivation=("react_pair_mean_avg", "mean"),
        semantic_mapping_asymmetry=("mapping_asymmetry", "mean"),
        semantic_mapping_asymmetry_self_corrected=("mapping_asymmetry_self_corrected", "mean"),
        semantic_target_source_over_self=("target_source_over_self", "mean"),
        n_semantic_rois=("roi", "nunique"),
    )
    hpc = table[table["roi"].isin(HPC_SPATIAL_ROIS)].groupby(key, as_index=False).agg(
        hpc_spatial_separation=("post_edge_specificity", "mean"),
        hpc_spatial_trained_drop=("trained_edge_drop", "mean"),
        retrieval_rebinding=("retrieval_minus_post_pair_similarity", "mean"),
        pre_pair_similarity=("pre_pair_similarity", "mean"),
        hpc_spatial_post_similarity=("post_pair_similarity", "mean"),
        retrieval_pair_similarity=("retrieval_pair_similarity", "mean"),
        memory=("memory", "mean"),
        learning_fluency_shift=("learning_fluency_shift", "mean"),
        n_hpc_spatial_rois=("roi", "nunique"),
    )
    out = semantic.merge(hpc, on=key, how="inner")
    for col in [
        "semantic_reactivation",
        "semantic_mapping_asymmetry",
        "semantic_mapping_asymmetry_self_corrected",
        "semantic_target_source_over_self",
        "hpc_spatial_separation",
        "hpc_spatial_trained_drop",
        "retrieval_rebinding",
        "pre_pair_similarity",
        "hpc_spatial_post_similarity",
        "retrieval_pair_similarity",
        "learning_fluency_shift",
    ]:
        out[f"{col}_z"] = _z(out[col])
    out["condition"] = pd.Categorical(out["condition"].astype(str).str.lower(), categories=["kj", "yy"])
    return out


def _fit_gee(frame: pd.DataFrame, formula: str, outcome: str, model_name: str, family: str = "gaussian") -> pd.DataFrame:
    data = frame.copy().dropna(subset=[outcome, "subject"])
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    rows: list[dict[str, object]] = []
    try:
        fit = smf.gee(
            formula=formula,
            groups="subject",
            data=data,
            family=fam,
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        for term in fit.params.index:
            rows.append(
                {
                    "model": model_name,
                    "outcome": outcome,
                    "term": term,
                    "estimate": float(fit.params[term]),
                    "se": float(fit.bse[term]),
                    "z": float(fit.tvalues[term]),
                    "p": float(fit.pvalues[term]),
                    "n_rows": int(len(data)),
                    "n_subjects": int(data["subject"].nunique()),
                    "status": "ok",
                    "formula": formula,
                }
            )
    except Exception as exc:
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": "__model__",
                "estimate": np.nan,
                "se": np.nan,
                "z": np.nan,
                "p": np.nan,
                "n_rows": int(len(data)),
                "n_subjects": int(data["subject"].nunique()) if "subject" in data else 0,
                "status": f"failed: {exc}",
                "formula": formula,
            }
        )
    return pd.DataFrame(rows)


def _subject_advantage(network_item: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "memory",
        "semantic_reactivation",
        "semantic_mapping_asymmetry",
        "semantic_mapping_asymmetry_self_corrected",
        "semantic_target_source_over_self",
        "hpc_spatial_separation",
        "retrieval_rebinding",
        "learning_fluency_shift",
    ]
    subject_condition = network_item.groupby(["subject", "condition"], as_index=False)[metrics].mean()
    wide = subject_condition.pivot(index="subject", columns="condition", values=metrics)
    rows = []
    for subject in wide.index:
        row = {"subject": subject}
        for metric in metrics:
            try:
                row[f"{metric}_yy_minus_kj"] = float(wide.loc[subject, (metric, "yy")] - wide.loc[subject, (metric, "kj")])
            except Exception:
                row[f"{metric}_yy_minus_kj"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_advantage_models(adv: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("adv_memory_by_semantic_reactivation", "memory_yy_minus_kj ~ semantic_reactivation_yy_minus_kj"),
        ("adv_memory_by_hpc_separation", "memory_yy_minus_kj ~ hpc_spatial_separation_yy_minus_kj"),
        ("adv_memory_by_retrieval_rebinding", "memory_yy_minus_kj ~ retrieval_rebinding_yy_minus_kj"),
        ("adv_memory_by_mapping", "memory_yy_minus_kj ~ semantic_mapping_asymmetry_yy_minus_kj"),
        (
            "adv_memory_by_mapping_self_corrected",
            "memory_yy_minus_kj ~ semantic_mapping_asymmetry_self_corrected_yy_minus_kj",
        ),
        (
            "adv_memory_by_target_source_over_self",
            "memory_yy_minus_kj ~ semantic_target_source_over_self_yy_minus_kj",
        ),
    ]
    rows = []
    for model_name, formula in specs:
        data = adv.copy()
        try:
            fit = smf.ols(formula, data=data).fit()
            for term in fit.params.index:
                rows.append(
                    {
                        "model": model_name,
                        "outcome": "memory_yy_minus_kj",
                        "term": term,
                        "estimate": float(fit.params[term]),
                        "se": float(fit.bse[term]),
                        "t": float(fit.tvalues[term]),
                        "p": float(fit.pvalues[term]),
                        "n_subjects": int(fit.nobs),
                        "r2": float(fit.rsquared),
                        "status": "ok",
                        "formula": formula,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "outcome": "memory_yy_minus_kj",
                    "term": "__model__",
                    "estimate": np.nan,
                    "se": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "n_subjects": int(len(data)),
                    "r2": np.nan,
                    "status": f"failed: {exc}",
                    "formula": formula,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        ok = out["status"].eq("ok") & out["p"].notna()
        for _, idx in out[ok].groupby("term").groups.items():
            out.loc[idx, "q"] = bh_fdr(out.loc[idx, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping")
    parser.add_argument("--table-out-dir", type=Path, default=base_dir / "paper_outputs" / "tables_exploratory" / "learning_reactivation_mapping")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.table_out_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.out_dir / "reactivation_mapping_analysis_table.tsv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis table; run reactivation_to_post_edge.py first: {analysis_path}")
    table = pd.read_csv(analysis_path, sep="\t")
    network_item = _build_network_item(table)
    write_tsv(network_item, args.out_dir / "network_bridge_item.tsv")

    specs = [
        (
            "semantic_to_hpc_separation",
            "hpc_spatial_separation ~ semantic_reactivation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
            "hpc_spatial_separation",
            "gaussian",
        ),
        (
            "mapping_to_hpc_separation",
            "hpc_spatial_separation ~ semantic_mapping_asymmetry_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
            "hpc_spatial_separation",
            "gaussian",
        ),
        (
            "mapping_self_corrected_to_hpc_separation",
            "hpc_spatial_separation ~ semantic_mapping_asymmetry_self_corrected_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
            "hpc_spatial_separation",
            "gaussian",
        ),
        (
            "target_source_over_self_to_hpc_separation",
            "hpc_spatial_separation ~ semantic_target_source_over_self_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
            "hpc_spatial_separation",
            "gaussian",
        ),
        (
            "hpc_separation_to_retrieval_raw_shared_post",
            "retrieval_rebinding ~ hpc_spatial_separation_z * C(condition, Treatment(reference='kj'))",
            "retrieval_rebinding",
            "gaussian",
        ),
        (
            "hpc_separation_to_retrieval_post_control",
            "retrieval_rebinding ~ hpc_spatial_separation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z + hpc_spatial_post_similarity_z",
            "retrieval_rebinding",
            "gaussian",
        ),
        (
            "hpc_separation_to_retrieval_similarity",
            "retrieval_pair_similarity ~ hpc_spatial_separation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z + hpc_spatial_post_similarity_z",
            "retrieval_pair_similarity",
            "gaussian",
        ),
        (
            "retrieval_to_memory",
            "memory ~ retrieval_rebinding_z * C(condition, Treatment(reference='kj'))",
            "memory",
            "binomial",
        ),
    ]
    rows = []
    for model_name, formula, outcome, family in specs:
        rows.append(_fit_gee(network_item, formula, outcome, model_name, family))
    models = pd.concat(rows, ignore_index=True)
    if not models.empty:
        models["q"] = np.nan
        ok = models["status"].eq("ok") & models["p"].notna()
        for _, idx in models[ok].groupby("term").groups.items():
            models.loc[idx, "q"] = bh_fdr(models.loc[idx, "p"])
    adv = _subject_advantage(network_item)
    adv_models = _fit_advantage_models(adv)
    write_tsv(models, args.out_dir / "network_bridge_models.tsv")
    write_tsv(adv, args.out_dir / "network_subject_advantage.tsv")
    write_tsv(adv_models, args.out_dir / "network_subject_advantage_models.tsv")
    write_tsv(models, args.table_out_dir / "table_network_bridge.tsv")
    summary = {
        "script": str(Path(__file__).resolve()),
        "n_network_rows": int(len(network_item)),
        "n_subjects": int(network_item["subject"].nunique()),
        "n_model_rows": int(len(models)),
        "n_advantage_rows": int(len(adv)),
    }
    (args.out_dir / "network_bridge_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
