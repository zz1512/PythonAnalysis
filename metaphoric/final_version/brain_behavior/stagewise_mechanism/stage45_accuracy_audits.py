#!/usr/bin/env python3
"""Stage 4.5 accuracy audits before evidence integration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
for path in [FINAL_ROOT, FINAL_ROOT / "brain_behavior"]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common.final_utils import ensure_dir, write_table  # noqa: E402
from brain_behavior.build_learning_post_memory_item_table import (  # noqa: E402
    CONDITIONS,
    VALID_ORIGINAL_IDS,
    _condition_item_id,
    _corr_matrix,
    _load_learning_condition,
    _load_mask,
    _load_template_map,
    _masked_samples,
)


MATERIAL_COVARS = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
KEY_COLS = ["subject", "condition", "original_pair_id", "template_pair_id", "condition_item_id"]
SEMANTIC_ROIS = {
    "meta_L_IFG",
    "meta_R_IFG",
    "meta_L_temporal_pole",
    "meta_R_temporal_pole",
    "meta_L_AG",
    "meta_R_AG",
    "meta_L_pMTG_pSTS",
    "meta_R_pMTG_pSTS",
}
HPC_SPATIAL_ROIS = {
    "meta_L_hippocampus",
    "meta_R_hippocampus",
    "meta_L_PPA_PHG",
    "meta_R_PPA_PHG",
    "meta_L_RSC_PCC",
    "meta_R_RSC_PCC",
    "meta_L_PPC_SPL",
    "meta_R_PPC_SPL",
    "meta_L_precuneus",
    "meta_R_precuneus",
}
QUALITY_METRICS = [
    "pattern_norm",
    "pattern_mean",
    "pattern_sd",
    "pattern_variance",
    "within_run_mean_similarity",
    "valid_voxel_count",
]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def zscore(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    sd = vals.std(ddof=0)
    if pd.isna(sd) or math.isclose(float(sd), 0.0):
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (vals - vals.mean()) / sd


def fdr_by_family(df: pd.DataFrame, family_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["q"] = np.nan
    if out.empty or "p" not in out:
        return out
    ok = out["p"].notna() & np.isfinite(pd.to_numeric(out["p"], errors="coerce"))
    for _, idx in out[ok].groupby(family_cols, dropna=False).groups.items():
        p = pd.to_numeric(out.loc[idx, "p"], errors="coerce")
        _, q, _, _ = multipletests(p, method="fdr_bh")
        out.loc[idx, "q"] = q
    return out


def available_covariates(frame: pd.DataFrame, min_coverage: float = 0.8) -> list[str]:
    covs: list[str] = []
    for col in MATERIAL_COVARS:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().mean() >= min_coverage and vals.nunique(dropna=True) > 1:
            covs.append(col)
    return covs


def md_table(frame: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def formula_columns(formula: str, outcome: str, frame: pd.DataFrame) -> list[str]:
    cols = ["subject", "condition", "condition_item_id", outcome]
    for col in frame.columns:
        if col in formula:
            cols.append(col)
    return sorted(set(col for col in cols if col in frame.columns))


def _term_rows(result, info: dict[str, object], data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term, estimate in result.params.items():
        if term.endswith(" Var") or term.startswith("item Var"):
            continue
        se = result.bse.get(term, np.nan)
        stat = estimate / se if pd.notna(se) and se != 0 else np.nan
        rows.append(
            {
                **info,
                "term": term,
                "contrast_type": "model_term",
                "estimate": float(estimate),
                "se": float(se) if pd.notna(se) else np.nan,
                "stat": float(stat) if pd.notna(stat) else np.nan,
                "p": float(result.pvalues.get(term, np.nan)),
                "n_rows": len(data),
                "n_subjects": int(data["subject"].nunique()),
                "n_condition_items": int(data["condition_item_id"].nunique())
                if "condition_item_id" in data
                else np.nan,
            }
        )
    return rows


def _linear_contrast(result, weights: dict[str, float]) -> tuple[float, float, float, float]:
    params = result.params
    cov = result.cov_params()
    if any(term not in params.index for term in weights):
        return np.nan, np.nan, np.nan, np.nan
    estimate = sum(float(params.loc[term]) * weight for term, weight in weights.items())
    var = 0.0
    for term_a, weight_a in weights.items():
        for term_b, weight_b in weights.items():
            if term_a not in cov.index or term_b not in cov.columns:
                return np.nan, np.nan, np.nan, np.nan
            var += weight_a * weight_b * float(cov.loc[term_a, term_b])
    se = math.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan
    stat = estimate / se if se and np.isfinite(se) else np.nan
    p = float(2 * stats.norm.sf(abs(stat))) if np.isfinite(stat) else np.nan
    return estimate, se, stat, p


def _condition_slope_rows(result, info: dict[str, object], data: pd.DataFrame, predictor: str) -> list[dict[str, object]]:
    interaction = f"{predictor}:C(condition, Treatment(reference='kj'))[T.yy]"
    if interaction not in result.params.index:
        interaction = f"C(condition, Treatment(reference='kj'))[T.yy]:{predictor}"
    specs = {
        f"{predictor}_slope_kj": {predictor: 1.0},
        f"{predictor}_slope_yy": {predictor: 1.0, interaction: 1.0},
        f"{predictor}_slope_yy_minus_kj": {interaction: 1.0},
    }
    rows = []
    for term, weights in specs.items():
        estimate, se, stat, p = _linear_contrast(result, weights)
        rows.append(
            {
                **info,
                "term": term,
                "contrast_type": "condition_slope",
                "estimate": estimate,
                "se": se,
                "stat": stat,
                "p": p,
                "n_rows": len(data),
                "n_subjects": int(data["subject"].nunique()),
                "n_condition_items": int(data["condition_item_id"].nunique()),
            }
        )
    return rows


def fit_mixed_or_cluster(
    frame: pd.DataFrame,
    *,
    formula: str,
    outcome: str,
    predictor: str | None,
    info: dict[str, object],
    min_rows: int = 80,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = frame.replace([np.inf, -np.inf], np.nan).copy()
    needed = formula_columns(formula, outcome, data)
    data = data.dropna(subset=needed).copy()
    data["subject"] = data["subject"].astype(str)
    if "condition" in data:
        data["condition"] = pd.Categorical(data["condition"].astype(str), categories=["kj", "yy"])
    if "condition_item_id" in data:
        data["condition_item_id"] = data["condition_item_id"].astype(str)
    info = {
        **info,
        "formula": formula,
        "status": "not_run",
        "converged": False,
        "fallback_used": "none",
        "message": "",
        "n_rows": len(data),
        "n_subjects": int(data["subject"].nunique()) if "subject" in data else 0,
        "n_condition_items": int(data["condition_item_id"].nunique()) if "condition_item_id" in data else np.nan,
    }
    if len(data) < min_rows or info["n_subjects"] < 5:
        info["status"] = "skipped_too_few_rows"
        return [], info

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            md = smf.mixedlm(
                formula,
                data,
                groups=data["subject"],
                vc_formula={"item": "0 + C(condition_item_id)"} if "condition_item_id" in data else None,
            )
            result = md.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            info["converged"] = bool(getattr(result, "converged", False))
            info["status"] = "ok" if info["converged"] else "mixed_not_converged"
        except Exception as exc:
            info["fallback_used"] = "ols_cluster_subject"
            info["message"] = repr(exc)
            try:
                result = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["subject"]})
                info["converged"] = True
                info["status"] = "fallback_ols_cluster_subject"
            except Exception as exc2:
                info["status"] = "failed"
                info["message"] = f"{info['message']}; fallback={repr(exc2)}"
                return [], info
        if caught:
            info["message"] = " | ".join(sorted({str(w.message)[:180] for w in caught}))

    rows = _term_rows(result, info, data)
    if predictor:
        rows.extend(_condition_slope_rows(result, info, data, predictor))
    return rows, info


def fit_cluster_ols(
    frame: pd.DataFrame,
    *,
    formula: str,
    outcome: str,
    info: dict[str, object],
    min_rows: int = 80,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = frame.replace([np.inf, -np.inf], np.nan).copy()
    needed = formula_columns(formula, outcome, data)
    data = data.dropna(subset=needed).copy()
    data["subject"] = data["subject"].astype(str)
    if "condition" in data:
        data["condition"] = pd.Categorical(data["condition"].astype(str), categories=["kj", "yy"])
    info = {
        **info,
        "formula": formula,
        "status": "not_run",
        "converged": False,
        "fallback_used": "none",
        "message": "",
        "n_rows": len(data),
        "n_subjects": int(data["subject"].nunique()) if "subject" in data else 0,
        "n_condition_items": int(data["condition_item_id"].nunique()) if "condition_item_id" in data else np.nan,
    }
    if len(data) < min_rows or info["n_subjects"] < 5:
        info["status"] = "skipped_too_few_rows"
        return [], info
    try:
        result = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["subject"]})
        info["status"] = "ok"
        info["converged"] = True
    except Exception as exc:
        info["status"] = "failed"
        info["message"] = repr(exc)
        return [], info
    return _term_rows(result, info, data), info


def audit1_memory_difference(stage1_network_item: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = [
        *KEY_COLS,
        "memory",
        "retrieval_pair_similarity",
        "hpc_spatial_post_similarity",
        "hpc_spatial_rebinding",
        *MATERIAL_COVARS,
    ]
    base = pd.read_csv(stage1_network_item, sep="\t", usecols=lambda c: c in keep, low_memory=False)
    base["condition"] = base["condition"].astype(str).str.lower()
    base = base[base["condition"].isin(["kj", "yy"])].copy()
    base["memory_score"] = pd.to_numeric(base["memory"], errors="coerce")
    metric_map = {
        "retrieval_pair_similarity": "hpc_spatial_retrieval_pair_similarity",
        "hpc_spatial_post_similarity": "hpc_spatial_post_pair_similarity",
        "hpc_spatial_rebinding": "hpc_spatial_rebinding_difference",
    }
    rows: list[pd.DataFrame] = []
    id_cols = [*KEY_COLS, "memory", "memory_score", *[c for c in MATERIAL_COVARS if c in base.columns]]
    for col, name in metric_map.items():
        tmp = base[id_cols + [col]].rename(columns={col: "component_metric"})
        tmp["component_name"] = name
        rows.append(tmp)
    components = pd.concat(rows, ignore_index=True)
    components["component_metric"] = pd.to_numeric(components["component_metric"], errors="coerce")
    components["component_metric_z"] = components.groupby("component_name", dropna=False)["component_metric"].transform(zscore)
    covs = available_covariates(components)
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    formula = f"component_metric_z ~ memory_score * C(condition, Treatment(reference='kj')){cov_expr}"
    model_rows, conv = [], []
    for component_name, sub in components.groupby("component_name", sort=False):
        rows_i, info = fit_mixed_or_cluster(
            sub,
            formula=formula,
            outcome="component_metric_z",
            predictor="memory_score",
            info={
                "analysis_id": "stage45_memory_difference_decomposition",
                "model_name": "memory_component_primary",
                "component_name": component_name,
            },
        )
        model_rows.extend(rows_i)
        conv.append(info)
    models = pd.DataFrame(model_rows)
    if not models.empty:
        models = fdr_by_family(models, ["model_name", "term", "contrast_type"])
    convergence = pd.DataFrame(conv)
    write_table(components, out_dir / "stage45_memory_difference_components.tsv")
    write_table(models, out_dir / "stage45_memory_difference_models.tsv")
    write_table(convergence, out_dir / "stage45_memory_difference_convergence.tsv")
    write_review_audit1(out_dir, components, models, convergence)
    return components, models


def write_review_audit1(out_dir: Path, components: pd.DataFrame, models: pd.DataFrame, convergence: pd.DataFrame) -> None:
    slopes = models[models["contrast_type"].eq("condition_slope")].copy()
    show = slopes[
        ["component_name", "term", "estimate", "se", "p", "q", "n_rows", "n_subjects", "status", "fallback_used"]
    ].sort_values(["term", "p"], na_position="last")
    means = (
        components.groupby(["component_name", "condition"], dropna=False)
        .agg(
            n=("component_metric", "size"),
            mean_metric=("component_metric", "mean"),
            mean_memory=("memory_score", "mean"),
        )
        .reset_index()
    )
    conv_show = convergence[["component_name", "status", "fallback_used", "n_rows", "n_subjects", "n_condition_items"]]
    text = f"""# Stage 4.5 Audit 1: Memory Bridge Difference-Score Decomposition

Purpose: decompose the Stage 4 memory bridge so `hpc_spatial_rebinding = retrieval_pair_similarity - post_pair_similarity` is not interpreted blindly.

Primary model:

```text
component_metric_z ~ memory_score * condition + material_covariates
                   + (1 | subject) + (1 | condition_item_id)
```

Positive memory slopes mean later remembered items have larger component values.

## Component Means

{md_table(means)}

## Memory Slopes

{md_table(show)}

## Convergence

{md_table(conv_show)}

Interpretation rule:

- Retrieval-similarity-driven result supports retrieval endpoint re-binding / mnemonic reinstatement.
- Post-similarity-driven result should be written as prior post separation, not retrieval-specific re-binding.
- Mixed result must be reported as a difference-score endpoint with component caveat.
"""
    (out_dir / "stage45_memory_difference_review.md").write_text(text, encoding="utf-8")


def _load_roi_rows(roi_manifest: Path, roi_sets: list[str]) -> pd.DataFrame:
    manifest = pd.read_csv(roi_manifest, sep="\t")
    out = manifest[
        manifest["roi_set"].isin(roi_sets)
        & manifest["include_in_rsa"].astype(bool)
        & manifest["roi_name"].isin(SEMANTIC_ROIS | HPC_SPATIAL_ROIS)
    ].copy()
    return out.sort_values(["roi_set", "roi_name"])


def compute_stage2_pattern_quality(
    *,
    pattern_root: Path,
    roi_manifest: Path,
    stimuli_template: Path,
    roi_sets: list[str],
    subjects: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    template = _load_template_map(stimuli_template)
    template_lookup = {
        condition: template[template["condition"].eq(condition)].set_index("original_pair_id")["template_pair_id"].to_dict()
        for condition in CONDITIONS
    }
    rois = _load_roi_rows(roi_manifest, roi_sets)
    subject_dirs = sorted([p for p in pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if subjects:
        wanted = set(subjects)
        subject_dirs = [p for p in subject_dirs if p.name in wanted]
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        for condition in CONDITIONS:
            try:
                meta, data, image_path = _load_learning_condition(subject_dir, condition)
            except Exception as exc:
                qc_rows.append({"subject": subject_dir.name, "condition": condition, "ok": False, "message": repr(exc)})
                continue
            run3 = meta[meta["run"].eq(3)].drop_duplicates("original_pair_id", keep=False).copy()
            run4 = meta[meta["run"].eq(4)].drop_duplicates("original_pair_id", keep=False).copy()
            common_ids = sorted(
                set(int(v) for v in run3["original_pair_id"].dropna().tolist())
                & set(int(v) for v in run4["original_pair_id"].dropna().tolist())
                & set(VALID_ORIGINAL_IDS[condition])
            )
            run_index = {
                3: {int(row.original_pair_id): int(idx) for idx, row in run3.iterrows()},
                4: {int(row.original_pair_id): int(idx) for idx, row in run4.iterrows()},
            }
            for _, roi_row in rois.iterrows():
                roi_set = str(roi_row["roi_set"])
                roi = str(roi_row["roi_name"])
                mask_path = Path(str(roi_row["mask_path"]))
                try:
                    samples = _masked_samples(data, _load_mask(mask_path), image_path, mask_path)
                    by_run = {
                        run: samples[[run_index[run][item] for item in common_ids], :]
                        for run in [3, 4]
                    }
                    within = {}
                    for run, mat in by_run.items():
                        sim = _corr_matrix(mat, mat)
                        np.fill_diagonal(sim, np.nan)
                        within[run] = np.nanmean(sim, axis=1)
                    for run, mat in by_run.items():
                        for pos, original_id in enumerate(common_ids):
                            pattern = np.asarray(mat[pos, :], dtype=float)
                            finite = np.isfinite(pattern)
                            vals = pattern[finite]
                            rows.append(
                                {
                                    "subject": subject_dir.name,
                                    "roi_set": roi_set,
                                    "roi": roi,
                                    "network": "semantic" if roi in SEMANTIC_ROIS else "hpc_spatial",
                                    "condition": condition,
                                    "run": run,
                                    "original_pair_id": int(original_id),
                                    "template_pair_id": int(template_lookup[condition].get(original_id, np.nan)),
                                    "condition_item_id": _condition_item_id(condition, original_id),
                                    "pattern_norm": float(np.linalg.norm(vals)) if vals.size else np.nan,
                                    "pattern_mean": float(np.nanmean(vals)) if vals.size else np.nan,
                                    "pattern_sd": float(np.nanstd(vals, ddof=0)) if vals.size else np.nan,
                                    "pattern_variance": float(np.nanvar(vals, ddof=0)) if vals.size else np.nan,
                                    "within_run_mean_similarity": float(within[run][pos]) if np.isfinite(within[run][pos]) else np.nan,
                                    "valid_voxel_count": int(finite.sum()),
                                }
                            )
                    qc_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": roi_set,
                            "roi": roi,
                            "condition": condition,
                            "ok": True,
                            "n_items": len(common_ids),
                            "n_voxels": int(by_run[3].shape[1]),
                            "message": "",
                        }
                    )
                except Exception as exc:
                    qc_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": roi_set,
                            "roi": roi,
                            "condition": condition,
                            "ok": False,
                            "n_items": len(common_ids),
                            "n_voxels": np.nan,
                            "message": repr(exc),
                        }
                    )
    quality = pd.DataFrame(rows)
    for metric in QUALITY_METRICS:
        if metric in quality:
            quality[f"{metric}_z"] = quality.groupby(["roi_set", "roi", "run"], dropna=False)[metric].transform(zscore)
    qc = pd.DataFrame(qc_rows)
    return quality, qc


def build_network_quality(quality: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subject", "network", "condition", "run", "original_pair_id", "template_pair_id", "condition_item_id"]
    agg = {metric: "mean" for metric in QUALITY_METRICS if metric in quality.columns}
    agg["roi"] = "nunique"
    net = quality.groupby(group_cols, dropna=False).agg(agg).reset_index()
    net = net.rename(columns={"roi": "n_rois"})
    net["roi_set"] = "network"
    net["roi"] = net["network"]
    for metric in QUALITY_METRICS:
        if metric in net:
            net[f"{metric}_z"] = net.groupby(["network", "run"], dropna=False)[metric].transform(zscore)
    return net


def quality_descriptive_models(quality: pd.DataFrame, network_quality: pd.DataFrame) -> pd.DataFrame:
    rows, conv = [], []
    frames = [
        ("roi", quality, ["roi_set", "roi"]),
        ("network", network_quality, ["roi_set", "roi"]),
    ]
    for scope, frame, group_cols in frames:
        for keys, sub in frame.groupby(group_cols, dropna=False, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            roi_set, roi = keys
            for metric in QUALITY_METRICS:
                if metric not in sub or pd.to_numeric(sub[metric], errors="coerce").nunique(dropna=True) <= 1:
                    continue
                metric_z = f"{metric}_z"
                formula = f"{metric_z} ~ C(condition, Treatment(reference='kj')) + C(run)"
                rows_i, info = fit_cluster_ols(
                    sub,
                    formula=formula,
                    outcome=metric_z,
                    info={
                        "analysis_id": "stage45_stage2_pattern_quality",
                        "model_name": "quality_condition_qc",
                        "scope": scope,
                        "roi_set": roi_set,
                        "roi": roi,
                        "quality_metric": metric,
                    },
                )
                rows.extend(rows_i)
                conv.append(info)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = fdr_by_family(out, ["model_name", "scope", "quality_metric", "term"])
    return out


def _prepare_quality_for_merge(quality: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "run",
        "original_pair_id",
        "condition_item_id",
        *[f"{metric}_z" for metric in QUALITY_METRICS],
    ]
    out = quality[[c for c in cols if c in quality.columns]].copy()
    rename = {
        "original_pair_id": f"{prefix}_original_pair_id",
        "condition_item_id": f"{prefix}_condition_item_id",
        "run": f"{prefix}_run",
    }
    for metric in QUALITY_METRICS:
        z = f"{metric}_z"
        if z in out.columns:
            rename[z] = f"{prefix}_{z}"
    return out.rename(columns=rename)


def _attach_quality(matrix: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    q3 = _prepare_quality_for_merge(quality[quality["run"].eq(3)].copy(), "run3")
    q4 = _prepare_quality_for_merge(quality[quality["run"].eq(4)].copy(), "run4")
    out = matrix.merge(
        q3.drop(columns=["run3_run"], errors="ignore"),
        on=["subject", "roi_set", "roi", "condition", "run3_original_pair_id", "run3_condition_item_id"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        q4.drop(columns=["run4_run"], errors="ignore"),
        on=["subject", "roi_set", "roi", "condition", "run4_original_pair_id", "run4_condition_item_id"],
        how="left",
        validate="many_to_one",
    )
    return out


def stage2_condition_quality_sensitivity(
    matrix_path: Path,
    network_matrix_path: Path,
    quality: pd.DataFrame,
    network_quality: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(matrix_path, sep="\t", low_memory=False)
    network_matrix = pd.read_csv(network_matrix_path, sep="\t", low_memory=False)
    matrix_q = _attach_quality(matrix, quality)
    network_q = _attach_quality(network_matrix, network_quality)
    frames = [
        ("roi", matrix_q),
        ("network", network_q),
    ]
    rows, conv = [], []
    predictor_sets = {
        "stage2b_amplitude_controlled": [
            f"{prefix}_{metric}_z"
            for metric in ["pattern_norm", "pattern_mean", "pattern_sd", "pattern_variance", "valid_voxel_count"]
            for prefix in ["run3", "run4"]
        ],
        "stage2b_quality_controlled": [
            f"{prefix}_{metric}_z"
            for metric in QUALITY_METRICS
            for prefix in ["run3", "run4"]
        ],
    }
    for scope, frame in frames:
        for (roi_set, roi), sub in frame.groupby(["roi_set", "roi"], dropna=False, sort=False):
            for model_name, candidate_predictors in predictor_sets.items():
                predictors = []
                for col in candidate_predictors:
                    if col in sub.columns and pd.to_numeric(sub[col], errors="coerce").nunique(dropna=True) > 1:
                        predictors.append(col)
                quality_expr = (" + " + " + ".join(predictors)) if predictors else ""
                formula = (
                    "cross_run_similarity ~ same_pair * C(condition, Treatment(reference='kj'))"
                    + quality_expr
                )
                rows_i, info = fit_cluster_ols(
                    sub,
                    formula=formula,
                    outcome="cross_run_similarity",
                    info={
                        "analysis_id": "stage45_stage2_condition_quality_sensitivity",
                        "model_name": model_name,
                        "scope": scope,
                        "roi_set": roi_set,
                        "roi": roi,
                        "quality_predictors": ",".join(predictors),
                    },
                    min_rows=500,
                )
                rows.extend(rows_i)
                conv.append(info)
    models = pd.DataFrame(rows)
    if not models.empty:
        models = fdr_by_family(models, ["model_name", "scope", "roi_set", "term"])
    convergence = pd.DataFrame(conv)
    write_table(matrix_q, out_dir / "stage45_stage2_cross_run_matrix_with_quality.tsv")
    write_table(network_q, out_dir / "stage45_stage2_cross_run_network_matrix_with_quality.tsv")
    write_table(models, out_dir / "stage45_stage2_condition_quality_models.tsv")
    write_table(convergence, out_dir / "stage45_stage2_condition_quality_convergence.tsv")
    return models, convergence, network_q


def audit2_stage2_pattern_quality(args, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_path = out_dir / "stage45_stage2_pattern_quality.tsv"
    network_quality_path = out_dir / "stage45_stage2_pattern_quality_network.tsv"
    qc_path = out_dir / "stage45_stage2_pattern_quality_qc.tsv"
    if args.reuse_quality and quality_path.exists() and network_quality_path.exists():
        quality = pd.read_csv(quality_path, sep="\t", low_memory=False)
        network_quality = pd.read_csv(network_quality_path, sep="\t", low_memory=False)
        qc = pd.read_csv(qc_path, sep="\t", low_memory=False) if qc_path.exists() else pd.DataFrame()
    else:
        quality, qc = compute_stage2_pattern_quality(
            pattern_root=args.pattern_root,
            roi_manifest=args.roi_manifest,
            stimuli_template=args.stimuli_template,
            roi_sets=args.roi_sets,
            subjects=args.subjects,
        )
        network_quality = build_network_quality(quality)
        write_table(quality, quality_path)
        write_table(network_quality, network_quality_path)
        write_table(qc, qc_path)
    quality_models = quality_descriptive_models(quality, network_quality)
    write_table(quality_models, out_dir / "stage45_stage2_pattern_quality_descriptive_models.tsv")
    sensitivity, convergence, network_matrix_q = stage2_condition_quality_sensitivity(
        args.stage2_matrix,
        args.stage2_network_matrix,
        quality,
        network_quality,
        out_dir,
    )
    write_review_audit2(out_dir, quality, network_quality, qc, quality_models, sensitivity, convergence)
    return quality_models, sensitivity


def write_review_audit2(
    out_dir: Path,
    quality: pd.DataFrame,
    network_quality: pd.DataFrame,
    qc: pd.DataFrame,
    quality_models: pd.DataFrame,
    sensitivity: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    condition_term = "C(condition, Treatment(reference='kj'))[T.yy]"
    desc = quality_models[
        quality_models["term"].eq(condition_term)
        & quality_models["scope"].eq("network")
    ][["roi", "quality_metric", "estimate", "p", "q", "n_rows"]].sort_values(["roi", "q"], na_position="last")
    sens_terms = sensitivity[
        sensitivity["term"].isin([condition_term, f"same_pair:{condition_term}", f"{condition_term}:same_pair"])
    ].copy()
    sens_show = sens_terms[
        ["model_name", "scope", "roi_set", "roi", "term", "estimate", "se", "p", "q", "n_rows", "status"]
    ].sort_values(["scope", "term", "q"], na_position="last")
    qc_summary = pd.DataFrame(
        [
            {
                "n_quality_rows": len(quality),
                "n_network_quality_rows": len(network_quality),
                "n_qc_rows": len(qc),
                "n_failed_qc": int((~qc["ok"].astype(bool)).sum()) if "ok" in qc else np.nan,
                "subjects": quality["subject"].nunique(),
                "rois": quality["roi"].nunique(),
            }
        ]
    )
    text = f"""# Stage 4.5 Audit 2: Stage 2b Pattern-Quality QC

Purpose: test whether the Stage 2b cross-run condition effect is plausibly explained by global pattern quality.

Computed quality metrics:

- pattern norm
- pattern mean
- voxelwise SD / variance
- within-run mean similarity to other items in the same condition/run
- valid voxel count

Strict item-level split-half reliability cannot be computed from the current learning beta tables because each item has one beta per run.
The within-run mean similarity metric is therefore a quality/global-geometry proxy rather than a true reliability estimate.

## QC Summary

{md_table(qc_summary)}

## Network-Level YY-KJ Quality Differences

{md_table(desc)}

## Quality-Controlled Stage 2b Condition Terms

{md_table(sens_show)}

## Model Convergence

{md_table(convergence[['model_name', 'scope', 'roi_set', 'roi', 'status', 'n_rows', 'n_subjects', 'message']])}
"""
    (out_dir / "stage45_stage2_condition_quality_review.md").write_text(text, encoding="utf-8")


def audit3_stage3_edge_direction(args, out_dir: Path) -> pd.DataFrame:
    paths: list[Path] = []
    for root in args.stage3_audit_roots:
        root = Path(root)
        if root.exists():
            paths.extend(sorted(root.rglob("*_rdms.npz")))
    rows = []
    for path in paths:
        rel = path
        try:
            subject = path.parent.parent.name
            roi = path.parent.name
            stem = path.name.replace("_rdms.npz", "")
            time, condition_group = stem.split("_", 1) if "_" in stem else (stem, "unknown")
            z = np.load(path, allow_pickle=False)
            if "neural_rdm" not in z.files or "M8_reverse_pair" not in z.files:
                rows.append({"path": str(path), "status": "missing_neural_or_edge"})
                continue
            neural = np.asarray(z["neural_rdm"], dtype=float)
            edge = np.asarray(z["M8_reverse_pair"], dtype=float)
            valid = np.isfinite(neural) & np.isfinite(edge)
            edge_values = sorted(pd.unique(pd.Series(edge[valid]).dropna()).tolist())
            same = valid & np.isclose(edge, 1.0)
            other = valid & np.isclose(edge, 0.0)
            rho, p = stats.spearmanr(neural[valid], edge[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
            rows.append(
                {
                    "path": str(rel),
                    "subject": subject,
                    "roi": roi,
                    "roi_set": "meta_metaphor" if roi in SEMANTIC_ROIS else "meta_spatial",
                    "network": "semantic" if roi in SEMANTIC_ROIS else "hpc_spatial",
                    "time": time,
                    "condition_group": condition_group,
                    "status": "ok",
                    "neural_rdm_type": "correlation_distance_1_minus_r",
                    "neural_min": float(np.nanmin(neural[valid])) if valid.any() else np.nan,
                    "neural_max": float(np.nanmax(neural[valid])) if valid.any() else np.nan,
                    "neural_mean": float(np.nanmean(neural[valid])) if valid.any() else np.nan,
                    "edge_model_name": "M8_reverse_pair",
                    "edge_values": ",".join(map(str, edge_values)),
                    "edge_one_mean_neural_distance": float(np.nanmean(neural[same])) if same.any() else np.nan,
                    "edge_zero_mean_neural_distance": float(np.nanmean(neural[other])) if other.any() else np.nan,
                    "same_minus_other_neural_distance": (
                        float(np.nanmean(neural[same]) - np.nanmean(neural[other])) if same.any() and other.any() else np.nan
                    ),
                    "spearman_neural_edge": float(rho) if pd.notna(rho) else np.nan,
                    "spearman_p": float(p) if pd.notna(p) else np.nan,
                    "direction_interpretation": "positive means same-pair items have larger neural distance / stronger differentiation",
                    "unique_r2_directional": False,
                }
            )
        except Exception as exc:
            rows.append({"path": str(path), "status": "failed", "message": repr(exc)})
    audit = pd.DataFrame(rows)
    write_table(audit, out_dir / "stage45_stage3_edge_direction_audit.tsv")
    write_review_audit3(out_dir, audit)
    return audit


def write_review_audit3(out_dir: Path, audit: pd.DataFrame) -> None:
    ok = audit[audit["status"].eq("ok")].copy()
    summary = (
        ok.groupby(["network", "condition_group"], dropna=False)
        .agg(
            n_cells=("same_minus_other_neural_distance", "size"),
            mean_same_minus_other_distance=("same_minus_other_neural_distance", "mean"),
            mean_spearman_neural_edge=("spearman_neural_edge", "mean"),
            min_neural_rdm=("neural_min", "min"),
            max_neural_rdm=("neural_max", "max"),
        )
        .reset_index()
        if not ok.empty
        else pd.DataFrame()
    )
    text = f"""# Stage 4.5 Audit 3: Stage 3 Edge-Model Direction Audit

Coding audit:

- Neural RDM uses correlation distance (`1 - r`), so larger values mean more dissimilar neural patterns.
- `M8_reverse_pair` is coded as same pair = 1, different pair = 0.
- A positive neural-edge association means same-pair words are farther apart in neural RDM, i.e. stronger differentiation.
- `unique_edge_r2 = full R2 - reduced R2` is an explained-variance increment and has no directional sign by itself.
- Direction must therefore be interpreted from model coding plus neural-edge coefficient/correlation, not from unique R2 alone.

## Audit Summary

{md_table(summary)}

Conclusion: if no failed coding rows appear, Stage 3 can remain a boundary result without additional reruns.

Failed/non-ok rows:

{md_table(audit[~audit['status'].eq('ok')][['path', 'status', 'message']] if 'message' in audit else pd.DataFrame())}
"""
    (out_dir / "stage45_stage3_edge_direction_review.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4.5 accuracy audits.")
    base_dir = _default_base_dir()
    qc_root = base_dir / "paper_outputs" / "qc"
    out_dir = qc_root / "stagewise_mechanism"
    parser.add_argument("--out-dir", type=Path, default=out_dir)
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--reuse-quality", action="store_true")
    parser.add_argument(
        "--stage1-network-item",
        type=Path,
        default=out_dir / "stage1_post_to_retrieval_network_item.tsv",
    )
    parser.add_argument(
        "--stage2-matrix",
        type=Path,
        default=out_dir / "stage2b_cross_run_matrix.tsv",
    )
    parser.add_argument(
        "--stage2-network-matrix",
        type=Path,
        default=out_dir / "stage2b_cross_run_network_matrix.tsv",
    )
    parser.add_argument(
        "--stage3-audit-roots",
        nargs="+",
        type=Path,
        default=[
            qc_root / "model_rdm_results_meta_metaphor" / "model_rdm_audit",
            qc_root / "model_rdm_results_meta_spatial" / "model_rdm_audit",
            qc_root / "stagewise_mechanism" / "stage3_model_rdm_by_condition_meta_metaphor" / "model_rdm_audit",
            qc_root / "stagewise_mechanism" / "stage3_model_rdm_by_condition_meta_spatial" / "model_rdm_audit",
        ],
    )
    parser.add_argument("--skip-audit2", action="store_true")
    args = parser.parse_args()

    out = ensure_dir(args.out_dir)
    comp, mem_models = audit1_memory_difference(args.stage1_network_item, out)
    if args.skip_audit2:
        quality_models = pd.DataFrame()
        quality_sensitivity = pd.DataFrame()
    else:
        quality_models, quality_sensitivity = audit2_stage2_pattern_quality(args, out)
    edge_audit = audit3_stage3_edge_direction(args, out)
    manifest = {
        "analysis_id": "stage45_accuracy_audits",
        "n_memory_component_rows": int(len(comp)),
        "n_memory_model_rows": int(len(mem_models)),
        "n_quality_model_rows": int(len(quality_models)),
        "n_quality_sensitivity_rows": int(len(quality_sensitivity)),
        "n_stage3_edge_audit_rows": int(len(edge_audit)),
        "out_dir": str(out),
    }
    (out / "stage45_accuracy_audits_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
