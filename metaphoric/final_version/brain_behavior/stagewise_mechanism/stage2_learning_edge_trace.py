#!/usr/bin/env python3
"""Stage 2a/2b: learning-phase item trace and cross-run matrix geometry."""

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
import statsmodels.api as sm
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

from common.roi_library import filter_roi_manifest, load_roi_manifest  # noqa: E402
from brain_behavior.build_learning_post_memory_item_table import (  # noqa: E402
    CONDITIONS,
    CONDITION_LABELS,
    VALID_ORIGINAL_IDS,
    _condition_item_id,
    _corr_matrix,
    _load_learning_condition,
    _load_mask,
    _load_template_map,
    _masked_samples,
)


META_METAPHOR_ROIS = {
    "meta_L_IFG",
    "meta_R_IFG",
    "meta_L_temporal_pole",
    "meta_R_temporal_pole",
    "meta_L_AG",
    "meta_R_AG",
    "meta_L_pMTG_pSTS",
    "meta_R_pMTG_pSTS",
}
META_SPATIAL_ROIS = {
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
MATERIAL_COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
TRACE_METRICS = ["row_specificity", "column_specificity", "bidirectional_specificity"]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _bh_fdr(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric)
    if valid.any():
        _, qvals, _, _ = multipletests(numeric.loc[valid], method="fdr_bh")
        out.loc[numeric.loc[valid].index] = qvals
    return out


def _z_by_group(frame: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    if not group_cols:
        vals = pd.to_numeric(frame[col], errors="coerce")
        sd = vals.std(ddof=1)
        if np.isfinite(sd) and not np.isclose(sd, 0.0):
            out.loc[frame.index] = (vals - vals.mean()) / sd
        return out
    for _, idx in frame.groupby(group_cols, sort=False, dropna=False, observed=True).groups.items():
        vals = pd.to_numeric(frame.loc[idx, col], errors="coerce")
        sd = vals.std(ddof=1)
        if np.isfinite(sd) and not np.isclose(sd, 0.0):
            out.loc[idx] = (vals - vals.mean()) / sd
    return out


def _available_covariates(frame: pd.DataFrame, min_coverage: float = 0.8) -> list[str]:
    covs: list[str] = []
    for col in MATERIAL_COVARIATES:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().mean() >= min_coverage and vals.nunique(dropna=True) > 1:
            covs.append(col)
    return covs


def _replace_inf(frame: pd.DataFrame) -> pd.DataFrame:
    with pd.option_context("future.no_silent_downcasting", True):
        return frame.replace([np.inf, -np.inf], np.nan)


def _load_item_context(item_table: Path) -> pd.DataFrame:
    keep = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "original_pair_id",
        "template_pair_id",
        "condition_item_id",
        "pre_pair_similarity",
        "pre_pair_similarity_z",
        "post_edge_specificity",
        "post_edge_specificity_z",
        *MATERIAL_COVARIATES,
    ]
    frame = pd.read_csv(item_table, sep="\t", usecols=lambda c: c in keep, low_memory=False)
    frame["condition"] = frame["condition"].astype(str).str.lower()
    return frame


def _compute_subject_roi_condition(
    subject_dir: Path,
    *,
    roi_set: str,
    roi: str,
    mask_path: Path,
    condition: str,
    template_lookup: dict[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    subject = subject_dir.name
    meta, data, image_path = _load_learning_condition(subject_dir, condition)
    samples = _masked_samples(data, _load_mask(mask_path), image_path, mask_path)
    run3 = meta[meta["run"].eq(3)].drop_duplicates("original_pair_id", keep=False).copy()
    run4 = meta[meta["run"].eq(4)].drop_duplicates("original_pair_id", keep=False).copy()
    common_ids = sorted(
        set(int(v) for v in run3["original_pair_id"].dropna().tolist())
        & set(int(v) for v in run4["original_pair_id"].dropna().tolist())
        & set(VALID_ORIGINAL_IDS[condition])
    )
    run3_index = {int(row.original_pair_id): int(idx) for idx, row in run3.iterrows()}
    run4_index = {int(row.original_pair_id): int(idx) for idx, row in run4.iterrows()}
    mat3 = samples[[run3_index[idx] for idx in common_ids], :]
    mat4 = samples[[run4_index[idx] for idx in common_ids], :]
    sim = _corr_matrix(mat3, mat4)
    n = len(common_ids)

    item_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    diag = np.diag(sim)
    for pos, original_id in enumerate(common_ids):
        row_other = np.delete(sim[pos, :], pos)
        col_other = np.delete(sim[:, pos], pos)
        row_mean = float(np.nanmean(row_other)) if np.isfinite(row_other).any() else np.nan
        col_mean = float(np.nanmean(col_other)) if np.isfinite(col_other).any() else np.nan
        self_sim = float(diag[pos]) if np.isfinite(diag[pos]) else np.nan
        row_spec = self_sim - row_mean if np.isfinite(self_sim) and np.isfinite(row_mean) else np.nan
        col_spec = self_sim - col_mean if np.isfinite(self_sim) and np.isfinite(col_mean) else np.nan
        bidir_base = np.nanmean([row_mean, col_mean])
        bidir = self_sim - bidir_base if np.isfinite(self_sim) and np.isfinite(bidir_base) else np.nan
        item_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "original_pair_id": int(original_id),
                "template_pair_id": int(template_lookup.get(original_id, np.nan)),
                "condition_item_id": _condition_item_id(condition, original_id),
                "learning_self_similarity": self_sim,
                "row_other_similarity": row_mean,
                "column_other_similarity": col_mean,
                "row_specificity": row_spec,
                "column_specificity": col_spec,
                "bidirectional_specificity": bidir,
                "n_same_condition_items": n,
            }
        )

    for i, row_id in enumerate(common_ids):
        for j, col_id in enumerate(common_ids):
            matrix_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "run3_original_pair_id": int(row_id),
                    "run4_original_pair_id": int(col_id),
                    "run3_condition_item_id": _condition_item_id(condition, row_id),
                    "run4_condition_item_id": _condition_item_id(condition, col_id),
                    "same_pair": int(row_id == col_id),
                    "cross_run_similarity": float(sim[i, j]) if np.isfinite(sim[i, j]) else np.nan,
                    "n_same_condition_items": n,
                }
            )
    qc = {
        "subject": subject,
        "roi_set": roi_set,
        "roi": roi,
        "condition": condition,
        "ok": True,
        "n_items": n,
        "n_matrix_rows": int(n * n),
        "finite_similarity_cells": int(np.isfinite(sim).sum()),
        "total_similarity_cells": int(sim.size),
        "fail_reason": "",
    }
    return item_rows, matrix_rows, qc


def _compute_learning_geometry(
    *,
    pattern_root: Path,
    roi_manifest: Path,
    roi_sets: list[str],
    stimuli_template: Path,
    subjects: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template = _load_template_map(stimuli_template)
    template_lookup = {
        condition: template[template["condition"].eq(condition)].set_index("original_pair_id")["template_pair_id"].to_dict()
        for condition in CONDITIONS
    }
    manifest = load_roi_manifest(roi_manifest)
    rois = pd.concat(
        [filter_roi_manifest(manifest, roi_set=roi_set, include_flag="include_in_rsa") for roi_set in roi_sets],
        ignore_index=True,
    )
    subject_dirs = sorted([p for p in pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if subjects:
        wanted = set(subjects)
        subject_dirs = [p for p in subject_dirs if p.name in wanted]

    item_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        for roi_row in rois.itertuples(index=False):
            for condition in CONDITIONS:
                try:
                    items, matrix, qc = _compute_subject_roi_condition(
                        subject_dir,
                        roi_set=str(roi_row.roi_set),
                        roi=str(roi_row.roi_name),
                        mask_path=Path(str(roi_row.mask_path)),
                        condition=condition,
                        template_lookup=template_lookup[condition],
                    )
                    item_rows.extend(items)
                    matrix_rows.extend(matrix)
                    qc_rows.append(qc)
                except Exception as exc:
                    qc_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": str(roi_row.roi_set),
                            "roi": str(roi_row.roi_name),
                            "condition": condition,
                            "ok": False,
                            "n_items": 0,
                            "n_matrix_rows": 0,
                            "finite_similarity_cells": 0,
                            "total_similarity_cells": 0,
                            "fail_reason": repr(exc),
                        }
                    )
    return pd.DataFrame(item_rows), pd.DataFrame(matrix_rows), pd.DataFrame(qc_rows)


def _merge_context(item: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    merged = item.merge(
        context,
        on=["subject", "roi_set", "roi", "condition", "original_pair_id", "template_pair_id", "condition_item_id"],
        how="left",
    )
    for metric in TRACE_METRICS:
        merged[f"{metric}_z"] = _z_by_group(merged, metric, ["roi_set", "roi"])
    return merged


def _make_network_item(item: pd.DataFrame) -> pd.DataFrame:
    key = ["subject", "condition", "original_pair_id", "template_pair_id", "condition_item_id"]
    cov_cols = [c for c in MATERIAL_COVARIATES if c in item.columns]
    frames = []
    for name, roi_set in [("semantic", META_METAPHOR_ROIS), ("hpc_spatial", META_SPATIAL_ROIS)]:
        sub = item[item["roi"].isin(roi_set)]
        agg = {
            f"{name}_{metric}": (metric, "mean")
            for metric in [*TRACE_METRICS, "post_edge_specificity", "pre_pair_similarity"]
            if metric in sub.columns
        }
        agg[f"n_{name}_rois"] = ("roi", "nunique")
        frames.append(sub.groupby(key, observed=True, as_index=False).agg(**agg))
    out = frames[0].merge(frames[1], on=key, how="outer")
    if cov_cols:
        covs = item.groupby(key, observed=True, as_index=False)[cov_cols].mean()
        out = out.merge(covs, on=key, how="left")
    for col in out.columns:
        if any(col.endswith(f"_{metric}") for metric in TRACE_METRICS) or col.endswith("_post_edge_specificity") or col.endswith("_pre_pair_similarity"):
            out[f"{col}_z"] = _z_by_group(out, col, [])
    return out


def _make_network_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    key = [
        "subject",
        "condition",
        "run3_original_pair_id",
        "run4_original_pair_id",
        "run3_condition_item_id",
        "run4_condition_item_id",
        "same_pair",
    ]
    rows = []
    for network, roi_set in [("semantic", META_METAPHOR_ROIS), ("hpc_spatial", META_SPATIAL_ROIS)]:
        sub = matrix[matrix["roi"].isin(roi_set)]
        agg = sub.groupby(key, observed=True, as_index=False).agg(
            cross_run_similarity=("cross_run_similarity", "mean"),
            n_rois=("roi", "nunique"),
        )
        agg["roi_set"] = "network"
        agg["roi"] = network
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def _fit_mixed_or_fallback(frame: pd.DataFrame, formula: str, *, outcome: str, model: str, roi_set: str, roi: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    cols = sorted(set(["subject", "condition_item_id", outcome, *[c for c in frame.columns if c in formula]]))
    data = _replace_inf(frame).dropna(subset=[c for c in cols if c in frame.columns]).copy()
    if len(data) < 30 or data["subject"].nunique() < 5:
        return [], {"model": model, "roi_set": roi_set, "roi": roi, "outcome": outcome, "status": "skipped_too_few_rows", "n_rows": len(data), "n_subjects": data.get("subject", pd.Series(dtype=object)).nunique(), "n_items": data.get("condition_item_id", pd.Series(dtype=object)).nunique(), "fallback_used": "none", "message": ""}
    data["subject"] = data["subject"].astype(str)
    data["condition_item_id"] = data["condition_item_id"].astype(str)
    data["condition"] = data["condition"].astype(str)
    status, fallback, message = "ok", "none", ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm(
                formula,
                data,
                groups=data["subject"],
                vc_formula={"item": "0 + C(condition_item_id)"},
            ).fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            if not bool(getattr(fit, "converged", False)):
                status = "mixed_not_converged"
        except Exception as exc:
            status, fallback, message = "fallback_ols_cluster_subject", "ols_cluster_subject", repr(exc)
            fit = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data["subject"]})
        if caught and not message:
            message = " | ".join(sorted({str(w.message)[:200] for w in caught}))
    rows = []
    for term, estimate in fit.params.items():
        if term.startswith("item Var") or term.endswith(" Var"):
            continue
        se = fit.bse.get(term, np.nan)
        rows.append(
            {
                "model": model,
                "roi_set": roi_set,
                "roi": roi,
                "outcome": outcome,
                "term": term,
                "estimate": float(estimate),
                "se": float(se) if pd.notna(se) else np.nan,
                "stat": float(estimate / se) if pd.notna(se) and not np.isclose(se, 0.0) else np.nan,
                "p": float(fit.pvalues.get(term, np.nan)),
                "n_rows": int(len(data)),
                "n_subjects": int(data["subject"].nunique()),
                "n_items": int(data["condition_item_id"].nunique()),
                "formula": formula,
                "status": status,
                "fallback_used": fallback,
                "message": message,
            }
        )
    info = {"model": model, "roi_set": roi_set, "roi": roi, "outcome": outcome, "status": status, "n_rows": int(len(data)), "n_subjects": int(data["subject"].nunique()), "n_items": int(data["condition_item_id"].nunique()), "fallback_used": fallback, "message": message}
    return rows, info


def _fit_ols_cluster(frame: pd.DataFrame, formula: str, *, outcome: str, model: str, roi_set: str, roi: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    cols = sorted(set(["subject", outcome, *[c for c in frame.columns if c in formula]]))
    data = _replace_inf(frame).dropna(subset=[c for c in cols if c in frame.columns]).copy()
    if len(data) < 100 or data["subject"].nunique() < 5:
        return [], {"model": model, "roi_set": roi_set, "roi": roi, "outcome": outcome, "status": "skipped_too_few_rows", "n_rows": len(data), "n_subjects": data.get("subject", pd.Series(dtype=object)).nunique(), "fallback_used": "none", "message": ""}
    data["subject"] = data["subject"].astype(str)
    data["condition"] = data["condition"].astype(str)
    try:
        fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["subject"]})
        status, message = "ok", ""
    except Exception as exc:
        return [], {"model": model, "roi_set": roi_set, "roi": roi, "outcome": outcome, "status": "failed", "n_rows": len(data), "n_subjects": data["subject"].nunique(), "fallback_used": "none", "message": repr(exc)}
    rows = []
    for term, estimate in fit.params.items():
        se = fit.bse.get(term, np.nan)
        rows.append(
            {
                "model": model,
                "roi_set": roi_set,
                "roi": roi,
                "outcome": outcome,
                "term": term,
                "estimate": float(estimate),
                "se": float(se) if pd.notna(se) else np.nan,
                "stat": float(estimate / se) if pd.notna(se) and not np.isclose(se, 0.0) else np.nan,
                "p": float(fit.pvalues.get(term, np.nan)),
                "n_rows": int(len(data)),
                "n_subjects": int(data["subject"].nunique()),
                "formula": formula,
                "status": status,
                "fallback_used": "ols_cluster_subject",
                "message": message,
            }
        )
    info = {"model": model, "roi_set": roi_set, "roi": roi, "outcome": outcome, "status": status, "n_rows": int(len(data)), "n_subjects": int(data["subject"].nunique()), "fallback_used": "ols_cluster_subject", "message": message}
    return rows, info


def _fit_item_fixed_fwl(frame: pd.DataFrame, *, outcome: str, model: str, roi_set: str, roi: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Two-way item fixed-effect sensitivity via FWL residualization.

    The condition main effect is intentionally omitted because row/column item
    fixed effects absorb it. Residualizing first also avoids estimating dozens
    of nuisance FE coefficients in the final clustered covariance step.
    """
    needed = [
        "subject",
        "condition",
        "same_pair",
        "run3_condition_item_id",
        "run4_condition_item_id",
        outcome,
    ]
    data = _replace_inf(frame).dropna(subset=needed).copy()
    formula = (
        "FWL: cross_run_similarity ~ same_pair + "
        "same_pair:C(condition, Treatment(reference='kj')) + "
        "C(run3_condition_item_id) + C(run4_condition_item_id); "
        "condition main effect omitted because item fixed effects absorb it"
    )
    if len(data) < 100 or data["subject"].nunique() < 5:
        return [], {
            "model": model,
            "roi_set": roi_set,
            "roi": roi,
            "outcome": outcome,
            "status": "skipped_too_few_rows",
            "n_rows": len(data),
            "n_subjects": data.get("subject", pd.Series(dtype=object)).nunique(),
            "fallback_used": "none",
            "message": "",
        }

    data["subject"] = data["subject"].astype(str)
    data["condition"] = data["condition"].astype(str)
    data["same_pair"] = pd.to_numeric(data["same_pair"], errors="coerce").astype(float)
    data["same_pair_yy"] = data["same_pair"] * data["condition"].eq("yy").astype(float)
    y = pd.to_numeric(data[outcome], errors="coerce").astype(float)
    ok = y.notna() & data["same_pair"].notna() & data["same_pair_yy"].notna()
    data = data.loc[ok].copy()
    y = y.loc[ok]

    row_fe = pd.get_dummies(data["run3_condition_item_id"].astype(str), prefix="run3", drop_first=True, dtype=float)
    col_fe = pd.get_dummies(data["run4_condition_item_id"].astype(str), prefix="run4", drop_first=True, dtype=float)
    fe = pd.concat([pd.Series(1.0, index=data.index, name="Intercept"), row_fe, col_fe], axis=1)
    targets = pd.concat(
        [
            y.rename("__y"),
            data["same_pair"].rename("same_pair"),
            data["same_pair_yy"].rename("same_pair:C(condition, Treatment(reference='kj'))[T.yy]"),
        ],
        axis=1,
    )
    try:
        fe_mat = fe.to_numpy(dtype=float)
        target_mat = targets.to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(fe_mat, target_mat, rcond=None)
        resid = target_mat - fe_mat @ beta
        y_resid = resid[:, 0]
        x_resid = pd.DataFrame(resid[:, 1:], columns=targets.columns[1:], index=data.index)
        if np.linalg.matrix_rank(x_resid.to_numpy(dtype=float)) < x_resid.shape[1]:
            raise np.linalg.LinAlgError("target regressors are rank deficient after item-FE residualization")
        fit = sm.OLS(y_resid, x_resid).fit(cov_type="cluster", cov_kwds={"groups": data["subject"]})
        status, message = "ok", ""
    except Exception as exc:
        return [], {
            "model": model,
            "roi_set": roi_set,
            "roi": roi,
            "outcome": outcome,
            "status": "failed",
            "n_rows": int(len(data)),
            "n_subjects": int(data["subject"].nunique()),
            "fallback_used": "none",
            "message": repr(exc),
            "formula": formula,
        }

    rows = []
    for term, estimate in fit.params.items():
        se = fit.bse.get(term, np.nan)
        rows.append(
            {
                "model": model,
                "roi_set": roi_set,
                "roi": roi,
                "outcome": outcome,
                "term": term,
                "estimate": float(estimate),
                "se": float(se) if pd.notna(se) else np.nan,
                "stat": float(estimate / se) if pd.notna(se) and not np.isclose(se, 0.0) else np.nan,
                "p": float(fit.pvalues.get(term, np.nan)),
                "n_rows": int(len(data)),
                "n_subjects": int(data["subject"].nunique()),
                "formula": formula,
                "status": status,
                "fallback_used": "fwl_item_fe_ols_cluster_subject",
                "message": message,
            }
        )
    info = {
        "model": model,
        "roi_set": roi_set,
        "roi": roi,
        "outcome": outcome,
        "status": status,
        "n_rows": int(len(data)),
        "n_subjects": int(data["subject"].nunique()),
        "fallback_used": "fwl_item_fe_ols_cluster_subject",
        "message": message,
        "formula": formula,
    }
    return rows, info


def _one_sample_models(item: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in TRACE_METRICS:
        for (roi_set, roi), sub in item.groupby(["roi_set", "roi"], observed=True, sort=False):
            subject_mean = sub.groupby("subject", as_index=False)[metric].mean().dropna(subset=[metric])
            vals = subject_mean[metric].to_numpy(dtype=float)
            if len(vals) >= 3:
                test = stats.ttest_1samp(vals, 0.0)
                estimate = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                p = float(test.pvalue)
                stat = float(test.statistic)
                status = "ok"
            else:
                estimate = se = p = stat = np.nan
                status = "skipped_too_few_subjects"
            rows.append(
                {
                    "model": "stage2a_one_sample_trace_gt_zero",
                    "metric": metric,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": "pooled",
                    "term": "subject_mean_intercept",
                    "estimate": estimate,
                    "se": se,
                    "stat": stat,
                    "p": p,
                    "n_subjects": int(len(vals)),
                    "n_items": int(sub["condition_item_id"].nunique()),
                    "status": status,
                }
            )
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    ok = out["status"].eq("ok") & out["p"].notna()
    for _, idx in out[ok].groupby(["metric", "roi_set"], observed=True, dropna=False).groups.items():
        out.loc[idx, "q"] = _bh_fdr(out.loc[idx, "p"])
    return out


def _stage2a_models(item: pd.DataFrame, covs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    covar_part = " + " + " + ".join(covs) if covs else ""
    rows, conv = [], []
    for metric in TRACE_METRICS:
        item[f"{metric}_z"] = _z_by_group(item, metric, ["roi_set", "roi"])
        specs = [
            (
                f"stage2a_{metric}_adjusted_trace",
                metric,
                f"{metric} ~ pre_pair_similarity_z" + covar_part,
            ),
            (
                f"stage2a_{metric}_to_post_edge_pooled",
                "post_edge_specificity",
                f"post_edge_specificity ~ {metric}_z + pre_pair_similarity_z" + covar_part,
            ),
        ]
        for (roi_set, roi), sub in item.groupby(["roi_set", "roi"], observed=True, sort=False):
            for model, outcome, formula in specs:
                model_rows, info = _fit_mixed_or_fallback(sub, formula, outcome=outcome, model=model, roi_set=str(roi_set), roi=str(roi))
                rows.extend(model_rows)
                conv.append(info)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        ok = out["p"].notna() & out["status"].isin(["ok", "mixed_not_converged", "fallback_ols_cluster_subject"])
        for _, idx in out[ok].groupby(["model", "roi_set", "term"], observed=True, dropna=False).groups.items():
            out.loc[idx, "q"] = _bh_fdr(out.loc[idx, "p"])
    return out, pd.DataFrame(conv)


def _stage2b_models(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, conv = [], []
    for (roi_set, roi), sub in matrix.groupby(["roi_set", "roi"], observed=True, sort=False):
        primary_rows, primary_info = _fit_ols_cluster(
            sub,
            "cross_run_similarity ~ same_pair * C(condition, Treatment(reference='kj'))",
            outcome="cross_run_similarity",
            model="stage2b_matrix_primary",
            roi_set=str(roi_set),
            roi=str(roi),
        )
        rows.extend(primary_rows)
        conv.append(primary_info)
        fixed_rows, fixed_info = _fit_item_fixed_fwl(
            sub,
            outcome="cross_run_similarity",
            model="stage2b_matrix_item_fixed",
            roi_set=str(roi_set),
            roi=str(roi),
        )
        rows.extend(fixed_rows)
        conv.append(fixed_info)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        ok = out["p"].notna() & out["status"].eq("ok")
        for _, idx in out[ok].groupby(["model", "roi_set", "term"], observed=True, dropna=False).groups.items():
            out.loc[idx, "q"] = _bh_fdr(out.loc[idx, "p"])
    return out, pd.DataFrame(conv)


def _qc_table(item: pd.DataFrame, matrix: pd.DataFrame, network_item: pd.DataFrame, network_matrix: pd.DataFrame, geom_qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in [("stage2a_roi_item", item), ("stage2b_roi_matrix", matrix), ("stage2a_network_item", network_item), ("stage2b_network_matrix", network_matrix)]:
        rows.append(
            {
                "table": name,
                "n_rows": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()) if "subject" in frame else 0,
                "n_rois": int(frame["roi"].nunique()) if "roi" in frame else 0,
                "yy_items": int(frame.loc[frame["condition"].astype(str).eq("yy"), "condition"].shape[0]) if "condition" in frame else 0,
                "kj_items": int(frame.loc[frame["condition"].astype(str).eq("kj"), "condition"].shape[0]) if "condition" in frame else 0,
            }
        )
    rows.append(
        {
            "table": "geometry_qc",
            "n_rows": int(len(geom_qc)),
            "n_subjects": int(geom_qc["subject"].nunique()) if not geom_qc.empty else 0,
            "n_rois": int(geom_qc["roi"].nunique()) if not geom_qc.empty else 0,
            "yy_items": int(geom_qc.loc[geom_qc["condition"].eq("yy"), "n_items"].sum()) if not geom_qc.empty else 0,
            "kj_items": int(geom_qc.loc[geom_qc["condition"].eq("kj"), "n_items"].sum()) if not geom_qc.empty else 0,
        }
    )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return ""
    work = df[[c for c in cols if c in df.columns]].head(n).copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            work[col] = work[col].astype("object").where(work[col].notna(), "").astype(str)
    header = "| " + " | ".join(work.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in work.columns) + " |" for _, row in work.iterrows()]
    return "\n".join([header, sep, *body])


def _write_review(out_dir: Path, qc: pd.DataFrame, one: pd.DataFrame, models2a: pd.DataFrame, models2b: pd.DataFrame) -> None:
    one_focus = one[one["metric"].eq("bidirectional_specificity")].sort_values("q", na_position="last")
    adjusted_focus = models2a[
        models2a["model"].str.contains("_bidirectional_specificity_adjusted_trace", na=False)
        & models2a["term"].astype(str).eq("Intercept")
    ].sort_values("q", na_position="last")
    post_focus = models2a[
        models2a["model"].str.contains("_bidirectional_specificity_to_post_edge_pooled", na=False)
        & models2a["term"].astype(str).eq("bidirectional_specificity_z")
    ].sort_values("q", na_position="last")
    matrix_focus = models2b[
        models2b["term"].astype(str).isin(
            [
                "same_pair",
                "C(condition, Treatment(reference='kj'))[T.yy]",
                "same_pair:C(condition, Treatment(reference='kj'))[T.yy]",
            ]
        )
    ].sort_values(["model", "q"], na_position="last")
    text = f"""# Stage 2 Learning Cross-Run Geometry Review

## Design

Stage 2a tests pooled item-specific trace using row, column, and bidirectional specificity.
Stage 2b keeps the full run3 x run4 matrix and tests condition, same-pair, and same-pair-by-condition effects.

## QC

{_md_table(qc, list(qc.columns), 10)}

## Stage 2a: Pooled Bidirectional Specificity > 0

{_md_table(one_focus, ["roi_set", "roi", "condition", "estimate", "se", "p", "q", "n_subjects", "status"], 30)}

## Stage 2a: Pooled Adjusted Bidirectional Specificity

{_md_table(adjusted_focus, ["roi_set", "roi", "estimate", "se", "p", "q", "status"], 30)}

## Stage 2a: Pooled Bidirectional Specificity -> Post Edge

{_md_table(post_focus, ["roi_set", "roi", "term", "estimate", "se", "p", "q", "status"], 40)}

## Stage 2b: Cross-Run Matrix Models

{_md_table(matrix_focus, ["model", "roi_set", "roi", "term", "estimate", "se", "p", "q", "status"], 60)}

## Interpretation

- Stage 2a pools YY and KJ because it only tests whether learning items have run3-to-run4 trace.
- Stage 2b is the primary test for learning-stage cross-run geometry because it keeps the full matrix.
- If same_pair is weak but condition is present, learning-stage evidence is better described as condition-level geometry rather than item-specific edge formation.
"""
    (out_dir / "stage2_learning_trace_review.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--item-table", type=Path, default=base_dir / "paper_outputs" / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv")
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "stagewise_mechanism")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-geometry", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_geometry and (args.out_dir / "stage2a_learning_item_trace_item.tsv").exists():
        item = pd.read_csv(args.out_dir / "stage2a_learning_item_trace_item.tsv", sep="\t", low_memory=False)
        matrix = pd.read_csv(args.out_dir / "stage2b_cross_run_matrix.tsv", sep="\t", low_memory=False)
        network_item = pd.read_csv(args.out_dir / "stage2a_learning_item_trace_network_item.tsv", sep="\t", low_memory=False)
        network_matrix = pd.read_csv(args.out_dir / "stage2b_cross_run_network_matrix.tsv", sep="\t", low_memory=False)
        geom_qc = pd.read_csv(args.out_dir / "stage2_learning_geometry_qc.tsv", sep="\t", low_memory=False)
    else:
        item_raw, matrix, geom_qc = _compute_learning_geometry(
            pattern_root=args.pattern_root,
            roi_manifest=args.roi_manifest,
            roi_sets=args.roi_sets,
            stimuli_template=args.stimuli_template,
            subjects=args.subjects,
        )
        context = _load_item_context(args.item_table)
        item = _merge_context(item_raw, context)
        network_item = _make_network_item(item)
        network_matrix = _make_network_matrix(matrix)
    covs = _available_covariates(item)
    matrix_all = pd.concat([matrix, network_matrix], ignore_index=True)
    qc = _qc_table(item, matrix, network_item, network_matrix, geom_qc)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": "stage2_learning_cross_run_geometry",
                    "n_item_rows": int(len(item)),
                    "n_matrix_rows": int(len(matrix)),
                    "n_network_item_rows": int(len(network_item)),
                    "n_network_matrix_rows": int(len(network_matrix)),
                    "covariates": covs,
                    "qc": qc.to_dict(orient="records"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    one = _one_sample_models(item)
    models2a, conv2a = _stage2a_models(item, covs)
    models2b, conv2b = _stage2b_models(matrix_all)
    convergence = pd.concat([conv2a, conv2b], ignore_index=True)

    item.to_csv(args.out_dir / "stage2a_learning_item_trace_item.tsv", sep="\t", index=False)
    network_item.to_csv(args.out_dir / "stage2a_learning_item_trace_network_item.tsv", sep="\t", index=False)
    matrix.to_csv(args.out_dir / "stage2b_cross_run_matrix.tsv", sep="\t", index=False)
    network_matrix.to_csv(args.out_dir / "stage2b_cross_run_network_matrix.tsv", sep="\t", index=False)
    qc.to_csv(args.out_dir / "stage2_learning_trace_qc.tsv", sep="\t", index=False)
    geom_qc.to_csv(args.out_dir / "stage2_learning_geometry_qc.tsv", sep="\t", index=False)
    one.to_csv(args.out_dir / "stage2a_learning_item_trace_one_sample.tsv", sep="\t", index=False)
    models2a.to_csv(args.out_dir / "stage2a_learning_item_trace_models.tsv", sep="\t", index=False)
    models2b.to_csv(args.out_dir / "stage2b_cross_run_matrix_models.tsv", sep="\t", index=False)
    convergence.to_csv(args.out_dir / "stage2_learning_trace_convergence.tsv", sep="\t", index=False)
    _write_review(args.out_dir, qc, one, models2a, models2b)

    manifest = {
        "analysis_id": "stage2_learning_cross_run_geometry",
        "input_item_table": str(args.item_table),
        "output_dir": str(args.out_dir),
        "n_item_rows": int(len(item)),
        "n_matrix_rows": int(len(matrix)),
        "n_network_item_rows": int(len(network_item)),
        "n_network_matrix_rows": int(len(network_matrix)),
        "n_stage2a_model_rows": int(len(models2a)),
        "n_stage2b_model_rows": int(len(models2b)),
        "covariates": covs,
    }
    (args.out_dir / "stage2_learning_trace_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
