from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PRIMARY_COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def read_item_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame["condition"] = frame["condition"].astype(str).str.lower()
    frame = frame[frame["condition"].isin(["kj", "yy"])].copy()
    frame["condition"] = pd.Categorical(frame["condition"], categories=["kj", "yy"])
    return frame


def available_covariates(frame: pd.DataFrame, min_coverage: float = 0.8) -> list[str]:
    covs: list[str] = []
    for col in PRIMARY_COVARIATES:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if float(vals.notna().mean()) >= min_coverage and vals.nunique(dropna=True) > 1:
            covs.append(col)
    return covs


def prepare_model_frame(frame: pd.DataFrame, columns: list[str], covariates: list[str] | None = None) -> pd.DataFrame:
    needed = ["subject", "roi_set", "roi", "condition", "condition_item_id", *columns]
    if covariates:
        needed.extend(covariates)
    existing = list(dict.fromkeys([col for col in needed if col in frame.columns]))
    out = frame[existing].copy()
    for col in columns + list(covariates or []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col for col in columns if col in out.columns])
    for col in covariates or []:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    out["condition"] = pd.Categorical(out["condition"].astype(str).str.lower(), categories=["kj", "yy"])
    return out


def _fit_gee(formula: str, data: pd.DataFrame, family) -> object:
    return smf.gee(
        formula=formula,
        groups="subject",
        data=data,
        cov_struct=sm.cov_struct.Exchangeable(),
        family=family,
    ).fit()


def fit_roi_gee(
    frame: pd.DataFrame,
    *,
    formula: str,
    outcome: str,
    model_name: str,
    family_name: str,
    covariates: list[str] | None = None,
    min_rows: int = 80,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    family = sm.families.Binomial() if family_name == "binomial" else sm.families.Gaussian()
    for (roi_set, roi), sub in frame.groupby(["roi_set", "roi"], sort=False):
        model_cols = [outcome]
        for token in ["learning_specificity_z", "post_edge_specificity_z", "post_edge_specificity_resid_z",
                      "pre_pair_similarity_z", "pseudo_edge_drop_z", "shuffled_predictor_z",
                      "retrieval_pair_similarity_z", "retrieval_minus_post_pair_similarity_z"]:
            if token in formula:
                model_cols.append(token)
        model = prepare_model_frame(sub, sorted(set(model_cols)), covariates)
        n_subjects = int(model["subject"].nunique()) if "subject" in model else 0
        n_items = int(model["condition_item_id"].nunique()) if "condition_item_id" in model else 0
        if len(model) < min_rows or n_subjects < 5:
            rows.append(
                {
                    "model": model_name,
                    "roi_set": roi_set,
                    "roi": roi,
                    "outcome": outcome,
                    "term": "__model__",
                    "estimate": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "n_rows": len(model),
                    "n_subjects": n_subjects,
                    "n_condition_items": n_items,
                    "status": "skipped_insufficient_data",
                    "formula": formula,
                }
            )
            continue
        try:
            result = _fit_gee(formula, model, family)
            for term in result.params.index:
                rows.append(
                    {
                        "model": model_name,
                        "roi_set": roi_set,
                        "roi": roi,
                        "outcome": outcome,
                        "term": term,
                        "estimate": float(result.params.get(term, np.nan)),
                        "se": float(result.bse.get(term, np.nan)),
                        "z": float(result.tvalues.get(term, np.nan)),
                        "p": float(result.pvalues.get(term, np.nan)),
                        "n_rows": len(model),
                        "n_subjects": n_subjects,
                        "n_condition_items": n_items,
                        "status": "ok",
                        "formula": formula,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "roi_set": roi_set,
                    "roi": roi,
                    "outcome": outcome,
                    "term": "__model__",
                    "estimate": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "n_rows": len(model),
                    "n_subjects": n_subjects,
                    "n_condition_items": n_items,
                    "status": f"failed: {exc}",
                    "formula": formula,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        ok = out["status"].eq("ok") & out["p"].notna()
        for (_, _, _), idx in out[ok].groupby(["model", "roi_set", "term"]).groups.items():
            out.loc[idx, "q"] = bh_fdr(out.loc[idx, "p"])
    return out


def add_residualized_post_specificity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["post_edge_specificity_resid"] = np.nan
    covs = [col for col in ["pre_pair_similarity", "sentence_char_len", "word_frequency_mean", "stroke_count_mean", "valence_mean", "arousal_mean"] if col in out.columns]
    for (roi_set, roi), sub in out.groupby(["roi_set", "roi"], sort=False):
        cols = ["post_edge_specificity", *covs]
        model = sub[cols].copy()
        for col in cols:
            model[col] = pd.to_numeric(model[col], errors="coerce")
        valid = model["post_edge_specificity"].notna()
        if "pre_pair_similarity" in model.columns:
            valid &= model["pre_pair_similarity"].notna()
        if valid.sum() < 20:
            continue
        x_cols = [col for col in covs if model.loc[valid, col].notna().mean() >= 0.8 and model.loc[valid, col].nunique(dropna=True) > 1]
        x = model.loc[valid, x_cols].copy()
        for col in x_cols:
            x[col] = x[col].fillna(x[col].mean())
        x = sm.add_constant(x, has_constant="add")
        y = model.loc[valid, "post_edge_specificity"]
        try:
            fit = sm.OLS(y, x).fit()
            out.loc[sub.index[valid], "post_edge_specificity_resid"] = fit.resid
        except Exception:
            continue
    for (roi_set, roi), sub in out.groupby(["roi_set", "roi"], sort=False):
        vals = pd.to_numeric(sub["post_edge_specificity_resid"], errors="coerce")
        sd = vals.std(ddof=1)
        if math.isclose(float(sd), 0.0) if np.isfinite(sd) else True:
            out.loc[sub.index, "post_edge_specificity_resid_z"] = 0.0
        else:
            out.loc[sub.index, "post_edge_specificity_resid_z"] = (vals - vals.mean()) / sd
    return out
