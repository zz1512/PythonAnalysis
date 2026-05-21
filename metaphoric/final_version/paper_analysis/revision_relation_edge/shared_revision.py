#!/usr/bin/env python3
"""Shared helpers for relation-edge story revision scripts."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
DEFAULT_PAPER_OUTPUTS = DEFAULT_BASE_DIR / "paper_outputs"
FINAL_NC = DEFAULT_PAPER_OUTPUTS / "qc" / "final_nc_converge"
REVISION_QC = PROJECT_ROOT / "paper_outputs" / "qc" / "revision_relation_edge"
REVISION_FIG = PROJECT_ROOT / "paper_outputs" / "figures_revision"

NETWORKS = ("semantic", "hpc_spatial")
MATERIAL_COVARS = [
    "novelty_z",
    "familiarity_z",
    "comprehensibility_z",
    "difficulty_z",
    "wordfreq_z",
    "charlen_z",
]
EXTRA_ITEM_COVARS = ["valence_z", "arousal_z", "stroke_z"]


def module_dir(module: str) -> Path:
    root = REVISION_QC.resolve()
    target = (root / module).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe output path: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def out_path(module: str, filename: str) -> Path:
    return module_dir(module) / filename


def fig_path(filename: str) -> Path:
    REVISION_FIG.mkdir(parents=True, exist_ok=True)
    return REVISION_FIG / filename


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False, **kwargs)
    raise ValueError(f"Unsupported table format: {path}")


def write_tsv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def write_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or math.isclose(float(sd), 0.0):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - values.mean(skipna=True)) / sd


def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    values = pd.to_numeric(pd.Series(pvalues), errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    vals = values[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[np.flatnonzero(ok)[order]] = np.clip(adj, 0, 1)
    return out


def clean_condition(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["condition"] = out["condition"].astype(str).str.lower().replace(
        {"metaphor": "yy", "spatial": "kj", "yy": "yy", "kj": "kj"}
    )
    return out


def ensure_condition_item_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = clean_condition(frame)
    if "condition_item_id" in out.columns:
        out["condition_item_id"] = out["condition_item_id"].astype(str).str.lower()
        return out
    id_col = next((c for c in ["original_pair_id", "pair_id", "template_pair_id", "item_id"] if c in out.columns), None)
    if id_col is None:
        raise ValueError("No usable item id column found")
    raw = out[id_col].astype(str).str.replace(r"\.0$", "", regex=True).str.lower()
    out["condition_item_id"] = out["condition"] + "_" + raw
    return out


def load_revision_master_network() -> pd.DataFrame:
    path = REVISION_QC / "r0_build_revision_master_table" / "revision_master_network.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Run r0 first: {path}")
    return read_table(path)


def numeric_cols_available(frame: pd.DataFrame, cols: Iterable[str], min_nonmissing: int = 20) -> list[str]:
    keep = []
    for col in cols:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= min_nonmissing:
            keep.append(col)
    return keep


def fit_ols_hc3(frame: pd.DataFrame, formula: str, *, model: str, network: str | None = None) -> pd.DataFrame:
    data = frame.copy()
    try:
        import statsmodels.formula.api as smf

        res = smf.ols(formula, data=data).fit(cov_type="HC3")
        rows = []
        for term, estimate in res.params.items():
            rows.append({
                "network": network,
                "model": model,
                "term": term,
                "estimate": estimate,
                "se": res.bse.get(term, np.nan),
                "stat": res.tvalues.get(term, np.nan),
                "p": res.pvalues.get(term, np.nan),
                "status": "ok",
                "model_type": "ols_hc3_revision_fallback",
                "formula": formula,
                "n_obs": int(res.nobs),
            })
        out = pd.DataFrame(rows)
        out["q_bh"] = bh_fdr(out["p"])
        return out
    except ModuleNotFoundError:
        return fit_ols_hc3_numpy(data, formula, model=model, network=network)
    except Exception as exc:
        return pd.DataFrame([{
            "network": network,
            "model": model,
            "term": "__model__",
            "status": f"model_failed: {exc}",
            "model_type": "ols_hc3_revision_fallback",
            "formula": formula,
            "n_obs": len(data),
        }])


def _formula_terms(rhs: str) -> list[str]:
    terms: list[str] = []
    for raw in rhs.split("+"):
        term = raw.strip()
        if not term or term == "1":
            continue
        if "*" in term:
            factors = [part.strip() for part in term.split("*")]
            terms.extend(factors)
            if len(factors) == 2:
                terms.append(f"{factors[0]}:{factors[1]}")
            else:
                terms.append(":".join(factors))
        else:
            terms.append(term)
    out: list[str] = []
    for term in terms:
        if term not in out:
            out.append(term)
    return out


def _factor_columns(data: pd.DataFrame, factor: str) -> dict[str, pd.Series]:
    factor = factor.strip()
    cat = re.fullmatch(r"C\(([^,)]+)(?:,[^)]+)?\)", factor)
    if cat:
        var = cat.group(1).strip()
        if var not in data.columns:
            return {}
        values = data[var].astype(str)
        levels = sorted(v for v in values.dropna().unique() if v.lower() != "nan")
        if len(levels) <= 1:
            return {}
        cols = {}
        for level in levels[1:]:
            cols[f"C({var})[T.{level}]"] = values.eq(level).astype(float)
        return cols

    square = re.fullmatch(r"I\(([^*]+)\*\*\s*2\)", factor.replace(" ", ""))
    if square:
        var = square.group(1).strip()
        if var not in data.columns:
            return {}
        vals = pd.to_numeric(data[var], errors="coerce")
        return {factor: vals * vals}

    if factor in data.columns:
        return {factor: pd.to_numeric(data[factor], errors="coerce")}
    return {}


def _term_columns(data: pd.DataFrame, term: str) -> dict[str, pd.Series]:
    factors = [part.strip() for part in term.split(":")]
    factor_cols = [_factor_columns(data, factor) for factor in factors]
    if any(not cols for cols in factor_cols):
        return {}
    out = factor_cols[0]
    for cols in factor_cols[1:]:
        merged: dict[str, pd.Series] = {}
        for name_a, vals_a in out.items():
            for name_b, vals_b in cols.items():
                merged[f"{name_a}:{name_b}"] = vals_a * vals_b
        out = merged
    return out


def fit_ols_hc3_numpy(frame: pd.DataFrame, formula: str, *, model: str, network: str | None = None) -> pd.DataFrame:
    """Small formula fallback for environments without statsmodels."""
    try:
        lhs, rhs = [part.strip() for part in formula.split("~", 1)]
        y = pd.to_numeric(frame[lhs], errors="coerce")
        columns: dict[str, pd.Series] = {"Intercept": pd.Series(1.0, index=frame.index)}
        for term in _formula_terms(rhs):
            columns.update(_term_columns(frame, term))
        design = pd.DataFrame(columns, index=frame.index)
        model_data = pd.concat([y.rename(lhs), design], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_data) <= max(2, len(design.columns)):
            raise ValueError("insufficient complete rows for numpy OLS")
        yv = model_data[lhs].to_numpy(dtype=float)
        x = model_data[list(columns)].to_numpy(dtype=float)
        xtx_inv = np.linalg.pinv(x.T @ x)
        beta = xtx_inv @ x.T @ yv
        fitted = x @ beta
        resid = yv - fitted
        hat = np.sum((x @ xtx_inv) * x, axis=1)
        denom = np.clip(1.0 - hat, 1e-8, None)
        scaled = (resid / denom) ** 2
        meat = x.T @ (x * scaled[:, None])
        cov = xtx_inv @ meat @ xtx_inv
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        stat = beta / se
        p = np.array([math.erfc(abs(float(t)) / math.sqrt(2.0)) if np.isfinite(t) else np.nan for t in stat])
        rows = []
        for name, estimate, std_err, zval, pval in zip(columns, beta, se, stat, p):
            rows.append({
                "network": network,
                "model": model,
                "term": name,
                "estimate": estimate,
                "se": std_err,
                "stat": zval,
                "p": pval,
                "status": "ok",
                "model_type": "ols_hc3_numpy_fallback",
                "formula": formula,
                "n_obs": int(len(model_data)),
            })
        out = pd.DataFrame(rows)
        out["q_bh"] = bh_fdr(out["p"])
        return out
    except Exception as exc:
        return pd.DataFrame([{
            "network": network,
            "model": model,
            "term": "__model__",
            "status": f"model_failed: {exc}",
            "model_type": "ols_hc3_numpy_fallback",
            "formula": formula,
            "n_obs": len(frame),
        }])


def mean_sem(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, 0
    return float(values.mean()), float(values.sem()) if len(values) > 1 else np.nan, int(len(values))
