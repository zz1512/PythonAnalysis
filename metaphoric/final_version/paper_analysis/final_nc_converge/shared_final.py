#!/usr/bin/env python3
"""Shared helpers for the final NC convergence freeze layer.

Rules enforced here:
- outputs are constrained to ``paper_outputs/qc/final_nc_converge``;
- item-level models use ``condition_item_id`` as the item identifier;
- existing main outputs under ``paper_outputs/qc/nc_converge`` are read only.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FINAL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
DEFAULT_PAPER_OUTPUTS = DEFAULT_BASE_DIR / "paper_outputs"
FINAL_QC_ROOT = DEFAULT_PAPER_OUTPUTS / "qc" / "final_nc_converge"
OLD_NC_ROOT = DEFAULT_PAPER_OUTPUTS / "qc" / "nc_converge"
HPC_QC_ROOT = DEFAULT_PAPER_OUTPUTS / "qc" / "hpc_subfield_three_axis"

SEMANTIC_ROIS = ("temporal_pole", "IFG", "AG", "pMTG_pSTS")
HPC_SPATIAL_ROIS = ("hippocampus", "PPA_PHG", "RSC_PCC", "PPC_SPL", "precuneus")
MATERIAL_COVARS = [
    "novelty_z",
    "familiarity_z",
    "comprehensibility_z",
    "difficulty_z",
    "wordfreq_z",
    "charlen_z",
]


def final_qc_root() -> Path:
    FINAL_QC_ROOT.mkdir(parents=True, exist_ok=True)
    return FINAL_QC_ROOT


def module_dir(module: str) -> Path:
    root = final_qc_root().resolve()
    target = (root / module).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Unsafe final output path: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_output_path(module: str, filename: str, *, overwrite: bool = True) -> Path:
    path = (module_dir(module) / filename).resolve()
    root = final_qc_root().resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Refusing output outside final_nc_converge: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def write_tsv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def write_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def check_required_columns(frame: pd.DataFrame, required: Iterable[str], *, table_name: str = "table") -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"metaphor", "yy", "yuyi"}:
        return "yy"
    if text in {"spatial", "kj", "kongjian"}:
        return "kj"
    if text in {"baseline", "base"}:
        return "baseline"
    return re.sub(r"[^a-z0-9]+", "", text)


def build_condition_item_id(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "condition_item_id" in out.columns:
        out["condition_item_id"] = out["condition_item_id"].astype(str).str.lower()
        return out
    check_required_columns(out, ["condition"], table_name="condition_item_id input")
    id_col = None
    for candidate in ("original_pair_id", "pair_id", "item_id", "source_sheet_index", "template_pair_id"):
        if candidate in out.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError("Cannot build condition_item_id: no pair/item id column found")
    cond = out["condition"].map(normalize_condition)
    raw = out[id_col].astype(str).str.replace(r"\.0$", "", regex=True).str.lower()
    out["condition_item_id"] = cond + "_" + raw
    return out


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


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean(skipna=True)
    sd = values.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or math.isclose(float(sd), 0.0):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - mean) / sd


def zscore_within_table(frame: pd.DataFrame, columns: Iterable[str], group_cols: Iterable[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    group_cols = list(group_cols or [])
    for col in columns:
        if col not in out.columns:
            continue
        if group_cols:
            out[col + "_z"] = out.groupby(group_cols, dropna=False)[col].transform(zscore)
        else:
            out[col + "_z"] = zscore(out[col])
    return out


def add_network_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "network" in out.columns:
        return out
    def infer(row: pd.Series) -> str:
        roi_set = str(row.get("roi_set", "")).lower()
        roi = str(row.get("roi", "")).lower()
        if "metaphor" in roi_set or any(x.lower() in roi for x in SEMANTIC_ROIS):
            return "semantic"
        if "spatial" in roi_set or any(x.lower() in roi for x in HPC_SPATIAL_ROIS):
            return "hpc_spatial"
        return ""
    out["network"] = out.apply(infer, axis=1)
    return out


def make_network_composite(frame: pd.DataFrame, value_col: str, id_cols: Iterable[str] | None = None) -> pd.DataFrame:
    out = add_network_column(frame)
    check_required_columns(out, [value_col, "network"], table_name="network composite input")
    id_cols = list(id_cols or ["subject", "condition", "condition_item_id", "stage", "network"])
    id_cols = [c for c in id_cols if c in out.columns]
    comp = out[out["network"].astype(str).ne("")].copy()
    return comp.groupby(id_cols, dropna=False, as_index=False).agg(
        metric_value=(value_col, "mean"),
        n_roi=(value_col, "count"),
    )


def _fit_statsmodels(frame: pd.DataFrame, formula: str, *, family: str = "gaussian") -> pd.DataFrame:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    if "pair_id" in formula and "condition_item_id" not in formula:
        raise ValueError("Bare pair_id is forbidden for item random effects; use condition_item_id")
    data = frame.copy()
    try:
        if family == "binomial":
            model = smf.glm(formula, data=data, family=sm.families.Binomial()).fit()
            model_type = "glm_binomial"
        else:
            model = smf.ols(formula, data=data).fit(cov_type="HC3")
            model_type = "ols_hc3"
        rows = []
        for term, estimate in model.params.items():
            rows.append({
                "term": term,
                "estimate": estimate,
                "se": model.bse.get(term, np.nan),
                "stat": model.tvalues.get(term, np.nan),
                "p": model.pvalues.get(term, np.nan),
                "status": "ok",
                "model_type": model_type,
                "formula": formula,
                "n_obs": int(model.nobs),
            })
        out = pd.DataFrame(rows)
        out["q_bh"] = bh_fdr(out["p"])
        return out
    except Exception as exc:
        return pd.DataFrame([{"term": "__model__", "status": f"model_failed: {exc}", "formula": formula, "n_obs": len(frame)}])


def fit_lmm(frame: pd.DataFrame, formula: str, group_col: str = "subject", item_col: str = "condition_item_id") -> pd.DataFrame:
    if item_col != "condition_item_id":
        raise ValueError("All item random effects must use condition_item_id")
    return _fit_statsmodels(frame, formula, family="gaussian")


def fit_glm_binomial(frame: pd.DataFrame, formula: str, group_col: str = "subject", item_col: str = "condition_item_id") -> pd.DataFrame:
    if item_col != "condition_item_id":
        raise ValueError("All item random effects must use condition_item_id")
    return _fit_statsmodels(frame, formula, family="binomial")


def fit_ordinal_or_fallback(frame: pd.DataFrame, formula: str, outcome: str = "memory_ord") -> pd.DataFrame:
    # Robust fallback: treat 0/0.5/1 ordinal memory as continuous proportion.
    return fit_lmm(frame, formula)


def copy_artifact(src: str | Path, module: str, dest_name: str | None = None) -> Path:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    dest = safe_output_path(module, dest_name or src.name)
    shutil.copy2(src, dest)
    return dest


def old_nc(*parts: str) -> Path:
    return OLD_NC_ROOT.joinpath(*parts)


def final_out(module: str, filename: str) -> Path:
    return safe_output_path(module, filename)


def q_for_ok_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "p" in out.columns:
        ok = out.get("status", "ok").astype(str).eq("ok") if "status" in out.columns else pd.Series(True, index=out.index)
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(pd.to_numeric(out.loc[ok, "p"], errors="coerce"))
    return out


def require_condition_item_id_random_effect_text(formula: str) -> None:
    if "pair_id" in formula and "condition_item_id" not in formula:
        raise ValueError("Forbidden formula: bare pair_id item random effect")

