#!/usr/bin/env python3
"""A4b: ordinal Dm with three-level memory rating granularity.

参考 ``.trae/specs/expand-dm-narrative/spec.md`` §4.2。

目标
----
扩展 a4 的 binary memory_strict / memory_lenient + gaussian memory_prop 套路：
显式把 ``memory_score`` ∈ {0, 0.5, 1} 与 ``memory_successes`` ∈ {0, 1, 2}
作为 ordinal outcome 跑 ordered logit；在 statsmodels OrderedModel 不可用
或拟合失败时 fallback 到 binomial(memory_strict) + gaussian(memory_score)
双跑作为 sanity。

不修改 [a4_post_memory_component.py](a4_post_memory_component.py) 与其输出沙箱；
复用其 prepare 后的全局 z 列（``post_pair_similarity_(_network)_z`` 等）。

输出沙箱: ``paper_outputs/qc/nc_converge/a4b_ordinal_dm/``。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from a4_post_memory_component import (
    PATTERNS as A4_PATTERNS,
    formula_with_covariates,
    load as a4_load,
    prepare as a4_prepare,
    usable_covariates,
)
from shared_nc import (
    add_common_args,
    bh_fdr,
    default_config,
    discover_tables,
    fit_formula,
    q_for_ok_rows,
    write_outputs,
)

MODULE = "a4b_ordinal_dm"
DEFAULT_INPUT = Path("paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    parser.add_argument(
        "--ordinal-engine",
        choices=("ordered_model", "gaussian_sanity", "binomial_sanity"),
        default="ordered_model",
        help="Ordinal logit 引擎；ordered_model 不可用时 fallback 自动启用。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# OrderedModel fitting
# ---------------------------------------------------------------------------


def _try_ordered_model(
    frame: pd.DataFrame,
    formula: str,
    *,
    outcome: str,
) -> pd.DataFrame:
    """Try statsmodels.miscmodels.ordinal_model.OrderedModel; safe fallback row."""
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel  # type: ignore
        from patsy import dmatrices  # type: ignore
    except Exception as exc:
        return pd.DataFrame([
            {
                "term": "__model__",
                "status": f"ordered_model_unavailable: {exc}",
                "formula": formula,
                "n_obs": int(len(frame)),
            }
        ])
    work = frame.copy()
    if work[outcome].isna().any():
        work = work.dropna(subset=[outcome])
    if len(work) < 30 or work[outcome].nunique(dropna=True) < 2:
        return pd.DataFrame([
            {
                "term": "__model__",
                "status": "too_few_levels_or_rows",
                "formula": formula,
                "n_obs": int(len(work)),
            }
        ])
    rhs = formula.split("~", 1)[1].strip()
    try:
        y, X = dmatrices(f"{outcome} ~ {rhs}", data=work, return_type="dataframe")
    except Exception as exc:
        return pd.DataFrame([
            {
                "term": "__model__",
                "status": f"design_matrix_failed: {exc}",
                "formula": formula,
                "n_obs": int(len(work)),
            }
        ])
    # OrderedModel 不接受截距列，去掉 patsy 默认 Intercept
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    y_arr = np.asarray(y).reshape(-1)
    try:
        model = OrderedModel(y_arr, X, distr="logit")
        result = model.fit(method="bfgs", disp=False, maxiter=200)
    except Exception as exc:
        return pd.DataFrame([
            {
                "term": "__model__",
                "status": f"ordered_fit_failed: {exc}",
                "formula": formula,
                "n_obs": int(len(work)),
            }
        ])
    rows: List[dict] = []
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    for term in params.index:
        rows.append(
            {
                "term": str(term),
                "estimate": float(params[term]),
                "std_error": float(bse[term]) if term in bse.index else float("nan"),
                "p": float(pvalues[term]) if term in pvalues.index else float("nan"),
                "z": float(params[term] / bse[term]) if term in bse.index and bse[term] not in (0, np.nan) else float("nan"),
                "status": "ok",
                "n_obs": int(len(work)),
                "formula": formula,
            }
        )
    out = pd.DataFrame(rows)
    return out


def _binary_sanity(
    frame: pd.DataFrame,
    formula: str,
    outcome: str,
) -> pd.DataFrame:
    return fit_formula(frame, formula.replace(outcome, "memory_strict", 1), family="binomial")


def _gaussian_sanity(
    frame: pd.DataFrame,
    formula: str,
    outcome: str,
) -> pd.DataFrame:
    return fit_formula(frame, formula.replace(outcome, "memory_score", 1), family="gaussian")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fit_ordinal_models(data: pd.DataFrame, engine: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    post_col = "post_pair_similarity_network_z" if "post_pair_similarity_network_z" in data.columns else "post_pair_similarity_z"
    retr_col = "retrieval_pair_similarity_network_z" if "retrieval_pair_similarity_network_z" in data.columns else "retrieval_pair_similarity_z"
    rows: List[pd.DataFrame] = []
    for outcome in ("memory_score", "memory_successes"):
        if outcome not in data.columns:
            rows.append(pd.DataFrame([{"outcome": outcome, "term": "__model__", "status": "missing_outcome"}]))
            continue
        for network, sub in data.groupby("network", dropna=False):
            covariates = usable_covariates(sub) if not sub.empty else []
            cov_expr = (" + " + " + ".join(covariates)) if covariates else ""
            base_formula = (
                f"{outcome} ~ C(condition, Treatment('kj')) * {post_col}"
                f" + C(condition, Treatment('kj')) * {retr_col}{cov_expr}"
            )
            need_cols = [outcome, post_col, retr_col, "condition", *covariates]
            model_data = sub.dropna(subset=[c for c in need_cols if c in sub.columns]).copy()
            if len(model_data) < 30 or model_data[outcome].nunique(dropna=True) < 2:
                rows.append(pd.DataFrame([
                    {
                        "network": network,
                        "outcome": outcome,
                        "engine": engine,
                        "term": "__model__",
                        "status": "too_few_rows_or_levels",
                        "n_obs": int(len(model_data)),
                    }
                ]))
                continue
            if engine == "ordered_model":
                res = _try_ordered_model(model_data, base_formula, outcome=outcome)
                if "status" in res.columns and res.iloc[0]["status"] != "ok":
                    # 自动 fallback 到 binomial sanity
                    res_fallback_b = _binary_sanity(model_data, base_formula, outcome)
                    res_fallback_b.insert(0, "fallback", "binomial_sanity")
                    res_fallback_g = _gaussian_sanity(model_data, base_formula, outcome)
                    res_fallback_g.insert(0, "fallback", "gaussian_sanity")
                    res = pd.concat([res, res_fallback_b, res_fallback_g], ignore_index=True, sort=False)
            elif engine == "binomial_sanity":
                res = _binary_sanity(model_data, base_formula, outcome)
            else:
                res = _gaussian_sanity(model_data, base_formula, outcome)
            res.insert(0, "network", network)
            res.insert(1, "outcome", outcome)
            res.insert(2, "engine", engine)
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True, sort=False)
    if "p" in table.columns:
        ok_mask = table.get("status", pd.Series(dtype=str)).astype(str).eq("ok")
        if ok_mask.any():
            table["q_bh"] = np.nan
            table.loc[ok_mask, "q_bh"] = bh_fdr(table.loc[ok_mask, "p"])
    return table


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    cols = [c for c in ["network", "condition"] if c in data.columns]
    if not cols:
        return pd.DataFrame()
    rows = []
    for outcome in ("memory_strict", "memory_lenient", "memory_score", "memory_successes"):
        if outcome not in data.columns:
            continue
        agg = data.groupby(cols, dropna=False).agg(
            n=(outcome, "count"),
            mean=(outcome, "mean"),
            sd=(outcome, "std"),
            n_subjects=("subject", "nunique") if "subject" in data.columns else (outcome, "count"),
            n_items=("condition_item_id", "nunique") if "condition_item_id" in data.columns else (outcome, "count"),
        ).reset_index()
        agg["outcome"] = outcome
        rows.append(agg)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_manifest(args: argparse.Namespace, data: pd.DataFrame, paths: List[Path]) -> pd.DataFrame:
    return pd.DataFrame([
        {"key": "engine", "value": args.ordinal_engine},
        {"key": "n_input_paths", "value": len(paths)},
        {"key": "n_rows", "value": int(len(data))},
        {"key": "n_subjects", "value": int(data["subject"].nunique()) if "subject" in data.columns and not data.empty else 0},
        {"key": "networks", "value": ",".join(sorted(data["network"].dropna().astype(str).unique())) if "network" in data.columns and not data.empty else ""},
    ])


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths: List[Path] = list(args.input or [])
    if not paths:
        paths = discover_tables(cfg.paper_output_root, A4_PATTERNS)
        default_path = Path(cfg.base_dir) / DEFAULT_INPUT
        if default_path.exists() and default_path not in paths:
            paths.insert(0, default_path)
    raw = a4_load(paths)
    if raw.empty and not args.allow_empty:
        raise FileNotFoundError("No item_mechanism candidate tables found; use --input or --allow-empty.")
    data = a4_prepare(raw) if not raw.empty else pd.DataFrame()
    ordinal_models = fit_ordinal_models(data, args.ordinal_engine)
    binary_sanity = fit_ordinal_models(data, "binomial_sanity") if args.ordinal_engine == "ordered_model" and not data.empty else pd.DataFrame()
    desc = descriptives(data)
    mani = build_manifest(args, data, paths)
    write_outputs(
        cfg,
        MODULE,
        {
            "ordinal_dm_models.tsv": ordinal_models if not ordinal_models.empty else pd.DataFrame(columns=["network", "outcome", "engine", "term", "status"]),
            "binary_sanity_models.tsv": binary_sanity if not binary_sanity.empty else pd.DataFrame(columns=["network", "outcome", "engine", "term", "status"]),
            "ordinal_dm_descriptives.tsv": desc if not desc.empty else pd.DataFrame(columns=["network", "condition", "outcome", "n", "mean"]),
            "manifest.tsv": mani,
        },
    )


if __name__ == "__main__":
    main()
