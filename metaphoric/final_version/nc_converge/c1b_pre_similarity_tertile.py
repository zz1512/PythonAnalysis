#!/usr/bin/env python3
"""C1b: pre-similarity tertile moderation of post-stage separation.

Goal
----
回答审稿/老师追问："对于学习前更相似的句对，YY 学习是否产生更强的精细化分离？"
C1 的 ``post_pair_similarity_z ~ pre_pair_similarity_z * condition`` 已经说明 pre→post
的总体斜率为强正向，但 ``pre × condition`` 交互不显著（hpc-spatial q≈0.75，
semantic q≈0.45）。本脚本提供一个互补的 tertile 设计：

1. 按 ``subject × network × condition_item_id`` 聚合 raw ROI 相似度，得到 network 层的
   ``pre_pair_similarity_network_raw`` / ``post_pair_similarity_network_raw`` 与
   ``post_edge_differentiation_raw``（``post_edge_specificity`` 优先，缺失时 fallback
   到 ``trained_edge_drop``）。
2. 在每个 network × condition 内按 pre_pair_similarity_network_raw 划分 ``low / mid /
   high`` tertile（默认 30 / 40 / 30）。tertile 标签是 condition-internal，避免 KJ/YY
   分布差异让 high tertile 全部落在某一 condition。
3. 跨 condition 拟合两条 confirmatory model：

   * ``post_edge_differentiation_z ~ condition * pre_tertile + covariates``
   * ``post_pair_similarity_z      ~ condition * pre_tertile + covariates``

   其中 outcome 用单次全局 z（沿用 §28.4 修复后的单位策略）。tertile 用
   ``Treatment(reference='low')`` 编码，因此 ``condition[T.yy]:pre_tertile[T.high]`` 直接
   对应 "YY 在 high pre-similarity tertile 相对 low tertile 的额外分化"。
4. 额外输出 ``focal_contrasts.tsv``：

   * YY high vs YY low 的均值差（带 SE / p / q）
   * KJ high vs KJ low 的均值差（作为 active control）
   * difference-in-differences (YY[high-low] − KJ[high-low])

Outputs
-------
``paper_outputs/qc/nc_converge/c1b_pre_similarity_tertile/`` 下：

* ``pre_tertile_descriptives.tsv``：每个 network × condition × tertile 的均值
* ``pre_tertile_model_coefficients.tsv``：两条 model 的全部回归系数 + BH-FDR
* ``focal_contrasts.tsv``：上述三类对照
* ``pre_tertile_manifest.tsv``：输入路径 / 行数 / 状态
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    add_network_column,
    bh_fdr,
    build_condition_item_id,
    default_config,
    fit_formula,
    q_for_ok_rows,
    read_table,
    write_outputs,
    zscore,
)

MODULE = "c1b_pre_similarity_tertile"
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")

LOW_QUANTILE = 0.30
HIGH_QUANTILE = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=LOW_QUANTILE,
        help="Quantile cutoff for the low pre-similarity tertile (default 0.30).",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=HIGH_QUANTILE,
        help="Quantile cutoff for the high pre-similarity tertile (default 0.70).",
    )
    return parser.parse_args()


def usable_covariates(frame: pd.DataFrame) -> list[str]:
    out = []
    for col in COVARIATES:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10:
            out.append(col)
    return out


def prepare(frame: pd.DataFrame, low_q: float, high_q: float) -> pd.DataFrame:
    """Build subject × network × condition_item_id table with tertile labels."""
    out = add_network_column(build_condition_item_id(frame))
    metric_col = (
        "post_edge_specificity"
        if "post_edge_specificity" in out.columns
        else ("trained_edge_drop" if "trained_edge_drop" in out.columns else None)
    )
    raw_metrics = [c for c in ["pre_pair_similarity", "post_pair_similarity"] if c in out.columns]
    if metric_col is None or not raw_metrics:
        return pd.DataFrame()
    keep_cols = list(dict.fromkeys([*raw_metrics, metric_col, *COVARIATES]))
    for col in keep_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "network" not in out.columns or "condition_item_id" not in out.columns:
        return pd.DataFrame()
    out = out[out["network"].notna() & out["condition_item_id"].astype(str).ne("")].copy()
    id_keys = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in out.columns]
    agg_cols = [c for c in keep_cols if c in out.columns]
    network_raw = out.groupby(id_keys, dropna=False, as_index=False)[agg_cols].mean()
    rename_map = {
        "pre_pair_similarity": "pre_pair_similarity_network_raw",
        "post_pair_similarity": "post_pair_similarity_network_raw",
        metric_col: "post_edge_differentiation_raw",
    }
    network_raw = network_raw.rename(columns=rename_map)
    # Single global z-score (consistent with §28.4 fix policy).
    if "pre_pair_similarity_network_raw" in network_raw.columns:
        network_raw["pre_pair_similarity_z"] = zscore(network_raw["pre_pair_similarity_network_raw"])
    if "post_pair_similarity_network_raw" in network_raw.columns:
        network_raw["post_pair_similarity_z"] = zscore(network_raw["post_pair_similarity_network_raw"])
    if "post_edge_differentiation_raw" in network_raw.columns:
        network_raw["post_edge_differentiation_z"] = zscore(network_raw["post_edge_differentiation_raw"])
    network_raw["pre_tertile"] = _assign_tertile(network_raw, low_q=low_q, high_q=high_q)
    return network_raw


def _assign_tertile(frame: pd.DataFrame, *, low_q: float, high_q: float) -> pd.Series:
    """Assign condition-internal low/mid/high tertile labels per network."""
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    if "pre_pair_similarity_network_raw" not in frame.columns:
        return labels
    if not {"network", "condition"}.issubset(frame.columns):
        return labels
    for (_network, _condition), idx in frame.groupby(["network", "condition"], dropna=False).groups.items():
        sub = frame.loc[idx, "pre_pair_similarity_network_raw"].astype(float)
        if sub.notna().sum() < 6:
            continue
        low_cut = sub.quantile(low_q)
        high_cut = sub.quantile(high_q)
        if not (np.isfinite(low_cut) and np.isfinite(high_cut)) or high_cut <= low_cut:
            continue
        for ix in idx:
            value = sub.loc[ix]
            if not np.isfinite(value):
                continue
            if value <= low_cut:
                labels.loc[ix] = "low"
            elif value >= high_cut:
                labels.loc[ix] = "high"
            else:
                labels.loc[ix] = "mid"
    return labels


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "network",
        "condition",
        "pre_tertile",
        "pre_pair_similarity_network_raw",
        "post_pair_similarity_network_raw",
        "post_edge_differentiation_raw",
    }
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    grouped = data.groupby(["network", "condition", "pre_tertile"], dropna=False, as_index=False).agg(
        n=("post_edge_differentiation_raw", "count"),
        n_subjects=("subject", "nunique") if "subject" in data.columns else ("post_edge_differentiation_raw", "size"),
        n_items=("condition_item_id", "nunique") if "condition_item_id" in data.columns else ("post_edge_differentiation_raw", "size"),
        pre_mean_raw=("pre_pair_similarity_network_raw", "mean"),
        post_mean_raw=("post_pair_similarity_network_raw", "mean"),
        post_edge_mean_raw=("post_edge_differentiation_raw", "mean"),
        post_edge_sd_raw=("post_edge_differentiation_raw", "std"),
        post_edge_mean_z=("post_edge_differentiation_z", "mean") if "post_edge_differentiation_z" in data.columns else ("post_edge_differentiation_raw", "size"),
    )
    return grouped


def fit_models(data: pd.DataFrame) -> pd.DataFrame:
    needed = {"subject", "condition_item_id", "condition", "network", "pre_tertile"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"term": "__model__", "status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    rows: list[pd.DataFrame] = []
    outcome_specs = [
        ("post_edge_differentiation_z", "edge_separation_by_pre_tertile"),
        ("post_pair_similarity_z", "post_similarity_by_pre_tertile"),
    ]
    for network, sub in data.groupby("network", dropna=False):
        covs = usable_covariates(sub)
        for outcome, model_name in outcome_specs:
            if outcome not in sub.columns:
                continue
            model_data = sub.dropna(subset=[outcome, "pre_tertile", *covs]).copy()
            model_data = model_data[model_data["pre_tertile"].isin(["low", "mid", "high"])]
            if len(model_data) < 30:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "network": network,
                                "outcome": outcome,
                                "mechanism_model": model_name,
                                "term": "__model__",
                                "status": "too_few_rows",
                                "n_obs": len(model_data),
                            }
                        ]
                    )
                )
                continue
            # Treatment-coded with reference 'low' so that condition[T.yy]:pre_tertile[T.high]
            # encodes the YY-specific extra separation in the high pre tertile.
            formula = (
                f"{outcome} ~ C(condition, Treatment(reference='kj')) "
                f"* C(pre_tertile, Treatment(reference='low'))"
            )
            if covs:
                formula += " + " + " + ".join(covs)
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "outcome", outcome)
            res.insert(2, "mechanism_model", model_name)
            rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def focal_contrasts(data: pd.DataFrame) -> pd.DataFrame:
    """Compute YY high-low / KJ high-low / DiD on post_edge_differentiation_z."""
    needed = {"network", "condition", "pre_tertile", "post_edge_differentiation_z"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    rows: list[dict] = []
    for network, sub in data.groupby("network", dropna=False):
        contrasts = {}
        for condition in sub["condition"].dropna().unique():
            cond_data = sub[sub["condition"] == condition]
            high = cond_data[cond_data["pre_tertile"] == "high"]["post_edge_differentiation_z"].astype(float)
            low = cond_data[cond_data["pre_tertile"] == "low"]["post_edge_differentiation_z"].astype(float)
            high = high[np.isfinite(high)]
            low = low[np.isfinite(low)]
            if len(high) < 5 or len(low) < 5:
                continue
            est = float(high.mean() - low.mean())
            se = float(np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low)))
            stat = est / se if se > 0 else float("nan")
            from math import erf, sqrt
            p = float(2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))) if np.isfinite(stat) else float("nan")
            contrasts[str(condition).lower()] = {
                "estimate": est,
                "se": se,
                "stat": stat,
                "p": p,
                "n_high": int(len(high)),
                "n_low": int(len(low)),
            }
            rows.append(
                {
                    "network": network,
                    "contrast": f"{str(condition).lower()}_high_minus_low",
                    "estimate": est,
                    "se": se,
                    "stat": stat,
                    "p": p,
                    "n_high": int(len(high)),
                    "n_low": int(len(low)),
                }
            )
        if "yy" in contrasts and "kj" in contrasts:
            yy = contrasts["yy"]
            kj = contrasts["kj"]
            est = yy["estimate"] - kj["estimate"]
            se = float(np.sqrt(yy["se"] ** 2 + kj["se"] ** 2))
            stat = est / se if se > 0 else float("nan")
            from math import erf, sqrt
            p = float(2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))) if np.isfinite(stat) else float("nan")
            rows.append(
                {
                    "network": network,
                    "contrast": "did_yy_minus_kj_high_low",
                    "estimate": est,
                    "se": se,
                    "stat": stat,
                    "p": p,
                    "n_high": yy["n_high"] + kj["n_high"],
                    "n_low": yy["n_low"] + kj["n_low"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and "p" in out.columns:
        out["q_bh"] = bh_fdr(out["p"])
    return out


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    raw = read_table(path)
    data = prepare(raw, low_q=float(args.low_quantile), high_q=float(args.high_quantile))
    desc = descriptives(data)
    models = fit_models(data)
    contrasts = focal_contrasts(data)
    write_outputs(
        cfg,
        MODULE,
        {
            "pre_tertile_descriptives.tsv": desc,
            "pre_tertile_model_coefficients.tsv": models,
            "focal_contrasts.tsv": contrasts,
            "pre_tertile_manifest.tsv": pd.DataFrame(
                [
                    {
                        "path": str(path),
                        "status": "ok",
                        "n_rows": len(raw),
                        "low_quantile": float(args.low_quantile),
                        "high_quantile": float(args.high_quantile),
                    }
                ]
            ),
        },
    )


if __name__ == "__main__":
    main()
