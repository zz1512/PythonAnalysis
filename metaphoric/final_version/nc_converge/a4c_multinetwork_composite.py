#!/usr/bin/env python3
"""A4c: multi-network composite Dm (PCA + cross-network mean).

参考 ``.trae/specs/expand-dm-narrative/spec.md`` §4.3。

目标
----
回应 reviewer 对 single-network cherry-pick 的潜在质疑：把 a4 prepare 后的
``{post,retrieval}_pair_similarity_network_z`` 在两个 network 上拼成 4 维
特征，做 PCA + cross-network mean，得到一个 subject-level / item-level
``composite_score_*``，再以 composite 作为 predictor 跑 memory_strict 模型，
并与单 network 模型对比 ΔAIC / ΔBIC / Δpseudo-R²。

不修改 a4 沙箱，只读取 a4 prepare 后的列与 a4 输出表（用于 ΔAIC 对比）。

输出沙箱: ``paper_outputs/qc/nc_converge/a4c_multinetwork_composite/``。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from a4_post_memory_component import (
    PATTERNS as A4_PATTERNS,
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
    read_table,
    write_outputs,
)

MODULE = "a4c_multinetwork_composite"
DEFAULT_INPUT = Path("paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_A4_NETWORK_TABLE = Path("paper_outputs/qc/nc_converge/a4_post_memory_component/memory_component_model.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    parser.add_argument("--a4-network-table", type=Path, default=None,
                        help="a4 输出 memory_component_model.tsv（用于 ΔAIC 对比；缺失会跳过对比但保留主模型）。")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------


def build_composite(data: pd.DataFrame) -> pd.DataFrame:
    """把 (subject, condition_item_id) × network × {post,retrieval} pivot 为 4 列宽表。"""
    if data.empty:
        return pd.DataFrame()
    needed = {"subject", "condition_item_id", "network"}
    missing = needed - set(data.columns)
    if missing:
        warnings.warn(f"build_composite missing columns: {sorted(missing)}")
        return pd.DataFrame()
    pivot_cols = []
    for stage_col in ("post_pair_similarity_network_z", "retrieval_pair_similarity_network_z"):
        if stage_col in data.columns:
            pivot_cols.append(stage_col)
        elif stage_col.replace("_network_z", "_z") in data.columns:
            pivot_cols.append(stage_col.replace("_network_z", "_z"))
    if not pivot_cols:
        warnings.warn("No *_pair_similarity_network_z columns; skip composite.")
        return pd.DataFrame()
    keep_cols = ["subject", "condition_item_id", "condition", "network", *pivot_cols]
    extra_keep = [c for c in (
        "memory_strict", "memory_lenient", "memory_score", "memory_prop",
        "sentence_char_len_z", "word_frequency_mean_z", "stroke_count_mean_z",
        "valence_mean_z", "arousal_mean_z",
    ) if c in data.columns]
    work = data[keep_cols + extra_keep].copy()
    # 多个 ROI 同 (subject, condition_item_id, network) 取平均
    grouped_keys = ["subject", "condition_item_id", "condition", "network"]
    grouped = work.groupby(grouped_keys, dropna=False, as_index=False)[pivot_cols + extra_keep].mean()
    # pivot
    wide = grouped.pivot_table(
        index=["subject", "condition_item_id", "condition"],
        columns="network",
        values=pivot_cols,
        aggfunc="mean",
    )
    if wide.empty:
        return pd.DataFrame()
    wide.columns = [f"{stage}__{net}" for stage, net in wide.columns]
    wide = wide.reset_index()
    # 补上 item-level memory + covariates（network 维度已 collapse，取 mean 即可）
    item_meta = grouped.groupby(["subject", "condition_item_id", "condition"], dropna=False, as_index=False)[extra_keep].mean()
    wide = wide.merge(item_meta, on=["subject", "condition_item_id", "condition"], how="left")
    return wide


def fit_pca(features: pd.DataFrame, columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (composite_df, loadings_df)。无 sklearn 时退化为 numpy SVD。"""
    work = features.dropna(subset=columns)
    if work.empty:
        return pd.DataFrame(), pd.DataFrame([{"status": "empty"}])
    matrix = work[columns].to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0)
    if matrix.shape[0] < 4 or len(columns) < 2:
        return pd.DataFrame(), pd.DataFrame([{"status": "too_few_obs_or_features"}])
    try:
        from numpy.linalg import svd
        U, S, Vt = svd(centered, full_matrices=False)
        scores = U * S  # (n, k)
        n_components = min(2, scores.shape[1])
        comp_df = work[["subject", "condition_item_id", "condition"]].copy()
        for k in range(n_components):
            comp_df[f"composite_score_pca{k+1}"] = scores[:, k]
        comp_df["composite_score_mean"] = matrix.mean(axis=1)
        loadings_rows = []
        for k in range(n_components):
            for j, col in enumerate(columns):
                loadings_rows.append({
                    "component": f"pca{k+1}",
                    "feature": col,
                    "loading": float(Vt[k, j]),
                    "explained_variance_ratio": float((S[k] ** 2) / np.sum(S ** 2)) if S.sum() > 0 else float("nan"),
                })
        return comp_df, pd.DataFrame(loadings_rows)
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame([{"status": f"svd_failed: {exc}"}])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_composite_models(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    covariates = usable_covariates(data) if hasattr(data, "columns") else []
    cov_expr = (" + " + " + ".join(covariates)) if covariates else ""
    for outcome in ("memory_strict", "memory_lenient", "memory_prop"):
        if outcome not in data.columns:
            continue
        family = "binomial" if outcome in ("memory_strict", "memory_lenient") else "gaussian"
        for predictor in ("composite_score_pca1", "composite_score_pca2", "composite_score_mean"):
            if predictor not in data.columns:
                continue
            need = [outcome, predictor, "condition", *covariates]
            model_data = data.dropna(subset=[c for c in need if c in data.columns]).copy()
            if len(model_data) < 30 or model_data[predictor].nunique(dropna=True) < 2:
                rows.append(pd.DataFrame([{
                    "outcome": outcome,
                    "predictor": predictor,
                    "term": "__model__",
                    "status": "too_few_rows",
                    "n_obs": int(len(model_data)),
                }]))
                continue
            formula = f"{outcome} ~ C(condition, Treatment('kj')) * {predictor}{cov_expr}"
            res = fit_formula(model_data, formula, family=family)
            res.insert(0, "outcome", outcome)
            res.insert(1, "predictor", predictor)
            res.insert(2, "model_kind", "composite")
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


def composite_vs_network_comparison(
    composite_models: pd.DataFrame,
    network_models: pd.DataFrame,
) -> pd.DataFrame:
    """比较 composite_score_pca1 与 a4 单 network 的 AIC/BIC（基于已落表的指标）。"""
    if composite_models.empty:
        return pd.DataFrame([{"status": "empty_composite_models"}])
    if network_models is None or network_models.empty:
        return pd.DataFrame([{"status": "no_a4_table_for_comparison"}])
    rows = []
    for outcome in composite_models.get("outcome", pd.Series(dtype=str)).dropna().unique():
        comp_rows = composite_models[
            (composite_models["outcome"] == outcome)
            & (composite_models.get("term", pd.Series(dtype=str)).astype(str) == "__model__")
            & (composite_models.get("predictor", pd.Series(dtype=str)).astype(str) == "composite_score_pca1")
        ]
        net_rows = network_models[
            (network_models["outcome"] == outcome)
            & (network_models.get("term", pd.Series(dtype=str)).astype(str) == "__model__")
        ] if "outcome" in network_models.columns else pd.DataFrame()
        if comp_rows.empty or net_rows.empty:
            rows.append({"outcome": outcome, "status": "missing_model_row"})
            continue
        for _, comp_row in comp_rows.iterrows():
            for _, net_row in net_rows.iterrows():
                rows.append({
                    "outcome": outcome,
                    "network": net_row.get("network"),
                    "composite_aic": float(comp_row.get("aic")) if "aic" in comp_row.index and pd.notna(comp_row.get("aic")) else float("nan"),
                    "network_aic": float(net_row.get("aic")) if "aic" in net_row.index and pd.notna(net_row.get("aic")) else float("nan"),
                    "delta_aic": (float(comp_row.get("aic")) - float(net_row.get("aic"))) if ("aic" in comp_row.index and "aic" in net_row.index and pd.notna(comp_row.get("aic")) and pd.notna(net_row.get("aic"))) else float("nan"),
                    "composite_bic": float(comp_row.get("bic")) if "bic" in comp_row.index and pd.notna(comp_row.get("bic")) else float("nan"),
                    "network_bic": float(net_row.get("bic")) if "bic" in net_row.index and pd.notna(net_row.get("bic")) else float("nan"),
                    "delta_bic": (float(comp_row.get("bic")) - float(net_row.get("bic"))) if ("bic" in comp_row.index and "bic" in net_row.index and pd.notna(comp_row.get("bic")) and pd.notna(net_row.get("bic"))) else float("nan"),
                    "composite_n_obs": int(comp_row.get("n_obs")) if "n_obs" in comp_row.index and pd.notna(comp_row.get("n_obs")) else None,
                    "network_n_obs": int(net_row.get("n_obs")) if "n_obs" in net_row.index and pd.notna(net_row.get("n_obs")) else None,
                    "status": "ok",
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "empty_comparison"}])


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(args: argparse.Namespace, composite_df: pd.DataFrame, paths: List[Path]) -> pd.DataFrame:
    return pd.DataFrame([
        {"key": "n_input_paths", "value": len(paths)},
        {"key": "n_composite_rows", "value": int(len(composite_df))},
        {"key": "n_subjects", "value": int(composite_df["subject"].nunique()) if "subject" in composite_df.columns and not composite_df.empty else 0},
        {"key": "a4_network_table", "value": str(args.a4_network_table) if args.a4_network_table else "default"},
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
    composite_df = build_composite(data)
    if not composite_df.empty:
        feature_cols = [c for c in composite_df.columns if "_pair_similarity" in c]
        comp_with_score, loadings = fit_pca(composite_df, feature_cols)
        if not comp_with_score.empty:
            composite_df = composite_df.merge(
                comp_with_score[["subject", "condition_item_id", "condition", "composite_score_pca1", "composite_score_pca2", "composite_score_mean"]],
                on=["subject", "condition_item_id", "condition"],
                how="left",
            )
        else:
            loadings = pd.DataFrame([{"status": "pca_failed"}])
    else:
        loadings = pd.DataFrame([{"status": "empty_composite"}])
    composite_models = fit_composite_models(composite_df)
    a4_table_path = Path(args.a4_network_table or cfg.base_dir / DEFAULT_A4_NETWORK_TABLE)
    if a4_table_path.exists():
        try:
            a4_network = read_table(a4_table_path)
        except Exception as exc:
            warnings.warn(f"Failed to read a4 network table: {exc}")
            a4_network = pd.DataFrame()
    else:
        a4_network = pd.DataFrame()
    comparison = composite_vs_network_comparison(composite_models, a4_network)
    mani = build_manifest(args, composite_df, paths)
    write_outputs(
        cfg,
        MODULE,
        {
            "composite_models.tsv": composite_models if not composite_models.empty else pd.DataFrame(columns=["outcome", "predictor", "term", "status"]),
            "composite_vs_network_comparison.tsv": comparison if not comparison.empty else pd.DataFrame(columns=["outcome", "network", "delta_aic", "status"]),
            "composite_loadings.tsv": loadings if not loadings.empty else pd.DataFrame(columns=["component", "feature", "loading"]),
            "composite_item_table.tsv": composite_df if not composite_df.empty else pd.DataFrame(columns=["subject", "condition_item_id", "condition"]),
            "manifest.tsv": mani,
        },
    )


if __name__ == "__main__":
    main()
