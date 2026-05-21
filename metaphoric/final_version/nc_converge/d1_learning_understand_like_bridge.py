#!/usr/bin/env python3
"""D1: 学习阶段 understand / like 行为 → post-stage 表征 / retrieval 记忆 桥模型.

Goal
----
回答："run3 (理解判断) 与 run4 (喜好判断) 是否与 run5/run6 的 post-stage representational
separation、以及 run7 的 retrieval memory 存在稳定相关？该相关是否具有 condition specific
特征 (YY vs KJ)？是否经由 post-stage representational separation 中介到 memory？"

设计与单位策略均沿用 §28.4 修复后的口径：

* outcome 用 raw 网络均值后做单次全局 z (避免 z-of-z 镜像)；
* understand_yes / like_yes 是二元 0/1 直接进；
* RT 已在上游 within subject×condition z (run3/run4_rt_z_subject_condition)，原样使用；
* memory_strict 是二元 (remembered_strict=1) 进 binomial GLMM，不再 z；
* fluency 用 learning_fluency_shift 的全局 z 作为 RT-side 控制；
* covariates 复用 sentence_char_len_z / word_frequency_mean_z / valence_mean_z /
  arousal_mean_z / pre_pair_similarity_z (raw → 全局 z)。

四组模型 (`fit_models`):

A. understand/like → post-stage representational separation::

    post_edge_differentiation_z ~
        C(condition, Treatment('kj')) * (run3_understand_yes + run4_like_yes)
      + run3_rt_z_subject_condition + run4_rt_z_subject_condition
      + learning_fluency_shift_z + pre_pair_similarity_z + covariates
      + (1|subject) + (1|condition_item_id)

   每个 network 各跑一次。confirmatory term:
   ``C(condition)[T.yy]:run3_understand_yes`` / ``...:run4_like_yes``。

B. understand/like → post pair similarity (输入-输出函数交叉验证)::

    post_pair_similarity_z ~ pre_pair_similarity_z
      + C(condition, Treatment('kj')) * (run3_understand_yes + run4_like_yes)
      + run3_rt_z_subject_condition + run4_rt_z_subject_condition
      + learning_fluency_shift_z + covariates
      + (1|subject) + (1|condition_item_id)

C. understand/like → memory (retrieval 行为终点，binomial)，跑两版做"条件中介"读取::

    C-direct: remembered_strict ~
        C(condition) * (run3_understand_yes + run4_like_yes)
      + run3_rt_z_subject_condition + run4_rt_z_subject_condition
      + learning_fluency_shift_z + pre_pair_similarity_z + covariates
      + (1|subject) + (1|condition_item_id)

    C-with-neural: 在 C-direct 基础上加入 post_edge_differentiation_z 作为 mediator。
    比较两版中 understand/like 的系数变化即可读出"是否经过 post-separation 中介"。

D. learning_response_profile (understand_yes × like_yes 4 cell) 描述性::
    每 network × condition × profile 报 cell mean post_edge_differentiation_raw / z
    与 memory_prop。

输出沙箱: ``paper_outputs/qc/nc_converge/d1_learning_understand_like_bridge/``。
不修改任何已有脚本或既有输出。
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

MODULE = "d1_learning_understand_like_bridge"
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_LEARNING_BEHAVIOR = Path("qc/learning_reactivation_mapping/learning_behavior_item.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
RT_COVARS = [
    "run3_rt_z_subject_condition",
    "run4_rt_z_subject_condition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--learning-behavior",
        type=Path,
        default=None,
        help="learning_behavior_item.tsv with run3/run4 understand, like and RT columns.",
    )
    return parser.parse_args()


def usable_columns(frame: pd.DataFrame, columns) -> list[str]:
    out: list[str] = []
    for col in columns:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10:
            out.append(col)
    return out


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Coerce bool/string/numeric response columns to 0/1/NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "t": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "1": 1.0,
            "1.0": 1.0,
            "false": 0.0,
            "f": 0.0,
            "no": 0.0,
            "n": 0.0,
            "0": 0.0,
            "0.0": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.where(mapped.notna(), numeric)


def merge_learning_behavior(
    frame: pd.DataFrame,
    behavior_path: Path,
    *,
    allow_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach run3/run4 behavior source columns by subject + condition_item_id.

    The item mechanism table does not contain these columns in the current
    source tree, so D1 must merge the source behavior table explicitly.
    """
    audit: dict[str, object] = {
        "learning_behavior_path": str(behavior_path),
        "learning_behavior_exists": bool(Path(behavior_path).exists()),
        "learning_behavior_status": "not_loaded",
    }
    if not Path(behavior_path).exists():
        if allow_missing:
            audit["learning_behavior_status"] = "missing_allowed"
            return frame, pd.DataFrame([audit])
        raise FileNotFoundError(f"learning_behavior_item.tsv not found: {behavior_path}")

    out = build_condition_item_id(frame)
    behavior = build_condition_item_id(read_table(behavior_path))
    keys = [c for c in ["subject", "condition_item_id"] if c in out.columns and c in behavior.columns]
    if len(keys) < 2:
        audit["learning_behavior_status"] = f"missing_merge_keys: {keys}"
        if allow_missing:
            return out, pd.DataFrame([audit])
        raise ValueError(f"Cannot merge learning behavior; available keys={keys}")

    behavior_cols = [
        "run3_understand_yes",
        "run4_like_yes",
        "run3_rt_z_subject_condition",
        "run4_rt_z_subject_condition",
        "learning_fluency_shift",
        "learning_response_profile",
    ]
    keep = [*keys, *[c for c in behavior_cols if c in behavior.columns]]
    behavior = behavior[keep].copy()
    for col in ["run3_understand_yes", "run4_like_yes"]:
        if col in behavior.columns:
            behavior[col] = _coerce_binary(behavior[col])
    for col in ["run3_rt_z_subject_condition", "run4_rt_z_subject_condition", "learning_fluency_shift"]:
        if col in behavior.columns:
            behavior[col] = pd.to_numeric(behavior[col], errors="coerce")

    numeric_cols = [c for c in behavior.columns if c not in keys and c != "learning_response_profile"]
    text_cols = [c for c in behavior.columns if c not in keys and c == "learning_response_profile"]
    agg_spec = {c: "mean" for c in numeric_cols}
    agg_spec.update({c: "first" for c in text_cols})
    behavior_unique = behavior.groupby(keys, dropna=False, as_index=False).agg(agg_spec)

    # Prefer source behavior columns if stale placeholder columns are present upstream.
    drop_existing = [c for c in behavior_cols if c in out.columns]
    if drop_existing:
        out = out.drop(columns=drop_existing)
    merged = out.merge(behavior_unique, on=keys, how="left", validate="many_to_one")
    audit.update(
        {
            "learning_behavior_status": "ok",
            "learning_behavior_rows": int(len(behavior)),
            "learning_behavior_unique_rows": int(len(behavior_unique)),
            "merge_keys": ",".join(keys),
            "merged_rows": int(len(merged)),
        }
    )
    for col in behavior_cols:
        if col in merged.columns:
            audit[f"{col}_n_nonnull_after_merge"] = int(merged[col].notna().sum())
    return merged, pd.DataFrame([audit])


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Build subject × network × condition_item_id table with raw + global z columns."""
    out = add_network_column(build_condition_item_id(frame))
    metric_col = (
        "post_edge_specificity"
        if "post_edge_specificity" in out.columns
        else ("trained_edge_drop" if "trained_edge_drop" in out.columns else None)
    )
    raw_metrics = [c for c in ["pre_pair_similarity", "post_pair_similarity"] if c in out.columns]
    if metric_col is None or not raw_metrics:
        return pd.DataFrame()

    numeric_cols = list(
        dict.fromkeys(
            [
                *raw_metrics,
                metric_col,
                *COVARIATES,
                *RT_COVARS,
                "learning_fluency_shift",
                "run3_understand_yes",
                "run4_like_yes",
                "memory",
                "memory_score",
                "memory_successes",
                "remembered_strict",
            ]
        )
    )
    for col in numeric_cols:
        if col in out.columns:
            if col in {"run3_understand_yes", "run4_like_yes"}:
                out[col] = _coerce_binary(out[col])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")

    if "network" not in out.columns or "condition_item_id" not in out.columns:
        return pd.DataFrame()
    out = out[out["network"].notna() & out["condition_item_id"].astype(str).ne("")].copy()

    id_keys = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in out.columns]
    agg_cols = [c for c in numeric_cols if c in out.columns]
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
    if "learning_fluency_shift" in network_raw.columns:
        network_raw["learning_fluency_shift_z"] = zscore(network_raw["learning_fluency_shift"])

    # learning_response_profile = understand × like 4-cell label (item averaged ⇒ 0/0.5/1).
    if {"run3_understand_yes", "run4_like_yes"}.issubset(network_raw.columns):
        network_raw["learning_response_profile"] = [
            _profile_label(u, l) for u, l in zip(
                network_raw["run3_understand_yes"], network_raw["run4_like_yes"]
            )
        ]

    # Binary memory outcome aligned with stage4_subsequent_memory_rework.
    if "remembered_strict" in network_raw.columns:
        network_raw["memory_strict"] = pd.to_numeric(
            network_raw["remembered_strict"], errors="coerce"
        )
    elif "memory_score" in network_raw.columns:
        ms = pd.to_numeric(network_raw["memory_score"], errors="coerce")
        network_raw["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    elif "memory" in network_raw.columns:
        ms = pd.to_numeric(network_raw["memory"], errors="coerce")
        network_raw["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    elif "memory_successes" in network_raw.columns:
        ms = pd.to_numeric(network_raw["memory_successes"], errors="coerce")
        network_raw["memory_strict"] = np.where(ms.ge(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
    return network_raw


def _profile_label(understand: float, like: float) -> str:
    if not (np.isfinite(understand) and np.isfinite(like)):
        return ""
    u = "understand" if understand >= 0.5 else "no_understand"
    l = "like" if like >= 0.5 else "no_like"
    return f"{u}+{l}"


def _format_covariates(frame: pd.DataFrame) -> tuple[str, list[str]]:
    """Return ``+ cov1 + cov2 ...`` along with retained list."""
    covs = usable_columns(frame, COVARIATES + RT_COVARS + ["learning_fluency_shift_z"])
    if not covs:
        return "", []
    return " + " + " + ".join(covs), covs


def _post_separation_models(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    needed = {
        "subject",
        "condition_item_id",
        "condition",
        "network",
        "post_edge_differentiation_z",
        "run3_understand_yes",
        "run4_like_yes",
    }
    if not needed.issubset(data.columns):
        return pd.DataFrame([
            {
                "term": "__model__",
                "status": f"missing_columns: {sorted(needed - set(data.columns))}",
            }
        ])
    for network, sub in data.groupby("network", dropna=False):
        cov_expr, covs = _format_covariates(sub)
        keep = [
            "post_edge_differentiation_z",
            "run3_understand_yes",
            "run4_like_yes",
            "pre_pair_similarity_z",
            *covs,
        ]
        keep = [c for c in keep if c in sub.columns]
        model_data = sub.dropna(subset=keep).copy()
        if len(model_data) < 30:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "network": network,
                            "outcome": "post_edge_differentiation_z",
                            "mechanism_model": "understand_like_to_post_separation",
                            "term": "__model__",
                            "status": "too_few_rows",
                            "n_obs": len(model_data),
                        }
                    ]
                )
            )
            continue
        formula = (
            "post_edge_differentiation_z ~ C(condition, Treatment('kj')) "
            "* (run3_understand_yes + run4_like_yes) + pre_pair_similarity_z"
            + cov_expr
        )
        res = fit_formula(model_data, formula, family="gaussian")
        res.insert(0, "network", network)
        res.insert(1, "outcome", "post_edge_differentiation_z")
        res.insert(2, "mechanism_model", "understand_like_to_post_separation")
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def _post_similarity_models(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    needed = {
        "subject",
        "condition_item_id",
        "condition",
        "network",
        "post_pair_similarity_z",
        "pre_pair_similarity_z",
        "run3_understand_yes",
        "run4_like_yes",
    }
    if not needed.issubset(data.columns):
        return pd.DataFrame([
            {
                "term": "__model__",
                "status": f"missing_columns: {sorted(needed - set(data.columns))}",
            }
        ])
    for network, sub in data.groupby("network", dropna=False):
        cov_expr, covs = _format_covariates(sub)
        keep = [
            "post_pair_similarity_z",
            "pre_pair_similarity_z",
            "run3_understand_yes",
            "run4_like_yes",
            *covs,
        ]
        keep = [c for c in keep if c in sub.columns]
        model_data = sub.dropna(subset=keep).copy()
        if len(model_data) < 30:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "network": network,
                            "outcome": "post_pair_similarity_z",
                            "mechanism_model": "understand_like_to_post_pair_similarity",
                            "term": "__model__",
                            "status": "too_few_rows",
                            "n_obs": len(model_data),
                        }
                    ]
                )
            )
            continue
        formula = (
            "post_pair_similarity_z ~ pre_pair_similarity_z "
            "+ C(condition, Treatment('kj')) * (run3_understand_yes + run4_like_yes)"
            + cov_expr
        )
        res = fit_formula(model_data, formula, family="gaussian")
        res.insert(0, "network", network)
        res.insert(1, "outcome", "post_pair_similarity_z")
        res.insert(2, "mechanism_model", "understand_like_to_post_pair_similarity")
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def _memory_models(data: pd.DataFrame, *, with_neural: bool) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    model_label = "understand_like_to_memory_with_neural" if with_neural else "understand_like_to_memory_direct"
    needed = {
        "subject",
        "condition_item_id",
        "condition",
        "network",
        "memory_strict",
        "run3_understand_yes",
        "run4_like_yes",
    }
    if with_neural:
        needed = needed | {"post_edge_differentiation_z"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([
            {
                "term": "__model__",
                "mechanism_model": model_label,
                "status": f"missing_columns: {sorted(needed - set(data.columns))}",
            }
        ])
    for network, sub in data.groupby("network", dropna=False):
        cov_expr, covs = _format_covariates(sub)
        keep = [
            "memory_strict",
            "run3_understand_yes",
            "run4_like_yes",
            "pre_pair_similarity_z",
            *covs,
        ]
        if with_neural:
            keep.append("post_edge_differentiation_z")
        keep = [c for c in keep if c in sub.columns]
        model_data = sub.dropna(subset=keep).copy()
        # binomial requires 0/1 outcome with both classes present.
        if len(model_data) < 30:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "network": network,
                            "outcome": "memory_strict",
                            "mechanism_model": model_label,
                            "term": "__model__",
                            "status": "too_few_rows",
                            "n_obs": len(model_data),
                        }
                    ]
                )
            )
            continue
        if model_data["memory_strict"].nunique(dropna=True) < 2:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "network": network,
                            "outcome": "memory_strict",
                            "mechanism_model": model_label,
                            "term": "__model__",
                            "status": "single_class_outcome",
                            "n_obs": len(model_data),
                        }
                    ]
                )
            )
            continue
        formula = (
            "memory_strict ~ C(condition, Treatment('kj')) "
            "* (run3_understand_yes + run4_like_yes) + pre_pair_similarity_z"
            + cov_expr
        )
        if with_neural:
            formula += " + post_edge_differentiation_z"
        res = fit_formula(model_data, formula, family="binomial")
        res.insert(0, "network", network)
        res.insert(1, "outcome", "memory_strict")
        res.insert(2, "mechanism_model", model_label)
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def fit_models(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "understand_like_to_post_separation.tsv": _post_separation_models(data),
        "understand_like_to_post_pair_similarity.tsv": _post_similarity_models(data),
        "understand_like_to_memory_direct.tsv": _memory_models(data, with_neural=False),
        "understand_like_to_memory_with_neural.tsv": _memory_models(data, with_neural=True),
    }


def descriptives(data: pd.DataFrame) -> pd.DataFrame:
    needed = {"network", "condition", "learning_response_profile"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    cols = {
        "n": ("post_edge_differentiation_raw", "size"),
    }
    if "subject" in data.columns:
        cols["n_subjects"] = ("subject", "nunique")
    if "condition_item_id" in data.columns:
        cols["n_items"] = ("condition_item_id", "nunique")
    if "post_edge_differentiation_raw" in data.columns:
        cols["post_edge_mean_raw"] = ("post_edge_differentiation_raw", "mean")
        cols["post_edge_sd_raw"] = ("post_edge_differentiation_raw", "std")
    if "post_edge_differentiation_z" in data.columns:
        cols["post_edge_mean_z"] = ("post_edge_differentiation_z", "mean")
    if "post_pair_similarity_network_raw" in data.columns:
        cols["post_similarity_mean_raw"] = ("post_pair_similarity_network_raw", "mean")
    if "memory_strict" in data.columns:
        cols["memory_prop"] = ("memory_strict", "mean")
    if "learning_fluency_shift" in data.columns:
        cols["fluency_shift_mean"] = ("learning_fluency_shift", "mean")
    grouped = data.groupby(
        ["network", "condition", "learning_response_profile"], dropna=False, as_index=False
    ).agg(**cols)
    return grouped


def manifest(path: Path, behavior_audit: pd.DataFrame, raw: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, object] = {
        "input_path": str(path),
        "input_rows": len(raw),
        "prepared_rows": len(prepared),
        "n_subjects": int(prepared["subject"].nunique()) if "subject" in prepared.columns else None,
        "n_items": int(prepared["condition_item_id"].nunique()) if "condition_item_id" in prepared.columns else None,
        "n_networks": int(prepared["network"].nunique()) if "network" in prepared.columns else None,
    }
    for col in [
        "run3_understand_yes",
        "run4_like_yes",
        "memory_strict",
        "post_edge_differentiation_raw",
        "pre_pair_similarity_network_raw",
        "post_pair_similarity_network_raw",
        "learning_fluency_shift",
        "run3_rt_z_subject_condition",
        "run4_rt_z_subject_condition",
    ]:
        if col in prepared.columns:
            counts[f"{col}_n_nonnull"] = int(prepared[col].notna().sum())
    out = pd.DataFrame([counts])
    if not behavior_audit.empty:
        for col, value in behavior_audit.iloc[0].items():
            out[col] = value
    return out


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    behavior_path = Path(args.learning_behavior or cfg.paper_output_root / DEFAULT_LEARNING_BEHAVIOR)
    raw = read_table(path)
    raw, behavior_audit = merge_learning_behavior(raw, behavior_path, allow_missing=args.allow_empty)
    data = prepare(raw)
    desc = descriptives(data)
    models = fit_models(data)
    write_outputs(
        cfg,
        MODULE,
        {
            **models,
            "learning_response_profile_descriptives.tsv": desc,
            "manifest.tsv": manifest(path, behavior_audit, raw, data),
        },
    )


if __name__ == "__main__":
    main()
