#!/usr/bin/env python3
"""A4: separate post/retrieval memory components and stage trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import add_common_args, build_condition_item_id, default_config, discover_tables, first_existing_column, fit_formula, make_network_composite, q_for_ok_rows, read_table, write_outputs, zscore

MODULE = "a4_post_memory_component"
PATTERNS = ["**/*item_mechanism*.tsv", "**/*memory*bridge*.tsv", "**/stage45*/*.tsv", "**/post_to_retrieval*.tsv"]
COVARIATE_CANDIDATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def load(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = read_table(path)
        except Exception:
            continue
        frame["source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    out = build_condition_item_id(raw)
    for target, choices in {
        "pre_pair_similarity": ["pre_pair_similarity", "pre_similarity", "pre_sim"],
        "post_pair_similarity": ["post_pair_similarity", "post_similarity", "post_sim", "post_edge_specificity"],
        "retrieval_pair_similarity": ["retrieval_pair_similarity", "retrieval_similarity", "retrieval_sim", "retrieval_rebinding"],
        "memory_prop": ["memory_prop", "memory", "accuracy", "remembered", "memory_score"],
    }.items():
        col = first_existing_column(out, choices)
        if col and col != target:
            out = out.rename(columns={col: target})
    if "network" not in out.columns:
        from shared_nc import add_network_column
        out = add_network_column(out)
    for metric in ["pre_pair_similarity", "post_pair_similarity", "retrieval_pair_similarity"]:
        if metric not in out.columns:
            continue
        comp = make_network_composite(out, value_col=metric, metric_name=metric)
        if comp.empty:
            continue
        key_cols = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in comp.columns]
        comp = comp.rename(columns={"metric_value": f"{metric}_network"})
        out = out.merge(comp[key_cols + [f"{metric}_network"]].drop_duplicates(), on=key_cols, how="left")
    for col in [
        "pre_pair_similarity",
        "post_pair_similarity",
        "retrieval_pair_similarity",
        "pre_pair_similarity_network",
        "post_pair_similarity_network",
        "retrieval_pair_similarity_network",
        "memory_prop",
        *COVARIATE_CANDIDATES,
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in [
        "pre_pair_similarity",
        "post_pair_similarity",
        "retrieval_pair_similarity",
        "pre_pair_similarity_network",
        "post_pair_similarity_network",
        "retrieval_pair_similarity_network",
    ]:
        if col in out.columns:
            out[f"{col}_z"] = out.groupby("network", dropna=False)[col].transform(zscore) if "network" in out.columns else zscore(out[col])
    if "memory_successes" not in out.columns and "memory_prop" in out.columns:
        out["memory_successes"] = np.rint(out["memory_prop"].clip(0, 1) * 2).astype("Int64")
    if "memory_prop" not in out.columns and "memory_successes" in out.columns:
        out["memory_prop"] = pd.to_numeric(out["memory_successes"], errors="coerce") / 2.0
    if "memory_prop" in out.columns:
        out["memory_strict"] = (out["memory_prop"] >= 1.0).astype(float)
        out["memory_lenient"] = (out["memory_prop"] >= 0.5).astype(float)
    return out


def usable_covariates(data: pd.DataFrame) -> list[str]:
    terms = []
    for col in COVARIATE_CANDIDATES:
        if col not in data.columns:
            continue
        values = pd.to_numeric(data[col], errors="coerce")
        if values.notna().sum() < 10 or values.nunique(dropna=True) < 2:
            continue
        terms.append(col)
    return terms


def formula_with_covariates(base: str, covariates: list[str]) -> str:
    if not covariates:
        return base
    return base + " + " + " + ".join(covariates)


def fit_memory_component(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    post_col = "post_pair_similarity_network_z" if "post_pair_similarity_network_z" in data.columns else "post_pair_similarity_z"
    retrieval_col = "retrieval_pair_similarity_network_z" if "retrieval_pair_similarity_network_z" in data.columns else "retrieval_pair_similarity_z"
    covariates = usable_covariates(data)
    outcome_specs = [
        ("memory_strict", "binomial"),
        ("memory_lenient", "binomial"),
        ("memory_prop", "gaussian"),
    ]
    for outcome, family in outcome_specs:
        needed = {outcome, "condition", post_col, retrieval_col, "subject", "condition_item_id", "network"}
        if not needed.issubset(data.columns):
            rows.append(pd.DataFrame([{"outcome": outcome, "term": "__model__", "status": f"missing_columns: {sorted(needed - set(data.columns))}"}]))
            continue
        for network, sub in data.dropna(subset=[outcome, post_col, retrieval_col]).groupby("network", dropna=False):
            if len(sub) < 10:
                rows.append(pd.DataFrame([{"network": network, "outcome": outcome, "term": "__model__", "status": "too_few_rows", "n_obs": len(sub)}]))
                continue
            covs = usable_covariates(sub[covariates]) if covariates else []
            formula = formula_with_covariates(f"{outcome} ~ condition * {post_col} + condition * {retrieval_col}", covs)
            model_data = sub.dropna(subset=[outcome, post_col, retrieval_col, *covs])
            res = fit_formula(model_data, formula, family=family)
            res.insert(0, "network", network)
            res.insert(1, "outcome", outcome)
            res.insert(2, "component_model", "memory_post_vs_retrieval")
            rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def roi_followup(data: pd.DataFrame, network_table: pd.DataFrame) -> pd.DataFrame:
    if "roi" not in data.columns or network_table.empty:
        return pd.DataFrame()
    sig = network_table[(network_table.get("q_bh", pd.Series(dtype=float)) < 0.05) & network_table.get("status", pd.Series(dtype=str)).astype(str).eq("ok")]
    if sig.empty:
        return pd.DataFrame([{"status": "not_run_network_not_significant"}])
    sig_networks = set(sig["network"].dropna().astype(str)) if "network" in sig.columns else set()
    if not sig_networks:
        return pd.DataFrame([{"status": "not_run_network_not_identified"}])
    rows = []
    covariates = usable_covariates(data)
    for (network, roi), sub in data.groupby(["network", "roi"], dropna=False):
        if str(network) not in sig_networks:
            continue
        if len(sub) < 10 or "post_pair_similarity_z" not in sub.columns:
            continue
        if "retrieval_pair_similarity_z" not in sub.columns:
            rows.append(pd.DataFrame([{"network": network, "roi": roi, "term": "__model__", "status": "missing_retrieval_pair_similarity_z"}]))
            continue
        covs = usable_covariates(sub[covariates]) if covariates else []
        formula = formula_with_covariates(
            "memory_strict ~ condition * post_pair_similarity_z + condition * retrieval_pair_similarity_z",
            covs,
        )
        res = fit_formula(
            sub.dropna(subset=["memory_strict", "post_pair_similarity_z", "retrieval_pair_similarity_z", *covs]),
            formula,
            family="binomial",
        )
        res.insert(0, "network", network)
        res.insert(1, "roi", roi)
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame([{"status": "no_roi_followup_fit"}])


def stage_similarity_long(data: pd.DataFrame) -> pd.DataFrame:
    covariates = usable_covariates(data)
    id_cols = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in data.columns]
    if not {"subject", "condition_item_id", "condition"}.issubset(id_cols):
        return pd.DataFrame([{"status": f"missing_columns: {sorted({'subject','condition_item_id','condition'} - set(data.columns))}"}])

    # 关键修正：必须使用 raw（未 z-score）的同尺度 stage 列，避免逐 stage 独立 z
    # 旧实现按 _z 列 stack 会让每个 stage 内 mean 强制为 0，造成 KJ/YY 严格镜像，
    # 也使 post-pre 的差混合不同尺度。注意：`*_network` 列也是经 make_network_composite
    # 内部 z 后的 ROI 平均（仍然是 z 单位），所以必须优先选 raw 的原始相似性列。
    stage_cols = [
        ("pre", first_existing_column(data, [
            "pre_pair_similarity", "pre_pair_similarity_network",
        ])),
        ("post", first_existing_column(data, [
            "post_pair_similarity", "post_pair_similarity_network",
        ])),
        ("retrieval", first_existing_column(data, [
            "retrieval_pair_similarity", "retrieval_pair_similarity_network",
        ])),
    ]
    frames = []
    for stage, col in stage_cols:
        if col is None or col not in data.columns:
            continue
        keep = id_cols + covariates + [col]
        temp = data[keep].copy()
        temp = temp.rename(columns={col: "pair_similarity"})
        temp["stage"] = stage
        temp["source_column"] = col
        temp["pair_similarity"] = pd.to_numeric(temp["pair_similarity"], errors="coerce")
        group_cols = id_cols + covariates + ["stage", "source_column"]
        temp = temp.groupby(group_cols, dropna=False, as_index=False).agg(pair_similarity=("pair_similarity", "mean"))
        frames.append(temp)
    if not frames:
        return pd.DataFrame([{"status": "missing_stage_similarity_columns"}])
    long = pd.concat(frames, ignore_index=True, sort=False)
    # 在 long 表上做一次性全局 z（按 network 内部统一），保证三个 stage 共用一套 mean/sd
    if "network" in long.columns:
        long["pair_similarity_z"] = long.groupby("network", dropna=False)["pair_similarity"].transform(zscore)
    else:
        long["pair_similarity_z"] = zscore(long["pair_similarity"])
    return long


def stage_trajectory_descriptives(stage_long: pd.DataFrame) -> pd.DataFrame:
    needed = {"pair_similarity", "condition", "stage"}
    if not needed.issubset(stage_long.columns):
        return pd.DataFrame([{"status": f"missing_columns: {sorted(needed - set(stage_long.columns))}"}])
    group_cols = [c for c in ["network", "condition", "stage"] if c in stage_long.columns]
    return stage_long.groupby(group_cols, dropna=False, as_index=False).agg(
        n=("pair_similarity", "count"),
        mean_raw=("pair_similarity", "mean"),
        sd_raw=("pair_similarity", "std"),
        mean_z=("pair_similarity_z", "mean") if "pair_similarity_z" in stage_long.columns else ("pair_similarity", "mean"),
        n_subjects=("subject", "nunique") if "subject" in stage_long.columns else ("pair_similarity", "count"),
        n_items=("condition_item_id", "nunique") if "condition_item_id" in stage_long.columns else ("pair_similarity", "count"),
    )


def fit_stage_trajectory(stage_long: pd.DataFrame) -> pd.DataFrame:
    needed = {"pair_similarity", "condition", "stage", "subject", "condition_item_id"}
    if not needed.issubset(stage_long.columns):
        return pd.DataFrame([{"trajectory_model": "condition_by_stage", "term": "__model__", "status": f"missing_columns: {sorted(needed - set(stage_long.columns))}"}])
    rows = []
    covariates = usable_covariates(stage_long)
    groups = stage_long.groupby("network", dropna=False) if "network" in stage_long.columns else [("all", stage_long)]
    for network, sub in groups:
        covs = usable_covariates(sub[covariates]) if covariates else []
        model_data = sub.dropna(subset=["pair_similarity", "condition", "stage", *covs]).copy()
        if len(model_data) < 30 or model_data["stage"].nunique(dropna=True) < 2:
            rows.append(pd.DataFrame([{"network": network, "trajectory_model": "condition_by_stage", "term": "__model__", "status": "too_few_rows_or_stages", "n_obs": len(model_data)}]))
            continue
        formula = formula_with_covariates("pair_similarity ~ condition * C(stage)", covs)
        res = fit_formula(model_data, formula, family="gaussian")
        res.insert(0, "network", network)
        res.insert(1, "trajectory_model", "condition_by_stage")
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.paper_output_root, PATTERNS)
    raw = load(paths)
    if raw.empty and not args.allow_empty:
        raise FileNotFoundError("No post-memory candidate tables found; use --input or --allow-empty.")
    data = prepare(raw)
    memory_component = fit_memory_component(data)
    roi = roi_followup(data, memory_component)
    stage_long = stage_similarity_long(data)
    trajectory = fit_stage_trajectory(stage_long)
    descriptives = stage_trajectory_descriptives(stage_long)
    write_outputs(cfg, MODULE, {
        "memory_component_model.tsv": memory_component,
        "memory_component_roi_followup.tsv": roi,
        "stage_trajectory_model.tsv": trajectory,
        "stage_trajectory_descriptives.tsv": descriptives,
        "memory_component_input_manifest.tsv": pd.DataFrame({"path": [str(p) for p in paths]}),
    })


if __name__ == "__main__":
    main()
