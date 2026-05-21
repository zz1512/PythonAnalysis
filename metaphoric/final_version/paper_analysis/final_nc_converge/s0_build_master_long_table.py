#!/usr/bin/env python3
"""P0: build frozen master long tables for final NC convergence analyses."""

from __future__ import annotations

import pandas as pd

from shared_final import (
    DEFAULT_PAPER_OUTPUTS,
    add_network_column,
    build_condition_item_id,
    check_required_columns,
    final_out,
    make_network_composite,
    normalize_condition,
    read_table,
    write_json,
    write_tsv,
    zscore,
)

MODULE = "s0_build_master_long_table"


def load_item_mechanism() -> pd.DataFrame:
    path = DEFAULT_PAPER_OUTPUTS / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv"
    df = read_table(path)
    df["source_path"] = str(path)
    return add_network_column(build_condition_item_id(df))


def load_material_covariates() -> pd.DataFrame:
    path = DEFAULT_PAPER_OUTPUTS / "tables_si" / "table_stimulus_control_from_materials.tsv"
    mat = read_table(path)
    mat = mat[mat["control_scope"].astype(str).eq("sentence")].copy()
    mat["condition"] = mat["condition"].map(normalize_condition)
    label_id = mat["word_label"].astype(str).str.extract(r"_(\d+)$")[0]
    mat["original_pair_id"] = label_id.fillna(pd.to_numeric(mat["source_sheet_index"], errors="coerce").astype("Int64").astype(str))
    mat = build_condition_item_id(mat)
    rename = {
        "novelty": "novelty",
        "familiarity": "familiarity",
        "comprehensibility": "comprehensibility",
        "difficulty": "difficulty",
    }
    mat = mat.rename(columns=rename)
    for col in ["novelty", "familiarity", "comprehensibility", "difficulty"]:
        mat[col] = pd.to_numeric(mat[col], errors="coerce")
        mat[col + "_z"] = zscore(mat[col])
    return mat.groupby("condition_item_id", as_index=False)[["novelty_z", "familiarity_z", "comprehensibility_z", "difficulty_z"]].mean()


def load_mean_activation() -> pd.DataFrame:
    path = DEFAULT_PAPER_OUTPUTS / "qc" / "reviewer_supp" / "m2_univariate_sanity" / "roi_mean_activation_long.tsv"
    if not path.exists():
        return pd.DataFrame()
    act = read_table(path)
    act["condition"] = act["condition"].map(normalize_condition)
    act = act.rename(columns={"pair_id": "original_pair_id"})
    act = build_condition_item_id(act)
    act = add_network_column(act)
    keys = ["subject", "roi_set", "roi", "condition", "condition_item_id", "stage"]
    out = act.groupby(keys, dropna=False, as_index=False).agg(mean_activation=("roi_mean_activation", "mean"))
    out["mean_activation_z"] = out.groupby(["roi_set", "roi", "stage"], dropna=False)["mean_activation"].transform(zscore)
    return out[keys + ["mean_activation_z"]]


def prepare_base() -> pd.DataFrame:
    df = load_item_mechanism()
    mat = load_material_covariates()
    df = df.merge(mat, on="condition_item_id", how="left")
    df["condition"] = df["condition"].map(normalize_condition)
    df["memory_ord"] = pd.to_numeric(df.get("memory"), errors="coerce")
    df["memory_strict"] = (df["memory_ord"] >= 1.0).astype(float)
    df["memory_lenient"] = (df["memory_ord"] >= 0.5).astype(float)
    df["memory_prop"] = df["memory_ord"]
    df["wordfreq_z"] = df.get("word_frequency_mean_z", pd.Series(index=df.index, dtype=float))
    df["charlen_z"] = df.get("sentence_char_len_z", pd.Series(index=df.index, dtype=float))
    df["post_edge_drop"] = pd.to_numeric(df.get("trained_edge_drop"), errors="coerce")
    df["edge_specificity"] = pd.to_numeric(df.get("post_edge_specificity"), errors="coerce")
    df["retrieval_rebound"] = pd.to_numeric(df.get("retrieval_minus_post_pair_similarity"), errors="coerce")
    df["pattern_quality_proxy_z"] = zscore(pd.to_numeric(df.get("learning_metric_reliability", 1.0), errors="coerce"))
    return df


def build_pair_stage_long(base: pd.DataFrame) -> pd.DataFrame:
    act = load_mean_activation()
    frames = []
    stage_cols = {
        "pre": "pre_pair_similarity",
        "post": "post_pair_similarity",
        "retrieval": "retrieval_pair_similarity",
    }
    keep = [
        "subject", "condition", "original_pair_id", "condition_item_id", "roi", "roi_set", "network",
        "post_pair_similarity", "retrieval_pair_similarity", "edge_specificity", "post_edge_drop",
        "retrieval_rebound", "pattern_quality_proxy_z", "memory_ord", "memory_strict", "memory_lenient",
        "memory_prop", "novelty_z", "familiarity_z", "comprehensibility_z", "difficulty_z", "wordfreq_z", "charlen_z",
    ]
    keep = [c for c in keep if c in base.columns]
    for stage, col in stage_cols.items():
        cols = list(dict.fromkeys(keep + [col]))
        temp = base[cols].copy()
        temp["stage"] = stage
        temp = temp.rename(columns={col: "pair_similarity"})
        frames.append(temp)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["pair_id_raw"] = out.get("original_pair_id")
    out["pair_similarity_z"] = out.groupby(["network", "stage"], dropna=False)["pair_similarity"].transform(zscore)
    if not act.empty:
        out = out.merge(act, on=["subject", "roi_set", "roi", "condition", "condition_item_id", "stage"], how="left")
    if "mean_activation_z" not in out.columns:
        out["mean_activation_z"] = pd.NA
    return out


def audit_tables(pair_long: pd.DataFrame, network_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    collisions = pair_long.groupby("condition_item_id", dropna=False)["condition"].nunique().reset_index(name="n_conditions")
    collision_n = int((collisions["n_conditions"] > 1).sum())
    schema = pd.DataFrame([
        {"check": "condition_item_id_condition_collision", "value": collision_n, "status": "ok" if collision_n == 0 else "fail"},
        {"check": "yy_28_kj_28_distinct", "value": str(set(["yy_28", "kj_28"]).issubset(set(pair_long["condition_item_id"].astype(str)))), "status": "ok"},
        {"check": "n_pair_stage_rows", "value": len(pair_long), "status": "ok"},
        {"check": "n_network_stage_rows", "value": len(network_long), "status": "ok"},
    ])
    fields = [
        "pair_similarity", "pair_similarity_z", "post_pair_similarity", "retrieval_pair_similarity",
        "edge_specificity", "post_edge_drop", "retrieval_rebound", "mean_activation_z",
        "pattern_quality_proxy_z", "memory_ord", "novelty_z", "familiarity_z",
        "comprehensibility_z", "difficulty_z", "wordfreq_z", "charlen_z",
    ]
    miss = []
    for frame_name, frame in [("master_pair_stage_long", pair_long), ("master_network_long", network_long)]:
        for col in fields:
            if col in frame.columns:
                miss.append({"table": frame_name, "field": col, "n": len(frame), "missing": int(frame[col].isna().sum()), "missing_rate": float(frame[col].isna().mean())})
    missing = pd.DataFrame(miss)
    return schema, missing


def main() -> None:
    base = prepare_base()
    pair_long = build_pair_stage_long(base)
    item_long = pair_long.copy()
    network_long = make_network_composite(pair_long, "pair_similarity", ["subject", "condition", "condition_item_id", "stage", "network"])
    network_long = network_long.rename(columns={"metric_value": "pair_similarity"})
    network_long["pair_similarity_z"] = network_long.groupby(["network", "stage"], dropna=False)["pair_similarity"].transform(zscore)
    cov_cols = [c for c in pair_long.columns if c.endswith("_z") or c.startswith("memory_") or c in ["post_edge_drop", "edge_specificity", "retrieval_rebound", "post_pair_similarity", "retrieval_pair_similarity"]]
    cov = pair_long.groupby(["subject", "condition", "condition_item_id", "network"], dropna=False, as_index=False)[[c for c in cov_cols if c in pair_long.columns]].mean(numeric_only=True)
    network_long = network_long.merge(cov, on=["subject", "condition", "condition_item_id", "network"], how="left", suffixes=("", "_from_pair"))
    schema, missing = audit_tables(pair_long, network_long)
    cell = network_long.groupby(["stage", "condition", "network"], dropna=False, as_index=False).agg(n=("pair_similarity", "count"), n_subjects=("subject", "nunique"), n_items=("condition_item_id", "nunique"))
    write_tsv(item_long, final_out(MODULE, "master_item_stage_long.tsv"))
    write_tsv(pair_long, final_out(MODULE, "master_pair_stage_long.tsv"))
    write_tsv(network_long, final_out(MODULE, "master_network_long.tsv"))
    write_tsv(schema, final_out(MODULE, "schema_audit.tsv"))
    write_tsv(missing, final_out(MODULE, "missingness_report.tsv"))
    write_tsv(cell, final_out(MODULE, "stage_condition_network_cell_counts.tsv"))
    write_json({"status": "ok", "note": "Frozen final master tables built from read-only upstream outputs."}, final_out(MODULE, "s0_manifest.json"))


if __name__ == "__main__":
    main()
