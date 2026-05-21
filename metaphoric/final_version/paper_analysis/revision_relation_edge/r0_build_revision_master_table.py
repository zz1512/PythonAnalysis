#!/usr/bin/env python3
"""Task r0: build revision master tables with locked condition_item_id."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd

from shared_revision import (
    DEFAULT_BASE_DIR,
    DEFAULT_PAPER_OUTPUTS,
    EXTRA_ITEM_COVARS,
    FINAL_NC,
    MATERIAL_COVARS,
    NETWORKS,
    PROJECT_ROOT,
    ensure_condition_item_id,
    out_path,
    read_table,
    write_json,
    write_tsv,
    zscore,
)

MODULE = "r0_build_revision_master_table"


def load_final_network_long() -> pd.DataFrame:
    path = FINAL_NC / "s0_build_master_long_table" / "master_network_long.tsv"
    df = read_table(path)
    df = ensure_condition_item_id(df)
    return df[df["network"].isin(NETWORKS)].copy()


def load_final_item_long() -> pd.DataFrame:
    path = FINAL_NC / "s0_build_master_long_table" / "master_item_stage_long.tsv"
    df = read_table(path)
    df = ensure_condition_item_id(df)
    return df[df["network"].isin(NETWORKS)].copy()


def load_extra_item_covars() -> pd.DataFrame:
    path = DEFAULT_PAPER_OUTPUTS / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv"
    df = read_table(path)
    df = ensure_condition_item_id(df)
    rename = {
        "valence_mean_z": "valence_z",
        "arousal_mean_z": "arousal_z",
        "stroke_count_mean_z": "stroke_z",
        "sentence_text": "sentence_text",
        "word_a_text": "word_a_text",
        "word_b_text": "word_b_text",
    }
    cols = ["condition_item_id"] + [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename).drop_duplicates("condition_item_id")
    return out


def ensure_learning_behavior_table() -> pd.DataFrame:
    """Load source-derived run3/run4 behavior, rebuilding it from raw CSVs if needed."""
    path = DEFAULT_PAPER_OUTPUTS / "qc" / "learning_reactivation_mapping" / "learning_behavior_item.tsv"
    if not path.exists():
        script = PROJECT_ROOT / "brain_behavior" / "learning_reactivation" / "build_learning_behavior_table.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--base-dir",
                str(DEFAULT_BASE_DIR),
                "--data-events",
                str(DEFAULT_BASE_DIR / "data_events"),
                "--stimuli-template",
                str(DEFAULT_BASE_DIR / "stimuli_template.csv"),
                "--out-dir",
                str(path.parent),
            ],
            check=True,
        )
    df = read_table(path)
    df = ensure_condition_item_id(df)
    rename = {
        "run3_understand_yes": "run3_understand",
        "run4_like_yes": "run4_like",
        "run3_rt_z_subject_condition": "run3_rt_z",
        "run4_rt_z_subject_condition": "run4_rt_z",
    }
    cols = [
        "subject",
        "condition_item_id",
        "run3_rt",
        "run4_rt",
        "run3_resp",
        "run4_resp",
        "run3_acc",
        "run4_acc",
        "run3_understand_yes",
        "run4_like_yes",
        "run3_rt_z_subject_condition",
        "run4_rt_z_subject_condition",
        "learning_fluency_shift",
        "learning_response_profile",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].rename(columns=rename).drop_duplicates(["subject", "condition_item_id"])
    return out


def merge_learning_behavior(out: pd.DataFrame) -> pd.DataFrame:
    learning_cols = [
        "run3_rt",
        "run4_rt",
        "run3_resp",
        "run4_resp",
        "run3_acc",
        "run4_acc",
        "run3_understand",
        "run4_like",
        "run3_rt_z",
        "run4_rt_z",
        "learning_fluency_shift",
        "learning_response_profile",
    ]
    out = out.drop(columns=[c for c in learning_cols if c in out.columns], errors="ignore")
    learning = ensure_learning_behavior_table()
    return out.merge(learning, on=["subject", "condition_item_id"], how="left")


def pivot_stage_table(stage_long: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    base_cols = keys + ["stage", "pair_similarity", "pair_similarity_z"]
    stage = stage_long[base_cols].dropna(subset=["stage"]).copy()
    raw = stage.pivot_table(index=keys, columns="stage", values="pair_similarity", aggfunc="mean").reset_index()
    zed = stage.pivot_table(index=keys, columns="stage", values="pair_similarity_z", aggfunc="mean").reset_index()
    raw = raw.rename(columns={
        "pre": "pre_pair_similarity",
        "post": "post_pair_similarity",
        "retrieval": "retrieval_pair_similarity",
    })
    zed = zed.rename(columns={
        "pre": "pre_pair_similarity_z",
        "post": "post_pair_similarity_z",
        "retrieval": "retrieval_pair_similarity_z",
    })
    out = raw.merge(zed, on=keys, how="left")
    return out


def add_static_metrics(out: pd.DataFrame, stage_long: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    static_candidates = [
        "post_edge_drop",
        "edge_specificity",
        "retrieval_rebound",
        "memory_ord",
        "memory_strict",
        "memory_lenient",
        "memory_prop",
        "pattern_quality_proxy_z",
        *MATERIAL_COVARS,
    ]
    keep = keys + [c for c in static_candidates if c in stage_long.columns]
    static = stage_long[keep].groupby(keys, dropna=False, as_index=False).mean(numeric_only=True)
    out = out.merge(static, on=keys, how="left")
    if "post_edge_drop" in out.columns:
        out["post_separation"] = pd.to_numeric(out["post_edge_drop"], errors="coerce")
    else:
        out["post_separation"] = -pd.to_numeric(out.get("post_pair_similarity"), errors="coerce")
    out["post_separation_z"] = out.groupby("network", dropna=False)["post_separation"].transform(zscore)
    if "retrieval_rebound" in out.columns:
        out["retrieval_rebound_z"] = out.groupby("network", dropna=False)["retrieval_rebound"].transform(zscore)
    else:
        out["retrieval_rebound"] = pd.to_numeric(out["retrieval_pair_similarity"], errors="coerce") - pd.to_numeric(out["post_pair_similarity"], errors="coerce")
        out["retrieval_rebound_z"] = out.groupby("network", dropna=False)["retrieval_rebound"].transform(zscore)
    return out


def add_required_placeholders(out: pd.DataFrame) -> pd.DataFrame:
    for col in ["run3_understand", "run4_like", "run3_rt_z", "run4_rt_z", "learning_fluency_shift"]:
        if col not in out.columns:
            out[col] = pd.NA
    if "pair_id_raw" not in out.columns:
        out["pair_id_raw"] = out["condition_item_id"].astype(str).str.extract(r"_(.+)$")[0]
    if "wordfreq_z" not in out.columns and "word_frequency_mean_z" in out.columns:
        out["wordfreq_z"] = out["word_frequency_mean_z"]
    if "charlen_z" not in out.columns and "sentence_char_len_z" in out.columns:
        out["charlen_z"] = out["sentence_char_len_z"]
    return out


def make_network_master() -> pd.DataFrame:
    long = load_final_network_long()
    keys = ["subject", "condition", "condition_item_id", "network"]
    out = pivot_stage_table(long, keys)
    out = add_static_metrics(out, long, keys)
    out = add_required_placeholders(out)
    extra = load_extra_item_covars()
    out = out.merge(extra, on="condition_item_id", how="left")
    out = merge_learning_behavior(out)
    out = add_required_placeholders(out)
    return out


def make_item_master() -> pd.DataFrame:
    long = load_final_item_long()
    keys = ["subject", "condition", "condition_item_id", "network", "roi_set", "roi"]
    out = pivot_stage_table(long, keys)
    out = add_static_metrics(out, long, keys)
    out = add_required_placeholders(out)
    extra = load_extra_item_covars()
    out = out.merge(extra, on="condition_item_id", how="left")
    out = merge_learning_behavior(out)
    out = add_required_placeholders(out)
    return out


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [
        "pre_pair_similarity_z",
        "post_pair_similarity_z",
        "retrieval_pair_similarity_z",
        "post_separation_z",
        "retrieval_rebound_z",
        "memory_ord",
        "memory_strict",
        "memory_lenient",
        *MATERIAL_COVARS,
        *EXTRA_ITEM_COVARS,
        "run3_understand",
        "run4_like",
        "run3_rt_z",
        "run4_rt_z",
        "learning_fluency_shift",
    ]
    for condition, sub in df.groupby("condition", dropna=False):
        for col in cols:
            if col in sub.columns:
                rows.append({
                    "condition": condition,
                    "field": col,
                    "n_rows": len(sub),
                    "n_missing": int(sub[col].isna().sum()),
                    "missing_rate": float(sub[col].isna().mean()),
                    "coverage_rate": float(sub[col].notna().mean()),
                })
    return pd.DataFrame(rows)


def schema_audit(network: pd.DataFrame) -> pd.DataFrame:
    collisions = network.groupby("condition_item_id", dropna=False)["condition"].nunique().reset_index(name="n_conditions")
    cell = network.groupby(["network", "condition"], dropna=False).agg(
        n_rows=("subject", "count"),
        n_subjects=("subject", "nunique"),
        n_items=("condition_item_id", "nunique"),
    ).reset_index()
    rows = [
        {
            "check": "condition_item_id_condition_collision",
            "value": int((collisions["n_conditions"] > 1).sum()),
            "status": "ok" if int((collisions["n_conditions"] > 1).sum()) == 0 else "fail",
        },
        {
            "check": "yy_28_and_kj_28_distinct",
            "value": str({"yy_28", "kj_28"}.issubset(set(network["condition_item_id"].astype(str)))),
            "status": "ok" if {"yy_28", "kj_28"}.issubset(set(network["condition_item_id"].astype(str))) else "warn",
        },
        {"check": "n_network_rows", "value": len(network), "status": "ok"},
        {"check": "n_network_subjects", "value": network["subject"].nunique(), "status": "ok"},
        {"check": "n_network_condition_items", "value": network["condition_item_id"].nunique(), "status": "ok"},
    ]
    audit = pd.concat([pd.DataFrame(rows), cell.assign(check="network_condition_cell").rename(columns={"network": "value"})], ignore_index=True, sort=False)
    return audit


def main() -> None:
    network = make_network_master()
    item = make_item_master()
    cov = coverage_table(network)
    audit = schema_audit(network)
    write_tsv(item, out_path(MODULE, "revision_master_item.tsv"))
    write_tsv(network, out_path(MODULE, "revision_master_network.tsv"))
    write_tsv(cov, out_path(MODULE, "revision_covariate_coverage.tsv"))
    write_tsv(audit, out_path(MODULE, "revision_schema_audit.tsv"))
    write_json({
        "status": "ok",
        "source": str(FINAL_NC / "s0_build_master_long_table"),
        "learning_behavior_source": str(DEFAULT_PAPER_OUTPUTS / "qc" / "learning_reactivation_mapping" / "learning_behavior_item.tsv"),
        "note": "Neural metrics are read from the locked nc_converge master; run3/run4 behavioral fields are source-derived from raw E-Prime CSV files via build_learning_behavior_table.py.",
    }, out_path(MODULE, "revision_master_manifest.json"))
    print(f"Wrote r0 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
