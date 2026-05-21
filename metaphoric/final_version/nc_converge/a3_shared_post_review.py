#!/usr/bin/env python3
"""A3: shared-post component review for post/retrieval memory bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import add_common_args, build_condition_item_id, default_config, discover_tables, first_existing_column, fit_formula, markdown_table, q_for_ok_rows, read_table, write_outputs, zscore

MODULE = "a3_shared_post_review"
PATTERNS = ["**/stage45*/*.tsv", "**/stage4*/*.tsv", "**/post_to_retrieval*.tsv", "**/*memory*bridge*.tsv", "**/*item_mechanism*.tsv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def load_frame(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = read_table(path)
        except Exception:
            continue
        frame["source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = build_condition_item_id(frame)
    rename_map = {}
    for target, choices in {
        "post_pair_similarity": ["post_pair_similarity", "post_similarity", "post_sim", "post"],
        "retrieval_pair_similarity": ["retrieval_pair_similarity", "retrieval_similarity", "retrieval_sim", "retrieval"],
        "memory_prop": ["memory_prop", "memory", "accuracy", "remembered", "memory_score"],
        "rebinding": ["retrieval_rebinding", "rebinding", "retrieval_minus_post"],
    }.items():
        col = first_existing_column(out, choices)
        if col and col != target:
            rename_map[col] = target
    out = out.rename(columns=rename_map)
    for col in ["post_pair_similarity", "retrieval_pair_similarity", "memory_prop", "rebinding"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "memory_successes" not in out.columns and "memory_prop" in out.columns:
        out["memory_successes"] = np.rint(out["memory_prop"].clip(0, 1) * 2).astype("Int64")
    if "memory_prop" not in out.columns and "memory_successes" in out.columns:
        out["memory_prop"] = pd.to_numeric(out["memory_successes"], errors="coerce") / 2.0
    for col in ["post_pair_similarity", "retrieval_pair_similarity", "rebinding"]:
        if col in out.columns:
            out[f"{col}_z"] = zscore(out[col])
    for cov in ["sentence_char_len", "word_frequency_mean", "stroke_count_mean", "valence_mean", "arousal_mean", "pre_pair_similarity"]:
        if cov in out.columns:
            out[f"{cov}_z"] = zscore(out[cov])
    return out


def model_terms(frame: pd.DataFrame) -> str:
    covs = [c for c in ["sentence_char_len_z", "word_frequency_mean_z", "stroke_count_mean_z", "valence_mean_z", "arousal_mean_z", "pre_pair_similarity_z"] if c in frame.columns]
    return " + ".join(covs)


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.paper_output_root, PATTERNS)
    raw = load_frame(paths)
    if raw.empty and not args.allow_empty:
        raise FileNotFoundError("No shared-post candidate tables found; use --input or --allow-empty.")
    data = prepare(raw)
    rows = []
    required = {"memory_prop", "condition", "post_pair_similarity_z", "retrieval_pair_similarity_z", "subject", "condition_item_id"}
    if required.issubset(data.columns):
        cov = model_terms(data)
        rhs = "condition * post_pair_similarity_z + condition * retrieval_pair_similarity_z" + (" + " + cov if cov else "")
        res = fit_formula(data.dropna(subset=list(required)), f"memory_prop ~ {rhs}", family="binomial")
        res.insert(0, "analysis", "component_post_retrieval")
        rows.append(res)
    else:
        rows.append(pd.DataFrame([{"analysis": "component_post_retrieval", "term": "__model__", "status": f"missing_columns: {sorted(required - set(data.columns))}"}]))
    required_diff = {"rebinding_z", "condition", "post_pair_similarity_z", "subject", "condition_item_id"}
    if required_diff.issubset(data.columns):
        cov = model_terms(data)
        rhs = "condition * post_pair_similarity_z" + (" + " + cov if cov else "")
        res = fit_formula(data.dropna(subset=list(required_diff)), f"rebinding_z ~ {rhs}", family="gaussian")
        res.insert(0, "analysis", "difference_score_sensitivity")
        rows.append(res)
    else:
        rows.append(pd.DataFrame([{"analysis": "difference_score_sensitivity", "term": "__model__", "status": f"missing_columns: {sorted(required_diff - set(data.columns))}"}]))
    table = q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))
    md = "# Shared-post Review\n\nComponent model is the primary analysis. Difference-score rebinding is reported only as sensitivity.\n\n"
    md += markdown_table(table[[c for c in ["analysis", "term", "estimate", "p", "q_bh", "status"] if c in table.columns]])
    write_outputs(cfg, MODULE, {"shared_post_review.tsv": table, "shared_post_review.md": md, "shared_post_input_manifest.tsv": pd.DataFrame({"path": [str(p) for p in paths]})})


if __name__ == "__main__":
    main()
