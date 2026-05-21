#!/usr/bin/env python3
"""S3: 在 6 个 subROI 上重做 Step5C 型 condition × time RSA。

输入：
    S2 输出的 beta_long.tsv（含每 item 的 beta_vector_path）。

建模：
    pair_similarity ~ condition * time + (1|subject) + (1|condition_item_id)
    mixed model，allow_fallback=False；FDR family = "hpc_subfield_three_axis"。

输出：
    s3_step5c_subfield/subfield_step5c.tsv
    s3_step5c_subfield/subfield_step5c_vs_main_review.md
    s3_step5c_subfield/s3_log.txt
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from shared_subfield import (
    FDR_FAMILY_LABEL,
    add_common_args,
    bh_fdr,
    build_condition_item_id,
    default_config,
    fit_formula,
    log_text,
    markdown_table,
    read_table,
    write_outputs,
)

MODULE = "s3_step5c_subfield"
TIME_ORDER = ("pre", "learning", "post", "retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--beta-long", type=Path, default=None,
                        help="S2 输出的 beta_long.tsv；缺省时在 output_root/s2_beta_extract 下自动定位。")
    parser.add_argument("--pair-table", type=Path, default=None,
                        help="pair-level 结构表（subject, condition, item_id_a, item_id_b, edge_type）。")
    return parser.parse_args()


def load_vector(path: str) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    if "beta" not in data.files:
        return None
    return np.asarray(data["beta"], dtype=float)


def compute_pair_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size != vec_b.size or vec_a.size < 2:
        return float("nan")
    if np.allclose(vec_a.std(), 0) or np.allclose(vec_b.std(), 0):
        return float("nan")
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def build_pair_similarity(beta_long: pd.DataFrame, pair_table: pd.DataFrame | None) -> pd.DataFrame:
    if beta_long.empty:
        return pd.DataFrame()
    frame = build_condition_item_id(beta_long.rename(columns={"item_id": "original_pair_id"}))
    rows: list[dict] = []
    key_cols = ["subject", "subROI", "run_phase"]
    if pair_table is not None and not pair_table.empty:
        merged = frame.merge(
            pair_table,
            left_on=["subject", "condition", "original_pair_id"],
            right_on=["subject", "condition", "item_id_a"],
            how="inner",
            suffixes=("", "_a"),
        )
        merged = merged.merge(
            frame,
            left_on=["subject", "subROI", "run_phase", "item_id_b"],
            right_on=["subject", "subROI", "run_phase", "original_pair_id"],
            how="inner",
            suffixes=("_a", "_b"),
        )
        for _, row in merged.iterrows():
            vec_a = load_vector(row.get("beta_vector_path_a"))
            vec_b = load_vector(row.get("beta_vector_path_b"))
            if vec_a is None or vec_b is None:
                continue
            sim = compute_pair_similarity(vec_a, vec_b)
            condition = row.get("condition") or row.get("condition_a") or row.get("condition_x")
            rows.append({
                "subject": row["subject"],
                "subROI": row["subROI"],
                "run_phase": row["run_phase"],
                "condition": condition,
                "edge_type": row.get("edge_type"),
                "condition_item_id": f"{condition}_{row['item_id_a']}_{row['item_id_b']}",
                "pair_similarity": sim,
            })
    else:
        # fallback：在同 condition 内按 item_id 成对配对（仅同 condition）
        for (sub, roi, phase, cond), grp in frame.groupby(
            ["subject", "subROI", "run_phase", "condition"], dropna=False
        ):
            grp = grp.dropna(subset=["beta_vector_path"])
            items = grp["original_pair_id"].astype(str).tolist()
            paths = grp["beta_vector_path"].astype(str).tolist()
            vecs = [load_vector(p) for p in paths]
            for (i, a), (j, b) in combinations(enumerate(items), 2):
                if vecs[i] is None or vecs[j] is None:
                    continue
                sim = compute_pair_similarity(vecs[i], vecs[j])
                rows.append({
                    "subject": sub,
                    "subROI": roi,
                    "run_phase": phase,
                    "condition": cond,
                    "edge_type": "any",
                    "condition_item_id": f"{cond}_{a}_{b}",
                    "pair_similarity": sim,
                })
    return pd.DataFrame(rows)


def run_model_per_subroi(pair_sim: pd.DataFrame) -> pd.DataFrame:
    if pair_sim.empty:
        return pd.DataFrame()
    frame = pair_sim.copy()
    frame["time"] = pd.Categorical(frame["run_phase"], categories=list(TIME_ORDER), ordered=True)
    records: list[dict] = []
    for roi, grp in frame.groupby("subROI", dropna=False):
        grp = grp.dropna(subset=["pair_similarity", "condition", "time"])
        if grp.empty:
            continue
        model_df = fit_formula(
            grp,
            formula="pair_similarity ~ C(condition) * C(time)",
            group_col="subject",
            item_col="condition_item_id",
        )
        for _, row in model_df.iterrows():
            records.append({
                "subROI": roi,
                "family": FDR_FAMILY_LABEL,
                "term": row.get("term"),
                "estimate": row.get("estimate"),
                "se": row.get("se"),
                "t_or_z": row.get("t_or_z") if "t_or_z" in row else row.get("z"),
                "df": row.get("df"),
                "p": row.get("p"),
                "status": row.get("status"),
                "n_obs": row.get("n_obs"),
                "formula": "pair_similarity ~ C(condition) * C(time)",
            })
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["q"] = bh_fdr(out["p"]) if "p" in out.columns else np.nan
    out["tier"] = np.where(out.get("q", np.nan) < 0.05, "A", "B")
    return out


def vs_main_review(result: pd.DataFrame) -> str:
    if result.empty:
        return "# Subfield Step5C vs §3\n\n_No model output._"
    key = result[result["term"].astype(str).str.contains("condition", na=False)].copy()
    summary = key.groupby("subROI").agg(
        min_p=("p", "min"),
        min_q=("q", "min"),
        any_tier_a=("tier", lambda s: "A" in set(s)),
    ).reset_index()
    summary["direction_vs_main_section3"] = "TBD after data machine run"
    md = "# Subfield Step5C vs §3 (main result)\n\n"
    md += "本文档在数据机跑出结果后由分析者填写 direction_vs_main_section3，对照 §3 的 R hpc 主效应。\n\n"
    md += markdown_table(summary)
    return md


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    beta_path = args.beta_long or (cfg.output_root / "s2_beta_extract" / "beta_long.tsv")
    beta_long = read_table(Path(beta_path)) if Path(beta_path).exists() else pd.DataFrame()
    default_pair_table = cfg.output_root / "p0_inputs" / "hpc_pair_table.tsv"
    pair_path = Path(args.pair_table) if args.pair_table else default_pair_table
    pair_table = read_table(pair_path) if pair_path.exists() else None
    pair_sim = build_pair_similarity(beta_long, pair_table)
    result = run_model_per_subroi(pair_sim)
    review = vs_main_review(result)
    write_outputs(cfg, MODULE, {
        "subfield_step5c.tsv": result,
        "subfield_pair_similarity.tsv": pair_sim,
        "subfield_step5c_vs_main_review.md": review,
        "s3_log.txt": log_text("S3 Step5C subfield log", [
            f"beta_long_rows={len(beta_long)}",
            f"pair_similarity_rows={len(pair_sim)}",
            f"model_rows={len(result)}",
        ]),
    })


if __name__ == "__main__":
    main()
