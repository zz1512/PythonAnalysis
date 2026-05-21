#!/usr/bin/env python3
"""S4: subfield edge specificity（4 个 contrast）。

C1：YY trained pre→post drop（pre 基线作为对照，同 subROI 内时间对比）
C2：YY trained drop > KJ trained drop
C3：YY trained drop > pseudo-edge drop（specificity）
C4：YY trained drop > non-edge drop（trained-untrained）

所有模型 allow_fallback=False；family = "hpc_subfield_three_axis"。

输出：
    s4_edge_specificity_subfield/subfield_edge_specificity.tsv
    s4_edge_specificity_subfield/subfield_edge_specificity_review.md
    s4_edge_specificity_subfield/s4_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_subfield import (
    FDR_FAMILY_LABEL,
    add_common_args,
    bh_fdr,
    default_config,
    fit_formula,
    log_text,
    markdown_table,
    read_table,
    write_outputs,
)

MODULE = "s4_edge_specificity_subfield"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pair-similarity", type=Path, default=None,
                        help="S3 产出的 pair similarity 长表；缺省时在上游 step5c 输出中找。")
    parser.add_argument("--edge-label-col", default="edge_type",
                        help="标识 trained / pseudo / non-edge 的列名。")
    return parser.parse_args()


def load_pair_similarity(cfg, args: argparse.Namespace) -> pd.DataFrame:
    if args.pair_similarity and Path(args.pair_similarity).exists():
        return read_table(Path(args.pair_similarity))
    # 尝试在 S3 输出附近找；若没有就返回空（数据机执行时提供完整路径）
    candidates = [
        cfg.output_root / "s3_step5c_subfield" / "subfield_pair_similarity.tsv",
        cfg.base_dir / "paper_outputs" / "step5c" / "pair_similarity_long.tsv",
    ]
    for p in candidates:
        if p.exists():
            return read_table(p)
    return pd.DataFrame()


def compute_drop(pair_sim: pd.DataFrame) -> pd.DataFrame:
    """对每 (subject, subROI, condition, edge_type, condition_item_id)
    计算 post - pre 的相似性变化（drop）。"""
    if pair_sim.empty or "run_phase" not in pair_sim.columns:
        return pd.DataFrame()
    sub = pair_sim[pair_sim["run_phase"].isin(["pre", "post"])].copy()
    key_cols = [c for c in ["subject", "subROI", "condition", "condition_item_id"] if c in sub.columns]
    if "edge_type" in sub.columns:
        key_cols.append("edge_type")
    wide = sub.pivot_table(
        index=key_cols,
        columns="run_phase",
        values="pair_similarity",
        aggfunc="mean",
    ).reset_index()
    if "pre" not in wide.columns or "post" not in wide.columns:
        return pd.DataFrame()
    wide["drop"] = wide["post"] - wide["pre"]
    return wide


def fit_contrast(drop_df: pd.DataFrame, contrast_name: str, formula: str, subset: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if subset.empty:
        return rows
    for roi, grp in subset.groupby("subROI", dropna=False):
        grp = grp.dropna(subset=["drop"])
        if grp.empty:
            continue
        model_df = fit_formula(
            grp,
            formula=formula,
            group_col="subject",
            item_col="condition_item_id" if "condition_item_id" in grp.columns else None,
        )
        for _, row in model_df.iterrows():
            rows.append({
                "contrast": contrast_name,
                "subROI": roi,
                "family": FDR_FAMILY_LABEL,
                "formula": formula,
                "term": row.get("term"),
                "estimate": row.get("estimate"),
                "se": row.get("se"),
                "t_or_z": row.get("t_or_z") if "t_or_z" in row else row.get("z"),
                "df": row.get("df"),
                "p": row.get("p"),
                "status": row.get("status"),
                "n_obs": row.get("n_obs"),
            })
    return rows


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    pair_sim = load_pair_similarity(cfg, args)
    drop_df = compute_drop(pair_sim)

    rows: list[dict] = []

    if not drop_df.empty:
        # C1：YY trained drop 非零（单样本相当于 intercept）
        c1 = drop_df[(drop_df.get("condition") == "YY") & (drop_df.get("edge_type") == "trained")]
        rows.extend(fit_contrast(drop_df, "C1_YY_trained_drop_vs_zero", "drop ~ 1", c1))

        # C2：YY trained drop > KJ trained drop
        c2 = drop_df[drop_df.get("edge_type") == "trained"]
        rows.extend(fit_contrast(drop_df, "C2_YY_vs_KJ_trained_drop", "drop ~ C(condition)", c2))

        # C3：YY trained drop > pseudo-edge drop（同 YY 条件内）
        c3 = drop_df[(drop_df.get("condition") == "YY") & (drop_df.get("edge_type").isin(["trained", "pseudo"]))]
        rows.extend(fit_contrast(drop_df, "C3_YY_trained_vs_pseudo_drop", "drop ~ C(edge_type)", c3))

        # C4：YY trained drop > non-edge drop（同 YY 条件内）
        c4 = drop_df[(drop_df.get("condition") == "YY") & (drop_df.get("edge_type").isin(["trained", "non_edge"]))]
        rows.extend(fit_contrast(drop_df, "C4_YY_trained_vs_non_edge_drop", "drop ~ C(edge_type)", c4))

    out = pd.DataFrame(rows)
    if not out.empty and "p" in out.columns:
        out["q"] = bh_fdr(out["p"])
        out["tier"] = np.where(out["q"] < 0.05, "A", "B")
    review = "# Subfield edge specificity review\n\n"
    if out.empty:
        review += "_No contrast output; check upstream pair-similarity table._"
    else:
        review += markdown_table(
            out[["contrast", "subROI", "term", "estimate", "se", "p", "q", "tier"]].sort_values(["contrast", "subROI"]) ,
        )
    write_outputs(cfg, MODULE, {
        "subfield_edge_specificity.tsv": out,
        "subfield_drop_long.tsv": drop_df,
        "subfield_edge_specificity_review.md": review,
        "s4_log.txt": log_text("S4 edge specificity log", [
            f"pair_similarity_rows={len(pair_sim)}",
            f"drop_rows={len(drop_df)}",
            f"result_rows={len(out)}",
        ]),
    })


if __name__ == "__main__":
    main()
