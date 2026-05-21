#!/usr/bin/env python3
"""S5: anterior-posterior contrast（Schlichting 轴）。

模型：
    drop ~ axis_position * C(condition) + (1|subject) + (1|condition_item_id)
其中 axis_position：head=+1 / body=0 / tail=-1。

半球分别跑 + 双侧 pooled，共三套结果。

输出：
    s5_anterior_posterior_contrast/anterior_posterior_contrast.tsv
    s5_anterior_posterior_contrast/anterior_posterior_review.md
    s5_anterior_posterior_contrast/s5_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_subfield import (
    AXIS_POSITION,
    FDR_FAMILY_LABEL,
    add_common_args,
    bh_fdr,
    default_config,
    fit_formula,
    log_text,
    markdown_table,
    parse_subfield_label,
    read_table,
    write_outputs,
)

MODULE = "s5_anterior_posterior_contrast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--drop-table", type=Path, default=None,
                        help="S4 计算 drop 后的长表（subject / subROI / condition / drop）。"
                        "缺省时尝试从 S4 输出复用。")
    return parser.parse_args()


def load_drop(cfg, args: argparse.Namespace) -> pd.DataFrame:
    if args.drop_table and Path(args.drop_table).exists():
        return read_table(Path(args.drop_table))
    candidate = cfg.output_root / "s4_edge_specificity_subfield" / "subfield_drop_long.tsv"
    if candidate.exists():
        return read_table(candidate)
    return pd.DataFrame()


def annotate_axis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "subROI" not in df.columns:
        return df
    out = df.copy()
    rows = []
    for label in out["subROI"].astype(str):
        try:
            hemi, segment = parse_subfield_label(label)
        except ValueError:
            hemi, segment = (np.nan, np.nan)
        rows.append((hemi, segment))
    out[["hemisphere", "segment"]] = pd.DataFrame(rows, index=out.index)
    out["axis_position"] = out["segment"].map(AXIS_POSITION)
    return out


def fit_axis_model(frame: pd.DataFrame, hemi_label: str) -> list[dict]:
    if frame.empty:
        return []
    formula = "drop ~ axis_position * C(condition)"
    model_df = fit_formula(
        frame.dropna(subset=["drop", "axis_position", "condition"]),
        formula=formula,
        group_col="subject",
        item_col="condition_item_id" if "condition_item_id" in frame.columns else None,
    )
    rows = []
    for _, row in model_df.iterrows():
        rows.append({
            "hemisphere_set": hemi_label,
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
    drop = annotate_axis(load_drop(cfg, args))
    model_drop = drop
    if not drop.empty and "edge_type" in drop.columns:
        model_drop = drop[drop["edge_type"].astype(str) == "trained"].copy()

    rows: list[dict] = []
    if not model_drop.empty:
        for hemi_label in ("L", "R"):
            sub = model_drop[model_drop["hemisphere"] == hemi_label]
            rows.extend(fit_axis_model(sub, f"hemi_{hemi_label}"))
        rows.extend(fit_axis_model(model_drop, "pooled_LR"))

    out = pd.DataFrame(rows)
    if not out.empty and "p" in out.columns:
        # FDR 仅对 axis 相关 term；保持简单：对所有 term 一并校正
        out["q"] = bh_fdr(out["p"])
        out["tier"] = np.where(out["q"] < 0.05, "A", "B")

    review = "# Anterior-posterior contrast (Schlichting axis)\n\n"
    review += "**假设**：YY trained drop 在 head 端整合（drop 接近 0 或正向）、tail 端分化（drop 显著为负）；\n"
    review += "因此 `axis_position : C(condition)[YY]` 项预期显著。\n\n"
    if out.empty:
        review += "_No model output; check upstream drop table._"
    else:
        cols = [c for c in ["hemisphere_set", "term", "estimate", "se", "p", "q", "tier"] if c in out.columns]
        review += markdown_table(out[cols])

    write_outputs(cfg, MODULE, {
        "anterior_posterior_contrast.tsv": out,
        "anterior_posterior_review.md": review,
        "s5_log.txt": log_text("S5 anterior-posterior log", [
            f"drop_rows={len(drop)}",
            f"trained_drop_rows={len(model_drop)}",
            f"result_rows={len(out)}",
        ]),
    })


if __name__ == "__main__":
    main()
