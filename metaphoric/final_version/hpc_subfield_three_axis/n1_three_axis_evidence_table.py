#!/usr/bin/env python3
"""N1: 三轴证据整合表（Favila / Schlichting / Bein）。

把 §3 / §4 / §9 / §27（既有主结果，只读）+ S3 / S4 / S5（本次产出）
按三个轴归类，输出统一的 evidence table。

输出：
    n1_three_axis_evidence/three_axis_evidence_table.tsv
    n1_three_axis_evidence/three_axis_evidence_review.md
    n1_three_axis_evidence/n1_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_subfield import (
    add_common_args,
    default_config,
    log_text,
    markdown_table,
    parse_subfield_label,
    read_table,
    write_outputs,
)

MODULE = "n1_three_axis_evidence"

AXIS_FAVILA = "favila_differentiation"
AXIS_SCHLICHTING = "schlichting_anterior_posterior"
AXIS_BEIN = "bein_assimilation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--main-step5c", type=Path, default=None,
                        help="§3 主结果对应的 step5c tsv（只读）。")
    parser.add_argument("--main-edge-specificity", type=Path, default=None,
                        help="§4 / 主结果对应的 edge specificity tsv（只读）。")
    parser.add_argument("--main-bein-table", type=Path, default=None,
                        help="左 IFG / 颞极相关结果（只读）。")
    return parser.parse_args()


def safe_read(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        return read_table(Path(path))
    except Exception:
        return pd.DataFrame()


def collect_subfield_evidence(cfg) -> list[dict]:
    rows: list[dict] = []
    s3 = safe_read(cfg.output_root / "s3_step5c_subfield" / "subfield_step5c.tsv")
    s4 = safe_read(cfg.output_root / "s4_edge_specificity_subfield" / "subfield_edge_specificity.tsv")
    s5 = safe_read(cfg.output_root / "s5_anterior_posterior_contrast" / "anterior_posterior_contrast.tsv")
    if not s3.empty:
        for _, r in s3.iterrows():
            rows.append({
                "axis": AXIS_FAVILA,
                "source": "S3",
                "roi": r.get("subROI"),
                "metric": "step5c_condition_x_time",
                "term": r.get("term"),
                "effect": r.get("estimate"),
                "p": r.get("p"),
                "q": r.get("q"),
                "tier": r.get("tier"),
            })
    if not s4.empty:
        for _, r in s4.iterrows():
            rows.append({
                "axis": AXIS_FAVILA,
                "source": "S4",
                "roi": r.get("subROI"),
                "metric": str(r.get("contrast")),
                "term": r.get("term"),
                "effect": r.get("estimate"),
                "p": r.get("p"),
                "q": r.get("q"),
                "tier": r.get("tier"),
            })
    if not s5.empty:
        for _, r in s5.iterrows():
            rows.append({
                "axis": AXIS_SCHLICHTING,
                "source": "S5",
                "roi": str(r.get("hemisphere_set")),
                "metric": "anterior_posterior_contrast",
                "term": r.get("term"),
                "effect": r.get("estimate"),
                "p": r.get("p"),
                "q": r.get("q"),
                "tier": r.get("tier"),
            })
    return rows


def collect_main_evidence(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    main_step5c = safe_read(args.main_step5c)
    if not main_step5c.empty:
        for _, r in main_step5c.iterrows():
            roi = str(r.get("roi", ""))
            axis = AXIS_FAVILA
            if "hippocampus" in roi.lower() and ("ant" in roi.lower() or "post" in roi.lower()):
                axis = AXIS_SCHLICHTING
            rows.append({
                "axis": axis,
                "source": "§3",
                "roi": roi,
                "metric": "main_step5c",
                "term": r.get("term"),
                "effect": r.get("estimate"),
                "p": r.get("p"),
                "q": r.get("q"),
                "tier": r.get("tier"),
            })
    main_edge = safe_read(args.main_edge_specificity)
    if not main_edge.empty:
        for _, r in main_edge.iterrows():
            rows.append({
                "axis": AXIS_FAVILA,
                "source": "§4",
                "roi": r.get("roi"),
                "metric": str(r.get("contrast", "edge_specificity")),
                "term": r.get("term"),
                "effect": r.get("estimate"),
                "p": r.get("p"),
                "q": r.get("q"),
                "tier": r.get("tier"),
            })
    bein = safe_read(args.main_bein_table)
    if not bein.empty:
        for _, r in bein.iterrows():
            rows.append({
                "axis": AXIS_BEIN,
                "source": "§4/§6/§13",
                "roi": r.get("roi"),
                "metric": r.get("metric", "left_semantic_assimilation"),
                "term": r.get("term"),
                "effect": r.get("estimate"),
                "p": r.get("p"),
                "q": r.get("q"),
                "tier": r.get("tier"),
            })
    return rows


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    rows = collect_subfield_evidence(cfg) + collect_main_evidence(args)
    table = pd.DataFrame(rows, columns=[
        "axis", "source", "roi", "metric", "term", "effect", "p", "q", "tier",
    ])
    review = "# Three-Axis Evidence (Favila / Schlichting / Bein)\n\n"
    if table.empty:
        review += "_No evidence collected; provide --main-step5c / --main-edge-specificity / --main-bein-table on data machine._"
    else:
        for axis_label in (AXIS_FAVILA, AXIS_SCHLICHTING, AXIS_BEIN):
            sub = table[table["axis"] == axis_label]
            review += f"## {axis_label}\n\n"
            if sub.empty:
                review += "_No evidence._\n\n"
            else:
                review += markdown_table(sub.drop(columns=["axis"]))
                review += "\n\n"
    write_outputs(cfg, MODULE, {
        "three_axis_evidence_table.tsv": table,
        "three_axis_evidence_review.md": review,
        "n1_log.txt": log_text("N1 three-axis evidence log", [
            f"total_rows={len(table)}",
            f"axes={sorted(table['axis'].unique()) if not table.empty else []}",
        ]),
    })


if __name__ == "__main__":
    main()
