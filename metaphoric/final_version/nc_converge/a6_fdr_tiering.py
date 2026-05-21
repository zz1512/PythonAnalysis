#!/usr/bin/env python3
"""A6: apply evidence-tier-specific BH-FDR families."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_nc import add_common_args, bh_fdr, default_config, discover_tables, markdown_table, read_table, write_outputs

MODULE = "a6_fdr_tiering"
TIER_RULES = [
    ("a4_post_memory_component", "Tier A"),
    ("a5_network_composite", "Tier A"),
    ("b1_staged_path", "Tier A"),
    ("b2_crossnobis_robustness", "Tier B"),
    ("b3_repr_coupling", "Tier A"),
    ("reviewer_supp", "Reviewer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", nargs="*", type=Path, default=None)
    return parser.parse_args()


def infer_tier(path: Path, frame: pd.DataFrame) -> str:
    text = path.as_posix().lower()
    for key, tier in TIER_RULES:
        if key in text:
            return tier
    if "roi_followup" in text:
        return "Tier B"
    if "explor" in text or "directional" in text:
        return "Tier C"
    return "Tier C"


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    paths = args.input or discover_tables(cfg.output_root, ["**/*.tsv"])
    rows = []
    for path in paths:
        if path.name == "fdr_tier_manifest.tsv":
            continue
        try:
            frame = read_table(path)
        except Exception:
            continue
        if "p" not in frame.columns:
            continue
        tier = infer_tier(path, frame)
        temp = frame.copy()
        temp["module"] = path.parent.name
        temp["source_file"] = path.name
        temp["tier"] = tier
        # family 粒度修正：按 (tier, module, network) 拆分，避免异质模型被混入同一 BH family
        # 导致弱效应的 q 被大尺寸 family 推上去。若表里没有 network，用 'all' 兜底。
        network_series = temp["network"].astype(str) if "network" in temp.columns else pd.Series(["all"] * len(temp), index=temp.index)
        temp["family_id"] = temp["module"].astype(str) + "::" + network_series
        rows.append(temp)
    manifest = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["module", "source_file", "tier", "family_id", "p", "q"])
    if not manifest.empty:
        manifest["q"] = np.nan
        # 同时按 (tier, family_id) 分组做 BH-FDR：每个 (tier, module, network) 独立 family
        for family, idx in manifest.groupby(["tier", "family_id"], dropna=False).groups.items():
            manifest.loc[list(idx), "q"] = bh_fdr(manifest.loc[list(idx), "p"])
    summary = "# FDR Tier Summary\n\nMain tables should cite Tier A/B only. Tier C and Reviewer are supplement-only families.\n\n"
    summary += "FDR families are partitioned by (tier, module, network) to avoid merging heterogeneous models.\n\n"
    if not manifest.empty:
        summary += markdown_table(manifest.groupby(["tier", "family_id"], dropna=False).size().reset_index(name="n_tests"))
    write_outputs(cfg, MODULE, {"fdr_tier_manifest.tsv": manifest, "fdr_tier_summary.md": summary})


if __name__ == "__main__":
    main()
