#!/usr/bin/env python3
"""Hippocampal-binding subfamily summary (B2).

把 HPC / 旁海马（PPA/PHG）相关 ROI 作为一个独立的 binding-subfamily，
从已经跑好的主 RSA 产物里抽取它们的行、做 subfamily-only BH-FDR，
便于论文里围绕"学习 → binding → 记忆"单独讲一条线。

本脚本**不重新计算 RSA**；它只做：

1. 用 ROI manifest 的 ``theory_role`` 或 ROI 命名规则筛出 subfamily ROI。
2. 读取 4 个主 RSA 脚本已经产出的 group/subject 级表格
   （step5c / edge / trajectory / retrieval / ERS if present）。
3. 在 subfamily 内部重新做 BH-FDR，输出 tidy 表。

subfamily 定义（默认）：
- ``theory_role ∈ {"hippocampal_binding", "scene_context_binding"}``，或
- ROI 名命中正则 ``(?i)hippocamp|PPA|PHG``。

输入：``$paper/qc/*_{roi_tag}/…`` + ``$base/roi_library/manifest.tsv``
输出：``$paper/qc/hippocampal_binding_summary_{roi_tag}/`` 与
``$paper/tables_main/table_hippocampal_binding_summary_{roi_tag}.tsv``。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402
from common.roi_library import sanitize_roi_tag  # noqa: E402


DEFAULT_THEORY_ROLES = ("hippocampal_binding", "scene_context_binding")
DEFAULT_NAME_REGEX = r"(?i)hippocamp|PPA|PHG"

# Candidate source tables to scan. Keys are analysis labels; values are relative
# paths under $paper (with {roi_tag} substitution). Missing ones are skipped.
CANDIDATE_SOURCES: dict[str, list[str]] = {
    "step5c": [
        "qc/step5c_rsa_{roi_tag}/step5c_group_summary.tsv",
        "qc/step5c_rsa_{roi_tag}/step5c_pair_similarity_group.tsv",
    ],
    "edge": [
        "qc/edge_specificity_{roi_tag}/edge_specificity_group.tsv",
    ],
    "trajectory": [
        "qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_one_sample.tsv",
        "qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_pairwise.tsv",
    ],
    "retrieval": [
        "qc/retrieval_geometry_{roi_tag}/retrieval_geometry_group_fdr.tsv",
        "qc/retrieval_pair_similarity_{roi_tag}/retrieval_geometry_group_fdr.tsv",
    ],
    "ers": [
        "qc/encoding_retrieval_similarity_{roi_tag}/ers_group_one_sample.tsv",
    ],
}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _read_manifest(manifest_path: Path) -> pd.DataFrame:
    frame = read_table(manifest_path)
    for col in ("roi_set", "roi_name", "theory_role"):
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.strip()
    return frame


def _select_subfamily_rois(
    manifest: pd.DataFrame,
    roi_sets: list[str],
    theory_roles: tuple[str, ...],
    name_regex: str,
) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    selector = manifest["roi_set"].isin(roi_sets)
    role_mask = (
        manifest["theory_role"].isin(theory_roles)
        if "theory_role" in manifest.columns
        else pd.Series(False, index=manifest.index)
    )
    name_mask = manifest["roi_name"].str.contains(name_regex, na=False, regex=True)
    return manifest[selector & (role_mask | name_mask)].copy()


def _resolve_existing(paper_root: Path, candidates: list[str], roi_tag: str) -> Path | None:
    for rel in candidates:
        path = paper_root / rel.format(roi_tag=roi_tag)
        if path.exists():
            return path
    return None


def _load_source(path: Path) -> pd.DataFrame:
    try:
        frame = read_table(path)
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame({"_load_error": [str(exc)], "_source_path": [str(path)]})
    frame = frame.copy()
    frame["_source_path"] = str(path)
    return frame


def _filter_rows(frame: pd.DataFrame, subfamily: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or subfamily.empty:
        return pd.DataFrame()
    if "roi" not in frame.columns and "roi_name" in frame.columns:
        frame = frame.rename(columns={"roi_name": "roi"})
    if "roi" not in frame.columns:
        return pd.DataFrame()
    keys = subfamily[["roi_set", "roi_name"]].drop_duplicates()
    keys = keys.rename(columns={"roi_name": "roi"})
    if "roi_set" in frame.columns:
        merged = frame.merge(keys, on=["roi_set", "roi"], how="inner")
    else:
        merged = frame.merge(keys[["roi"]], on="roi", how="inner")
    return merged


def _rerun_bh_within_subfamily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "p" not in out.columns:
        return out
    out["q_bh_hipp_subfamily"] = np.nan
    group_cols = [
        col
        for col in ("analysis_type", "condition", "variant", "phase", "contrast", "edge")
        if col in out.columns
    ]
    if not group_cols:
        out["q_bh_hipp_subfamily"] = _bh_fdr(out["p"])
        return out
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_hipp_subfamily"] = _bh_fdr(out.loc[idx, "p"])
    return out


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Aggregate existing RSA outputs into a hippocampal-binding subfamily table.",
    )
    parser.add_argument(
        "--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv"
    )
    parser.add_argument(
        "--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"]
    )
    parser.add_argument(
        "--theory-roles", nargs="+", default=list(DEFAULT_THEORY_ROLES)
    )
    parser.add_argument("--name-regex", default=DEFAULT_NAME_REGEX)
    parser.add_argument(
        "--paper-output-root", type=Path, default=base_dir / "paper_outputs"
    )
    args = parser.parse_args()

    manifest = _read_manifest(args.roi_manifest)
    subfamily = _select_subfamily_rois(
        manifest,
        roi_sets=list(args.roi_sets),
        theory_roles=tuple(args.theory_roles),
        name_regex=args.name_regex,
    )

    # 使用 current_roi_set 不合适（我们要按传入的 roi_sets 各自处理），
    # 直接用 sanitize_roi_tag 对每个 roi_set 枚举一次。
    tables_main = ensure_dir(args.paper_output_root / "tables_main")

    combined_rows: list[pd.DataFrame] = []
    missing_rows_all: list[dict[str, object]] = []

    for roi_set in args.roi_sets:
        missing_rows: list[dict[str, object]] = []
        roi_tag = sanitize_roi_tag(roi_set)
        subset = subfamily[subfamily["roi_set"] == roi_set]
        if subset.empty:
            continue
        out_dir = ensure_dir(
            args.paper_output_root / "qc" / f"hippocampal_binding_summary_{roi_tag}"
        )
        subfamily_out = subset.copy()
        subfamily_out["subfamily_label"] = "hippocampal_binding"
        write_table(subfamily_out, out_dir / "subfamily_rois.tsv")

        for analysis, candidates in CANDIDATE_SOURCES.items():
            source = _resolve_existing(args.paper_output_root, candidates, roi_tag)
            if source is None:
                row = {
                    "roi_set": roi_set,
                    "roi_tag": roi_tag,
                    "analysis": analysis,
                    "status": "missing",
                    "candidates": "; ".join(
                        str(args.paper_output_root / rel.format(roi_tag=roi_tag))
                        for rel in candidates
                    ),
                }
                missing_rows.append(row)
                missing_rows_all.append(row)
                continue
            raw = _load_source(source)
            filtered = _filter_rows(raw, subset)
            if filtered.empty:
                row = {
                    "roi_set": roi_set,
                    "roi_tag": roi_tag,
                    "analysis": analysis,
                    "status": "no_subfamily_rows",
                    "candidates": str(source),
                }
                missing_rows.append(row)
                missing_rows_all.append(row)
                continue
            adjusted = _rerun_bh_within_subfamily(filtered)
            adjusted["analysis_label"] = analysis
            adjusted["subfamily_label"] = "hippocampal_binding"
            adjusted["roi_set_requested"] = roi_set
            write_table(adjusted, out_dir / f"{analysis}_subfamily_fdr.tsv")
            combined_rows.append(adjusted)

        write_table(pd.DataFrame(missing_rows), out_dir / "missing_sources.tsv")

    combined = (
        pd.concat(combined_rows, ignore_index=True, sort=False)
        if combined_rows
        else pd.DataFrame()
    )

    # 防止重复汇总/重跑导致的双计数：按稳定键去重（忽略 _source_path 这种 provenance 列）
    if not combined.empty:
        stable_keys = [
            "roi_set",
            "roi",
            "analysis_label",
            "analysis_type",
            "condition",
            "variant",
            "phase",
            "contrast",
            "edge",
            "test_type",
        ]
        subset_cols = [c for c in stable_keys if c in combined.columns]
        if subset_cols:
            combined = combined.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
        else:
            combined = combined.drop_duplicates().reset_index(drop=True)
    # Combined main table covers both meta_metaphor and meta_spatial subfamily rows.
    combined_tag = sanitize_roi_tag("meta_hipp_subfamily")
    combined_dir = ensure_dir(
        args.paper_output_root / "qc" / f"hippocampal_binding_summary_{combined_tag}"
    )
    write_table(combined, combined_dir / "combined_subfamily_fdr.tsv")
    write_table(
        combined,
        tables_main / f"table_hippocampal_binding_summary_{combined_tag}.tsv",
    )

    write_table(pd.DataFrame(missing_rows_all), combined_dir / "missing_sources.tsv")

    save_json(
        {
            "roi_sets": list(args.roi_sets),
            "theory_roles": list(args.theory_roles),
            "name_regex": args.name_regex,
            "subfamily_label": "hippocampal_binding",
            "n_subfamily_rois": int(len(subfamily)),
            "analyses": list(CANDIDATE_SOURCES.keys()),
            "n_combined_rows": int(len(combined)),
            "n_missing_sources": int(len(missing_rows_all)),
            "note": (
                "Aggregation only. Hippocampal subfamily BH-FDR is rerun within-subfamily; "
                "the primary ROI family BH-FDR in upstream tables remains authoritative."
            ),
        },
        combined_dir / "subfamily_manifest.json",
    )


if __name__ == "__main__":
    main()
