#!/usr/bin/env python3
"""D4: HPC subfield × condition × memory subsequent-memory analysis.

参考 ``.trae/specs/expand-dm-narrative/spec.md`` §4.4。

目标
----
把 hpc_subfield_three_axis 的 head/body/tail × L/R 6 段（不切 CA1/CA3/DG）
作为 a4-同构 Dm 的 within-subject factor，回答 anterior vs posterior subfield
specificity（YY × memory × segment 三阶交互）。

输入
----
- hpc_subfield_three_axis 的 ``s2_subfield_extract_beta`` 输出 ``beta_long.tsv``
  （含 subject / run_phase / condition / item_id / subROI / beta_vector_path）；
- item_mechanism_table.tsv（含 memory_strict / memory_lenient / memory_score 与 covariates）。

不修改 hpc_subfield_three_axis 任一脚本与其输出沙箱；只读取 beta_long.tsv 与按
beta_vector_path 装载的 NPZ。开发机 dry-run 下输入缺失时返回空表。

输出沙箱: ``paper_outputs/qc/nc_converge/d4_subfield_dm/``。
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    bh_fdr,
    build_condition_item_id,
    default_config,
    fit_formula,
    q_for_ok_rows,
    read_table,
    write_outputs,
    zscore,
)

MODULE = "d4_subfield_dm"
DEFAULT_BETA_LONG = Path("paper_outputs/qc/hpc_subfield_three_axis/s2_beta_extract/beta_long.tsv")
DEFAULT_ITEM_TABLE = Path("paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
SEGMENTS = ("head", "body", "tail")
HEMISPHERES = ("L", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--beta-long", type=Path, default=None, help="hpc_subfield_three_axis/s2_subfield_extract_beta/beta_long.tsv")
    parser.add_argument("--item-table", type=Path, default=None, help="item_mechanism_table.tsv（含 memory_*）")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# subROI -> (segment, hemisphere) parser
# ---------------------------------------------------------------------------


def parse_subROI(name: str) -> Tuple[str, str]:
    """从 subROI 名称推断 (segment, hemisphere)；失败返回 ('', '')."""
    text = str(name).lower()
    seg = ""
    for cand in SEGMENTS:
        if cand in text:
            seg = cand
            break
    hemi = ""
    if "_l" in text or "left" in text or text.endswith("_l"):
        hemi = "L"
    elif "_r" in text or "right" in text or text.endswith("_r"):
        hemi = "R"
    return seg, hemi


def _normal_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        value_f = float(text)
    except Exception:
        return text
    if not np.isfinite(value_f):
        return ""
    return str(int(value_f)) if value_f.is_integer() else str(value_f)


def _pair_id_from_item_id(value: object) -> str:
    match = re.search(r"(\d+)", str(value))
    return _normal_pair_id(match.group(1)) if match else ""


# ---------------------------------------------------------------------------
# Beta NPZ loading + pair similarity
# ---------------------------------------------------------------------------


def load_beta_vectors(beta_long: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    cache: Dict[str, np.ndarray] = {}
    rows: List[dict] = []
    for _, row in beta_long.iterrows():
        path = row.get("beta_vector_path")
        if not isinstance(path, str) or not path:
            continue
        key = f"{row.get('subject')}::{row.get('subROI')}::{row.get('run_phase')}::{row.get('item_id')}"
        try:
            with np.load(path) as data:
                arr = np.asarray(data["beta"], dtype=float) if "beta" in data.files else None
        except Exception as exc:
            rows.append({"key": key, "status": f"load_failed: {exc}", "n_voxels": 0})
            continue
        if arr is None or arr.size == 0:
            rows.append({"key": key, "status": "empty_array", "n_voxels": 0})
            continue
        cache[key] = arr
        rows.append({"key": key, "status": "ok", "n_voxels": int(arr.size)})
    return cache, pd.DataFrame(rows)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    a_v = a - a.mean()
    b_v = b - b.mean()
    denom = float(np.sqrt((a_v ** 2).sum() * (b_v ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((a_v * b_v).sum() / denom)


def _fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def compute_pair_similarity(beta_long: pd.DataFrame, cache: Dict[str, np.ndarray]) -> pd.DataFrame:
    """对每 (subject, subROI, run_phase, condition_item_id) 内的 item 集合两两 Fisher-z(Pearson)。"""
    if beta_long.empty:
        return pd.DataFrame()
    if "condition_item_id" not in beta_long.columns and {"condition", "item_id"}.issubset(beta_long.columns):
        beta_long = beta_long.copy()
        beta_long["original_pair_id"] = beta_long["item_id"].map(_pair_id_from_item_id)
        beta_long = build_condition_item_id(beta_long)
    elif "condition_item_id" not in beta_long.columns:
        beta_long = build_condition_item_id(beta_long)
    keys = ["subject", "subROI", "run_phase", "condition", "condition_item_id"]
    keep_keys = [k for k in keys if k in beta_long.columns]
    if not {"subject", "subROI", "run_phase", "condition_item_id"}.issubset(beta_long.columns):
        warnings.warn(f"beta_long missing key columns: {set(keys) - set(beta_long.columns)}")
        return pd.DataFrame()
    rows: List[dict] = []
    # 同一 condition_item_id 下的两 item / 两 word 求平均后再两两计算
    for grp_keys, sub in beta_long.groupby(keep_keys, dropna=False, sort=False):
        ids = sub.get("item_id") if "item_id" in sub.columns else None
        # 用 beta_vector_path 为唯一 trial key，pair similarity = within-pair items 两两 Fisher-z
        arrays: List[np.ndarray] = []
        for _, row in sub.iterrows():
            key = f"{row.get('subject')}::{row.get('subROI')}::{row.get('run_phase')}::{row.get('item_id')}"
            arr = cache.get(key)
            if arr is not None and arr.size > 0:
                arrays.append(arr)
        if len(arrays) < 2:
            continue
        sims = []
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                if arrays[i].size != arrays[j].size:
                    continue
                sims.append(_fisher_z(_safe_corr(arrays[i], arrays[j])))
        sims = [s for s in sims if np.isfinite(s)]
        if not sims:
            continue
        rec = dict(zip(keep_keys, grp_keys if isinstance(grp_keys, tuple) else (grp_keys,)))
        rec["pair_similarity_raw"] = float(np.mean(sims))
        rec["n_items"] = int(len(arrays))
        rows.append(rec)
    item = pd.DataFrame(rows)
    if item.empty:
        return item
    item[["segment", "hemisphere"]] = item["subROI"].apply(lambda x: pd.Series(parse_subROI(x)))
    if "condition" in item.columns:
        item["condition"] = item["condition"].astype(str).str.lower()
    # 单位策略：在 (subROI, run_phase) 内做 z（与 hpc_subfield_three_axis 内部口径一致）
    item["pair_similarity_z"] = item.groupby(["subROI", "run_phase"], dropna=False)["pair_similarity_raw"].transform(zscore)
    return item


# ---------------------------------------------------------------------------
# Merge memory
# ---------------------------------------------------------------------------


def merge_item_memory(item: pd.DataFrame, item_table: pd.DataFrame) -> pd.DataFrame:
    if item.empty or item_table.empty:
        return item
    keep_cols = [c for c in [
        "subject", "condition_item_id",
        "memory",
        "memory_strict", "memory_lenient", "memory_prop", "memory_score", "memory_successes",
        *COVARIATES,
    ] if c in item_table.columns]
    if not keep_cols:
        return item
    meta = item_table[keep_cols].drop_duplicates(subset=["subject", "condition_item_id"])
    if "memory_prop" not in meta.columns:
        for source in ("memory", "memory_score", "memory_successes"):
            if source in meta.columns:
                if source == "memory_successes":
                    meta["memory_prop"] = pd.to_numeric(meta[source], errors="coerce") / 2.0
                else:
                    meta["memory_prop"] = pd.to_numeric(meta[source], errors="coerce")
                break
    if "memory_strict" not in meta.columns and "memory_prop" in meta.columns:
        meta["memory_strict"] = (pd.to_numeric(meta["memory_prop"], errors="coerce") >= 1.0).astype(float)
    if "memory_lenient" not in meta.columns and "memory_prop" in meta.columns:
        meta["memory_lenient"] = (pd.to_numeric(meta["memory_prop"], errors="coerce") >= 0.5).astype(float)
    if "memory_successes" not in meta.columns and "memory_prop" in meta.columns:
        meta["memory_successes"] = np.rint(pd.to_numeric(meta["memory_prop"], errors="coerce").clip(0, 1) * 2).astype("Int64")
    return item.merge(meta, on=["subject", "condition_item_id"], how="left")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_per_segment_models(item: pd.DataFrame) -> pd.DataFrame:
    """每 (segment, hemisphere) 跑 a4-同构 Dm 模型（仅 post run_phase）."""
    if item.empty:
        return pd.DataFrame()
    if "run_phase" in item.columns:
        post = item[item["run_phase"].astype(str).str.lower().isin({"post", "post_encoding"})]
    else:
        post = item
    if post.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    covs = [c for c in COVARIATES if c in post.columns and pd.to_numeric(post[c], errors="coerce").notna().sum() >= 10]
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    for outcome in ("memory_strict", "memory_lenient"):
        if outcome not in post.columns:
            continue
        for (segment, hemisphere), sub in post.groupby(["segment", "hemisphere"], dropna=False):
            need = [outcome, "pair_similarity_z", "condition", *covs]
            model_data = sub.dropna(subset=[c for c in need if c in sub.columns]).copy()
            if len(model_data) < 30 or model_data[outcome].nunique(dropna=True) < 2:
                rows.append(pd.DataFrame([{
                    "segment": segment, "hemisphere": hemisphere,
                    "outcome": outcome, "term": "__model__",
                    "status": "too_few_rows", "n_obs": int(len(model_data)),
                }]))
                continue
            formula = f"{outcome} ~ C(condition, Treatment('kj')) * pair_similarity_z{cov_expr}"
            res = fit_formula(model_data, formula, family="binomial")
            res.insert(0, "segment", segment)
            res.insert(1, "hemisphere", hemisphere)
            res.insert(2, "outcome", outcome)
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


def fit_pooled_three_way(item: pd.DataFrame) -> pd.DataFrame:
    """跨 segment pooled 三阶交互."""
    if item.empty:
        return pd.DataFrame()
    if "run_phase" in item.columns:
        post = item[item["run_phase"].astype(str).str.lower().isin({"post", "post_encoding"})]
    else:
        post = item
    if post.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    covs = [c for c in COVARIATES if c in post.columns and pd.to_numeric(post[c], errors="coerce").notna().sum() >= 10]
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    for outcome in ("memory_strict", "memory_lenient"):
        if outcome not in post.columns:
            continue
        need = [outcome, "pair_similarity_z", "condition", "segment", "hemisphere", *covs]
        model_data = post.dropna(subset=[c for c in need if c in post.columns]).copy()
        if len(model_data) < 50:
            rows.append(pd.DataFrame([{
                "outcome": outcome, "term": "__model__",
                "status": "too_few_rows", "n_obs": int(len(model_data)),
            }]))
            continue
        formula = (
            f"{outcome} ~ C(condition, Treatment('kj')) * pair_similarity_z * C(segment, Treatment('body'))"
            f" + C(hemisphere){cov_expr}"
        )
        res = fit_formula(model_data, formula, family="binomial")
        res.insert(0, "outcome", outcome)
        res.insert(1, "model_kind", "pooled_three_way")
        rows.append(res)
    if not rows:
        return pd.DataFrame()
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


def descriptives(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    cols = [c for c in ["segment", "hemisphere", "condition", "run_phase"] if c in item.columns]
    if not cols:
        return pd.DataFrame()
    return item.groupby(cols, dropna=False, as_index=False).agg(
        n=("pair_similarity_raw", "count"),
        mean_raw=("pair_similarity_raw", "mean"),
        sd_raw=("pair_similarity_raw", "std"),
        mean_z=("pair_similarity_z", "mean"),
        n_subjects=("subject", "nunique") if "subject" in item.columns else ("pair_similarity_raw", "count"),
    )


def build_manifest(args: argparse.Namespace, item: pd.DataFrame, beta_long: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"key": "beta_long", "value": str(args.beta_long) if args.beta_long else "default"},
        {"key": "item_table", "value": str(args.item_table) if args.item_table else "default"},
        {"key": "n_beta_rows", "value": int(len(beta_long))},
        {"key": "n_pair_rows", "value": int(len(item))},
        {"key": "segments", "value": ",".join(sorted(item["segment"].dropna().astype(str).unique())) if "segment" in item.columns and not item.empty else ""},
        {"key": "hemispheres", "value": ",".join(sorted(item["hemisphere"].dropna().astype(str).unique())) if "hemisphere" in item.columns and not item.empty else ""},
    ])


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    beta_long_path = Path(args.beta_long or cfg.base_dir / DEFAULT_BETA_LONG)
    item_path = Path(args.item_table or cfg.base_dir / DEFAULT_ITEM_TABLE)

    if beta_long_path.exists():
        beta_long = read_table(beta_long_path)
    else:
        if not args.allow_empty:
            raise FileNotFoundError(f"beta_long not found: {beta_long_path}")
        warnings.warn(f"beta_long missing: {beta_long_path}")
        beta_long = pd.DataFrame()

    if item_path.exists():
        item_table = build_condition_item_id(read_table(item_path))
    else:
        if not args.allow_empty:
            raise FileNotFoundError(f"item_table not found: {item_path}")
        warnings.warn(f"item_table missing: {item_path}")
        item_table = pd.DataFrame()

    if not beta_long.empty:
        if "condition_item_id" not in beta_long.columns and {"condition", "item_id"}.issubset(beta_long.columns):
            beta_long = beta_long.copy()
            beta_long["original_pair_id"] = beta_long["item_id"].map(_pair_id_from_item_id)
        beta_long = build_condition_item_id(beta_long)

    cache, qc = load_beta_vectors(beta_long) if not beta_long.empty else ({}, pd.DataFrame())
    pair_item = compute_pair_similarity(beta_long, cache) if not beta_long.empty else pd.DataFrame()
    pair_item = merge_item_memory(pair_item, item_table)
    per_seg = fit_per_segment_models(pair_item)
    pooled = fit_pooled_three_way(pair_item)
    desc = descriptives(pair_item)
    mani = build_manifest(args, pair_item, beta_long)
    write_outputs(
        cfg,
        MODULE,
        {
            "subfield_models_per_segment.tsv": per_seg if not per_seg.empty else pd.DataFrame(columns=["segment", "hemisphere", "outcome", "term", "status"]),
            "subfield_pooled_three_way.tsv": pooled if not pooled.empty else pd.DataFrame(columns=["outcome", "term", "status"]),
            "subfield_descriptives.tsv": desc if not desc.empty else pd.DataFrame(columns=["segment", "hemisphere", "condition", "n"]),
            "pair_similarity_long.tsv": pair_item if not pair_item.empty else pd.DataFrame(columns=["subject", "subROI", "run_phase", "segment", "hemisphere", "pair_similarity_raw"]),
            "beta_qc.tsv": qc if not qc.empty else pd.DataFrame(columns=["key", "status"]),
            "manifest.tsv": mani,
        },
    )


if __name__ == "__main__":
    main()
