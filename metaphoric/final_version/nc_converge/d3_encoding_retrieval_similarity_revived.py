#!/usr/bin/env python3
"""D3: encoding-retrieval similarity revived (own − other × memory × condition).

参考 ``.trae/specs/expand-dm-narrative/spec.md`` §4.1。

目标
----
闭合 §31.5 reactivation caveat：在 voxel-level 重做 ERS，但用 **own − other** 基准
扣除 baseline漂移，并以 ``memory_strict / memory_lenient / memory_prop / memory_successes``
作为 moderator 检验 reinstatement-memory 关联。

ERS 公式（每 (subject, roi, pair) 在 condition 内）
1. ``ers_post_to_retrieval_own = corr(pair_post_i, pair_retrieval_i)``（Fisher-z(Pearson)）；
2. ``ers_post_to_retrieval_other = mean_{j ≠ i, condition(j) == condition(i)} corr(pair_post_i, pair_retrieval_j)``；
3. ``ers_post_to_retrieval_diff = own - other``（核心 outcome，扣除 condition 内 baseline）；
4. 同时跑 ``ers_pre_to_retrieval_*`` 作为 baseline 对照（pre 阶段是否本身就达到 retrieval 相似度）。

入口与 D2 一致：voxel_pattern_long.tsv 优先；npz pattern_root fallback；都不可用时
manifest 标 ``no_pattern_input`` 不抛异常（dev sanity）。

输出沙箱: ``paper_outputs/qc/nc_converge/d3_encoding_retrieval_similarity_revived/``。

不耦合 ``brain_behavior/encoding_retrieval_similarity.py`` 与
``reviewer_supp/m4_correct_ers.py``；保持各自结果不动。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    add_network_column,
    bh_fdr,
    build_condition_item_id,
    default_config,
    fit_formula,
    q_for_ok_rows,
    read_table,
    write_outputs,
    zscore,
)

MODULE = "d3_encoding_retrieval_similarity_revived"
DEFAULT_INPUT = Path("paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_PATTERN_LONG = Path("paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv")
DEFAULT_PATTERN_ROOT = Path("pattern_root")
DEFAULT_ROI_MANIFEST = Path("roi_library/manifest.tsv")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
STAGES = ("pre", "post", "retrieval")
MEMORY_VARS = ("memory_strict", "memory_lenient", "memory_prop", "memory_successes")
ERS_OUTCOMES = (
    "ers_post_to_retrieval_diff",
    "ers_post_to_retrieval_own",
    "ers_pre_to_retrieval_diff",
    "ers_pre_to_retrieval_own",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None, help="item_mechanism_table.tsv")
    parser.add_argument("--pattern-long-tsv", type=Path, default=None)
    parser.add_argument("--pattern-root", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--include-pre-baseline", action="store_true", default=True)
    parser.add_argument("--no-pre-baseline", dest="include_pre_baseline", action="store_false")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pattern loading (与 D2 同口径)
# ---------------------------------------------------------------------------


def load_pair_patterns(
    pattern_long_tsv: Path | None,
    pattern_root: Path | None,
    *,
    roi_manifest: Path | None = None,
    max_subjects: int | None = None,
    allow_empty: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """返回 (manifest, cache, source_tag).

    cache: ``pair_pattern_id`` -> 1-D voxel pattern (两词平均)。
    """
    if pattern_long_tsv is not None and Path(pattern_long_tsv).exists():
        manifest, cache = _load_long_to_pair(Path(pattern_long_tsv))
        return manifest, cache, "long_tsv"
    if pattern_root is not None and Path(pattern_root).exists():
        try:
            from d2_voxel_trajectory_alignment import load_voxel_patterns as d2_load_voxel_patterns

            word_manifest, word_cache, source = d2_load_voxel_patterns(
                Path(pattern_root),
                Path("__missing_voxel_pattern_long__.tsv"),
                roi_manifest=roi_manifest,
                max_subjects=max_subjects,
                allow_empty=allow_empty,
            )
        except Exception as exc:
            warnings.warn(f"D3 NIfTI fallback failed: {exc}")
            word_manifest = pd.DataFrame()
            word_cache = {}
            source = "none"
        if not word_manifest.empty and word_cache:
            return _word_manifest_to_pair(word_manifest, word_cache), _pair_cache_from_word_manifest(word_manifest, word_cache), f"{source}_pair_mean"
        warnings.warn("D3 npz/NIfTI fallback did not produce patterns; manifest will be empty.")
    if allow_empty:
        return pd.DataFrame(), {}, "none"
    raise FileNotFoundError(
        f"No voxel pattern input found. Tried long_tsv={pattern_long_tsv}, root={pattern_root}."
    )


def _word_manifest_to_pair(word_manifest: pd.DataFrame, word_cache: Dict[str, np.ndarray]) -> pd.DataFrame:
    if word_manifest.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    keys = ["subject", "network", "roi", "stage", "condition_item_id"]
    for grp_keys, sub in word_manifest[word_manifest.get("status", "ok").astype(str).eq("ok")].groupby(keys, dropna=False, sort=False):
        word_rows = sub.sort_values("word_index")
        arrs = [
            word_cache.get(str(row.get("pattern_array_id", "")), np.zeros((0,)))
            for _, row in word_rows.iterrows()
            if int(row.get("word_index", -1)) in (0, 1)
        ]
        if len(arrs) < 2 or arrs[0].size != arrs[1].size or arrs[0].size == 0:
            status = "missing_word"
            arr_pair = np.zeros((0,))
        else:
            arr_pair = (arrs[0] + arrs[1]) / 2.0
            status = "ok" if np.isfinite(arr_pair).all() else "nonfinite_voxels"
        rec = dict(zip(keys, grp_keys))
        rec["pair_pattern_id"] = "::".join(str(x) for x in grp_keys)
        rec["n_voxels"] = int(arr_pair.size)
        rec["status"] = status
        rows.append(rec)
    return pd.DataFrame(rows)


def _pair_cache_from_word_manifest(word_manifest: pd.DataFrame, word_cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    if word_manifest.empty:
        return cache
    keys = ["subject", "network", "roi", "stage", "condition_item_id"]
    for grp_keys, sub in word_manifest[word_manifest.get("status", "ok").astype(str).eq("ok")].groupby(keys, dropna=False, sort=False):
        word_rows = sub.sort_values("word_index")
        arrs = [
            word_cache.get(str(row.get("pattern_array_id", "")), np.zeros((0,)))
            for _, row in word_rows.iterrows()
            if int(row.get("word_index", -1)) in (0, 1)
        ]
        pid = "::".join(str(x) for x in grp_keys)
        if len(arrs) >= 2 and arrs[0].size == arrs[1].size and arrs[0].size > 0:
            cache[pid] = (arrs[0] + arrs[1]) / 2.0
        else:
            cache[pid] = np.zeros((0,))
    return cache


def _load_long_to_pair(path: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    frame = read_table(path)
    needed = {"subject", "roi", "stage", "condition_item_id", "word_index", "voxel", "value"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"voxel_pattern_long.tsv missing columns: {sorted(missing)}")
    frame = build_condition_item_id(frame)
    frame = add_network_column(frame)
    frame["stage"] = frame["stage"].astype(str).str.lower()
    frame = frame[frame["stage"].isin(STAGES)].copy()
    frame["voxel"] = pd.to_numeric(frame["voxel"], errors="coerce").astype("Int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["voxel"])
    keys = ["subject", "network", "roi", "stage", "condition_item_id"]
    cache: Dict[str, np.ndarray] = {}
    rows: List[dict] = []
    for grp_keys, sub in frame.groupby(keys, dropna=False, sort=False):
        # pair pattern = 两词平均：先对每个 word_index 排序后取均值
        words = sub.groupby("word_index", dropna=False)
        word_arrs: List[np.ndarray] = []
        for _wi, ws in words:
            ws_sorted = ws.sort_values("voxel")
            arr = ws_sorted["value"].to_numpy(dtype=float)
            word_arrs.append(arr)
        if len(word_arrs) < 2:
            status = "missing_word"
            arr_pair = np.zeros((0,))
        elif word_arrs[0].size != word_arrs[1].size or word_arrs[0].size == 0:
            status = "shape_mismatch"
            arr_pair = np.zeros((0,))
        else:
            arr_pair = (word_arrs[0] + word_arrs[1]) / 2.0
            status = "ok" if np.isfinite(arr_pair).all() else "nonfinite_voxels"
        pid = "::".join(str(x) for x in grp_keys)
        cache[pid] = arr_pair
        rec = dict(zip(keys, grp_keys))
        rec.update({"pair_pattern_id": pid, "n_voxels": int(arr_pair.size), "status": status})
        rows.append(rec)
    return pd.DataFrame(rows), cache


# ---------------------------------------------------------------------------
# ERS own-other
# ---------------------------------------------------------------------------


def _fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return float("nan")
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    a_v = a - a.mean()
    b_v = b - b.mean()
    denom = float(np.sqrt((a_v ** 2).sum() * (b_v ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((a_v * b_v).sum() / denom)


def compute_ers_own_other(
    manifest: pd.DataFrame,
    cache: Dict[str, np.ndarray],
    item_table: pd.DataFrame,
    *,
    include_pre_baseline: bool,
) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    needed_meta = ["subject", "condition_item_id", "condition"]
    if not set(needed_meta).issubset(item_table.columns):
        warnings.warn(f"item_table missing columns: {set(needed_meta) - set(item_table.columns)}")
        return pd.DataFrame()
    cond_map = (
        item_table[needed_meta]
        .drop_duplicates(subset=["subject", "condition_item_id"])
        .set_index(["subject", "condition_item_id"])["condition"]
        .to_dict()
    )
    rows: List[dict] = []
    base_keys = ["subject", "network", "roi"]
    stages_to_use = ("post", "retrieval") + (("pre",) if include_pre_baseline else ())
    for grp_keys, sub in manifest.groupby(base_keys, dropna=False, sort=False):
        # 把 (stage, pair) -> pattern id 的索引拉出来
        per_stage_pair: Dict[str, Dict[str, np.ndarray]] = {st: {} for st in stages_to_use}
        for _, row in sub.iterrows():
            st = str(row.get("stage"))
            if st not in per_stage_pair:
                continue
            cid = str(row.get("condition_item_id"))
            arr = cache.get(str(row.get("pair_pattern_id", "")), np.zeros((0,)))
            if arr.size == 0:
                continue
            per_stage_pair[st][cid] = arr
        if "post" not in per_stage_pair or "retrieval" not in per_stage_pair:
            continue
        post_pairs = per_stage_pair.get("post", {})
        retr_pairs = per_stage_pair.get("retrieval", {})
        pre_pairs = per_stage_pair.get("pre", {}) if include_pre_baseline else {}
        common_post_retr = set(post_pairs.keys()) & set(retr_pairs.keys())
        if not common_post_retr:
            continue
        # condition 内 other：先按 condition 分桶
        condition_buckets: Dict[str, List[str]] = {}
        for cid in common_post_retr:
            cond = cond_map.get((row_dict_subject(grp_keys, base_keys), cid))
            if cond is None:
                continue
            condition_buckets.setdefault(str(cond), []).append(cid)
        for cid in common_post_retr:
            cond = cond_map.get((row_dict_subject(grp_keys, base_keys), cid))
            if cond is None:
                continue
            cond = str(cond)
            others = [c for c in condition_buckets.get(cond, []) if c != cid]
            post_i = post_pairs[cid]
            retr_i = retr_pairs[cid]
            own_post = _fisher_z(_safe_corr(post_i, retr_i))
            other_post_vals = []
            for c2 in others:
                if c2 not in retr_pairs:
                    continue
                other_post_vals.append(_fisher_z(_safe_corr(post_i, retr_pairs[c2])))
            other_post = float(np.nanmean(other_post_vals)) if other_post_vals else float("nan")
            row_out = {
                "subject": row_dict_subject(grp_keys, base_keys),
                "network": row_dict_value(grp_keys, base_keys, "network"),
                "roi": row_dict_value(grp_keys, base_keys, "roi"),
                "condition": cond,
                "condition_item_id": cid,
                "ers_post_to_retrieval_own": own_post,
                "ers_post_to_retrieval_other": other_post,
                "ers_post_to_retrieval_diff": own_post - other_post if np.isfinite(own_post) and np.isfinite(other_post) else float("nan"),
            }
            if include_pre_baseline and cid in pre_pairs:
                pre_i = pre_pairs[cid]
                own_pre = _fisher_z(_safe_corr(pre_i, retr_i))
                other_pre_vals = []
                for c2 in others:
                    if c2 not in retr_pairs:
                        continue
                    other_pre_vals.append(_fisher_z(_safe_corr(pre_i, retr_pairs[c2])))
                other_pre = float(np.nanmean(other_pre_vals)) if other_pre_vals else float("nan")
                row_out["ers_pre_to_retrieval_own"] = own_pre
                row_out["ers_pre_to_retrieval_other"] = other_pre
                row_out["ers_pre_to_retrieval_diff"] = (
                    own_pre - other_pre if np.isfinite(own_pre) and np.isfinite(other_pre) else float("nan")
                )
            else:
                row_out["ers_pre_to_retrieval_own"] = float("nan")
                row_out["ers_pre_to_retrieval_other"] = float("nan")
                row_out["ers_pre_to_retrieval_diff"] = float("nan")
            row_out["status"] = "ok"
            rows.append(row_out)
    return pd.DataFrame(rows)


def row_dict_subject(grp_keys, keys):
    return row_dict_value(grp_keys, keys, "subject")


def row_dict_value(grp_keys, keys, name):
    if isinstance(grp_keys, tuple):
        for k, v in zip(keys, grp_keys):
            if k == name:
                return v
        return None
    return grp_keys if name == keys[0] else None


# ---------------------------------------------------------------------------
# Item join + memory columns
# ---------------------------------------------------------------------------


def merge_item_memory(ers_item: pd.DataFrame, item_table: pd.DataFrame) -> pd.DataFrame:
    if ers_item.empty:
        return ers_item
    keep_cols = [
        c
        for c in [
            "subject",
            "condition_item_id",
            "memory",
            "memory_strict",
            "memory_lenient",
            "memory_prop",
            "memory_score",
            "memory_successes",
            *COVARIATES,
        ]
        if c in item_table.columns
    ]
    if not keep_cols:
        return ers_item
    item_small = item_table[keep_cols].drop_duplicates(subset=["subject", "condition_item_id"])
    if "memory_prop" not in item_small.columns:
        for source in ("memory", "memory_score", "memory_successes"):
            if source in item_small.columns:
                if source == "memory_successes":
                    item_small["memory_prop"] = pd.to_numeric(item_small[source], errors="coerce") / 2.0
                else:
                    item_small["memory_prop"] = pd.to_numeric(item_small[source], errors="coerce")
                break
    if "memory_strict" not in item_small.columns and "memory_prop" in item_small.columns:
        item_small["memory_strict"] = (pd.to_numeric(item_small["memory_prop"], errors="coerce") >= 1.0).astype(float)
    if "memory_lenient" not in item_small.columns and "memory_prop" in item_small.columns:
        item_small["memory_lenient"] = (pd.to_numeric(item_small["memory_prop"], errors="coerce") >= 0.5).astype(float)
    if "memory_successes" not in item_small.columns and "memory_prop" in item_small.columns:
        item_small["memory_successes"] = np.rint(pd.to_numeric(item_small["memory_prop"], errors="coerce").clip(0, 1) * 2).astype("Int64")
    out = ers_item.merge(item_small, on=["subject", "condition_item_id"], how="left")
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_ers_models(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for outcome in ERS_OUTCOMES:
        if outcome not in item.columns:
            continue
        for memory_var in MEMORY_VARS:
            if memory_var not in item.columns:
                continue
            for network, sub in item.groupby("network", dropna=False):
                model_data = sub.dropna(subset=[outcome, memory_var, "condition"])
                if len(model_data) < 30 or model_data[memory_var].nunique(dropna=True) < 2:
                    rows.append(pd.DataFrame([{
                        "network": network,
                        "outcome": outcome,
                        "memory_var": memory_var,
                        "term": "__model__",
                        "status": "too_few_rows",
                        "n_obs": len(model_data),
                    }]))
                    continue
                covs = [c for c in COVARIATES if c in model_data.columns and pd.to_numeric(model_data[c], errors="coerce").notna().sum() >= 10]
                rhs = f"C(condition, Treatment('kj')) * {memory_var}"
                if covs:
                    rhs = rhs + " + " + " + ".join(covs)
                formula = f"{outcome} ~ {rhs}"
                res = fit_formula(model_data.dropna(subset=[outcome, memory_var, *covs]), formula, family="gaussian")
                res.insert(0, "network", network)
                res.insert(1, "outcome", outcome)
                res.insert(2, "memory_var", memory_var)
                rows.append(res)
    if not rows:
        return pd.DataFrame()
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


def fit_baseline_pre(item: pd.DataFrame) -> pd.DataFrame:
    """专门跑 pre→retrieval baseline 的 own/diff，作为 caveat 对照。"""
    if item.empty or "ers_pre_to_retrieval_diff" not in item.columns:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for outcome in ("ers_pre_to_retrieval_diff", "ers_pre_to_retrieval_own"):
        if outcome not in item.columns:
            continue
        for network, sub in item.groupby("network", dropna=False):
            model_data = sub.dropna(subset=[outcome, "condition"])
            if len(model_data) < 30:
                rows.append(pd.DataFrame([{
                    "network": network, "outcome": outcome, "term": "__model__",
                    "status": "too_few_rows", "n_obs": len(model_data),
                }]))
                continue
            res = fit_formula(model_data, f"{outcome} ~ C(condition, Treatment('kj'))", family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "outcome", outcome)
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


# ---------------------------------------------------------------------------
# Descriptives & manifest
# ---------------------------------------------------------------------------


def descriptives(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["network", "condition"] if c in item.columns]
    if not group_cols:
        return pd.DataFrame()
    rows = []
    for outcome in ERS_OUTCOMES:
        if outcome not in item.columns:
            continue
        agg = item.groupby(group_cols, dropna=False).agg(
            n=(outcome, "count"),
            mean=(outcome, "mean"),
            sd=(outcome, "std"),
            n_subjects=("subject", "nunique") if "subject" in item.columns else (outcome, "count"),
            n_items=("condition_item_id", "nunique") if "condition_item_id" in item.columns else (outcome, "count"),
        ).reset_index()
        agg["outcome"] = outcome
        rows.append(agg)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_manifest(args: argparse.Namespace, item: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    rows = [
        {"key": "input_item_table", "value": str(args.input) if args.input else "default"},
        {"key": "pattern_long_tsv", "value": str(args.pattern_long_tsv) if args.pattern_long_tsv else "default"},
        {"key": "pattern_root", "value": str(args.pattern_root) if args.pattern_root else "default"},
        {"key": "pattern_source", "value": source_tag},
        {"key": "include_pre_baseline", "value": str(args.include_pre_baseline)},
        {"key": "n_rows_item", "value": int(len(item))},
        {"key": "n_subjects", "value": int(item["subject"].nunique()) if "subject" in item.columns and not item.empty else 0},
        {"key": "networks", "value": ",".join(sorted(item["network"].dropna().astype(str).unique())) if "network" in item.columns and not item.empty else ""},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    item_path = Path(args.input or cfg.base_dir / DEFAULT_INPUT)
    long_path = Path(args.pattern_long_tsv or cfg.base_dir / DEFAULT_PATTERN_LONG)
    root_path = Path(args.pattern_root or cfg.base_dir / DEFAULT_PATTERN_ROOT)
    roi_manifest = Path(args.roi_manifest or cfg.base_dir / DEFAULT_ROI_MANIFEST)

    if item_path.exists():
        item_table = build_condition_item_id(read_table(item_path))
        item_table = add_network_column(item_table)
    else:
        if not args.allow_empty:
            raise FileNotFoundError(f"item_table not found: {item_path}")
        warnings.warn(f"item_table missing: {item_path}")
        item_table = pd.DataFrame()

    manifest, cache, source_tag = load_pair_patterns(
        long_path,
        root_path,
        roi_manifest=roi_manifest,
        max_subjects=args.max_subjects,
        allow_empty=args.allow_empty,
    )
    ers_item = compute_ers_own_other(manifest, cache, item_table, include_pre_baseline=args.include_pre_baseline)
    ers_item = merge_item_memory(ers_item, item_table)

    models = fit_ers_models(ers_item)
    baseline_models = fit_baseline_pre(ers_item)
    desc = descriptives(ers_item)
    mani = build_manifest(args, ers_item, source_tag)

    write_outputs(
        cfg,
        MODULE,
        {
            "ers_item.tsv": ers_item if not ers_item.empty else pd.DataFrame(columns=["subject", "network", "roi", "condition", "condition_item_id", *ERS_OUTCOMES, "status"]),
            "ers_models.tsv": models if not models.empty else pd.DataFrame(columns=["network", "outcome", "memory_var", "term", "status"]),
            "ers_baseline_pre_to_retrieval.tsv": baseline_models if not baseline_models.empty else pd.DataFrame(columns=["network", "outcome", "term", "status"]),
            "ers_descriptives.tsv": desc if not desc.empty else pd.DataFrame(columns=["network", "condition", "outcome", "n", "mean", "sd"]),
            "manifest.tsv": mani,
        },
    )


if __name__ == "__main__":
    main()
