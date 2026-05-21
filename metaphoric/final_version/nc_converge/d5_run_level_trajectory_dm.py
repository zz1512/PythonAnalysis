#!/usr/bin/env python3
"""D5: run-level trajectory subsequent memory (run3-run6).

参考 ``.trae/specs/expand-dm-narrative/spec.md`` §4.5。

目标
----
把 a4 的 stage-level (pre / post / retrieval) 轨迹细化到 run 级：在 learning
(run3 / run4) 与 post (run5 / run6) 内分别按 metadata.run 拆分，得到每
(subject, roi, run ∈ {3, 4, 5, 6}, condition_item_id) 的 pair_similarity（与
retrieval_pair_similarity 同公式 Fisher-z(Pearson)），然后跑 condition × run ×
memory 三阶交互（连续 slope + categorical 各一次），回答"YY 的 Dm 分化是否从
learning 阶段就开始累积"。

入口
----
- voxel_pattern_long.tsv（D2 入口；含 metadata.run）；
- npz pattern_root fallback：每 (subject, roi, run, condition_item_id) 算 pair
  similarity；都不可用时返回空表。
- item_mechanism_table.tsv 提供 memory_strict / memory_lenient / covariates。

输出沙箱: ``paper_outputs/qc/nc_converge/d5_run_level_trajectory_dm/``。
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

MODULE = "d5_run_level_trajectory_dm"
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
RUN_INDICES = (3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None, help="item_mechanism_table.tsv")
    parser.add_argument("--pattern-long-tsv", type=Path, default=None)
    parser.add_argument("--pattern-root", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pattern + run loading
# ---------------------------------------------------------------------------


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


def load_long(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    needed = {"subject", "roi", "condition_item_id", "word_index", "voxel", "value"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"voxel_pattern_long.tsv missing columns for D5: {sorted(missing)}")
    # run 列可能命名为 run / metadata_run / run_index
    run_col = next((c for c in ("run", "metadata_run", "run_index", "run_label") if c in frame.columns), None)
    if run_col is None:
        warnings.warn("voxel_pattern_long.tsv 缺少 run 列；D5 将无法拆 run3/4/5/6。")
        frame["run"] = pd.NA
    else:
        if run_col != "run":
            frame = frame.rename(columns={run_col: "run"})
    frame["run"] = pd.to_numeric(frame["run"].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce").astype("Int64")
    frame = frame[frame["run"].isin(RUN_INDICES)].copy()
    frame = build_condition_item_id(frame)
    frame = add_network_column(frame)
    frame["voxel"] = pd.to_numeric(frame["voxel"], errors="coerce").astype("Int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["voxel"])
    return frame


def compute_run_pair_similarity(long_frame: pd.DataFrame) -> pd.DataFrame:
    """每 (subject, network, roi, run, condition_item_id) 把两词 voxel pattern 做 Fisher-z(Pearson)."""
    if long_frame.empty:
        return pd.DataFrame()
    keys = ["subject", "network", "roi", "run", "condition_item_id"]
    rows: List[dict] = []
    for grp_keys, sub in long_frame.groupby(keys, dropna=False, sort=False):
        words = sub.groupby("word_index", dropna=False)
        word_arrs: List[np.ndarray] = []
        for _wi, ws in words:
            ws_sorted = ws.sort_values("voxel")
            arr = ws_sorted["value"].to_numpy(dtype=float)
            word_arrs.append(arr)
        if len(word_arrs) < 2:
            continue
        if word_arrs[0].size != word_arrs[1].size or word_arrs[0].size == 0:
            continue
        r = _safe_corr(word_arrs[0], word_arrs[1])
        rec = dict(zip(keys, grp_keys))
        rec["pair_similarity_raw"] = _fisher_z(r)
        rec["n_voxels"] = int(word_arrs[0].size)
        rows.append(rec)
    item = pd.DataFrame(rows)
    if item.empty:
        return item
    # 与 retrieval_pair_similarity 同口径：在 (network, run) 内做 z
    item["pair_similarity_z"] = item.groupby(["network", "run"], dropna=False)["pair_similarity_raw"].transform(zscore)
    return item


def compute_post_run_pair_similarity_from_nifti(
    pattern_root: Path,
    roi_manifest: Path,
    *,
    max_subjects: int | None = None,
) -> pd.DataFrame:
    """Stream post run5/run6 pair similarity from pattern_root without writing voxel TSV."""
    try:
        from d2_voxel_trajectory_alignment import _load_4d, _load_mask, _load_roi_manifest, _metadata_for_stage
    except Exception as exc:
        warnings.warn(f"Cannot import D2 NIfTI helpers: {exc}")
        return pd.DataFrame()

    try:
        roi_table = _load_roi_manifest(roi_manifest, ("meta_metaphor", "meta_spatial"))
    except Exception as exc:
        warnings.warn(f"Cannot load ROI manifest for D5: {exc}")
        return pd.DataFrame()

    mask_cache = {}
    for roi_row in roi_table.itertuples(index=False):
        try:
            mask_cache[str(roi_row.roi_name)] = _load_mask(Path(str(roi_row.mask_path)))
        except Exception as exc:
            warnings.warn(f"Cannot load mask {roi_row.mask_path}: {exc}")

    subject_dirs = sorted(p for p in pattern_root.glob("sub-*") if p.is_dir())
    if max_subjects is not None and max_subjects > 0:
        subject_dirs = subject_dirs[: int(max_subjects)]

    rows: List[dict] = []
    for subject_dir in subject_dirs:
        for condition in ("yy", "kj"):
            image_path = subject_dir / f"post_{condition}.nii.gz"
            meta_path = subject_dir / f"post_{condition}_metadata.tsv"
            if not image_path.exists() or not meta_path.exists():
                continue
            try:
                meta = _metadata_for_stage(subject_dir, "post", condition)
                data = _load_4d(image_path)
            except Exception as exc:
                warnings.warn(f"Cannot load post pattern for {subject_dir.name}/{condition}: {exc}")
                continue
            if data.shape[3] != len(meta):
                warnings.warn(f"volume_metadata_mismatch: {image_path} {data.shape[3]} vs {len(meta)}")
                continue
            meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
            meta = meta[meta["run"].isin((5, 6)) & meta["word_index"].isin((0, 1))].copy()
            if meta.empty:
                continue
            for roi_row in roi_table.itertuples(index=False):
                roi = str(roi_row.roi_name)
                mask = mask_cache.get(roi)
                if mask is None or data.shape[:3] != mask.shape:
                    continue
                samples = data[mask, :].T.astype(float, copy=False)
                for (run, cid), sub in meta.groupby(["run", "condition_item_id"], dropna=False, sort=False):
                    word_rows = sub.sort_values("word_index")
                    if set(pd.to_numeric(word_rows["word_index"], errors="coerce").dropna().astype(int)) != {0, 1}:
                        continue
                    arrs = [samples[int(row_index)] for row_index in word_rows.index[:2]]
                    if len(arrs) < 2 or arrs[0].size != arrs[1].size:
                        continue
                    rows.append({
                        "subject": subject_dir.name,
                        "network": getattr(roi_row, "network", None),
                        "roi": roi,
                        "run": int(run),
                        "condition": condition,
                        "condition_item_id": str(cid),
                        "pair_similarity_raw": _fisher_z(_safe_corr(arrs[0], arrs[1])),
                        "n_voxels": int(arrs[0].size),
                    })
    item = pd.DataFrame(rows)
    if item.empty:
        return item
    item = add_network_column(item)
    item["pair_similarity_z"] = item.groupby(["network", "run"], dropna=False)["pair_similarity_raw"].transform(zscore)
    return item


# ---------------------------------------------------------------------------
# Merge memory + condition
# ---------------------------------------------------------------------------


def merge_item_memory(item: pd.DataFrame, item_table: pd.DataFrame) -> pd.DataFrame:
    if item.empty or item_table.empty:
        return item
    keep = [c for c in [
        "subject", "condition_item_id",
        "memory",
        "memory_strict", "memory_lenient", "memory_prop", "memory_score", "memory_successes",
        *COVARIATES,
    ] if c in item_table.columns]
    if not keep:
        return item
    meta = item_table[keep].drop_duplicates(subset=["subject", "condition_item_id"])
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
    out = item.merge(meta, on=["subject", "condition_item_id"], how="left")
    if "condition" in out.columns:
        out["condition"] = out["condition"].astype(str).str.lower()
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_run_models(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    covs = [c for c in COVARIATES if c in item.columns and pd.to_numeric(item[c], errors="coerce").notna().sum() >= 10]
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    for outcome in ("memory_strict", "memory_lenient"):
        if outcome not in item.columns:
            continue
        for network, sub in item.groupby("network", dropna=False):
            need = ["pair_similarity_z", outcome, "condition", "run", *covs]
            model_data = sub.dropna(subset=[c for c in need if c in sub.columns]).copy()
            if len(model_data) < 50:
                rows.append(pd.DataFrame([{
                    "network": network, "outcome": outcome,
                    "model_kind": "linear_run_slope", "term": "__model__",
                    "status": "too_few_rows", "n_obs": int(len(model_data)),
                }]))
                continue
            model_data["run_index"] = pd.to_numeric(model_data["run"], errors="coerce")
            # 1) 连续 run_index 线性 slope
            formula_lin = (
                f"pair_similarity_z ~ C(condition, Treatment('kj')) * run_index * {outcome}"
                f"{cov_expr}"
            )
            res_lin = fit_formula(model_data, formula_lin, family="gaussian")
            res_lin.insert(0, "network", network)
            res_lin.insert(1, "outcome", outcome)
            res_lin.insert(2, "model_kind", "linear_run_slope")
            rows.append(res_lin)
            # 2) categorical 4 levels
            run_ref = int(pd.to_numeric(model_data["run"], errors="coerce").dropna().min())
            formula_cat = (
                f"pair_similarity_z ~ C(condition, Treatment('kj')) * C(run, Treatment({run_ref})) * {outcome}"
                f"{cov_expr}"
            )
            res_cat = fit_formula(model_data, formula_cat, family="gaussian")
            res_cat.insert(0, "network", network)
            res_cat.insert(1, "outcome", outcome)
            res_cat.insert(2, "model_kind", "categorical_run")
            rows.append(res_cat)
    if not rows:
        return pd.DataFrame()
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False))


def descriptives(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty:
        return pd.DataFrame()
    cols = [c for c in ["network", "condition", "run"] if c in item.columns]
    if not cols:
        return pd.DataFrame()
    return item.groupby(cols, dropna=False, as_index=False).agg(
        n=("pair_similarity_raw", "count"),
        mean_raw=("pair_similarity_raw", "mean"),
        sd_raw=("pair_similarity_raw", "std"),
        mean_z=("pair_similarity_z", "mean"),
        n_subjects=("subject", "nunique") if "subject" in item.columns else ("pair_similarity_raw", "count"),
    )


def build_manifest(args: argparse.Namespace, item: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"key": "input", "value": str(args.input) if args.input else "default"},
        {"key": "pattern_long_tsv", "value": str(args.pattern_long_tsv) if args.pattern_long_tsv else "default"},
        {"key": "pattern_source", "value": source_tag},
        {"key": "n_pair_rows", "value": int(len(item))},
        {"key": "n_subjects", "value": int(item["subject"].nunique()) if "subject" in item.columns and not item.empty else 0},
        {"key": "runs_present", "value": ",".join(sorted(item["run"].dropna().astype(str).unique())) if "run" in item.columns and not item.empty else ""},
    ])


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    item_path = Path(args.input or cfg.base_dir / DEFAULT_INPUT)
    long_path = Path(args.pattern_long_tsv or cfg.base_dir / DEFAULT_PATTERN_LONG)
    pattern_root = Path(args.pattern_root or cfg.base_dir / DEFAULT_PATTERN_ROOT)
    roi_manifest = Path(args.roi_manifest or cfg.base_dir / DEFAULT_ROI_MANIFEST)

    if item_path.exists():
        item_table = build_condition_item_id(read_table(item_path))
    else:
        if not args.allow_empty:
            raise FileNotFoundError(f"item_table not found: {item_path}")
        warnings.warn(f"item_table missing: {item_path}")
        item_table = pd.DataFrame()

    source_tag = "none"
    if long_path.exists():
        try:
            long_frame = load_long(long_path)
            source_tag = "long_tsv"
        except Exception as exc:
            warnings.warn(f"Failed to load voxel_pattern_long.tsv: {exc}")
            long_frame = pd.DataFrame()
    else:
        warnings.warn(f"voxel_pattern_long.tsv missing: {long_path}; trying direct post NIfTI stream.")
        long_frame = pd.DataFrame()

    if not long_frame.empty:
        pair_item = compute_run_pair_similarity(long_frame)
    elif pattern_root.exists():
        pair_item = compute_post_run_pair_similarity_from_nifti(
            pattern_root,
            roi_manifest,
            max_subjects=args.max_subjects,
        )
        source_tag = "post_nifti_stream" if not pair_item.empty else "none"
    else:
        if not args.allow_empty:
            raise FileNotFoundError(f"No D5 pattern input found: {long_path} or {pattern_root}")
        pair_item = pd.DataFrame()
    pair_item = merge_item_memory(pair_item, item_table)
    models = fit_run_models(pair_item)
    desc = descriptives(pair_item)
    mani = build_manifest(args, pair_item, source_tag)
    write_outputs(
        cfg,
        MODULE,
        {
            "run_level_trajectory_long.tsv": pair_item if not pair_item.empty else pd.DataFrame(columns=["subject", "network", "roi", "run", "condition_item_id", "pair_similarity_raw", "pair_similarity_z"]),
            "run_level_models.tsv": models if not models.empty else pd.DataFrame(columns=["network", "outcome", "model_kind", "term", "status"]),
            "run_level_descriptives.tsv": desc if not desc.empty else pd.DataFrame(columns=["network", "condition", "run", "n"]),
            "manifest.tsv": mani,
        },
    )


if __name__ == "__main__":
    main()
