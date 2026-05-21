#!/usr/bin/env python3
"""voxel_pattern_export: 从 stack_patterns 4D + meta ROI mask 导出 voxel_pattern_long.tsv.

参考 ``.trae/specs/expand-dm-narrative/spec.md`` §3.1 与 §4.6。

目标
----
解锁 d2 / d3 / d5 共用的 voxel 入口。当前 D2 默认读取
``paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv``，
但仓库中缺少该 long-tsv 的生成脚本。本 helper 把
[stack_patterns.py](../representation_analysis/stack_patterns.py) 写出的
``pattern_root/sub-XX/{pre,post,learn,retrieval}_{yy,kj}.nii.gz`` + ``*_metadata.tsv``
按 meta ROI mask 提取并 pivot 为 long 格式。

唯一跨 nc_converge 沙箱的写入白名单（spec §3.1）：
``paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv``
仅在文件不存在时写入；存在则 abort 不覆盖。

输出沙箱: ``paper_outputs/qc/nc_converge/voxel_pattern_export/`` 仅落 manifest。

不重新跑 LSS / 不重新生成 4D pattern; 不修改 stack_patterns / encoding_retrieval_similarity 等任一上游脚本。
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from shared_nc import (
    add_common_args,
    add_network_column,
    build_condition_item_id,
    default_config,
    read_table,
    write_outputs,
)

MODULE = "voxel_pattern_export"
DEFAULT_PATTERN_ROOT_REL = Path("pattern_root")
DEFAULT_ITEM_TABLE_REL = Path("paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_MASK_ROOT_REL = Path("roi_library/masks")
DEFAULT_TARGET_LONG_REL = Path("paper_outputs/qc/learning_post_memory_prediction/voxel_pattern_long.tsv")
STAGES = ("pre", "post", "learn", "retrieval")
CONDITIONS = ("yy", "kj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pattern-root", type=Path, default=None, help="stack_patterns 输出根目录（含 sub-XX/{stage}_{cond}.nii.gz 与 metadata）")
    parser.add_argument("--item-table", type=Path, default=None, help="item_mechanism_table.tsv（提供 condition_item_id / pair word index 映射）")
    parser.add_argument("--mask-root", type=Path, default=None, help="meta ROI mask 根目录（roi_library/meta_sources/）")
    parser.add_argument(
        "--target-long-tsv",
        type=Path,
        default=None,
        help="目标 long-tsv 路径；存在则 abort（spec §3.1 白名单唯一例外）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def _candidate_metadata_paths(pattern_root: Path) -> List[Path]:
    paths: List[Path] = []
    if not pattern_root.exists():
        return paths
    for subject_dir in sorted(p for p in pattern_root.iterdir() if p.is_dir()):
        for stage in STAGES:
            for cond in CONDITIONS:
                meta = subject_dir / f"{stage}_{cond}_metadata.tsv"
                if meta.exists():
                    paths.append(meta)
    return paths


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


def _condition_item_id(condition: object, pair_id: object) -> str:
    cond = re.sub(r"[^a-z]+", "", str(condition).strip().lower())
    pair = _normal_pair_id(pair_id)
    return f"{cond}_{pair}" if cond and pair else ""


def _extract_label_pair_id(frame: pd.DataFrame) -> pd.Series:
    for col in ("word_label", "unique_label", "real_word", "trial_id"):
        if col in frame.columns:
            extracted = frame[col].astype(str).str.extract(r"(\d+)", expand=False).map(_normal_pair_id)
            if extracted.replace("", pd.NA).notna().any():
                return extracted
    if "pair_id" in frame.columns:
        return frame["pair_id"].map(_normal_pair_id)
    return pd.Series([""] * len(frame), index=frame.index)


def _derive_word_index(frame: pd.DataFrame) -> pd.Series:
    if "word_index" in frame.columns:
        return pd.to_numeric(frame["word_index"], errors="coerce")
    label = None
    for col in ("word_label", "unique_label"):
        if col in frame.columns:
            label = frame[col].astype(str).str.lower()
            break
    if label is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out[label.str.contains(r"(?:^|_)(?:yy|kj)?w_", regex=True, na=False)] = 0
    out[label.str.contains(r"(?:^|_)(?:yy|kj)?ew_", regex=True, na=False)] = 1
    return out


def load_metadata(pattern_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in _candidate_metadata_paths(pattern_root):
        try:
            frame = read_table(path)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to load metadata {path}: {exc}")
            continue
        frame = frame.copy()
        # subject 与 stage / condition 由路径推断（与 stack_patterns 输出对齐）
        subject = path.parent.name
        stem = path.stem.replace("_metadata", "")
        if "_" in stem:
            stage, cond = stem.split("_", 1)
        else:
            stage, cond = stem, ""
        frame["subject"] = frame["subject"].astype(str) if "subject" in frame.columns else subject
        frame["stage"] = str(stage).lower()
        frame["condition"] = str(cond).lower()
        label_pair_id = _extract_label_pair_id(frame)
        frame["condition_item_id"] = [
            _condition_item_id(cond, pair_id) for pair_id in label_pair_id
        ]
        frame["word_index"] = _derive_word_index(frame)
        frame["metadata_path"] = str(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = build_condition_item_id(out)
    return out


# ---------------------------------------------------------------------------
# Long-tsv pivoting
# ---------------------------------------------------------------------------


def _list_mask_files(mask_root: Path) -> List[Path]:
    if not mask_root.exists():
        return []
    mask_files = []
    for path in mask_root.glob("**/*.nii.gz"):
        if not path.is_file() or not path.name.startswith("meta_"):
            continue
        parts = {part.lower() for part in path.parts}
        if {"meta_metaphor", "meta_spatial"} & parts:
            mask_files.append(path)
    return sorted(mask_files)


def _try_import_nibabel():
    try:
        import nibabel as nib  # type: ignore
        return nib
    except Exception:
        return None


def pivot_voxel_long(
    metadata: pd.DataFrame,
    pattern_root: Path,
    mask_root: Path,
) -> pd.DataFrame:
    """提取 (subject, roi, stage, condition_item_id, word_index, voxel) -> value.

    在开发机 dry-run 下 metadata / pattern 通常缺失，函数直接返回空表，由调用方
    决定是否写 long-tsv。真实计算在数据机执行。
    """
    if metadata.empty:
        return pd.DataFrame(columns=[
            "subject", "network", "roi", "stage", "run", "condition_item_id", "word_index", "voxel", "value",
        ])
    nib = _try_import_nibabel()
    if nib is None:
        warnings.warn("nibabel not available; voxel_pattern_long.tsv will not be written from this run.")
        return pd.DataFrame(columns=[
            "subject", "network", "roi", "stage", "run", "condition_item_id", "word_index", "voxel", "value",
        ])
    mask_files = _list_mask_files(mask_root)
    if not mask_files:
        warnings.warn(f"No mask files under {mask_root}; abort pivot.")
        return pd.DataFrame(columns=[
            "subject", "network", "roi", "stage", "run", "condition_item_id", "word_index", "voxel", "value",
        ])

    # 数据机真实分支：枚举 (subject, stage, condition) -> 4D 文件 + metadata.run/word_index/pair
    # 对每个 ROI mask 应用，pivot 到 long。本函数实现保留稳定接口，由数据机执行。
    out_frames: List[pd.DataFrame] = []
    for (subject, stage, condition), sub in metadata.groupby(["subject", "stage", "condition"], dropna=False):
        nii_path = pattern_root / subject / f"{stage}_{condition}.nii.gz"
        if not nii_path.exists():
            continue
        try:
            img = nib.load(str(nii_path))
            data = np.asarray(img.dataobj, dtype=float)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to read {nii_path}: {exc}")
            continue
        if data.ndim != 4:
            continue
        for mask_path in mask_files:
            roi_name = mask_path.name.replace(".nii.gz", "")
            try:
                mask_img = nib.load(str(mask_path))
                mask = np.asarray(mask_img.dataobj) > 0
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Failed to load mask {mask_path}: {exc}")
                continue
            if mask.shape != data.shape[:3]:
                continue
            mask_idx = np.flatnonzero(mask.reshape(-1))
            n_vox = mask_idx.size
            if n_vox == 0:
                continue
            flat = data.reshape(-1, data.shape[-1])  # (vox*, T)
            # 对每行 metadata（trial）抽取并按 (condition_item_id, word_index) 平均
            trial_keys = sub.reset_index(drop=True)
            for col_needed in ("condition_item_id", "word_index"):
                if col_needed not in trial_keys.columns:
                    trial_keys[col_needed] = pd.NA
            if "run" not in trial_keys.columns:
                trial_keys["run"] = pd.NA
            agg: dict[tuple, list[np.ndarray]] = {}
            for row_i, row in trial_keys.iterrows():
                if row_i >= flat.shape[1]:
                    break
                cid = str(row.get("condition_item_id", ""))
                word_index = row.get("word_index", pd.NA)
                if pd.isna(word_index):
                    continue
                run_value = pd.to_numeric(pd.Series([row.get("run", pd.NA)]), errors="coerce").iloc[0]
                key = (cid, int(word_index), int(run_value) if pd.notna(run_value) else pd.NA)
                trial_vec = flat[mask_idx, row_i]
                agg.setdefault(key, []).append(trial_vec)
            if not agg:
                continue
            rec_rows: List[dict] = []
            for (cid, word_index, run_value), vecs in agg.items():
                mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
                for v_idx, v_value in enumerate(mean_vec):
                    rec_rows.append(
                        {
                            "subject": subject,
                            "roi": roi_name,
                            "stage": stage,
                            "run": run_value,
                            "condition_item_id": cid,
                            "word_index": int(word_index),
                            "voxel": int(v_idx),
                            "value": float(v_value),
                        }
                    )
            if rec_rows:
                out_frames.append(pd.DataFrame(rec_rows))
    if not out_frames:
        return pd.DataFrame(columns=[
            "subject", "network", "roi", "stage", "run", "condition_item_id", "word_index", "voxel", "value",
        ])
    long = pd.concat(out_frames, ignore_index=True, sort=False)
    long = add_network_column(long)
    return long[[
        "subject", "network", "roi", "stage", "run", "condition_item_id", "word_index", "voxel", "value",
    ]]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    metadata: pd.DataFrame,
    long: pd.DataFrame,
    args: argparse.Namespace,
    pattern_root: Path,
    mask_root: Path,
    target_long_tsv: Path,
    long_tsv_status: str,
) -> pd.DataFrame:
    rows = [
        {"key": "pattern_root", "value": str(pattern_root)},
        {"key": "mask_root", "value": str(mask_root)},
        {"key": "item_table", "value": str(args.item_table) if args.item_table else ""},
        {"key": "target_long_tsv", "value": str(target_long_tsv)},
        {"key": "target_long_tsv_status", "value": long_tsv_status},
        {"key": "n_metadata_rows", "value": int(len(metadata))},
        {"key": "n_long_rows", "value": int(len(long))},
        {"key": "n_subjects_meta", "value": int(metadata["subject"].nunique()) if "subject" in metadata.columns and not metadata.empty else 0},
        {"key": "n_subjects_long", "value": int(long["subject"].nunique()) if "subject" in long.columns and not long.empty else 0},
        {"key": "n_rois_long", "value": int(long["roi"].nunique()) if "roi" in long.columns and not long.empty else 0},
        {"key": "stages_long", "value": ",".join(sorted(long["stage"].dropna().astype(str).unique())) if "stage" in long.columns and not long.empty else ""},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    pattern_root = Path(args.pattern_root or cfg.base_dir / DEFAULT_PATTERN_ROOT_REL)
    mask_root = Path(args.mask_root or cfg.base_dir / DEFAULT_MASK_ROOT_REL)
    target_long_tsv = Path(args.target_long_tsv or cfg.base_dir / DEFAULT_TARGET_LONG_REL)

    metadata = load_metadata(pattern_root)
    long = pivot_voxel_long(metadata, pattern_root, mask_root)

    long_tsv_status = "not_attempted"
    if not long.empty:
        if target_long_tsv.exists():
            long_tsv_status = "exists_skip_no_overwrite"
            warnings.warn(f"Target long-tsv exists; refusing to overwrite: {target_long_tsv}")
        else:
            target_long_tsv.parent.mkdir(parents=True, exist_ok=True)
            long.to_csv(target_long_tsv, sep="\t", index=False)
            long_tsv_status = "written"
    else:
        if not args.allow_empty:
            warnings.warn("Empty pivot result; nothing to write.")
        long_tsv_status = "empty_input"

    manifest = build_manifest(metadata, long, args, pattern_root, mask_root, target_long_tsv, long_tsv_status)
    write_outputs(cfg, MODULE, {"manifest.tsv": manifest})


if __name__ == "__main__":
    main()
