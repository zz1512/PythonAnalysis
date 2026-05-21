#!/usr/bin/env python3
"""D2: voxel-level pre-post-retrieval trajectory alignment.

参考 ``.trae/specs/add-voxel-trajectory-alignment/spec.md``。

目标
----
闭合 ``result_new_meta_roi.md`` §31.5 自承的 caveat：在每个 (subject, ROI, pair)
三个 stage (pre / post / retrieval) 上取 voxel-level pattern，构造 pair 状态向量
``s = (m1 + m2) / 2`` 与 differentiation 轴 ``Δ = m1 - m2``，再计算五件套 cosine
+ 两个投影标量，回答 "KJ 与 YY 在 pre→post→retrieval 中重建方向是否相似" 的问题。

输入入口三档 fallback (按优先级)
1. ``--pattern-long-tsv`` 指向的 long 表 (subject / network / roi / stage /
   condition_item_id / word_index / voxel / value)，pivot 为 pattern matrix；
2. ``--pattern-root`` 下按 ``{stage}/sub-XX/roi-XX_pair-XX.npz`` 装载 (npz 内
   ``pattern`` 形状 ``(2, n_voxel)``，0 行 = word_index 0，1 行 = word_index 1)；
3. 都不可用 ⇒ manifest 记录 status=``no_pattern_input``，所有 cosine 留 NaN，
   留给数据机决定补哪条入口（脚本不重新提取 pattern）。

输出沙箱: ``paper_outputs/qc/nc_converge/d2_voxel_trajectory_alignment/``。

不修改 R3 / nc_converge 既有脚本与既有输出，沿用 ``shared_nc.safe_output_path``。
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
)

MODULE = "d2_voxel_trajectory_alignment"
DEFAULT_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
DEFAULT_PATTERN_ROOT = Path("qc/lss_voxel_patterns")
DEFAULT_PATTERN_LONG = Path("qc/learning_post_memory_prediction/voxel_pattern_long.tsv")
DEFAULT_ROI_MANIFEST = Path("roi_library/manifest.tsv")
DEFAULT_ROI_SETS = ("meta_metaphor", "meta_spatial")
COVARIATES = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]
STAGES = ("pre", "post", "retrieval")

COSINE_OUTCOMES = (
    "return_to_pre_cos",
    "continued_diff_cos",
    "bend_cos",
    "pre_retrieval_axis_cos",
    "differentiation_axis_cos",
)
PROJECTION_OUTCOMES = (
    "pre_recovery_projection",
    "post_alignment_projection",
)
GROUP_OUTCOMES = (
    "v_post_direction_agreement",
    "v_retrieval_direction_agreement",
    "differentiation_direction_agreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--input", type=Path, default=None, help="item_mechanism_table.tsv (与 d1/c1b 同源)")
    parser.add_argument("--pattern-root", type=Path, default=None)
    parser.add_argument("--pattern-long-tsv", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--roi-sets", nargs="+", default=list(DEFAULT_ROI_SETS))
    parser.add_argument("--max-subjects", type=int, default=None, help="Development/test limit for NIfTI loading.")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pattern loading (3 fallback tiers)
# ---------------------------------------------------------------------------


def _empty_manifest() -> pd.DataFrame:
    cols = [
        "subject",
        "network",
        "roi",
        "stage",
        "condition_item_id",
        "word_index",
        "pattern_path",
        "pattern_array_id",
        "n_voxels",
        "status",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def load_voxel_patterns(
    pattern_root: Path,
    pattern_long_tsv: Path,
    *,
    roi_manifest: Path | None = None,
    roi_sets: Iterable[str] = DEFAULT_ROI_SETS,
    max_subjects: int | None = None,
    allow_empty: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """返回 (manifest, pattern_cache, source_tag).

    pattern_cache: ``pattern_array_id -> np.ndarray`` (1-D voxel pattern)。
    source_tag: ``long_tsv | npz_root | none``。
    """
    if pattern_long_tsv is not None and Path(pattern_long_tsv).exists():
        manifest, cache = _load_from_long_tsv(Path(pattern_long_tsv))
        return manifest, cache, "long_tsv"
    if pattern_root is not None and Path(pattern_root).exists():
        if _looks_like_nifti_pattern_root(Path(pattern_root)):
            manifest, cache = _load_from_nifti_pattern_root(
                Path(pattern_root),
                roi_manifest=roi_manifest,
                roi_sets=roi_sets,
                max_subjects=max_subjects,
            )
            return manifest, cache, "nifti_pattern_root"
        manifest, cache = _load_from_npz_root(Path(pattern_root))
        return manifest, cache, "npz_root"
    if allow_empty:
        return _empty_manifest(), {}, "none"
    raise FileNotFoundError(
        f"No voxel pattern input found. Tried long_tsv={pattern_long_tsv}, root={pattern_root}."
    )


def _load_from_long_tsv(path: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    frame = read_table(path)
    needed = {"subject", "roi", "stage", "condition_item_id", "word_index", "voxel", "value"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"voxel_pattern_long.tsv missing columns: {sorted(missing)}")
    frame = build_condition_item_id(frame)
    frame = add_network_column(frame)
    frame["stage"] = frame["stage"].astype(str).str.lower()
    frame = frame[frame["stage"].isin(STAGES)].copy()
    frame["word_index"] = pd.to_numeric(frame["word_index"], errors="coerce").astype("Int64")
    frame["voxel"] = pd.to_numeric(frame["voxel"], errors="coerce").astype("Int64")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["word_index", "voxel"])
    keys = ["subject", "network", "roi", "stage", "condition_item_id", "word_index"]
    cache: Dict[str, np.ndarray] = {}
    rows: List[dict] = []
    for grp_keys, sub in frame.groupby(keys, dropna=False, sort=False):
        sub_sorted = sub.sort_values("voxel")
        arr = sub_sorted["value"].to_numpy(dtype=float)
        if arr.size == 0:
            status = "empty_pattern"
        elif not np.isfinite(arr).all():
            status = "nonfinite_voxels"
        else:
            status = "ok"
        array_id = "::".join(str(x) for x in grp_keys)
        cache[array_id] = arr
        record = dict(zip(keys, grp_keys))
        record.update(
            {
                "pattern_path": str(path),
                "pattern_array_id": array_id,
                "n_voxels": int(arr.size),
                "status": status,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows), cache


def _looks_like_nifti_pattern_root(root: Path) -> bool:
    return any(root.glob("sub-*/*_metadata.tsv")) and any(root.glob("sub-*/*.nii.gz"))


def _load_roi_manifest(roi_manifest: Path, roi_sets: Iterable[str]) -> pd.DataFrame:
    manifest = read_table(roi_manifest).copy()
    required = {"roi_name", "roi_set", "mask_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"ROI manifest missing columns: {sorted(missing)}")
    selected_sets = {str(item).strip().lower() for item in roi_sets}
    manifest = manifest[manifest["roi_set"].astype(str).str.lower().isin(selected_sets)].copy()
    if "include_in_rsa" in manifest.columns:
        manifest = manifest[manifest["include_in_rsa"].astype(str).str.lower().isin({"1", "true", "yes", "y", "t"})]
    manifest["mask_path"] = manifest["mask_path"].map(lambda item: Path(str(item)))
    manifest = manifest[manifest["mask_path"].map(lambda item: item.exists())].copy()
    if manifest.empty:
        raise FileNotFoundError(f"No ROI masks selected from {roi_manifest} for roi_sets={sorted(selected_sets)}")
    return manifest.reset_index(drop=True)


def _normal_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        value_f = float(text)
    except Exception:
        return text
    if math.isnan(value_f):
        return ""
    return str(int(value_f)) if value_f.is_integer() else str(value_f)


def _extract_original_id(value: object) -> str:
    text = str(value)
    match = re.search(r"(\d+)", text)
    return _normal_pair_id(match.group(1)) if match else ""


def _condition_item_id(condition: object, pair_id: object) -> str:
    cond = re.sub(r"[^a-z]+", "", str(condition).strip().lower())
    pair = _normal_pair_id(pair_id)
    return f"{cond}_{pair}" if cond and pair else ""


def _load_mask(mask_path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(mask_path)).get_fdata()) > 0


def _load_4d(image_path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asarray(nib.load(str(image_path)).get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D/4D image: {image_path}")
    return data


def _metadata_for_stage(subject_dir: Path, stage: str, condition: str) -> pd.DataFrame:
    meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
    frame = read_table(meta_path).copy()
    if "pair_id" not in frame.columns:
        raise ValueError(f"Missing pair_id in {meta_path}")
    if "condition" not in frame.columns:
        frame["condition"] = condition
    frame["condition"] = condition
    frame["stage"] = stage
    frame["subject"] = frame.get("subject", pd.Series([subject_dir.name] * len(frame))).astype(str)
    if "pic_num" in frame.columns:
        original_id = pd.to_numeric(frame["pic_num"], errors="coerce").map(_normal_pair_id)
    elif "word_label" in frame.columns:
        original_id = frame["word_label"].map(_extract_original_id)
    else:
        original_id = frame["pair_id"].map(_normal_pair_id)
    frame["original_pair_id_for_join"] = original_id
    frame["condition_item_id"] = [_condition_item_id(condition, pair) for pair in original_id]
    sort_cols = [c for c in ["condition_item_id", "word_label", "real_word", "trial_index"] if c in frame.columns]
    frame = frame.sort_values(sort_cols).reset_index(drop=True)
    frame["word_index"] = frame.groupby("condition_item_id", dropna=False).cumcount()
    return frame


def _load_from_nifti_pattern_root(
    root: Path,
    *,
    roi_manifest: Path | None,
    roi_sets: Iterable[str],
    max_subjects: int | None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if roi_manifest is None:
        raise ValueError("NIfTI pattern loading requires --roi-manifest")
    roi_table = _load_roi_manifest(Path(roi_manifest), roi_sets)
    mask_cache: dict[str, np.ndarray] = {}
    for row in roi_table.itertuples(index=False):
        mask_cache[str(row.roi_name)] = _load_mask(Path(str(row.mask_path)))

    subject_dirs = sorted(p for p in root.glob("sub-*") if p.is_dir())
    if max_subjects is not None and max_subjects > 0:
        subject_dirs = subject_dirs[: int(max_subjects)]

    rows: List[dict] = []
    cache: Dict[str, np.ndarray] = {}
    for subject_dir in subject_dirs:
        for stage in STAGES:
            for condition in ("yy", "kj"):
                image_path = subject_dir / f"{stage}_{condition}.nii.gz"
                meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
                if not image_path.exists() or not meta_path.exists():
                    rows.append(
                        {
                            "subject": subject_dir.name,
                            "network": None,
                            "roi": "",
                            "stage": stage,
                            "condition_item_id": "",
                            "word_index": -1,
                            "pattern_path": str(image_path),
                            "pattern_array_id": "",
                            "n_voxels": 0,
                            "status": "missing_stage_file",
                        }
                    )
                    continue
                try:
                    meta = _metadata_for_stage(subject_dir, stage, condition)
                    data = _load_4d(image_path)
                    if data.shape[3] != len(meta):
                        raise ValueError(f"volume_metadata_mismatch: {data.shape[3]} vs {len(meta)}")
                except Exception as exc:
                    rows.append(
                        {
                            "subject": subject_dir.name,
                            "network": None,
                            "roi": "",
                            "stage": stage,
                            "condition_item_id": "",
                            "word_index": -1,
                            "pattern_path": str(image_path),
                            "pattern_array_id": "",
                            "n_voxels": 0,
                            "status": f"stage_load_failed: {exc}",
                        }
                    )
                    continue
                for roi_row in roi_table.itertuples(index=False):
                    roi = str(roi_row.roi_name)
                    mask = mask_cache[roi]
                    if data.shape[:3] != mask.shape:
                        rows.append(
                            {
                                "subject": subject_dir.name,
                                "network": None,
                                "roi": roi,
                                "stage": stage,
                                "condition_item_id": "",
                                "word_index": -1,
                                "pattern_path": str(image_path),
                                "pattern_array_id": "",
                                "n_voxels": 0,
                                "status": "image_mask_shape_mismatch",
                            }
                        )
                        continue
                    samples = data[mask, :].T.astype(float, copy=False)
                    for row_index, meta_row in meta.iterrows():
                        word_index = int(meta_row["word_index"])
                        if word_index not in (0, 1):
                            continue
                        pair = str(meta_row["condition_item_id"])
                        array_id = f"{subject_dir.name}::{roi}::{stage}::{pair}::{word_index}"
                        arr = samples[int(row_index)]
                        cache[array_id] = arr
                        status = "ok" if arr.size and np.isfinite(arr).all() else "nonfinite_voxels"
                        rows.append(
                            {
                                "subject": subject_dir.name,
                                "network": None,
                                "roi": roi,
                                "stage": stage,
                                "condition_item_id": pair,
                                "word_index": word_index,
                                "pattern_path": str(image_path),
                                "pattern_array_id": array_id,
                                "n_voxels": int(arr.size),
                                "status": status,
                            }
                        )
    manifest = pd.DataFrame(rows) if rows else _empty_manifest()
    if not manifest.empty:
        manifest = add_network_column(manifest)
    return manifest, cache


def _load_from_npz_root(root: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[dict] = []
    cache: Dict[str, np.ndarray] = {}
    for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stage = stage_dir.name.lower()
        if stage not in STAGES:
            continue
        for subject_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            subject = subject_dir.name
            for npz_path in sorted(subject_dir.glob("*.npz")):
                stem = npz_path.stem
                roi, pair = _parse_roi_pair_from_stem(stem)
                if roi is None:
                    continue
                try:
                    with np.load(npz_path) as data:
                        if "pattern" not in data.files:
                            status = "missing_pattern_key"
                            arr2 = np.zeros((0,))
                            for word_index in (0, 1):
                                array_id = f"{subject}::{roi}::{stage}::{pair}::{word_index}"
                                cache[array_id] = arr2
                                rows.append(
                                    {
                                        "subject": subject,
                                        "network": None,
                                        "roi": roi,
                                        "stage": stage,
                                        "condition_item_id": pair,
                                        "word_index": word_index,
                                        "pattern_path": str(npz_path),
                                        "pattern_array_id": array_id,
                                        "n_voxels": 0,
                                        "status": status,
                                    }
                                )
                            continue
                        pattern = np.asarray(data["pattern"], dtype=float)
                except Exception as exc:  # pragma: no cover - runtime hardening
                    rows.append(
                        {
                            "subject": subject,
                            "network": None,
                            "roi": roi,
                            "stage": stage,
                            "condition_item_id": pair,
                            "word_index": -1,
                            "pattern_path": str(npz_path),
                            "pattern_array_id": "",
                            "n_voxels": 0,
                            "status": f"npz_load_failed: {exc}",
                        }
                    )
                    continue
                if pattern.ndim != 2 or pattern.shape[0] != 2:
                    status = f"bad_shape: {pattern.shape}"
                    array_id = f"{subject}::{roi}::{stage}::{pair}::na"
                    cache[array_id] = np.zeros((0,))
                    rows.append(
                        {
                            "subject": subject,
                            "network": None,
                            "roi": roi,
                            "stage": stage,
                            "condition_item_id": pair,
                            "word_index": -1,
                            "pattern_path": str(npz_path),
                            "pattern_array_id": array_id,
                            "n_voxels": 0,
                            "status": status,
                        }
                    )
                    continue
                for word_index in (0, 1):
                    arr = pattern[word_index].astype(float)
                    array_id = f"{subject}::{roi}::{stage}::{pair}::{word_index}"
                    cache[array_id] = arr
                    if not np.isfinite(arr).all():
                        status = "nonfinite_voxels"
                    else:
                        status = "ok"
                    rows.append(
                        {
                            "subject": subject,
                            "network": None,
                            "roi": roi,
                            "stage": stage,
                            "condition_item_id": pair,
                            "word_index": word_index,
                            "pattern_path": str(npz_path),
                            "pattern_array_id": array_id,
                            "n_voxels": int(arr.size),
                            "status": status,
                        }
                    )
    manifest = pd.DataFrame(rows) if rows else _empty_manifest()
    if not manifest.empty:
        manifest = add_network_column(manifest)
    return manifest, cache


def _parse_roi_pair_from_stem(stem: str) -> Tuple[str | None, str | None]:
    parts = stem.split("_")
    roi = None
    pair = None
    for chunk in parts:
        if chunk.startswith("roi-"):
            roi = chunk.split("-", 1)[1]
        elif chunk.startswith("pair-"):
            pair = chunk.split("-", 1)[1]
    return roi, pair


# ---------------------------------------------------------------------------
# Pair state vectors and cosines
# ---------------------------------------------------------------------------


def pair_state_vectors(
    manifest: pd.DataFrame,
    cache: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """每 (subject, roi, stage, pair) 输出 ``s`` 与 ``Δ`` 向量 id 与 voxel 数."""
    if manifest.empty:
        return pd.DataFrame()
    keys = ["subject", "network", "roi", "stage", "condition_item_id"]
    out_rows: List[dict] = []
    for grp_keys, sub in manifest.groupby(keys, dropna=False, sort=False):
        word_rows = sub.sort_values("word_index")
        if len(word_rows) < 2:
            out_rows.append(
                dict(zip(keys, grp_keys), s_id="", delta_id="", n_voxels=0, status="missing_word")
            )
            continue
        m1_id = word_rows.iloc[0]["pattern_array_id"]
        m2_id = word_rows.iloc[1]["pattern_array_id"]
        m1 = cache.get(m1_id, np.zeros((0,)))
        m2 = cache.get(m2_id, np.zeros((0,)))
        if m1.size == 0 or m2.size == 0 or m1.size != m2.size:
            out_rows.append(
                dict(zip(keys, grp_keys), s_id="", delta_id="", n_voxels=0, status="shape_mismatch")
            )
            continue
        s = (m1 + m2) / 2.0
        delta = m1 - m2
        s_id = f"S::{m1_id}::{m2_id}"
        delta_id = f"D::{m1_id}::{m2_id}"
        cache[s_id] = s
        cache[delta_id] = delta
        out_rows.append(
            dict(zip(keys, grp_keys), s_id=s_id, delta_id=delta_id, n_voxels=int(s.size), status="ok")
        )
    return pd.DataFrame(out_rows)


def _vector(cache: Dict[str, np.ndarray], key: str) -> np.ndarray | None:
    arr = cache.get(key)
    if arr is None or arr.size == 0:
        return None
    return arr


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if not (math.isfinite(na) and math.isfinite(nb)) or na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _projection(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    if not math.isfinite(nb) or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / nb)


def compute_cosines(
    states: pd.DataFrame,
    cache: Dict[str, np.ndarray],
    item_meta: pd.DataFrame,
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    pivot_keys = ["subject", "network", "roi", "condition_item_id"]
    rows: List[dict] = []
    for grp_keys, sub in states.groupby(pivot_keys, dropna=False, sort=False):
        s_ids = {r["stage"]: r["s_id"] for _, r in sub.iterrows()}
        d_ids = {r["stage"]: r["delta_id"] for _, r in sub.iterrows()}
        statuses = {r["stage"]: r["status"] for _, r in sub.iterrows()}
        if not all(stage in s_ids for stage in STAGES):
            rows.append(
                dict(
                    zip(pivot_keys, grp_keys),
                    status="missing_stage",
                    **{name: float("nan") for name in COSINE_OUTCOMES + PROJECTION_OUTCOMES},
                )
            )
            continue
        if any(statuses[stage] != "ok" for stage in STAGES):
            rows.append(
                dict(
                    zip(pivot_keys, grp_keys),
                    status="upstream_" + ",".join(sorted({statuses[s] for s in STAGES if statuses[s] != "ok"})),
                    **{name: float("nan") for name in COSINE_OUTCOMES + PROJECTION_OUTCOMES},
                )
            )
            continue
        s_pre = _vector(cache, s_ids["pre"])
        s_post = _vector(cache, s_ids["post"])
        s_ret = _vector(cache, s_ids["retrieval"])
        d_pre = _vector(cache, d_ids["pre"])
        d_post = _vector(cache, d_ids["post"])
        d_ret = _vector(cache, d_ids["retrieval"])
        if any(v is None for v in (s_pre, s_post, s_ret, d_pre, d_post, d_ret)):
            rows.append(
                dict(
                    zip(pivot_keys, grp_keys),
                    status="missing_vector",
                    **{name: float("nan") for name in COSINE_OUTCOMES + PROJECTION_OUTCOMES},
                )
            )
            continue
        sizes = {v.size for v in (s_pre, s_post, s_ret, d_pre, d_post, d_ret)}
        if len(sizes) != 1:
            rows.append(
                dict(
                    zip(pivot_keys, grp_keys),
                    status="shape_mismatch",
                    **{name: float("nan") for name in COSINE_OUTCOMES + PROJECTION_OUTCOMES},
                )
            )
            continue
        v_post = s_post - s_pre
        v_retrieval = s_ret - s_post
        v_pre_retrieval = s_ret - s_pre
        u_post = d_post - d_pre
        u_retrieval = d_ret - d_post
        rec = dict(zip(pivot_keys, grp_keys))
        rec["status"] = "ok"
        rec["return_to_pre_cos"] = _cosine(v_retrieval, -v_post)
        rec["continued_diff_cos"] = _cosine(v_retrieval, v_post)
        rec["bend_cos"] = _cosine(v_post, v_retrieval)
        rec["pre_retrieval_axis_cos"] = _cosine(v_pre_retrieval, v_post)
        rec["differentiation_axis_cos"] = _cosine(u_retrieval, u_post)
        rec["pre_recovery_projection"] = _projection(v_retrieval, s_pre - s_post)
        rec["post_alignment_projection"] = _projection(v_pre_retrieval, v_post)
        rec["v_post_norm"] = float(np.linalg.norm(v_post))
        rec["v_retrieval_norm"] = float(np.linalg.norm(v_retrieval))
        rec["u_retrieval_norm"] = float(np.linalg.norm(u_retrieval))
        rec["v_post_id"] = "::".join([str(grp_keys[0]), str(grp_keys[1]), str(grp_keys[2]), str(grp_keys[3]), "v_post"])
        rec["v_retrieval_id"] = "::".join([str(grp_keys[0]), str(grp_keys[1]), str(grp_keys[2]), str(grp_keys[3]), "v_retrieval"])
        rec["u_retrieval_id"] = "::".join([str(grp_keys[0]), str(grp_keys[1]), str(grp_keys[2]), str(grp_keys[3]), "u_retrieval"])
        cache[rec["v_post_id"]] = v_post
        cache[rec["v_retrieval_id"]] = v_retrieval
        cache[rec["u_retrieval_id"]] = u_retrieval
        rows.append(rec)
    item_table = pd.DataFrame(rows)
    if item_table.empty or item_meta.empty:
        return item_table
    join_cols = [c for c in ["subject", "condition_item_id"] if c in item_meta.columns]
    if not join_cols:
        return item_table
    keep_meta_cols = [
        *join_cols,
        *[c for c in ["condition", "memory_strict", *COVARIATES] if c in item_meta.columns],
    ]
    item_table = item_table.merge(
        item_meta[keep_meta_cols].drop_duplicates(subset=join_cols),
        on=join_cols,
        how="left",
    )
    return item_table


# ---------------------------------------------------------------------------
# Item table → network aggregation → mixed-effects models
# ---------------------------------------------------------------------------


def _aggregate_to_network(item_table: pd.DataFrame) -> pd.DataFrame:
    if item_table.empty:
        return item_table
    if "network" in item_table.columns:
        item_table = item_table[item_table["network"].notna()].copy()
    keep_cols = [
        c
        for c in [
            "subject",
            "network",
            "condition",
            "condition_item_id",
            "memory_strict",
            *COVARIATES,
            *COSINE_OUTCOMES,
            *PROJECTION_OUTCOMES,
        ]
        if c in item_table.columns
    ]
    if not keep_cols:
        return pd.DataFrame()
    grouped = item_table[keep_cols].copy()
    keys = [c for c in ["subject", "network", "condition", "condition_item_id"] if c in grouped.columns]
    if not keys:
        return grouped
    numeric_cols = [c for c in grouped.columns if c not in keys]
    aggregated = grouped.groupby(keys, dropna=False, as_index=False)[numeric_cols].mean()
    return aggregated


def _usable_columns(frame: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for col in columns:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10:
            out.append(col)
    return out


def fit_models(network_table: pd.DataFrame) -> pd.DataFrame:
    if network_table.empty:
        return pd.DataFrame()
    needed_base = {"subject", "condition_item_id", "condition", "network"}
    if not needed_base.issubset(network_table.columns):
        return pd.DataFrame(
            [
                {
                    "term": "__model__",
                    "status": f"missing_columns: {sorted(needed_base - set(network_table.columns))}",
                }
            ]
        )
    rows: List[pd.DataFrame] = []
    outcomes = list(COSINE_OUTCOMES) + list(PROJECTION_OUTCOMES)
    for network, sub in network_table.groupby("network", dropna=False, sort=False):
        covs = _usable_columns(sub, COVARIATES)
        cov_expr = (" + " + " + ".join(covs)) if covs else ""
        has_memory = "memory_strict" in sub.columns and sub["memory_strict"].notna().any()
        for outcome in outcomes:
            if outcome not in sub.columns:
                continue
            keep = [outcome, *covs]
            if has_memory:
                keep.append("memory_strict")
            model_data = sub.dropna(subset=keep).copy()
            if len(model_data) < 30:
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "network": network,
                                "outcome": outcome,
                                "mechanism_model": "trajectory_cosine_mixed",
                                "term": "__model__",
                                "status": "too_few_rows",
                                "n_obs": len(model_data),
                            }
                        ]
                    )
                )
                continue
            interaction = " * memory_strict" if has_memory else ""
            formula = (
                f"{outcome} ~ C(condition, Treatment('kj')){interaction}{cov_expr}"
            )
            res = fit_formula(model_data, formula, family="gaussian")
            res.insert(0, "network", network)
            res.insert(1, "outcome", outcome)
            res.insert(2, "mechanism_model", "trajectory_cosine_mixed")
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True, sort=False)
    table = q_for_ok_rows(table)
    return table


# ---------------------------------------------------------------------------
# Group direction agreement (subject-level KJ vs YY cosine + permutation)
# ---------------------------------------------------------------------------


def _subject_condition_means(
    item_table: pd.DataFrame,
    cache: Dict[str, np.ndarray],
    vector_id_col: str,
) -> pd.DataFrame:
    """对每 (subject, network, condition) 内 item 平均位移向量，返回新的向量 id."""
    if item_table.empty or vector_id_col not in item_table.columns:
        return pd.DataFrame()
    keys = ["subject", "network", "condition"]
    rows: List[dict] = []
    for grp_keys, sub in item_table.groupby(keys, dropna=False, sort=False):
        ids = [vid for vid in sub[vector_id_col].tolist() if isinstance(vid, str) and vid]
        arrays = [cache.get(vid) for vid in ids]
        arrays = [a for a in arrays if a is not None and a.size > 0]
        if not arrays:
            rows.append(
                dict(zip(keys, grp_keys), mean_id="", n_items=0, status="empty"),
            )
            continue
        sizes = {a.size for a in arrays}
        if len(sizes) != 1:
            rows.append(
                dict(zip(keys, grp_keys), mean_id="", n_items=len(arrays), status="shape_mismatch"),
            )
            continue
        mean_vec = np.mean(np.stack(arrays, axis=0), axis=0)
        mean_id = "MEAN::" + "::".join(str(x) for x in grp_keys) + f"::{vector_id_col}"
        cache[mean_id] = mean_vec
        rows.append(
            dict(zip(keys, grp_keys), mean_id=mean_id, n_items=int(len(arrays)), status="ok"),
        )
    return pd.DataFrame(rows)


def _within_subject_cross_condition_cos(
    means: pd.DataFrame,
    cache: Dict[str, np.ndarray],
) -> pd.DataFrame:
    if means.empty:
        return pd.DataFrame()
    keys = ["subject", "network"]
    rows: List[dict] = []
    for grp_keys, sub in means.groupby(keys, dropna=False, sort=False):
        cond_to_id = {str(r["condition"]).lower(): r["mean_id"] for _, r in sub.iterrows() if r["status"] == "ok"}
        if "kj" not in cond_to_id or "yy" not in cond_to_id:
            rows.append(dict(zip(keys, grp_keys), cosine=float("nan"), status="missing_condition"))
            continue
        a = cache.get(cond_to_id["kj"])
        b = cache.get(cond_to_id["yy"])
        if a is None or b is None or a.size == 0 or b.size != a.size:
            rows.append(dict(zip(keys, grp_keys), cosine=float("nan"), status="shape_mismatch"))
            continue
        rows.append(dict(zip(keys, grp_keys), cosine=_cosine(a, b), status="ok"))
    return pd.DataFrame(rows)


def _permutation_baseline(
    item_table: pd.DataFrame,
    cache: Dict[str, np.ndarray],
    vector_id_col: str,
    *,
    n_perm: int,
    rng_seed: int,
) -> pd.DataFrame:
    if item_table.empty or vector_id_col not in item_table.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(rng_seed)
    rows: List[dict] = []
    keys = ["subject", "network"]
    for grp_keys, sub in item_table.groupby(keys, dropna=False, sort=False):
        sub_ok = sub[sub["status"].astype(str).eq("ok") & sub[vector_id_col].astype(str).ne("")]
        if sub_ok.empty or "condition" not in sub_ok.columns:
            rows.append(dict(zip(keys, grp_keys), cos_perm_mean=float("nan"), cos_perm_lower=float("nan"), cos_perm_upper=float("nan"), p_perm=float("nan"), status="empty"))
            continue
        ids = sub_ok[vector_id_col].tolist()
        conditions = sub_ok["condition"].astype(str).str.lower().tolist()
        arrays = [cache.get(i) for i in ids]
        valid = [(c, a) for c, a in zip(conditions, arrays) if a is not None and a.size > 0]
        if not valid:
            rows.append(dict(zip(keys, grp_keys), cos_perm_mean=float("nan"), cos_perm_lower=float("nan"), cos_perm_upper=float("nan"), p_perm=float("nan"), status="empty"))
            continue
        sizes = {a.size for _, a in valid}
        if len(sizes) != 1:
            rows.append(dict(zip(keys, grp_keys), cos_perm_mean=float("nan"), cos_perm_lower=float("nan"), cos_perm_upper=float("nan"), p_perm=float("nan"), status="shape_mismatch"))
            continue
        condition_arr = np.array([c for c, _ in valid])
        stack = np.stack([a for _, a in valid], axis=0)
        if not (np.unique(condition_arr).tolist() == ["kj", "yy"] or np.unique(condition_arr).tolist() == ["yy", "kj"] or set(np.unique(condition_arr).tolist()) == {"kj", "yy"}):
            rows.append(dict(zip(keys, grp_keys), cos_perm_mean=float("nan"), cos_perm_lower=float("nan"), cos_perm_upper=float("nan"), p_perm=float("nan"), status="missing_condition"))
            continue
        observed_kj = stack[condition_arr == "kj"].mean(axis=0)
        observed_yy = stack[condition_arr == "yy"].mean(axis=0)
        observed = _cosine(observed_kj, observed_yy)
        perm_values = np.full(int(n_perm), np.nan, dtype=float)
        idx = np.arange(stack.shape[0])
        for k in range(int(n_perm)):
            perm_idx = rng.permutation(idx)
            perm_cond = condition_arr[perm_idx]
            try:
                perm_kj = stack[perm_cond == "kj"].mean(axis=0)
                perm_yy = stack[perm_cond == "yy"].mean(axis=0)
            except Exception:
                continue
            perm_values[k] = _cosine(perm_kj, perm_yy)
        valid_perm = perm_values[np.isfinite(perm_values)]
        if valid_perm.size == 0:
            rows.append(dict(zip(keys, grp_keys), observed_cos=observed, cos_perm_mean=float("nan"), cos_perm_lower=float("nan"), cos_perm_upper=float("nan"), p_perm=float("nan"), status="no_perm"))
            continue
        lower = float(np.percentile(valid_perm, 2.5))
        upper = float(np.percentile(valid_perm, 97.5))
        if math.isfinite(observed):
            extreme = np.sum(np.abs(valid_perm) >= abs(observed))
            p_perm = float((extreme + 1) / (valid_perm.size + 1))
        else:
            p_perm = float("nan")
        rows.append(
            dict(
                zip(keys, grp_keys),
                observed_cos=observed,
                cos_perm_mean=float(np.mean(valid_perm)),
                cos_perm_lower=lower,
                cos_perm_upper=upper,
                p_perm=p_perm,
                status="ok",
            )
        )
    return pd.DataFrame(rows)


def group_direction_agreement(
    item_table: pd.DataFrame,
    cache: Dict[str, np.ndarray],
    *,
    n_perm: int,
    rng_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []
    if item_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "network" in item_table.columns:
        item_table = item_table[item_table["network"].notna()].copy()
    if item_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    spec = [
        ("v_post_id", "v_post_direction_agreement"),
        ("v_retrieval_id", "v_retrieval_direction_agreement"),
        ("u_retrieval_id", "differentiation_direction_agreement"),
    ]
    for vector_col, outcome_name in spec:
        if vector_col not in item_table.columns:
            continue
        means = _subject_condition_means(item_table, cache, vector_col)
        cosines = _within_subject_cross_condition_cos(means, cache)
        perm = _permutation_baseline(
            item_table, cache, vector_col, n_perm=n_perm, rng_seed=rng_seed
        )
        if not cosines.empty:
            cosines = cosines.assign(outcome=outcome_name)
            if not perm.empty:
                cosines = cosines.merge(perm, on=["subject", "network"], how="left", suffixes=("", "_perm"))
            detail_frames.append(cosines)
        # network-level summary: one-sample t against 0
        if cosines.empty:
            continue
        for network, sub in cosines.groupby("network", dropna=False, sort=False):
            valid = sub[sub["status"].astype(str).eq("ok")]["cosine"].astype(float)
            valid = valid[np.isfinite(valid)]
            n = int(valid.size)
            if n < 3:
                summary_rows.append(
                    {
                        "network": network,
                        "outcome": outcome_name,
                        "n_subjects": n,
                        "mean_cos": float("nan"),
                        "se_cos": float("nan"),
                        "t_stat": float("nan"),
                        "p_t": float("nan"),
                        "status": "too_few_subjects",
                    }
                )
                continue
            mean = float(valid.mean())
            sd = float(valid.std(ddof=1))
            se = sd / math.sqrt(n) if n > 0 and sd > 0 else float("nan")
            t_stat = mean / se if se and math.isfinite(se) and se > 0 else float("nan")
            if math.isfinite(t_stat):
                # two-sided p with t distribution approximated via standard normal
                # to avoid scipy; for n>=3 the difference is acceptable for screening.
                p_t = float(2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))
            else:
                p_t = float("nan")
            summary_rows.append(
                {
                    "network": network,
                    "outcome": outcome_name,
                    "n_subjects": n,
                    "mean_cos": mean,
                    "se_cos": se,
                    "t_stat": t_stat,
                    "p_t": p_t,
                    "status": "ok",
                }
            )
    detail = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty and "p_t" in summary.columns:
        ok = summary["status"].astype(str).eq("ok")
        summary["q_bh"] = float("nan")
        if ok.any():
            summary.loc[ok, "q_bh"] = bh_fdr(summary.loc[ok, "p_t"])
    return detail, summary


# ---------------------------------------------------------------------------
# Descriptives & manifest
# ---------------------------------------------------------------------------


def descriptives(network_table: pd.DataFrame) -> pd.DataFrame:
    if network_table.empty:
        return pd.DataFrame([{"status": "empty_input"}])
    keys = [c for c in ["network", "condition"] if c in network_table.columns]
    if not keys:
        return pd.DataFrame([{"status": "missing_keys"}])
    rows: List[dict] = []
    for grp_keys, sub in network_table.groupby(keys, dropna=False, sort=False):
        record = dict(zip(keys, grp_keys))
        record["n"] = int(len(sub))
        if "subject" in sub.columns:
            record["n_subjects"] = int(sub["subject"].nunique())
        if "condition_item_id" in sub.columns:
            record["n_items"] = int(sub["condition_item_id"].nunique())
        for col in COSINE_OUTCOMES + PROJECTION_OUTCOMES:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                values = values[np.isfinite(values)]
                record[f"{col}_mean"] = float(values.mean()) if not values.empty else float("nan")
                record[f"{col}_sd"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def manifest(
    args: argparse.Namespace,
    raw_input_path: Path,
    selected_pattern_root: Path,
    selected_pattern_long: Path,
    selected_roi_manifest: Path,
    raw: pd.DataFrame,
    pattern_manifest: pd.DataFrame,
    item_table: pd.DataFrame,
    pattern_source: str,
) -> pd.DataFrame:
    record: dict = {
        "input_path": str(raw_input_path),
        "pattern_source": pattern_source,
        "pattern_long_tsv": str(selected_pattern_long),
        "pattern_root": str(selected_pattern_root),
        "roi_manifest": str(selected_roi_manifest),
        "roi_sets": ",".join(str(item) for item in args.roi_sets),
        "max_subjects": args.max_subjects if args.max_subjects is not None else "",
        "n_perm": int(args.n_permutations),
        "rng_seed": int(args.rng_seed),
        "input_rows": int(len(raw)),
        "pattern_manifest_rows": int(len(pattern_manifest)),
        "pattern_manifest_ok": int((pattern_manifest.get("status", pd.Series(dtype=str)) == "ok").sum()) if not pattern_manifest.empty else 0,
        "item_table_rows": int(len(item_table)),
        "item_table_ok": int((item_table.get("status", pd.Series(dtype=str)) == "ok").sum()) if not item_table.empty else 0,
    }
    if not item_table.empty and "subject" in item_table.columns:
        record["n_subjects"] = int(item_table["subject"].nunique())
    if not item_table.empty and "network" in item_table.columns:
        record["n_networks"] = int(item_table["network"].nunique())
    return pd.DataFrame([record])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    input_path = Path(args.input or cfg.paper_output_root / DEFAULT_INPUT)
    pattern_root = Path(args.pattern_root or cfg.base_dir / "pattern_root")
    pattern_long_tsv = Path(args.pattern_long_tsv or cfg.paper_output_root / DEFAULT_PATTERN_LONG)
    roi_manifest = Path(args.roi_manifest or cfg.base_dir / DEFAULT_ROI_MANIFEST)

    if input_path.exists():
        raw = read_table(input_path)
    else:
        if not args.allow_empty:
            raise FileNotFoundError(f"item_mechanism_table not found: {input_path}")
        warnings.warn(f"item_mechanism_table missing, running with empty input: {input_path}")
        raw = pd.DataFrame()

    item_meta = pd.DataFrame()
    if not raw.empty:
        item_meta = add_network_column(build_condition_item_id(raw))
        for col in [*COVARIATES, "memory_strict", "remembered_strict", "memory_score", "memory"]:
            if col in item_meta.columns:
                item_meta[col] = pd.to_numeric(item_meta[col], errors="coerce")
        if "memory_strict" not in item_meta.columns and "remembered_strict" in item_meta.columns:
            item_meta["memory_strict"] = item_meta["remembered_strict"]
        elif "memory_strict" not in item_meta.columns and "memory_score" in item_meta.columns:
            ms = item_meta["memory_score"]
            item_meta["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))
        elif "memory_strict" not in item_meta.columns and "memory" in item_meta.columns:
            ms = item_meta["memory"]
            item_meta["memory_strict"] = np.where(ms.eq(1.0), 1.0, np.where(ms.eq(0.0), 0.0, np.nan))

    pattern_manifest_df, cache, pattern_source = load_voxel_patterns(
        pattern_root,
        pattern_long_tsv,
        roi_manifest=roi_manifest,
        roi_sets=tuple(args.roi_sets),
        max_subjects=args.max_subjects,
        allow_empty=args.allow_empty,
    )
    states = pair_state_vectors(pattern_manifest_df, cache)
    item_table = compute_cosines(states, cache, item_meta)
    network_table = _aggregate_to_network(item_table)
    desc = descriptives(network_table)
    models = fit_models(network_table)
    direction_detail, direction_summary = group_direction_agreement(
        item_table, cache, n_perm=int(args.n_permutations), rng_seed=int(args.rng_seed)
    )
    cosine_export = item_table.copy()
    for col in ("v_post_id", "v_retrieval_id", "u_retrieval_id"):
        if col in cosine_export.columns:
            del cosine_export[col]

    write_outputs(
        cfg,
        MODULE,
        {
            "voxel_pattern_manifest.tsv": pattern_manifest_df,
            "trajectory_cosines_item.tsv": cosine_export,
            "trajectory_cosine_models.tsv": models,
            "group_direction_agreement.tsv": direction_detail,
            "group_direction_agreement_summary.tsv": direction_summary,
            "descriptives.tsv": desc,
            "manifest.tsv": manifest(args, input_path, pattern_root, pattern_long_tsv, roi_manifest, raw, pattern_manifest_df, item_table, pattern_source),
        },
    )


if __name__ == "__main__":
    main()
