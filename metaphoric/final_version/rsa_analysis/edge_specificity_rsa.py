#!/usr/bin/env python3
"""
Learned-edge specificity analysis for pre/post RSA patterns.

This script asks whether the Step 5C pair-similarity decrease is specific to
trained cue-target edges rather than a global decrease among all words. It
computes ROI-level mean similarities for:

- trained_edge: the learned YY/KJ pair in stimuli_template.csv
- untrained_nonedge: all within-condition word pairs that were not trained pairs
- baseline_pseudo_edge: fixed jx A/B pseudo-pairs from stimuli_template.csv

The main group metrics use `drop = pre - post`, so positive values indicate a
pre-to-post similarity decrease.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats


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
from common.roi_library import sanitize_roi_tag, select_roi_masks  # noqa: E402


CONDITION_ALIASES = {
    "metaphor": "yy",
    "yy": "yy",
    "yyw": "yy",
    "yyew": "yy",
    "spatial": "kj",
    "kj": "kj",
    "kjw": "kj",
    "kjew": "kj",
    "baseline": "baseline",
    "jx": "baseline",
}
CONDITION_LABELS = {"yy": "YY", "kj": "KJ", "baseline": "Baseline"}
PRIMARY_CONTRASTS = {
    "yy_trained_drop_minus_baseline_pseudo_drop",
    "yy_trained_drop_minus_untrained_nonedge_drop",
    "yy_specificity_minus_kj_specificity",
    "yy_trained_drop_minus_kj_trained_drop",
}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _normalize_condition(value: object) -> str | None:
    return CONDITION_ALIASES.get(str(value).strip().lower())


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


def _cohens_dz(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def _bootstrap_ci(values: np.ndarray, *, seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = float(sample.mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _one_sample_summary(values: Iterable[float], *, seed: int = 42) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "n_subjects": int(arr.size),
            "mean_effect": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "cohens_dz": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    t_val, p_val = stats.ttest_1samp(arr, popmean=0.0, nan_policy="omit")
    ci_low, ci_high = _bootstrap_ci(arr, seed=seed)
    return {
        "n_subjects": int(arr.size),
        "mean_effect": float(arr.mean()),
        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "cohens_dz": _cohens_dz(arr),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _load_template_pairs(template_path: Path, conditions: list[str]) -> dict[str, pd.DataFrame]:
    frame = _read_any_table(template_path).copy()
    frame["condition_norm"] = frame["condition"].map(_normalize_condition)
    frame = frame[frame["condition_norm"].isin(conditions)].copy()
    frame["word_label"] = frame["word_label"].astype(str).str.strip()
    frame["pair_id"] = frame["pair_id"]
    if "sort_id" in frame.columns:
        frame["_sort_key"] = pd.to_numeric(frame["sort_id"], errors="coerce")
    else:
        frame["_sort_key"] = np.arange(len(frame), dtype=float)

    out: dict[str, pd.DataFrame] = {}
    for condition, condition_frame in frame.groupby("condition_norm", sort=False):
        rows: list[dict[str, object]] = []
        for pair_id, group in condition_frame.groupby("pair_id", sort=False):
            group = group.sort_values("_sort_key").reset_index(drop=True)
            if len(group) != 2:
                raise ValueError(f"{condition} pair_id={pair_id} has {len(group)} rows; expected 2.")
            rows.append(
                {
                    "condition": condition,
                    "pair_id": pair_id,
                    "word_a": str(group.loc[0, "word_label"]).strip(),
                    "word_b": str(group.loc[1, "word_label"]).strip(),
                }
            )
        out[str(condition)] = pd.DataFrame(rows)
    return out


def _load_4d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {path}")
    return data


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


def _masked_samples(data: np.ndarray, mask: np.ndarray, *, image_path: Path, mask_path: Path) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    samples = data[mask, :].T.astype(np.float64, copy=False)
    return samples


def _corr_similarity(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    centered = samples - np.nanmean(samples, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denom = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = centered @ centered.T / denom
    sim[~np.isfinite(sim)] = np.nan
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, np.nan)
    return sim


def _metadata_word_index(metadata: pd.DataFrame) -> dict[str, int]:
    meta = metadata.reset_index(drop=True).copy()
    candidates: dict[str, int] = {}
    for idx, row in meta.iterrows():
        for col in ["word_label", "unique_label"]:
            if col not in meta.columns:
                continue
            key = str(row[col]).strip()
            if key and key.lower() != "nan":
                if key in candidates and candidates[key] != int(idx):
                    raise ValueError(f"Duplicate metadata word label: {key}")
                candidates[key] = int(idx)
    return candidates


def _edge_values(sim: np.ndarray, pair_frame: pd.DataFrame, word_to_index: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    missing: list[str] = []
    for row in pair_frame.itertuples(index=False):
        word_a = str(row.word_a).strip()
        word_b = str(row.word_b).strip()
        idx_a = word_to_index.get(word_a)
        idx_b = word_to_index.get(word_b)
        if idx_a is None or idx_b is None:
            missing.append(f"{word_a}|{word_b}")
            continue
        values.append(float(sim[idx_a, idx_b]))
    return np.asarray(values, dtype=float), missing


def _nonedge_values(sim: np.ndarray, pair_frame: pd.DataFrame, word_to_index: dict[str, int]) -> tuple[np.ndarray, int]:
    rows: list[tuple[int, object]] = []
    for row in pair_frame.itertuples(index=False):
        for word in [row.word_a, row.word_b]:
            idx = word_to_index.get(str(word).strip())
            if idx is not None:
                rows.append((idx, row.pair_id))
    values: list[float] = []
    for left in range(len(rows)):
        idx_a, pair_a = rows[left]
        for right in range(left + 1, len(rows)):
            idx_b, pair_b = rows[right]
            if pair_a == pair_b:
                continue
            values.append(float(sim[idx_a, idx_b]))
    return np.asarray(values, dtype=float), len(rows)


def _mean_row(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "n_edges": int(arr.size),
        "mean_similarity": float(arr.mean()) if arr.size else float("nan"),
        "sd_similarity": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
    }


def _resolve_roi_masks(manifest_path: Path, roi_sets: list[str]) -> dict[str, dict[str, Path]]:
    resolved: dict[str, dict[str, Path]] = {}
    for roi_set in roi_sets:
        masks = select_roi_masks(manifest_path, roi_set=roi_set, include_flag="include_in_rsa")
        if not masks:
            raise RuntimeError(f"No ROI masks found for roi_set={roi_set} in {manifest_path}")
        resolved[roi_set] = {str(name): Path(path) for name, path in masks.items()}
    return resolved


def _subject_dirs(pattern_root: Path) -> list[Path]:
    if not pattern_root.exists():
        raise FileNotFoundError(f"Missing pattern root: {pattern_root}")
    return sorted(path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-"))


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_sets = _resolve_roi_masks(args.roi_manifest, list(args.roi_sets))
    mask_cache = {str(path): _load_mask(path) for masks in roi_sets.values() for path in masks.values()}
    pair_specs = _load_template_pairs(args.stimuli_template, list(args.conditions))

    metric_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in _subject_dirs(args.pattern_root):
        subject = subject_dir.name
        for condition in args.conditions:
            condition = str(condition).strip().lower()
            if condition not in pair_specs:
                continue
            pair_frame = pair_specs[condition]
            for time in ["pre", "post"]:
                image_path = subject_dir / f"{time}_{condition}.nii.gz"
                metadata_path = subject_dir / f"{time}_{condition}_metadata.tsv"
                try:
                    if not image_path.exists():
                        raise FileNotFoundError(image_path)
                    if not metadata_path.exists():
                        raise FileNotFoundError(metadata_path)
                    data = _load_4d(image_path)
                    metadata = _read_any_table(metadata_path)
                    if len(metadata) != data.shape[3]:
                        raise ValueError(f"metadata rows={len(metadata)} but image volumes={data.shape[3]}")
                    word_to_index = _metadata_word_index(metadata)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "subject": subject,
                            "condition": condition,
                            "time": time,
                            "roi_set": "all",
                            "roi": "all",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    if not args.allow_partial:
                        raise
                    continue

                for roi_set, masks in roi_sets.items():
                    for roi_name, mask_path in masks.items():
                        base = {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "condition_label": CONDITION_LABELS.get(condition, condition),
                            "time": time,
                            "image_path": str(image_path),
                            "metadata_path": str(metadata_path),
                            "roi_mask_path": str(mask_path),
                        }
                        try:
                            samples = _masked_samples(data, mask_cache[str(mask_path)], image_path=image_path, mask_path=mask_path)
                            sim = _corr_similarity(samples)
                            edge_values, missing_edges = _edge_values(sim, pair_frame, word_to_index)
                            nonedge_values, n_words = _nonedge_values(sim, pair_frame, word_to_index)
                            if missing_edges:
                                raise ValueError(f"Missing template pair members: {missing_edges[:5]}")

                            edge_status = "baseline_pseudo_edge" if condition == "baseline" else "trained_edge"
                            metric_rows.append(
                                {
                                    **base,
                                    "edge_status": edge_status,
                                    "pairing_scheme": "template_pseudo_pair" if condition == "baseline" else "learned_pair",
                                    "n_words": int(n_words),
                                    **_mean_row(edge_values),
                                }
                            )
                            if args.include_untrained_nonedge:
                                metric_rows.append(
                                    {
                                        **base,
                                        "edge_status": "untrained_nonedge",
                                        "pairing_scheme": "different_template_pairs",
                                        "n_words": int(n_words),
                                        **_mean_row(nonedge_values),
                                    }
                                )

                            qc_rows.append(
                                {
                                    **base,
                                    "status": "ok",
                                    "skip_reason": "",
                                    "n_voxels": int(mask_cache[str(mask_path)].sum()),
                                    "n_words": int(n_words),
                                    "n_template_edges": int(np.isfinite(edge_values).sum()),
                                    "n_nonedges": int(np.isfinite(nonedge_values).sum()),
                                }
                            )
                        except Exception as exc:
                            qc_rows.append(
                                {
                                    **base,
                                    "status": "failed",
                                    "skip_reason": repr(exc),
                                    "n_voxels": 0,
                                    "n_words": 0,
                                    "n_template_edges": 0,
                                    "n_nonedges": 0,
                                }
                            )
                            failure_rows.append(
                                {
                                    **base,
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            if not args.allow_partial:
                                raise

    return pd.DataFrame(metric_rows), pd.DataFrame(qc_rows), pd.DataFrame(failure_rows)


def build_delta_subject(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    pivot = (
        metrics.pivot_table(
            index=["subject", "roi_set", "roi", "condition", "condition_label", "edge_status", "pairing_scheme"],
            columns="time",
            values="mean_similarity",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "pre" not in pivot.columns or "post" not in pivot.columns:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["pre", "post"]).copy()
    pivot["delta_post_minus_pre"] = pivot["post"] - pivot["pre"]
    pivot["drop_pre_minus_post"] = pivot["pre"] - pivot["post"]
    return pivot.reset_index(drop=True)


def _wide_delta(delta: pd.DataFrame) -> pd.DataFrame:
    wide = delta.pivot_table(
        index=["subject", "roi_set", "roi"],
        columns=["condition", "edge_status"],
        values="drop_pre_minus_post",
        aggfunc="mean",
    )
    wide.columns = [f"{cond}__{edge}" for cond, edge in wide.columns]
    return wide.reset_index()


def summarize_contrasts(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []

    # Direct condition x edge drops.
    for (roi_set, roi, condition, edge_status), subset in delta.groupby(["roi_set", "roi", "condition", "edge_status"], sort=False):
        summary = _one_sample_summary(subset["drop_pre_minus_post"])
        rows.append(
            {
                "analysis_type": "edge_drop_vs_zero",
                "contrast": f"{condition}_{edge_status}_drop",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "edge_status": edge_status,
                "contrast_family": "edge_drop",
                "is_primary_contrast": False,
                **summary,
            }
        )

    wide = _wide_delta(delta)
    contrast_specs = [
        (
            "yy_trained_drop_minus_untrained_nonedge_drop",
            "yy__trained_edge",
            "yy__untrained_nonedge",
            "edge_specificity",
            True,
        ),
        (
            "kj_trained_drop_minus_untrained_nonedge_drop",
            "kj__trained_edge",
            "kj__untrained_nonedge",
            "edge_specificity",
            False,
        ),
        (
            "baseline_pseudo_drop_minus_untrained_nonedge_drop",
            "baseline__baseline_pseudo_edge",
            "baseline__untrained_nonedge",
            "edge_specificity",
            False,
        ),
        (
            "yy_trained_drop_minus_kj_trained_drop",
            "yy__trained_edge",
            "kj__trained_edge",
            "condition_trained_edge",
            True,
        ),
        (
            "yy_trained_drop_minus_baseline_pseudo_drop",
            "yy__trained_edge",
            "baseline__baseline_pseudo_edge",
            "baseline_control",
            True,
        ),
        (
            "kj_trained_drop_minus_baseline_pseudo_drop",
            "kj__trained_edge",
            "baseline__baseline_pseudo_edge",
            "baseline_control",
            False,
        ),
    ]
    if {"yy__trained_edge", "yy__untrained_nonedge", "kj__trained_edge", "kj__untrained_nonedge"}.issubset(wide.columns):
        wide["yy_specificity"] = wide["yy__trained_edge"] - wide["yy__untrained_nonedge"]
        wide["kj_specificity"] = wide["kj__trained_edge"] - wide["kj__untrained_nonedge"]
        contrast_specs.append(
            (
                "yy_specificity_minus_kj_specificity",
                "yy_specificity",
                "kj_specificity",
                "condition_edge_specificity",
                True,
            )
        )

    for contrast, col_a, col_b, family, primary in contrast_specs:
        if col_a not in wide.columns or col_b not in wide.columns:
            continue
        work = wide[["subject", "roi_set", "roi", col_a, col_b]].dropna().copy()
        work["contrast_value"] = work[col_a] - work[col_b]
        for (roi_set, roi), subset in work.groupby(["roi_set", "roi"], sort=False):
            summary = _one_sample_summary(subset["contrast_value"])
            rows.append(
                {
                    "analysis_type": "drop_contrast",
                    "contrast": contrast,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": "",
                    "edge_status": "",
                    "contrast_family": family,
                    "is_primary_contrast": bool(primary),
                    **summary,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_roi_set_family"] = np.nan
    out["q_bh_primary_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "contrast_family"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set_family"] = _bh_fdr(out.loc[idx, "p"])
    primary = out["is_primary_contrast"].astype(bool)
    for _, idx in out[primary].groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_primary_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["is_primary_contrast", "q_bh_primary_family", "p"], ascending=[False, True, True]).reset_index(drop=True)


def summarize_edge_process(delta: pd.DataFrame) -> pd.DataFrame:
    """Summarize the raw pre/post trajectory behind each edge contrast."""
    if delta.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "condition", "condition_label", "edge_status", "pairing_scheme"]
    for keys, subset in delta.groupby(group_cols, sort=False):
        roi_set, roi, condition, condition_label, edge_status, pairing_scheme = keys
        work = subset[["subject", "pre", "post", "delta_post_minus_pre", "drop_pre_minus_post"]].dropna().copy()
        drop_summary = _one_sample_summary(work["drop_pre_minus_post"])
        pre_ci_low, pre_ci_high = _bootstrap_ci(work["pre"].to_numpy(dtype=float), seed=43)
        post_ci_low, post_ci_high = _bootstrap_ci(work["post"].to_numpy(dtype=float), seed=44)
        delta_ci_low, delta_ci_high = _bootstrap_ci(work["delta_post_minus_pre"].to_numpy(dtype=float), seed=45)
        rows.append(
            {
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": condition_label,
                "edge_status": edge_status,
                "pairing_scheme": pairing_scheme,
                "n_subjects": int(len(work)),
                "mean_pre": float(work["pre"].mean()) if not work.empty else float("nan"),
                "mean_post": float(work["post"].mean()) if not work.empty else float("nan"),
                "mean_delta_post_minus_pre": float(work["delta_post_minus_pre"].mean()) if not work.empty else float("nan"),
                "mean_drop_pre_minus_post": float(work["drop_pre_minus_post"].mean()) if not work.empty else float("nan"),
                "pre_ci_low": pre_ci_low,
                "pre_ci_high": pre_ci_high,
                "post_ci_low": post_ci_low,
                "post_ci_high": post_ci_high,
                "delta_ci_low": delta_ci_low,
                "delta_ci_high": delta_ci_high,
                "drop_t_vs_zero": drop_summary["t"],
                "drop_p_vs_zero": drop_summary["p"],
                "drop_cohens_dz": drop_summary["cohens_dz"],
                "drop_ci_low": drop_summary["ci_low"],
                "drop_ci_high": drop_summary["ci_high"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["drop_q_bh_within_roi_set"] = np.nan
    for _, idx in out.groupby("roi_set", dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "drop_q_bh_within_roi_set"] = _bh_fdr(out.loc[idx, "drop_p_vs_zero"])
    return out.sort_values(["roi_set", "roi", "condition", "edge_status"]).reset_index(drop=True)


def summarize_contrast_process(summary: pd.DataFrame, process: pd.DataFrame) -> pd.DataFrame:
    """Attach component pre/post/drop values to each primary contrast."""
    if summary.empty or process.empty:
        return pd.DataFrame()
    component_map = {
        "yy_trained_drop_minus_untrained_nonedge_drop": [("yy", "trained_edge"), ("yy", "untrained_nonedge")],
        "kj_trained_drop_minus_untrained_nonedge_drop": [("kj", "trained_edge"), ("kj", "untrained_nonedge")],
        "baseline_pseudo_drop_minus_untrained_nonedge_drop": [
            ("baseline", "baseline_pseudo_edge"),
            ("baseline", "untrained_nonedge"),
        ],
        "yy_trained_drop_minus_kj_trained_drop": [("yy", "trained_edge"), ("kj", "trained_edge")],
        "yy_trained_drop_minus_baseline_pseudo_drop": [("yy", "trained_edge"), ("baseline", "baseline_pseudo_edge")],
        "kj_trained_drop_minus_baseline_pseudo_drop": [("kj", "trained_edge"), ("baseline", "baseline_pseudo_edge")],
        "yy_specificity_minus_kj_specificity": [
            ("yy", "trained_edge"),
            ("yy", "untrained_nonedge"),
            ("kj", "trained_edge"),
            ("kj", "untrained_nonedge"),
        ],
    }
    rows: list[dict[str, object]] = []
    primary = summary[summary["is_primary_contrast"].astype(bool)].copy()
    for row in primary.itertuples(index=False):
        components = component_map.get(str(row.contrast), [])
        for condition, edge_status in components:
            match = process[
                process["roi_set"].eq(row.roi_set)
                & process["roi"].eq(row.roi)
                & process["condition"].eq(condition)
                & process["edge_status"].eq(edge_status)
            ]
            if match.empty:
                continue
            comp = match.iloc[0]
            rows.append(
                {
                    "roi_set": row.roi_set,
                    "roi": row.roi,
                    "contrast": row.contrast,
                    "contrast_mean_effect": row.mean_effect,
                    "contrast_p": row.p,
                    "contrast_q_bh_primary_family": row.q_bh_primary_family,
                    "component_condition": condition,
                    "component_edge_status": edge_status,
                    "component_mean_pre": comp["mean_pre"],
                    "component_mean_post": comp["mean_post"],
                    "component_mean_delta_post_minus_pre": comp["mean_delta_post_minus_pre"],
                    "component_mean_drop_pre_minus_post": comp["mean_drop_pre_minus_post"],
                    "component_drop_p_vs_zero": comp["drop_p_vs_zero"],
                    "component_drop_q_bh_within_roi_set": comp["drop_q_bh_within_roi_set"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Test learned-edge specificity in pre/post pattern similarities.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--roi-sets", nargs="+", default=["main_functional", "literature", "literature_spatial"])
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj", "baseline"])
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-untrained-nonedge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    args.conditions = [str(item).strip().lower() for item in args.conditions]
    output_dir = ensure_dir(args.output_dir or (args.paper_output_root / "qc" / "edge_specificity"))
    tagged_si_outputs = args.output_dir is not None and len(args.roi_sets) == 1
    table_suffix = f"_{sanitize_roi_tag(args.roi_sets[0])}" if tagged_si_outputs else ""
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    metrics, qc, failures = run_analysis(args)
    failure_columns = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "time",
        "image_path",
        "metadata_path",
        "roi_mask_path",
        "error_type",
        "error_message",
        "traceback",
    ]
    if failures.empty:
        failures = pd.DataFrame(columns=failure_columns)
    write_table(metrics, output_dir / "edge_similarity_long.tsv")
    write_table(qc, output_dir / "edge_similarity_qc.tsv")
    write_table(failures, output_dir / "edge_similarity_failures.tsv")
    if not failures.empty and not args.allow_partial:
        raise RuntimeError(f"Edge specificity analysis has {len(failures)} failures.")

    delta = build_delta_subject(metrics)
    summary = summarize_contrasts(delta)
    process = summarize_edge_process(delta)
    contrast_process = summarize_contrast_process(summary, process)
    write_table(delta, output_dir / "edge_delta_subject.tsv")
    write_table(process, output_dir / "edge_process_group.tsv")
    write_table(contrast_process, output_dir / "edge_contrast_process_components.tsv")
    write_table(summary, output_dir / "edge_specificity_group_fdr.tsv")
    write_table(summary, tables_si / f"table_edge_specificity_full{table_suffix}.tsv")
    write_table(process, tables_si / f"table_edge_process_group{table_suffix}.tsv")
    write_table(contrast_process, tables_si / f"table_edge_contrast_process_components{table_suffix}.tsv")
    main_table = summary[summary["is_primary_contrast"].astype(bool)].copy() if not summary.empty else summary
    if tagged_si_outputs:
        write_table(main_table, tables_si / f"table_edge_specificity{table_suffix}.tsv")
    else:
        write_table(main_table, tables_main / "table_edge_specificity.tsv")

    save_json(
        {
            "pattern_root": str(args.pattern_root),
            "stimuli_template": str(args.stimuli_template),
            "roi_manifest": str(args.roi_manifest),
            "roi_sets": list(args.roi_sets),
            "conditions": list(args.conditions),
            "output_dir": str(output_dir),
            "n_metric_rows": int(len(metrics)),
            "n_delta_rows": int(len(delta)),
            "n_process_rows": int(len(process)),
            "n_contrast_process_rows": int(len(contrast_process)),
            "n_summary_rows": int(len(summary)),
            "n_failed_cells": int(len(failures)),
            "primary_contrasts": sorted(PRIMARY_CONTRASTS),
        },
        output_dir / "edge_specificity_manifest.json",
    )
    print(f"[edge-specificity] wrote {len(metrics)} metric rows and {len(summary)} summary rows to {output_dir}")


if __name__ == "__main__":
    main()
