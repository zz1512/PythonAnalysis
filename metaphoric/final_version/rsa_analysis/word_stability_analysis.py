#!/usr/bin/env python3
"""
Single-word stability and neighborhood-change analysis.

This A++ control checks whether learned-edge differentiation can coexist with
stable single-word identities. For every subject x ROI x condition, it computes:

- same_word_stability: corr(pre pattern for word i, post pattern for word i)
- neighborhood_pre/post: mean similarity from word i to all other words in the
  same condition and stage
- neighborhood_delta_post_minus_pre
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

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.roi_library import select_roi_masks  # noqa: E402


CONDITION_ALIASES = {
    "metaphor": "yy",
    "yy": "yy",
    "spatial": "kj",
    "kj": "kj",
    "baseline": "baseline",
}
CONDITION_LABELS = {"yy": "YY", "kj": "KJ", "baseline": "Baseline"}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_any_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


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


def _bootstrap_ci(values: Iterable[float], *, seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        means[idx] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _one_sample_summary(values: Iterable[float], *, popmean: float = 0.0, seed: int = 42) -> dict[str, float]:
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
    t_val, p_val = stats.ttest_1samp(arr, popmean=popmean, nan_policy="omit")
    ci_low, ci_high = _bootstrap_ci(arr, seed=seed)
    return {
        "n_subjects": int(arr.size),
        "mean_effect": float(arr.mean()),
        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "cohens_dz": _cohens_dz(arr - popmean),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _load_template_words(template_path: Path, conditions: list[str]) -> dict[str, pd.DataFrame]:
    frame = _read_any_table(template_path).copy()
    frame["condition_norm"] = frame["condition"].map(_normalize_condition)
    frame = frame[frame["condition_norm"].isin(conditions)].copy()
    frame["word_label"] = frame["word_label"].astype(str).str.strip()
    if "sort_id" in frame.columns:
        frame["_sort_key"] = pd.to_numeric(frame["sort_id"], errors="coerce")
    else:
        frame["_sort_key"] = np.arange(len(frame), dtype=float)
    out = {}
    for condition, group in frame.groupby("condition_norm", sort=False):
        out[str(condition)] = group.sort_values("_sort_key")[["condition_norm", "pair_id", "type", "word_label"]].reset_index(drop=True)
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
    return data[mask, :].T.astype(np.float64, copy=False)


def _metadata_word_index(metadata: pd.DataFrame) -> dict[str, int]:
    meta = metadata.reset_index(drop=True).copy()
    lookup: dict[str, int] = {}
    for idx, row in meta.iterrows():
        for col in ["word_label", "unique_label"]:
            if col not in meta.columns:
                continue
            key = str(row[col]).strip()
            if key and key.lower() != "nan":
                if key in lookup and lookup[key] != int(idx):
                    raise ValueError(f"Duplicate metadata word label: {key}")
                lookup[key] = int(idx)
    return lookup


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_centered = a - np.nanmean(a, axis=1, keepdims=True)
    b_centered = b - np.nanmean(b, axis=1, keepdims=True)
    numer = np.sum(a_centered * b_centered, axis=1)
    denom = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numer / denom
    corr[~np.isfinite(corr)] = np.nan
    return np.clip(corr, -1.0, 1.0)


def _corr_similarity(samples: np.ndarray) -> np.ndarray:
    centered = samples - np.nanmean(samples, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denom = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = centered @ centered.T / denom
    sim[~np.isfinite(sim)] = np.nan
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, np.nan)
    return sim


def _ordered_samples(samples: np.ndarray, metadata: pd.DataFrame, words: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    lookup = _metadata_word_index(metadata)
    indices = []
    missing = []
    rows = []
    for row in words.itertuples(index=False):
        word = str(row.word_label).strip()
        idx = lookup.get(word)
        if idx is None:
            missing.append(word)
            continue
        indices.append(idx)
        rows.append(
            {
                "pair_id": getattr(row, "pair_id", ""),
                "word_role": getattr(row, "type", ""),
                "word_label": word,
            }
        )
    if missing:
        return np.empty((0, samples.shape[1])), pd.DataFrame(rows), missing
    return samples[np.asarray(indices, dtype=int), :], pd.DataFrame(rows), missing


def _resolve_roi_masks(manifest_path: Path, roi_sets: list[str]) -> dict[str, dict[str, Path]]:
    resolved = {}
    for roi_set in roi_sets:
        masks = select_roi_masks(manifest_path, roi_set=roi_set, include_flag="include_in_rsa")
        if not masks:
            raise RuntimeError(f"No ROI masks found for roi_set={roi_set}")
        resolved[roi_set] = {str(name): Path(path) for name, path in masks.items()}
    return resolved


def _subject_dirs(pattern_root: Path) -> list[Path]:
    if not pattern_root.exists():
        raise FileNotFoundError(f"Missing pattern root: {pattern_root}")
    return sorted(path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-"))


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_sets = _resolve_roi_masks(args.roi_manifest, list(args.roi_sets))
    mask_cache = {str(path): _load_mask(path) for masks in roi_sets.values() for path in masks.values()}
    template_words = _load_template_words(args.stimuli_template, list(args.conditions))

    word_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in _subject_dirs(args.pattern_root):
        subject = subject_dir.name
        for condition in args.conditions:
            condition = str(condition).strip().lower()
            words = template_words.get(condition)
            if words is None or words.empty:
                continue
            paths = {
                "pre_image": subject_dir / f"pre_{condition}.nii.gz",
                "post_image": subject_dir / f"post_{condition}.nii.gz",
                "pre_metadata": subject_dir / f"pre_{condition}_metadata.tsv",
                "post_metadata": subject_dir / f"post_{condition}_metadata.tsv",
            }
            try:
                for path in paths.values():
                    if not path.exists():
                        raise FileNotFoundError(path)
                pre_data = _load_4d(paths["pre_image"])
                post_data = _load_4d(paths["post_image"])
                pre_meta = _read_any_table(paths["pre_metadata"])
                post_meta = _read_any_table(paths["post_metadata"])
                if len(pre_meta) != pre_data.shape[3]:
                    raise ValueError(f"{paths['pre_metadata']} rows do not match pre image volumes")
                if len(post_meta) != post_data.shape[3]:
                    raise ValueError(f"{paths['post_metadata']} rows do not match post image volumes")
            except Exception as exc:
                failure_rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
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
                for roi, mask_path in masks.items():
                    base = {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS.get(condition, condition),
                        "roi_mask_path": str(mask_path),
                    }
                    try:
                        mask = mask_cache[str(mask_path)]
                        pre_samples_all = _masked_samples(pre_data, mask, image_path=paths["pre_image"], mask_path=mask_path)
                        post_samples_all = _masked_samples(post_data, mask, image_path=paths["post_image"], mask_path=mask_path)
                        pre_samples, word_frame, pre_missing = _ordered_samples(pre_samples_all, pre_meta, words)
                        post_samples, _, post_missing = _ordered_samples(post_samples_all, post_meta, words)
                        missing = sorted(set(pre_missing + post_missing))
                        if missing:
                            raise ValueError(f"Missing word labels: {missing[:8]}")
                        if pre_samples.shape != post_samples.shape or pre_samples.shape[0] == 0:
                            raise ValueError("Aligned pre/post samples are empty or mismatched")

                        stability = _corr_rows(pre_samples, post_samples)
                        pre_sim = _corr_similarity(pre_samples)
                        post_sim = _corr_similarity(post_samples)
                        pre_neighborhood = np.nanmean(pre_sim, axis=1)
                        post_neighborhood = np.nanmean(post_sim, axis=1)
                        neighborhood_delta = post_neighborhood - pre_neighborhood
                        for idx, word_row in word_frame.reset_index(drop=True).iterrows():
                            word_rows.append(
                                {
                                    **base,
                                    "pair_id": word_row["pair_id"],
                                    "word_role": word_row["word_role"],
                                    "word_label": word_row["word_label"],
                                    "same_word_stability": float(stability[idx]),
                                    "neighborhood_pre": float(pre_neighborhood[idx]),
                                    "neighborhood_post": float(post_neighborhood[idx]),
                                    "neighborhood_delta_post_minus_pre": float(neighborhood_delta[idx]),
                                }
                            )
                        subject_rows.append(
                            {
                                **base,
                                "n_words": int(pre_samples.shape[0]),
                                "n_voxels": int(mask.sum()),
                                "mean_same_word_stability": float(np.nanmean(stability)),
                                "mean_neighborhood_pre": float(np.nanmean(pre_neighborhood)),
                                "mean_neighborhood_post": float(np.nanmean(post_neighborhood)),
                                "mean_neighborhood_delta_post_minus_pre": float(np.nanmean(neighborhood_delta)),
                            }
                        )
                        qc_rows.append(
                            {
                                **base,
                                "status": "ok",
                                "skip_reason": "",
                                "n_words": int(pre_samples.shape[0]),
                                "n_voxels": int(mask.sum()),
                            }
                        )
                    except Exception as exc:
                        qc_rows.append(
                            {
                                **base,
                                "status": "failed",
                                "skip_reason": repr(exc),
                                "n_words": 0,
                                "n_voxels": 0,
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
    return pd.DataFrame(word_rows), pd.DataFrame(subject_rows), pd.DataFrame(qc_rows), pd.DataFrame(failure_rows)


def summarize(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    if subject_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    metric_specs = [
        ("mean_same_word_stability", "same_word_stability_gt_zero", "identity_stability"),
        ("mean_neighborhood_delta_post_minus_pre", "neighborhood_delta_post_minus_pre", "neighborhood_change"),
    ]
    for (roi_set, roi, condition), subset in subject_metrics.groupby(["roi_set", "roi", "condition"], sort=False):
        for metric, contrast, family in metric_specs:
            summary = _one_sample_summary(subset[metric])
            rows.append(
                {
                    "analysis_type": "one_sample",
                    "contrast": contrast,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(str(condition), str(condition)),
                    "metric": metric,
                    "contrast_family": family,
                    "is_primary_contrast": bool(metric == "mean_same_word_stability"),
                    **summary,
                }
            )

    wide = subject_metrics.pivot_table(
        index=["subject", "roi_set", "roi"],
        columns="condition",
        values="mean_neighborhood_delta_post_minus_pre",
        aggfunc="mean",
    ).reset_index()
    contrast_specs = [
        ("yy_neighborhood_delta_minus_kj", "yy", "kj", "condition_neighborhood_change"),
        ("yy_neighborhood_delta_minus_baseline", "yy", "baseline", "baseline_neighborhood_control"),
        ("kj_neighborhood_delta_minus_baseline", "kj", "baseline", "baseline_neighborhood_control"),
    ]
    for contrast, col_a, col_b, family in contrast_specs:
        if col_a not in wide.columns or col_b not in wide.columns:
            continue
        work = wide[["subject", "roi_set", "roi", col_a, col_b]].dropna().copy()
        work["contrast_value"] = work[col_a] - work[col_b]
        for (roi_set, roi), subset in work.groupby(["roi_set", "roi"], sort=False):
            summary = _one_sample_summary(subset["contrast_value"])
            rows.append(
                {
                    "analysis_type": "condition_contrast",
                    "contrast": contrast,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": "",
                    "condition_label": "",
                    "metric": "mean_neighborhood_delta_post_minus_pre",
                    "contrast_family": family,
                    "is_primary_contrast": False,
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
    return out.sort_values(["q_bh_primary_family", "q_bh_within_roi_set_family", "p"], na_position="last").reset_index(drop=True)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Single-word pre/post stability and neighborhood analysis.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--roi-sets", nargs="+", default=["main_functional", "literature", "literature_spatial"])
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj", "baseline"])
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.conditions = [str(item).strip().lower() for item in args.conditions]

    output_dir = ensure_dir(args.output_dir or (args.paper_output_root / "qc" / "word_stability"))
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    word_metrics, subject_metrics, qc, failures = run_analysis(args)
    failure_columns = ["subject", "roi_set", "roi", "condition", "roi_mask_path", "error_type", "error_message", "traceback"]
    if failures.empty:
        failures = pd.DataFrame(columns=failure_columns)
    write_table(word_metrics, output_dir / "word_stability_long.tsv")
    write_table(subject_metrics, output_dir / "word_stability_subject.tsv")
    write_table(qc, output_dir / "word_stability_qc.tsv")
    write_table(failures, output_dir / "word_stability_failures.tsv")
    if not failures.empty and not args.allow_partial:
        raise RuntimeError(f"Word stability analysis has {len(failures)} failures.")

    summary = summarize(subject_metrics)
    write_table(summary, output_dir / "word_stability_group_fdr.tsv")
    write_table(summary, tables_si / "table_word_stability_full.tsv")
    main_table = summary[summary["is_primary_contrast"].astype(bool)].copy() if not summary.empty else summary
    write_table(main_table, tables_main / "table_word_stability.tsv")

    save_json(
        {
            "pattern_root": str(args.pattern_root),
            "stimuli_template": str(args.stimuli_template),
            "roi_manifest": str(args.roi_manifest),
            "roi_sets": list(args.roi_sets),
            "conditions": list(args.conditions),
            "output_dir": str(output_dir),
            "n_word_rows": int(len(word_metrics)),
            "n_subject_rows": int(len(subject_metrics)),
            "n_summary_rows": int(len(summary)),
            "n_failed_cells": int(len(failures)),
        },
        output_dir / "word_stability_manifest.json",
    )
    print(f"[word-stability] wrote {len(subject_metrics)} subject rows and {len(summary)} summary rows to {output_dir}")


if __name__ == "__main__":
    main()
