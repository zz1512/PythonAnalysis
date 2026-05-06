#!/usr/bin/env python3
"""Learning run3 <-> run4 within-item pattern reliability.

For each subject x ROI x condition (yy / kj) this script:
- loads ``pattern_root/sub-XX/learn_{condition}.nii.gz`` plus metadata,
- splits the 4D stack into run3 / run4 sub-sets based on the ``run`` column,
- aligns the two sub-sets by ``word_label`` (fallback to ``pair_id``), and
- computes ``corr(L_i^run3, L_i^run4)`` for each paired item.

Item-level values are aggregated to subject x ROI x condition means, then
entered into group-level one-sample t vs 0 tests and the metaphor - spatial
paired t-test, with BH-FDR applied within ROI set.

Outputs (under ``paper_output_root / 'qc' / f'learning_within_item_reliability_{roi_tag}'``):
- ``learning_within_item_reliability_item.tsv``
- ``learning_within_item_reliability_subject.tsv``
- ``learning_within_item_reliability_group_one_sample.tsv``
- ``learning_within_item_reliability_group_yy_vs_kj.tsv``
- ``learning_within_item_reliability_qc.tsv``
- ``learning_within_item_reliability_manifest.json``
Additionally a copy of the subject table is mirrored into
``paper_output_root / 'tables_si' / f'table_learning_within_item_reliability_{roi_tag}.tsv'``.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback

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

from common.final_utils import (  # noqa: E402
    ensure_dir,
    one_sample_t_summary,
    paired_t_summary,
    read_table,
    save_json,
    write_table,
)
from common.pattern_metrics import load_4d_data, load_mask  # noqa: E402
from common.roi_library import current_roi_set, sanitize_roi_tag, select_roi_masks  # noqa: E402


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
RUNS = [3, 4]


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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    a = a[valid] - np.nanmean(a[valid])
    b = b[valid] - np.nanmean(b[valid])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if math.isclose(denom, 0.0):
        return float("nan")
    return float(np.dot(a, b) / denom)


def _masked_samples(data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _load_learn_condition(
    subject_dir: Path,
    condition: str,
) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"learn_{condition}.nii.gz"
    meta_path = subject_dir / f"learn_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing learn_{condition} image or metadata for {subject_dir.name}"
        )
    meta = read_table(meta_path).reset_index(drop=True)
    if "run" not in meta.columns:
        raise ValueError(
            f"Metadata for {image_path.name} missing required 'run' column; "
            "cannot split run3 / run4 for learning within-item reliability."
        )
    _, data = load_4d_data(image_path)
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    if "pair_id" in meta.columns:
        meta["pair_id"] = pd.to_numeric(meta["pair_id"], errors="coerce")
    if "word_label" in meta.columns:
        meta["word_label"] = meta["word_label"].astype(str).str.strip()
    return meta, data, image_path


def _pick_key_column(meta: pd.DataFrame) -> str:
    if "word_label" in meta.columns:
        labels = meta["word_label"].astype(str).str.strip()
        if labels.replace({"": np.nan, "nan": np.nan}).notna().any():
            return "word_label"
    if "pair_id" in meta.columns:
        return "pair_id"
    raise ValueError("Learning metadata missing both 'word_label' and 'pair_id' columns")


def _build_run_index(
    meta: pd.DataFrame,
    run: int,
    key_col: str,
) -> tuple[dict[object, int], list[object]]:
    """Return dict of key -> row index and list of keys that are duplicated in this run."""
    subset = meta[meta["run"] == run]
    idx_map: dict[object, int] = {}
    duplicates: list[object] = []
    counts = subset.groupby(key_col).size()
    dup_keys = set(counts[counts > 1].index.tolist())
    for idx, row in subset.iterrows():
        key = row[key_col]
        if isinstance(key, float) and math.isnan(key):
            continue
        if isinstance(key, str) and (not key or key.lower() == "nan"):
            continue
        if key in dup_keys:
            duplicates.append(key)
            continue
        idx_map[key] = int(idx)
    return idx_map, duplicates


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    item_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name

    _, mask = load_mask(mask_path)

    for condition in CONDITIONS:
        if condition not in learn_cache:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": "learn cache missing",
                    "key_column": "",
                    "n_run3_rows": 0,
                    "n_run4_rows": 0,
                    "n_paired_items": 0,
                    "n_skipped_duplicates": 0,
                    "n_unpaired": 0,
                }
            )
            continue

        meta, data, image_path = learn_cache[condition]
        try:
            samples = _masked_samples(data, mask, image_path, mask_path)
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": f"masking failed: {exc}",
                    "key_column": "",
                    "n_run3_rows": 0,
                    "n_run4_rows": 0,
                    "n_paired_items": 0,
                    "n_skipped_duplicates": 0,
                    "n_unpaired": 0,
                }
            )
            continue

        if samples.shape[0] != len(meta):
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": (
                        f"volume/metadata mismatch: {samples.shape[0]} vols vs {len(meta)} rows"
                    ),
                    "key_column": "",
                    "n_run3_rows": 0,
                    "n_run4_rows": 0,
                    "n_paired_items": 0,
                    "n_skipped_duplicates": 0,
                    "n_unpaired": 0,
                }
            )
            continue

        try:
            key_col = _pick_key_column(meta)
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "ok": False,
                    "fail_reason": f"key column error: {exc}",
                    "key_column": "",
                    "n_run3_rows": int((meta["run"] == 3).sum()),
                    "n_run4_rows": int((meta["run"] == 4).sum()),
                    "n_paired_items": 0,
                    "n_skipped_duplicates": 0,
                    "n_unpaired": 0,
                }
            )
            continue

        run3_idx, dup3 = _build_run_index(meta, 3, key_col)
        run4_idx, dup4 = _build_run_index(meta, 4, key_col)
        dup_keys = set(dup3) | set(dup4)

        paired_keys = [k for k in run3_idx if k in run4_idx]
        unpaired = (set(run3_idx) ^ set(run4_idx)) | dup_keys

        for key in paired_keys:
            i3 = run3_idx[key]
            i4 = run4_idx[key]
            reliability = _corr(samples[i3], samples[i4])
            row3 = meta.iloc[i3]
            pair_id_val = row3.get("pair_id", np.nan) if "pair_id" in meta.columns else np.nan
            word_label_val = row3.get("word_label", "") if "word_label" in meta.columns else ""
            item_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "key_column": key_col,
                    "key_value": key,
                    "word_label": word_label_val if key_col == "word_label" else word_label_val,
                    "pair_id": pair_id_val,
                    "reliability": reliability,
                }
            )

        reliabilities = [
            row["reliability"] for row in item_rows
            if row["subject"] == subject
            and row["roi_set"] == roi_set
            and row["roi"] == roi_name
            and row["condition"] == condition
        ]
        finite_vals = [v for v in reliabilities if isinstance(v, float) and math.isfinite(v)]
        subject_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi_name,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "mean_reliability": float(np.mean(finite_vals)) if finite_vals else float("nan"),
                "n_items": int(len(finite_vals)),
            }
        )

        qc_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi_name,
                "condition": condition,
                "ok": True,
                "fail_reason": "",
                "key_column": key_col,
                "n_run3_rows": int((meta["run"] == 3).sum()),
                "n_run4_rows": int((meta["run"] == 4).sum()),
                "n_paired_items": int(len(paired_keys)),
                "n_skipped_duplicates": int(len(dup_keys)),
                "n_unpaired": int(len(unpaired)),
            }
        )

    return item_rows, subject_rows, qc_rows


def summarize_one_sample(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    if subject_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "condition", "condition_label"]
    for keys, subset in subject_metrics.groupby(group_cols, sort=False):
        roi_set, roi, condition, condition_label = keys
        summary = one_sample_t_summary(subset["mean_reliability"])
        rows.append(
            {
                "analysis_type": "one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": condition_label,
                "n_subjects": int(summary["n"]),
                "mean_effect": summary["mean"],
                "t": summary["t"],
                "p": summary["p"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_roi_set"] = np.nan
    for _, idx in out.groupby(["roi_set", "condition"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["roi_set", "condition", "p"], na_position="last")


def summarize_yy_vs_kj(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    if subject_metrics.empty:
        return pd.DataFrame()
    pivot = subject_metrics.pivot_table(
        index=["subject", "roi_set", "roi"],
        columns="condition",
        values="mean_reliability",
        aggfunc="mean",
    ).reset_index()
    if "yy" not in pivot.columns or "kj" not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (roi_set, roi), subset in pivot.groupby(["roi_set", "roi"], sort=False):
        summary = paired_t_summary(subset["yy"], subset["kj"])
        rows.append(
            {
                "analysis_type": "yy_minus_kj_paired_t",
                "roi_set": roi_set,
                "roi": roi,
                "n_subjects": int(summary["n"]),
                "mean_yy": summary["mean_a"],
                "mean_kj": summary["mean_b"],
                "mean_diff": summary["mean_a"] - summary["mean_b"]
                if np.isfinite(summary["mean_a"]) and np.isfinite(summary["mean_b"])
                else float("nan"),
                "t": summary["t"],
                "p": summary["p"],
                "cohens_dz": summary["cohens_dz"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_roi_set"] = np.nan
    for _, idx in out.groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["roi_set", "p"], na_position="last")


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Learning run3<->run4 within-item pattern reliability in meta ROIs.",
    )
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument(
        "--roi-manifest",
        type=Path,
        default=base_dir / "roi_library" / "manifest.tsv",
    )
    parser.add_argument(
        "--roi-sets",
        nargs="+",
        default=["meta_metaphor", "meta_spatial"],
    )
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=base_dir / "paper_outputs",
    )
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(current_roi_set())
    output_dir = ensure_dir(
        args.paper_output_root / "qc" / f"learning_within_item_reliability_{roi_tag}"
    )
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    subjects = sorted([path for path in args.pattern_root.glob("sub-*") if path.is_dir()])
    subjects = [
        sub
        for sub in subjects
        if (sub / "learn_yy.nii.gz").exists() and (sub / "learn_kj.nii.gz").exists()
    ]

    roi_masks_by_set = {
        roi_set: select_roi_masks(
            args.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa"
        )
        for roi_set in args.roi_sets
    }

    item_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        load_failed = False
        for condition in CONDITIONS:
            try:
                learn_cache[condition] = _load_learn_condition(subject_dir, condition)
            except Exception as exc:
                failure_rows.append(
                    {
                        "subject": subject_dir.name,
                        "roi_set": "all",
                        "roi": "all",
                        "condition": condition,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                load_failed = True
                break
        if load_failed:
            continue

        for roi_set, masks in roi_masks_by_set.items():
            for roi_name, mask_path in masks.items():
                try:
                    items, subject_metrics, qc = process_subject_roi(
                        subject_dir,
                        roi_set,
                        roi_name,
                        mask_path,
                        learn_cache,
                    )
                    item_rows.extend(items)
                    subject_rows.extend(subject_metrics)
                    qc_rows.extend(qc)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": "",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

    item_metrics = pd.DataFrame(item_rows)
    subject_metrics = pd.DataFrame(subject_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group_one_sample = summarize_one_sample(subject_metrics)
    group_yy_vs_kj = summarize_yy_vs_kj(subject_metrics)

    write_table(item_metrics, output_dir / "learning_within_item_reliability_item.tsv")
    write_table(subject_metrics, output_dir / "learning_within_item_reliability_subject.tsv")
    write_table(
        group_one_sample,
        output_dir / "learning_within_item_reliability_group_one_sample.tsv",
    )
    write_table(
        group_yy_vs_kj,
        output_dir / "learning_within_item_reliability_group_yy_vs_kj.tsv",
    )
    write_table(qc, output_dir / "learning_within_item_reliability_qc.tsv")
    if not failures.empty:
        write_table(
            failures,
            output_dir / "learning_within_item_reliability_failures.tsv",
        )

    write_table(
        subject_metrics,
        tables_si / f"table_learning_within_item_reliability_{roi_tag}.tsv",
    )

    save_json(
        {
            "roi_sets": args.roi_sets,
            "roi_tag": roi_tag,
            "n_subjects_with_learn_patterns": len(subjects),
            "subjects": [path.name for path in subjects],
            "n_item_rows": int(len(item_metrics)),
            "n_subject_rows": int(len(subject_metrics)),
            "n_qc_rows": int(len(qc)),
            "n_failures": int(len(failures)),
            "pattern_root": str(args.pattern_root),
            "roi_manifest": str(args.roi_manifest),
            "paper_output_root": str(args.paper_output_root),
        },
        output_dir / "learning_within_item_reliability_manifest.json",
    )


if __name__ == "__main__":
    main()
