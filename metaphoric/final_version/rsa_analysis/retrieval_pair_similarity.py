#!/usr/bin/env python3
"""Run-7 retrieval geometry for meta ROI analyses.

The script uses run-7 LSS patterns stacked as
`pattern_root/sub-xx/retrieval_{yy,kj}.nii.gz`. It keeps only the subject/item
intersection that is present in neural patterns and retrieval metadata.

Main outputs:
- qc/retrieval_geometry/retrieval_item_metrics.tsv
- qc/retrieval_geometry/retrieval_pair_metrics.tsv
- qc/retrieval_geometry/retrieval_subject_metrics.tsv
- qc/retrieval_geometry/retrieval_geometry_group_fdr.tsv
- tables_si/table_retrieval_geometry.tsv
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback

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
from common.roi_library import select_roi_masks  # noqa: E402


CONDITIONS = ["yy", "kj"]
STAGES = ["pre", "post", "retrieval"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


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


def _cohens_dz(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def _one_sample(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return {
            "n_subjects": int(arr.size),
            "mean_effect": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "cohens_dz": float("nan"),
        }
    t_val, p_val = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
    return {
        "n_subjects": int(arr.size),
        "mean_effect": float(arr.mean()),
        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "cohens_dz": _cohens_dz(pd.Series(arr)),
    }


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


def _masked_samples(data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _load_stage(subject_dir: Path, stage: str, condition: str) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"{stage}_{condition}.nii.gz"
    meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {stage}_{condition} image or metadata for {subject_dir.name}")
    meta = _read_any(meta_path).reset_index(drop=True)
    data = _load_4d(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(f"Volume/metadata mismatch: {image_path} has {data.shape[3]} vols, meta has {len(meta)}")
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["stage"] = stage
    meta["pair_id"] = pd.to_numeric(meta["pair_id"], errors="coerce")
    meta["word_label"] = meta["word_label"].astype(str).str.strip()
    if "memory" in meta.columns:
        meta["memory"] = pd.to_numeric(meta["memory"], errors="coerce")
    if "action_time" in meta.columns:
        meta["action_time"] = pd.to_numeric(meta["action_time"], errors="coerce")
    return meta, data, image_path


def _load_subject_cache(subject_dir: Path) -> dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]]:
    cache: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]] = {}
    for condition in CONDITIONS:
        cache[condition] = {}
        for stage in STAGES:
            cache[condition][stage] = _load_stage(subject_dir, stage, condition)
    return cache


def _word_pattern_map(meta: pd.DataFrame, samples: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for idx, row in meta.iterrows():
        label = str(row["word_label"]).strip()
        if label and label.lower() != "nan":
            out[label] = samples[int(idx)]
    return out


def _template_pairs(template_path: Path) -> dict[str, pd.DataFrame]:
    frame = read_table(template_path).copy()
    frame["condition_norm"] = frame["condition"].astype(str).str.lower().map(
        {"metaphor": "yy", "spatial": "kj", "yy": "yy", "kj": "kj"}
    )
    frame = frame[frame["condition_norm"].isin(CONDITIONS)].copy()
    frame["word_label"] = frame["word_label"].astype(str).str.strip()
    if "sort_id" in frame.columns:
        frame["_sort"] = pd.to_numeric(frame["sort_id"], errors="coerce")
    else:
        frame["_sort"] = np.arange(len(frame), dtype=float)
    pairs: dict[str, list[dict[str, object]]] = {cond: [] for cond in CONDITIONS}
    for (condition, pair_id), group in frame.groupby(["condition_norm", "pair_id"], sort=False):
        group = group.sort_values("_sort")
        if len(group) != 2:
            continue
        rows = group.to_dict("records")
        pairs[str(condition)].append(
            {
                "condition": condition,
                "pair_id": int(pair_id),
                "word_a": str(rows[0]["word_label"]),
                "word_b": str(rows[1]["word_label"]),
            }
        )
    return {cond: pd.DataFrame(rows) for cond, rows in pairs.items()}


def _different_mean(anchor: np.ndarray, candidates: dict[str, np.ndarray], exclude: set[str]) -> float:
    vals = [_corr(anchor, value) for label, value in candidates.items() if label not in exclude]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    pairs_by_condition: dict[str, pd.DataFrame],
    stage_cache: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    item_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name
    mask = _load_mask(mask_path)

    for condition in CONDITIONS:
        pairs = pairs_by_condition[condition]
        stage_meta: dict[str, pd.DataFrame] = {}
        stage_samples: dict[str, np.ndarray] = {}
        stage_words: dict[str, dict[str, np.ndarray]] = {}
        ok = True
        fail_reason = ""
        for stage in STAGES:
            try:
                meta, data, image_path = stage_cache[condition][stage]
                samples = _masked_samples(data, mask, image_path, mask_path)
                stage_meta[stage] = meta
                stage_samples[stage] = samples
                stage_words[stage] = _word_pattern_map(meta, samples)
            except Exception as exc:
                ok = False
                fail_reason = f"{stage}: {exc}"
                break
        qc_rows.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi_name,
                "condition": condition,
                "ok": bool(ok),
                "fail_reason": fail_reason,
                "n_pairs_template": int(len(pairs)),
                "n_retrieval_rows": int(len(stage_meta.get("retrieval", []))) if ok else 0,
            }
        )
        if not ok:
            continue

        retrieval_meta = stage_meta["retrieval"].copy()
        retrieval_meta["pair_id"] = pd.to_numeric(retrieval_meta["pair_id"], errors="coerce")
        retrieval_info = retrieval_meta.set_index("word_label", drop=False)

        for pair in pairs.itertuples(index=False):
            pair_id = int(pair.pair_id)
            words = [str(pair.word_a), str(pair.word_b)]
            if not all(word in stage_words["retrieval"] for word in words):
                continue

            stage_pair_sim: dict[str, float] = {}
            for stage in STAGES:
                if all(word in stage_words[stage] for word in words):
                    stage_pair_sim[stage] = _corr(stage_words[stage][words[0]], stage_words[stage][words[1]])
                else:
                    stage_pair_sim[stage] = float("nan")

            post_word_same = []
            pre_word_same = []
            item_reinstatement = []
            pre_item_reinstatement = []
            for word in words:
                post_same = _corr(stage_words["post"][word], stage_words["retrieval"][word]) if word in stage_words["post"] else float("nan")
                pre_same = _corr(stage_words["pre"][word], stage_words["retrieval"][word]) if word in stage_words["pre"] else float("nan")
                post_diff = (
                    _different_mean(stage_words["post"][word], stage_words["retrieval"], set(words))
                    if word in stage_words["post"]
                    else float("nan")
                )
                pre_diff = (
                    _different_mean(stage_words["pre"][word], stage_words["retrieval"], set(words))
                    if word in stage_words["pre"]
                    else float("nan")
                )
                post_reinstatement = post_same - post_diff if np.isfinite(post_same) and np.isfinite(post_diff) else float("nan")
                pre_reinstatement = pre_same - pre_diff if np.isfinite(pre_same) and np.isfinite(pre_diff) else float("nan")
                post_word_same.append(post_same)
                pre_word_same.append(pre_same)
                item_reinstatement.append(post_reinstatement)
                pre_item_reinstatement.append(pre_reinstatement)
                info = retrieval_info.loc[word] if word in retrieval_info.index else pd.Series(dtype=object)
                item_rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "pair_id": pair_id,
                        "word_label": word,
                        "memory": info.get("memory", np.nan),
                        "action_time": info.get("action_time", np.nan),
                        "retrieval_pair_similarity": stage_pair_sim["retrieval"],
                        "post_pair_similarity": stage_pair_sim["post"],
                        "pre_pair_similarity": stage_pair_sim["pre"],
                        "post_to_retrieval_same_word": post_same,
                        "post_to_retrieval_different_word_mean": post_diff,
                        "post_to_retrieval_word_reinstatement": post_reinstatement,
                        "pre_to_retrieval_same_word": pre_same,
                        "pre_to_retrieval_different_word_mean": pre_diff,
                        "pre_to_retrieval_word_reinstatement": pre_reinstatement,
                    }
                )

            # Pair-level cross-stage reinstatement: all same-pair cross-stage
            # word correlations minus cross-stage correlations to words from
            # different pairs.
            same_cross = []
            diff_cross = []
            pre_same_cross = []
            pre_diff_cross = []
            for word_post in words:
                if word_post not in stage_words["post"]:
                    continue
                if word_post in stage_words["pre"]:
                    pre_same_cross.extend(
                        [_corr(stage_words["pre"][word_post], stage_words["retrieval"][word_ret]) for word_ret in words]
                    )
                    pre_diff_cross.append(_different_mean(stage_words["pre"][word_post], stage_words["retrieval"], set(words)))
                same_cross.extend(
                    [_corr(stage_words["post"][word_post], stage_words["retrieval"][word_ret]) for word_ret in words]
                )
                diff_cross.append(_different_mean(stage_words["post"][word_post], stage_words["retrieval"], set(words)))
            same_cross = [v for v in same_cross if np.isfinite(v)]
            diff_cross = [v for v in diff_cross if np.isfinite(v)]
            pre_same_cross = [v for v in pre_same_cross if np.isfinite(v)]
            pre_diff_cross = [v for v in pre_diff_cross if np.isfinite(v)]
            pair_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "pair_id": pair_id,
                    "word_a": words[0],
                    "word_b": words[1],
                    "pre_pair_similarity": stage_pair_sim["pre"],
                    "post_pair_similarity": stage_pair_sim["post"],
                    "retrieval_pair_similarity": stage_pair_sim["retrieval"],
                    "post_minus_pre_pair_similarity": stage_pair_sim["post"] - stage_pair_sim["pre"],
                    "retrieval_minus_post_pair_similarity": stage_pair_sim["retrieval"] - stage_pair_sim["post"],
                    "retrieval_minus_pre_pair_similarity": stage_pair_sim["retrieval"] - stage_pair_sim["pre"],
                    "post_to_retrieval_pair_same": float(np.mean(same_cross)) if same_cross else float("nan"),
                    "post_to_retrieval_pair_different": float(np.mean(diff_cross)) if diff_cross else float("nan"),
                    "post_to_retrieval_pair_reinstatement": (
                        float(np.mean(same_cross) - np.mean(diff_cross)) if same_cross and diff_cross else float("nan")
                    ),
                    "pre_to_retrieval_pair_same": float(np.mean(pre_same_cross)) if pre_same_cross else float("nan"),
                    "pre_to_retrieval_pair_different": float(np.mean(pre_diff_cross)) if pre_diff_cross else float("nan"),
                    "pre_to_retrieval_pair_reinstatement": (
                        float(np.mean(pre_same_cross) - np.mean(pre_diff_cross)) if pre_same_cross and pre_diff_cross else float("nan")
                    ),
                    "post_to_retrieval_word_reinstatement": float(np.nanmean(item_reinstatement)),
                    "pre_to_retrieval_word_reinstatement": float(np.nanmean(pre_item_reinstatement)),
                    "post_to_retrieval_same_word": float(np.nanmean(post_word_same)),
                    "pre_to_retrieval_same_word": float(np.nanmean(pre_word_same)),
                }
            )

        if pair_rows:
            pair_frame = pd.DataFrame([row for row in pair_rows if row["subject"] == subject and row["roi"] == roi_name and row["condition"] == condition])
            metric_cols = [
                "pre_pair_similarity",
                "post_pair_similarity",
                "retrieval_pair_similarity",
                "post_minus_pre_pair_similarity",
                "retrieval_minus_post_pair_similarity",
                "retrieval_minus_pre_pair_similarity",
                "post_to_retrieval_pair_reinstatement",
                "pre_to_retrieval_pair_reinstatement",
                "post_to_retrieval_word_reinstatement",
                "pre_to_retrieval_word_reinstatement",
            ]
            for metric in metric_cols:
                subject_rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "metric": metric,
                        "value": float(pd.to_numeric(pair_frame[metric], errors="coerce").mean()),
                        "n_pairs": int(pair_frame[metric].notna().sum()),
                    }
                )

    return item_rows, pair_rows, subject_rows, qc_rows


def summarize_subject_metrics(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "condition", "condition_label", "metric"]
    for keys, subset in subject_metrics.groupby(group_cols, sort=False):
        roi_set, roi, condition, condition_label, metric = keys
        summary = _one_sample(subset["value"])
        rows.append(
            {
                "analysis_type": "one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": condition_label,
                "metric": metric,
                **summary,
            }
        )

    contrast = subject_metrics.pivot_table(
        index=["subject", "roi_set", "roi", "metric"],
        columns="condition",
        values="value",
        aggfunc="mean",
    ).reset_index()
    if "yy" in contrast.columns and "kj" in contrast.columns:
        contrast["value"] = contrast["yy"] - contrast["kj"]
        for keys, subset in contrast.groupby(["roi_set", "roi", "metric"], sort=False):
            roi_set, roi, metric = keys
            summary = _one_sample(subset["value"])
            rows.append(
                {
                    "analysis_type": "yy_minus_kj",
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": "yy_minus_kj",
                    "condition_label": "YY-KJ",
                    "metric": metric,
                    **summary,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_roi_set_metric"] = np.nan
    out["q_bh_primary_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "metric"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set_metric"] = _bh_fdr(out.loc[idx, "p"])
    primary = out["metric"].isin(
        [
            "retrieval_pair_similarity",
            "retrieval_minus_post_pair_similarity",
            "post_to_retrieval_pair_reinstatement",
            "post_to_retrieval_word_reinstatement",
        ]
    )
    for _, idx in out[primary].groupby(["roi_set", "analysis_type"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_primary_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_primary_family", "q_bh_within_roi_set_metric", "p"], na_position="last")


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Run-7 retrieval geometry in meta ROIs.")
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    args = parser.parse_args()

    output_dir = ensure_dir(args.paper_output_root / "qc" / "retrieval_geometry")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")
    pairs_by_condition = _template_pairs(args.stimuli_template)
    subjects = sorted([path for path in args.pattern_root.glob("sub-*") if path.is_dir()])
    subjects = [
        sub for sub in subjects
        if (sub / "retrieval_yy.nii.gz").exists() and (sub / "retrieval_kj.nii.gz").exists()
    ]

    item_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    roi_masks_by_set = {
        roi_set: select_roi_masks(args.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa")
        for roi_set in args.roi_sets
    }
    for subject_dir in subjects:
        try:
            stage_cache = _load_subject_cache(subject_dir)
        except Exception as exc:
            failure_rows.append(
                {
                    "subject": subject_dir.name,
                    "roi_set": "all",
                    "roi": "all",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue
        for roi_set, masks in roi_masks_by_set.items():
            for roi_name, mask_path in masks.items():
                try:
                    items, pairs, subject_metrics, qc = process_subject_roi(
                        subject_dir,
                        roi_set,
                        roi_name,
                        mask_path,
                        pairs_by_condition,
                        stage_cache,
                    )
                    item_rows.extend(items)
                    pair_rows.extend(pairs)
                    subject_rows.extend(subject_metrics)
                    qc_rows.extend(qc)
                except Exception as exc:
                    failure_rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

    item_metrics = pd.DataFrame(item_rows)
    pair_metrics = pd.DataFrame(pair_rows)
    subject_metrics = pd.DataFrame(subject_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group = summarize_subject_metrics(subject_metrics)

    write_table(item_metrics, output_dir / "retrieval_item_metrics.tsv")
    write_table(pair_metrics, output_dir / "retrieval_pair_metrics.tsv")
    write_table(subject_metrics, output_dir / "retrieval_subject_metrics.tsv")
    write_table(qc, output_dir / "retrieval_geometry_qc.tsv")
    write_table(failures, output_dir / "retrieval_geometry_failures.tsv")
    write_table(group, output_dir / "retrieval_geometry_group_fdr.tsv")
    write_table(group, tables_si / "table_retrieval_geometry.tsv")

    save_json(
        {
            "roi_sets": args.roi_sets,
            "n_subjects_with_retrieval_patterns": len(subjects),
            "subjects": [path.name for path in subjects],
            "n_item_rows": int(len(item_metrics)),
            "n_pair_rows": int(len(pair_metrics)),
            "n_subject_metric_rows": int(len(subject_metrics)),
            "n_failures": int(len(failures)),
        },
        output_dir / "retrieval_geometry_manifest.json",
    )


if __name__ == "__main__":
    main()
