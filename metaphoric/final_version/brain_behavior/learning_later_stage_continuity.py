#!/usr/bin/env python3
"""Learning -> pre/post/retrieval cross-stage continuity ERS.

该脚本把 learning 阶段形成的 sentence-level pattern，与 later-stage
(`pre`, `post`, `retrieval`) 的 constituent-word pair pattern 做 item-level
continuity 配对，用于解释：

- learning trace 是否已在 `post` 阶段被部分调动；
- 这种调动是否弱于 `retrieval`、强于 `pre`；
- 现有 MVPA / RSA 阶段差异是否兼容一个 continuity gradient。

核心输出（默认写入 ``$paper/qc/learning_later_stage_continuity_{roi_tag}/``）：
- `learning_later_stage_continuity_item.tsv`
- `learning_later_stage_continuity_group_one_sample.tsv`
- `learning_later_stage_continuity_group_stage_contrasts.tsv`
- `learning_later_stage_continuity_group_yy_minus_kj.tsv`
- `learning_later_stage_continuity_specificity_group_one_sample.tsv`
- `learning_later_stage_continuity_specificity_group_stage_contrasts.tsv`
- `learning_later_stage_continuity_specificity_group_yy_minus_kj.tsv`
- `learning_later_stage_continuity_itemlevel_gee.tsv`
- `learning_later_stage_continuity_qc.tsv`
- `learning_later_stage_continuity_manifest.json`

later-stage continuity 是对已有结果的解释性 bridge / control analysis，
不替代原有主机制。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

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

from common.final_utils import ensure_dir, paired_t_summary, save_json, write_table  # noqa: E402
from common.pattern_metrics import load_4d_data, load_mask  # noqa: E402
from common.roi_library import current_roi_set, sanitize_roi_tag, select_roi_masks  # noqa: E402


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
LEARN_RUNS = [3, 4]
LATER_STAGES = ["pre", "post", "retrieval"]
VARIANTS = ["run3_to_stage", "run4_to_stage", "max_run3_run4_to_stage"]
STAGE_CONTRASTS = [
    ("post", "pre", "post_minus_pre"),
    ("retrieval", "post", "retrieval_minus_post"),
    ("retrieval", "pre", "retrieval_minus_pre"),
]


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


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    sd = float(valid.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - float(valid.mean())) / sd


def _normalize_pair_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        numeric = float(text)
    except Exception:
        return text
    if math.isnan(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _masked_samples(
    data: np.ndarray, mask: np.ndarray, image_path: Path, mask_path: Path
) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _load_stage(
    subject_dir: Path, stage: str, condition: str
) -> tuple[pd.DataFrame, np.ndarray, Path]:
    image_path = subject_dir / f"{stage}_{condition}.nii.gz"
    meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {stage}_{condition} image or metadata for {subject_dir.name}"
        )
    meta = _read_any(meta_path).reset_index(drop=True)
    _, data = load_4d_data(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(
            f"Volume/metadata mismatch: {image_path} has {data.shape[3]} vols, meta has {len(meta)}"
        )
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["stage"] = stage
    if "pair_id" in meta.columns:
        meta["pair_id"] = meta["pair_id"].map(_normalize_pair_id)
    if "word_label" in meta.columns:
        meta["word_label"] = meta["word_label"].astype(str).str.strip()
    if stage == "learn" and "run" in meta.columns:
        meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    if "memory" in meta.columns:
        meta["memory"] = pd.to_numeric(meta["memory"], errors="coerce")
    if "action_time" in meta.columns:
        meta["action_time"] = pd.to_numeric(meta["action_time"], errors="coerce")
    return meta, data, image_path


def _mean_pattern(samples: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.full(samples.shape[1], np.nan, dtype=np.float64)
    return np.nanmean(samples[indices, :], axis=0)


def _aggregate_run_patterns(
    samples: np.ndarray, meta: pd.DataFrame, run_value: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], int]:
    run_mask = meta["run"].fillna(-1).astype(int).eq(run_value).to_numpy(dtype=bool)
    out: dict[str, np.ndarray] = {}
    info: dict[str, dict[str, object]] = {}
    duplicate_rows = 0
    run_meta = meta.loc[run_mask].reset_index(drop=True)
    run_idx = np.flatnonzero(run_mask)
    if run_meta.empty:
        return out, info, duplicate_rows
    run_pair_ids = run_meta["pair_id"].astype(str).str.strip()
    for pair_id in sorted(set(run_pair_ids) - {"", "nan"}):
        local_mask = run_pair_ids.eq(pair_id).to_numpy(dtype=bool)
        local_indices = np.flatnonzero(local_mask)
        global_indices = run_idx[local_indices]
        if global_indices.size == 0:
            continue
        if global_indices.size > 1:
            duplicate_rows += int(global_indices.size - 1)
        out[pair_id] = _mean_pattern(samples, global_indices)
        row = run_meta.iloc[int(local_indices[0])]
        info[pair_id] = {
            "n_rows": int(global_indices.size),
            "word_label_example": str(row.get("word_label", "")).strip(),
        }
    return out, info, duplicate_rows


def _aggregate_stage_pairs(
    samples: np.ndarray, meta: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], int]:
    out: dict[str, np.ndarray] = {}
    info: dict[str, dict[str, object]] = {}
    duplicate_rows = 0
    pair_ids = meta["pair_id"].astype(str).str.strip()
    for pair_id in sorted(set(pair_ids) - {"", "nan"}):
        indices = np.flatnonzero(pair_ids.eq(pair_id).to_numpy(dtype=bool))
        if indices.size == 0:
            continue
        if indices.size > 1:
            duplicate_rows += int(indices.size - 1)
        out[pair_id] = _mean_pattern(samples, indices)
        row = meta.iloc[int(indices[0])]
        info[pair_id] = {
            "n_rows": int(indices.size),
            "word_label_example": str(row.get("word_label", "")).strip(),
        }
    return out, info, duplicate_rows


def _behavior_by_pair(meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty or "pair_id" not in meta.columns:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "n_behavior_trials",
                "n_behavior_missing_memory",
                "n_behavior_nonmissing_memory",
                "n_behavior_nonmissing_log_rt_correct",
                "memory",
                "action_time",
                "word_label_example",
            ]
        )
    valid = meta[meta["pair_id"].map(_normalize_pair_id).ne("")].copy()
    valid["pair_id"] = valid["pair_id"].map(_normalize_pair_id)
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for pair_id, subset in valid.groupby("pair_id", sort=False):
        memory_values = pd.to_numeric(subset.get("memory"), errors="coerce")
        rt_values = pd.to_numeric(subset.get("action_time"), errors="coerce")
        memory_nonmissing = memory_values.dropna()
        rt_nonmissing = rt_values.dropna()
        word_label = ""
        if "word_label" in subset.columns:
            word_label = str(subset.iloc[0].get("word_label", "")).strip()
        rows.append(
            {
                "pair_id": str(pair_id),
                "n_behavior_trials": int(len(subset)),
                "n_behavior_missing_memory": int(memory_values.isna().sum()),
                "n_behavior_nonmissing_memory": int(memory_nonmissing.size),
                "n_behavior_nonmissing_log_rt_correct": int(rt_nonmissing.size),
                "memory": float(memory_nonmissing.iloc[0]) if not memory_nonmissing.empty else float("nan"),
                "action_time": float(rt_nonmissing.iloc[0]) if not rt_nonmissing.empty else float("nan"),
                "word_label_example": word_label,
            }
        )
    return pd.DataFrame(rows)


def _mean_corr_to_other_pairs(
    anchor_pattern: np.ndarray,
    target_patterns: dict[str, np.ndarray],
    *,
    exclude_pair_id: str,
) -> tuple[float, int]:
    values: list[float] = []
    n_candidates = 0
    for pair_id, pattern in target_patterns.items():
        if pair_id == exclude_pair_id:
            continue
        n_candidates += 1
        corr = _corr(anchor_pattern, pattern)
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return float("nan"), int(n_candidates)
    return float(np.mean(values)), int(n_candidates)


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
    later_stage_cache: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject = subject_dir.name
    _, mask = load_mask(mask_path)
    item_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for condition in CONDITIONS:
        try:
            learn_meta, learn_data, learn_image = learn_cache[condition]
            learn_samples = _masked_samples(learn_data, mask, learn_image, mask_path)
        except Exception as exc:
            for later_stage in LATER_STAGES:
                qc_rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "later_stage": later_stage,
                        "ok": False,
                        "fail_reason": f"learn: {exc}",
                        "n_run3_pairs": 0,
                        "n_run4_pairs": 0,
                        "n_later_stage_pairs": 0,
                        "n_shared_pairs": 0,
                        "n_duplicate_rows_run3": 0,
                        "n_duplicate_rows_run4": 0,
                        "n_duplicate_rows_later_stage": 0,
                        "n_behavior_pairs": 0,
                        "n_behavior_pairs_joined": 0,
                        "n_behavior_trials_joined": 0,
                        "n_behavior_missing_memory": 0,
                        "n_behavior_nonmissing_log_rt_correct": 0,
                    }
                )
            continue

        run3_patterns, run3_info, dup3 = _aggregate_run_patterns(learn_samples, learn_meta, 3)
        run4_patterns, run4_info, dup4 = _aggregate_run_patterns(learn_samples, learn_meta, 4)

        try:
            retrieval_meta, _, _ = later_stage_cache[condition]["retrieval"]
            behavior_pairs = _behavior_by_pair(retrieval_meta)
        except Exception:
            behavior_pairs = pd.DataFrame()
        behavior_map = (
            behavior_pairs.set_index("pair_id").to_dict("index")
            if not behavior_pairs.empty
            else {}
        )
        behavior_pair_ids = set(behavior_map)

        for later_stage in LATER_STAGES:
            try:
                stage_meta, stage_data, stage_image = later_stage_cache[condition][later_stage]
                stage_samples = _masked_samples(stage_data, mask, stage_image, mask_path)
                later_patterns, later_info, dup_stage = _aggregate_stage_pairs(
                    stage_samples, stage_meta
                )
            except Exception as exc:
                qc_rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "later_stage": later_stage,
                        "ok": False,
                        "fail_reason": f"{later_stage}: {exc}",
                        "n_run3_pairs": int(len(run3_patterns)),
                        "n_run4_pairs": int(len(run4_patterns)),
                        "n_later_stage_pairs": 0,
                        "n_shared_pairs": 0,
                        "n_duplicate_rows_run3": int(dup3),
                        "n_duplicate_rows_run4": int(dup4),
                        "n_duplicate_rows_later_stage": 0,
                        "n_behavior_pairs": int(len(behavior_pair_ids)),
                        "n_behavior_pairs_joined": 0,
                        "n_behavior_trials_joined": 0,
                        "n_behavior_missing_memory": 0,
                        "n_behavior_nonmissing_log_rt_correct": 0,
                    }
                )
                continue

            shared_pairs = sorted(
                (set(run3_patterns) | set(run4_patterns)) & set(later_patterns) - {"", "nan"}
            )

            n_behavior_pairs_joined = 0
            n_behavior_trials_joined = 0
            n_behavior_missing_memory = 0
            n_behavior_nonmissing_log_rt_correct = 0

            for pair_id in shared_pairs:
                later_pattern = later_patterns[pair_id]
                ers_run3 = (
                    _corr(run3_patterns[pair_id], later_pattern)
                    if pair_id in run3_patterns
                    else float("nan")
                )
                ers_run4 = (
                    _corr(run4_patterns[pair_id], later_pattern)
                    if pair_id in run4_patterns
                    else float("nan")
                )
                finite_values = [v for v in (ers_run3, ers_run4) if np.isfinite(v)]
                ers_max = float(max(finite_values)) if finite_values else float("nan")
                mismatch_run3, n_mismatch_run3 = (
                    _mean_corr_to_other_pairs(
                        run3_patterns[pair_id], later_patterns, exclude_pair_id=pair_id
                    )
                    if pair_id in run3_patterns
                    else (float("nan"), 0)
                )
                mismatch_run4, n_mismatch_run4 = (
                    _mean_corr_to_other_pairs(
                        run4_patterns[pair_id], later_patterns, exclude_pair_id=pair_id
                    )
                    if pair_id in run4_patterns
                    else (float("nan"), 0)
                )
                use_run3_for_max = np.isfinite(ers_run3) and (
                    not np.isfinite(ers_run4) or ers_run3 >= ers_run4
                )
                if use_run3_for_max:
                    mismatch_max = mismatch_run3
                    n_mismatch_max = n_mismatch_run3
                else:
                    mismatch_max = mismatch_run4
                    n_mismatch_max = n_mismatch_run4
                match_minus_mismatch_run3 = (
                    float(ers_run3 - mismatch_run3)
                    if np.isfinite(ers_run3) and np.isfinite(mismatch_run3)
                    else float("nan")
                )
                match_minus_mismatch_run4 = (
                    float(ers_run4 - mismatch_run4)
                    if np.isfinite(ers_run4) and np.isfinite(mismatch_run4)
                    else float("nan")
                )
                match_minus_mismatch_max = (
                    float(ers_max - mismatch_max)
                    if np.isfinite(ers_max) and np.isfinite(mismatch_max)
                    else float("nan")
                )

                behavior = behavior_map.get(pair_id, {})
                if behavior:
                    n_behavior_pairs_joined += 1
                    n_behavior_trials_joined += int(behavior.get("n_behavior_trials", 0))
                    n_behavior_missing_memory += int(
                        behavior.get("n_behavior_missing_memory", 0)
                    )
                    n_behavior_nonmissing_log_rt_correct += int(
                        behavior.get("n_behavior_nonmissing_log_rt_correct", 0)
                    )

                for (
                    variant_name,
                    value,
                    mismatch_ers,
                    match_minus_mismatch,
                    n_mismatch_candidates,
                ) in (
                    (
                        "run3_to_stage",
                        ers_run3,
                        mismatch_run3,
                        match_minus_mismatch_run3,
                        n_mismatch_run3,
                    ),
                    (
                        "run4_to_stage",
                        ers_run4,
                        mismatch_run4,
                        match_minus_mismatch_run4,
                        n_mismatch_run4,
                    ),
                    (
                        "max_run3_run4_to_stage",
                        ers_max,
                        mismatch_max,
                        match_minus_mismatch_max,
                        n_mismatch_max,
                    ),
                ):
                    item_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "pair_id": pair_id,
                            "later_stage": later_stage,
                            "variant": variant_name,
                            "ers": float(value) if np.isfinite(value) else float("nan"),
                            "match_ers": float(value) if np.isfinite(value) else float("nan"),
                            "mismatch_ers": float(mismatch_ers) if np.isfinite(mismatch_ers) else float("nan"),
                            "match_minus_mismatch": (
                                float(match_minus_mismatch)
                                if np.isfinite(match_minus_mismatch)
                                else float("nan")
                            ),
                            "memory": float(behavior.get("memory", np.nan)),
                            "action_time": float(behavior.get("action_time", np.nan)),
                            "word_label": str(
                                later_info.get(pair_id, {}).get(
                                    "word_label_example",
                                    behavior.get("word_label_example", ""),
                                )
                            ).strip(),
                            "n_run3_trials": int(run3_info.get(pair_id, {}).get("n_rows", 0)),
                            "n_run4_trials": int(run4_info.get(pair_id, {}).get("n_rows", 0)),
                            "n_later_stage_trials": int(
                                later_info.get(pair_id, {}).get("n_rows", 0)
                            ),
                            "n_mismatch_candidates": int(n_mismatch_candidates),
                        }
                    )

            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "later_stage": later_stage,
                    "ok": True,
                    "fail_reason": "",
                    "n_run3_pairs": int(len(run3_patterns)),
                    "n_run4_pairs": int(len(run4_patterns)),
                    "n_later_stage_pairs": int(len(later_patterns)),
                    "n_shared_pairs": int(len(shared_pairs)),
                    "n_mismatch_candidates_total": int(
                        sum(max(len(shared_pairs) - 1, 0) for _ in shared_pairs)
                    ),
                    "n_mismatch_candidates_per_pair": int(max(len(shared_pairs) - 1, 0)),
                    "n_duplicate_rows_run3": int(dup3),
                    "n_duplicate_rows_run4": int(dup4),
                    "n_duplicate_rows_later_stage": int(dup_stage),
                    "n_behavior_pairs": int(len(behavior_pair_ids)),
                    "n_behavior_pairs_joined": int(n_behavior_pairs_joined),
                    "n_behavior_trials_joined": int(n_behavior_trials_joined),
                    "n_behavior_missing_memory": int(n_behavior_missing_memory),
                    "n_behavior_nonmissing_log_rt_correct": int(
                        n_behavior_nonmissing_log_rt_correct
                    ),
                }
            )

    return item_rows, qc_rows


def summarize_group_one_sample(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty:
        return pd.DataFrame()
    subject_mean = (
        item_metrics.dropna(subset=["ers"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "later_stage", "variant"],
            dropna=False,
        )["ers"]
        .mean()
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for keys, subset in subject_mean.groupby(
        ["roi_set", "roi", "condition", "later_stage", "variant"], sort=False
    ):
        roi_set, roi, condition, later_stage, variant = keys
        values = pd.to_numeric(subset["ers"], errors="coerce").dropna()
        if values.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(values.to_numpy(dtype=float), 0.0, nan_policy="omit")
        rows.append(
            {
                "analysis_type": "continuity_one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "later_stage": later_stage,
                "variant": variant,
                "n_subjects": int(values.size),
                "mean_ers": float(values.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                "cohens_dz": _cohens_dz(values),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(
        ["roi_set", "later_stage", "variant", "analysis_type"], dropna=False
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(
        ["later_stage", "variant", "q_bh_within_family", "p"],
        na_position="last",
    )


def summarize_specificity_group_one_sample(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty or "match_minus_mismatch" not in item_metrics.columns:
        return pd.DataFrame()
    subject_mean = (
        item_metrics.dropna(subset=["match_minus_mismatch"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "later_stage", "variant"],
            dropna=False,
        )["match_minus_mismatch"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for keys, subset in subject_mean.groupby(
        ["roi_set", "roi", "condition", "later_stage", "variant"], sort=False
    ):
        roi_set, roi, condition, later_stage, variant = keys
        values = pd.to_numeric(subset["match_minus_mismatch"], errors="coerce").dropna()
        if values.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(values.to_numpy(dtype=float), 0.0, nan_policy="omit")
        rows.append(
            {
                "analysis_type": "continuity_specificity_one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "later_stage": later_stage,
                "variant": variant,
                "n_subjects": int(values.size),
                "mean_match_minus_mismatch": float(values.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                "cohens_dz": _cohens_dz(values),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(
        ["roi_set", "later_stage", "variant", "analysis_type"], dropna=False
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(
        ["later_stage", "variant", "q_bh_within_family", "p"],
        na_position="last",
    )


def summarize_stage_contrasts(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty:
        return pd.DataFrame()
    subject_mean = (
        item_metrics.dropna(subset=["ers"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "later_stage", "variant"],
            dropna=False,
        )["ers"]
        .mean()
        .reset_index()
    )
    pivot = subject_mean.pivot_table(
        index=["subject", "roi_set", "roi", "condition", "variant"],
        columns="later_stage",
        values="ers",
        aggfunc="mean",
    ).reset_index()
    rows: list[dict[str, object]] = []
    for stage_a, stage_b, contrast_label in STAGE_CONTRASTS:
        if stage_a not in pivot.columns or stage_b not in pivot.columns:
            continue
        for keys, subset in pivot.groupby(["roi_set", "roi", "condition", "variant"], sort=False):
            roi_set, roi, condition, variant = keys
            summary = paired_t_summary(subset[stage_a], subset[stage_b])
            if summary["n"] < 2:
                continue
            rows.append(
                {
                    "analysis_type": "later_stage_paired_t",
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "variant": variant,
                    "contrast": contrast_label,
                    "later_stage_a": stage_a,
                    "later_stage_b": stage_b,
                    "n_subjects": int(summary["n"]),
                    "mean_a": summary["mean_a"],
                    "mean_b": summary["mean_b"],
                    "mean_diff": (
                        summary["mean_a"] - summary["mean_b"]
                        if np.isfinite(summary["mean_a"]) and np.isfinite(summary["mean_b"])
                        else float("nan")
                    ),
                    "t": summary["t"],
                    "p": summary["p"],
                    "cohens_dz": summary["cohens_dz"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "contrast", "variant", "condition"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["contrast", "variant", "q_bh_within_family", "p"], na_position="last")


def summarize_specificity_stage_contrasts(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty or "match_minus_mismatch" not in item_metrics.columns:
        return pd.DataFrame()
    subject_mean = (
        item_metrics.dropna(subset=["match_minus_mismatch"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "later_stage", "variant"],
            dropna=False,
        )["match_minus_mismatch"]
        .mean()
        .reset_index()
    )
    pivot = subject_mean.pivot_table(
        index=["subject", "roi_set", "roi", "condition", "variant"],
        columns="later_stage",
        values="match_minus_mismatch",
        aggfunc="mean",
    ).reset_index()
    rows: list[dict[str, object]] = []
    for stage_a, stage_b, contrast_label in STAGE_CONTRASTS:
        if stage_a not in pivot.columns or stage_b not in pivot.columns:
            continue
        for keys, subset in pivot.groupby(["roi_set", "roi", "condition", "variant"], sort=False):
            roi_set, roi, condition, variant = keys
            summary = paired_t_summary(subset[stage_a], subset[stage_b])
            if summary["n"] < 2:
                continue
            rows.append(
                {
                    "analysis_type": "continuity_specificity_paired_t",
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "variant": variant,
                    "contrast": contrast_label,
                    "later_stage_a": stage_a,
                    "later_stage_b": stage_b,
                    "n_subjects": int(summary["n"]),
                    "mean_a": summary["mean_a"],
                    "mean_b": summary["mean_b"],
                    "mean_diff": (
                        summary["mean_a"] - summary["mean_b"]
                        if np.isfinite(summary["mean_a"]) and np.isfinite(summary["mean_b"])
                        else float("nan")
                    ),
                    "t": summary["t"],
                    "p": summary["p"],
                    "cohens_dz": summary["cohens_dz"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "contrast", "variant", "condition"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["contrast", "variant", "q_bh_within_family", "p"], na_position="last")


def summarize_yy_minus_kj(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty:
        return pd.DataFrame()
    subject_mean = (
        item_metrics.dropna(subset=["ers"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "later_stage", "variant"],
            dropna=False,
        )["ers"]
        .mean()
        .reset_index()
    )
    pivot = subject_mean.pivot_table(
        index=["subject", "roi_set", "roi", "later_stage", "variant"],
        columns="condition",
        values="ers",
        aggfunc="mean",
    ).reset_index()
    if "yy" not in pivot.columns or "kj" not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, subset in pivot.groupby(["roi_set", "roi", "later_stage", "variant"], sort=False):
        roi_set, roi, later_stage, variant = keys
        diff = pd.to_numeric(subset["yy"] - subset["kj"], errors="coerce").dropna()
        if diff.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(diff.to_numpy(dtype=float), 0.0, nan_policy="omit")
        rows.append(
            {
                "analysis_type": "continuity_yy_minus_kj",
                "roi_set": roi_set,
                "roi": roi,
                "later_stage": later_stage,
                "variant": variant,
                "condition": "yy_minus_kj",
                "condition_label": "YY-KJ",
                "n_subjects": int(diff.size),
                "mean_ers": float(diff.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                "cohens_dz": _cohens_dz(diff),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "later_stage", "variant"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["later_stage", "variant", "q_bh_within_family", "p"], na_position="last")


def summarize_specificity_yy_minus_kj(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty or "match_minus_mismatch" not in item_metrics.columns:
        return pd.DataFrame()
    subject_mean = (
        item_metrics.dropna(subset=["match_minus_mismatch"])
        .groupby(
            ["subject", "roi_set", "roi", "condition", "later_stage", "variant"],
            dropna=False,
        )["match_minus_mismatch"]
        .mean()
        .reset_index()
    )
    pivot = subject_mean.pivot_table(
        index=["subject", "roi_set", "roi", "later_stage", "variant"],
        columns="condition",
        values="match_minus_mismatch",
        aggfunc="mean",
    ).reset_index()
    if "yy" not in pivot.columns or "kj" not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, subset in pivot.groupby(["roi_set", "roi", "later_stage", "variant"], sort=False):
        roi_set, roi, later_stage, variant = keys
        diff = pd.to_numeric(subset["yy"] - subset["kj"], errors="coerce").dropna()
        if diff.size < 2:
            continue
        t_val, p_val = stats.ttest_1samp(diff.to_numpy(dtype=float), 0.0, nan_policy="omit")
        rows.append(
            {
                "analysis_type": "continuity_specificity_yy_minus_kj",
                "roi_set": roi_set,
                "roi": roi,
                "later_stage": later_stage,
                "variant": variant,
                "condition": "yy_minus_kj",
                "condition_label": "YY-KJ",
                "n_subjects": int(diff.size),
                "mean_match_minus_mismatch": float(diff.mean()),
                "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                "cohens_dz": _cohens_dz(diff),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "later_stage", "variant"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["later_stage", "variant", "q_bh_within_family", "p"], na_position="last")


def _fit_gee(frame: pd.DataFrame, response: str) -> dict[str, object] | None:
    try:
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial, Gaussian
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except Exception as exc:  # pragma: no cover
        return {"status": "failed", "error": f"statsmodels missing: {exc}"}

    work = frame.dropna(subset=["ers", response, "subject", "condition"]).copy()
    if work["subject"].nunique() < 8 or work[response].nunique() < 2 or len(work) < 30:
        return None
    work["ers_z"] = _zscore(work["ers"])
    work["condition_yy"] = (work["condition"].astype(str) == "yy").astype(int)
    family = Binomial() if response == "memory" else Gaussian()
    try:
        model = GEE.from_formula(
            f"{response} ~ ers_z * condition_yy",
            groups="subject",
            data=work,
            family=family,
            cov_struct=Exchangeable(),
        )
        fit = model.fit()
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "n_trials": int(len(work)),
            "n_subjects": int(work["subject"].nunique()),
        }
    params = fit.params
    pvalues = fit.pvalues
    bse = fit.bse
    return {
        "status": "ok",
        "n_trials": int(len(work)),
        "n_subjects": int(work["subject"].nunique()),
        "ers_beta": float(params.get("ers_z", np.nan)),
        "ers_se": float(bse.get("ers_z", np.nan)),
        "ers_p": float(pvalues.get("ers_z", np.nan)),
        "condition_beta": float(params.get("condition_yy", np.nan)),
        "condition_p": float(pvalues.get("condition_yy", np.nan)),
        "interaction_beta": float(params.get("ers_z:condition_yy", np.nan)),
        "interaction_p": float(pvalues.get("ers_z:condition_yy", np.nan)),
        "family": "binomial" if response == "memory" else "gaussian",
    }


def summarize_itemlevel_gee(item_metrics: pd.DataFrame) -> pd.DataFrame:
    if item_metrics.empty:
        return pd.DataFrame()
    frame = item_metrics.copy()
    frame["log_rt_correct"] = np.nan
    valid_rt = frame["memory"].eq(1) & frame["action_time"].gt(0)
    frame.loc[valid_rt, "log_rt_correct"] = np.log(frame.loc[valid_rt, "action_time"])

    rows: list[dict[str, object]] = []
    for keys, subset in frame.groupby(["roi_set", "roi", "later_stage", "variant"], sort=False):
        roi_set, roi, later_stage, variant = keys
        for response in ("memory", "log_rt_correct"):
            fit = _fit_gee(subset, response)
            if fit is None:
                continue
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "later_stage": later_stage,
                    "variant": variant,
                    "response": response,
                    "analysis_note": "secondary behavior bridge; does not replace stage-level continuity summaries",
                    **fit,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ok = out["status"].eq("ok")
    out["q_bh_within_family"] = np.nan
    out["interaction_q_bh_within_family"] = np.nan
    for _, idx in out[ok].groupby(["roi_set", "later_stage", "variant", "response"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "ers_p"])
        out.loc[idx, "interaction_q_bh_within_family"] = _bh_fdr(
            out.loc[idx, "interaction_p"]
        )
    return out.sort_values(["later_stage", "variant", "response", "q_bh_within_family", "ers_p"], na_position="last")


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Learning -> pre/post/retrieval cross-stage continuity ERS.",
    )
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument(
        "--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv"
    )
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. If not set, uses default ROI-tagged path.",
    )
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(current_roi_set())
    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
    else:
        output_dir = ensure_dir(
            args.paper_output_root / "qc" / f"learning_later_stage_continuity_{roi_tag}"
        )
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    roi_masks_by_set = {
        roi_set: select_roi_masks(
            args.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa"
        )
        for roi_set in args.roi_sets
    }

    all_subject_dirs = sorted([path for path in args.pattern_root.glob("sub-*") if path.is_dir()])
    subjects: list[Path] = []
    excluded_subjects: list[dict[str, object]] = []
    for subject_dir in all_subject_dirs:
        required = [
            *(subject_dir / f"learn_{condition}.nii.gz" for condition in CONDITIONS),
            *(
                subject_dir / f"{stage}_{condition}.nii.gz"
                for stage in LATER_STAGES
                for condition in CONDITIONS
            ),
        ]
        missing = [str(path.name) for path in required if not path.exists()]
        if missing:
            excluded_subjects.append(
                {
                    "subject": subject_dir.name,
                    "reason": "missing_required_pattern",
                    "missing_files": ";".join(missing),
                }
            )
            continue
        subjects.append(subject_dir)

    item_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        later_stage_cache: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]] = {
            condition: {} for condition in CONDITIONS
        }
        load_error: str | None = None
        try:
            for condition in CONDITIONS:
                learn_cache[condition] = _load_stage(subject_dir, "learn", condition)
                for stage in LATER_STAGES:
                    later_stage_cache[condition][stage] = _load_stage(subject_dir, stage, condition)
        except Exception as exc:
            load_error = str(exc)
            failure_rows.append(
                {
                    "subject": subject_dir.name,
                    "roi_set": "all",
                    "roi": "all",
                    "error": load_error,
                    "traceback": traceback.format_exc(),
                }
            )
        if load_error is not None:
            continue

        for roi_set, masks in roi_masks_by_set.items():
            for roi_name, mask_path in masks.items():
                try:
                    items, qc = process_subject_roi(
                        subject_dir,
                        roi_set,
                        roi_name,
                        mask_path,
                        learn_cache,
                        later_stage_cache,
                    )
                    item_rows.extend(items)
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
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group_one_sample = summarize_group_one_sample(item_metrics)
    stage_contrasts = summarize_stage_contrasts(item_metrics)
    yy_minus_kj = summarize_yy_minus_kj(item_metrics)
    specificity_group_one_sample = summarize_specificity_group_one_sample(item_metrics)
    specificity_stage_contrasts = summarize_specificity_stage_contrasts(item_metrics)
    specificity_yy_minus_kj = summarize_specificity_yy_minus_kj(item_metrics)
    itemlevel_gee = summarize_itemlevel_gee(item_metrics)

    write_table(item_metrics, output_dir / "learning_later_stage_continuity_item.tsv")
    write_table(
        group_one_sample,
        output_dir / "learning_later_stage_continuity_group_one_sample.tsv",
    )
    write_table(
        stage_contrasts,
        output_dir / "learning_later_stage_continuity_group_stage_contrasts.tsv",
    )
    write_table(
        yy_minus_kj,
        output_dir / "learning_later_stage_continuity_group_yy_minus_kj.tsv",
    )
    write_table(
        specificity_group_one_sample,
        output_dir / "learning_later_stage_continuity_specificity_group_one_sample.tsv",
    )
    write_table(
        specificity_stage_contrasts,
        output_dir / "learning_later_stage_continuity_specificity_group_stage_contrasts.tsv",
    )
    write_table(
        specificity_yy_minus_kj,
        output_dir / "learning_later_stage_continuity_specificity_group_yy_minus_kj.tsv",
    )
    write_table(
        itemlevel_gee,
        output_dir / "learning_later_stage_continuity_itemlevel_gee.tsv",
    )
    write_table(qc, output_dir / "learning_later_stage_continuity_qc.tsv")
    write_table(failures, output_dir / "learning_later_stage_continuity_failures.tsv")
    write_table(
        group_one_sample,
        tables_si / f"table_learning_later_stage_continuity_{roi_tag}.tsv",
    )

    save_json(
        {
            "spec_id": "add-learning-later-stage-continuity-ers",
            "roi_sets": args.roi_sets,
            "roi_tag": roi_tag,
            "later_stages": LATER_STAGES,
            "variants": VARIANTS,
            "learn_runs": LEARN_RUNS,
            "n_subjects": len(subjects),
            "subjects": [path.name for path in subjects],
            "n_excluded_subjects_missing_required_patterns": len(excluded_subjects),
            "excluded_subjects_missing_required_patterns": excluded_subjects,
            "n_item_rows": int(len(item_metrics)),
            "n_group_one_sample_rows": int(len(group_one_sample)),
            "n_stage_contrast_rows": int(len(stage_contrasts)),
            "n_yy_minus_kj_rows": int(len(yy_minus_kj)),
            "n_specificity_group_one_sample_rows": int(len(specificity_group_one_sample)),
            "n_specificity_stage_contrast_rows": int(len(specificity_stage_contrasts)),
            "n_specificity_yy_minus_kj_rows": int(len(specificity_yy_minus_kj)),
            "n_behavior_rows": int(len(itemlevel_gee)),
            "n_qc_rows": int(len(qc)),
            "n_failures": int(len(failures)),
            "analysis_note": (
                "Interpretive bridge/control analysis. pre = baseline control, "
                "post = partial re-engagement stage, retrieval = strongest re-engagement candidate."
            ),
            "outputs": {
                "item": str(output_dir / "learning_later_stage_continuity_item.tsv"),
                "group_one_sample": str(
                    output_dir / "learning_later_stage_continuity_group_one_sample.tsv"
                ),
                "stage_contrasts": str(
                    output_dir / "learning_later_stage_continuity_group_stage_contrasts.tsv"
                ),
                "yy_minus_kj": str(
                    output_dir / "learning_later_stage_continuity_group_yy_minus_kj.tsv"
                ),
                "specificity_group_one_sample": str(
                    output_dir
                    / "learning_later_stage_continuity_specificity_group_one_sample.tsv"
                ),
                "specificity_stage_contrasts": str(
                    output_dir
                    / "learning_later_stage_continuity_specificity_group_stage_contrasts.tsv"
                ),
                "specificity_yy_minus_kj": str(
                    output_dir
                    / "learning_later_stage_continuity_specificity_group_yy_minus_kj.tsv"
                ),
                "itemlevel_gee": str(
                    output_dir / "learning_later_stage_continuity_itemlevel_gee.tsv"
                ),
                "qc": str(output_dir / "learning_later_stage_continuity_qc.tsv"),
                "table_si": str(
                    tables_si / f"table_learning_later_stage_continuity_{roi_tag}.tsv"
                ),
            },
        },
        output_dir / "learning_later_stage_continuity_manifest.json",
    )


if __name__ == "__main__":
    main()
