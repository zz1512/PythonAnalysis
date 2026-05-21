#!/usr/bin/env python3
"""Learning-stage decomposition for run3/run4 condition geometry.

This script mirrors the IO and ROI-selection style of
`learning_condition_rdm.py`, but restricts the analysis to the learning
stage and decomposes the summary signal into interpretable components.

For each subject x ROI x run in {3, 4}:
  - split `learn_yy` / `learn_kj` trials by the metadata `run` column,
  - compute the correlation-distance RDM on the concatenated trial set,
  - summarize:
      between
      within_yy
      within_kj
      within_mean
      between_minus_within = between - within_mean

Outputs (under
`paper_output_root / "qc" / f"learning_stage_decomposition_{roi_tag}"`):
  - learning_stage_decomposition_{roi_tag}.tsv
  - learning_stage_decomposition_group_one_sample.tsv
  - learning_stage_decomposition_group_run4_minus_run3.tsv
  - learning_stage_decomposition_group_within_yy_minus_within_kj.tsv
  - learning_stage_decomposition_qc.tsv
  - learning_stage_decomposition_manifest.json
  - paper_output_root/tables_si/table_learning_stage_decomposition_{roi_tag}.tsv
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

from common.final_utils import (  # noqa: E402
    ensure_dir,
    paired_t_summary,
    read_table,
    save_json,
    write_table,
)
from common.pattern_metrics import (  # noqa: E402
    correlation_distance_rdm,
    load_4d_data,
    load_mask,
)
from common.roi_library import (  # noqa: E402
    current_roi_set,
    default_roi_tagged_out_dir,
    sanitize_roi_tag,
    select_roi_masks,
)


CONDITIONS = ["yy", "kj"]
LEARNING_RUNS = [3, 4]
METRIC_ORDER = [
    "between",
    "within_yy",
    "within_kj",
    "within_mean",
    "between_minus_within",
]


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


def _cohens_dz_one(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def _one_sample(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
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
        "cohens_dz": _cohens_dz_one(arr),
    }


def _load_learning_condition(
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
            "cannot split learning trials into run3/run4."
        )
    _, data = load_4d_data(image_path)
    meta["run"] = pd.to_numeric(meta["run"], errors="coerce")
    return meta, data, image_path


def _masked_samples(
    data: np.ndarray,
    mask: np.ndarray,
    image_path: Path,
    mask_path: Path,
) -> np.ndarray:
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def _split_by_run(meta: pd.DataFrame, samples: np.ndarray, run_value: int) -> np.ndarray:
    if len(meta) != samples.shape[0]:
        raise ValueError(
            f"Volume/metadata mismatch: {samples.shape[0]} volumes vs {len(meta)} rows"
        )
    run_mask = meta["run"].to_numpy(dtype=float) == float(run_value)
    return samples[run_mask]


def _within_between_from_rdm(rdm: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n_trials = rdm.shape[0]
    iu, ju = np.triu_indices(n_trials, k=1)
    lab_i = labels[iu]
    lab_j = labels[ju]
    values = rdm[iu, ju]

    yy_mask = (lab_i == "yy") & (lab_j == "yy")
    kj_mask = (lab_i == "kj") & (lab_j == "kj")
    between_mask = lab_i != lab_j

    def _mean_or_nan(vec: np.ndarray) -> float:
        if vec.size == 0:
            return float("nan")
        return float(np.nanmean(vec))

    within_yy = _mean_or_nan(values[yy_mask])
    within_kj = _mean_or_nan(values[kj_mask])
    between = _mean_or_nan(values[between_mask])

    within_components = [v for v in (within_yy, within_kj) if np.isfinite(v)]
    within_mean = float(np.mean(within_components)) if within_components else float("nan")
    between_minus_within = (
        between - within_mean
        if np.isfinite(between) and np.isfinite(within_mean)
        else float("nan")
    )

    return {
        "between": between,
        "within_yy": within_yy,
        "within_kj": within_kj,
        "within_mean": within_mean,
        "between_minus_within": between_minus_within,
    }


def _compute_decomposition_metrics(
    yy_samples: np.ndarray,
    kj_samples: np.ndarray,
) -> dict[str, float]:
    if yy_samples.shape[0] == 0 or kj_samples.shape[0] == 0:
        raise ValueError(
            f"Need >=1 trial in each condition, got yy={yy_samples.shape[0]}, "
            f"kj={kj_samples.shape[0]}"
        )
    if yy_samples.shape[1] != kj_samples.shape[1]:
        raise ValueError(
            f"Voxel count mismatch between yy/kj samples: "
            f"{yy_samples.shape[1]} vs {kj_samples.shape[1]}"
        )
    samples = np.vstack([yy_samples, kj_samples])
    labels = np.asarray(["yy"] * yy_samples.shape[0] + ["kj"] * kj_samples.shape[0])
    rdm = correlation_distance_rdm(samples)
    if rdm.shape[0] != samples.shape[0]:
        raise ValueError(
            "correlation_distance_rdm removed constant trials; label alignment "
            "would be broken"
        )
    return _within_between_from_rdm(rdm, labels)


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name

    try:
        yy_meta, yy_data, yy_image = learn_cache["yy"]
        kj_meta, kj_data, kj_image = learn_cache["kj"]
        _, mask = load_mask(mask_path)
        yy_samples_all = _masked_samples(yy_data, mask, yy_image, mask_path)
        kj_samples_all = _masked_samples(kj_data, mask, kj_image, mask_path)
    except Exception as exc:
        for run_value in LEARNING_RUNS:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "run": run_value,
                    "ok": False,
                    "fail_reason": f"load: {exc}",
                    "n_yy_trials": 0,
                    "n_kj_trials": 0,
                }
            )
        return subject_rows, qc_rows

    for run_value in LEARNING_RUNS:
        try:
            yy_run = _split_by_run(yy_meta, yy_samples_all, run_value)
            kj_run = _split_by_run(kj_meta, kj_samples_all, run_value)
            metrics = _compute_decomposition_metrics(yy_run, kj_run)
            subject_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "run": run_value,
                    "n_yy_trials": int(yy_run.shape[0]),
                    "n_kj_trials": int(kj_run.shape[0]),
                    **metrics,
                }
            )
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "run": run_value,
                    "ok": True,
                    "fail_reason": "",
                    "n_yy_trials": int(yy_run.shape[0]),
                    "n_kj_trials": int(kj_run.shape[0]),
                }
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "roi_set": roi_set,
                    "roi": roi_name,
                    "run": run_value,
                    "ok": False,
                    "fail_reason": f"compute: {exc}",
                    "n_yy_trials": int((yy_meta["run"] == float(run_value)).sum()),
                    "n_kj_trials": int((kj_meta["run"] == float(run_value)).sum()),
                }
            )

    return subject_rows, qc_rows


def _metric_long_frame(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    if subject_metrics.empty:
        return pd.DataFrame()
    return subject_metrics.melt(
        id_vars=["subject", "roi_set", "roi", "run", "n_yy_trials", "n_kj_trials"],
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="value",
    )


def summarize_one_sample(metric_long: pd.DataFrame) -> pd.DataFrame:
    if metric_long.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, subset in metric_long.groupby(["roi_set", "roi", "run", "metric"], sort=False):
        roi_set, roi, run_value, metric = keys
        summary = _one_sample(subset["value"].to_numpy(dtype=float))
        rows.append(
            {
                "analysis_type": "one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "run": int(run_value),
                "metric": metric,
                **summary,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_roi_set"] = np.nan
    for _, idx in out.groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(out.loc[idx, "p"])
    metric_rank = {name: idx for idx, name in enumerate(METRIC_ORDER)}
    out["metric_order"] = out["metric"].map(metric_rank)
    return out.sort_values(
        ["roi_set", "run", "metric_order", "q_bh_within_roi_set", "p"],
        na_position="last",
    ).drop(columns=["metric_order"])


def summarize_run4_minus_run3(metric_long: pd.DataFrame) -> pd.DataFrame:
    if metric_long.empty:
        return pd.DataFrame()
    pivot = metric_long.pivot_table(
        index=["subject", "roi_set", "roi", "metric"],
        columns="run",
        values="value",
        aggfunc="mean",
    ).reset_index()
    if 3 not in pivot.columns or 4 not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (roi_set, roi, metric), subset in pivot.groupby(["roi_set", "roi", "metric"], sort=False):
        summary = paired_t_summary(subset[4], subset[3])
        rows.append(
            {
                "analysis_type": "run4_minus_run3_paired_t",
                "roi_set": roi_set,
                "roi": roi,
                "metric": metric,
                "n_subjects": int(summary["n"]),
                "mean_run4": summary["mean_a"],
                "mean_run3": summary["mean_b"],
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
    out["q_bh_within_roi_set"] = np.nan
    for _, idx in out.groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(out.loc[idx, "p"])
    metric_rank = {name: idx for idx, name in enumerate(METRIC_ORDER)}
    out["metric_order"] = out["metric"].map(metric_rank)
    return out.sort_values(
        ["roi_set", "metric_order", "q_bh_within_roi_set", "p"],
        na_position="last",
    ).drop(columns=["metric_order"])


def summarize_within_yy_minus_within_kj(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    if subject_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, subset in subject_metrics.groupby(["roi_set", "roi", "run"], sort=False):
        roi_set, roi, run_value = keys
        summary = paired_t_summary(subset["within_yy"], subset["within_kj"])
        rows.append(
            {
                "analysis_type": "within_yy_minus_within_kj_paired_t",
                "roi_set": roi_set,
                "roi": roi,
                "run": int(run_value),
                "n_subjects": int(summary["n"]),
                "mean_within_yy": summary["mean_a"],
                "mean_within_kj": summary["mean_b"],
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
    out["q_bh_within_roi_set"] = np.nan
    for _, idx in out.groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["roi_set", "run", "q_bh_within_roi_set", "p"], na_position="last")


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Learning-stage decomposition of run3/run4 condition geometry."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. If not set, uses default ROI-tagged path.",
    )
    args = parser.parse_args()

    roi_set_env = current_roi_set()
    roi_tag = sanitize_roi_tag(roi_set_env)

    if args.output_dir:
        output_dir = ensure_dir(args.output_dir)
    else:
        output_dir = ensure_dir(
            default_roi_tagged_out_dir(
                args.paper_output_root / "qc",
                "learning_stage_decomposition",
                roi_set=roi_set_env,
            )
        )
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    roi_masks_by_set: dict[str, dict[str, Path]] = {}
    for roi_set in args.roi_sets:
        roi_masks_by_set[roi_set] = select_roi_masks(
            args.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa"
        )

    subjects = sorted([path for path in args.pattern_root.glob("sub-*") if path.is_dir()])
    subjects = [
        sub
        for sub in subjects
        if (sub / "learn_yy.nii.gz").exists() and (sub / "learn_kj.nii.gz").exists()
    ]

    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in subjects:
        learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        load_failed = False
        for condition in CONDITIONS:
            try:
                learn_cache[condition] = _load_learning_condition(subject_dir, condition)
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
                    rows, qc = process_subject_roi(
                        subject_dir,
                        roi_set,
                        roi_name,
                        mask_path,
                        learn_cache,
                    )
                    subject_rows.extend(rows)
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

    subject_metrics = pd.DataFrame(subject_rows)
    if not subject_metrics.empty:
        subject_metrics = subject_metrics.sort_values(
            ["roi_set", "roi", "subject", "run"]
        ).reset_index(drop=True)
    qc = pd.DataFrame(qc_rows)
    if not qc.empty:
        qc = qc.sort_values(["roi_set", "roi", "subject", "run"]).reset_index(drop=True)
    failures = pd.DataFrame(failure_rows)

    metric_long = _metric_long_frame(subject_metrics)
    group_one_sample = summarize_one_sample(metric_long)
    group_run_change = summarize_run4_minus_run3(metric_long)
    group_within_condition = summarize_within_yy_minus_within_kj(subject_metrics)

    subject_path = output_dir / f"learning_stage_decomposition_{roi_tag}.tsv"
    write_table(metric_long, subject_path)
    write_table(
        group_one_sample,
        output_dir / "learning_stage_decomposition_group_one_sample.tsv",
    )
    write_table(
        group_run_change,
        output_dir / "learning_stage_decomposition_group_run4_minus_run3.tsv",
    )
    write_table(
        group_within_condition,
        output_dir / "learning_stage_decomposition_group_within_yy_minus_within_kj.tsv",
    )
    write_table(qc, output_dir / "learning_stage_decomposition_qc.tsv")
    if not failures.empty:
        write_table(failures, output_dir / "learning_stage_decomposition_failures.tsv")

    write_table(
        metric_long,
        tables_si / f"table_learning_stage_decomposition_{roi_tag}.tsv",
    )

    save_json(
        {
            "spec_id": "add-learning-stage-mechanism-analysis",
            "roi_sets": list(args.roi_sets),
            "metaphor_roi_set": roi_set_env,
            "roi_tag": roi_tag,
            "n_subjects_with_learn_patterns": len(subjects),
            "subjects": [path.name for path in subjects],
            "learning_runs": LEARNING_RUNS,
            "metrics": METRIC_ORDER,
            "n_subject_rows": int(len(metric_long)),
            "n_qc_rows": int(len(qc)),
            "n_failures": int(len(failures)),
            "outputs": {
                "subject_tsv": str(subject_path),
                "table_si_tsv": str(
                    tables_si / f"table_learning_stage_decomposition_{roi_tag}.tsv"
                ),
                "group_one_sample_tsv": str(
                    output_dir / "learning_stage_decomposition_group_one_sample.tsv"
                ),
                "group_run4_minus_run3_tsv": str(
                    output_dir / "learning_stage_decomposition_group_run4_minus_run3.tsv"
                ),
                "group_within_yy_minus_within_kj_tsv": str(
                    output_dir
                    / "learning_stage_decomposition_group_within_yy_minus_within_kj.tsv"
                ),
                "qc_tsv": str(output_dir / "learning_stage_decomposition_qc.tsv"),
            },
        },
        output_dir / "learning_stage_decomposition_manifest.json",
    )


if __name__ == "__main__":
    main()
