#!/usr/bin/env python3
"""Learning (run3/run4) condition-level RDM primary metric (Task 1 + Task 2).

This script consumes pre-stacked LSS patterns produced by
`representation_analysis/stack_patterns.py`:

  pattern_root/sub-XX/learn_yy.nii.gz   + learn_yy_metadata.tsv
  pattern_root/sub-XX/learn_kj.nii.gz   + learn_kj_metadata.tsv
  pattern_root/sub-XX/{pre,post,retrieval}_{yy,kj}.nii.gz (+ metadata)

For each subject x ROI x run in {3, 4}:
  - Concatenate the yy / kj trials (belonging to that run) along the trial axis.
  - Compute the correlation-distance RDM via
    `common.pattern_metrics.correlation_distance_rdm` (same operator used by the
    pre / post / retrieval geometry to keep the five trajectory points
    isomorphic).
  - within_yy  = mean upper-triangle RDM over yy-yy pairs
    within_kj  = mean upper-triangle RDM over kj-kj pairs
    within     = 0.5 * (within_yy + within_kj)
    between    = mean RDM over yy-kj cross pairs
    between_minus_within = between - within.

Task 2 then re-applies the same pipeline to pre / post / retrieval (phase-wise,
no run split) and produces a tidy 5-point trajectory table with phases
{pre, run3, run4, post, retrieval}, followed by one-sample / paired group
statistics with BH-FDR correction within each ROI set family.

Outputs (under `paper_output_root / "qc" / f"learning_condition_rdm_{roi_tag}"`):
  - learning_condition_rdm_subject.tsv        (Task 1 long table)
  - four_phase_trajectory_{roi_tag}.csv       (Task 2 tidy CSV)
  - four_phase_trajectory_group_one_sample.tsv
  - four_phase_trajectory_group_pairwise.tsv
  - learning_condition_rdm_qc.tsv
  - learning_condition_rdm_manifest.json
  - paper_output_root/tables_si/table_four_phase_trajectory_{roi_tag}.tsv
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
PHASE_ORDER = ["pre", "run3", "run4", "post", "retrieval"]
WORD_LEVEL_PHASES = ["pre", "post", "retrieval"]
LEARNING_PHASES = ["run3", "run4"]
WORD_LEVEL_CONTRASTS = [
    ("post", "pre"),
    ("retrieval", "post"),
    ("retrieval", "pre"),
]
LEARNING_CONTRASTS = [
    ("run4", "run3"),
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


def _load_metadata(path: Path) -> pd.DataFrame:
    frame = read_table(path).reset_index(drop=True)
    return frame


def _masked_samples(image_path: Path, mask_path: Path) -> np.ndarray:
    """Return (n_trials, n_voxels) matrix without re-reading the mask repeatedly.

    We go through `load_4d_data` + `load_mask` instead of `load_masked_samples`
    so the caller can share the loaded arrays (mask is loaded once per ROI by
    the caller, image is loaded once per subject/condition).
    """
    _, data = load_4d_data(image_path)
    _, mask = load_mask(mask_path)
    if data.shape[:3] != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image_path} ({data.shape[:3]}) vs "
            f"{mask_path} ({mask.shape})"
        )
    return data[mask, :].T.astype(np.float64, copy=False)


def _within_between_from_rdm(rdm: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n = rdm.shape[0]
    iu, ju = np.triu_indices(n, k=1)
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
    if within_components:
        within = float(np.mean(within_components))
    else:
        within = float("nan")

    if np.isfinite(between) and np.isfinite(within):
        diff = between - within
    else:
        diff = float("nan")

    return {
        "within_yy": within_yy,
        "within_kj": within_kj,
        "within": within,
        "between": between,
        "between_minus_within": diff,
    }


def _compute_condition_rdm_metrics(
    yy_samples: np.ndarray,
    kj_samples: np.ndarray,
) -> dict[str, float]:
    """Concatenate yy/kj trials and compute within/between distance metrics."""
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
    labels = np.asarray(
        ["yy"] * yy_samples.shape[0] + ["kj"] * kj_samples.shape[0]
    )
    rdm = correlation_distance_rdm(samples)
    # `correlation_distance_rdm` drops zero-norm trials, so the resulting size
    # may be smaller than n_trials. In that case we can no longer trust the
    # label alignment -> refuse silently and raise.
    if rdm.shape[0] != samples.shape[0]:
        raise ValueError(
            "correlation_distance_rdm removed constant trials; label alignment "
            "would be broken"
        )
    return _within_between_from_rdm(rdm, labels)


def _split_by_run(meta: pd.DataFrame, samples: np.ndarray, run_value: int) -> np.ndarray:
    if "run" not in meta.columns:
        raise KeyError(
            "metadata is missing required column 'run'; cannot split learning "
            "trials into run3/run4"
        )
    run_series = pd.to_numeric(meta["run"], errors="coerce")
    mask = run_series.to_numpy() == run_value
    return samples[mask]


def _load_phase_samples(
    subject_dir: Path,
    phase: str,
    condition: str,
    mask_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    image_path = subject_dir / f"{phase}_{condition}.nii.gz"
    meta_path = subject_dir / f"{phase}_{condition}_metadata.tsv"
    if not image_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {phase}_{condition} image or metadata for {subject_dir.name}"
        )
    meta = _load_metadata(meta_path)
    _, data = load_4d_data(image_path)
    _, mask = load_mask(mask_path)
    if data.shape[:3] != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image_path} vs {mask_path}"
        )
    if data.shape[3] != len(meta):
        raise ValueError(
            f"Volume/metadata mismatch: {image_path} has {data.shape[3]} "
            f"volumes, metadata has {len(meta)} rows"
        )
    samples = data[mask, :].T.astype(np.float64, copy=False)
    return meta, samples


def _process_learning_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Task 1: compute run3/run4 learning condition RDM metrics."""
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name

    try:
        yy_meta, yy_samples = _load_phase_samples(subject_dir, "learn", "yy", mask_path)
        kj_meta, kj_samples = _load_phase_samples(subject_dir, "learn", "kj", mask_path)
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
            yy_run = _split_by_run(yy_meta, yy_samples, run_value)
            kj_run = _split_by_run(kj_meta, kj_samples, run_value)
            metrics = _compute_condition_rdm_metrics(yy_run, kj_run)
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
                    "n_yy_trials": int(yy_meta.shape[0]) if yy_meta is not None else 0,
                    "n_kj_trials": int(kj_meta.shape[0]) if kj_meta is not None else 0,
                }
            )

    return subject_rows, qc_rows


def _compute_phase_metrics(
    subject_dir: Path,
    phase: str,
    mask_path: Path,
) -> dict[str, float]:
    """Compute the same within/between/between_minus_within for a non-learning
    phase (pre / post / retrieval)."""
    yy_meta, yy_samples = _load_phase_samples(subject_dir, phase, "yy", mask_path)
    kj_meta, kj_samples = _load_phase_samples(subject_dir, phase, "kj", mask_path)
    metrics = _compute_condition_rdm_metrics(yy_samples, kj_samples)
    metrics["n_yy_trials"] = int(yy_samples.shape[0])
    metrics["n_kj_trials"] = int(kj_samples.shape[0])
    # touch meta variables so linters keep them (they hold row counts).
    _ = len(yy_meta), len(kj_meta)
    return metrics


def build_four_phase_trajectory(
    learning_subject_rows: list[dict[str, object]],
    subjects: list[Path],
    roi_masks_by_set: dict[str, dict[str, Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Task 2: assemble the five-phase tidy trajectory and associated QC.

    Pre / post / retrieval metrics are re-computed with the same isomorphic
    formula (correlation_distance_rdm -> within/between) so that all five
    trajectory points live on the same scale.
    """
    trajectory_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    # Pre-fill trajectory with learning run3 / run4 results.
    for row in learning_subject_rows:
        trajectory_rows.append(
            {
                "subject": row["subject"],
                "roi_set": row["roi_set"],
                "roi": row["roi"],
                "phase": f"run{int(row['run'])}",
                "within_yy": row["within_yy"],
                "within_kj": row["within_kj"],
                "within": row["within"],
                "between": row["between"],
                "between_minus_within": row["between_minus_within"],
            }
        )

    # Pre / post / retrieval: recompute with identical operator.
    for subject_dir in subjects:
        subject = subject_dir.name
        for roi_set, masks in roi_masks_by_set.items():
            for roi_name, mask_path in masks.items():
                for phase in ["pre", "post", "retrieval"]:
                    try:
                        metrics = _compute_phase_metrics(subject_dir, phase, mask_path)
                        trajectory_rows.append(
                            {
                                "subject": subject,
                                "roi_set": roi_set,
                                "roi": roi_name,
                                "phase": phase,
                                "within_yy": metrics["within_yy"],
                                "within_kj": metrics["within_kj"],
                                "within": metrics["within"],
                                "between": metrics["between"],
                                "between_minus_within": metrics["between_minus_within"],
                            }
                        )
                    except FileNotFoundError as exc:
                        qc_rows.append(
                            {
                                "subject": subject,
                                "roi_set": roi_set,
                                "roi": roi_name,
                                "phase": phase,
                                "ok": False,
                                "fail_reason": f"missing: {exc}",
                            }
                        )
                    except Exception as exc:
                        qc_rows.append(
                            {
                                "subject": subject,
                                "roi_set": roi_set,
                                "roi": roi_name,
                                "phase": phase,
                                "ok": False,
                                "fail_reason": f"compute: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                        )

    return trajectory_rows, qc_rows


def _subset_phase_frame(frame: pd.DataFrame, phases: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame[frame["phase"].astype(str).isin(phases)].copy()
    if out.empty:
        return out
    out["phase"] = pd.Categorical(out["phase"], categories=phases, ordered=True)
    return out.sort_values(["roi_set", "roi", "subject", "phase"]).reset_index(drop=True)


def _group_one_sample_table(frame: pd.DataFrame, *, phases: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, subset in frame.groupby(["roi_set", "roi", "phase"], sort=False):
        roi_set, roi, phase = keys
        summary = _one_sample(subset["between_minus_within"].to_numpy(dtype=float))
        rows.append(
            {
                "analysis_type": "one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "phase": phase,
                "metric": "between_minus_within",
                **summary,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["q_bh_within_roi_set"] = np.nan
    for _, idx in frame.groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        frame.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(frame.loc[idx, "p"])
    frame["phase"] = pd.Categorical(frame["phase"], categories=phases, ordered=True)
    return frame.sort_values(["roi_set", "phase", "q_bh_within_roi_set", "p"], na_position="last")


def _group_pairwise_table(
    frame: pd.DataFrame,
    *,
    phases: list[str],
    pairwise_contrasts: list[tuple[str, str]],
    module_label: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index=["roi_set", "roi", "subject"],
        columns="phase",
        values="between_minus_within",
        aggfunc="mean",
    ).reset_index()
    rows: list[dict[str, object]] = []
    for (roi_set, roi), subset in pivot.groupby(["roi_set", "roi"], sort=False):
        for phase_a, phase_b in pairwise_contrasts:
            if phase_a not in subset.columns or phase_b not in subset.columns:
                continue
            a_values = pd.to_numeric(subset[phase_a], errors="coerce").to_numpy(dtype=float)
            b_values = pd.to_numeric(subset[phase_b], errors="coerce").to_numpy(dtype=float)
            summary = paired_t_summary(a_values, b_values)
            rows.append(
                {
                    "analysis_type": "paired_t",
                    "module": module_label,
                    "roi_set": roi_set,
                    "roi": roi,
                    "metric": "between_minus_within",
                    "phase_a": phase_a,
                    "phase_b": phase_b,
                    "n": summary.get("n", float("nan")),
                    "mean_a": summary.get("mean_a", float("nan")),
                    "mean_b": summary.get("mean_b", float("nan")),
                    "mean_diff": (
                        summary.get("mean_a", float("nan"))
                        - summary.get("mean_b", float("nan"))
                        if np.isfinite(summary.get("mean_a", float("nan")))
                        and np.isfinite(summary.get("mean_b", float("nan")))
                        else float("nan")
                    ),
                    "t": summary.get("t", float("nan")),
                    "p": summary.get("p", float("nan")),
                    "cohens_dz": summary.get("cohens_dz", float("nan")),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["q_bh_within_roi_set"] = np.nan
    for _, idx in frame.groupby(["roi_set"], dropna=False).groups.items():
        idx = list(idx)
        frame.loc[idx, "q_bh_within_roi_set"] = _bh_fdr(frame.loc[idx, "p"])
    phase_rank = {phase: idx for idx, phase in enumerate(phases)}
    frame["phase_a_order"] = frame["phase_a"].map(phase_rank)
    frame["phase_b_order"] = frame["phase_b"].map(phase_rank)
    return frame.sort_values(
        ["roi_set", "roi", "phase_a_order", "phase_b_order", "q_bh_within_roi_set", "p"],
        na_position="last",
    ).drop(columns=["phase_a_order", "phase_b_order"])


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Learning (run3/run4) condition-level RDM primary metric "
        "and five-phase trajectory assembly."
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
                "learning_condition_rdm",
                roi_set=roi_set_env,
            )
        )
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    roi_masks_by_set: dict[str, dict[str, Path]] = {}
    for roi_set in args.roi_sets:
        roi_masks_by_set[roi_set] = select_roi_masks(
            args.roi_manifest, roi_set=roi_set, include_flag="include_in_rsa"
        )

    subjects = sorted(
        [path for path in args.pattern_root.glob("sub-*") if path.is_dir()]
    )
    # Keep subjects that at least have learning patterns; pre/post/retrieval
    # gaps are handled per-phase via QC.
    learning_subjects = [
        sub for sub in subjects
        if (sub / "learn_yy.nii.gz").exists() and (sub / "learn_kj.nii.gz").exists()
    ]

    learning_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for subject_dir in learning_subjects:
        for roi_set, masks in roi_masks_by_set.items():
            for roi_name, mask_path in masks.items():
                try:
                    rows, qc = _process_learning_subject_roi(
                        subject_dir, roi_set, roi_name, mask_path
                    )
                    learning_rows.extend(rows)
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

    learning_frame = pd.DataFrame(learning_rows)
    qc_frame = pd.DataFrame(qc_rows)
    failures_frame = pd.DataFrame(failure_rows)

    write_table(learning_frame, output_dir / "learning_condition_rdm_subject.tsv")
    write_table(qc_frame, output_dir / "learning_condition_rdm_qc.tsv")
    if not failures_frame.empty:
        write_table(failures_frame, output_dir / "learning_condition_rdm_failures.tsv")

    # -------- Task 2: five-phase trajectory --------
    # Only use learning rows that succeeded (between_minus_within is finite).
    valid_learning_rows = [
        row for row in learning_rows
        if np.isfinite(float(row.get("between_minus_within", float("nan"))))
    ]

    # Trajectory only makes sense for subjects that have every phase; we
    # compute per-subject and per-ROI, then let QC drop incomplete rows.
    traj_subjects = sorted(
        [path for path in args.pattern_root.glob("sub-*") if path.is_dir()]
    )
    trajectory_rows, trajectory_qc = build_four_phase_trajectory(
        valid_learning_rows, traj_subjects, roi_masks_by_set
    )

    trajectory_frame = pd.DataFrame(trajectory_rows)
    if not trajectory_frame.empty:
        trajectory_frame["phase"] = pd.Categorical(
            trajectory_frame["phase"], categories=PHASE_ORDER, ordered=True
        )
        trajectory_frame = trajectory_frame.sort_values(
            ["roi_set", "roi", "subject", "phase"]
        ).reset_index(drop=True)

    trajectory_csv = output_dir / f"four_phase_trajectory_{roi_tag}.csv"
    write_table(trajectory_frame, trajectory_csv)
    write_table(
        trajectory_frame,
        tables_si / f"table_four_phase_trajectory_{roi_tag}.tsv",
    )

    # -------- 主文拆成两个模块：word-level / learning-level --------
    word_level_frame = _subset_phase_frame(trajectory_frame, WORD_LEVEL_PHASES)
    learning_profile_frame = _subset_phase_frame(trajectory_frame, LEARNING_PHASES)

    write_table(word_level_frame, output_dir / f"word_level_profile_{roi_tag}.csv")
    write_table(
        word_level_frame,
        tables_si / f"table_word_level_profile_{roi_tag}.tsv",
    )
    write_table(
        learning_profile_frame,
        output_dir / f"learning_profile_{roi_tag}.csv",
    )
    write_table(
        learning_profile_frame,
        tables_si / f"table_learning_profile_{roi_tag}.tsv",
    )

    if trajectory_qc:
        write_table(
            pd.DataFrame(trajectory_qc),
            output_dir / "four_phase_trajectory_qc.tsv",
        )

    one_sample_frame = _group_one_sample_table(trajectory_frame, phases=PHASE_ORDER)
    write_table(one_sample_frame, output_dir / "four_phase_trajectory_group_one_sample.tsv")

    pairwise_frame = _group_pairwise_table(
        trajectory_frame,
        phases=PHASE_ORDER,
        pairwise_contrasts=WORD_LEVEL_CONTRASTS + LEARNING_CONTRASTS,
        module_label="combined_compatibility",
    )
    write_table(pairwise_frame, output_dir / "four_phase_trajectory_group_pairwise.tsv")

    word_level_one_sample = _group_one_sample_table(
        word_level_frame, phases=WORD_LEVEL_PHASES
    )
    write_table(
        word_level_one_sample,
        output_dir / "word_level_profile_group_one_sample.tsv",
    )
    word_level_pairwise = _group_pairwise_table(
        word_level_frame,
        phases=WORD_LEVEL_PHASES,
        pairwise_contrasts=WORD_LEVEL_CONTRASTS,
        module_label="word_level",
    )
    write_table(
        word_level_pairwise,
        output_dir / "word_level_profile_group_pairwise.tsv",
    )

    learning_one_sample = _group_one_sample_table(
        learning_profile_frame, phases=LEARNING_PHASES
    )
    write_table(
        learning_one_sample,
        output_dir / "learning_profile_group_one_sample.tsv",
    )
    learning_pairwise = _group_pairwise_table(
        learning_profile_frame,
        phases=LEARNING_PHASES,
        pairwise_contrasts=LEARNING_CONTRASTS,
        module_label="learning_sentence_level",
    )
    write_table(
        learning_pairwise,
        output_dir / "learning_profile_group_pairwise.tsv",
    )

    save_json(
        {
            "spec_id": "extend-learning-rsa-trajectory",
            "metaphor_roi_set": roi_set_env,
            "roi_tag": roi_tag,
            "roi_sets": list(args.roi_sets),
            "n_learning_subjects": len(learning_subjects),
            "learning_subjects": [path.name for path in learning_subjects],
            "n_trajectory_subjects": len(traj_subjects),
            "n_learning_rows": int(len(learning_frame)),
            "n_trajectory_rows": int(len(trajectory_frame)),
            "n_word_level_rows": int(len(word_level_frame)),
            "n_learning_profile_rows": int(len(learning_profile_frame)),
            "n_qc_rows": int(len(qc_frame)),
            "n_failures": int(len(failures_frame)),
            "phases": PHASE_ORDER,
            "word_level_phases": WORD_LEVEL_PHASES,
            "learning_phases": LEARNING_PHASES,
            "word_level_contrasts": [list(pair) for pair in WORD_LEVEL_CONTRASTS],
            "learning_contrasts": [list(pair) for pair in LEARNING_CONTRASTS],
            "outputs": {
                "learning_subject_tsv": str(
                    output_dir / "learning_condition_rdm_subject.tsv"
                ),
                "trajectory_csv": str(trajectory_csv),
                "trajectory_table_si": str(
                    tables_si / f"table_four_phase_trajectory_{roi_tag}.tsv"
                ),
                "one_sample_tsv": str(
                    output_dir / "four_phase_trajectory_group_one_sample.tsv"
                ),
                "pairwise_tsv": str(
                    output_dir / "four_phase_trajectory_group_pairwise.tsv"
                ),
                "word_level_csv": str(output_dir / f"word_level_profile_{roi_tag}.csv"),
                "word_level_one_sample_tsv": str(
                    output_dir / "word_level_profile_group_one_sample.tsv"
                ),
                "word_level_pairwise_tsv": str(
                    output_dir / "word_level_profile_group_pairwise.tsv"
                ),
                "learning_profile_csv": str(
                    output_dir / f"learning_profile_{roi_tag}.csv"
                ),
                "learning_profile_one_sample_tsv": str(
                    output_dir / "learning_profile_group_one_sample.tsv"
                ),
                "learning_profile_pairwise_tsv": str(
                    output_dir / "learning_profile_group_pairwise.tsv"
                ),
                "qc_tsv": str(output_dir / "learning_condition_rdm_qc.tsv"),
            },
        },
        output_dir / "learning_condition_rdm_manifest.json",
    )


if __name__ == "__main__":
    main()
