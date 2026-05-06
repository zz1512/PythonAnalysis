#!/usr/bin/env python3
"""Cross-phase constituent-to-sentence reinstatement (Task 3).

This script quantifies, for every learning-phase sentence L_i (run3 / run4),
how similar its pattern is to the two constituent patterns that make up the
same pair in another phase (pre / post / retrieval) relative to the other
pairs in the same condition:

    match_i     = mean_{p in pair_i}( corr(L_i, P_p) )
    mismatch_i  = mean_{j != i, same condition}( corr(L_i, P_{pair_j}) )
    reinstatement_i = match_i - mismatch_i

Inputs consumed (all already produced by ``stack_patterns.py``):
- ``pattern_root/sub-XX/learn_{yy,kj}.nii.gz`` + metadata with ``run``,
  ``word_label``, ``pair_id`` columns.
- ``pattern_root/sub-XX/{pre,post,retrieval}_{yy,kj}.nii.gz`` + metadata with
  ``word_label`` and ``pair_id`` columns.

Main outputs (written under
``paper_output_root/qc/cross_phase_reinstatement_{roi_tag}``):
- cross_phase_reinstatement_item.tsv
- cross_phase_reinstatement_subject.tsv
- cross_phase_reinstatement_group_one_sample.tsv
- cross_phase_reinstatement_group_run_contrast.tsv
- cross_phase_reinstatement_qc.tsv
- cross_phase_reinstatement_manifest.json

Plus a tidy mirror at
``paper_output_root/tables_si/table_cross_phase_reinstatement_{roi_tag}.tsv``.

Implements the Task 3 scenarios in
``.trae/specs/extend-learning-rsa-trajectory/spec.md``.
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
    save_json,
    write_table,
)
from common.pattern_metrics import load_4d_data, load_mask  # noqa: E402
from common.roi_library import current_roi_set, sanitize_roi_tag, select_roi_masks  # noqa: E402


CONDITIONS = ["yy", "kj"]
CONDITION_LABELS = {"yy": "YY", "kj": "KJ"}
TARGET_PHASES = ["pre", "post", "retrieval"]
LEARN_RUNS = [3, 4]


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
    _, data = load_4d_data(image_path)
    if data.shape[3] != len(meta):
        raise ValueError(
            f"Volume/metadata mismatch: {image_path} has {data.shape[3]} vols, meta has {len(meta)}"
        )
    meta["subject"] = meta.get("subject", subject_dir.name).astype(str)
    meta["condition"] = condition
    meta["stage"] = stage
    if "pair_id" in meta.columns:
        meta["pair_id"] = meta["pair_id"].astype(str).str.strip()
    if "word_label" in meta.columns:
        meta["word_label"] = meta["word_label"].astype(str).str.strip()
    return meta, data, image_path


def _load_learn(subject_dir: Path, condition: str) -> tuple[pd.DataFrame, np.ndarray, Path]:
    meta, data, image_path = _load_stage(subject_dir, "learn", condition)
    if "run" not in meta.columns:
        raise ValueError(
            f"learn_{condition}_metadata.tsv for {subject_dir.name} is missing the 'run' column"
        )
    meta["run"] = pd.to_numeric(meta["run"], errors="coerce").astype("Int64")
    return meta, data, image_path


def _phase_pair_to_patterns(meta: pd.DataFrame, samples: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Group phase patterns by pair_id (baseline_* pair_ids are naturally skipped upstream)."""
    mapping: dict[str, list[np.ndarray]] = {}
    for idx, row in meta.iterrows():
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id.lower() == "nan":
            continue
        mapping.setdefault(pair_id, []).append(samples[int(idx)])
    return mapping


def process_subject_roi(
    subject_dir: Path,
    roi_set: str,
    roi_name: str,
    mask_path: Path,
    learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]],
    phase_cache: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return (item_rows, subject_rows, qc_rows) for one subject × ROI."""

    item_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    subject = subject_dir.name
    _, mask = load_mask(mask_path)

    for condition in CONDITIONS:
        # learning
        try:
            learn_meta, learn_data, learn_image = learn_cache[condition]
            learn_samples = _masked_samples(learn_data, mask, learn_image, mask_path)
        except Exception as exc:  # pragma: no cover - defensive
            for target_phase in TARGET_PHASES:
                for run in LEARN_RUNS:
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "run": int(run),
                            "target_phase": target_phase,
                            "ok": False,
                            "n_learn_trials": 0,
                            "n_phase_patterns": 0,
                            "fail_reason": f"learn: {exc}",
                        }
                    )
            continue

        # target phases
        phase_samples: dict[str, np.ndarray] = {}
        phase_meta_map: dict[str, pd.DataFrame] = {}
        phase_pair_map: dict[str, dict[str, list[np.ndarray]]] = {}
        phase_fail: dict[str, str] = {}
        for target_phase in TARGET_PHASES:
            try:
                p_meta, p_data, p_image = phase_cache[condition][target_phase]
                p_samples = _masked_samples(p_data, mask, p_image, mask_path)
                phase_meta_map[target_phase] = p_meta
                phase_samples[target_phase] = p_samples
                phase_pair_map[target_phase] = _phase_pair_to_patterns(p_meta, p_samples)
            except Exception as exc:
                phase_fail[target_phase] = str(exc)

        for run in LEARN_RUNS:
            run_mask = (learn_meta["run"] == run).fillna(False).to_numpy(dtype=bool)
            run_indices = np.flatnonzero(run_mask)
            if run_indices.size == 0:
                for target_phase in TARGET_PHASES:
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "run": int(run),
                            "target_phase": target_phase,
                            "ok": False,
                            "n_learn_trials": 0,
                            "n_phase_patterns": int(
                                sum(len(v) for v in phase_pair_map.get(target_phase, {}).values())
                            ) if target_phase in phase_pair_map else 0,
                            "fail_reason": f"no learn rows for run={run}",
                        }
                    )
                continue

            for target_phase in TARGET_PHASES:
                if target_phase in phase_fail:
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "run": int(run),
                            "target_phase": target_phase,
                            "ok": False,
                            "n_learn_trials": int(run_indices.size),
                            "n_phase_patterns": 0,
                            "fail_reason": f"{target_phase}: {phase_fail[target_phase]}",
                        }
                    )
                    continue

                pair_map = phase_pair_map[target_phase]
                n_phase_patterns = int(sum(len(v) for v in pair_map.values()))
                if n_phase_patterns == 0:
                    qc_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "run": int(run),
                            "target_phase": target_phase,
                            "ok": False,
                            "n_learn_trials": int(run_indices.size),
                            "n_phase_patterns": 0,
                            "fail_reason": f"{target_phase}: no pair patterns",
                        }
                    )
                    continue

                reinstatement_values: list[float] = []
                for idx in run_indices:
                    row = learn_meta.iloc[int(idx)]
                    pair_id = str(row.get("pair_id", "")).strip()
                    word_label = str(row.get("word_label", "")).strip()
                    if not pair_id or pair_id.lower() == "nan":
                        continue
                    match_patterns = pair_map.get(pair_id, [])
                    mismatch_patterns: list[np.ndarray] = []
                    for pid, patterns in pair_map.items():
                        if pid == pair_id:
                            continue
                        mismatch_patterns.extend(patterns)

                    anchor = learn_samples[int(idx)]
                    match_vals = [
                        c for c in (_corr(anchor, p) for p in match_patterns) if np.isfinite(c)
                    ]
                    mismatch_vals = [
                        c for c in (_corr(anchor, p) for p in mismatch_patterns) if np.isfinite(c)
                    ]
                    match_mean = float(np.mean(match_vals)) if match_vals else float("nan")
                    mismatch_mean = (
                        float(np.mean(mismatch_vals)) if mismatch_vals else float("nan")
                    )
                    if np.isfinite(match_mean) and np.isfinite(mismatch_mean):
                        reinstatement = match_mean - mismatch_mean
                    else:
                        reinstatement = float("nan")
                    if np.isfinite(reinstatement):
                        reinstatement_values.append(reinstatement)

                    item_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "run": int(run),
                            "target_phase": target_phase,
                            "pair_id": pair_id,
                            "word_label": word_label,
                            "match": match_mean,
                            "mismatch": mismatch_mean,
                            "reinstatement": reinstatement,
                            "n_match_patterns": int(len(match_patterns)),
                            "n_mismatch_patterns": int(len(mismatch_patterns)),
                        }
                    )

                ok = len(reinstatement_values) > 0
                qc_rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi_name,
                        "condition": condition,
                        "run": int(run),
                        "target_phase": target_phase,
                        "ok": bool(ok),
                        "n_learn_trials": int(run_indices.size),
                        "n_phase_patterns": n_phase_patterns,
                        "fail_reason": "" if ok else "no finite reinstatement values",
                    }
                )
                if ok:
                    subject_rows.append(
                        {
                            "subject": subject,
                            "roi_set": roi_set,
                            "roi": roi_name,
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "run": int(run),
                            "target_phase": target_phase,
                            "reinstatement": float(np.mean(reinstatement_values)),
                            "n_trials": int(len(reinstatement_values)),
                        }
                    )

    return item_rows, subject_rows, qc_rows


def summarize_one_sample(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if subject_metrics.empty:
        return pd.DataFrame(rows)
    group_cols = ["roi_set", "roi", "condition", "condition_label", "run", "target_phase"]
    for keys, subset in subject_metrics.groupby(group_cols, sort=False):
        roi_set, roi, condition, condition_label, run, target_phase = keys
        values = pd.to_numeric(subset["reinstatement"], errors="coerce").dropna()
        summary = one_sample_t_summary(values.to_numpy(dtype=float), popmean=0.0)
        rows.append(
            {
                "analysis_type": "one_sample_gt_zero",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": condition_label,
                "run": int(run),
                "target_phase": target_phase,
                "n_subjects": int(summary.get("n", 0)),
                "mean_effect": float(summary.get("mean", float("nan"))),
                "t": float(summary.get("t", float("nan"))),
                "p": float(summary.get("p", float("nan"))),
                "cohens_dz": _cohens_dz(values),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH-FDR within roi_set × target_phase × condition family.
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(
        ["roi_set", "target_phase", "condition"], dropna=False
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_within_family", "p"], na_position="last")


def summarize_run_contrast(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if subject_metrics.empty:
        return pd.DataFrame(rows)
    pivot = subject_metrics.pivot_table(
        index=["subject", "roi_set", "roi", "condition", "condition_label", "target_phase"],
        columns="run",
        values="reinstatement",
        aggfunc="mean",
    ).reset_index()
    if 3 not in pivot.columns or 4 not in pivot.columns:
        return pd.DataFrame(rows)
    pivot["diff"] = pivot[4] - pivot[3]
    group_cols = ["roi_set", "roi", "condition", "condition_label", "target_phase"]
    for keys, subset in pivot.groupby(group_cols, sort=False):
        roi_set, roi, condition, condition_label, target_phase = keys
        a = pd.to_numeric(subset[4], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(subset[3], errors="coerce").to_numpy(dtype=float)
        summary = paired_t_summary(a, b)
        rows.append(
            {
                "analysis_type": "run4_minus_run3",
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "condition_label": condition_label,
                "target_phase": target_phase,
                "n_subjects": int(summary.get("n", 0)),
                "mean_run3": float(summary.get("mean_b", float("nan"))),
                "mean_run4": float(summary.get("mean_a", float("nan"))),
                "mean_diff": float(
                    pd.to_numeric(subset["diff"], errors="coerce").mean()
                ),
                "t": float(summary.get("t", float("nan"))),
                "p": float(summary.get("p", float("nan"))),
                "cohens_dz": float(summary.get("cohens_dz", float("nan"))),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_within_family"] = np.nan
    for _, idx in out.groupby(
        ["roi_set", "target_phase", "condition"], dropna=False
    ).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_within_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_within_family", "p"], na_position="last")


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="Learning ↔ pre/post/retrieval constituent-to-sentence reinstatement.",
    )
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument(
        "--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv"
    )
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument(
        "--paper-output-root", type=Path, default=base_dir / "paper_outputs"
    )
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(current_roi_set())
    output_dir = ensure_dir(
        args.paper_output_root / "qc" / f"cross_phase_reinstatement_{roi_tag}"
    )
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    subjects_all = sorted(
        [path for path in args.pattern_root.glob("sub-*") if path.is_dir()]
    )
    subjects = [
        sub for sub in subjects_all
        if all((sub / f"learn_{cond}.nii.gz").exists() for cond in CONDITIONS)
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
        # Pre-load per-subject caches (learn + phase × condition)
        learn_cache: dict[str, tuple[pd.DataFrame, np.ndarray, Path]] = {}
        phase_cache: dict[str, dict[str, tuple[pd.DataFrame, np.ndarray, Path]]] = {
            cond: {} for cond in CONDITIONS
        }
        load_error: str | None = None
        try:
            for condition in CONDITIONS:
                learn_cache[condition] = _load_learn(subject_dir, condition)
                for target_phase in TARGET_PHASES:
                    phase_cache[condition][target_phase] = _load_stage(
                        subject_dir, target_phase, condition
                    )
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
                    items, subj_metrics, qc = process_subject_roi(
                        subject_dir,
                        roi_set,
                        roi_name,
                        mask_path,
                        learn_cache,
                        phase_cache,
                    )
                    item_rows.extend(items)
                    subject_rows.extend(subj_metrics)
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
    subject_metrics = pd.DataFrame(subject_rows)
    qc = pd.DataFrame(qc_rows)
    failures = pd.DataFrame(failure_rows)
    group_one_sample = summarize_one_sample(subject_metrics)
    group_run_contrast = summarize_run_contrast(subject_metrics)

    write_table(item_metrics, output_dir / "cross_phase_reinstatement_item.tsv")
    write_table(subject_metrics, output_dir / "cross_phase_reinstatement_subject.tsv")
    write_table(
        group_one_sample, output_dir / "cross_phase_reinstatement_group_one_sample.tsv"
    )
    write_table(
        group_run_contrast, output_dir / "cross_phase_reinstatement_group_run_contrast.tsv"
    )
    write_table(qc, output_dir / "cross_phase_reinstatement_qc.tsv")
    write_table(failures, output_dir / "cross_phase_reinstatement_failures.tsv")
    write_table(
        subject_metrics,
        tables_si / f"table_cross_phase_reinstatement_{roi_tag}.tsv",
    )

    save_json(
        {
            "roi_sets": args.roi_sets,
            "roi_tag": roi_tag,
            "target_phases": TARGET_PHASES,
            "learn_runs": LEARN_RUNS,
            "n_subjects_with_learning_patterns": len(subjects),
            "subjects": [path.name for path in subjects],
            "n_item_rows": int(len(item_metrics)),
            "n_subject_rows": int(len(subject_metrics)),
            "n_failures": int(len(failures)),
            "output_dir": str(output_dir),
            "tables_si_path": str(
                tables_si / f"table_cross_phase_reinstatement_{roi_tag}.tsv"
            ),
        },
        output_dir / "cross_phase_reinstatement_manifest.json",
    )


if __name__ == "__main__":
    main()
