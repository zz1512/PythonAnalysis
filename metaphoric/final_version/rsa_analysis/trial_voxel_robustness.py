"""
trial_voxel_robustness.py

Purpose
- Summarize trial-count and voxel-count robustness for ROI-level RSA.
- Use existing pattern metadata, rsa_itemwise_details.csv, and rsa_summary_stats.csv.
- Answer three practical questions:
  1. Are pre/post trial counts balanced by subject and condition?
  2. Are downstream itemwise pair counts balanced by subject/ROI/stage/condition?
  3. Does n_voxels show any systematic pre/post drift across ROI?

Default inputs
- pattern_root: `${PYTHON_METAPHOR_ROOT}/pattern_root`
- rsa_dir: current ROI-set output under `${PYTHON_METAPHOR_ROOT}/rsa_results_optimized_<roi_set>`

Outputs
- `<rsa_dir>/robustness/trial_metadata_subject_stage_condition.tsv`
- `<rsa_dir>/robustness/trial_metadata_group_summary.tsv`
- `<rsa_dir>/robustness/trial_metadata_stage_tests.tsv`
- `<rsa_dir>/robustness/itemwise_counts_subject_roi_stage_condition.tsv`
- `<rsa_dir>/robustness/itemwise_counts_group_summary.tsv`
- `<rsa_dir>/robustness/itemwise_counts_stage_tests.tsv`
- `<rsa_dir>/robustness/voxel_counts_subject_roi_stage.tsv`
- `<rsa_dir>/robustness/voxel_counts_group_summary.tsv`
- `<rsa_dir>/robustness/voxel_counts_stage_tests.tsv`
- `<rsa_dir>/robustness/analysis_manifest.json`
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

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
RSA_ROOT = Path(__file__).resolve().parent
if str(RSA_ROOT) not in sys.path:
    sys.path.append(str(RSA_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.roi_library import sanitize_roi_tag  # noqa: E402
import rsa_config as cfg  # noqa: E402


BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", str(cfg.BASE_DIR)))


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _default_rsa_dir() -> Path:
    roi_tag = sanitize_roi_tag(getattr(cfg, "ROI_SET", ""))
    return BASE_DIR / f"rsa_results_optimized_{roi_tag}"


def _resolve_rsa_dir(path_arg: Path | None) -> Path:
    if path_arg is not None:
        return path_arg
    return _default_rsa_dir()


def _resolve_output_dir(path_arg: Path | None, *, rsa_dir: Path) -> Path:
    if path_arg is not None:
        return path_arg
    return rsa_dir / "robustness"


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _paired_stage_tests(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    label_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = {column: keys[idx] for idx, column in enumerate(group_cols)}
        pivot = group.pivot_table(index="subject", columns="stage", values=value_col, aggfunc="mean")
        if not {"Pre", "Post"}.issubset(pivot.columns):
            continue
        pair = pivot.dropna(subset=["Pre", "Post"])
        if pair.empty:
            continue
        pre = pair["Pre"]
        post = pair["Post"]
        diff = post - pre
        all_zero = bool((diff == 0).all())
        if all_zero:
            t_stat = 0.0
            t_p = 1.0
            w_stat = 0.0
            w_p = 1.0
        else:
            t_stat, t_p = stats.ttest_rel(post, pre, nan_policy="omit")
            try:
                w_stat, w_p = stats.wilcoxon(post, pre, zero_method="wilcox")
            except Exception:
                w_stat, w_p = float("nan"), float("nan")
        row = {
            **labels,
            "n_subjects": int(len(pair)),
            "pre_mean": _safe_float(pre.mean()),
            "post_mean": _safe_float(post.mean()),
            "post_minus_pre_mean": _safe_float(diff.mean()),
            "exact_match_all_subjects": all_zero,
            "ttest_t": _safe_float(t_stat),
            "ttest_p": _safe_float(t_p),
            "wilcoxon_w": _safe_float(w_stat),
            "wilcoxon_p": _safe_float(w_p),
        }
        rows.append(row)
    columns = [
        *label_cols,
        "n_subjects",
        "pre_mean",
        "post_mean",
        "post_minus_pre_mean",
        "exact_match_all_subjects",
        "ttest_t",
        "ttest_p",
        "wilcoxon_w",
        "wilcoxon_p",
    ]
    return pd.DataFrame(rows, columns=columns)


def collect_trial_metadata(pattern_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(pattern_root.glob("sub-*/*_metadata.tsv")):
        frame = _read_table(metadata_path)
        if frame.empty:
            continue
        subject = str(frame["subject"].iloc[0]) if "subject" in frame.columns else metadata_path.parent.name
        stage = str(frame["stage"].iloc[0]) if "stage" in frame.columns else metadata_path.stem.split("_", 1)[0].title()
        condition = str(frame["condition"].iloc[0]) if "condition" in frame.columns else metadata_path.stem.split("_", 1)[1]
        runs = sorted({str(value) for value in frame["run"].dropna().unique()}) if "run" in frame.columns else []
        rows.append(
            {
                "subject": subject,
                "stage": stage,
                "condition": condition,
                "n_trials": int(len(frame)),
                "n_unique_items": int(frame["unique_label"].nunique()) if "unique_label" in frame.columns else int(len(frame)),
                "n_runs": int(len(runs)),
                "runs": ",".join(runs),
                "source_file": str(metadata_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "stage", "condition"]).reset_index(drop=True)


def collect_itemwise_counts(itemwise_path: Path) -> pd.DataFrame:
    frame = _read_table(itemwise_path)
    counts = (
        frame.groupby(["subject", "roi", "stage", "condition"], as_index=False)
        .agg(
            n_pairs=("similarity", "size"),
            n_items=("item", "nunique"),
            n_pair_ids=("pair_id", "nunique"),
        )
        .sort_values(["subject", "roi", "stage", "condition"])
        .reset_index(drop=True)
    )
    return counts


def collect_voxel_counts(summary_path: Path) -> pd.DataFrame:
    frame = _read_table(summary_path)
    keep = ["subject", "roi", "stage", "n_voxels"]
    missing = [column for column in keep if column not in frame.columns]
    if missing:
        raise ValueError(f"Summary file missing required columns: {missing}")
    return frame[keep].copy().sort_values(["subject", "roi", "stage"]).reset_index(drop=True)


def group_summary(frame: pd.DataFrame, *, group_cols: list[str], value_col: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_cols, dropna=False)[value_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_rows", "mean": f"{value_col}_mean", "std": f"{value_col}_sd", "min": f"{value_col}_min", "max": f"{value_col}_max"})
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Trial-count and voxel-count robustness for ROI RSA outputs.")
    parser.add_argument("--rsa-dir", type=Path, default=None, help="RSA result directory. Default: current ROI-set output.")
    parser.add_argument("--pattern-root", type=Path, default=BASE_DIR / "pattern_root", help="pattern_root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <rsa_dir>/robustness")
    args = parser.parse_args()

    rsa_dir = _resolve_rsa_dir(args.rsa_dir)
    output_dir = ensure_dir(_resolve_output_dir(args.output_dir, rsa_dir=rsa_dir))

    itemwise_path = rsa_dir / "rsa_itemwise_details.csv"
    summary_path = rsa_dir / "rsa_summary_stats.csv"
    if not itemwise_path.exists():
        raise FileNotFoundError(f"Missing RSA itemwise file: {itemwise_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing RSA summary file: {summary_path}")
    if not args.pattern_root.exists():
        raise FileNotFoundError(f"Missing pattern_root: {args.pattern_root}")

    trial_metadata = collect_trial_metadata(args.pattern_root)
    write_table(trial_metadata, output_dir / "trial_metadata_subject_stage_condition.tsv")
    write_table(
        group_summary(trial_metadata, group_cols=["condition", "stage"], value_col="n_trials"),
        output_dir / "trial_metadata_group_summary.tsv",
    )
    trial_tests = _paired_stage_tests(
        trial_metadata,
        group_cols=["condition"],
        value_col="n_trials",
        label_cols=["condition"],
    )
    write_table(trial_tests, output_dir / "trial_metadata_stage_tests.tsv")

    itemwise_counts = collect_itemwise_counts(itemwise_path)
    write_table(itemwise_counts, output_dir / "itemwise_counts_subject_roi_stage_condition.tsv")
    write_table(
        group_summary(itemwise_counts, group_cols=["roi", "condition", "stage"], value_col="n_pairs"),
        output_dir / "itemwise_counts_group_summary.tsv",
    )
    itemwise_tests = _paired_stage_tests(
        itemwise_counts,
        group_cols=["roi", "condition"],
        value_col="n_pairs",
        label_cols=["roi", "condition"],
    )
    write_table(itemwise_tests, output_dir / "itemwise_counts_stage_tests.tsv")

    voxel_counts = collect_voxel_counts(summary_path)
    write_table(voxel_counts, output_dir / "voxel_counts_subject_roi_stage.tsv")
    write_table(
        group_summary(voxel_counts, group_cols=["roi", "stage"], value_col="n_voxels"),
        output_dir / "voxel_counts_group_summary.tsv",
    )
    voxel_tests = _paired_stage_tests(
        voxel_counts,
        group_cols=["roi"],
        value_col="n_voxels",
        label_cols=["roi"],
    )
    write_table(voxel_tests, output_dir / "voxel_counts_stage_tests.tsv")

    save_json(
        {
            "rsa_dir": str(rsa_dir),
            "pattern_root": str(args.pattern_root),
            "n_trial_metadata_rows": int(len(trial_metadata)),
            "n_itemwise_count_rows": int(len(itemwise_counts)),
            "n_voxel_count_rows": int(len(voxel_counts)),
            "trial_conditions": sorted({str(item) for item in trial_metadata["condition"].unique()}),
            "itemwise_conditions": sorted({str(item) for item in itemwise_counts["condition"].unique()}),
            "roi_count": int(voxel_counts["roi"].nunique()),
        },
        output_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
