from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dimension_analysis_core import DEFAULT_OUTPUT_ROOT, DEFAULT_PATTERN_DIR, SUBJECTS, setup_logging, write_json
from dimension_story_utils import (
    DEFAULT_HIPPOCAMPUS_LEFT_MASK,
    DEFAULT_HIPPOCAMPUS_RIGHT_MASK,
    build_cross_subject_voxel_keep_vector,
    iter_thresholds,
    run_roi_threshold_analysis,
    save_roi_keep_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the main ROI MATLAB story into a consolidated Python ROI suite."
    )
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--left-mask-path", type=Path, default=DEFAULT_HIPPOCAMPUS_LEFT_MASK)
    parser.add_argument("--right-mask-path", type=Path, default=DEFAULT_HIPPOCAMPUS_RIGHT_MASK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "story_roi_suite",
    )
    parser.add_argument("--voxel-alpha", type=float, default=0.05)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Running ROI story suite under %s", args.output_dir)

    summaries: dict[str, object] = {}

    summaries["hl_patternbutnotrdm"] = run_roi_threshold_analysis(
        analysis_name="hl_patternbutnotrdm",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        thresholds=iter_thresholds(70, 80),
        output_dir=args.output_dir / "hl_patternbutnotrdm",
        metric_mode="pattern",
        trial_strategy="all",
    )

    summaries["hl_rdm_baseline"] = run_roi_threshold_analysis(
        analysis_name="hl_rdm_baseline",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.right_mask_path,
        thresholds=[80],
        output_dir=args.output_dir / "hl_rdm_baseline",
        metric_mode="rdm",
        trial_strategy="all",
    )

    summaries["sametrial_rdm_fixed20"] = run_roi_threshold_analysis(
        analysis_name="sametrial_rdm_fixed20",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        thresholds=iter_thresholds(70, 80),
        output_dir=args.output_dir / "sametrial_rdm_fixed20",
        metric_mode="rdm",
        trial_strategy="fixed",
        fixed_trials=20,
        n_resamples=1000,
    )

    summaries["demean_pattern_min"] = run_roi_threshold_analysis(
        analysis_name="demean_pattern_min",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        thresholds=iter_thresholds(70, 90),
        output_dir=args.output_dir / "demean_pattern_min",
        metric_mode="pattern",
        trial_strategy="min",
        demean=True,
        n_resamples=1000,
    )

    summaries["demean_pattern_fixed25"] = run_roi_threshold_analysis(
        analysis_name="demean_pattern_fixed25",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        thresholds=iter_thresholds(70, 90),
        output_dir=args.output_dir / "demean_pattern_fixed25",
        metric_mode="pattern",
        trial_strategy="fixed",
        fixed_trials=25,
        demean=True,
        n_resamples=1000,
    )

    voxel_report = build_cross_subject_voxel_keep_vector(
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        alpha=args.voxel_alpha,
    )
    save_roi_keep_vector(
        args.left_mask_path,
        voxel_report["keep_vector"],
        args.output_dir / "voxel_mask_keep.nii",
    )
    voxel_summary = {
        "task": "cross_subject_voxel_mask",
        "config": {
            "pattern_dir": str(args.pattern_dir),
            "roi_mask_path": str(args.left_mask_path),
            "alpha": args.voxel_alpha,
        },
        "n_voxels_total": voxel_report["n_voxels_total"],
        "n_voxels_kept": voxel_report["n_voxels_kept"],
        "mask_path": str(args.output_dir / "voxel_mask_keep.nii"),
    }
    write_json(args.output_dir / "voxel_mask_summary.json", voxel_summary)
    summaries["cross_subject_voxel_mask"] = voxel_summary

    summaries["voxelmask_pattern_fixed20"] = run_roi_threshold_analysis(
        analysis_name="voxelmask_pattern_fixed20",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        thresholds=iter_thresholds(70, 80),
        output_dir=args.output_dir / "voxelmask_pattern_fixed20",
        metric_mode="pattern",
        trial_strategy="fixed",
        fixed_trials=20,
        voxel_keep_vector=voxel_report["keep_vector"],
        n_resamples=1000,
    )

    summaries["pc_window_3_5_fixed20"] = run_roi_threshold_analysis(
        analysis_name="pc_window_3_5_fixed20",
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        roi_mask_path=args.left_mask_path,
        thresholds=[80],
        output_dir=args.output_dir / "pc_window_3_5_fixed20",
        metric_mode="pc_window",
        trial_strategy="fixed",
        fixed_trials=20,
        n_resamples=1000,
        pc_window=(3, 5),
    )

    master_summary = {
        "task": "story_roi_suite",
        "output_dir": str(args.output_dir),
        "analyses": summaries,
    }
    write_json(args.output_dir / "summary.json", master_summary)
    logging.info("ROI story suite finished. Summary written to %s", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
