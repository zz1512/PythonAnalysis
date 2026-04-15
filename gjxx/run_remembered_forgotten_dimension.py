from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dimension_analysis_core import (
    DEFAULT_MASK_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATTERNS_HLRF_DIR,
    SUBJECTS,
    analyze_remembered_forgotten,
    exclusion_rows,
    load_mask,
    paired_t_test,
    setup_logging,
    subject_dimension_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 1: extend remembered-only analysis to forgotten trials."
    )
    parser.add_argument("--patterns-hlrf-dir", type=Path, default=DEFAULT_PATTERNS_HLRF_DIR)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "remembered_forgotten_dimension",
    )
    parser.add_argument("--explained-threshold", type=float, default=80.0)
    parser.add_argument("--min-trials", type=int, default=8)
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

    mask = load_mask(args.mask_path)
    results = analyze_remembered_forgotten(
        subjects=SUBJECTS,
        patterns_hlrf_dir=args.patterns_hlrf_dir,
        mask=mask,
        explained_threshold=args.explained_threshold,
        min_trials=args.min_trials,
    )

    remembered_stats = paired_t_test(results["remembered"].records)
    forgotten_stats = paired_t_test(results["forgotten"].records)
    summary = {
        "task": "remembered_vs_forgotten_dimension",
        "config": {
            "patterns_hlrf_dir": str(args.patterns_hlrf_dir),
            "mask_path": str(args.mask_path),
            "explained_threshold": args.explained_threshold,
            "min_trials": args.min_trials,
        },
        "remembered_included_subject_ids": results["remembered"].included_subject_ids,
        "remembered_excluded_due_to_trial_count": [
            record.__dict__ for record in results["remembered"].excluded_due_to_trial_count
        ],
        "forgotten_included_subject_ids": results["forgotten"].included_subject_ids,
        "forgotten_excluded_due_to_trial_count": [
            record.__dict__ for record in results["forgotten"].excluded_due_to_trial_count
        ],
        "remembered_hsc_vs_lsc": remembered_stats,
        "forgotten_hsc_vs_lsc": forgotten_stats,
    }

    rows = []
    rows.extend(subject_dimension_rows(results["remembered"].records, "analysis", "remembered"))
    rows.extend(subject_dimension_rows(results["forgotten"].records, "analysis", "forgotten"))
    write_csv(args.output_dir / "remembered_forgotten_dimensions.csv", rows)
    selection_rows = []
    selection_rows.extend(
        {
            "analysis": "remembered",
            "status": "included",
            "subject_id": subject_id,
            "hsc_trials": None,
            "lsc_trials": None,
            "reason": "included",
        }
        for subject_id in results["remembered"].included_subject_ids
    )
    selection_rows.extend(
        exclusion_rows(
            results["remembered"].excluded_due_to_trial_count,
            "analysis",
            "remembered",
            "excluded",
        )
    )
    selection_rows.extend(
        {
            "analysis": "forgotten",
            "status": "included",
            "subject_id": subject_id,
            "hsc_trials": None,
            "lsc_trials": None,
            "reason": "included",
        }
        for subject_id in results["forgotten"].included_subject_ids
    )
    selection_rows.extend(
        exclusion_rows(
            results["forgotten"].excluded_due_to_trial_count,
            "analysis",
            "forgotten",
            "excluded",
        )
    )
    write_csv(args.output_dir / "selection_details.csv", selection_rows)
    write_json(args.output_dir / "summary.json", summary)

    logging.info("Remembered HSC vs LSC: %s", remembered_stats)
    logging.info("Forgotten HSC vs LSC: %s", forgotten_stats)
    logging.info("Task 1 outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
