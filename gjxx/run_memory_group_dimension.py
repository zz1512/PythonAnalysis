from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dimension_analysis_core import (
    DEFAULT_EVENTS_DIR,
    DEFAULT_MASK_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATTERN_DIR,
    SUBJECTS,
    analyze_memory_groups,
    compute_memory_accuracies,
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
        description="Task 2: compare HSC vs LSC within memory-good and memory-poor subjects."
    )
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "memory_group_dimension",
    )
    parser.add_argument("--explained-threshold", type=float, default=80.0)
    parser.add_argument("--min-trials", type=int, default=8)
    parser.add_argument("--good-memory-threshold", type=float, default=0.2)
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

    accuracy_records = compute_memory_accuracies(
        subjects=SUBJECTS,
        events_dir=args.events_dir,
        good_memory_threshold=args.good_memory_threshold,
    )
    accuracy_by_subject = {record.subject_id: record for record in accuracy_records}

    logging.info("Per-subject memory accuracy:")
    for record in accuracy_records:
        logging.info(
            "%s | remembered=%d | total=%d | accuracy=%.4f | group=%s",
            record.subject_id,
            record.remembered_trials,
            record.total_trials,
            record.accuracy,
            record.memory_group,
        )

    good_subjects = [record.subject_id for record in accuracy_records if record.memory_group == "good"]
    poor_subjects = [record.subject_id for record in accuracy_records if record.memory_group == "poor"]
    logging.info("Memory-good subjects (> %.2f): %s", args.good_memory_threshold, good_subjects)
    logging.info("Memory-poor subjects (<= %.2f): %s", args.good_memory_threshold, poor_subjects)

    mask = load_mask(args.mask_path)
    grouped_records = analyze_memory_groups(
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        mask=mask,
        explained_threshold=args.explained_threshold,
        min_trials=args.min_trials,
        accuracy_by_subject=accuracy_by_subject,
    )

    good_stats = paired_t_test(grouped_records["good"].records)
    poor_stats = paired_t_test(grouped_records["poor"].records)
    summary = {
        "task": "memory_group_dimension",
        "config": {
            "pattern_dir": str(args.pattern_dir),
            "events_dir": str(args.events_dir),
            "mask_path": str(args.mask_path),
            "explained_threshold": args.explained_threshold,
            "min_trials": args.min_trials,
            "good_memory_threshold": args.good_memory_threshold,
        },
        "memory_good_subject_ids": good_subjects,
        "memory_poor_subject_ids": poor_subjects,
        "memory_good_included_subject_ids": grouped_records["good"].included_subject_ids,
        "memory_good_excluded_due_to_trial_count": [
            record.__dict__ for record in grouped_records["good"].excluded_due_to_trial_count
        ],
        "memory_poor_included_subject_ids": grouped_records["poor"].included_subject_ids,
        "memory_poor_excluded_due_to_trial_count": [
            record.__dict__ for record in grouped_records["poor"].excluded_due_to_trial_count
        ],
        "memory_good_hsc_vs_lsc": good_stats,
        "memory_poor_hsc_vs_lsc": poor_stats,
    }

    write_csv(
        args.output_dir / "subject_memory_accuracy.csv",
        [record.__dict__ for record in accuracy_records],
    )
    rows = []
    rows.extend(subject_dimension_rows(grouped_records["good"].records, "memory_group", "good"))
    rows.extend(subject_dimension_rows(grouped_records["poor"].records, "memory_group", "poor"))
    write_csv(args.output_dir / "memory_group_dimensions.csv", rows)
    selection_rows = []
    selection_rows.extend(
        {
            "memory_group": "good",
            "status": "included",
            "subject_id": subject_id,
            "hsc_trials": None,
            "lsc_trials": None,
            "reason": "included",
        }
        for subject_id in grouped_records["good"].included_subject_ids
    )
    selection_rows.extend(
        exclusion_rows(
            grouped_records["good"].excluded_due_to_trial_count,
            "memory_group",
            "good",
            "excluded",
        )
    )
    selection_rows.extend(
        {
            "memory_group": "poor",
            "status": "included",
            "subject_id": subject_id,
            "hsc_trials": None,
            "lsc_trials": None,
            "reason": "included",
        }
        for subject_id in grouped_records["poor"].included_subject_ids
    )
    selection_rows.extend(
        exclusion_rows(
            grouped_records["poor"].excluded_due_to_trial_count,
            "memory_group",
            "poor",
            "excluded",
        )
    )
    write_csv(args.output_dir / "selection_details.csv", selection_rows)
    write_json(args.output_dir / "summary.json", summary)

    logging.info("Memory-good HSC vs LSC: %s", good_stats)
    logging.info("Memory-poor HSC vs LSC: %s", poor_stats)
    logging.info("Task 2 outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
