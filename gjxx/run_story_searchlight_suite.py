from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dimension_analysis_core import DEFAULT_OUTPUT_ROOT, DEFAULT_PATTERN_DIR, SUBJECTS, setup_logging
from dimension_story_utils import DEFAULT_SEARCHLIGHT_MASK_ROOT, run_dimensionality_searchlight_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert searchlight_RD.m + my_var_measure.m + pairedttest.m into Python."
    )
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--subject-mask-root", type=Path, default=DEFAULT_SEARCHLIGHT_MASK_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "story_searchlight_suite",
    )
    parser.add_argument("--voxel-count", type=int, default=100)
    parser.add_argument("--explained-threshold", type=float, default=70.0)
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

    summary = run_dimensionality_searchlight_suite(
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        subject_mask_root=args.subject_mask_root,
        output_dir=args.output_dir,
        voxel_count=args.voxel_count,
        explained_threshold=args.explained_threshold,
    )
    logging.info("Searchlight suite finished: %s", summary["group_summary"])


if __name__ == "__main__":
    main()
