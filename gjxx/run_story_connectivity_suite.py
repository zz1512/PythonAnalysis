from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dimension_analysis_core import DEFAULT_OUTPUT_ROOT, DEFAULT_PATTERN_DIR, SUBJECTS, setup_logging
from dimension_story_utils import (
    DEFAULT_HIPPOCAMPUS_LEFT_MASK,
    DEFAULT_SEARCHLIGHT_MASK_ROOT,
    run_seed_connectivity_searchlight_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert stage1_pca_conn_searchlight.m + my_measure.m into Python."
    )
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--subject-mask-root", type=Path, default=DEFAULT_SEARCHLIGHT_MASK_ROOT)
    parser.add_argument("--seed-mask-path", type=Path, default=DEFAULT_HIPPOCAMPUS_LEFT_MASK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "story_connectivity_suite",
    )
    parser.add_argument("--voxel-count", type=int, default=100)
    parser.add_argument("--explained-threshold", type=float, default=80.0)
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

    summary = run_seed_connectivity_searchlight_suite(
        subjects=SUBJECTS,
        pattern_dir=args.pattern_dir,
        subject_mask_root=args.subject_mask_root,
        seed_mask_path=args.seed_mask_path,
        output_dir=args.output_dir,
        voxel_count=args.voxel_count,
        explained_threshold=args.explained_threshold,
    )
    logging.info("Connectivity suite finished: %s", summary["group_summary"])


if __name__ == "__main__":
    main()
