from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gjxx.dimension_analysis_core import DEFAULT_OUTPUT_ROOT, DEFAULT_PATTERN_DIR, SUBJECTS, setup_logging
from gjxx.dimension_story_utils import DEFAULT_SEARCHLIGHT_MASK_ROOT, run_dimensionality_searchlight_suite
from gjxx.runtime_config import load_config_json, pick_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 20: whole-brain RD dimensionality searchlight (MATLAB Dimension story)."
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Optional config JSON. Priority: CLI > config > legacy preset.",
    )
    parser.add_argument("--pattern-dir", type=Path, default=None)
    parser.add_argument("--subject-mask-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    config = load_config_json(args.config_json)
    args.pattern_dir = pick_path(
        cli_value=args.pattern_dir,
        config_payload=config,
        config_key="pattern_dir",
        preset_value=DEFAULT_PATTERN_DIR,
    )
    args.subject_mask_root = pick_path(
        cli_value=args.subject_mask_root,
        config_payload=config,
        config_key="subject_mask_root",
        preset_value=DEFAULT_SEARCHLIGHT_MASK_ROOT,
    )
    output_root = pick_path(
        cli_value=None,
        config_payload=config,
        config_key="output_root",
        preset_value=DEFAULT_OUTPUT_ROOT,
    )
    # Keep legacy output folder name.
    args.output_dir = pick_path(
        cli_value=args.output_dir,
        config_payload=config,
        config_key="output_dir",
        preset_value=output_root / "story_searchlight_suite",
    )
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
    logging.info("Stage 20 finished: %s", summary["group_summary"])


if __name__ == "__main__":
    main()

