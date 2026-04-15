from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dimension_analysis_core import DEFAULT_OUTPUT_ROOT, SUBJECTS, write_json, setup_logging
from dimension_story_utils import (
    DEFAULT_ALT_EVENTS_DIR,
    DEFAULT_HIPPOCAMPUS_LEFT_MASK,
    DEFAULT_RSA_ALL_DIR,
    run_behavioral_pooling_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the behavioral item-sampling MATLAB scripts into Python."
    )
    parser.add_argument("--rsa-all-dir", type=Path, default=DEFAULT_RSA_ALL_DIR)
    parser.add_argument("--roi-mask-path", type=Path, default=DEFAULT_HIPPOCAMPUS_LEFT_MASK)
    parser.add_argument("--events-dir", type=Path, default=Path(r"I:\FLXX1\events"))
    parser.add_argument("--alt-events-dir", type=Path, default=DEFAULT_ALT_EVENTS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "story_behavior_suite",
    )
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

    summaries: dict[str, object] = {}

    summaries["across_sub_dim"] = run_behavioral_pooling_analysis(
        analysis_name="across_sub_dim",
        subjects=SUBJECTS,
        rsa_all_dir=args.rsa_all_dir,
        events_dir=args.events_dir,
        roi_mask_path=args.roi_mask_path,
        score_column="jiaocha_score",
        top_fraction=0.3,
        bottom_fraction=0.3,
        top_sample_fraction=0.2,
        bottom_sample_fraction=0.2,
        n_resamples=30,
        explained_threshold=80,
        output_dir=args.output_dir / "across_sub_dim",
    )

    summaries["untitled2_top10_bottom50"] = run_behavioral_pooling_analysis(
        analysis_name="untitled2_top10_bottom50",
        subjects=SUBJECTS,
        rsa_all_dir=args.rsa_all_dir,
        events_dir=args.alt_events_dir,
        roi_mask_path=args.roi_mask_path,
        score_column="jiaocha_score_2",
        top_fraction=0.1,
        bottom_fraction=0.5,
        top_sample_fraction=None,
        bottom_sample_fraction=0.2,
        n_resamples=100,
        explained_threshold=80,
        output_dir=args.output_dir / "untitled2_top10_bottom50",
    )

    summaries["untitled3_top10_bottom50"] = run_behavioral_pooling_analysis(
        analysis_name="untitled3_top10_bottom50",
        subjects=SUBJECTS,
        rsa_all_dir=args.rsa_all_dir,
        events_dir=args.alt_events_dir,
        roi_mask_path=args.roi_mask_path,
        score_column="jiaocha_score_2",
        top_fraction=0.1,
        bottom_fraction=0.5,
        top_sample_fraction=0.5,
        bottom_sample_fraction=0.1,
        n_resamples=1000,
        explained_threshold=80,
        output_dir=args.output_dir / "untitled3_top10_bottom50",
    )

    summary = {
        "task": "story_behavior_suite",
        "output_dir": str(args.output_dir),
        "analyses": summaries,
    }
    write_json(args.output_dir / "summary.json", summary)
    logging.info("Behavior suite finished. Summary written to %s", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
