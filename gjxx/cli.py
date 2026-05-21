from __future__ import annotations

import argparse

from gjxx.config import GlmSettings


def _add_shared_glm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--high-pass-seconds", type=float, default=128.0)
    parser.add_argument("--hrf-model", default="spm")
    parser.add_argument("--noise-model", default="ar1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GJXX story Python pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    relabel = subparsers.add_parser("relabel-events", help="Create the no-too-easy-or-hard event files.")
    relabel.add_argument("--input-root", required=True)
    relabel.add_argument("--output-root", required=True)

    item_glm = subparsers.add_parser("item-glm", help="Fit item-level GLM for one story.")
    item_glm.add_argument("--story", choices=["no_too_easy_or_hard", "gps", "biggerthan3", "tichu_jiaocha", "rd_all_examples"], required=True)
    item_glm.add_argument("--bold-root", required=True)
    item_glm.add_argument("--events-root", required=True)
    item_glm.add_argument("--output-root", required=True)
    item_glm.add_argument("--bold-pattern", default="sub-*.nii*")
    _add_shared_glm_args(item_glm)

    activation = subparsers.add_parser("activation-glm", help="Fit condition-level activation GLM.")
    activation.add_argument("--story", choices=["four_types", "remember_forget"], required=True)
    activation.add_argument("--bold-root", required=True)
    activation.add_argument("--events-root", required=True)
    activation.add_argument("--output-root", required=True)
    activation.add_argument("--bold-pattern", default="sub-*.nii*")
    _add_shared_glm_args(activation)

    stack = subparsers.add_parser("stack-patterns", help="Stack trial maps into 4D pattern images.")
    stack.add_argument("--story", choices=["no_too_easy_or_hard_gps", "gps", "rca", "rd_all_examples"], required=True)
    stack.add_argument("--trial-map-root", required=True)
    stack.add_argument("--output-root", required=True)
    stack.add_argument("--filter-query")

    gps = subparsers.add_parser("gps", help="Run group GPS analysis from stacked patterns.")
    gps.add_argument("--pattern-root", required=True)
    gps.add_argument("--mask", required=True)
    gps.add_argument("--output-dir", required=True)
    gps.add_argument("--pattern-name", default="glm_T_gps.nii.gz")
    gps.add_argument("--metadata-name", default="glm_T_gps_metadata.tsv")

    roi_corr = subparsers.add_parser("roi-dsm", help="Run the ROI-to-ROI DSM correlation analysis.")
    roi_corr.add_argument("--pattern-root", required=True)
    roi_corr.add_argument("--mask-a", required=True)
    roi_corr.add_argument("--mask-b", required=True)
    roi_corr.add_argument("--output-dir", required=True)
    roi_corr.add_argument("--hsc-name", default="HSC.nii.gz")
    roi_corr.add_argument("--lsc-name", default="LSC.nii.gz")

    rd = subparsers.add_parser("rd", help="Run the standard RD analysis.")
    rd.add_argument("--pattern-root", required=True)
    rd.add_argument("--mask", required=True)
    rd.add_argument("--output-dir", required=True)
    rd.add_argument("--threshold", type=float, default=80.0)

    rd_equalized = subparsers.add_parser("rd-equalized", help="Run the equalized-trial RD analysis.")
    rd_equalized.add_argument("--pattern-root", required=True)
    rd_equalized.add_argument("--mask", required=True)
    rd_equalized.add_argument("--output-dir", required=True)
    rd_equalized.add_argument("--threshold", type=float, default=80.0)
    rd_equalized.add_argument("--iterations", type=int, default=1000)
    rd_equalized.add_argument("--seed", type=int, default=0)

    searchlight = subparsers.add_parser("rd-searchlight", help="Run searchlight RD for one 4D image.")
    searchlight.add_argument("--input-img", required=True)
    searchlight.add_argument("--mask", required=True)
    searchlight.add_argument("--output-path", required=True)
    searchlight.add_argument("--threshold", type=float, default=80.0)
    searchlight.add_argument("--neighbors", type=int, default=100)

    scores = subparsers.add_parser("subject-scores", help="Summarize one score per subject from events.")
    scores.add_argument("--events-root", required=True)
    scores.add_argument("--output-path", required=True)
    scores.add_argument("--score-column", required=True)

    regression = subparsers.add_parser("group-regression", help="Run group voxelwise regression.")
    regression.add_argument("--map-root", required=True)
    regression.add_argument("--image-name", required=True)
    regression.add_argument("--score-table", required=True)
    regression.add_argument("--mask", required=True)
    regression.add_argument("--output-dir", required=True)
    regression.add_argument("--score-column", default="score")
    regression.add_argument("--alpha", type=float, default=0.05)
    regression.add_argument("--min-cluster-size", type=int, default=5)
    regression.add_argument("--no-rank-inputs", action="store_true")

    # Dimension analysis stages (formerly B-line)
    subparsers.add_parser("roi-robustness", help="Stage 08: ROI dimensionality + robustness suite.")
    subparsers.add_parser("remembered-forgotten", help="Stage 09: remembered vs forgotten dimension analysis.")
    subparsers.add_parser("memory-group", help="Stage 10: memory-good vs memory-poor dimension analysis.")
    subparsers.add_parser("behavior-pooling", help="Stage 11: pooled behavioral item analyses.")
    subparsers.add_parser("dimension-searchlight", help="Stage 12: whole-brain RD dimensionality searchlight.")
    subparsers.add_parser("seed-connectivity", help="Stage 13: hippocampal seed PCA-connectivity searchlight.")

    return parser


def _glm_from_args(args: argparse.Namespace) -> GlmSettings:
    return GlmSettings(
        tr=args.tr,
        hrf_model=args.hrf_model,
        high_pass_seconds=args.high_pass_seconds,
        noise_model=args.noise_model,
    )


def main() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    if args.command == "relabel-events":
        from gjxx.events import relabel_events_excluding_extremes

        relabel_events_excluding_extremes(args.input_root, args.output_root)
        return

    if args.command == "item-glm":
        from gjxx.first_level import fit_item_level_story

        fit_item_level_story(
            bold_root=args.bold_root,
            events_root=args.events_root,
            output_root=args.output_root,
            story=args.story,
            glm_settings=_glm_from_args(args),
            bold_pattern=args.bold_pattern,
        )
        return

    if args.command == "activation-glm":
        from gjxx.first_level import fit_activation_story

        fit_activation_story(
            bold_root=args.bold_root,
            events_root=args.events_root,
            output_root=args.output_root,
            story=args.story,
            glm_settings=_glm_from_args(args),
            bold_pattern=args.bold_pattern,
        )
        return

    if args.command == "stack-patterns":
        from gjxx.patterns import build_story_pattern_outputs

        build_story_pattern_outputs(args.trial_map_root, args.output_root, story=args.story, filter_query=args.filter_query)
        return

    if args.command == "gps":
        from gjxx.roi import run_group_gps

        run_group_gps(
            pattern_root=args.pattern_root,
            mask_path=args.mask,
            output_dir=args.output_dir,
            pattern_name=args.pattern_name,
            metadata_name=args.metadata_name,
        )
        return

    if args.command == "roi-dsm":
        from gjxx.roi import run_group_roi_dsm_correlation

        run_group_roi_dsm_correlation(
            pattern_root=args.pattern_root,
            mask_a=args.mask_a,
            mask_b=args.mask_b,
            output_dir=args.output_dir,
            hsc_name=args.hsc_name,
            lsc_name=args.lsc_name,
        )
        return

    if args.command == "rd":
        from gjxx.rd import run_group_rd

        run_group_rd(
            pattern_root=args.pattern_root,
            mask_path=args.mask,
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
        return

    if args.command == "rd-equalized":
        from gjxx.rd import run_group_equalized_rd

        run_group_equalized_rd(
            pattern_root=args.pattern_root,
            mask_path=args.mask,
            output_dir=args.output_dir,
            threshold=args.threshold,
            n_iter=args.iterations,
            seed=args.seed,
        )
        return

    if args.command == "rd-searchlight":
        from gjxx.rd import run_rd_searchlight

        run_rd_searchlight(
            input_img=args.input_img,
            mask_path=args.mask,
            output_path=args.output_path,
            threshold=args.threshold,
            neighbors=args.neighbors,
        )
        return

    if args.command == "subject-scores":
        from gjxx.group import get_subject_mean_scores

        get_subject_mean_scores(
            events_root=args.events_root,
            output_path=args.output_path,
            score_column=args.score_column,
        )
        return

    if args.command == "group-regression":
        from gjxx.group import run_group_regression

        run_group_regression(
            map_root=args.map_root,
            image_name=args.image_name,
            score_table=args.score_table,
            mask_path=args.mask,
            output_dir=args.output_dir,
            score_column=args.score_column,
            alpha=args.alpha,
            min_cluster_size=args.min_cluster_size,
            rank_inputs=not args.no_rank_inputs,
        )
        return

    _STAGE_DISPATCH = {
        "roi-robustness": "gjxx.run_stage08_roi_robustness",
        "remembered-forgotten": "gjxx.run_stage09_remembered_forgotten",
        "memory-group": "gjxx.run_stage10_memory_group",
        "behavior-pooling": "gjxx.run_stage11_behavior_pooling",
        "dimension-searchlight": "gjxx.run_stage12_dimension_searchlight",
        "seed-connectivity": "gjxx.run_stage13_seed_connectivity",
    }

    if args.command in _STAGE_DISPATCH:
        import importlib
        import sys
        # The target module has its own argparse; pass remaining args through.
        sys.argv = [sys.argv[0]] + remaining
        module = importlib.import_module(_STAGE_DISPATCH[args.command])
        module.main()
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
