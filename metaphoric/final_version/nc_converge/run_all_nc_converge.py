#!/usr/bin/env python3
"""Run all NC convergence modules in spec order.

By default this only runs analysis scripts. Use --skip-append-section28 to avoid
appending to result_new_meta_roi.md while testing on a development machine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("a2_schema_lock.py", "a2_schema_lock", ["id_audit.tsv", "subject_intersection.tsv", "schema_lock.log"]),
    ("a3_shared_post_review.py", "a3_shared_post_review", ["shared_post_review.tsv", "shared_post_review.md", "shared_post_input_manifest.tsv"]),
    ("a4_post_memory_component.py", "a4_post_memory_component", ["memory_component_model.tsv", "memory_component_roi_followup.tsv", "stage_trajectory_model.tsv", "stage_trajectory_descriptives.tsv", "memory_component_input_manifest.tsv"]),
    ("a5_network_composite.py", "a5_network_composite", ["network_composite_summary.tsv", "network_trajectory_long.tsv", "network_composite_input_manifest.tsv"]),
    ("b1_staged_path.py", "b1_staged_path", ["stage5_network_trajectory_table.tsv", "staged_trajectory.md"]),
    ("b2a_mahalanobis_from_patterns.py", "b2a_mahalanobis_from_patterns", ["mahalanobis_pair_distance.tsv", "mahalanobis_network_stage.tsv", "mahalanobis_subject_delta.tsv", "mahalanobis_step5c.tsv", "mahalanobis_input_manifest.tsv", "mahalanobis_method_note.md"]),
    ("b2_crossnobis_robustness.py", "b2_crossnobis_robustness", ["crossnobis_step5c.tsv", "crossnobis_input_manifest.tsv"]),
    ("b3_repr_coupling.py", "b3_repr_coupling", ["repr_coupling.tsv", "repr_coupling_input_manifest.tsv"]),
    ("d_reviewer_supp_index.py", "d_reviewer_supp_index", ["reviewer_supp_index.tsv", "reviewer_supp_index_log.tsv"]),
    ("a6_fdr_tiering.py", "a6_fdr_tiering", ["fdr_tier_manifest.tsv", "fdr_tier_summary.md"]),
    ("go_no_go.py", "go_no_go", ["go_no_go_stage6.md", "go_no_go_evidence.tsv"]),
    ("append_section28.py", "append_section28", ["section28_preview.md", "append_section28.log"]),
    ("isolation_check.py", "isolation_check", ["output_manifest.tsv", "isolation_report.md", "upstream_git_status.tsv"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--final-root", type=Path, default=None)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--skip-append-section28", action="store_true")
    parser.add_argument("--dry-run-section28", action="store_true", help="Pass --dry-run to append_section28 instead of modifying result_new_meta_roi.md.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip modules whose expected outputs already exist.")
    return parser.parse_args()


def common_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for flag, value in [
        ("--base-dir", args.base_dir),
        ("--paper-output-root", args.paper_output_root),
        ("--output-root", args.output_root),
        ("--final-root", args.final_root),
    ]:
        if value is not None:
            out.extend([flag, str(value)])
    if args.allow_empty:
        out.append("--allow-empty")
    return out


def output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    paper_root = args.paper_output_root or ((args.base_dir or Path("E:/python_metaphor")) / "paper_outputs")
    return Path(paper_root) / "qc" / "nc_converge"


def module_completed(args: argparse.Namespace, module: str, expected: list[str]) -> bool:
    module_path = output_root(args) / module
    return all((module_path / name).exists() for name in expected)


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    failures = []
    for script, module, expected in SCRIPTS:
        if script == "append_section28.py" and args.skip_append_section28:
            continue
        if not args.no_resume and module_completed(args, module, expected):
            print(f"[{script}] skipped=existing_outputs")
            continue
        cmd = [sys.executable, str(here / script), *common_args(args)]
        if script == "append_section28.py" and args.dry_run_section28:
            cmd.append("--dry-run")
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        print(f"[{script}] exit={result.returncode}")
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
        if result.returncode != 0:
            failures.append(script)
            break
    if failures:
        raise SystemExit(f"Failed module(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()
