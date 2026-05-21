#!/usr/bin/env python3
"""按 spec 顺序运行海马亚区切分 + 三轴 + MDS 全流程。

顺序：S1 → S2 → S3 → S4 → S5 → V1 → N1 → ISO → append §29。
默认 --resume / --skip-existing：模块预期产物全部存在则跳过。
默认不追加 §29（避免在开发机上修改主结果文档）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("p0_prepare_inputs.py", "p0_inputs",
     ["hpc_items.tsv", "hpc_pair_table.tsv", "p0_manifest.tsv"]),
    ("s1_subfield_segmentation.py", "s1_segmentation",
     ["subfield_manifest.tsv", "s1_log.txt"]),
    ("s2_subfield_extract_beta.py", "s2_beta_extract",
     ["beta_long.tsv", "beta_qc.tsv", "s2_log.txt"]),
    ("s3_subfield_step5c.py", "s3_step5c_subfield",
     ["subfield_step5c.tsv", "subfield_pair_similarity.tsv", "subfield_step5c_vs_main_review.md", "s3_log.txt"]),
    ("s4_subfield_edge_specificity.py", "s4_edge_specificity_subfield",
     ["subfield_edge_specificity.tsv", "subfield_drop_long.tsv", "subfield_edge_specificity_review.md", "s4_log.txt"]),
    ("s5_subfield_anterior_posterior_test.py", "s5_anterior_posterior_contrast",
     ["anterior_posterior_contrast.tsv", "anterior_posterior_review.md", "s5_log.txt"]),
    ("v1_mds_rhpc_temporal_pole.py", "v1_mds_visualisation",
     ["mds_coords.tsv", "v1_log.txt"]),
    ("n1_three_axis_evidence_table.py", "n1_three_axis_evidence",
     ["three_axis_evidence_table.tsv", "three_axis_evidence_review.md", "n1_log.txt"]),
    ("isolation_check.py", "isolation_check",
     ["output_manifest.tsv", "isolation_report.md", "upstream_git_status.tsv"]),
    ("append_section29.py", "append_section29",
     ["section29_preview.md", "append_section29.log"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--final-root", type=Path, default=None)
    parser.add_argument("--fs60-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--skip-append-section29", action="store_true", default=True,
                        help="默认跳过 §29 追加；显式 --no-skip-append-section29 才会执行。")
    parser.add_argument("--no-skip-append-section29", dest="skip_append_section29",
                        action="store_false", help="允许在最后追加 §29 到主结果文档。")
    parser.add_argument("--dry-run-section29", action="store_true",
                        help="将 --dry-run 透传给 append_section29，不实际改动 result_new_meta_roi.md。")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用 resume；已有产物的模块也会重新触发（但不会覆盖现有文件，会因 FileExistsError 失败）。")
    return parser.parse_args()


def common_args(args: argparse.Namespace) -> list[str]:
    flags = []
    for flag, value in [
        ("--base-dir", args.base_dir),
        ("--paper-output-root", args.paper_output_root),
        ("--output-root", args.output_root),
        ("--final-root", args.final_root),
        ("--fs60-root", args.fs60_root),
        ("--mask-root", args.mask_root),
    ]:
        if value is not None:
            flags.extend([flag, str(value)])
    if args.allow_empty:
        flags.append("--allow-empty")
    return flags


def output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    paper_root = args.paper_output_root or ((args.base_dir or Path("E:/python_metaphor")) / "paper_outputs")
    return Path(paper_root) / "qc" / "hpc_subfield_three_axis"


def module_completed(args: argparse.Namespace, module: str, expected: list[str]) -> bool:
    module_path = output_root(args) / module
    return all((module_path / name).exists() for name in expected)


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    failures = []
    for script, module, expected in SCRIPTS:
        if script == "append_section29.py" and args.skip_append_section29:
            print(f"[{script}] skipped=skip_append_section29")
            continue
        if not args.no_resume and module_completed(args, module, expected):
            print(f"[{script}] skipped=existing_outputs")
            continue
        cmd = [sys.executable, str(here / script), *common_args(args)]
        if script == "append_section29.py" and args.dry_run_section29:
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
