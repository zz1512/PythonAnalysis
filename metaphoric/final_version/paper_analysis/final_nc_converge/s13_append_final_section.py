#!/usr/bin/env python3
"""Task 14: dry-run append final section to result documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared_final import FINAL_ROOT, final_out, write_text

MODULE = "s13_append_final_section"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=FINAL_ROOT / "result_final.md")
    parser.add_argument("--section-number", default="Final NC convergence")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--write", action="store_true", help="Actually append to target; dry-run is default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    section = f"\n\n## {args.section_number}\n\nFinal NC convergence outputs are available under `paper_outputs/qc/final_nc_converge/`. The final figures are collected in `figure_final/`. Interpret B2 as a Mahalanobis robustness proxy, HPC long-axis as exploratory, and memory bridging as prior post-stage separation rather than pure retrieval reinstatement.\n"
    write_text(section.strip() + "\n", final_out(MODULE, "append_preview.md"))
    log = f"target={args.target}\ndry_run={not args.write}\nskip_if_exists={args.skip_if_exists}\n"
    if args.write:
        text = args.target.read_text(encoding="utf-8") if args.target.exists() else ""
        if args.skip_if_exists and args.section_number in text:
            log += "status=skipped_existing_section\n"
        else:
            args.target.write_text(text + section, encoding="utf-8")
            log += "status=appended\n"
    else:
        log += "status=dry_run_only\n"
    write_text(log, final_out(MODULE, "append_log.txt"))


if __name__ == "__main__":
    main()

