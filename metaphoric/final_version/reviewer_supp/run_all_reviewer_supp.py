#!/usr/bin/env python3
"""Run reviewer_supp analysis modules in sequence on the analysis machine.

This file is only an orchestrator. It does not run unless called explicitly.
By default it runs only M2/M3/M4/M5/M6/M9/M11. Section 27 preview and
isolation checks are opt-in so a batch run cannot silently edit the result file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULES = [
    "m2_univariate_sanity.py",
    "m3_learning_dm.py",
    "m4_correct_ers.py",
    "m5_theoretical_reframing.py",
    "m6_novelty_repetition.py",
    "m9_logistic_dm.py",
    "m11_kj_fair_characterization.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append-preview", action="store_true", help="Generate section27_preview.md after module runs.")
    parser.add_argument("--isolation-check", action="store_true", help="Run isolation_check.py after module runs.")
    parser.add_argument("module_args", nargs=argparse.REMAINDER, help="Arguments forwarded to each module after --")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    forwarded = args.module_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    for module in MODULES:
        cmd = [sys.executable, str(root / module), *forwarded]
        subprocess.run(cmd, check=True)
    if args.append_preview:
        subprocess.run([sys.executable, str(root / "append_section27.py"), *forwarded], check=True)
    if args.isolation_check:
        subprocess.run([sys.executable, str(root / "isolation_check.py"), *forwarded], check=True)


if __name__ == "__main__":
    main()
