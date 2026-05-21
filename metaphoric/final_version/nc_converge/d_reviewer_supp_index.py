#!/usr/bin/env python3
"""D: index reviewer_supp outputs without rerunning analyses by default."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, default_config, discover_tables, sha256, write_outputs

MODULE = "d_reviewer_supp_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--run-reviewer-supp", action="store_true", help="Execute reviewer_supp/run_all_reviewer_supp.py before indexing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    logs = []
    script = cfg.final_root / "reviewer_supp" / "run_all_reviewer_supp.py"
    if args.run_reviewer_supp:
        result = subprocess.run(["python", str(script), "--base-dir", str(cfg.base_dir)], text=True, capture_output=True, check=False)
        logs.append({"step": "run_reviewer_supp", "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]})
        if result.returncode != 0:
            raise RuntimeError("reviewer_supp run failed; see log output.")
    reviewer_root = cfg.paper_output_root / "qc" / "reviewer_supp"
    paths = discover_tables(reviewer_root, ["**/*.tsv", "**/*.json", "**/*.md", "**/*.log"])
    rows = []
    for path in paths:
        rows.append({"path": str(path), "module": path.parent.name, "suffix": path.suffix, "sha256": sha256(path)})
    write_outputs(cfg, MODULE, {"reviewer_supp_index.tsv": pd.DataFrame(rows), "reviewer_supp_index_log.tsv": pd.DataFrame(logs)})


if __name__ == "__main__":
    main()
