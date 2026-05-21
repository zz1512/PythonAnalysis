#!/usr/bin/env python3
"""Isolation check for nc_converge outputs and untouched upstream files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, default_config, discover_tables, git_status, markdown_table, sha256, write_outputs

MODULE = "isolation_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--upstream", nargs="*", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    outputs = discover_tables(cfg.output_root, ["**/*"])
    output_rows = [{"path": str(p), "suffix": p.suffix, "sha256": sha256(p)} for p in outputs if p.is_file()]
    upstream_paths = args.upstream or [cfg.final_root / "result_new_meta_roi.md", cfg.final_root / "reviewer_supp", cfg.final_root / "brain_behavior", cfg.final_root / "rsa_analysis"]
    upstream_rows = []
    for path in upstream_paths:
        upstream_rows.append({"path": str(path), "git_status": git_status(path)})
    report = "# NC Converge Isolation Report\n\n"
    report += "All nc_converge outputs are expected under `paper_outputs/qc/nc_converge`. Existing upstream scripts/results should remain untouched except append-only Section 28 when explicitly run.\n\n"
    report += markdown_table(pd.DataFrame(upstream_rows))
    write_outputs(cfg, MODULE, {"output_manifest.tsv": pd.DataFrame(output_rows), "isolation_report.md": report, "upstream_git_status.tsv": pd.DataFrame(upstream_rows)})


if __name__ == "__main__":
    main()
