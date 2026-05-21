#!/usr/bin/env python3
"""ISO: 隔离自检。

- 列出所有 paper_outputs/qc/hpc_subfield_three_axis/ 下产物的 sha256；
- 检查 result_new_meta_roi.md 与既有 nc_converge / reviewer_supp / step5c 等关键路径
  的 git status 是否仅出现期望的"末尾追加 §29"变化。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shared_subfield import (
    add_common_args,
    default_config,
    git_status,
    markdown_table,
    sha256,
    write_outputs,
)

MODULE = "isolation_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--upstream", nargs="*", type=Path, default=None,
                        help="需检查 git status 的关键上游路径列表。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)

    out_paths = sorted(p for p in Path(cfg.output_root).rglob("*") if p.is_file())
    output_rows = [{"path": str(p), "suffix": p.suffix, "sha256": sha256(p)} for p in out_paths]

    upstream_paths = args.upstream or [
        cfg.final_root / "result_new_meta_roi.md",
        cfg.final_root / "reviewer_supp",
        cfg.final_root / "nc_converge",
        cfg.final_root / "brain_behavior",
        cfg.final_root / "rsa_analysis",
        cfg.final_root / "step5c",
    ]
    upstream_rows = [{"path": str(p), "git_status": git_status(p)} for p in upstream_paths]

    report = "# HPC Subfield Three-Axis Isolation Report\n\n"
    report += (
        "All hpc_subfield_three_axis outputs are expected under "
        "`paper_outputs/qc/hpc_subfield_three_axis`. The only allowed change to upstream "
        "is an append-only Section 29 in `result_new_meta_roi.md` (executed via `append_section29.py`).\n\n"
    )
    report += "## Upstream git status\n\n"
    report += markdown_table(pd.DataFrame(upstream_rows))

    write_outputs(cfg, MODULE, {
        "output_manifest.tsv": pd.DataFrame(output_rows),
        "isolation_report.md": report,
        "upstream_git_status.tsv": pd.DataFrame(upstream_rows),
    })


if __name__ == "__main__":
    main()
