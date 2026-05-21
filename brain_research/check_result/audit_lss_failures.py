#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_lss_failures.py

论文级审计（LSS 失败原因汇总）：
读取 lss_main 输出的 lss_audit_*.csv，汇总：
- run 级别失败（缺 inputs/confounds/mask 等）
- trial 级别 success/fail/skipped 的比例
- fail stage 分布（create_lss_events/fit/contrast/save 等）
- 最容易失败的被试（fail count / success rate）

输出：
- audit_lss_failures_summary.csv
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="汇总 lss_audit_*.csv 并输出论文可用的失败审计表")
    p.add_argument("--dir", type=Path, default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"), help="搜索目录")
    p.add_argument("--glob", type=str, default="**/lss_audit_*.csv", help="审计文件 glob")
    p.add_argument("--out-dir", type=Path, default=Path("./qc_audit_out"), help="输出目录")
    p.add_argument("--min-trial-success-rate", type=float, default=0.9, help="trial 成功率低于该阈值判为 FAIL")
    p.add_argument("--log-top", type=int, default=10, help="FAIL 时打印示例条目数量")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    files = sorted(Path(args.dir).glob(args.glob), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"未找到审计文件: dir={args.dir}, glob={args.glob}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["audit_file"] = str(f)
        dfs.append(df)
    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    for c in ["subject", "task", "status", "stage"]:
        if c in all_df.columns:
            all_df[c] = all_df[c].astype(str)

    run_fail = all_df[all_df["trial_idx"].isna()] if "trial_idx" in all_df.columns else all_df.iloc[0:0]
    trial_df = all_df[all_df["trial_idx"].notna()] if "trial_idx" in all_df.columns else all_df

    overall = (
        trial_df.groupby(["audit_file"], as_index=False)
        .agg(
            n_trials=("status", "size"),
            n_success=("status", lambda s: int((s == "success").sum())),
            n_fail=("status", lambda s: int((s == "fail").sum())),
            n_skipped=("status", lambda s: int((s == "skipped").sum())),
        )
    )
    overall["trial_success_rate"] = overall["n_success"] / overall["n_trials"].clip(lower=1)

    stage_counts = (
        trial_df[trial_df["status"] == "fail"]
        .groupby(["audit_file", "stage"], as_index=False)
        .size()
        .rename(columns={"size": "n_fail"})
        .sort_values(["audit_file", "n_fail"], ascending=[True, False])
    )

    subj = (
        trial_df.groupby(["audit_file", "task", "subject"], as_index=False)
        .agg(
            n_trials=("status", "size"),
            n_success=("status", lambda s: int((s == "success").sum())),
            n_fail=("status", lambda s: int((s == "fail").sum())),
            n_skipped=("status", lambda s: int((s == "skipped").sum())),
        )
    )
    subj["trial_success_rate"] = subj["n_success"] / subj["n_trials"].clip(lower=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_summary = out_dir / "audit_lss_failures_summary.csv"
    overall.to_csv(out_summary, index=False)
    out_stage = out_dir / "audit_lss_failures_by_stage.csv"
    stage_counts.to_csv(out_stage, index=False)
    out_subj = out_dir / "audit_lss_failures_by_subject.csv"
    subj.to_csv(out_subj, index=False)

    bad_run_fail = run_fail[run_fail["status"] == "fail"] if not run_fail.empty and "status" in run_fail.columns else run_fail
    bad_success = overall[overall["trial_success_rate"] < float(args.min_trial_success_rate)]

    ok = (len(bad_run_fail) == 0) and (len(bad_success) == 0)
    status = "PASS" if ok else "FAIL"
    print(f"[QC] audit_lss_failures: {status}")
    print(f"[QC] Saved: {out_summary}")
    print(f"[QC] Saved: {out_stage}")
    print(f"[QC] Saved: {out_subj}")
    print(
        f"[QC] AuditFiles={len(overall)} | RunLevelFail={len(bad_run_fail)} | "
        f"LowTrialSuccess(<{float(args.min_trial_success_rate):.3f})={len(bad_success)}"
    )

    if status == "FAIL":
        top_n = int(args.log_top)
        if top_n > 0 and len(bad_run_fail) > 0:
            cols = [c for c in ["audit_file", "task", "subject", "run", "stage", "error"] if c in bad_run_fail.columns]
            print(f"[QC] Run-level failures (showing up to {top_n}):")
            print(bad_run_fail[cols].head(top_n).to_string(index=False))

        if top_n > 0 and len(bad_success) > 0:
            print(f"[QC] Low trial success rate files (showing up to {top_n}):")
            print(bad_success.head(top_n).to_string(index=False))

        if top_n > 0 and not stage_counts.empty:
            print(f"[QC] Most common failure stages (showing up to {top_n}):")
            print(stage_counts.head(top_n).to_string(index=False))

        if top_n > 0 and not subj.empty:
            worst_sub = subj.sort_values(["trial_success_rate", "n_fail"], ascending=[True, False]).head(top_n)
            print(f"[QC] Worst subjects (showing up to {top_n}):")
            print(worst_sub.to_string(index=False))


if __name__ == "__main__":
    main()

