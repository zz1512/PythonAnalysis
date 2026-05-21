#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_lss_index_coverage.py

论文级审计（LSS 索引与 beta 文件一致性）：
给定 lss_index_*.csv（或对齐后的 lss_index_*_aligned.csv），审计以下风险点：
1) 每个 subject×task 的记录数、unique 文件数、beta 文件存在比例（防止 LSS 静默失败）
2) 每个 trial_type 的覆盖情况（用于检查某类 trial 是否系统性缺失）
3) beta_path 重复（多个被试指向同一输出文件是严重数据泄漏/路径错误）

输出（默认到 --out-dir）：
- audit_<index>_by_subject.csv
- audit_<index>_by_trial_type.csv
- audit_<index>_duplicate_beta_path.csv

扩展建议（未来）：
- 读取 lss_main 的运行日志，将“失败原因”映射到索引行，实现可审计的 Exclusion criteria
- 对每个被试计算刺激覆盖率（stimulus_content）并输出“公共刺激阈值”诊断

默认路径与 emo 分析输出保持一致：
- LSS_OUTPUT_ROOT：LSS 输出根目录（默认 /public/home/dingrui/fmri_analysis/zz_analysis/lss_results）
- QC_AUDIT_OUT_DIR：QC 输出目录（默认 <LSS_OUTPUT_ROOT 的上一级>/qc_audit_out）
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _default_lss_output_root() -> Path:
    return Path(os.environ.get("LSS_OUTPUT_ROOT", "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"))


def _default_qc_out_dir() -> Path:
    p = os.environ.get("QC_AUDIT_OUT_DIR", "")
    if p:
        return Path(p)
    return _default_lss_output_root().parent / "qc_audit_out"


def _resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    """
    将索引行解析为 beta 绝对路径。
    兼容两种常见结构：
    - <lss_root>/<task>/<subject>/<file>
    - <lss_root>/<task>/<subject>/run-<run>/<file>
    如果 row['file'] 已是绝对路径，则直接使用。
    """
    rel = str(row.get("file", ""))
    if rel.startswith("/"):
        return Path(rel)
    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))
    run = row.get("run", None)
    if run is None or (isinstance(run, float) and np.isnan(run)):
        return lss_root / task / sub / rel
    run_str = f"run-{int(run)}"
    p1 = lss_root / task / sub / rel
    if p1.exists():
        return p1
    return lss_root / task / sub / run_str / rel


def _basic_required_cols(df: pd.DataFrame) -> List[str]:
    """索引文件至少需要 subject/task/file 三列。"""
    base = ["subject", "task", "file"]
    missing = [c for c in base if c not in df.columns]
    return missing


def _trial_type_from_label(label: str) -> str:
    """从 lss_main 生成的 label（如 Passive_Emo_tr12）提取 trial_type（Passive_Emo）。"""
    s = str(label)
    if "_tr" in s:
        return s.split("_tr")[0]
    return s


def audit_index_file(index_csv: Path, lss_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回多个表：
    - by_subject: subject×task 级别覆盖率与文件存在性
    - by_trial: task×trial_type 级别覆盖与存在性
    - dup_path: beta_path 重复明细
    - dup_key: (subject, task, stimulus_content) 键重复明细（若存在 stimulus_content）
    - key_collision: stimulus_content 规范化后冲突明细（若存在 stimulus_content）
    - cov_stim: stimulus×task 覆盖率表（若存在 stimulus_content）
    - miss_sub: subject×task 缺失统计表（若存在 stimulus_content）
    """
    df = pd.read_csv(index_csv)
    missing = _basic_required_cols(df)
    if missing:
        raise ValueError(f"{index_csv} 缺少必要列: {missing}")

    df["subject"] = df["subject"].astype(str)
    df["task"] = df["task"].astype(str)
    df["file"] = df["file"].astype(str)
    if "stimulus_content" in df.columns:
        df["stimulus_content"] = df["stimulus_content"].astype(str)
    if "label" in df.columns:
        df["trial_type"] = df["label"].astype(str).map(_trial_type_from_label)
    elif "trial_type" not in df.columns:
        df["trial_type"] = ""

    if "run" not in df.columns:
        df["run"] = np.nan

    df["beta_path"] = df.apply(lambda r: str(_resolve_beta_path(lss_root, r)), axis=1)
    df["beta_exists"] = df["beta_path"].map(lambda p: Path(p).exists())

    by_sub = (
        df.groupby(["subject", "task"], as_index=False)
        .agg(
            n_rows=("file", "size"),
            n_unique_files=("file", "nunique"),
            n_exists=("beta_exists", "sum"),
            frac_exists=("beta_exists", "mean"),
            n_trial_types=("trial_type", "nunique"),
        )
        .sort_values(["task", "frac_exists", "n_rows"])
    )

    by_trial = (
        df.groupby(["task", "trial_type"], as_index=False)
        .agg(
            n_rows=("file", "size"),
            n_subjects=("subject", "nunique"),
            frac_exists=("beta_exists", "mean"),
        )
        .sort_values(["task", "n_rows"], ascending=[True, False])
    )

    dup_path = df[df["beta_path"].duplicated(keep=False)].copy()
    dup_path = dup_path.sort_values(["beta_path", "subject", "task"])

    dup_key = df.iloc[0:0].copy()
    key_collision = pd.DataFrame()
    cov_stim = pd.DataFrame()
    miss_sub = pd.DataFrame()

    if "stimulus_content" in df.columns:
        dup_key = df[df.duplicated(subset=["subject", "task", "stimulus_content"], keep=False)].copy()
        if not dup_key.empty:
            dup_key = dup_key.sort_values(["task", "subject", "stimulus_content", "file"])

        k = df["stimulus_content"].astype(str)
        k_norm = (
            k.str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
        df["_stimulus_norm"] = k_norm
        grp = (
            df.groupby(["task", "_stimulus_norm"], as_index=False)
            .agg(
                n_rows=("file", "size"),
                n_subjects=("subject", "nunique"),
                n_unique_keys=("stimulus_content", "nunique"),
                examples=("stimulus_content", lambda x: "|".join(sorted(set(x.astype(str).tolist()))[:10])),
            )
        )
        key_collision = grp[grp["n_unique_keys"] > 1].sort_values(["task", "n_unique_keys", "n_rows"], ascending=[True, False, False])

        cov_stim = (
            df.groupby(["task", "stimulus_content"], as_index=False)
            .agg(
                n_rows=("file", "size"),
                n_subjects=("subject", "nunique"),
                frac_exists=("beta_exists", "mean"),
            )
            .sort_values(["task", "n_subjects", "frac_exists", "stimulus_content"], ascending=[True, False, False, True])
        )

        sub_n = df.groupby(["task", "subject"], as_index=False).agg(n_rows=("file", "size"))
        sub_exist = df.groupby(["task", "subject"], as_index=False).agg(n_exists=("beta_exists", "sum"))
        miss_sub = sub_n.merge(sub_exist, on=["task", "subject"], how="outer")
        miss_sub["n_rows"] = pd.to_numeric(miss_sub["n_rows"], errors="coerce").fillna(0).astype(int)
        miss_sub["n_exists"] = pd.to_numeric(miss_sub["n_exists"], errors="coerce").fillna(0).astype(int)
        miss_sub["n_missing"] = (miss_sub["n_rows"] - miss_sub["n_exists"]).clip(lower=0).astype(int)
        miss_sub["frac_missing"] = (miss_sub["n_missing"] / miss_sub["n_rows"].clip(lower=1)).astype(float)
        miss_sub = miss_sub.sort_values(["task", "frac_missing", "n_missing", "n_rows"], ascending=[True, False, False, False])

    return by_sub, by_trial, dup_path, dup_key, key_collision, cov_stim, miss_sub


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="审计 LSS 索引 CSV 的覆盖率、重复与文件存在性")
    p.add_argument("--index-csv", type=Path, default=None, help="lss_index_*.csv 或 lss_index_*_aligned.csv")
    p.add_argument("--index-dir", type=Path, default=_default_lss_output_root(), help="索引目录（未提供 index-csv 时使用）")
    p.add_argument("--lss-root", type=Path, default=_default_lss_output_root(), help="LSS 输出根目录")
    p.add_argument("--out-dir", type=Path, default=_default_qc_out_dir(), help="输出目录")
    p.add_argument("--min-frac-exists", type=float, default=0.98, help="subject×task 级别文件存在比例下限（低于则判为未通过）")
    p.add_argument("--min-coverage", type=float, default=0.9, help="stimulus 覆盖率阈值（用于导出 common 刺激列表）")
    p.add_argument("--write-missing-matrix", action="store_true", help="导出 subject×stimulus 的存在性矩阵（可能较大）")
    p.add_argument("--max-matrix-stimuli", type=int, default=600, help="write-missing-matrix 时最多导出的刺激数量（按覆盖率排序截断）")
    p.add_argument("--log-top", type=int, default=10, help="FAIL 时在日志中打印的示例条目数量")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_files: List[Path] = []
    if args.index_csv is not None:
        index_files = [Path(args.index_csv)]
    else:
        index_files = sorted(args.index_dir.rglob("lss_index_*.csv"), key=lambda p: p.stat().st_mtime)
        if not index_files:
            raise FileNotFoundError(f"未找到索引文件: {args.index_dir}/lss_index_*.csv")
        index_files = [index_files[-1]]

    for f in index_files:
        by_sub, by_trial, dup_path, dup_key, key_collision, cov_stim, miss_sub = audit_index_file(f, args.lss_root)
        tag = f.stem
        p1 = out_dir / f"audit_{tag}_by_subject.csv"
        p2 = out_dir / f"audit_{tag}_by_trial_type.csv"
        p3 = out_dir / f"audit_{tag}_duplicate_beta_path.csv"
        by_sub.to_csv(p1, index=False)
        by_trial.to_csv(p2, index=False)
        dup_path.to_csv(p3, index=False)

        extra_paths = []
        if isinstance(dup_key, pd.DataFrame) and len(dup_key) > 0:
            p4 = out_dir / f"audit_{tag}_duplicate_subject_stimulus.csv"
            dup_key.to_csv(p4, index=False)
            extra_paths.append(p4)
        if isinstance(key_collision, pd.DataFrame) and len(key_collision) > 0:
            p5 = out_dir / f"audit_{tag}_stimulus_key_collisions.csv"
            key_collision.to_csv(p5, index=False)
            extra_paths.append(p5)
        if isinstance(cov_stim, pd.DataFrame) and len(cov_stim) > 0:
            p6 = out_dir / f"audit_{tag}_coverage_by_stimulus.csv"
            cov_stim.to_csv(p6, index=False)
            extra_paths.append(p6)
        if isinstance(miss_sub, pd.DataFrame) and len(miss_sub) > 0:
            p7 = out_dir / f"audit_{tag}_missing_by_subject.csv"
            miss_sub.to_csv(p7, index=False)
            extra_paths.append(p7)

        if bool(args.write_missing_matrix):
            df = pd.read_csv(f)
            if "stimulus_content" in df.columns and "subject" in df.columns and "task" in df.columns:
                df["subject"] = df["subject"].astype(str)
                df["task"] = df["task"].astype(str)
                df["stimulus_content"] = df["stimulus_content"].astype(str)
                df["beta_path"] = df.apply(lambda r: str(_resolve_beta_path(Path(args.lss_root), r)), axis=1)
                df["beta_exists"] = df["beta_path"].map(lambda p: Path(p).exists())
                for task_name in sorted(df["task"].unique().tolist()):
                    sub_df = df[df["task"] == task_name].copy()
                    if sub_df.empty:
                        continue
                    cov = sub_df.groupby("stimulus_content", as_index=False).agg(n_subjects=("subject", "nunique"))
                    cov = cov.sort_values(["n_subjects", "stimulus_content"], ascending=[False, True])
                    max_stim = int(args.max_matrix_stimuli)
                    stim_keep = cov["stimulus_content"].astype(str).tolist()[:max_stim]
                    sub_keep = sorted(sub_df["subject"].astype(str).unique().tolist())
                    mat = (
                        sub_df[sub_df["stimulus_content"].isin(stim_keep)]
                        .pivot_table(index="subject", columns="stimulus_content", values="beta_exists", aggfunc="max")
                        .reindex(index=sub_keep, columns=stim_keep)
                    )
                    mat = mat.fillna(False).astype(int)
                    out_m = out_dir / f"audit_{tag}_missing_matrix__{task_name}.csv"
                    mat.to_csv(out_m)
                    extra_paths.append(out_m)

        bad_frac = by_sub[by_sub["frac_exists"].to_numpy(dtype=float) < float(args.min_frac_exists)]
        bad_unique = by_sub[by_sub["n_unique_files"].to_numpy(dtype=float) != by_sub["n_rows"].to_numpy(dtype=float)]
        ok = (len(dup_path) == 0) and (len(bad_frac) == 0) and (len(bad_unique) == 0) and (len(dup_key) == 0 if isinstance(dup_key, pd.DataFrame) else True)
        status = "PASS" if ok else "FAIL"
        print(f"[QC] audit_lss_index_coverage ({tag}): {status}")
        print(f"[QC] Saved: {p1}")
        print(f"[QC] Saved: {p2}")
        print(f"[QC] Saved: {p3}")
        for ep in extra_paths:
            print(f"[QC] Saved: {ep}")
        print(
            f"[QC] Subjects×Task={len(by_sub)} | TrialType={len(by_trial)} | "
            f"DupPaths={len(dup_path)} | LowFrac(<{float(args.min_frac_exists):.3f})={len(bad_frac)} | "
            f"FileDuplicates(n_unique!=n_rows)={len(bad_unique)}"
        )

        if status == "FAIL":
            top_n = int(args.log_top)
            if top_n > 0 and len(bad_frac) > 0:
                worst = bad_frac.sort_values(["frac_exists", "n_rows"]).head(top_n)
                print(f"[QC] LowFrac examples (showing up to {top_n}):")
                print(worst[["subject", "task", "n_rows", "n_exists", "frac_exists"]].to_string(index=False))

            if top_n > 0 and len(bad_unique) > 0:
                ex = bad_unique.sort_values(["task", "subject"]).head(top_n)
                print(f"[QC] FileDuplicates examples (showing up to {top_n}):")
                print(ex[["subject", "task", "n_rows", "n_unique_files"]].to_string(index=False))

            if top_n > 0 and len(dup_path) > 0:
                dup_small = dup_path[["beta_path", "subject", "task", "run", "file"]].head(top_n)
                print(f"[QC] Duplicated beta_path examples (showing up to {top_n}):")
                print(dup_small.to_string(index=False))

            if top_n > 0 and len(by_trial) > 0:
                worst_trial = by_trial.sort_values(["frac_exists", "n_subjects", "n_rows"]).head(top_n)
                print(f"[QC] TrialType coverage (lowest frac_exists, up to {top_n}):")
                print(worst_trial[["task", "trial_type", "n_subjects", "n_rows", "frac_exists"]].to_string(index=False))


if __name__ == "__main__":
    main()
