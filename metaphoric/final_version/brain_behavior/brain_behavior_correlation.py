"""
brain_behavior_correlation.py

用途
- 汇总多个“subject-level 指标表”（例如 RSA/RD/GPS/MVPA/行为）到同一张表，并输出相关矩阵与显著性。
- 这个脚本主要负责“把结果拼起来”，不负责重新跑上游分析。

输入
- output_dir: 输出目录
- metrics: 若干 TSV/CSV 文件（每个文件必须包含 `subject` 列；默认要求已经是 subject-level）

输出（output_dir）
- `brain_behavior_merged.tsv`：合并后的 subject-level 指标表（供进一步回归/中介使用）
- `brain_behavior_correlations.tsv`：两两相关结果（含 p 值）
- `brain_behavior_meta.json`：元信息
"""

from __future__ import annotations



import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, safe_merge, save_json, write_table


def summarise_file(path: Path, *, force_subject_mean: bool = False) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    if "subject" not in frame.columns:
        raise ValueError(f"Missing subject column: {path}")
    numeric_cols = [column for column in frame.columns if column != "subject" and pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric_cols:
        return pd.DataFrame(columns=["subject"])
    duplicate_counts = frame["subject"].value_counts()
    has_duplicates = bool((duplicate_counts > 1).any())
    if has_duplicates and not force_subject_mean:
        example_subject = str(duplicate_counts[duplicate_counts > 1].index[0])
        raise ValueError(
            f"{path} contains multiple rows per subject (e.g. {example_subject}). "
            "This script now requires pre-aggregated subject-level inputs by default. "
            "Aggregate upstream or rerun with --force-subject-mean if collapsing rows is intentional."
        )
    grouped = (
        frame.groupby("subject", as_index=False)[numeric_cols].mean()
        if has_duplicates
        else frame[["subject", *numeric_cols]].copy()
    )
    rename = {column: f"{path.stem}_{column}" for column in numeric_cols}
    return grouped.rename(columns=rename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multi-layer neural metrics and compute brain-behavior correlations.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--force-subject-mean",
        action="store_true",
        help="Explicitly allow collapsing duplicated subject rows by numeric mean. Use only for intentionally preselected summaries.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    merged = safe_merge(
        [summarise_file(path, force_subject_mean=args.force_subject_mean) for path in args.inputs],
        on="subject",
    )
    write_table(merged, output_dir / "brain_behavior_merged.tsv")

    numeric_cols = [column for column in merged.columns if column != "subject" and pd.api.types.is_numeric_dtype(merged[column])]
    rows = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            pair = merged[[col_a, col_b]].dropna()
            if len(pair) < 3:
                continue
            pearson_r, pearson_p = stats.pearsonr(pair[col_a], pair[col_b])
            spearman_r, spearman_p = stats.spearmanr(pair[col_a], pair[col_b])
            rows.append({
                "metric_a": col_a,
                "metric_b": col_b,
                "n": len(pair),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })

    corr_frame = pd.DataFrame(rows).sort_values(["pearson_p", "spearman_p"], na_position="last") if rows else pd.DataFrame(columns=["metric_a", "metric_b", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
    write_table(corr_frame, output_dir / "brain_behavior_correlations.tsv")
    save_json(
        {
            "n_subjects": int(merged["subject"].nunique()),
            "n_numeric_metrics": int(len(numeric_cols)),
            "force_subject_mean": bool(args.force_subject_mean),
        },
        output_dir / "brain_behavior_summary.json",
    )


if __name__ == "__main__":
    main()
