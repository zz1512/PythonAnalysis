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


def summarise_file(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    if "subject" not in frame.columns:
        raise ValueError(f"Missing subject column: {path}")
    numeric_cols = [column for column in frame.columns if column != "subject" and pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric_cols:
        return pd.DataFrame(columns=["subject"])
    grouped = frame.groupby("subject", as_index=False)[numeric_cols].mean()
    rename = {column: f"{path.stem}_{column}" for column in numeric_cols}
    return grouped.rename(columns=rename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multi-layer neural metrics and compute brain-behavior correlations.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    merged = safe_merge([summarise_file(path) for path in args.inputs], on="subject")
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
    save_json({"n_subjects": int(merged["subject"].nunique()), "n_numeric_metrics": int(len(numeric_cols))}, output_dir / "brain_behavior_summary.json")


if __name__ == "__main__":
    main()
