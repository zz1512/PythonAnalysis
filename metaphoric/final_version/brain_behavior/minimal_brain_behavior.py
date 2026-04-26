"""
minimal_brain_behavior.py

Purpose
- Run a minimal brain-behavior analysis aligned with the current primary story.
- Primary behavior metric: YY-KJ memory advantage from refined run-7 behavior.
- Primary neural metric: metaphor-specific differentiation advantage in item-wise RSA.

Primary neural definition
- For each subject, compute condition-wise mean similarity in pre/post.
- Differentiation strength = pre_mean - post_mean (higher = stronger post-learning decrease).
- Primary neural metric = Metaphor differentiation - Spatial differentiation.
- Supplementary neural metric = Metaphor differentiation - Baseline differentiation.

Outputs
- subject_level_neural_summary.tsv
- brain_behavior_primary.tsv
- brain_behavior_primary_correlations.tsv
- analysis_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
RSA_ROOT = Path(__file__).resolve().parents[1] / "rsa_analysis"
if str(RSA_ROOT) not in sys.path:
    sys.path.append(str(RSA_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
import rsa_config as cfg  # noqa: E402


BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _canonical_time(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"pre", "pretest"}:
        return "pre"
    if text in {"post", "posttest"}:
        return "post"
    return text


def _subject_neural_summary(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    working = frame.copy()
    if "time" not in working.columns and "stage" in working.columns:
        working["time"] = working["stage"]
    working["time"] = working["time"].map(_canonical_time)
    working = working[working["condition"].isin(["Metaphor", "Spatial", "Baseline"])].copy()
    grouped = (
        working.groupby(["subject", "condition", "time"], as_index=False)["similarity"]
        .mean()
    )
    pivot = grouped.pivot(index="subject", columns=["condition", "time"], values="similarity")
    pivot.columns = [f"{prefix}_{condition.lower()}_{time}_similarity" for condition, time in pivot.columns]
    pivot = pivot.reset_index()

    for condition in ["metaphor", "spatial", "baseline"]:
        pre_col = f"{prefix}_{condition}_pre_similarity"
        post_col = f"{prefix}_{condition}_post_similarity"
        if pre_col in pivot.columns and post_col in pivot.columns:
            pivot[f"{prefix}_{condition}_differentiation"] = pivot[pre_col] - pivot[post_col]

    meta = f"{prefix}_metaphor_differentiation"
    spa = f"{prefix}_spatial_differentiation"
    base = f"{prefix}_baseline_differentiation"
    if meta in pivot.columns and spa in pivot.columns:
        pivot[f"{prefix}_primary_diff_yy_minus_kj"] = pivot[meta] - pivot[spa]
    if meta in pivot.columns and base in pivot.columns:
        pivot[f"{prefix}_secondary_diff_yy_minus_baseline"] = pivot[meta] - pivot[base]
    return pivot


def _pairwise_correlations(frame: pd.DataFrame, metric_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for neural_col, behavior_col in metric_pairs:
        if neural_col not in frame.columns or behavior_col not in frame.columns:
            continue
        pair = frame[["subject", neural_col, behavior_col]].dropna()
        if len(pair) < 3:
            continue
        pearson_r, pearson_p = stats.pearsonr(pair[neural_col], pair[behavior_col])
        spearman_r, spearman_p = stats.spearmanr(pair[neural_col], pair[behavior_col])
        rows.append(
            {
                "neural_metric": neural_col,
                "behavior_metric": behavior_col,
                "n": int(len(pair)),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal brain-behavior analysis for current primary story.")
    parser.add_argument(
        "--behavior-summary",
        type=Path,
        default=BASE_DIR / "behavior_results" / "refined" / "subject_summary_wide.tsv",
        help="Refined subject-level behavior summary.",
    )
    parser.add_argument(
        "--main-rsa",
        type=Path,
        default=BASE_DIR / "rsa_results_optimized_main_functional" / "rsa_itemwise_details.csv",
        help="Main functional item-wise RSA table.",
    )
    parser.add_argument(
        "--literature-rsa",
        type=Path,
        default=BASE_DIR / "rsa_results_optimized_literature" / "rsa_itemwise_details.csv",
        help="Literature item-wise RSA table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "brain_behavior_results" / "minimal",
        help="Output directory.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    behavior = _read_table(args.behavior_summary).copy()
    behavior["subject"] = behavior["subject"].astype(str)

    main_neural = _subject_neural_summary(_read_table(args.main_rsa), prefix="main_functional")
    lit_neural = _subject_neural_summary(_read_table(args.literature_rsa), prefix="literature")
    neural = behavior.merge(main_neural, how="inner", on="subject").merge(lit_neural, how="left", on="subject")
    write_table(neural, output_dir / "brain_behavior_primary.tsv")

    metric_pairs = [
        ("main_functional_primary_diff_yy_minus_kj", "behavior_primary_diff_yy_minus_kj"),
        ("main_functional_secondary_diff_yy_minus_baseline", "behavior_primary_diff_yy_minus_kj"),
        ("literature_primary_diff_yy_minus_kj", "behavior_primary_diff_yy_minus_kj"),
        ("main_functional_primary_diff_yy_minus_kj", "rt_correct_diff_yy_minus_kj"),
    ]
    corr = _pairwise_correlations(neural, metric_pairs)
    if not corr.empty:
        corr = corr.sort_values(["spearman_p", "pearson_p"], na_position="last").reset_index(drop=True)
    write_table(corr, output_dir / "brain_behavior_primary_correlations.tsv")

    subject_level_neural = main_neural.merge(lit_neural, how="outer", on="subject")
    write_table(subject_level_neural, output_dir / "subject_level_neural_summary.tsv")

    save_json(
        {
            "behavior_summary": str(args.behavior_summary),
            "main_rsa": str(args.main_rsa),
            "literature_rsa": str(args.literature_rsa),
            "primary_behavior_metric": "behavior_primary_diff_yy_minus_kj",
            "supplementary_behavior_metric": "rt_correct_diff_yy_minus_kj",
            "primary_neural_metric": "main_functional_primary_diff_yy_minus_kj",
            "supplementary_neural_metric": "main_functional_secondary_diff_yy_minus_baseline",
            "n_subjects_merged": int(neural["subject"].nunique()),
        },
        output_dir / "analysis_manifest.json",
    )


if __name__ == "__main__":
    main()
