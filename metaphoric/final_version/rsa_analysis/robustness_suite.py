"""
robustness_suite.py

用途
- 为 Step 5C 的主结果提供最小但高价值的稳健性证据包：
  - Bayes Factor
  - bootstrap CI
  - LOSO (leave-one-subject-out)
  - split-half replication
- 统一围绕 subject-level interaction delta 展开，避免不同稳健性分析使用不同的效应定义。

效应定义
- 先按 `subject × roi_scope × condition × time` 求均值 similarity
- 再构造差异中的差异（interaction delta）：
  - yy_vs_baseline: (yy_post - yy_pre) - (baseline_post - baseline_pre)
  - yy_vs_kj:       (yy_post - yy_pre) - (kj_post - kj_pre)

默认输入
- paper_outputs/qc/rsa_results_optimized_main_functional/rsa_itemwise_details.csv
- paper_outputs/qc/rsa_results_optimized_literature/rsa_itemwise_details.csv
- paper_outputs/qc/rsa_results_optimized_literature_spatial/rsa_itemwise_details.csv

默认输出
- paper_outputs/tables_si/table_bayes_factors.tsv
- paper_outputs/tables_si/table_bootstrap_ci.tsv
- paper_outputs/tables_si/table_loso.tsv
- paper_outputs/tables_si/table_splithalf.tsv
- paper_outputs/qc/robustness_subject_interaction.tsv
- paper_outputs/qc/robustness_meta.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import (  # noqa: E402
    difference_in_differences,
    ensure_dir,
    one_sample_t_summary,
    read_table,
    save_json,
    write_table,
)
from utils.bayesian_stats import bayes_factor_ttest  # noqa: E402
from utils.bootstrap_ci import bootstrap_mean_ci  # noqa: E402


CONDITION_MAP = {
    "metaphor": "yy",
    "yy": "yy",
    "spatial": "kj",
    "kj": "kj",
    "baseline": "baseline",
}


def _default_base_dir() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402
    return Path(BASE_DIR)


def _default_rsa_files(base_dir: Path) -> list[Path]:
    return [
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_main_functional" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_literature" / "rsa_itemwise_details.csv",
        base_dir / "paper_outputs" / "qc" / "rsa_results_optimized_literature_spatial" / "rsa_itemwise_details.csv",
    ]


def _canonical_time(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "time" not in out.columns and "stage" in out.columns:
        out["time"] = out["stage"]
    out["time"] = (
        out["time"].astype(str).str.strip().str.lower().map({"pre": "pre", "post": "post", "1": "pre", "2": "post"})
    )
    return out


def _infer_roi_set(path: Path) -> str:
    name = path.parent.name
    prefix = "rsa_results_optimized_"
    if name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix):]
    return name


def _infer_main_family(roi_name: str) -> str:
    if "Metaphor_gt_Spatial" in roi_name:
        return "Metaphor_gt_Spatial"
    if "Spatial_gt_Metaphor" in roi_name:
        return "Spatial_gt_Metaphor"
    return "main_functional_other"


def _build_subject_scope_frame(rsa_files: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in rsa_files:
        if not path.exists():
            continue
        frame = _canonical_time(read_table(path))
        if frame.empty:
            continue
        frame["condition"] = frame["condition"].astype(str).str.strip().str.lower().map(CONDITION_MAP)
        frame = frame[frame["condition"].isin(["yy", "kj", "baseline"])].copy()
        frame["similarity"] = pd.to_numeric(frame["similarity"], errors="coerce")
        frame = frame.dropna(subset=["subject", "roi", "condition", "time", "similarity"])
        if frame.empty:
            continue

        roi_set = _infer_roi_set(path)
        pooled = (
            frame.groupby(["subject", "condition", "time"], as_index=False)["similarity"].mean()
            .assign(roi_scope=f"{roi_set}__all", roi_set=roi_set)
        )
        rows.append(pooled)

        if roi_set == "main_functional":
            fam = frame.copy()
            fam["roi_scope"] = fam["roi"].astype(str).map(_infer_main_family).map(lambda x: f"main_functional__{x}")
            fam = fam.groupby(["subject", "condition", "time", "roi_scope"], as_index=False)["similarity"].mean()
            fam["roi_set"] = "main_functional"
            rows.append(fam)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["subject", "condition", "time", "roi_scope"], keep="last")
    return out


def _compute_interaction_rows(subject_scope_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if subject_scope_frame.empty:
        return pd.DataFrame()
    contrasts = [
        ("yy_vs_baseline", "yy", "baseline"),
        ("yy_vs_kj", "yy", "kj"),
    ]
    for roi_scope, scope_frame in subject_scope_frame.groupby("roi_scope"):
        roi_set = scope_frame["roi_set"].iloc[0]
        for contrast_name, metaphor_label, control_label in contrasts:
            try:
                did = difference_in_differences(
                    scope_frame.rename(columns={"similarity": "value"}),
                    value_col="value",
                    metaphor_label=metaphor_label,
                    control_label=control_label,
                )
            except Exception:
                continue
            rows.append(
                did[["subject", "yy_delta", "kj_delta", "interaction_delta"]]
                .assign(roi_scope=roi_scope, roi_set=roi_set, contrast=contrast_name)
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bayes_table(interaction_frame: pd.DataFrame, *, alternative: str, r_scale: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (roi_scope, contrast), frame in interaction_frame.groupby(["roi_scope", "contrast"]):
        result = bayes_factor_ttest(
            frame["interaction_delta"].to_numpy(dtype=float),
            paired=False,
            alternative=alternative,
            r=r_scale,
        )
        rows.append(
            {
                "roi_scope": roi_scope,
                "roi_set": frame["roi_set"].iloc[0],
                "contrast": contrast,
                **result,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_table(
    interaction_frame: pd.DataFrame,
    *,
    confidence: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (roi_scope, contrast), frame in interaction_frame.groupby(["roi_scope", "contrast"]):
        summary = bootstrap_mean_ci(frame["interaction_delta"].to_numpy(dtype=float), confidence, n_boot, seed)
        rows.append(
            {
                "roi_scope": roi_scope,
                "roi_set": frame["roi_set"].iloc[0],
                "contrast": contrast,
                "n": int(frame["interaction_delta"].notna().sum()),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _loso_table(interaction_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (roi_scope, contrast), frame in interaction_frame.groupby(["roi_scope", "contrast"]):
        subjects = sorted(frame["subject"].astype(str).unique().tolist())
        full_summary = one_sample_t_summary(frame["interaction_delta"].to_numpy(dtype=float), popmean=0.0)
        full_sign = np.sign(full_summary["mean"]) if np.isfinite(full_summary["mean"]) else np.nan
        for held_out in subjects:
            subset = frame[frame["subject"] != held_out]
            summary = one_sample_t_summary(subset["interaction_delta"].to_numpy(dtype=float), popmean=0.0)
            subset_sign = np.sign(summary["mean"]) if np.isfinite(summary["mean"]) else np.nan
            rows.append(
                {
                    "roi_scope": roi_scope,
                    "roi_set": frame["roi_set"].iloc[0],
                    "contrast": contrast,
                    "held_out_subject": held_out,
                    "n_remaining": int(summary["n"]),
                    "mean_interaction_delta": summary["mean"],
                    "t": summary["t"],
                    "p": summary["p"],
                    "full_sample_mean": full_summary["mean"],
                    "same_direction_as_full": bool(np.isfinite(full_sign) and np.isfinite(subset_sign) and subset_sign == full_sign),
                }
            )
    return pd.DataFrame(rows)


def _split_half_table(interaction_frame: pd.DataFrame, *, n_splits: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for (roi_scope, contrast), frame in interaction_frame.groupby(["roi_scope", "contrast"]):
        work = frame[["subject", "interaction_delta"]].dropna().copy()
        subjects = sorted(work["subject"].astype(str).unique().tolist())
        if len(subjects) < 6:
            continue
        n_half = len(subjects) // 2
        for split_idx in range(n_splits):
            perm = rng.permutation(subjects)
            half_a = set(perm[:n_half].tolist())
            half_b = set(perm[n_half:].tolist())
            a_vals = work.loc[work["subject"].isin(half_a), "interaction_delta"].to_numpy(dtype=float)
            b_vals = work.loc[work["subject"].isin(half_b), "interaction_delta"].to_numpy(dtype=float)
            a_sum = one_sample_t_summary(a_vals, popmean=0.0)
            b_sum = one_sample_t_summary(b_vals, popmean=0.0)
            rows.append(
                {
                    "roi_scope": roi_scope,
                    "roi_set": frame["roi_set"].iloc[0],
                    "contrast": contrast,
                    "split_index": int(split_idx),
                    "n_half_a": int(a_sum["n"]),
                    "n_half_b": int(b_sum["n"]),
                    "mean_half_a": a_sum["mean"],
                    "mean_half_b": b_sum["mean"],
                    "t_half_a": a_sum["t"],
                    "t_half_b": b_sum["t"],
                    "p_half_a": a_sum["p"],
                    "p_half_b": b_sum["p"],
                    "same_direction": bool(
                        np.isfinite(a_sum["mean"]) and np.isfinite(b_sum["mean"]) and np.sign(a_sum["mean"]) == np.sign(b_sum["mean"])
                    ),
                    "both_positive": bool(np.isfinite(a_sum["mean"]) and np.isfinite(b_sum["mean"]) and a_sum["mean"] > 0 and b_sum["mean"] > 0),
                    "both_p_lt_0_05": bool(
                        np.isfinite(a_sum["p"]) and np.isfinite(b_sum["p"]) and a_sum["p"] < 0.05 and b_sum["p"] < 0.05
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness suite for Step 5C primary endpoint (BF, bootstrap, LOSO, split-half).")
    parser.add_argument("--rsa-details", nargs="*", type=Path, default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")
    parser.add_argument("--r-scale", type=float, default=0.707)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=200)
    args = parser.parse_args()

    base_dir = _default_base_dir()
    paper_root = args.paper_output_root or (base_dir / "paper_outputs")
    tables_si = ensure_dir(paper_root / "tables_si")
    qc_dir = ensure_dir(paper_root / "qc")
    rsa_files = args.rsa_details or _default_rsa_files(base_dir)

    subject_scope = _build_subject_scope_frame(rsa_files)
    interaction_frame = _compute_interaction_rows(subject_scope)

    bayes_df = _bayes_table(interaction_frame, alternative=args.alternative, r_scale=args.r_scale)
    bootstrap_df = _bootstrap_table(
        interaction_frame,
        confidence=args.confidence,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    loso_df = _loso_table(interaction_frame)
    splithalf_df = _split_half_table(interaction_frame, n_splits=args.n_splits, seed=args.seed)

    write_table(bayes_df, tables_si / "table_bayes_factors.tsv")
    write_table(bootstrap_df, tables_si / "table_bootstrap_ci.tsv")
    write_table(loso_df, tables_si / "table_loso.tsv")
    write_table(splithalf_df, tables_si / "table_splithalf.tsv")
    write_table(interaction_frame, qc_dir / "robustness_subject_interaction.tsv")
    save_json(
        {
            "rsa_details_files": [str(path) for path in rsa_files],
            "alternative": args.alternative,
            "r_scale": float(args.r_scale),
            "confidence": float(args.confidence),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "n_splits": int(args.n_splits),
            "n_subject_rows": int(len(interaction_frame)),
            "roi_scopes": sorted(interaction_frame["roi_scope"].unique().tolist()) if not interaction_frame.empty else [],
            "contrasts": sorted(interaction_frame["contrast"].unique().tolist()) if not interaction_frame.empty else [],
        },
        qc_dir / "robustness_meta.json",
    )


if __name__ == "__main__":
    main()
