"""
export_gppi_summary.py

Purpose
- Post-hoc exporter for S2 gPPI outputs.
- It reads the per-seed SI tables already written by `gPPI_analysis.py`,
  then consolidates them into:
  - `paper_outputs/tables_main/table_gppi_summary_main.tsv`
  - `paper_outputs/tables_si/table_gppi_summary_full.tsv`

Why a separate script
- Do not touch or restart the currently running gPPI job.
- Keep the heavy first-level/second-level computation unchanged.
- Only add a light-weight export layer that normalizes final paper tables.
"""

from __future__ import annotations

import argparse
import os
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

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402


PSEUDO_TARGETS = {"__scope_mean__", "__cross_sets__"}


def _default_paper_root() -> Path:
    try:
        from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

        return Path(BASE_DIR) / "paper_outputs"
    except Exception:
        return Path(os.environ.get("PYTHON_METAPHOR_ROOT", ".")).resolve() / "paper_outputs"


def _candidate_tables(tables_si: Path) -> list[Path]:
    paths = sorted(tables_si.glob("table_gppi_summary_*.tsv"))
    excluded = {
        "table_gppi_summary_main.tsv",
        "table_gppi_summary_full.tsv",
    }
    return [path for path in paths if path.name not in excluded]


def _candidate_subject_metric_tables(qc_root: Path) -> list[Path]:
    combined = sorted(qc_root.glob("gppi_combined*/gppi_subject_metrics_all_seeds.tsv"))
    if combined:
        return combined
    return sorted(qc_root.glob("gppi_results_*/gppi_subject_metrics.tsv"))


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q_values = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return pd.Series(q_values, index=p_values.index)
    valid_indices = np.flatnonzero(valid)
    order = np.argsort(values[valid])
    ranked = values[valid][order]
    adjusted = ranked * float(len(ranked)) / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values[valid_indices[order]] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q_values, index=p_values.index)


def _read_seed_tables(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = read_table(path).copy()
        if frame.empty:
            continue
        frame["source_table"] = str(path)
        if "seed" not in frame.columns and "seed_roi" in frame.columns:
            frame["seed"] = frame["seed_roi"]
        if "roi_set" not in frame.columns:
            frame["roi_set"] = pd.NA
        if "deconvolution" not in frame.columns:
            frame["deconvolution"] = pd.NA
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    numeric_cols = ["n", "mean_post", "mean_pre", "t", "p", "cohens_dz", "mean_diff"]
    for col in numeric_cols:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")
    return full


def _read_subject_metric_tables(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = read_table(path).copy()
        if frame.empty:
            continue
        frame["source_metric_table"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True).drop_duplicates()
    if "seed" not in metrics.columns and "seed_roi" in metrics.columns:
        metrics["seed"] = metrics["seed_roi"]
    metrics["gppi_beta"] = pd.to_numeric(metrics["gppi_beta"], errors="coerce")
    return metrics


def _paired_t_from_diff(diff: np.ndarray) -> tuple[int, float, float, float, float, float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n < 2:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    if not np.isfinite(sd) or np.isclose(sd, 0.0):
        t_val = 0.0
        p_val = 1.0
        dz = 0.0
    else:
        t_val = float(mean_diff / (sd / np.sqrt(n)))
        p_val = float(2.0 * stats.t.sf(abs(t_val), df=n - 1))
        dz = float(mean_diff / sd)
    return n, mean_diff, float(np.nan), float(np.nan), t_val, p_val, dz


def _sign_flip_permutation_p(diff: np.ndarray, *, n_permutations: int, rng: np.random.Generator) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float("nan")
    observed = abs(float(np.mean(diff)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diff.size), replace=True)
    null = np.abs((signs * diff[None, :]).mean(axis=1))
    return float((np.sum(null >= observed) + 1.0) / (n_permutations + 1.0))


def _build_target_level_table(
    metrics: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    required = {"subject", "seed_roi", "target_roi", "target_roi_set", "target_scope", "ppi_condition", "phase", "gppi_beta"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"gPPI subject metrics missing columns: {sorted(missing)}")
    work = metrics[metrics["phase"].astype(str).isin(["pre", "post"])].copy()
    if work.empty:
        return pd.DataFrame()
    subj_phase = (
        work.groupby(
            ["subject", "seed_roi", "target_roi", "target_roi_set", "target_scope", "ppi_condition", "phase"],
            as_index=False,
        )["gppi_beta"]
        .mean()
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for keys, sub_frame in subj_phase.groupby(["seed_roi", "target_roi", "target_roi_set", "target_scope", "ppi_condition"], sort=False):
        seed_roi, target_roi, target_set, target_scope, ppi_condition = keys
        pivot = sub_frame.pivot(index="subject", columns="phase", values="gppi_beta").dropna()
        if not {"pre", "post"}.issubset(pivot.columns) or pivot.empty:
            continue
        diff = (pivot["post"] - pivot["pre"]).to_numpy(dtype=float)
        n, mean_diff, _, _, t_val, p_parametric, dz = _paired_t_from_diff(diff)
        p_permutation = _sign_flip_permutation_p(diff, n_permutations=n_permutations, rng=rng)
        rows.append(
            {
                "seed": seed_roi,
                "seed_roi": seed_roi,
                "target_roi": target_roi,
                "target_roi_set": target_set,
                "target_scope": target_scope,
                "ppi_condition": ppi_condition,
                "test_type": "post_vs_pre",
                "n": n,
                "mean_diff": mean_diff,
                "t": t_val,
                "p": p_permutation,
                "p_permutation": p_permutation,
                "p_parametric": p_parametric,
                "cohens_dz": dz,
            }
        )
    target = pd.DataFrame(rows)
    if target.empty:
        return target

    target["q_within_seed_scope_condition"] = np.nan
    for _, idx in target.groupby(["seed_roi", "target_scope", "ppi_condition"], dropna=False).groups.items():
        target.loc[idx, "q_within_seed_scope_condition"] = _benjamini_hochberg(target.loc[idx, "p_permutation"])

    target["significant_fdr_within_scope"] = target["q_within_seed_scope_condition"].lt(0.05)
    return target.sort_values(
        ["seed", "target_scope", "ppi_condition", "q_within_seed_scope_condition", "p"],
        na_position="last",
    ).reset_index(drop=True)


def _build_scope_level_permutation_table(
    metrics: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    work = metrics[metrics["phase"].astype(str).isin(["pre", "post"])].copy()
    work = work[work["target_scope"].astype(str).isin(["within_literature", "within_literature_spatial"])].copy()
    if work.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    subj_scope = (
        work.groupby(["subject", "seed_roi", "target_scope", "ppi_condition", "phase"], as_index=False)["gppi_beta"]
        .mean()
    )
    for (seed_roi, target_scope, ppi_condition), sub_frame in subj_scope.groupby(["seed_roi", "target_scope", "ppi_condition"], sort=False):
        pivot = sub_frame.pivot(index="subject", columns="phase", values="gppi_beta").dropna()
        if not {"pre", "post"}.issubset(pivot.columns) or pivot.empty:
            continue
        diff = (pivot["post"] - pivot["pre"]).to_numpy(dtype=float)
        n, mean_diff, _, _, t_val, p_parametric, dz = _paired_t_from_diff(diff)
        p_permutation = _sign_flip_permutation_p(diff, n_permutations=n_permutations, rng=rng)
        rows.append(
            {
                "seed": seed_roi,
                "target_roi": "__scope_mean__",
                "target_scope": target_scope,
                "ppi_condition": ppi_condition,
                "test_type": "post_vs_pre",
                "n": n,
                "mean_diff": mean_diff,
                "t_permutation_source": t_val,
                "p_permutation": p_permutation,
                "p_parametric_scope_recomputed": p_parametric,
                "cohens_dz_recomputed": dz,
            }
        )

    wide = subj_scope.pivot_table(
        index=["subject", "seed_roi", "ppi_condition", "phase"],
        columns="target_scope",
        values="gppi_beta",
        aggfunc="mean",
    ).reset_index()
    if not wide.empty and {"within_literature", "within_literature_spatial"}.issubset(wide.columns):
        for (seed_roi, ppi_condition), sub_frame in wide.groupby(["seed_roi", "ppi_condition"], sort=False):
            pre = sub_frame[sub_frame["phase"] == "pre"].set_index("subject")
            post = sub_frame[sub_frame["phase"] == "post"].set_index("subject")
            common = sorted(set(pre.index) & set(post.index))
            if not common:
                continue
            pre_delta = (pre.loc[common, "within_literature"] - pre.loc[common, "within_literature_spatial"]).to_numpy(dtype=float)
            post_delta = (post.loc[common, "within_literature"] - post.loc[common, "within_literature_spatial"]).to_numpy(dtype=float)
            diff = post_delta - pre_delta
            n, mean_diff, _, _, t_val, p_parametric, dz = _paired_t_from_diff(diff)
            p_permutation = _sign_flip_permutation_p(diff, n_permutations=n_permutations, rng=rng)
            rows.append(
                {
                    "seed": seed_roi,
                    "target_roi": "__cross_sets__",
                    "target_scope": "cross_sets",
                    "ppi_condition": ppi_condition,
                    "test_type": "post_vs_pre",
                    "n": n,
                    "mean_diff": mean_diff,
                    "t_permutation_source": t_val,
                    "p_permutation": p_permutation,
                    "p_parametric_scope_recomputed": p_parametric,
                    "cohens_dz_recomputed": dz,
                }
            )
    return pd.DataFrame(rows)


def _build_target_peak_table(target: pd.DataFrame) -> pd.DataFrame:
    if target.empty:
        return target
    rows = []
    for _, sub_frame in target.groupby(["seed", "target_scope", "ppi_condition"], sort=False):
        ranked = sub_frame.sort_values(
            ["q_within_seed_scope_condition", "p"],
            na_position="last",
        )
        if not ranked.empty:
            rows.append(ranked.iloc[0].to_dict())
    if not rows:
        return pd.DataFrame()
    peaks = pd.DataFrame(rows)
    return peaks.sort_values(
        ["ppi_condition", "seed", "target_scope", "q_within_seed_scope_condition", "p"],
        na_position="last",
    ).reset_index(drop=True)


def _merge_target_corrections(full: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    if full.empty or target.empty:
        return full
    corrections = target[
        [
            "seed",
            "target_roi",
            "target_scope",
            "ppi_condition",
            "test_type",
            "p_permutation",
            "p_parametric",
            "q_within_seed_scope_condition",
            "significant_fdr_within_scope",
        ]
    ].copy()
    merged = full.merge(
        corrections,
        how="left",
        on=["seed", "target_roi", "target_scope", "ppi_condition", "test_type"],
    )
    return merged


def _merge_scope_corrections(full: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    if full.empty or scope.empty:
        return full
    corrections = scope[
        [
            "seed",
            "target_roi",
            "target_scope",
            "ppi_condition",
            "test_type",
            "p_permutation",
            "p_parametric_scope_recomputed",
        ]
    ].copy()
    corrections = corrections.rename(columns={"p_permutation": "p_permutation_scope"})
    return full.merge(
        corrections,
        how="left",
        on=["seed", "target_roi", "target_scope", "ppi_condition", "test_type"],
    )


def _build_main_table(full: pd.DataFrame) -> pd.DataFrame:
    if full.empty:
        return pd.DataFrame()
    scope_targets = {"__scope_mean__", "__cross_sets__"}
    scope_order = {
        "within_literature": 0,
        "within_literature_spatial": 1,
        "cross_sets": 2,
    }
    main = full.copy()
    main = main[main["test_type"].astype(str).eq("post_vs_pre")]
    main = main[main["ppi_condition"].astype(str).eq("yy_minus_kj")]
    main = main[main["target_scope"].astype(str).isin(scope_order)]
    main = main[main["target_roi"].astype(str).isin(scope_targets)]
    if main.empty:
        return main
    main = main.assign(
        significant_p_uncorrected=main["p"].lt(0.05),
        scope_rank=main["target_scope"].map(scope_order).fillna(999).astype(int),
    )
    if "p_permutation" in main.columns:
        main["q_within_main_tests"] = _benjamini_hochberg(main["p_permutation"])
        main["significant_fdr_main_tests"] = main["q_within_main_tests"].lt(0.05)
    if "p_permutation_scope" in main.columns:
        main["q_within_main_tests"] = _benjamini_hochberg(main["p_permutation_scope"])
        main["significant_fdr_main_tests"] = main["q_within_main_tests"].lt(0.05)
    ordered = [
        "seed",
        "roi_set",
        "deconvolution",
        "target_scope",
        "ppi_condition",
        "test_type",
        "n",
        "mean_post",
        "mean_pre",
        "mean_diff",
        "t",
        "p",
        "p_permutation_scope",
        "q_within_main_tests",
        "cohens_dz",
        "significant_p_uncorrected",
        "significant_fdr_main_tests",
        "source_table",
    ]
    present = [col for col in ordered if col in main.columns]
    main = main.sort_values(["seed", "scope_rank", "p"], ascending=[True, True, True]).reset_index(drop=True)
    return main[present]


def _build_target_main_table(target_peaks: pd.DataFrame) -> pd.DataFrame:
    if target_peaks.empty:
        return target_peaks
    main = target_peaks[target_peaks["ppi_condition"].astype(str).eq("yy_minus_kj")].copy()
    ordered = [
        "seed",
        "target_scope",
        "ppi_condition",
        "target_roi",
        "target_roi_set",
        "n",
        "mean_diff",
        "t",
        "p",
        "p_permutation",
        "p_parametric",
        "q_within_seed_scope_condition",
        "cohens_dz",
        "significant_fdr_within_scope",
    ]
    present = [col for col in ordered if col in main.columns]
    return main[present].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export consolidated paper tables from finished gPPI outputs.")
    parser.add_argument("paper_output_root", nargs="?", type=Path, default=None)
    parser.add_argument("--n-target-permutations", type=int, default=5000, help="Sign-flip permutations for target-level post-pre tests.")
    parser.add_argument("--target-permutation-seed", type=int, default=42)
    args = parser.parse_args()

    paper_root = ensure_dir(args.paper_output_root or _default_paper_root())
    tables_main = ensure_dir(paper_root / "tables_main")
    tables_si = ensure_dir(paper_root / "tables_si")
    qc_root = ensure_dir(paper_root / "qc")

    source_tables = _candidate_tables(tables_si)
    subject_metric_tables = _candidate_subject_metric_tables(qc_root)
    full_df = _read_seed_tables(source_tables)
    subject_metrics = _read_subject_metric_tables(subject_metric_tables)
    target_df = _build_target_level_table(
        subject_metrics,
        n_permutations=args.n_target_permutations,
        seed=args.target_permutation_seed,
    )
    scope_df = _build_scope_level_permutation_table(
        subject_metrics,
        n_permutations=args.n_target_permutations,
        seed=args.target_permutation_seed + 10000,
    )
    target_peaks = _build_target_peak_table(target_df)
    full_df = _merge_target_corrections(full_df, target_df)
    full_df = _merge_scope_corrections(full_df, scope_df)
    main_df = _build_main_table(full_df)
    target_main_df = _build_target_main_table(target_peaks)

    write_table(full_df, tables_si / "table_gppi_summary_full.tsv")
    write_table(target_df, tables_si / "table_gppi_target_level.tsv")
    write_table(target_peaks, tables_si / "table_gppi_target_peaks_full.tsv")
    write_table(scope_df, tables_si / "table_gppi_scope_permutation.tsv")
    write_table(main_df, tables_main / "table_gppi_summary_main.tsv")
    write_table(target_main_df, tables_main / "table_gppi_target_peaks.tsv")
    save_json(
        {
            "paper_output_root": str(paper_root),
            "n_source_tables": int(len(source_tables)),
            "source_tables": [str(path) for path in source_tables],
            "n_subject_metric_tables": int(len(subject_metric_tables)),
            "subject_metric_tables": [str(path) for path in subject_metric_tables],
            "target_p_value": "sign_flip_permutation",
            "target_fdr": "BH within seed_roi x target_scope x ppi_condition",
            "n_target_permutations": int(args.n_target_permutations),
            "target_permutation_seed": int(args.target_permutation_seed),
            "n_full_rows": int(len(full_df)),
            "n_main_rows": int(len(main_df)),
            "n_target_rows": int(len(target_df)),
            "n_target_peak_rows": int(len(target_peaks)),
            "n_scope_rows": int(len(scope_df)),
            "full_table": str(tables_si / "table_gppi_summary_full.tsv"),
            "main_table": str(tables_main / "table_gppi_summary_main.tsv"),
            "target_table": str(tables_si / "table_gppi_target_level.tsv"),
            "target_peaks_full": str(tables_si / "table_gppi_target_peaks_full.tsv"),
            "target_peaks_main": str(tables_main / "table_gppi_target_peaks.tsv"),
            "scope_permutation_table": str(tables_si / "table_gppi_scope_permutation.tsv"),
        },
        qc_root / "gppi_summary_export_manifest.json",
    )


if __name__ == "__main__":
    main()
