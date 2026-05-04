#!/usr/bin/env python3
"""Network-level dissociation for relation-vector decoupling."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Iterable
import warnings

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

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402


PRIMARY_MODELS = {"M9_relation_vector_direct", "M9_relation_vector_abs"}
NETWORK_LABELS = {
    "semantic_metaphor": "Semantic/metaphor",
    "spatial_context": "Spatial/context",
    "hippocampus_core": "Hippocampus",
    "main_metaphor_functional": "Main Metaphor>Spatial",
    "main_spatial_functional": "Main Spatial>Metaphor",
}
PLANNED_NETWORK_CONTRASTS = [
    ("semantic_minus_spatial", "semantic_metaphor", "spatial_context"),
    ("semantic_minus_hippocampus", "semantic_metaphor", "hippocampus_core"),
    ("hippocampus_minus_spatial", "hippocampus_core", "spatial_context"),
    ("main_metaphor_minus_main_spatial", "main_metaphor_functional", "main_spatial_functional"),
    ("semantic_minus_main_spatial", "semantic_metaphor", "main_spatial_functional"),
]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _cohens_dz(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if math.isclose(sd, 0.0) or not math.isfinite(sd):
        return float("nan")
    return float(arr.mean() / sd)


def _bootstrap_ci(values: Iterable[float], *, seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        means[idx] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _one_sample_summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "n_subjects": int(arr.size),
            "mean_effect": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "cohens_dz": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    t_val, p_val = stats.ttest_1samp(arr, popmean=0.0, nan_policy="omit")
    ci_low, ci_high = _bootstrap_ci(arr)
    return {
        "n_subjects": int(arr.size),
        "mean_effect": float(arr.mean()),
        "t": float(t_val) if np.isfinite(t_val) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "cohens_dz": _cohens_dz(arr),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def infer_network(roi_set: str, roi: str) -> str:
    roi_lower = roi.lower()
    if roi_set == "meta_metaphor":
        return "semantic_metaphor"
    if roi_set == "meta_spatial":
        if "hippocampus" in roi_lower:
            return "hippocampus_core"
        return "spatial_context"
    if roi_set == "literature":
        if "precuneus" in roi_lower:
            return "spatial_context"
        return "semantic_metaphor"
    if roi_set == "literature_spatial":
        if "hippocampus" in roi_lower:
            return "hippocampus_core"
        return "spatial_context"
    if roi_set == "main_functional":
        if "metaphor_gt_spatial" in roi_lower:
            return "main_metaphor_functional"
        if "spatial_gt_metaphor" in roi_lower:
            return "main_spatial_functional"
    return "unassigned"


def load_network_long(subject_path: Path) -> pd.DataFrame:
    frame = _read_table(subject_path).copy()
    frame = frame[frame["neural_rdm_type"].eq("relation_vector") & frame["model"].isin(PRIMARY_MODELS)].copy()
    long = frame.melt(
        id_vars=["subject", "roi_set", "roi", "neural_rdm_type", "model", "model_role", "is_primary_model"],
        value_vars=["decoupling_yy", "decoupling_kj"],
        var_name="condition",
        value_name="relation_decoupling_pre_minus_post",
    )
    long["condition"] = long["condition"].str.replace("decoupling_", "", regex=False)
    long["network"] = [infer_network(roi_set, roi) for roi_set, roi in zip(long["roi_set"], long["roi"])]
    long = long[long["network"].ne("unassigned")].copy()
    long["network_label"] = long["network"].map(NETWORK_LABELS).fillna(long["network"])
    long["condition_code"] = np.where(long["condition"].eq("yy"), 0.5, -0.5)
    return long.reset_index(drop=True)


def build_network_subject(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["subject", "network", "network_label", "condition", "model"], as_index=False)
        .agg(
            relation_decoupling_pre_minus_post=("relation_decoupling_pre_minus_post", "mean"),
            n_rois=("roi", "nunique"),
            roi_names=("roi", lambda values: "|".join(sorted(set(map(str, values))))),
        )
        .reset_index(drop=True)
    )


def summarize_network_condition_contrasts(network_subject: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = (
        network_subject.pivot_table(
            index=["subject", "network", "network_label", "model", "n_rois", "roi_names"],
            columns="condition",
            values="relation_decoupling_pre_minus_post",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "yy" not in pivot.columns or "kj" not in pivot.columns:
        raise ValueError("Need YY and KJ network rows.")
    pivot["yy_minus_kj_decoupling"] = pivot["yy"] - pivot["kj"]
    rows: list[dict[str, object]] = []
    for keys, subset in pivot.groupby(["network", "network_label", "model"], sort=False):
        network, network_label, model = keys
        summary = _one_sample_summary(subset["yy_minus_kj_decoupling"])
        rows.append(
            {
                "analysis_type": "network_condition_contrast",
                "contrast": "yy_minus_kj_decoupling",
                "network": network,
                "network_label": network_label,
                "model": model,
                "n_rois": int(subset["n_rois"].max()),
                **summary,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["q_bh_within_model"] = np.nan
        for _, idx in summary.groupby("model", dropna=False).groups.items():
            idx = list(idx)
            summary.loc[idx, "q_bh_within_model"] = _bh_fdr(summary.loc[idx, "p"])
    return pivot, summary


def summarize_planned_network_contrasts(network_contrast_subject: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_subject: list[dict[str, object]] = []
    rows_summary: list[dict[str, object]] = []
    for model, model_df in network_contrast_subject.groupby("model", sort=False):
        wide = model_df.pivot_table(index="subject", columns="network", values="yy_minus_kj_decoupling", aggfunc="mean")
        for contrast_name, left, right in PLANNED_NETWORK_CONTRASTS:
            if left not in wide.columns or right not in wide.columns:
                continue
            effect = wide[left] - wide[right]
            for subject, value in effect.items():
                rows_subject.append(
                    {
                        "subject": subject,
                        "model": model,
                        "contrast": contrast_name,
                        "left_network": left,
                        "right_network": right,
                        "network_difference_of_condition_contrast": value,
                    }
                )
            summary = _one_sample_summary(effect)
            rows_summary.append(
                {
                    "analysis_type": "network_difference_of_condition_contrast",
                    "contrast": contrast_name,
                    "left_network": left,
                    "right_network": right,
                    "model": model,
                    **summary,
                }
            )
    subject = pd.DataFrame(rows_subject)
    summary = pd.DataFrame(rows_summary)
    if not summary.empty:
        summary["q_bh_within_model"] = np.nan
        for _, idx in summary.groupby("model", dropna=False).groups.items():
            idx = list(idx)
            summary.loc[idx, "q_bh_within_model"] = _bh_fdr(summary.loc[idx, "p"])
    return subject, summary


def fit_lmm_params(long: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    rows: list[dict[str, object]] = []
    data = long.copy()
    network_order = [
        "spatial_context",
        "semantic_metaphor",
        "hippocampus_core",
        "main_metaphor_functional",
        "main_spatial_functional",
    ]
    data["network"] = pd.Categorical(data["network"], categories=network_order)
    formula = "relation_decoupling_pre_minus_post ~ condition_code * C(network, Treatment(reference='spatial_context'))"

    def append_result(model_name: str, subset: pd.DataFrame, result, method: str, converged: bool) -> None:
        for term in result.params.index:
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "estimate": float(result.params[term]),
                    "se": float(result.bse.get(term, np.nan)),
                    "z": float(result.tvalues.get(term, np.nan)),
                    "p": float(result.pvalues.get(term, np.nan)),
                    "converged": bool(converged),
                    "fit_method": method,
                    "n_rows": int(len(subset)),
                }
            )

    for model, subset in data.groupby("model", sort=False):
        subset = subset.dropna(subset=["relation_decoupling_pre_minus_post", "network", "condition_code"]).copy()
        errors: list[str] = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.mixedlm(
                    formula,
                    subset,
                    groups=subset["subject"],
                    vc_formula={"roi": "0 + C(roi)"},
                ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            append_result(model, subset, result, "mixedlm_subject_plus_roi_vc", bool(result.converged))
            continue
        except Exception as exc:  # pragma: no cover - only used for robust batch execution
            errors.append(f"mixedlm_subject_plus_roi_vc: {exc}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.mixedlm(formula, subset, groups=subset["subject"]).fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=500,
                    disp=False,
                )
            append_result(model, subset, result, "mixedlm_subject_random_intercept", bool(result.converged))
            continue
        except Exception as exc:  # pragma: no cover - only used for robust batch execution
            errors.append(f"mixedlm_subject_random_intercept: {exc}")
        try:
            result = smf.ols(formula, subset).fit(cov_type="cluster", cov_kwds={"groups": subset["subject"]})
            append_result(model, subset, result, "ols_subject_clustered_se", True)
        except Exception as exc:  # pragma: no cover - only used for robust batch execution
            errors.append(f"ols_subject_clustered_se: {exc}")
            rows.append(
                {
                    "model": model,
                    "term": "MODEL_FAILED",
                    "estimate": float("nan"),
                    "se": float("nan"),
                    "z": float("nan"),
                    "p": float("nan"),
                    "converged": False,
                    "fit_method": "all_failed",
                    "n_rows": int(len(subset)),
                    "error": " | ".join(errors),
                }
            )
    params = pd.DataFrame(rows)
    if not params.empty:
        params["q_bh_fixed_effects_within_model"] = np.nan
        testable = params["term"].ne("Intercept") & params["term"].ne("MODEL_FAILED")
        for _, idx in params[testable].groupby("model", dropna=False).groups.items():
            idx = list(idx)
            params.loc[idx, "q_bh_fixed_effects_within_model"] = _bh_fdr(params.loc[idx, "p"])
    return params


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Network-level relation-vector dissociation.")
    parser.add_argument(
        "--relation-subject-metrics",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "relation_vector_contrast" / "relation_vector_condition_contrast_subject.tsv",
    )
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir or args.paper_output_root / "qc" / "relation_vector_network")
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")

    long = load_network_long(args.relation_subject_metrics)
    network_subject = build_network_subject(long)
    network_contrast_subject, network_summary = summarize_network_condition_contrasts(network_subject)
    planned_subject, planned_summary = summarize_planned_network_contrasts(network_contrast_subject)
    lmm_params = fit_lmm_params(long)
    main_table = pd.concat([network_summary, planned_summary], ignore_index=True, sort=False)
    if not main_table.empty:
        main_table = main_table.sort_values(["q_bh_within_model", "p", "model"], na_position="last").reset_index(drop=True)

    write_table(long, output_dir / "relation_vector_network_long.tsv")
    write_table(network_subject, output_dir / "relation_vector_network_subject.tsv")
    write_table(network_contrast_subject, output_dir / "relation_vector_network_condition_contrast_subject.tsv")
    write_table(network_summary, output_dir / "relation_vector_network_condition_contrast_group.tsv")
    write_table(planned_subject, output_dir / "relation_vector_network_planned_contrast_subject.tsv")
    write_table(planned_summary, output_dir / "relation_vector_network_planned_contrast_group.tsv")
    write_table(lmm_params, output_dir / "relation_vector_network_lmm_params.tsv")
    write_table(main_table, tables_main / "table_relation_vector_network_dissociation.tsv")
    write_table(main_table, tables_si / "table_relation_vector_network_dissociation_full.tsv")
    save_json(
        {
            "relation_subject_metrics": str(args.relation_subject_metrics),
            "output_dir": str(output_dir),
            "n_long_rows": int(len(long)),
            "n_network_subject_rows": int(len(network_subject)),
            "n_main_rows": int(len(main_table)),
            "network_labels": NETWORK_LABELS,
            "condition_contrast": "yy_minus_kj_decoupling = pre-post YY minus pre-post KJ",
        },
        output_dir / "relation_vector_network_manifest.json",
    )
    print(f"[relation-vector-network] wrote {len(long)} long rows and {len(main_table)} main rows to {output_dir}")


if __name__ == "__main__":
    main()
