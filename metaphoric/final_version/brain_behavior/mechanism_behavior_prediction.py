#!/usr/bin/env python3
"""
mechanism_behavior_prediction.py

Purpose
- Implement S1: subject-level mechanism -> behavior / memory prediction.
- Reuse Step 5D model-RSA outputs (`model_rdm_subject_metrics.tsv`) to build
  subject-level mechanism scores, then test whether stronger mechanism signals
  predict better run-7 behavior.
- Keep the logic explicitly different from MVPA:
  MVPA uses trial-level condition decoding, while this script uses subject-level
  mechanism scores to explain individual differences in learning outcomes.

Primary outputs
- `paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv`
- `paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv`
- `paper_outputs/figures_main/fig_mechanism_behavior_prediction.png`

Supporting outputs
- `paper_outputs/qc/mechanism_behavior_prediction_long.tsv`
- `paper_outputs/qc/mechanism_behavior_prediction_manifest.json`
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from behavior_analysis.behavior_lmm import normalize_subject_id  # noqa: E402


DEFAULT_MODELS = [
    "M8_reverse_pair",
    "M3_embedding",
]
PRIMARY_MODELS = {"M8_reverse_pair", "M3_embedding"}
PRIMARY_BEHAVIOR_METRICS = {
    "run7_memory_accuracy",
    "run7_correct_retrieval_efficiency",
}
CONDITION_ORDER = ["yy", "kj"]
CONDITION_NAME_MAP = {
    "yy": "Metaphor",
    "metaphor": "Metaphor",
    "kj": "Spatial",
    "spatial": "Spatial",
}
MODEL_SIGN_MAP = {
    # Larger aligned score = stronger hypothesized post-learning mechanism.
    "M8_reverse_pair": 1.0,
    "M3_embedding": -1.0,
    # Secondary / sensitivity: still align larger values with stronger post-learning change.
    "M7_continuous_confidence": -1.0,
}
BEHAVIOR_LABELS = {
    "run7_memory_accuracy": "Run-7 memory accuracy",
    "run7_correct_retrieval_efficiency": "Run-7 retrieval efficiency",
    "run7_correct_retrieval_speed": "Run-7 correct-trial speed",
}


def _default_base_dir() -> Path:
    override = os.environ.get("PYTHON_METAPHOR_ROOT", "").strip()
    if override:
        return Path(override)
    return FINAL_ROOT.parent


BASE_DIR = _default_base_dir()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _canonical_condition(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"yy", "metaphor"}:
        return "yy"
    if text in {"kj", "spatial"}:
        return "kj"
    return None


def _condition_display(value: str) -> str:
    return CONDITION_NAME_MAP.get(str(value).strip().lower(), str(value))


def _infer_roi_set(path: Path) -> str:
    parent = path.parent.name.strip()
    prefix = "model_rdm_results_"
    if parent.startswith(prefix) and len(parent) > len(prefix):
        return parent[len(prefix):]
    return "unknown"


def _resolve_default_model_metric_paths(base_dir: Path) -> list[Path]:
    qc_root = base_dir / "paper_outputs" / "qc"
    candidates = [
        qc_root / "model_rdm_results_main_functional" / "model_rdm_subject_metrics.tsv",
        qc_root / "model_rdm_results_literature" / "model_rdm_subject_metrics.tsv",
        qc_root / "model_rdm_results_literature_spatial" / "model_rdm_subject_metrics.tsv",
    ]
    return [path for path in candidates if path.exists()]


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if mask.sum() < 2:
        return out
    subset = arr[mask]
    std = float(subset.std(ddof=1))
    if math.isclose(std, 0.0) or np.isnan(std):
        out[mask] = 0.0
        return out
    mean = float(subset.mean())
    out[mask] = (subset - mean) / std
    return out


def _bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    low = np.quantile(arr, alpha / 2.0)
    high = np.quantile(arr, 1.0 - alpha / 2.0)
    return float(low), float(high)


def _standardized_beta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xz = _zscore(x[mask])
    yz = _zscore(y[mask])
    if np.isnan(xz).all() or np.isnan(yz).all():
        return float("nan")
    design = np.column_stack([np.ones(mask.sum()), xz])
    beta, *_ = np.linalg.lstsq(design, yz, rcond=None)
    return float(beta[1])


def _spearman_effect(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 5:
        return {
            "n_subjects": n,
            "beta_std": float("nan"),
            "beta_ci_low": float("nan"),
            "beta_ci_high": float("nan"),
            "spearman_r": float("nan"),
            "spearman_ci_low": float("nan"),
            "spearman_ci_high": float("nan"),
            "permutation_p": float("nan"),
            "null_mean": float("nan"),
            "null_sd": float("nan"),
        }
    beta_std = _standardized_beta(x, y)
    spearman_r, _ = stats.spearmanr(x, y)
    spearman_r = float(spearman_r)

    boot_r: list[float] = []
    boot_beta: list[float] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample_x = x[sample_idx]
        sample_y = y[sample_idx]
        r_boot, _ = stats.spearmanr(sample_x, sample_y)
        boot_r.append(float(r_boot) if np.isfinite(r_boot) else float("nan"))
        boot_beta.append(_standardized_beta(sample_x, sample_y))
    r_low, r_high = _bootstrap_ci(boot_r)
    beta_low, beta_high = _bootstrap_ci(boot_beta)

    null_values = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        perm_y = rng.permutation(y)
        perm_r, _ = stats.spearmanr(x, perm_y)
        null_values[index] = float(perm_r) if np.isfinite(perm_r) else 0.0
    permutation_p = float((np.sum(np.abs(null_values) >= abs(spearman_r)) + 1.0) / (n_permutations + 1.0))

    return {
        "n_subjects": n,
        "beta_std": beta_std,
        "beta_ci_low": beta_low,
        "beta_ci_high": beta_high,
        "spearman_r": spearman_r,
        "spearman_ci_low": r_low,
        "spearman_ci_high": r_high,
        "permutation_p": permutation_p,
        "null_mean": float(np.mean(null_values)),
        "null_sd": float(np.std(null_values, ddof=1)) if n_permutations > 1 else float("nan"),
    }


def _interaction_beta(frame: pd.DataFrame) -> float:
    work = frame.copy()
    work["condition_bin"] = work["condition_group"].map({"yy": 0.0, "kj": 1.0})
    work["x_z"] = np.nan
    work["y_z"] = np.nan
    for condition, idx in work.groupby("condition_group").groups.items():
        work.loc[idx, "x_z"] = _zscore(work.loc[idx, "mechanism_score"].to_numpy(dtype=float))
        work.loc[idx, "y_z"] = _zscore(work.loc[idx, "behavior_value"].to_numpy(dtype=float))
    valid = work[["x_z", "y_z", "condition_bin"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 8 or valid["condition_bin"].nunique() < 2:
        return float("nan")
    x = valid["x_z"].to_numpy(dtype=float)
    y = valid["y_z"].to_numpy(dtype=float)
    cond = valid["condition_bin"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(valid)), x, cond, x * cond])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[3])


def _interaction_effect(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    required = {"subject", "condition_group", "mechanism_score", "behavior_value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Interaction frame missing columns: {sorted(missing)}")

    pivot = frame.pivot_table(
        index="subject",
        columns="condition_group",
        values=["mechanism_score", "behavior_value"],
        aggfunc="mean",
    )
    required_cols = [
        ("mechanism_score", "yy"),
        ("mechanism_score", "kj"),
        ("behavior_value", "yy"),
        ("behavior_value", "kj"),
    ]
    for col in required_cols:
        if col not in pivot.columns:
            return {
                "n_subjects": int(frame["subject"].nunique()),
                "interaction_beta": float("nan"),
                "interaction_ci_low": float("nan"),
                "interaction_ci_high": float("nan"),
                "r_yy": float("nan"),
                "r_kj": float("nan"),
                "r_diff_yy_minus_kj": float("nan"),
                "permutation_p": float("nan"),
                "null_mean": float("nan"),
                "null_sd": float("nan"),
            }
    pivot = pivot.dropna(subset=required_cols).copy()
    n_subjects = int(len(pivot))
    if n_subjects < 5:
        return {
            "n_subjects": n_subjects,
            "interaction_beta": float("nan"),
            "interaction_ci_low": float("nan"),
            "interaction_ci_high": float("nan"),
            "r_yy": float("nan"),
            "r_kj": float("nan"),
            "r_diff_yy_minus_kj": float("nan"),
            "permutation_p": float("nan"),
            "null_mean": float("nan"),
            "null_sd": float("nan"),
        }

    long_rows: list[dict[str, object]] = []
    for subject, row in pivot.iterrows():
        long_rows.append(
            {
                "subject": str(subject),
                "condition_group": "yy",
                "mechanism_score": float(row[("mechanism_score", "yy")]),
                "behavior_value": float(row[("behavior_value", "yy")]),
            }
        )
        long_rows.append(
            {
                "subject": str(subject),
                "condition_group": "kj",
                "mechanism_score": float(row[("mechanism_score", "kj")]),
                "behavior_value": float(row[("behavior_value", "kj")]),
            }
        )
    long_frame = pd.DataFrame(long_rows)
    observed_beta = _interaction_beta(long_frame)
    r_yy, _ = stats.spearmanr(
        pivot[("mechanism_score", "yy")].to_numpy(dtype=float),
        pivot[("behavior_value", "yy")].to_numpy(dtype=float),
    )
    r_kj, _ = stats.spearmanr(
        pivot[("mechanism_score", "kj")].to_numpy(dtype=float),
        pivot[("behavior_value", "kj")].to_numpy(dtype=float),
    )
    r_yy = float(r_yy) if np.isfinite(r_yy) else float("nan")
    r_kj = float(r_kj) if np.isfinite(r_kj) else float("nan")

    boot_beta: list[float] = []
    subject_ids = pivot.index.to_numpy()
    for _ in range(n_bootstrap):
        sample_ids = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        sample_pivot = pivot.loc[sample_ids]
        sample_rows: list[dict[str, object]] = []
        for subject, row in sample_pivot.iterrows():
            sample_rows.append(
                {
                    "subject": str(subject),
                    "condition_group": "yy",
                    "mechanism_score": float(row[("mechanism_score", "yy")]),
                    "behavior_value": float(row[("behavior_value", "yy")]),
                }
            )
            sample_rows.append(
                {
                    "subject": str(subject),
                    "condition_group": "kj",
                    "mechanism_score": float(row[("mechanism_score", "kj")]),
                    "behavior_value": float(row[("behavior_value", "kj")]),
                }
            )
        boot_beta.append(_interaction_beta(pd.DataFrame(sample_rows)))
    beta_low, beta_high = _bootstrap_ci(boot_beta)

    null_values = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        perm_rows: list[dict[str, object]] = []
        swap_mask = rng.random(len(pivot)) < 0.5
        for swap, (subject, row) in zip(swap_mask, pivot.iterrows()):
            if swap:
                yy_mech = float(row[("mechanism_score", "kj")])
                yy_beh = float(row[("behavior_value", "kj")])
                kj_mech = float(row[("mechanism_score", "yy")])
                kj_beh = float(row[("behavior_value", "yy")])
            else:
                yy_mech = float(row[("mechanism_score", "yy")])
                yy_beh = float(row[("behavior_value", "yy")])
                kj_mech = float(row[("mechanism_score", "kj")])
                kj_beh = float(row[("behavior_value", "kj")])
            perm_rows.append(
                {
                    "subject": str(subject),
                    "condition_group": "yy",
                    "mechanism_score": yy_mech,
                    "behavior_value": yy_beh,
                }
            )
            perm_rows.append(
                {
                    "subject": str(subject),
                    "condition_group": "kj",
                    "mechanism_score": kj_mech,
                    "behavior_value": kj_beh,
                }
            )
        null_values[index] = _interaction_beta(pd.DataFrame(perm_rows))
    permutation_p = float((np.sum(np.abs(null_values) >= abs(observed_beta)) + 1.0) / (n_permutations + 1.0))

    return {
        "n_subjects": n_subjects,
        "interaction_beta": observed_beta,
        "interaction_ci_low": beta_low,
        "interaction_ci_high": beta_high,
        "r_yy": r_yy,
        "r_kj": r_kj,
        "r_diff_yy_minus_kj": float(r_yy - r_kj) if np.isfinite(r_yy) and np.isfinite(r_kj) else float("nan"),
        "permutation_p": permutation_p,
        "null_mean": float(np.mean(null_values)),
        "null_sd": float(np.std(null_values, ddof=1)) if n_permutations > 1 else float("nan"),
    }


def _prepare_mechanism_scores(
    metrics_path: Path,
    *,
    enabled_models: list[str],
) -> pd.DataFrame:
    frame = _read_table(metrics_path).copy()
    required = {"subject", "roi", "time", "model", "rho"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{metrics_path} is missing required columns: {sorted(missing)}")
    if "condition_group" not in frame.columns:
        raise ValueError(f"{metrics_path} must contain `condition_group` for S1.")

    frame["subject"] = frame["subject"].map(normalize_subject_id)
    frame["model"] = frame["model"].astype(str).str.strip()
    frame["condition_group"] = frame["condition_group"].map(_canonical_condition)
    frame["time"] = frame["time"].astype(str).str.strip().str.lower()
    frame["rho"] = pd.to_numeric(frame["rho"], errors="coerce")
    if "skip_reason" in frame.columns:
        frame = frame[frame["skip_reason"].fillna("").eq("")].copy()
    frame = frame[
        frame["model"].isin(enabled_models)
        & frame["condition_group"].isin(CONDITION_ORDER)
        & frame["time"].isin(["pre", "post"])
    ].copy()

    pivot = (
        frame.pivot_table(
            index=["subject", "roi", "condition_group", "model"],
            columns="time",
            values="rho",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "pre" not in pivot.columns or "post" not in pivot.columns:
        raise ValueError(f"{metrics_path} does not contain matched pre/post rows.")

    roi_set = _infer_roi_set(metrics_path)
    pivot["delta_rho"] = pivot["post"] - pivot["pre"]
    pivot["mechanism_sign"] = pivot["model"].map(MODEL_SIGN_MAP).fillna(1.0)
    pivot["mechanism_score"] = pivot["delta_rho"] * pivot["mechanism_sign"]
    pivot["mechanism_direction_note"] = pivot["model"].map(
        {
            "M8_reverse_pair": "larger score = stronger pair differentiation because positive delta_rho is retained",
            "M3_embedding": "larger score = stronger semantic reweighting because delta_rho is sign-flipped",
            "M7_continuous_confidence": "secondary model; larger score indexes stronger post-learning directional change",
        }
    ).fillna("larger score indexes stronger aligned post-learning change")
    pivot["roi_set"] = roi_set
    return pivot[
        [
            "subject",
            "roi_set",
            "roi",
            "condition_group",
            "model",
            "pre",
            "post",
            "delta_rho",
            "mechanism_sign",
            "mechanism_score",
            "mechanism_direction_note",
        ]
    ].copy()


def _append_behavior_rows(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    metric: str,
    wide_prefix: str,
    value_transform,
    source_name: str,
) -> None:
    for raw_condition in ["YY", "KJ"]:
        column = f"{wide_prefix}_{raw_condition}"
        if column not in frame.columns:
            continue
        condition_group = _canonical_condition(raw_condition)
        if condition_group is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        for subject, raw_value in zip(frame["subject"], values):
            if not np.isfinite(raw_value):
                continue
            rows.append(
                {
                    "subject": str(subject),
                    "condition_group": condition_group,
                    "behavior_metric": metric,
                    "behavior_value_raw": float(raw_value),
                    "behavior_value": float(value_transform(float(raw_value))),
                    "behavior_source": source_name,
                }
            )


def _prepare_behavior_summary(path: Path) -> pd.DataFrame:
    frame = _read_table(path).copy()
    if "subject" not in frame.columns:
        raise ValueError(f"{path} must contain `subject`.")
    frame["subject"] = frame["subject"].map(normalize_subject_id)

    rows: list[dict[str, object]] = []
    _append_behavior_rows(
        rows,
        frame,
        metric="run7_memory_accuracy",
        wide_prefix="accuracy",
        value_transform=lambda value: value,
        source_name=path.name,
    )
    _append_behavior_rows(
        rows,
        frame,
        metric="run7_correct_retrieval_efficiency",
        wide_prefix="inverse_efficiency",
        value_transform=lambda value: -value,
        source_name=path.name,
    )
    _append_behavior_rows(
        rows,
        frame,
        metric="run7_correct_retrieval_speed",
        wide_prefix="mean_rt_correct",
        value_transform=lambda value: -value,
        source_name=path.name,
    )
    return pd.DataFrame(rows)


def _fit_condition_specific_rows(
    merged: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
    min_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "model", "behavior_metric", "condition_group"]
    for keys, subset in merged.groupby(group_cols, sort=False):
        roi_set, roi, model, behavior_metric, condition_group = keys
        pair = subset[["subject", "mechanism_score", "behavior_value", "delta_rho"]].dropna().copy()
        pair = pair.drop_duplicates(subset=["subject"], keep="first")
        effect = _spearman_effect(
            pair["mechanism_score"].to_numpy(dtype=float),
            pair["behavior_value"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            rng=rng,
        )
        if effect["n_subjects"] < min_subjects:
            continue
        predictor_name = "aligned_mechanism_score"
        summary_rows.append(
            {
                "analysis_type": "condition_specific",
                "roi_set": roi_set,
                "roi": roi,
                "model": model,
                "behavior_metric": behavior_metric,
                "condition_group": condition_group,
                "condition_label": _condition_display(condition_group),
                "predictor_name": predictor_name,
                "n_subjects": int(effect["n_subjects"]),
                "beta_std": effect["beta_std"],
                "beta_ci_low": effect["beta_ci_low"],
                "beta_ci_high": effect["beta_ci_high"],
                "spearman_r": effect["spearman_r"],
                "spearman_ci_low": effect["spearman_ci_low"],
                "spearman_ci_high": effect["spearman_ci_high"],
                "permutation_p": effect["permutation_p"],
                "mean_mechanism_score": float(pair["mechanism_score"].mean()),
                "mean_delta_rho_raw": float(pair["delta_rho"].mean()),
                "mean_behavior_value": float(pair["behavior_value"].mean()),
                "model_priority": "primary" if model in PRIMARY_MODELS else "secondary",
            }
        )
        perm_rows.append(
            {
                "analysis_type": "condition_specific",
                "roi_set": roi_set,
                "roi": roi,
                "model": model,
                "behavior_metric": behavior_metric,
                "condition_group": condition_group,
                "n_subjects": int(effect["n_subjects"]),
                "null_mean": effect["null_mean"],
                "null_sd": effect["null_sd"],
                "permutation_p": effect["permutation_p"],
                "n_bootstrap": int(n_bootstrap),
                "n_permutations": int(n_permutations),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(perm_rows)


def _fit_interaction_rows(
    merged: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
    min_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "model", "behavior_metric"]
    for keys, subset in merged.groupby(group_cols, sort=False):
        roi_set, roi, model, behavior_metric = keys
        subset = subset[subset["condition_group"].isin(CONDITION_ORDER)].copy()
        effect = _interaction_effect(
            subset,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            rng=rng,
        )
        if effect["n_subjects"] < min_subjects:
            continue
        summary_rows.append(
            {
                "analysis_type": "condition_interaction",
                "roi_set": roi_set,
                "roi": roi,
                "model": model,
                "behavior_metric": behavior_metric,
                "condition_group": "yy_vs_kj",
                "condition_label": "Metaphor vs Spatial",
                "predictor_name": "aligned_mechanism_score_x_condition",
                "n_subjects": int(effect["n_subjects"]),
                "beta_std": effect["interaction_beta"],
                "beta_ci_low": effect["interaction_ci_low"],
                "beta_ci_high": effect["interaction_ci_high"],
                "spearman_r": effect["r_diff_yy_minus_kj"],
                "spearman_ci_low": float("nan"),
                "spearman_ci_high": float("nan"),
                "permutation_p": effect["permutation_p"],
                "mean_mechanism_score": float("nan"),
                "mean_delta_rho_raw": float("nan"),
                "mean_behavior_value": float("nan"),
                "model_priority": "primary" if model in PRIMARY_MODELS else "secondary",
                "r_yy": effect["r_yy"],
                "r_kj": effect["r_kj"],
            }
        )
        perm_rows.append(
            {
                "analysis_type": "condition_interaction",
                "roi_set": roi_set,
                "roi": roi,
                "model": model,
                "behavior_metric": behavior_metric,
                "condition_group": "yy_vs_kj",
                "n_subjects": int(effect["n_subjects"]),
                "null_mean": effect["null_mean"],
                "null_sd": effect["null_sd"],
                "permutation_p": effect["permutation_p"],
                "n_bootstrap": int(n_bootstrap),
                "n_permutations": int(n_permutations),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(perm_rows)


def _plot_top_associations(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    figure_path: Path,
) -> None:
    if merged.empty or summary.empty:
        return
    plot_rows = summary[
        (summary["analysis_type"] == "condition_specific")
        & (summary["behavior_metric"].isin(PRIMARY_BEHAVIOR_METRICS))
        & (summary["model"].isin(PRIMARY_MODELS))
    ].copy()
    if plot_rows.empty:
        return
    plot_rows = plot_rows.sort_values(
        ["permutation_p", "behavior_metric", "model_priority", "roi_set", "roi"],
        na_position="last",
    ).head(6)
    n_panels = len(plot_rows)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, max(4.8, n_rows * 4.1)), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes:
        ax.set_visible(False)

    color_map = {"yy": "#b23a48", "kj": "#3574b7"}
    for index, row in enumerate(plot_rows.itertuples(index=False)):
        ax = flat_axes[index]
        ax.set_visible(True)
        mask = (
            merged["roi_set"].eq(row.roi_set)
            & merged["roi"].eq(row.roi)
            & merged["model"].eq(row.model)
            & merged["behavior_metric"].eq(row.behavior_metric)
            & merged["condition_group"].eq(row.condition_group)
        )
        sub = merged.loc[mask, ["subject", "mechanism_score", "behavior_value"]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        color = color_map.get(str(row.condition_group), "#4b5563")
        ax.scatter(sub["mechanism_score"], sub["behavior_value"], color=color, alpha=0.85, s=34)

        x = sub["mechanism_score"].to_numpy(dtype=float)
        y = sub["behavior_value"].to_numpy(dtype=float)
        if len(sub) >= 2 and np.nanstd(x, ddof=1) > 0:
            slope, intercept = np.polyfit(x, y, deg=1)
            x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(x_line, intercept + slope * x_line, color=color, linewidth=1.6)

        title = (
            f"{row.model} | {_condition_display(row.condition_group)}\n"
            f"{Path(str(row.roi)).name} | p_perm={_safe_float(row.permutation_p):.3g}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Aligned mechanism score")
        ax.set_ylabel(BEHAVIOR_LABELS.get(str(row.behavior_metric), str(row.behavior_metric)))
        ax.axhline(0.0, color="#d1d5db", linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color="#d1d5db", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 subject-level mechanism -> behavior / memory prediction.")
    parser.add_argument(
        "--model-metrics",
        nargs="*",
        type=Path,
        default=None,
        help="One or more model_rdm_subject_metrics.tsv files. If omitted, use paper_outputs/qc defaults.",
    )
    parser.add_argument(
        "--behavior-summary",
        type=Path,
        default=BASE_DIR / "paper_outputs" / "qc" / "behavior_results" / "refined" / "subject_summary_wide.tsv",
        help="Run-7 refined subject summary wide table.",
    )
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=BASE_DIR / "paper_outputs",
        help="Unified paper output root.",
    )
    parser.add_argument(
        "--include-m7",
        action="store_true",
        help="Include M7_continuous_confidence as a secondary model. Default is off.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations for CIs.")
    parser.add_argument("--n-permutations", type=int, default=5000, help="Permutation iterations.")
    parser.add_argument("--min-subjects", type=int, default=8, help="Minimum n required to report a result.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    metric_paths = args.model_metrics if args.model_metrics else _resolve_default_model_metric_paths(BASE_DIR)
    if not metric_paths:
        raise FileNotFoundError(
            "No model_rdm_subject_metrics.tsv files were found. "
            "Provide them explicitly with --model-metrics."
        )
    missing_paths = [path for path in metric_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing model metric files: {missing_paths}")
    if not args.behavior_summary.exists():
        raise FileNotFoundError(f"Missing behavior summary table: {args.behavior_summary}")

    rng = np.random.default_rng(args.seed)
    paper_root = ensure_dir(args.paper_output_root)
    tables_main = ensure_dir(paper_root / "tables_main")
    tables_si = ensure_dir(paper_root / "tables_si")
    qc_dir = ensure_dir(paper_root / "qc")
    figures_main = ensure_dir(paper_root / "figures_main")

    enabled_models = list(DEFAULT_MODELS)
    if args.include_m7:
        enabled_models.append("M7_continuous_confidence")

    mechanism_frames = [
        _prepare_mechanism_scores(path, enabled_models=enabled_models)
        for path in metric_paths
    ]
    mechanism = pd.concat(mechanism_frames, ignore_index=True)
    behavior = _prepare_behavior_summary(args.behavior_summary)
    if behavior.empty:
        raise RuntimeError("No usable behavior rows were extracted from the provided tables.")

    merged = mechanism.merge(
        behavior,
        how="inner",
        on=["subject", "condition_group"],
    )
    if merged.empty:
        raise RuntimeError("Mechanism scores and behavior metrics could not be merged on subject + condition.")
    merged["condition_label"] = merged["condition_group"].map(_condition_display)
    merged = merged.sort_values(["roi_set", "roi", "model", "behavior_metric", "condition_group", "subject"]).reset_index(drop=True)
    write_table(merged, qc_dir / "mechanism_behavior_prediction_long.tsv")

    summary_specific, perm_specific = _fit_condition_specific_rows(
        merged,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        rng=rng,
        min_subjects=args.min_subjects,
    )
    summary_interaction, perm_interaction = _fit_interaction_rows(
        merged,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        rng=rng,
        min_subjects=args.min_subjects,
    )

    summary = pd.concat([summary_specific, summary_interaction], ignore_index=True)
    perm_table = pd.concat([perm_specific, perm_interaction], ignore_index=True)
    if summary.empty:
        raise RuntimeError("No eligible S1 results met the minimum-subject requirement.")

    summary["behavior_metric_label"] = summary["behavior_metric"].map(BEHAVIOR_LABELS).fillna(summary["behavior_metric"])
    summary["condition_sort"] = summary["condition_group"].map({"yy": 0, "kj": 1, "yy_vs_kj": 2}).fillna(9)
    summary = summary.sort_values(
        ["analysis_type", "behavior_metric", "model_priority", "permutation_p", "roi_set", "condition_sort", "roi"],
        na_position="last",
    ).reset_index(drop=True)
    summary = summary.drop(columns=["condition_sort"])
    perm_table = perm_table.sort_values(
        ["analysis_type", "behavior_metric", "model", "roi_set", "roi", "condition_group"],
        na_position="last",
    ).reset_index(drop=True)

    write_table(summary, tables_main / "table_mechanism_behavior_prediction.tsv")
    write_table(perm_table, tables_si / "table_mechanism_behavior_prediction_perm.tsv")

    figure_path = figures_main / "fig_mechanism_behavior_prediction.png"
    _plot_top_associations(merged, summary, figure_path)

    save_json(
        {
            "model_metrics": [str(path) for path in metric_paths],
            "behavior_summary": str(args.behavior_summary),
            "paper_output_root": str(args.paper_output_root),
            "models": enabled_models,
            "include_m7": bool(args.include_m7),
            "n_bootstrap": int(args.n_bootstrap),
            "n_permutations": int(args.n_permutations),
            "min_subjects": int(args.min_subjects),
            "seed": int(args.seed),
            "n_merged_rows": int(len(merged)),
            "n_subjects_merged": int(merged["subject"].nunique()),
            "n_rois": int(merged["roi"].nunique()),
            "n_roi_sets": int(merged["roi_set"].nunique()),
            "main_table": str(tables_main / "table_mechanism_behavior_prediction.tsv"),
            "perm_table": str(tables_si / "table_mechanism_behavior_prediction_perm.tsv"),
            "figure": str(figure_path),
        },
        qc_dir / "mechanism_behavior_prediction_manifest.json",
    )


if __name__ == "__main__":
    main()
