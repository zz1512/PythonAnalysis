#!/usr/bin/env python3
"""YY-KJ contrast for relation-vector decoupling."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Iterable

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
MODEL_ROLES = {
    "M9_relation_vector_direct": "primary",
    "M9_relation_vector_abs": "primary",
    "M9_relation_vector_length": "control",
    "M3_embedding_pair_distance": "control",
    "M3_embedding_pair_centroid": "control",
}


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


def _infer_roi_set(path: Path) -> str:
    name = path.name
    prefix = "relation_vector_rsa_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def load_subject_contrasts(relation_dirs: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for relation_dir in relation_dirs:
        path = relation_dir / "relation_vector_subject_metrics.tsv"
        frame = _read_table(path)
        roi_set = frame["roi_set"].iloc[0] if "roi_set" in frame.columns and not frame.empty else _infer_roi_set(relation_dir)
        frame = frame.copy()
        frame["roi_set"] = str(roi_set)
        frame["condition"] = frame["condition"].astype(str).str.strip().str.lower()
        frame["time"] = frame["time"].astype(str).str.strip().str.lower()
        frame["rho"] = pd.to_numeric(frame["rho"], errors="coerce")
        frame["model_role"] = frame["model"].map(MODEL_ROLES).fillna(frame.get("model_role", "exploratory"))
        pivot = (
            frame.pivot_table(
                index=["subject", "roi_set", "roi", "condition", "neural_rdm_type", "model", "model_role"],
                columns="time",
                values="rho",
                aggfunc="mean",
            )
            .reset_index()
        )
        if "pre" not in pivot.columns or "post" not in pivot.columns:
            raise ValueError(f"{path} lacks matched pre/post rows")
        pivot["relation_delta_post_minus_pre"] = pivot["post"] - pivot["pre"]
        pivot["relation_decoupling_pre_minus_post"] = pivot["pre"] - pivot["post"]
        rows.append(pivot)
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot_table(
        index=["subject", "roi_set", "roi", "neural_rdm_type", "model", "model_role"],
        columns="condition",
        values="relation_decoupling_pre_minus_post",
        aggfunc="mean",
    ).reset_index()
    if "yy" not in wide.columns or "kj" not in wide.columns:
        raise ValueError("Need both yy and kj relation metrics to compute condition contrast.")
    wide["decoupling_yy"] = wide["yy"]
    wide["decoupling_kj"] = wide["kj"]
    wide["yy_minus_kj_decoupling"] = wide["decoupling_yy"] - wide["decoupling_kj"]
    wide["is_primary_model"] = wide["model"].isin(PRIMARY_MODELS)
    return wide.drop(columns=[col for col in ["yy", "kj"] if col in wide.columns])


def summarize(subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["roi_set", "roi", "neural_rdm_type", "model", "model_role"]
    for keys, subset in subject.groupby(group_cols, sort=False):
        roi_set, roi, neural_rdm_type, model, model_role = keys
        summary = _one_sample_summary(subset["yy_minus_kj_decoupling"])
        rows.append(
            {
                "roi_set": roi_set,
                "roi": roi,
                "neural_rdm_type": neural_rdm_type,
                "model": model,
                "model_role": model_role,
                "is_primary_model": model in PRIMARY_MODELS,
                "contrast": "yy_minus_kj_decoupling",
                **summary,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh_model_family"] = np.nan
    out["q_bh_primary_family"] = np.nan
    out["q_bh_model_role_family"] = np.nan
    for _, idx in out.groupby(["roi_set", "neural_rdm_type"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_model_family"] = _bh_fdr(out.loc[idx, "p"])
        primary_idx = out.index[out.index.isin(idx) & out["is_primary_model"]]
        if len(primary_idx):
            out.loc[primary_idx, "q_bh_primary_family"] = _bh_fdr(out.loc[primary_idx, "p"])
    for _, idx in out.groupby(["roi_set", "neural_rdm_type", "model_role"], dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, "q_bh_model_role_family"] = _bh_fdr(out.loc[idx, "p"])
    return out.sort_values(["q_bh_primary_family", "p"], na_position="last").reset_index(drop=True)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="YY-KJ relation-vector decoupling contrast.")
    parser.add_argument(
        "--relation-dirs",
        nargs="+",
        type=Path,
        default=[
            base_dir / "paper_outputs" / "qc" / "relation_vector_rsa_main_functional",
            base_dir / "paper_outputs" / "qc" / "relation_vector_rsa_literature",
            base_dir / "paper_outputs" / "qc" / "relation_vector_rsa_literature_spatial",
        ],
    )
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir or (args.paper_output_root / "qc" / "relation_vector_contrast"))
    tables_main = ensure_dir(args.paper_output_root / "tables_main")
    tables_si = ensure_dir(args.paper_output_root / "tables_si")
    subject = load_subject_contrasts(list(args.relation_dirs))
    summary = summarize(subject)
    write_table(subject, output_dir / "relation_vector_condition_contrast_subject.tsv")
    write_table(summary, output_dir / "relation_vector_condition_contrast_group_fdr.tsv")
    write_table(summary, tables_si / "table_relation_vector_condition_contrast_full.tsv")
    main_table = summary[
        summary["is_primary_model"].astype(bool)
        & summary["neural_rdm_type"].eq("relation_vector")
    ].copy()
    write_table(main_table, tables_main / "table_relation_vector_condition_contrast.tsv")
    save_json(
        {
            "relation_dirs": [str(path) for path in args.relation_dirs],
            "output_dir": str(output_dir),
            "n_subject_rows": int(len(subject)),
            "n_summary_rows": int(len(summary)),
            "metric": "yy_minus_kj_decoupling = (pre-post YY) - (pre-post KJ)",
        },
        output_dir / "relation_vector_condition_contrast_manifest.json",
    )
    print(f"[relation-vector-contrast] wrote {len(subject)} subject rows and {len(summary)} summary rows to {output_dir}")


if __name__ == "__main__":
    main()
