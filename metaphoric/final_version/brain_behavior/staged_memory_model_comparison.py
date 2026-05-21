#!/usr/bin/env python3
"""Secondary staged incremental prediction of run7 memory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, write_table  # noqa: E402
from brain_behavior.learning_post_memory_model_utils import (  # noqa: E402
    add_residualized_post_specificity,
    available_covariates,
    bh_fdr,
    prepare_model_frame,
    read_item_table,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _fit(formula: str, data: pd.DataFrame):
    return smf.gee(
        formula=formula,
        groups="subject",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument(
        "--item-table",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.paper_output_root / "qc" / "learning_post_memory_prediction")
    table_dir = ensure_dir(args.paper_output_root / "tables_main")
    frame = add_residualized_post_specificity(read_item_table(args.item_table))
    covs = available_covariates(frame)
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    models = [
        ("M0_condition_covariates", "memory ~ C(condition, Treatment(reference='kj'))" + cov_expr, []),
        ("M1_learning", "memory ~ C(condition, Treatment(reference='kj')) + learning_specificity_z" + cov_expr, ["learning_specificity_z"]),
        ("M2_post_edge", "memory ~ C(condition, Treatment(reference='kj')) + learning_specificity_z + post_edge_specificity_z" + cov_expr, ["learning_specificity_z", "post_edge_specificity_z"]),
        ("M3_retrieval_pair", "memory ~ C(condition, Treatment(reference='kj')) + learning_specificity_z + post_edge_specificity_z + retrieval_pair_similarity_z" + cov_expr, ["learning_specificity_z", "post_edge_specificity_z", "retrieval_pair_similarity_z"]),
        ("M4_retrieval_minus_post", "memory ~ C(condition, Treatment(reference='kj')) + learning_specificity_z + post_edge_specificity_z + retrieval_minus_post_pair_similarity_z" + cov_expr, ["learning_specificity_z", "post_edge_specificity_z", "retrieval_minus_post_pair_similarity_z"]),
    ]

    rows = []
    for (roi_set, roi), sub in frame.groupby(["roi_set", "roi"], sort=False):
        required = sorted(set(["memory", "learning_specificity_z", "post_edge_specificity_z", "retrieval_pair_similarity_z", "retrieval_minus_post_pair_similarity_z", *covs]))
        data = prepare_model_frame(sub, required, covs)
        if len(data) < 80 or data["subject"].nunique() < 5:
            rows.append(
                {
                    "roi_set": roi_set,
                    "roi": roi,
                    "model": "__all__",
                    "status": "skipped_insufficient_data",
                    "n_rows": len(data),
                    "n_subjects": data["subject"].nunique() if "subject" in data else 0,
                }
            )
            continue
        previous_loss = None
        for model_name, formula, predictors in models:
            model_data = data.dropna(subset=["memory", *predictors]).copy()
            try:
                result = _fit(formula, model_data)
                pred = result.predict(model_data)
                loss = _log_loss(model_data["memory"].to_numpy(dtype=float), pred)
                rows.append(
                    {
                        "roi_set": roi_set,
                        "roi": roi,
                        "model": model_name,
                        "status": "ok",
                        "n_rows": len(model_data),
                        "n_subjects": int(model_data["subject"].nunique()),
                        "n_condition_items": int(model_data["condition_item_id"].nunique()),
                        "log_loss": loss,
                        "delta_log_loss_vs_previous": previous_loss - loss if previous_loss is not None else np.nan,
                        "formula": formula,
                    }
                )
                previous_loss = loss
            except Exception as exc:
                rows.append(
                    {
                        "roi_set": roi_set,
                        "roi": roi,
                        "model": model_name,
                        "status": f"failed: {exc}",
                        "n_rows": len(model_data),
                        "n_subjects": int(model_data["subject"].nunique()) if "subject" in model_data else 0,
                        "n_condition_items": int(model_data["condition_item_id"].nunique()) if "condition_item_id" in model_data else 0,
                        "log_loss": np.nan,
                        "delta_log_loss_vs_previous": np.nan,
                        "formula": formula,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty and "delta_log_loss_vs_previous" in out.columns:
        out["rank_within_roi"] = out.groupby(["roi_set", "roi"])["log_loss"].rank(method="dense")
        # Smaller log loss is better; this q column is only a descriptive family rank, not a p-value FDR.
        out["descriptive_best_model"] = out.groupby(["roi_set", "roi"])["log_loss"].transform("min").eq(out["log_loss"])
    write_table(out, out_dir / "staged_model_comparison.tsv")
    write_table(out, table_dir / "table_staged_memory_model_comparison.tsv")
    print(f"Wrote {len(out)} staged-comparison rows; covariates={covs}")


if __name__ == "__main__":
    main()
