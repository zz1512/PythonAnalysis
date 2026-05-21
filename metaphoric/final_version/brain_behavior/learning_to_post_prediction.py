#!/usr/bin/env python3
"""Learning-stage item specificity predicting post-pre edge specificity."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

from common.final_utils import ensure_dir, write_table  # noqa: E402
from brain_behavior.learning_post_memory_model_utils import (  # noqa: E402
    add_residualized_post_specificity,
    available_covariates,
    fit_roi_gee,
    read_item_table,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


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

    formulas = [
        (
            "learning_to_post_primary",
            "post_edge_specificity ~ learning_specificity_z * C(condition, Treatment(reference='kj'))"
            + (" + " + " + ".join(covs) if covs else ""),
            "post_edge_specificity",
        ),
        (
            "learning_to_post_pre_control",
            "post_edge_specificity ~ learning_specificity_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + (" + " + " + ".join(covs) if covs else ""),
            "post_edge_specificity",
        ),
        (
            "learning_to_post_residualized",
            "post_edge_specificity_resid ~ learning_specificity_z * C(condition, Treatment(reference='kj'))"
            + (" + " + " + ".join(covs) if covs else ""),
            "post_edge_specificity_resid",
        ),
    ]
    results = []
    for model_name, formula, outcome in formulas:
        results.append(
            fit_roi_gee(
                frame,
                formula=formula,
                outcome=outcome,
                model_name=model_name,
                family_name="gaussian",
                covariates=covs,
            )
        )
    out = pd.concat(results, ignore_index=True)
    write_table(out, out_dir / "learning_to_post_models.tsv")
    write_table(out[out["term"].str.contains("learning_specificity", regex=False, na=False)].copy(), table_dir / "table_learning_to_post_prediction.tsv")
    write_table(out[out["model"].str.contains("control|residualized", regex=True, na=False)].copy(), out_dir / "learning_to_post_robustness.tsv")
    print(f"Wrote {len(out)} model rows; covariates={covs}")


if __name__ == "__main__":
    main()
