#!/usr/bin/env python3
"""Negative controls for the learning-post-memory mechanism."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

from common.final_utils import ensure_dir, write_table  # noqa: E402
from brain_behavior.learning_post_memory_model_utils import (  # noqa: E402
    available_covariates,
    fit_roi_gee,
    read_item_table,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _add_shuffled_predictor(frame: pd.DataFrame, seed: int = 20260509) -> pd.DataFrame:
    out = frame.copy()
    out["shuffled_predictor_z"] = np.nan
    rng = np.random.default_rng(seed)
    for _, idx in out.groupby(["subject", "roi_set", "roi", "condition"]).groups.items():
        vals = out.loc[idx, "learning_specificity_z"].to_numpy(dtype=float)
        if len(vals) > 1:
            vals = rng.permutation(vals)
        out.loc[idx, "shuffled_predictor_z"] = vals
    return out


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
    table_dir = ensure_dir(args.paper_output_root / "tables_si")
    frame = _add_shuffled_predictor(read_item_table(args.item_table))
    covs = available_covariates(frame)

    specs = [
        (
            "negative_pseudo_edge",
            "pseudo_edge_drop ~ learning_specificity_z * C(condition, Treatment(reference='kj'))"
            + (" + " + " + ".join(covs) if covs else ""),
            "pseudo_edge_drop",
            "gaussian",
        ),
        (
            "negative_pre_similarity",
            "pre_pair_similarity ~ learning_specificity_z * C(condition, Treatment(reference='kj'))"
            + (" + " + " + ".join(covs) if covs else ""),
            "pre_pair_similarity",
            "gaussian",
        ),
        (
            "negative_shuffled_memory",
            "memory ~ shuffled_predictor_z * C(condition, Treatment(reference='kj'))"
            + (" + " + " + ".join(covs) if covs else ""),
            "memory",
            "binomial",
        ),
    ]
    results = []
    for model_name, formula, outcome, family in specs:
        results.append(
            fit_roi_gee(
                frame,
                formula=formula,
                outcome=outcome,
                model_name=model_name,
                family_name=family,
                covariates=covs,
            )
        )
    out = pd.concat(results, ignore_index=True)
    write_table(out, out_dir / "negative_control_models.tsv")
    write_table(out, table_dir / "table_learning_post_negative_controls.tsv")
    # Placeholder for later bootstrap expansion; keeps spec output contract.
    summary = out[out["status"].eq("ok")].groupby(["model", "roi_set"], as_index=False).agg(
        n_terms=("term", "count"),
        min_q=("q", "min"),
    )
    write_table(summary, out_dir / "bootstrap_sensitivity.tsv")
    print(f"Wrote {len(out)} negative-control rows; covariates={covs}")


if __name__ == "__main__":
    main()
