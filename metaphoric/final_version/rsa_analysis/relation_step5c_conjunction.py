#!/usr/bin/env python3
"""Conjoin pair-differentiation evidence with relation-vector contrasts."""

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


EDGE_CONTRAST_LABELS = {
    "yy_trained_drop_minus_untrained_nonedge_drop": "YY trained > YY untrained",
    "yy_trained_drop_minus_kj_trained_drop": "YY trained > KJ trained",
    "yy_trained_drop_minus_baseline_pseudo_drop": "YY trained > baseline pseudo",
    "yy_specificity_minus_kj_specificity": "YY specificity > KJ specificity",
}
STEP5C_CONTRAST_LABELS = {
    "step5c_metaphor_drop": "Step5C Metaphor drop",
    "step5c_metaphor_drop_minus_spatial_drop": "Step5C Metaphor > Spatial drop",
    "step5c_metaphor_drop_minus_baseline_drop": "Step5C Metaphor > Baseline drop",
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


def build_step5c_pair_evidence(step5c_subject_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _read_table(step5c_subject_path).copy()
    frame["condition"] = frame["condition"].astype(str).str.strip()
    frame["delta_post_minus_pre"] = pd.to_numeric(frame["delta_post_minus_pre"], errors="coerce")
    frame["drop_pre_minus_post"] = -frame["delta_post_minus_pre"]
    pivot = (
        frame.pivot_table(
            index=["subject", "roi"],
            columns="condition",
            values="drop_pre_minus_post",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    required = {"Metaphor", "Spatial", "Baseline"}
    missing = required.difference(pivot.columns)
    if missing:
        raise ValueError(f"Step5C subject table lacks conditions: {sorted(missing)}")
    pivot["step5c_metaphor_drop"] = pivot["Metaphor"]
    pivot["step5c_metaphor_drop_minus_spatial_drop"] = pivot["Metaphor"] - pivot["Spatial"]
    pivot["step5c_metaphor_drop_minus_baseline_drop"] = pivot["Metaphor"] - pivot["Baseline"]
    subject_long = pivot.melt(
        id_vars=["subject", "roi"],
        value_vars=list(STEP5C_CONTRAST_LABELS),
        var_name="pair_contrast",
        value_name="pair_subject_effect",
    )
    subject_long["roi_set"] = "main_functional"
    subject_long["pair_evidence_source"] = "step5c_condition_drop"
    rows: list[dict[str, object]] = []
    for keys, subset in subject_long.groupby(["roi_set", "roi", "pair_evidence_source", "pair_contrast"], sort=False):
        roi_set, roi, source, contrast = keys
        summary = _one_sample_summary(subset["pair_subject_effect"])
        rows.append(
            {
                "pair_evidence_source": source,
                "pair_contrast": contrast,
                "pair_contrast_label": STEP5C_CONTRAST_LABELS.get(contrast, contrast),
                "roi_set": roi_set,
                "roi": roi,
                **{f"pair_{key}": value for key, value in summary.items()},
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["pair_q"] = np.nan
        for _, idx in summary.groupby(["pair_evidence_source", "pair_contrast"], dropna=False).groups.items():
            idx = list(idx)
            summary.loc[idx, "pair_q"] = _bh_fdr(summary.loc[idx, "pair_p"])
    return subject_long, summary


def load_edge_pair_evidence(edge_path: Path) -> pd.DataFrame:
    frame = _read_table(edge_path).copy()
    frame = frame[frame["contrast"].isin(EDGE_CONTRAST_LABELS)].copy()
    frame["pair_evidence_source"] = "edge_specificity"
    frame["pair_contrast"] = frame["contrast"]
    frame["pair_contrast_label"] = frame["pair_contrast"].map(EDGE_CONTRAST_LABELS).fillna(frame["pair_contrast"])
    rename = {
        "n_subjects": "pair_n_subjects",
        "mean_effect": "pair_mean_effect",
        "t": "pair_t",
        "p": "pair_p",
        "cohens_dz": "pair_cohens_dz",
        "ci_low": "pair_ci_low",
        "ci_high": "pair_ci_high",
        "q_bh_primary_family": "pair_q",
    }
    frame = frame.rename(columns=rename)
    columns = [
        "pair_evidence_source",
        "pair_contrast",
        "pair_contrast_label",
        "roi_set",
        "roi",
        "pair_n_subjects",
        "pair_mean_effect",
        "pair_t",
        "pair_p",
        "pair_cohens_dz",
        "pair_ci_low",
        "pair_ci_high",
        "pair_q",
    ]
    return frame[columns].copy()


def load_relation_evidence(relation_path: Path) -> pd.DataFrame:
    frame = _read_table(relation_path).copy()
    frame = frame[frame["neural_rdm_type"].eq("relation_vector")].copy()
    rename = {
        "model": "relation_model",
        "model_role": "relation_model_role",
        "n_subjects": "relation_n_subjects",
        "mean_effect": "relation_mean_effect",
        "t": "relation_t",
        "p": "relation_p",
        "cohens_dz": "relation_cohens_dz",
        "ci_low": "relation_ci_low",
        "ci_high": "relation_ci_high",
        "q_bh_primary_family": "relation_q",
    }
    frame = frame.rename(columns=rename)
    columns = [
        "roi_set",
        "roi",
        "relation_model",
        "relation_model_role",
        "relation_n_subjects",
        "relation_mean_effect",
        "relation_t",
        "relation_p",
        "relation_cohens_dz",
        "relation_ci_low",
        "relation_ci_high",
        "relation_q",
    ]
    return frame[columns].copy()


def classify_conjunction(row: pd.Series, alpha: float) -> str:
    pair_sig = bool(row["pair_q"] <= alpha) if pd.notna(row["pair_q"]) else False
    relation_sig = bool(row["relation_q"] <= alpha) if pd.notna(row["relation_q"]) else False
    pair_pos = bool(row["pair_mean_effect"] > 0)
    relation_pos = bool(row["relation_mean_effect"] > 0)
    relation_neg = bool(row["relation_mean_effect"] < 0)
    if pair_sig and pair_pos and relation_sig and relation_pos:
        return "yy_pair_and_yy_relation"
    if pair_sig and pair_pos and relation_sig and relation_neg:
        return "yy_pair_kj_relation_dissociation"
    if pair_sig and pair_pos and not relation_sig:
        return "yy_pair_only"
    if (not pair_sig) and relation_sig and relation_pos:
        return "yy_relation_only"
    if (not pair_sig) and relation_sig and relation_neg:
        return "kj_relation_only"
    if pair_sig and not pair_pos:
        return "nonpositive_pair_effect"
    return "not_conjoined"


def build_conjunction(pair_evidence: pd.DataFrame, relation: pd.DataFrame, alpha: float) -> pd.DataFrame:
    joined = pair_evidence.merge(relation, on=["roi_set", "roi"], how="inner", validate="many_to_many")
    for col in ["pair_mean_effect", "pair_q", "relation_mean_effect", "relation_q", "pair_p", "relation_p"]:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
    joined["same_direction_product"] = joined["pair_mean_effect"] * joined["relation_mean_effect"]
    joined["abs_relation_effect"] = joined["relation_mean_effect"].abs()
    joined["pair_significant"] = joined["pair_q"] <= alpha
    joined["relation_significant"] = joined["relation_q"] <= alpha
    joined["pair_positive"] = joined["pair_mean_effect"] > 0
    joined["relation_direction"] = np.where(
        joined["relation_mean_effect"] > 0,
        "YY > KJ decoupling",
        np.where(joined["relation_mean_effect"] < 0, "KJ > YY decoupling", "zero"),
    )
    joined["conjunction_class"] = joined.apply(classify_conjunction, axis=1, alpha=alpha)
    priority = {
        "yy_pair_and_yy_relation": 0,
        "yy_pair_kj_relation_dissociation": 1,
        "yy_pair_only": 2,
        "yy_relation_only": 3,
        "kj_relation_only": 4,
        "nonpositive_pair_effect": 5,
        "not_conjoined": 6,
    }
    joined["conjunction_priority"] = joined["conjunction_class"].map(priority).fillna(99).astype(int)
    return joined.sort_values(
        ["conjunction_priority", "pair_q", "relation_q", "pair_p", "relation_p"],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Conjoin Step5C/A++ pair evidence with relation-vector contrasts.")
    parser.add_argument("--paper-output-root", type=Path, default=base_dir / "paper_outputs")
    parser.add_argument("--edge-table", type=Path, default=None)
    parser.add_argument("--relation-table", type=Path, default=None)
    parser.add_argument("--step5c-subject-table", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    edge_path = args.edge_table or paper_root / "tables_main" / "table_edge_specificity.tsv"
    relation_path = args.relation_table or paper_root / "tables_main" / "table_relation_vector_condition_contrast.tsv"
    step5c_path = args.step5c_subject_table or paper_root / "figures_main" / "table_step5c_roi_delta.tsv"
    output_dir = ensure_dir(args.output_dir or paper_root / "qc" / "relation_vector_conjunction")
    tables_main = ensure_dir(paper_root / "tables_main")
    tables_si = ensure_dir(paper_root / "tables_si")

    step5c_subject, step5c_summary = build_step5c_pair_evidence(step5c_path)
    edge_summary = load_edge_pair_evidence(edge_path)
    relation_summary = load_relation_evidence(relation_path)
    pair_evidence = pd.concat([edge_summary, step5c_summary], ignore_index=True, sort=False)
    conjunction = build_conjunction(pair_evidence, relation_summary, args.alpha)

    write_table(step5c_subject, output_dir / "step5c_pair_subject_contrasts.tsv")
    write_table(step5c_summary, output_dir / "step5c_pair_group_fdr.tsv")
    write_table(pair_evidence, output_dir / "pair_evidence_for_conjunction.tsv")
    write_table(conjunction, output_dir / "relation_step5c_conjunction.tsv")
    write_table(conjunction, tables_si / "table_relation_step5c_conjunction_full.tsv")
    main_table = conjunction[
        conjunction["conjunction_class"].isin(
            [
                "yy_pair_and_yy_relation",
                "yy_pair_kj_relation_dissociation",
                "yy_pair_only",
                "yy_relation_only",
                "kj_relation_only",
            ]
        )
    ].copy()
    write_table(main_table, tables_main / "table_relation_step5c_conjunction.tsv")
    save_json(
        {
            "edge_table": str(edge_path),
            "relation_table": str(relation_path),
            "step5c_subject_table": str(step5c_path),
            "output_dir": str(output_dir),
            "alpha": args.alpha,
            "n_pair_evidence_rows": int(len(pair_evidence)),
            "n_conjunction_rows": int(len(conjunction)),
            "n_main_rows": int(len(main_table)),
            "pair_positive_metric": "positive = stronger pair differentiation/drop evidence",
            "relation_metric": "positive = stronger YY than KJ relation-vector decoupling; negative = stronger KJ than YY",
        },
        output_dir / "relation_step5c_conjunction_manifest.json",
    )
    print(f"[relation-step5c-conjunction] wrote {len(conjunction)} conjunction rows to {output_dir}")
    print(f"[relation-step5c-conjunction] wrote {len(main_table)} main rows to {tables_main / 'table_relation_step5c_conjunction.tsv'}")


if __name__ == "__main__":
    main()
