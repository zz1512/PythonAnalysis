#!/usr/bin/env python3
"""C3: novelty/familiarity moderation of post edge differentiation.

Unit policy
-----------
The outcome ``post_edge_differentiation`` is now aggregated to the
``subject x network x condition_item_id`` raw network mean, then z-scored once
globally to feed the regression. The previous version ran ``make_network_composite``
(internal subject x roi z-score) followed by ``groupby('network').transform(zscore)``
which forced YY/KJ outcome means toward symmetric reflections (the same
``z-of-z`` pattern as the §28.4 mirror artefact) and mixed two scales with
``novelty_z`` (which is global). After this fix, the ``condition``,
``novelty_z`` and ``condition:novelty_z`` terms all live on consistent z scales.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from shared_nc import add_common_args, add_network_column, build_condition_item_id, default_config, fit_formula, q_for_ok_rows, read_table, write_outputs, zscore

MODULE = "c3_novelty_familiarity_moderation"
ITEM_INPUT = Path("qc/learning_post_memory_prediction/item_mechanism_table.tsv")
MATERIAL_INPUT = Path("tables_si/table_stimulus_control_from_materials.tsv")
COVARIATES = ["sentence_char_len_z", "word_frequency_mean_z", "stroke_count_mean_z", "valence_mean_z", "arousal_mean_z"]
MATERIAL_TERMS = ["novelty", "familiarity", "comprehensibility", "difficulty"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--item-input", type=Path, default=None)
    parser.add_argument("--material-input", type=Path, default=None)
    return parser.parse_args()


def condition_norm(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"metaphor", "yy"}:
        return "yy"
    if text in {"spatial", "kj"}:
        return "kj"
    return re.sub(r"[^a-z]+", "", text)


def prepare_materials(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "control_scope" in out.columns:
        out = out[out["control_scope"].astype(str).str.lower().eq("sentence")].copy()
    out["condition_norm"] = out["condition"].map(condition_norm)
    for col in MATERIAL_TERMS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_z"] = zscore(out[col])
    keep = [c for c in ["condition_norm", "sentence", *MATERIAL_TERMS, *[f"{c}_z" for c in MATERIAL_TERMS]] if c in out.columns]
    return out[keep].drop_duplicates()


def prepare_items(frame: pd.DataFrame, materials: pd.DataFrame) -> pd.DataFrame:
    out = add_network_column(build_condition_item_id(frame))
    out["condition_norm"] = out["condition"].map(condition_norm)
    sentence_col = "sentence_text" if "sentence_text" in out.columns else "sentence"
    if sentence_col in out.columns and "sentence" in materials.columns:
        out = out.merge(materials, left_on=["condition_norm", sentence_col], right_on=["condition_norm", "sentence"], how="left", suffixes=("", "_material"))
    for col in ["trained_edge_drop", "post_edge_specificity", *COVARIATES]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "network" not in out.columns or "condition_item_id" not in out.columns:
        return out
    out = out[out["network"].notna() & out["condition_item_id"].astype(str).ne("")].copy()
    metric_col = "post_edge_specificity" if "post_edge_specificity" in out.columns else ("trained_edge_drop" if "trained_edge_drop" in out.columns else None)
    if metric_col is None:
        return out
    id_keys = [c for c in ["subject", "condition_item_id", "condition", "network"] if c in out.columns]
    material_z_cols = [f"{c}_z" for c in MATERIAL_TERMS if f"{c}_z" in out.columns]
    cov_keep = [c for c in COVARIATES if c in out.columns]
    agg_cols = list(dict.fromkeys([metric_col, *material_z_cols, *cov_keep]))
    network_raw = out.groupby(id_keys, dropna=False, as_index=False)[agg_cols].mean()
    network_raw = network_raw.rename(columns={metric_col: "post_edge_differentiation_raw"})
    # Single global z-score for the outcome so that condition, novelty_z and
    # condition:novelty_z terms share a consistent z-scale.
    network_raw["post_edge_differentiation_z"] = zscore(network_raw["post_edge_differentiation_raw"])
    return network_raw


def coverage(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network, sub in data.groupby("network", dropna=False):
        row = {"network": network, "n_rows": len(sub), "n_subjects": sub["subject"].nunique() if "subject" in sub else None, "n_items": sub["condition_item_id"].nunique() if "condition_item_id" in sub else None}
        for col in MATERIAL_TERMS:
            z_col = f"{col}_z"
            row[f"{col}_n"] = pd.to_numeric(sub.get(z_col), errors="coerce").notna().sum() if z_col in sub.columns else 0
        rows.append(row)
    return pd.DataFrame(rows)


def usable_terms(frame: pd.DataFrame, terms: list[str]) -> list[str]:
    out = []
    for col in terms:
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().sum() >= 10 and frame[col].nunique(dropna=True) >= 2:
            out.append(col)
    return out


def fit_models(data: pd.DataFrame) -> pd.DataFrame:
    needed = {"subject", "condition_item_id", "condition", "network", "post_edge_differentiation_z"}
    if not needed.issubset(data.columns):
        return pd.DataFrame([{"term": "__model__", "status": f"missing_columns: {sorted(needed - set(data.columns))}"}])
    rows = []
    for network, sub in data.groupby("network", dropna=False):
        material_terms = usable_terms(sub, [f"{c}_z" for c in MATERIAL_TERMS])
        covs = usable_terms(sub, COVARIATES)
        if "novelty_z" not in material_terms:
            rows.append(pd.DataFrame([{"network": network, "term": "__model__", "status": "missing_novelty_coverage", "n_obs": len(sub)}]))
            continue
        rhs = ["condition", "novelty_z", "condition:novelty_z"]
        rhs.extend([term for term in material_terms if term != "novelty_z"])
        rhs.extend(covs)
        required_cols = ["post_edge_differentiation_z", "condition", "novelty_z", *[term for term in material_terms if term != "novelty_z"], *covs]
        model_data = sub.dropna(subset=required_cols).copy()
        if len(model_data) < 30:
            rows.append(pd.DataFrame([{"network": network, "term": "__model__", "status": "too_few_complete_rows", "n_obs": len(model_data)}]))
            continue
        res = fit_formula(model_data, "post_edge_differentiation_z ~ " + " + ".join(rhs), family="gaussian")
        res.insert(0, "network", network)
        res.insert(1, "mechanism_model", "novelty_familiarity_moderation")
        rows.append(res)
    return q_for_ok_rows(pd.concat(rows, ignore_index=True, sort=False)) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    item_path = Path(args.item_input or cfg.paper_output_root / ITEM_INPUT)
    material_path = Path(args.material_input or cfg.paper_output_root / MATERIAL_INPUT)
    item = read_table(item_path)
    materials = prepare_materials(read_table(material_path))
    data = prepare_items(item, materials)
    models = fit_models(data)
    cov = coverage(data)
    write_outputs(cfg, MODULE, {
        "novelty_familiarity_moderation.tsv": models,
        "novelty_familiarity_input_coverage.tsv": cov,
        "novelty_familiarity_manifest.tsv": pd.DataFrame([
            {"path": str(item_path), "role": "item_neural", "n_rows": len(item)},
            {"path": str(material_path), "role": "materials", "n_rows": len(materials)},
        ]),
    })


if __name__ == "__main__":
    main()
