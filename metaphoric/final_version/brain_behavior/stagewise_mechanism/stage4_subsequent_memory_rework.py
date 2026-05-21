#!/usr/bin/env python3
"""Stage 4: subsequent-memory rework for predefined mechanism metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


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


MATERIAL_COVARS = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]

KEY_COLS = [
    "subject",
    "condition",
    "original_pair_id",
    "template_pair_id",
    "condition_item_id",
]


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    if pd.isna(sd) or math.isclose(float(sd), 0.0):
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd


def fdr_by_family(df: pd.DataFrame, family_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["q"] = np.nan
    if out.empty or "p" not in out:
        return out
    ok = out["p"].notna() & np.isfinite(pd.to_numeric(out["p"], errors="coerce"))
    for _, idx in out[ok].groupby(family_cols, dropna=False).groups.items():
        p = pd.to_numeric(out.loc[idx, "p"], errors="coerce")
        _, q, _, _ = multipletests(p, method="fdr_bh")
        out.loc[idx, "q"] = q
    return out


def available_covariates(frame: pd.DataFrame, min_coverage: float = 0.8) -> list[str]:
    covs: list[str] = []
    for col in MATERIAL_COVARS:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().mean() >= min_coverage and vals.nunique(dropna=True) > 1:
            covs.append(col)
    return covs


def load_stage2_learning(path: Path) -> pd.DataFrame:
    keep = [
        *KEY_COLS,
        "semantic_bidirectional_specificity",
        "hpc_spatial_bidirectional_specificity",
        *MATERIAL_COVARS,
    ]
    frame = pd.read_csv(path, sep="\t", usecols=lambda c: c in keep, low_memory=False)
    frame = frame.rename(
        columns={
            "semantic_bidirectional_specificity": "semantic_learning_edge_trace",
            "hpc_spatial_bidirectional_specificity": "hpc_spatial_learning_edge_trace",
        }
    )
    return frame


def load_stage1_post_retrieval(path: Path) -> pd.DataFrame:
    keep = [
        *KEY_COLS,
        "memory",
        "hpc_spatial_post_separation",
        "hpc_spatial_rebinding",
        *MATERIAL_COVARS,
    ]
    return pd.read_csv(path, sep="\t", usecols=lambda c: c in keep, low_memory=False)


def build_stage4_item_table(stage1: pd.DataFrame, stage2: pd.DataFrame) -> pd.DataFrame:
    merged = stage1.merge(
        stage2,
        on=KEY_COLS,
        how="inner",
        suffixes=("", "_stage2"),
        validate="one_to_one",
    )
    for cov in MATERIAL_COVARS:
        alt = f"{cov}_stage2"
        if cov not in merged and alt in merged:
            merged[cov] = merged[alt]
        elif cov in merged and alt in merged:
            merged[cov] = pd.to_numeric(merged[cov], errors="coerce").fillna(
                pd.to_numeric(merged[alt], errors="coerce")
            )
    for col in merged.columns:
        if col not in {"subject", "condition", "condition_item_id"}:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["condition"] = merged["condition"].astype(str).str.lower()
    merged = merged[merged["condition"].isin(["kj", "yy"])].copy()
    merged["memory_score"] = pd.to_numeric(merged["memory"], errors="coerce")
    merged["memory_successes"] = np.rint(merged["memory_score"] * 2).astype("Int64")
    merged["remembered_strict"] = np.where(
        merged["memory_score"].eq(1.0),
        1.0,
        np.where(merged["memory_score"].eq(0.0), 0.0, np.nan),
    )

    metric_map = {
        "semantic_learning_edge_trace": "learning_edge_trace",
        "hpc_spatial_learning_edge_trace": "learning_edge_trace",
        "hpc_spatial_post_separation": "post_edge_specificity",
        "hpc_spatial_rebinding": "retrieval_rebinding",
    }
    rows: list[pd.DataFrame] = []
    id_cols = [
        *KEY_COLS,
        "memory",
        "memory_score",
        "memory_successes",
        "remembered_strict",
        *[c for c in MATERIAL_COVARS if c in merged.columns],
    ]
    for col, metric_family in metric_map.items():
        if col not in merged.columns:
            continue
        network = "semantic" if col.startswith("semantic_") else "hpc_spatial"
        tmp = merged[id_cols + [col]].rename(columns={col: "neural_metric"})
        tmp["metric_name"] = col
        tmp["metric_family"] = metric_family
        tmp["network"] = network
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out["neural_metric"] = pd.to_numeric(out["neural_metric"], errors="coerce")
    out["neural_metric_z"] = out.groupby("metric_name", dropna=False)["neural_metric"].transform(zscore)
    return out


def formula_columns(formula: str, outcome: str, frame: pd.DataFrame) -> list[str]:
    cols = ["subject", "condition", "condition_item_id", outcome]
    for col in frame.columns:
        if col in formula:
            cols.append(col)
    return sorted(set(cols))


def _extract_rows(result, model_info: dict[str, object], data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    params = result.params
    for term, estimate in params.items():
        if term.endswith(" Var") or term.startswith("item Var"):
            continue
        se = result.bse.get(term, np.nan)
        stat = estimate / se if pd.notna(se) and se != 0 else np.nan
        rows.append(
            {
                **model_info,
                "term": term,
                "contrast_type": "model_term",
                "estimate": float(estimate),
                "se": float(se) if pd.notna(se) else np.nan,
                "stat": float(stat) if pd.notna(stat) else np.nan,
                "p": float(result.pvalues.get(term, np.nan)),
                "n_rows": len(data),
                "n_subjects": int(data["subject"].nunique()),
                "n_condition_items": int(data["condition_item_id"].nunique()),
            }
        )
    return rows


def _linear_contrast(
    result,
    terms: dict[str, float],
) -> tuple[float, float, float, float]:
    params = result.params
    cov = result.cov_params()
    estimate = 0.0
    for term, weight in terms.items():
        if term not in params.index:
            return np.nan, np.nan, np.nan, np.nan
        estimate += weight * float(params.loc[term])
    var = 0.0
    for term_a, weight_a in terms.items():
        for term_b, weight_b in terms.items():
            if term_a not in cov.index or term_b not in cov.columns:
                return np.nan, np.nan, np.nan, np.nan
            var += weight_a * weight_b * float(cov.loc[term_a, term_b])
    se = math.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan
    stat = estimate / se if se and np.isfinite(se) else np.nan
    p = float(2 * stats.norm.sf(abs(stat))) if np.isfinite(stat) else np.nan
    return estimate, se, stat, p


def _condition_slope_rows(
    result,
    *,
    model_info: dict[str, object],
    data: pd.DataFrame,
    predictor: str,
) -> list[dict[str, object]]:
    interaction = f"{predictor}:C(condition, Treatment(reference='kj'))[T.yy]"
    if interaction not in result.params.index:
        interaction = f"C(condition, Treatment(reference='kj'))[T.yy]:{predictor}"
    specs = {
        f"{predictor}_slope_kj": {predictor: 1.0},
        f"{predictor}_slope_yy": {predictor: 1.0, interaction: 1.0},
        f"{predictor}_slope_yy_minus_kj": {interaction: 1.0},
    }
    rows: list[dict[str, object]] = []
    for term, weights in specs.items():
        estimate, se, stat, p = _linear_contrast(result, weights)
        rows.append(
            {
                **model_info,
                "term": term,
                "contrast_type": "condition_slope",
                "estimate": estimate,
                "se": se,
                "stat": stat,
                "p": p,
                "n_rows": len(data),
                "n_subjects": int(data["subject"].nunique()),
                "n_condition_items": int(data["condition_item_id"].nunique()),
            }
        )
    return rows


def fit_gaussian_model(
    frame: pd.DataFrame,
    *,
    model_name: str,
    predictor: str,
    outcome: str,
    formula: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = frame.replace([np.inf, -np.inf], np.nan)
    needed = formula_columns(formula, outcome, data)
    data = data.dropna(subset=needed).copy()
    data["subject"] = data["subject"].astype(str)
    data["condition"] = pd.Categorical(data["condition"].astype(str), categories=["kj", "yy"])
    data["condition_item_id"] = data["condition_item_id"].astype(str)

    info = {
        "analysis_id": "stage4_subsequent_memory",
        "model_name": model_name,
        "metric_name": str(frame["metric_name"].iloc[0]),
        "metric_family": str(frame["metric_family"].iloc[0]),
        "network": str(frame["network"].iloc[0]),
        "outcome": outcome,
        "predictor": predictor,
        "formula": formula,
        "status": "not_run",
        "converged": False,
        "fallback_used": "none",
        "message": "",
        "n_rows": len(data),
        "n_subjects": int(data["subject"].nunique()) if "subject" in data else 0,
        "n_condition_items": int(data["condition_item_id"].nunique()) if "condition_item_id" in data else 0,
    }
    if len(data) < 80 or info["n_subjects"] < 5 or info["n_condition_items"] < 10:
        info["status"] = "skipped_too_few_rows"
        return [], info

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            md = smf.mixedlm(
                formula,
                data,
                groups=data["subject"],
                vc_formula={"item": "0 + C(condition_item_id)"},
            )
            result = md.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            info["converged"] = bool(getattr(result, "converged", False))
            info["status"] = "ok" if info["converged"] else "mixed_not_converged"
        except Exception as exc:
            info["fallback_used"] = "ols_cluster_subject"
            info["message"] = repr(exc)
            try:
                result = smf.ols(formula, data).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": data["subject"]},
                )
                info["converged"] = True
                info["status"] = "fallback_ols_cluster_subject"
            except Exception as exc2:
                info["status"] = "failed"
                info["message"] = f"{info['message']}; fallback={repr(exc2)}"
                return [], info
        if caught:
            info["message"] = " | ".join(sorted({str(w.message)[:180] for w in caught}))

    rows = _extract_rows(result, info, data)
    rows.extend(_condition_slope_rows(result, model_info=info, data=data, predictor=predictor))
    return rows, info


def run_models(item: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    covs = available_covariates(item)
    cov_expr = (" + " + " + ".join(covs)) if covs else ""
    specs = [
        (
            "primary_continuous_memory",
            "memory_score",
            f"neural_metric_z ~ memory_score * C(condition, Treatment(reference='kj')){cov_expr}",
        ),
        (
            "sensitivity_success_count",
            "memory_successes",
            f"neural_metric_z ~ memory_successes * C(condition, Treatment(reference='kj')){cov_expr}",
        ),
        (
            "sensitivity_strict_remembered",
            "remembered_strict",
            f"neural_metric_z ~ remembered_strict * C(condition, Treatment(reference='kj')){cov_expr}",
        ),
    ]
    model_rows: list[dict[str, object]] = []
    conv_rows: list[dict[str, object]] = []
    for metric_name, sub in item.groupby("metric_name", sort=False):
        for model_name, predictor, formula in specs:
            rows, info = fit_gaussian_model(
                sub,
                model_name=model_name,
                predictor=predictor,
                outcome="neural_metric_z",
                formula=formula,
            )
            model_rows.extend(rows)
            conv_rows.append(info)
    models = pd.DataFrame(model_rows)
    if not models.empty:
        models = fdr_by_family(models, ["model_name", "term", "contrast_type"])
    convergence = pd.DataFrame(conv_rows)
    return models, convergence


def build_qc(item: pd.DataFrame, convergence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "table": "stage4_subsequent_memory_item",
            "n_rows": len(item),
            "n_subjects": item["subject"].nunique(),
            "n_condition_items": item["condition_item_id"].nunique(),
            "n_metrics": item["metric_name"].nunique(),
            "memory_unique_values": ",".join(
                map(str, sorted(pd.to_numeric(item["memory"], errors="coerce").dropna().unique()))
            ),
            "strict_rows": int(item["remembered_strict"].notna().sum()),
            "condition_item_cross_condition_collisions": int(
                item[["condition_item_id", "condition"]]
                .drop_duplicates()
                .groupby("condition_item_id")["condition"]
                .nunique()
                .gt(1)
                .sum()
            ),
        }
    )
    for (metric_name, condition), sub in item.groupby(["metric_name", "condition"], dropna=False):
        rows.append(
            {
                "table": "metric_condition",
                "metric_name": metric_name,
                "condition": condition,
                "n_rows": len(sub),
                "n_subjects": sub["subject"].nunique(),
                "n_condition_items": sub["condition_item_id"].nunique(),
                "mean_memory": float(pd.to_numeric(sub["memory"], errors="coerce").mean()),
                "mean_neural_metric": float(pd.to_numeric(sub["neural_metric"], errors="coerce").mean()),
            }
        )
    for _, row in convergence.iterrows():
        rows.append(
            {
                "table": "model_convergence",
                "metric_name": row.get("metric_name"),
                "model_name": row.get("model_name"),
                "n_rows": row.get("n_rows"),
                "n_subjects": row.get("n_subjects"),
                "n_condition_items": row.get("n_condition_items"),
                "status": row.get("status"),
                "fallback_used": row.get("fallback_used"),
                "converged": row.get("converged"),
            }
        )
    return pd.DataFrame(rows)


def write_review(out_dir: Path, item: pd.DataFrame, models: pd.DataFrame, convergence: pd.DataFrame) -> None:
    def md_table(frame: pd.DataFrame, floatfmt: str = ".4g") -> str:
        if frame.empty:
            return "_No rows._"
        cols = list(frame.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in frame.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    vals.append("")
                elif isinstance(val, (float, np.floating)):
                    vals.append(format(float(val), floatfmt))
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    primary = models[
        models["model_name"].eq("primary_continuous_memory")
        & models["contrast_type"].eq("condition_slope")
    ].copy()
    primary = primary.sort_values(["term", "p"], na_position="last")
    sens = models[
        models["model_name"].ne("primary_continuous_memory")
        & models["contrast_type"].eq("condition_slope")
    ].copy()
    terms = [
        "memory_score_slope_kj",
        "memory_score_slope_yy",
        "memory_score_slope_yy_minus_kj",
    ]
    primary_show = md_table(primary[primary["term"].isin(terms)][
        ["metric_name", "network", "term", "estimate", "se", "p", "q", "n_rows", "n_subjects"]
    ])
    sensitivity_show = md_table(sens.sort_values(["model_name", "term", "p"], na_position="last")[
        ["model_name", "metric_name", "term", "estimate", "p", "q"]
    ])
    conv_show = md_table(convergence[
        ["model_name", "metric_name", "status", "fallback_used", "n_rows", "n_subjects", "n_condition_items"]
    ])
    memory_counts = (
        item.drop_duplicates(["subject", "condition_item_id"])
        .groupby(["condition", "memory"], dropna=False)
        .size()
        .reset_index(name="n_items")
    )
    memory_counts = md_table(memory_counts)
    text = f"""# Stage 4 Subsequent-Memory Rework Review

Stage 4 asks whether later remembered items differ in predefined neural mechanism metrics.
The model direction is intentionally subsequent-memory style:

```text
neural_metric_z ~ memory_score * condition + material_covariates
                + (1 | subject) + (1 | condition_item_id)
```

Primary metrics are limited to network composites:

- semantic learning edge trace
- hpc-spatial learning edge trace
- hpc-spatial post edge separation
- hpc-spatial retrieval re-binding

## Input/QC

- item rows: {len(item)}
- subjects: {item['subject'].nunique()}
- condition items: {item['condition_item_id'].nunique()}
- metric count: {item['metric_name'].nunique()}

Memory distribution:

{memory_counts}

## Primary Continuous-Memory Slopes

Positive slopes mean later remembered items have larger neural metric values.

{primary_show}

## Sensitivity Models

`memory_successes` treats 0/0.5/1 as 0/1/2 remembered trials.
`remembered_strict` excludes ambiguous 0.5 items.

{sensitivity_show}

## Convergence

{conv_show}
"""
    (out_dir / "stage4_subsequent_memory_review.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 subsequent-memory rework.")
    base_dir = _default_base_dir()
    default_out = base_dir / "paper_outputs" / "qc" / "stagewise_mechanism"
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument(
        "--stage1-network-item",
        type=Path,
        default=default_out / "stage1_post_to_retrieval_network_item.tsv",
    )
    parser.add_argument(
        "--stage2-network-item",
        type=Path,
        default=default_out / "stage2a_learning_item_trace_network_item.tsv",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    stage1 = load_stage1_post_retrieval(args.stage1_network_item)
    stage2 = load_stage2_learning(args.stage2_network_item)
    item = build_stage4_item_table(stage1, stage2)
    models, convergence = run_models(item)
    sensitivity = models[models["model_name"].ne("primary_continuous_memory")].copy()
    qc = build_qc(item, convergence)

    write_table(item, out_dir / "stage4_subsequent_memory_item.tsv")
    write_table(models, out_dir / "stage4_subsequent_memory_models.tsv")
    write_table(sensitivity, out_dir / "stage4_subsequent_memory_sensitivity.tsv")
    write_table(convergence, out_dir / "stage4_subsequent_memory_convergence.tsv")
    write_table(qc, out_dir / "stage4_subsequent_memory_qc.tsv")
    write_review(out_dir, item, models, convergence)

    manifest = {
        "analysis_id": "stage4_subsequent_memory_rework",
        "n_item_rows": int(len(item)),
        "n_subjects": int(item["subject"].nunique()),
        "n_condition_items": int(item["condition_item_id"].nunique()),
        "n_metrics": int(item["metric_name"].nunique()),
        "n_model_rows": int(len(models)),
        "out_dir": str(out_dir),
    }
    (out_dir / "stage4_subsequent_memory_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
