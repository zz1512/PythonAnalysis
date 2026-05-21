from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


QC_ROOT = Path("E:/python_metaphor/paper_outputs/qc")
OUT_DIR = QC_ROOT / "mixed_effects"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_TABLE = QC_ROOT / "learning_reactivation_mapping" / "reactivation_mapping_analysis_table.tsv"
NETWORK_TABLE = QC_ROOT / "learning_reactivation_mapping" / "network_bridge_item.tsv"
ITEM_TABLE = QC_ROOT / "learning_post_memory_prediction" / "item_mechanism_table.tsv"

META_METAPHOR_N = 8
META_SPATIAL_N = 10

MATERIAL_COVARS = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]


def fdr_by_family(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if results.empty or "p" not in results.columns:
        results["q"] = np.nan
        return results
    results = results.copy()
    results["q"] = np.nan
    for _, idx in results.groupby(group_cols, dropna=False).groups.items():
        p = pd.to_numeric(results.loc[idx, "p"], errors="coerce")
        ok = p.notna()
        if ok.any():
            _, q, _, _ = multipletests(p[ok], method="fdr_bh")
            results.loc[p[ok].index, "q"] = q
    return results


def available_covars(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    keep: list[str] = []
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            keep.append(col)
    return keep


def term_table_from_result(
    res,
    *,
    analysis_id: str,
    model_name: str,
    roi_set: str,
    roi: str,
    outcome: str,
    formula: str,
    data: pd.DataFrame,
    converged: bool,
    singular: bool,
    fallback_used: str,
    status: str,
) -> list[dict]:
    params = getattr(res, "params", pd.Series(dtype=float))
    bse = getattr(res, "bse", pd.Series(dtype=float))
    pvalues = getattr(res, "pvalues", pd.Series(dtype=float))
    rows = []
    for term, est in params.items():
        if term.startswith("item Var") or term.endswith(" Var"):
            continue
        se = bse.get(term, np.nan)
        p = pvalues.get(term, np.nan)
        stat = est / se if se is not None and pd.notna(se) and se != 0 else np.nan
        rows.append(
            {
                "analysis_id": analysis_id,
                "model_name": model_name,
                "roi_set": roi_set,
                "roi": roi,
                "outcome": outcome,
                "term": term,
                "estimate": est,
                "se": se,
                "stat": stat,
                "p": p,
                "n_rows": len(data),
                "n_subjects": data["subject"].nunique() if "subject" in data else np.nan,
                "n_items": data["condition_item_id"].nunique()
                if "condition_item_id" in data
                else np.nan,
                "formula": formula,
                "converged": converged,
                "singular": singular,
                "fallback_used": fallback_used,
                "status": status,
            }
        )
    return rows


def fit_mixed_or_fallback(
    data: pd.DataFrame,
    formula: str,
    *,
    analysis_id: str,
    model_name: str,
    roi_set: str,
    roi: str,
    outcome: str,
) -> tuple[list[dict], dict]:
    needed = ["subject", "condition_item_id", outcome]
    formula_cols = [c for c in data.columns if c in formula]
    drop_cols = sorted(set([c for c in needed if c in data.columns] + formula_cols))
    model_data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=drop_cols)
    if len(model_data) < 30 or model_data["subject"].nunique() < 5:
        info = {
            "analysis_id": analysis_id,
            "model_name": model_name,
            "roi_set": roi_set,
            "roi": roi,
            "outcome": outcome,
            "n_rows": len(model_data),
            "n_subjects": model_data["subject"].nunique() if "subject" in model_data else 0,
            "n_items": model_data["condition_item_id"].nunique()
            if "condition_item_id" in model_data
            else 0,
            "converged": False,
            "singular": False,
            "fallback_used": "none",
            "status": "skipped_too_few_rows",
            "message": "",
        }
        return [], info

    model_data = model_data.copy()
    model_data["subject"] = model_data["subject"].astype(str)
    model_data["condition_item_id"] = model_data["condition_item_id"].astype(str)
    model_data["condition"] = model_data["condition"].astype(str)

    status = "ok"
    converged = False
    singular = False
    fallback_used = "none"
    message = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            md = smf.mixedlm(
                formula,
                model_data,
                groups=model_data["subject"],
                vc_formula={"item": "0 + C(condition_item_id)"},
            )
            res = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            converged = bool(getattr(res, "converged", False))
            cov_re = getattr(res, "cov_re", None)
            singular = bool(cov_re is not None and np.asarray(cov_re).size > 0 and np.nanmin(np.linalg.eigvalsh(cov_re)) < 1e-8)
            if not converged:
                status = "mixed_not_converged"
        except Exception as exc:
            status = "fallback_ols_cluster_subject"
            fallback_used = "ols_cluster_subject"
            message = repr(exc)
            try:
                res = smf.ols(formula, model_data).fit(
                    cov_type="cluster", cov_kwds={"groups": model_data["subject"]}
                )
                converged = True
            except Exception as exc2:
                info = {
                    "analysis_id": analysis_id,
                    "model_name": model_name,
                    "roi_set": roi_set,
                    "roi": roi,
                    "outcome": outcome,
                    "n_rows": len(model_data),
                    "n_subjects": model_data["subject"].nunique(),
                    "n_items": model_data["condition_item_id"].nunique(),
                    "converged": False,
                    "singular": False,
                    "fallback_used": "ols_cluster_subject",
                    "status": "failed",
                    "message": f"{message}; fallback={repr(exc2)}",
                }
                return [], info
        if caught and not message:
            message = " | ".join(sorted({str(w.message)[:200] for w in caught}))

    rows = term_table_from_result(
        res,
        analysis_id=analysis_id,
        model_name=model_name,
        roi_set=roi_set,
        roi=roi,
        outcome=outcome,
        formula=formula,
        data=model_data,
        converged=converged,
        singular=singular,
        fallback_used=fallback_used,
        status=status,
    )
    info = {
        "analysis_id": analysis_id,
        "model_name": model_name,
        "roi_set": roi_set,
        "roi": roi,
        "outcome": outcome,
        "n_rows": len(model_data),
        "n_subjects": model_data["subject"].nunique(),
        "n_items": model_data["condition_item_id"].nunique(),
        "converged": converged,
        "singular": singular,
        "fallback_used": fallback_used,
        "status": status,
        "message": message,
    }
    return rows, info


def load_analysis_table() -> pd.DataFrame:
    cols = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "original_pair_id",
        "condition_item_id",
        "post_edge_specificity",
        "trained_edge_drop",
        "pre_pair_similarity",
        "post_pair_similarity",
        "retrieval_minus_post_pair_similarity",
        "retrieval_minus_pre_pair_similarity",
        "retrieval_pair_similarity",
        "memory",
        "sentence_char_len_z",
        "word_frequency_mean_z",
        "stroke_count_mean_z",
        "valence_mean_z",
        "arousal_mean_z",
        "react_pair_mean_avg_z",
        "learning_fluency_shift_z",
        "target_source_over_self_z",
        "mapping_asymmetry_self_corrected_z",
        "pre_pair_similarity_z",
        "post_edge_specificity_z",
    ]
    df = pd.read_csv(ANALYSIS_TABLE, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    for c in df.columns:
        if c not in {"subject", "roi_set", "roi", "condition", "condition_item_id"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_network_table() -> pd.DataFrame:
    df = pd.read_csv(NETWORK_TABLE, sep="\t", low_memory=False)
    for c in df.columns:
        if c not in {"subject", "condition", "condition_item_id"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["roi_set"] = "network"
    df["roi"] = "semantic_to_hpc_spatial"
    return df


def build_trajectory_table() -> pd.DataFrame:
    cols = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "condition_item_id",
        "pre_pair_similarity",
        "learning_self_similarity",
        "post_pair_similarity",
        "retrieval_pair_similarity",
        "sentence_char_len_z",
        "word_frequency_mean_z",
        "stroke_count_mean_z",
        "valence_mean_z",
        "arousal_mean_z",
    ]
    base = pd.read_csv(ITEM_TABLE, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    for c in cols:
        if c in base.columns and c not in {"subject", "roi_set", "roi", "condition", "condition_item_id"}:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    phase_map = {
        "pre": "pre_pair_similarity",
        "learning": "learning_self_similarity",
        "post": "post_pair_similarity",
        "retrieval": "retrieval_pair_similarity",
    }
    rows = []
    id_cols = ["subject", "roi_set", "roi", "condition", "condition_item_id"] + available_covars(base, MATERIAL_COVARS)
    for phase, col in phase_map.items():
        tmp = base[id_cols + [col]].rename(columns={col: "pair_similarity"})
        tmp["phase"] = phase
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.dropna(subset=["pair_similarity"])
    return long_df


def write_qc(analysis_df: pd.DataFrame, network_df: pd.DataFrame, trajectory_df: pd.DataFrame) -> None:
    qc_rows = []
    for name, df in [
        ("analysis_item", analysis_df),
        ("network_item", network_df),
        ("trajectory_long", trajectory_df),
    ]:
        item_collision = (
            df[["condition_item_id", "condition"]]
            .drop_duplicates()
            .groupby("condition_item_id")["condition"]
            .nunique()
            .gt(1)
            .sum()
            if "condition_item_id" in df and "condition" in df
            else np.nan
        )
        qc_rows.append(
            {
                "table": name,
                "n_rows": len(df),
                "n_subjects": df["subject"].nunique() if "subject" in df else np.nan,
                "n_items": df["condition_item_id"].nunique() if "condition_item_id" in df else np.nan,
                "n_rois": df["roi"].nunique() if "roi" in df else np.nan,
                "yy_items": df.loc[df["condition"].eq("yy"), "condition_item_id"].nunique()
                if "condition" in df
                else np.nan,
                "kj_items": df.loc[df["condition"].eq("kj"), "condition_item_id"].nunique()
                if "condition" in df
                else np.nan,
                "condition_item_cross_condition_collisions": item_collision,
                "pre_pair_similarity_z_present": "pre_pair_similarity_z" in df.columns,
                "memory_unique_values": ",".join(
                    map(str, sorted(pd.to_numeric(df["memory"], errors="coerce").dropna().unique()))
                )
                if "memory" in df
                else "",
            }
        )
    pd.DataFrame(qc_rows).to_csv(OUT_DIR / "mixed_model_table_qc.tsv", sep="\t", index=False)

    phase_qc = (
        trajectory_df.groupby(["roi_set", "roi", "condition", "phase"], dropna=False)
        .agg(
            n_rows=("pair_similarity", "size"),
            n_subjects=("subject", "nunique"),
            n_items=("condition_item_id", "nunique"),
            mean_similarity=("pair_similarity", "mean"),
        )
        .reset_index()
    )
    phase_qc.to_csv(OUT_DIR / "mixed_model_phase_qc.tsv", sep="\t", index=False)


def run_roi_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    covars = available_covars(df, MATERIAL_COVARS)
    covar_str = " + ".join(covars)
    covar_part = f" + {covar_str}" if covar_str else ""
    model_specs = [
        (
            "M1_post_edge_condition",
            "post_edge_specificity",
            "post_edge_specificity ~ C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + covar_part,
        ),
        (
            "M3_reactivation_to_post",
            "post_edge_specificity",
            "post_edge_specificity ~ react_pair_mean_avg_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + covar_part,
        ),
        (
            "M3_fluency_to_post",
            "post_edge_specificity",
            "post_edge_specificity ~ learning_fluency_shift_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + covar_part,
        ),
        (
            "M3_reactivation_negative_control",
            "pre_pair_similarity",
            "pre_pair_similarity ~ react_pair_mean_avg_z * C(condition, Treatment(reference='kj'))"
            + covar_part,
        ),
        (
            "M4_target_source_to_post",
            "post_edge_specificity",
            "post_edge_specificity ~ target_source_over_self_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + covar_part,
        ),
        (
            "M4_corrected_mapping_to_post",
            "post_edge_specificity",
            "post_edge_specificity ~ mapping_asymmetry_self_corrected_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + covar_part,
        ),
    ]
    rows, conv = [], []
    for (roi_set, roi), sub in df.groupby(["roi_set", "roi"], sort=False):
        for model_name, outcome, formula in model_specs:
            model_rows, info = fit_mixed_or_fallback(
                sub,
                formula,
                analysis_id=model_name.split("_")[0],
                model_name=model_name,
                roi_set=roi_set,
                roi=roi,
                outcome=outcome,
            )
            rows.extend(model_rows)
            conv.append(info)
    res = pd.DataFrame(rows)
    if not res.empty:
        res = fdr_by_family(res, ["model_name", "roi_set", "term"])
    return res, pd.DataFrame(conv)


def run_trajectory_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    covars = available_covars(df, MATERIAL_COVARS)
    covar_part = " + " + " + ".join(covars) if covars else ""
    formula = (
        "pair_similarity ~ C(phase, Treatment(reference='pre')) * "
        "C(condition, Treatment(reference='kj'))"
        + covar_part
    )
    rows, conv = [], []
    for (roi_set, roi), sub in df.groupby(["roi_set", "roi"], sort=False):
        model_rows, info = fit_mixed_or_fallback(
            sub,
            formula,
            analysis_id="M2",
            model_name="M2_four_phase_trajectory",
            roi_set=roi_set,
            roi=roi,
            outcome="pair_similarity",
        )
        rows.extend(model_rows)
        conv.append(info)
    res = pd.DataFrame(rows)
    if not res.empty:
        res = fdr_by_family(res, ["model_name", "roi_set", "term"])
    return res, pd.DataFrame(conv)


def run_network_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_specs = [
        (
            "M5_semantic_reactivation_to_hpc_separation",
            "hpc_spatial_separation",
            "hpc_spatial_separation ~ semantic_reactivation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
        ),
        (
            "M5_target_source_to_hpc_separation",
            "hpc_spatial_separation",
            "hpc_spatial_separation ~ semantic_target_source_over_self_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
        ),
        (
            "M5_corrected_mapping_to_hpc_separation",
            "hpc_spatial_separation",
            "hpc_spatial_separation ~ semantic_mapping_asymmetry_self_corrected_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
        ),
        (
            "M5_hpc_separation_to_retrieval_raw_shared_post",
            "retrieval_rebinding",
            "retrieval_rebinding ~ hpc_spatial_separation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z",
        ),
        (
            "M5_hpc_separation_to_retrieval_post_control",
            "retrieval_rebinding",
            "retrieval_rebinding ~ hpc_spatial_separation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z + hpc_spatial_post_similarity_z",
        ),
        (
            "M5_hpc_separation_to_retrieval_similarity",
            "retrieval_pair_similarity",
            "retrieval_pair_similarity ~ hpc_spatial_separation_z * C(condition, Treatment(reference='kj')) + pre_pair_similarity_z + hpc_spatial_post_similarity_z",
        ),
    ]
    rows, conv = [], []
    for model_name, outcome, formula in model_specs:
        model_rows, info = fit_mixed_or_fallback(
            df,
            formula,
            analysis_id="M5",
            model_name=model_name,
            roi_set="network",
            roi="semantic_to_hpc_spatial",
            outcome=outcome,
        )
        rows.extend(model_rows)
        conv.append(info)
    res = pd.DataFrame(rows)
    if not res.empty:
        res = fdr_by_family(res, ["model_name", "term"])
    return res, pd.DataFrame(conv)


def run_m6_check(network_df: pd.DataFrame) -> pd.DataFrame:
    mem = pd.to_numeric(network_df["memory"], errors="coerce").dropna()
    unique = sorted(mem.unique())
    binary = set(unique).issubset({0, 1})
    status = "ready_binary_memory" if binary else "skipped_memory_not_binary"
    message = (
        "memory contains non-binary values; logistic mixed model was not run"
        if not binary
        else "memory is binary; logistic mixed model can be added with glmer or BinomialBayesMixedGLM"
    )
    out = pd.DataFrame(
        [
            {
                "analysis_id": "M6",
                "status": status,
                "message": message,
                "n_rows": len(mem),
                "n_subjects": network_df["subject"].nunique(),
                "n_items": network_df["condition_item_id"].nunique(),
                "memory_unique_values": ",".join(map(str, unique)),
            }
        ]
    )
    return out


def write_review(
    roi_results: pd.DataFrame,
    traj_results: pd.DataFrame,
    network_results: pd.DataFrame,
    convergence: pd.DataFrame,
    m6_qc: pd.DataFrame,
) -> None:
    def df_to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        work = df.copy()
        for col in work.columns:
            if pd.api.types.is_float_dtype(work[col]):
                work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
            else:
                work[col] = work[col].fillna("").astype(str)
        header = "| " + " | ".join(work.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in work.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep] + body)

    def top_terms(df: pd.DataFrame, terms: list[str], n: int = 8) -> pd.DataFrame:
        if df.empty:
            return df
        mask = pd.Series(False, index=df.index)
        for t in terms:
            mask |= df["term"].str.contains(t, regex=False, na=False)
        return df.loc[mask].sort_values("q").head(n)

    m1 = top_terms(roi_results[roi_results["model_name"].eq("M1_post_edge_condition")], ["T.yy"], 10)
    m3 = top_terms(
        roi_results[roi_results["model_name"].isin(["M3_reactivation_to_post", "M3_fluency_to_post"])],
        [":C(condition", "react_pair_mean_avg_z", "learning_fluency_shift_z"],
        12,
    )
    m4 = top_terms(
        roi_results[roi_results["model_name"].isin(["M4_target_source_to_post", "M4_corrected_mapping_to_post"])],
        [":C(condition"],
        10,
    )
    m5 = network_results.sort_values("q").head(20) if not network_results.empty else network_results
    conv_summary = (
        convergence.groupby(["model_name", "status", "fallback_used"], dropna=False)
        .size()
        .reset_index(name="n_models")
        .sort_values(["model_name", "status"])
    )

    lines = [
        "# Mixed-Effects Model Review",
        "",
        "## Scope",
        "",
        "This analysis re-estimates key item-level effects with random intercepts for subject and condition-specific item.",
        "",
        "```text",
        "random structure: (1 | subject) + (1 | condition_item_id)",
        "implementation: Python statsmodels MixedLM; OLS cluster-subject fallback only when MixedLM fails",
        "```",
        "",
        "## Convergence",
        "",
        df_to_md(conv_summary),
        "",
        "## M1 Post Edge Differentiation",
        "",
        df_to_md(m1[["model_name", "roi_set", "roi", "term", "estimate", "se", "p", "q", "status", "fallback_used"]])
        if not m1.empty
        else "No M1 rows.",
        "",
        "## M3 Learning Reactivation / Fluency -> Post Edge",
        "",
        df_to_md(m3[["model_name", "roi_set", "roi", "term", "estimate", "se", "p", "q", "status", "fallback_used"]])
        if not m3.empty
        else "No M3 rows.",
        "",
        "## M4 Corrected Directional Mapping -> Post Edge",
        "",
        df_to_md(m4[["model_name", "roi_set", "roi", "term", "estimate", "se", "p", "q", "status", "fallback_used"]])
        if not m4.empty
        else "No M4 rows.",
        "",
        "## M5 Network Bridge",
        "",
        df_to_md(m5[["model_name", "term", "estimate", "se", "p", "q", "status", "fallback_used"]])
        if not m5.empty
        else "No M5 rows.",
        "",
        "## M6 Memory Logistic Check",
        "",
        df_to_md(m6_qc),
        "",
        "## Interpretation Guardrails",
        "",
        "- YY and KJ were not numerically paired; `condition_item_id` was used as the item random unit.",
        "- `pre_pair_similarity_z` came from neural pattern similarity columns in existing QC tables.",
        "- M6 was not run if memory was not binary at item level.",
        "- Mixed signs in directional mapping remain exploratory and should not be written as a unified assimilation mechanism.",
        "",
    ]
    (OUT_DIR / "mixed_effects_review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    analysis_df = load_analysis_table()
    network_df = load_network_table()
    trajectory_df = build_trajectory_table()
    write_qc(analysis_df, network_df, trajectory_df)

    # Persist compact analysis inputs for auditability.
    analysis_df.to_csv(OUT_DIR / "mixed_model_roi_item_table.tsv", sep="\t", index=False)
    network_df.to_csv(OUT_DIR / "mixed_model_network_item_table.tsv", sep="\t", index=False)
    trajectory_df.to_csv(OUT_DIR / "mixed_model_trajectory_long.tsv", sep="\t", index=False)

    roi_results, roi_conv = run_roi_models(analysis_df)
    traj_results, traj_conv = run_trajectory_models(trajectory_df)
    network_results, network_conv = run_network_models(network_df)
    m6_qc = run_m6_check(network_df)

    convergence = pd.concat([roi_conv, traj_conv, network_conv], ignore_index=True)

    roi_results.to_csv(OUT_DIR / "mixed_effects_roi_models.tsv", sep="\t", index=False)
    traj_results.to_csv(OUT_DIR / "mixed_effects_trajectory_models.tsv", sep="\t", index=False)
    network_results.to_csv(OUT_DIR / "mixed_effects_network_models.tsv", sep="\t", index=False)
    convergence.to_csv(OUT_DIR / "mixed_effects_convergence.tsv", sep="\t", index=False)
    m6_qc.to_csv(OUT_DIR / "mixed_effects_memory_logistic_qc.tsv", sep="\t", index=False)

    manifest = {
        "analysis_table": str(ANALYSIS_TABLE),
        "network_table": str(NETWORK_TABLE),
        "item_table": str(ITEM_TABLE),
        "output_dir": str(OUT_DIR),
        "n_roi_result_rows": int(len(roi_results)),
        "n_trajectory_result_rows": int(len(traj_results)),
        "n_network_result_rows": int(len(network_results)),
        "n_convergence_rows": int(len(convergence)),
        "m6_status": m6_qc.loc[0, "status"],
        "implementation": "statsmodels MixedLM with subject groups and item variance component",
    }
    (OUT_DIR / "mixed_effects_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_review(roi_results, traj_results, network_results, convergence, m6_qc)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
