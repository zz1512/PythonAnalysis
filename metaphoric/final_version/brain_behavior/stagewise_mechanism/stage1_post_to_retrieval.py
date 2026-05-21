from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


QC_ROOT = Path("E:/python_metaphor/paper_outputs/qc")
OUT_DIR = QC_ROOT / "stagewise_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TABLE = (
    QC_ROOT
    / "learning_post_memory_prediction"
    / "item_mechanism_table.tsv"
)

MATERIAL_COVARS = [
    "sentence_char_len_z",
    "word_frequency_mean_z",
    "stroke_count_mean_z",
    "valence_mean_z",
    "arousal_mean_z",
]

HPC_SPATIAL_ROI_SET = "meta_spatial"


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def fdr_by_family(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["q"] = np.nan
    if df.empty:
        return df
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        p = pd.to_numeric(df.loc[idx, "p"], errors="coerce")
        ok = p.notna()
        if ok.any():
            _, q, _, _ = multipletests(p[ok], method="fdr_bh")
            df.loc[p[ok].index, "q"] = q
    return df


def load_item_table() -> pd.DataFrame:
    columns = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "original_pair_id",
        "template_pair_id",
        "condition_item_id",
        "post_edge_specificity",
        "trained_edge_drop",
        "pre_pair_similarity",
        "post_pair_similarity",
        "retrieval_minus_post_pair_similarity",
        "retrieval_pair_similarity",
        "memory",
        "pre_pair_similarity_z",
        "post_pair_similarity_z",
        "post_edge_specificity_z",
        "trained_edge_drop_z",
        *MATERIAL_COVARS,
    ]
    df = pd.read_csv(
        INPUT_TABLE,
        sep="\t",
        usecols=lambda c: c in columns,
        low_memory=False,
    )
    numeric_cols = [c for c in df.columns if c not in {"subject", "roi_set", "roi", "condition", "condition_item_id"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"retrieval_minus_post_pair_similarity": "retrieval_rebinding"})
    return df


def build_network_table(df: pd.DataFrame) -> pd.DataFrame:
    hpc = df[df["roi_set"].eq(HPC_SPATIAL_ROI_SET)].copy()
    group_cols = [
        "subject",
        "condition",
        "original_pair_id",
        "template_pair_id",
        "condition_item_id",
    ]
    agg = {
        "post_edge_specificity": "mean",
        "trained_edge_drop": "mean",
        "retrieval_rebinding": "mean",
        "retrieval_pair_similarity": "mean",
        "pre_pair_similarity": "mean",
        "post_pair_similarity": "mean",
        "memory": "first",
        **{c: "first" for c in MATERIAL_COVARS if c in hpc.columns},
        "roi": "nunique",
    }
    net = hpc.groupby(group_cols, dropna=False).agg(agg).reset_index()
    net = net.rename(
        columns={
            "post_edge_specificity": "hpc_spatial_post_separation",
            "trained_edge_drop": "hpc_spatial_trained_drop",
            "retrieval_rebinding": "hpc_spatial_rebinding",
            "pre_pair_similarity": "hpc_spatial_pre_similarity",
            "post_pair_similarity": "hpc_spatial_post_similarity",
            "roi": "n_hpc_spatial_rois",
        }
    )
    net["roi_set"] = "network"
    net["roi"] = "hpc_spatial_composite"
    net["hpc_spatial_post_separation_z"] = zscore(net["hpc_spatial_post_separation"])
    net["hpc_spatial_trained_drop_z"] = zscore(net["hpc_spatial_trained_drop"])
    net["hpc_spatial_pre_similarity_z"] = zscore(net["hpc_spatial_pre_similarity"])
    net["hpc_spatial_post_similarity_z"] = zscore(net["hpc_spatial_post_similarity"])
    net["hpc_spatial_rebinding_z"] = zscore(net["hpc_spatial_rebinding"])
    return net


def formula_columns(formula: str, data: pd.DataFrame, outcome: str) -> list[str]:
    cols = ["subject", "condition_item_id", "condition", outcome]
    for col in data.columns:
        if col in formula:
            cols.append(col)
    return sorted(set(c for c in cols if c in data.columns))


def term_rows(
    result,
    *,
    model_name: str,
    roi_set: str,
    roi: str,
    outcome: str,
    formula: str,
    data: pd.DataFrame,
    status: str,
    converged: bool,
    fallback_used: str,
) -> list[dict]:
    rows: list[dict] = []
    for term, estimate in result.params.items():
        if term.endswith(" Var") or term.startswith("item Var"):
            continue
        se = result.bse.get(term, np.nan)
        p = result.pvalues.get(term, np.nan)
        stat = estimate / se if pd.notna(se) and se != 0 else np.nan
        rows.append(
            {
                "analysis_id": "stage1_post_to_retrieval",
                "model_name": model_name,
                "roi_set": roi_set,
                "roi": roi,
                "outcome": outcome,
                "term": term,
                "estimate": estimate,
                "se": se,
                "stat": stat,
                "p": p,
                "n_rows": len(data),
                "n_subjects": data["subject"].nunique(),
                "n_items": data["condition_item_id"].nunique(),
                "formula": formula,
                "status": status,
                "converged": converged,
                "fallback_used": fallback_used,
            }
        )
    return rows


def fit_model(
    data: pd.DataFrame,
    *,
    formula: str,
    model_name: str,
    roi_set: str,
    roi: str,
    outcome: str,
) -> tuple[list[dict], dict]:
    needed = formula_columns(formula, data, outcome)
    model_data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    model_data["subject"] = model_data["subject"].astype(str)
    model_data["condition"] = model_data["condition"].astype(str)
    model_data["condition_item_id"] = model_data["condition_item_id"].astype(str)

    info = {
        "analysis_id": "stage1_post_to_retrieval",
        "model_name": model_name,
        "roi_set": roi_set,
        "roi": roi,
        "outcome": outcome,
        "n_rows": len(model_data),
        "n_subjects": model_data["subject"].nunique() if "subject" in model_data else 0,
        "n_items": model_data["condition_item_id"].nunique() if "condition_item_id" in model_data else 0,
        "status": "not_run",
        "converged": False,
        "fallback_used": "none",
        "message": "",
    }
    if len(model_data) < 50 or info["n_subjects"] < 5 or info["n_items"] < 10:
        info["status"] = "skipped_too_few_rows"
        return [], info

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            md = smf.mixedlm(
                formula,
                model_data,
                groups=model_data["subject"],
                vc_formula={"item": "0 + C(condition_item_id)"},
            )
            res = md.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            info["converged"] = bool(getattr(res, "converged", False))
            info["status"] = "ok" if info["converged"] else "mixed_not_converged"
        except Exception as exc:
            info["fallback_used"] = "ols_cluster_subject"
            info["message"] = repr(exc)
            try:
                res = smf.ols(formula, model_data).fit(
                    cov_type="cluster", cov_kwds={"groups": model_data["subject"]}
                )
                info["converged"] = True
                info["status"] = "fallback_ols_cluster_subject"
            except Exception as exc2:
                info["status"] = "failed"
                info["message"] = f"{info['message']}; fallback={repr(exc2)}"
                return [], info
        if caught:
            warnings_text = " | ".join(sorted({str(w.message)[:180] for w in caught}))
            info["message"] = (
                f"{info['message']} | {warnings_text}" if info["message"] else warnings_text
            )

    rows = term_rows(
        res,
        model_name=model_name,
        roi_set=roi_set,
        roi=roi,
        outcome=outcome,
        formula=formula,
        data=model_data,
        status=info["status"],
        converged=info["converged"],
        fallback_used=info["fallback_used"],
    )
    return rows, info


def write_qc(item_df: pd.DataFrame, network_df: pd.DataFrame) -> None:
    qc_rows = []
    for name, df in [("roi_item", item_df), ("hpc_spatial_network", network_df)]:
        collisions = (
            df[["condition_item_id", "condition"]]
            .drop_duplicates()
            .groupby("condition_item_id")["condition"]
            .nunique()
            .gt(1)
            .sum()
        )
        retrieval_rows = df["retrieval_rebinding"].notna().sum() if "retrieval_rebinding" in df else df["hpc_spatial_rebinding"].notna().sum()
        qc_rows.append(
            {
                "table": name,
                "n_rows": len(df),
                "n_rows_with_retrieval": retrieval_rows,
                "n_subjects": df["subject"].nunique(),
                "n_subjects_with_retrieval": df.loc[
                    df["retrieval_rebinding"].notna()
                    if "retrieval_rebinding" in df
                    else df["hpc_spatial_rebinding"].notna(),
                    "subject",
                ].nunique(),
                "n_items": df["condition_item_id"].nunique(),
                "yy_items": df.loc[df["condition"].eq("yy"), "condition_item_id"].nunique(),
                "kj_items": df.loc[df["condition"].eq("kj"), "condition_item_id"].nunique(),
                "n_rois": df["roi"].nunique(),
                "condition_item_cross_condition_collisions": int(collisions),
                "pre_pair_similarity_z_present": "pre_pair_similarity_z" in df.columns
                or "hpc_spatial_pre_similarity_z" in df.columns,
            }
        )
    pd.DataFrame(qc_rows).to_csv(OUT_DIR / "stage1_post_to_retrieval_qc.tsv", sep="\t", index=False)

    process = (
        item_df.dropna(subset=["retrieval_rebinding"])
        .groupby(["roi_set", "roi", "condition"], dropna=False)
        .agg(
            n_rows=("retrieval_rebinding", "size"),
            n_subjects=("subject", "nunique"),
            n_items=("condition_item_id", "nunique"),
            post_edge_specificity_mean=("post_edge_specificity", "mean"),
            retrieval_rebinding_mean=("retrieval_rebinding", "mean"),
            trained_edge_drop_mean=("trained_edge_drop", "mean"),
            pre_pair_similarity_mean=("pre_pair_similarity", "mean"),
            post_pair_similarity_mean=("post_pair_similarity", "mean"),
        )
        .reset_index()
    )
    process.to_csv(OUT_DIR / "stage1_post_to_retrieval_process.tsv", sep="\t", index=False)


def run_roi_models(item_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    covars = " + ".join(MATERIAL_COVARS)
    covar_part = f" + {covars}" if covars else ""
    specs = [
        (
            "roi_raw_shared_post_specificity_to_rebinding",
            "retrieval_rebinding",
            "retrieval_rebinding ~ post_edge_specificity_z * C(condition, Treatment(reference='kj'))"
            + " + pre_pair_similarity_z"
            + covar_part,
        ),
        (
            "roi_post_control_specificity_to_rebinding",
            "retrieval_rebinding",
            "retrieval_rebinding ~ post_edge_specificity_z * C(condition, Treatment(reference='kj'))"
            + " + pre_pair_similarity_z + post_pair_similarity_z"
            + covar_part,
        ),
        (
            "roi_specificity_to_retrieval_similarity",
            "retrieval_pair_similarity",
            "retrieval_pair_similarity ~ post_edge_specificity_z * C(condition, Treatment(reference='kj'))"
            + " + pre_pair_similarity_z + post_pair_similarity_z"
            + covar_part,
        ),
        (
            "roi_raw_shared_post_trained_drop_to_rebinding",
            "retrieval_rebinding",
            "retrieval_rebinding ~ trained_edge_drop_z * C(condition, Treatment(reference='kj'))"
            + " + pre_pair_similarity_z"
            + covar_part,
        ),
    ]
    rows, conv = [], []
    for (roi_set, roi), sub in item_df.groupby(["roi_set", "roi"], sort=False):
        for model_name, outcome, formula in specs:
            model_rows, info = fit_model(
                sub,
                formula=formula,
                model_name=model_name,
                roi_set=roi_set,
                roi=roi,
                outcome=outcome,
            )
            rows.extend(model_rows)
            conv.append(info)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = fdr_by_family(results, ["model_name", "roi_set", "term"])
    return results, pd.DataFrame(conv)


def run_network_models(network_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    covars = " + ".join([c for c in MATERIAL_COVARS if c in network_df.columns])
    covar_part = f" + {covars}" if covars else ""
    specs = [
        (
            "network_raw_shared_post_hpc_separation_to_rebinding",
            "hpc_spatial_rebinding",
            "hpc_spatial_rebinding ~ hpc_spatial_post_separation_z * C(condition, Treatment(reference='kj'))"
            + " + hpc_spatial_pre_similarity_z"
            + covar_part,
        ),
        (
            "network_post_control_hpc_separation_to_rebinding",
            "hpc_spatial_rebinding",
            "hpc_spatial_rebinding ~ hpc_spatial_post_separation_z * C(condition, Treatment(reference='kj'))"
            + " + hpc_spatial_pre_similarity_z + hpc_spatial_post_similarity_z"
            + covar_part,
        ),
        (
            "network_hpc_separation_to_retrieval_similarity",
            "retrieval_pair_similarity",
            "retrieval_pair_similarity ~ hpc_spatial_post_separation_z * C(condition, Treatment(reference='kj'))"
            + " + hpc_spatial_pre_similarity_z + hpc_spatial_post_similarity_z"
            + covar_part,
        ),
        (
            "network_raw_shared_post_hpc_trained_drop_to_rebinding",
            "hpc_spatial_rebinding",
            "hpc_spatial_rebinding ~ hpc_spatial_trained_drop_z * C(condition, Treatment(reference='kj'))"
            + " + hpc_spatial_pre_similarity_z"
            + covar_part,
        ),
    ]
    rows, conv = [], []
    for model_name, outcome, formula in specs:
        model_rows, info = fit_model(
            network_df,
            formula=formula,
            model_name=model_name,
            roi_set="network",
            roi="hpc_spatial_composite",
            outcome=outcome,
        )
        rows.extend(model_rows)
        conv.append(info)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = fdr_by_family(results, ["model_name", "term"])
    return results, pd.DataFrame(conv)


def make_review(
    roi_results: pd.DataFrame,
    network_results: pd.DataFrame,
    convergence: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "No rows."
        work = df[cols].copy()
        for col in work.columns:
            if pd.api.types.is_numeric_dtype(work[col]):
                work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
            else:
                work[col] = work[col].fillna("").astype(str)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in work.astype(str).itertuples(index=False, name=None)]
        return "\n".join([header, sep] + body)

    conv_summary = (
        convergence.groupby(["model_name", "status", "fallback_used"], dropna=False)
        .size()
        .reset_index(name="n_models")
    )
    primary_terms = [
        "post_edge_specificity_z",
        "post_edge_specificity_z:C(condition, Treatment(reference='kj'))[T.yy]",
    ]
    roi_primary = roi_results[
        roi_results["model_name"].isin(
            [
                "roi_raw_shared_post_specificity_to_rebinding",
                "roi_post_control_specificity_to_rebinding",
                "roi_specificity_to_retrieval_similarity",
            ]
        )
        & roi_results["term"].isin(primary_terms)
    ].sort_values("q")
    network_top = network_results.sort_values("q")

    lines = [
        "# Stage 1 Review: Post Differentiation -> Run7 Rebinding Coupling",
        "",
        "## Scope",
        "",
        "Stage 1 tests whether item-level post-stage edge separation predicts run7 retrieval re-binding.",
        "",
        "Important design warning: the raw difference metrics `pre - post` and `retrieval - post` share the same post term. Therefore, raw models are reported for continuity, but post-controlled models are the inferential priority.",
        "",
        "Primary ROI model:",
        "",
        "```text",
        "retrieval_rebinding ~ post_edge_specificity_z * condition",
        "                    + pre_pair_similarity_z + material_covariates",
        "                    + (1 | subject) + (1 | condition_item_id)",
        "```",
        "",
        "Post-controlled sensitivity:",
        "",
        "```text",
        "retrieval_rebinding ~ post_edge_specificity_z * condition",
        "                    + pre_pair_similarity_z + post_pair_similarity_z + material_covariates",
        "retrieval_pair_similarity ~ post_edge_specificity_z * condition",
        "                          + pre_pair_similarity_z + post_pair_similarity_z + material_covariates",
        "```",
        "",
        "Network model uses the predeclared hpc-spatial composite built from `meta_spatial` ROIs.",
        "",
        "## QC",
        "",
        md_table(qc, list(qc.columns)),
        "",
        "## Convergence",
        "",
        md_table(conv_summary, list(conv_summary.columns)),
        "",
        "## Primary ROI Results",
        "",
        md_table(
            roi_primary.head(20),
            ["model_name", "roi_set", "roi", "term", "estimate", "se", "p", "q", "status"],
        ),
        "",
        "## Network Results",
        "",
        md_table(
            network_top,
            ["model_name", "term", "estimate", "se", "p", "q", "status"],
        ),
        "",
        "## Interpretation Checklist",
        "",
        "- Raw difference models are vulnerable to shared-post coupling; prioritize post-controlled and retrieval-similarity models.",
        "- If the post-controlled main slope is positive and stable, post separation is coupled with run7 re-binding beyond shared post similarity.",
        "- If the YY interaction is not significant, the coupling should be written as general rather than YY-specific.",
        "- If the network model is significant but ROI-wise models are mixed, emphasize the predeclared hpc-spatial composite.",
        "- If neither model is significant, keep the existing condition-level bridge but do not claim item-level coupling.",
        "",
    ]
    (OUT_DIR / "stage1_post_to_retrieval_review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    item_df = load_item_table()
    network_df = build_network_table(item_df)

    item_df.to_csv(OUT_DIR / "stage1_post_to_retrieval_item.tsv", sep="\t", index=False)
    network_df.to_csv(OUT_DIR / "stage1_post_to_retrieval_network_item.tsv", sep="\t", index=False)
    write_qc(item_df, network_df)

    roi_results, roi_conv = run_roi_models(item_df)
    network_results, network_conv = run_network_models(network_df)
    convergence = pd.concat([roi_conv, network_conv], ignore_index=True)

    roi_results.to_csv(OUT_DIR / "stage1_post_to_retrieval_models.tsv", sep="\t", index=False)
    network_results.to_csv(OUT_DIR / "stage1_post_to_retrieval_network_models.tsv", sep="\t", index=False)
    convergence.to_csv(OUT_DIR / "stage1_post_to_retrieval_convergence.tsv", sep="\t", index=False)

    qc = pd.read_csv(OUT_DIR / "stage1_post_to_retrieval_qc.tsv", sep="\t")
    make_review(roi_results, network_results, convergence, qc)

    manifest = {
        "analysis_id": "stage1_post_to_retrieval",
        "input_table": str(INPUT_TABLE),
        "output_dir": str(OUT_DIR),
        "n_item_rows": int(len(item_df)),
        "n_network_rows": int(len(network_df)),
        "n_roi_model_rows": int(len(roi_results)),
        "n_network_model_rows": int(len(network_results)),
        "n_convergence_rows": int(len(convergence)),
        "primary_model": "post-controlled retrieval_rebinding/retrieval_similarity models; raw shared-post model reported as continuity only",
    }
    (OUT_DIR / "stage1_post_to_retrieval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
