#!/usr/bin/env python3
"""Task r4: semantic-hpc network coupling revision."""

from __future__ import annotations

import pandas as pd

from shared_revision import MATERIAL_COVARS, fig_path, fit_ols_hc3, load_revision_master_network, numeric_cols_available, out_path, write_text, write_tsv, zscore

MODULE = "r4_network_coupling_revision"


def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "condition", "condition_item_id"]
    stage_cols = ["pre_pair_similarity_z", "post_pair_similarity_z", "retrieval_pair_similarity_z"]
    metric_cols = ["post_separation_z", "retrieval_rebound_z", "memory_ord", *MATERIAL_COVARS]
    wide = df.pivot_table(index=keys, columns="network", values=stage_cols + ["post_separation_z", "retrieval_rebound_z"], aggfunc="mean")
    wide.columns = [f"{net}_{metric}" for metric, net in wide.columns]
    wide = wide.reset_index()
    static = df.groupby(keys, as_index=False)[[c for c in ["memory_ord", *MATERIAL_COVARS] if c in df.columns]].mean(numeric_only=True)
    wide = wide.merge(static, on=keys, how="left")
    for stage in ["pre", "post", "retrieval"]:
        sem = f"semantic_{stage}_pair_similarity_z"
        hpc = f"hpc_spatial_{stage}_pair_similarity_z"
        out = f"semantic_hpc_coupling_{stage}"
        if sem in wide.columns and hpc in wide.columns:
            wide[out] = wide[sem] * wide[hpc]
            wide[out + "_z"] = zscore(wide[out])
    if "semantic_retrieval_rebound_z" in wide.columns and "hpc_spatial_retrieval_rebound_z" in wide.columns:
        wide["retrieval_rebound_coupling"] = wide["semantic_retrieval_rebound_z"] * wide["hpc_spatial_retrieval_rebound_z"]
        wide["retrieval_rebound_coupling_z"] = zscore(wide["retrieval_rebound_coupling"])
    if "semantic_post_separation_z" in wide.columns and "hpc_spatial_post_separation_z" in wide.columns:
        wide["post_separation_coupling"] = wide["semantic_post_separation_z"] * wide["hpc_spatial_post_separation_z"]
        wide["post_separation_coupling_z"] = zscore(wide["post_separation_coupling"])
    return wide


def stage_long(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ["pre", "post", "retrieval"]:
        col = f"semantic_hpc_coupling_{stage}_z"
        if col not in wide.columns:
            continue
        temp = wide[["subject", "condition", "condition_item_id", col]].copy()
        temp["stage"] = stage
        temp = temp.rename(columns={col: "network_coupling_z"})
        rows.append(temp)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def fit_stage_model(stage: pd.DataFrame) -> pd.DataFrame:
    if stage.empty:
        return pd.DataFrame([{"model": "stage_coupling", "term": "__model__", "status": "not_run_no_stage_coupling"}])
    return fit_ols_hc3(stage.dropna(subset=["network_coupling_z"]), "network_coupling_z ~ C(condition) * C(stage)", model="stage_coupling", network="semantic_hpc")


def fit_rebound_model(wide: pd.DataFrame) -> pd.DataFrame:
    covars = numeric_cols_available(wide, MATERIAL_COVARS, min_nonmissing=30)
    covar_text = (" + " + " + ".join(covars)) if covars else ""
    if not {"hpc_spatial_retrieval_rebound_z", "semantic_retrieval_rebound_z"}.issubset(wide.columns):
        return pd.DataFrame([{"model": "retrieval_rebound_coupling", "term": "__model__", "status": "not_run_missing_rebound_columns"}])
    data = wide.dropna(subset=["hpc_spatial_retrieval_rebound_z", "semantic_retrieval_rebound_z"]).copy()
    return fit_ols_hc3(data, f"hpc_spatial_retrieval_rebound_z ~ semantic_retrieval_rebound_z * C(condition){covar_text}", model="retrieval_rebound_coupling", network="semantic_to_hpc")


def fit_memory_model(wide: pd.DataFrame) -> pd.DataFrame:
    covars = numeric_cols_available(wide, MATERIAL_COVARS, min_nonmissing=30)
    covar_text = (" + " + " + ".join(covars)) if covars else ""
    needed = ["memory_ord", "semantic_hpc_coupling_retrieval_z", "semantic_hpc_coupling_post_z"]
    if not set(needed).issubset(wide.columns):
        return pd.DataFrame([{"model": "memory_coupling", "term": "__model__", "status": "not_run_missing_coupling_columns"}])
    data = wide.dropna(subset=needed).copy()
    formula = f"memory_ord ~ C(condition) * semantic_hpc_coupling_retrieval_z + C(condition) * semantic_hpc_coupling_post_z{covar_text}"
    return fit_ols_hc3(data, formula, model="memory_coupling", network="semantic_hpc")


def make_figure(stage: pd.DataFrame, wide: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return

    if not stage.empty:
        desc = stage.groupby(["condition", "stage"], as_index=False).agg(mean=("network_coupling_z", "mean"), sem=("network_coupling_z", "sem"))
        labels = desc["condition"] + " " + desc["stage"]
        y = np.arange(len(desc))
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(desc))))
        ax.barh(y, desc["mean"], xerr=desc["sem"], color="#315f72", alpha=0.85)
        ax.axvline(0, color="0.7", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("semantic-hpc coupling z")
        ax.set_title("Stage-wise semantic-hpc coupling")
        fig.tight_layout()
        fig.savefig(fig_path("fig_network_coupling_revision.png"), dpi=200)
        fig.savefig(fig_path("fig_network_coupling_revision.pdf"))
        plt.close(fig)


def review(stage_model: pd.DataFrame, rebound: pd.DataFrame, memory: pd.DataFrame) -> str:
    lines = [
        "# Network coupling revision review",
        "",
        "Analyses are restricted to semantic/metaphor and hpc-spatial network composites.",
        "Coupling is a predefined item-level product proxy from network z-scored metrics; it is not causal communication.",
        "",
        "## Stage model key terms",
        stage_model.to_csv(sep="\t", index=False),
        "",
        "## Retrieval rebound coupling",
        rebound.to_csv(sep="\t", index=False),
        "",
        "## Memory coupling",
        memory.to_csv(sep="\t", index=False),
    ]
    return "\n".join(lines)


def main() -> None:
    df = load_revision_master_network()
    wide = build_wide(df)
    stage = stage_long(wide)
    stage_model = fit_stage_model(stage)
    rebound = fit_rebound_model(wide)
    memory = fit_memory_model(wide)
    write_tsv(wide, out_path(MODULE, "network_coupling_item_metrics.tsv"))
    write_tsv(stage, out_path(MODULE, "network_coupling_stage_long.tsv"))
    write_tsv(stage_model, out_path(MODULE, "network_coupling_stage_models.tsv"))
    write_tsv(memory, out_path(MODULE, "network_coupling_memory_models.tsv"))
    write_tsv(rebound, out_path(MODULE, "network_coupling_retrieval_rebound_models.tsv"))
    write_tsv(pd.DataFrame([{"status": "not_run", "reason": "HPC long-axis coupling remains exploratory and is not required for the main revision pass."}]), out_path(MODULE, "hpc_long_axis_coupling_exploratory.tsv"))
    make_figure(stage, wide)
    write_text(review(stage_model, rebound, memory), out_path(MODULE, "network_coupling_revision_review.md"))
    print(f"Wrote r4 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
