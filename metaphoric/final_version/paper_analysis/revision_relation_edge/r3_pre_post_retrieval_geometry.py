#!/usr/bin/env python3
"""Task r3: pre-post-retrieval trajectory geometry."""

from __future__ import annotations

import pandas as pd

from shared_revision import MATERIAL_COVARS, fig_path, fit_ols_hc3, load_revision_master_network, numeric_cols_available, out_path, write_text, write_tsv

MODULE = "r3_pre_post_retrieval_geometry"


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    use = frame[columns].copy() if columns else frame.copy()
    if use.empty:
        return "_No rows._"
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            use[col] = use[col].fillna("").astype(str)
    header = "| " + " | ".join(use.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(use.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in use.to_numpy()]
    return "\n".join([header, sep, *rows])


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pre = pd.to_numeric(out["pre_pair_similarity"], errors="coerce")
    post = pd.to_numeric(out["post_pair_similarity"], errors="coerce")
    retrieval = pd.to_numeric(out["retrieval_pair_similarity"], errors="coerce")
    # These are scalar trajectory proxies from pair-similarity summaries, not voxel-pattern distances.
    out["d_pre_post"] = (post - pre).abs()
    out["d_post_retrieval"] = (retrieval - post).abs()
    out["d_pre_retrieval"] = (retrieval - pre).abs()
    out["post_drop_raw"] = pre - post
    out["retrieval_rebound_raw"] = retrieval - post
    out["return_to_pre_index"] = out["d_post_retrieval"] - out["d_pre_retrieval"]
    out["retrieval_new_state_index"] = out["d_pre_retrieval"] - out["d_post_retrieval"]
    out["triangle_area_proxy"] = (out["d_pre_post"] + out["d_post_retrieval"] + out["d_pre_retrieval"]) / 2
    return out


def fit_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network, sub in metrics.groupby("network"):
        covars = numeric_cols_available(sub, MATERIAL_COVARS, min_nonmissing=30)
        covar_text = (" + " + " + ".join(covars)) if covars else ""
        for outcome in ["d_pre_post", "d_post_retrieval", "d_pre_retrieval", "return_to_pre_index", "retrieval_new_state_index", "post_drop_raw", "retrieval_rebound_raw"]:
            tmp = sub.dropna(subset=[outcome, "memory_ord"]).copy()
            if len(tmp) < 50:
                rows.append(pd.DataFrame([{"network": network, "model": outcome, "term": "__model__", "status": "not_run_insufficient_rows", "n_obs": len(tmp)}]))
                continue
            formula = f"{outcome} ~ C(condition) * memory_ord{covar_text}"
            rows.append(fit_ols_hc3(tmp, formula, model=outcome, network=network))
    return pd.concat(rows, ignore_index=True, sort=False)


def vector_alignment_placeholder(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "condition", "condition_item_id", "network"]
    return metrics[keys].drop_duplicates().assign(
        trajectory_cosine=pd.NA,
        status="not_available_no_voxel_pattern_input_in_revision_master",
    )


def make_figure(metrics: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return

    desc = metrics.groupby(["network", "condition"], as_index=False).agg(
        d_pre_post=("d_pre_post", "mean"),
        d_post_retrieval=("d_post_retrieval", "mean"),
        d_pre_retrieval=("d_pre_retrieval", "mean"),
    )
    if desc.empty:
        return
    labels = []
    vals = []
    for _, row in desc.iterrows():
        for metric in ["d_pre_post", "d_post_retrieval", "d_pre_retrieval"]:
            labels.append(f"{row['network']} | {row['condition']} | {metric}")
            vals.append(row[metric])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(labels))))
    ax.barh(y, vals, color="#496f5d")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Distance proxy from pair similarities")
    ax.set_title("Pre-post-retrieval triangle metrics")
    fig.tight_layout()
    fig.savefig(fig_path("fig_pre_post_retrieval_triangle.png"), dpi=200)
    fig.savefig(fig_path("fig_pre_post_retrieval_triangle.pdf"))
    plt.close(fig)

    align = metrics.groupby(["network", "condition"], as_index=False).agg(
        post_drop_raw=("post_drop_raw", "mean"),
        retrieval_rebound_raw=("retrieval_rebound_raw", "mean"),
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for _, row in align.iterrows():
        ax.scatter(row["post_drop_raw"], row["retrieval_rebound_raw"], label=f"{row['network']} {row['condition']}", s=70)
    ax.axhline(0, color="0.7", lw=1)
    ax.axvline(0, color="0.7", lw=1)
    ax.set_xlabel("pre - post similarity")
    ax.set_ylabel("retrieval - post similarity")
    ax.set_title("Displacement direction proxy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path("fig_displacement_vector_alignment.png"), dpi=200)
    fig.savefig(fig_path("fig_displacement_vector_alignment.pdf"))
    plt.close(fig)


def review_md(metrics: pd.DataFrame, models: pd.DataFrame) -> str:
    desc = metrics.groupby(["network", "condition"], as_index=False).agg(
        d_pre_post=("d_pre_post", "mean"),
        d_post_retrieval=("d_post_retrieval", "mean"),
        d_pre_retrieval=("d_pre_retrieval", "mean"),
        post_drop_raw=("post_drop_raw", "mean"),
        retrieval_rebound_raw=("retrieval_rebound_raw", "mean"),
        n=("subject", "count"),
    )
    lines = [
        "# Trajectory geometry review",
        "",
        "Distance metrics are proxy metrics derived from available pair similarities, not voxel-level displacement vectors.",
        "Vector alignment is marked unavailable unless voxel pattern inputs are later added.",
        "",
        "## Descriptives",
        "",
        markdown_table(desc),
        "",
        "## Key model terms",
        "",
    ]
    key = models[models["term"].astype(str).str.contains("C\\(condition\\)|memory_ord", regex=True, na=False)].copy()
    if key.empty:
        lines.append("No key model terms were available.")
    else:
        lines.append(markdown_table(key, ["network", "model", "term", "estimate", "p", "q_bh", "status", "n_obs"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_revision_master_network()
    metrics = build_metrics(df)
    models = fit_models(metrics)
    align = vector_alignment_placeholder(metrics)
    write_tsv(metrics, out_path(MODULE, "trajectory_triangle_metrics.tsv"))
    write_tsv(align, out_path(MODULE, "trajectory_vector_alignment.tsv"))
    write_tsv(models, out_path(MODULE, "trajectory_geometry_models.tsv"))
    make_figure(metrics)
    write_text(review_md(metrics, models), out_path(MODULE, "trajectory_geometry_review.md"))
    print(f"Wrote r3 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
