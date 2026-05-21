#!/usr/bin/env python3
"""Task r1: material semantic distance / novelty moderation."""

from __future__ import annotations

import pandas as pd

from shared_revision import MATERIAL_COVARS, bh_fdr, fig_path, fit_ols_hc3, load_revision_master_network, numeric_cols_available, out_path, write_tsv

MODULE = "r1_material_distance_novelty_moderation"


def make_formula(outcome: str, frame: pd.DataFrame, *, yy_only: bool = False, quadratic: bool = False) -> str:
    covars = numeric_cols_available(frame, MATERIAL_COVARS, min_nonmissing=30)
    terms = []
    if yy_only:
        if "pre_pair_similarity_z" in frame.columns:
            terms.append("pre_pair_similarity_z")
        if "novelty_z" in covars:
            terms.append("novelty_z")
    else:
        terms.append("C(condition) * pre_pair_similarity_z")
        if "novelty_z" in covars and frame["novelty_z"].notna().sum() >= 30:
            terms.append("C(condition) * novelty_z")
    for covar in covars:
        if covar not in {"novelty_z"}:
            terms.append(covar)
    if quadratic:
        terms.append("I(pre_pair_similarity_z ** 2)")
        if not yy_only:
            terms.append("C(condition):I(pre_pair_similarity_z ** 2)")
    if not terms:
        terms = ["1"]
    return f"{outcome} ~ " + " + ".join(terms)


def fit_by_network(df: pd.DataFrame, outcome: str, model_name: str, *, yy_only: bool = False, quadratic: bool = False) -> pd.DataFrame:
    rows = []
    data = df.copy()
    if yy_only:
        data = data[data["condition"].eq("yy")].copy()
    for network, sub in data.groupby("network"):
        needed = [outcome, "pre_pair_similarity_z"]
        sub = sub.dropna(subset=[c for c in needed if c in sub.columns]).copy()
        if len(sub) < 50:
            rows.append(pd.DataFrame([{
                "network": network,
                "model": model_name,
                "term": "__model__",
                "status": "not_run_insufficient_rows",
                "n_obs": len(sub),
            }]))
            continue
        formula = make_formula(outcome, sub, yy_only=yy_only, quadratic=quadratic)
        rows.append(fit_ols_hc3(sub, formula, model=model_name, network=network))
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["q_bh_all_models"] = bh_fdr(out.get("p", pd.Series(dtype=float)))
    return out


def make_figure(memory: pd.DataFrame, post: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return

    terms = ["C(condition)[T.yy]:pre_pair_similarity_z", "C(condition)[T.yy]:novelty_z", "pre_pair_similarity_z", "novelty_z"]
    plot = pd.concat([
        memory.assign(outcome="memory_ord"),
        post.assign(outcome="post_separation_z"),
    ], ignore_index=True)
    plot = plot[plot["term"].isin(terms) & plot["status"].eq("ok")].copy()
    if plot.empty:
        return
    plot["lo"] = plot["estimate"] - 1.96 * plot["se"]
    plot["hi"] = plot["estimate"] + 1.96 * plot["se"]
    plot["label"] = plot["outcome"] + " | " + plot["network"] + " | " + plot["term"].str.replace("C(condition)[T.yy]:", "YY x ", regex=False)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(plot))))
    ax.axvline(0, color="0.6", lw=1)
    ax.errorbar(plot["estimate"], y, xerr=[plot["estimate"] - plot["lo"], plot["hi"] - plot["estimate"]], fmt="o", color="#2f5d7c")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=8)
    ax.set_xlabel("Estimate (OLS HC3 fallback)")
    ax.set_title("Material distance / novelty moderation")
    fig.tight_layout()
    fig.savefig(fig_path("fig_material_moderation_effects.png"), dpi=200)
    fig.savefig(fig_path("fig_material_moderation_effects.pdf"))
    plt.close(fig)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network, sub in df.groupby("network"):
        for condition, cell in sub.groupby("condition"):
            for col in ["pre_pair_similarity_z", "novelty_z", "familiarity_z", "comprehensibility_z", "difficulty_z", "wordfreq_z", "charlen_z"]:
                if col in cell.columns:
                    rows.append({
                        "network": network,
                        "condition": condition,
                        "field": col,
                        "n": len(cell),
                        "coverage": float(cell[col].notna().mean()),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_revision_master_network()
    memory = fit_by_network(df, "memory_ord", "memory_material_distance_novelty")
    post = fit_by_network(df, "post_separation_z", "post_separation_material_distance_novelty")
    yy_memory = fit_by_network(df, "memory_ord", "yy_only_memory_material", yy_only=True)
    yy_post = fit_by_network(df, "post_separation_z", "yy_only_post_separation_material", yy_only=True)
    quad = fit_by_network(df, "memory_ord", "memory_semantic_distance_quadratic", quadratic=True)
    yy = pd.concat([yy_memory, yy_post, quad], ignore_index=True, sort=False)
    write_tsv(memory, out_path(MODULE, "material_moderation_memory.tsv"))
    write_tsv(post, out_path(MODULE, "material_moderation_post_separation.tsv"))
    write_tsv(yy, out_path(MODULE, "yy_only_material_models.tsv"))
    write_tsv(coverage(df), out_path(MODULE, "material_covariate_coverage.tsv"))
    make_figure(memory, post)
    print(f"Wrote r1 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
