#!/usr/bin/env python3
"""Task r2: profile YY items with high vs low post-stage separation."""

from __future__ import annotations

import pandas as pd

from shared_revision import EXTRA_ITEM_COVARS, MATERIAL_COVARS, fig_path, fit_ols_hc3, load_revision_master_network, mean_sem, out_path, write_tsv

MODULE = "r2_high_low_post_separation_item_profile"


FEATURES = [
    "pre_pair_similarity_z",
    "novelty_z",
    "familiarity_z",
    "comprehensibility_z",
    "difficulty_z",
    "wordfreq_z",
    "charlen_z",
    *EXTRA_ITEM_COVARS,
    "run3_understand",
    "run4_like",
    "run3_rt",
    "run4_rt",
    "run3_rt_z",
    "run4_rt_z",
    "learning_fluency_shift",
    "memory_ord",
]


def assign_groups(df: pd.DataFrame, split: str) -> pd.DataFrame:
    item = df[df["condition"].eq("yy")].groupby(["network", "condition_item_id"], as_index=False).agg(
        post_separation_z=("post_separation_z", "mean")
    )
    groups = []
    for network, sub in item.groupby("network"):
        if split == "top_bottom_30":
            lo = sub["post_separation_z"].quantile(0.30)
            hi = sub["post_separation_z"].quantile(0.70)
            temp = sub[(sub["post_separation_z"] <= lo) | (sub["post_separation_z"] >= hi)].copy()
            temp["separation_group"] = temp["post_separation_z"].where(temp["post_separation_z"] >= hi, other=pd.NA)
            temp["separation_group"] = temp["separation_group"].notna().map({True: "high", False: "low"})
        else:
            med = sub["post_separation_z"].median()
            temp = sub.copy()
            temp["separation_group"] = (temp["post_separation_z"] >= med).map({True: "high", False: "low"})
        temp["split"] = split
        groups.append(temp)
    return pd.concat(groups, ignore_index=True, sort=False)


def profile_table(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = df.merge(groups[["network", "condition_item_id", "separation_group", "split"]], on=["network", "condition_item_id"], how="inner")
    item_mean = merged.groupby(["network", "condition_item_id", "separation_group", "split"], as_index=False)[
        [c for c in FEATURES if c in merged.columns] + ["post_separation_z"]
    ].mean(numeric_only=True)
    for (network, split, group), sub in item_mean.groupby(["network", "split", "separation_group"]):
        for feature in [c for c in FEATURES + ["post_separation_z"] if c in item_mean.columns]:
            mean, sem, n = mean_sem(sub[feature])
            rows.append({"network": network, "split": split, "separation_group": group, "feature": feature, "mean": mean, "sem": sem, "n_items": n})
    return pd.DataFrame(rows), item_mean


def tests(item_mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (network, split), sub in item_mean.groupby(["network", "split"]):
        for feature in [c for c in FEATURES if c in sub.columns]:
            tmp = sub.dropna(subset=[feature, "separation_group"]).copy()
            if tmp["separation_group"].nunique() < 2 or len(tmp) < 10:
                rows.append(pd.DataFrame([{"network": network, "model": f"{split}_{feature}", "term": "__model__", "status": "not_run_insufficient_groups", "n_obs": len(tmp)}]))
            else:
                rows.append(fit_ols_hc3(tmp, f"{feature} ~ C(separation_group)", model=f"{split}_{feature}", network=network))
    return pd.concat(rows, ignore_index=True, sort=False)


def make_figure(profile: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return

    keep = profile[(profile["split"].eq("top_bottom_30")) & (profile["feature"].isin(["pre_pair_similarity_z", "novelty_z", "difficulty_z", "memory_ord", "post_separation_z"]))].copy()
    if keep.empty:
        return
    labels = []
    x = []
    err = []
    colors = []
    for _, row in keep.iterrows():
        labels.append(f"{row['network']} | {row['feature']} | {row['separation_group']}")
        x.append(row["mean"])
        err.append(row["sem"] if pd.notna(row["sem"]) else 0)
        colors.append("#7c4d3a" if row["separation_group"] == "high" else "#597a8c")
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(labels))))
    ax.barh(y, x, xerr=err, color=colors, alpha=0.85)
    ax.axvline(0, color="0.6", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Item mean")
    ax.set_title("High vs low YY post-separation item profile")
    fig.tight_layout()
    fig.savefig(fig_path("fig_high_low_separation_profile.png"), dpi=200)
    fig.savefig(fig_path("fig_high_low_separation_profile.pdf"))
    plt.close(fig)


def main() -> None:
    df = load_revision_master_network()
    groups = pd.concat([assign_groups(df, "top_bottom_30"), assign_groups(df, "median_split")], ignore_index=True, sort=False)
    profile, item_mean = profile_table(df, groups)
    test = tests(item_mean)
    write_tsv(profile, out_path(MODULE, "high_low_post_separation_item_profile.tsv"))
    write_tsv(test, out_path(MODULE, "high_low_post_separation_feature_tests.tsv"))
    write_tsv(item_mean, out_path(MODULE, "high_low_post_separation_item_means.tsv"))
    make_figure(profile)
    print(f"Wrote r2 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
