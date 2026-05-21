#!/usr/bin/env python3
"""Plot top subject-level relation-vector brain-behavior associations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from figure_utils import (
    abbreviate_roi_name,
    add_panel_label,
    apply_publication_rcparams,
    default_paper_output_root,
    save_png_pdf,
)


MODEL_LABELS = {
    "M9_relation_vector_direct": "direct",
    "M9_relation_vector_abs": "absolute",
}
BEHAVIOR_LABELS = {
    "run7_memory_accuracy": "memory accuracy",
    "run7_correct_retrieval_efficiency": "retrieval efficiency",
    "run7_correct_retrieval_speed": "retrieval speed",
}
PREDICTOR_LABELS = {
    "relation_pre": "pre alignment",
    "relation_post": "post alignment",
    "relation_delta": "delta alignment",
    "relation_decoupling": "decoupling",
}
CONDITION_LABELS = {"yy": "YY", "kj": "KJ", "yy_minus_kj": "YY-KJ"}


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _select_top(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frame = summary[
        summary["is_primary_model"].astype(str).str.lower().isin(["true", "1"])
        & summary["is_primary_behavior"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    frame["spearman_p"] = pd.to_numeric(frame["spearman_p"], errors="coerce")
    frame["q_bh_primary_family"] = pd.to_numeric(frame["q_bh_primary_family"], errors="coerce")
    frame["rank_value"] = frame["q_bh_primary_family"].fillna(1.0) * 1000 + frame["spearman_p"].fillna(1.0)
    return frame.sort_values(["rank_value", "spearman_p"]).head(top_n).reset_index(drop=True)


def _panel_title(row: pd.Series) -> str:
    roi = abbreviate_roi_name(row["roi"])
    condition = CONDITION_LABELS.get(str(row["condition_group"]), str(row["condition_group"]).upper())
    model = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
    predictor = PREDICTOR_LABELS.get(str(row["predictor_name"]), str(row["predictor_name"]))
    behavior = BEHAVIOR_LABELS.get(str(row["behavior_metric"]), str(row["behavior_metric"]))
    q_value = pd.to_numeric(row.get("q_bh_primary_family"), errors="coerce")
    q_text = "q=NA" if pd.isna(q_value) else f"q={q_value:.3f}"
    return f"{roi} | {condition} | {model}\n{predictor} -> {behavior} ({q_text})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot relation-vector brain-behavior candidate associations.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    summary_path = paper_root / "tables_main" / "table_relation_behavior_prediction.tsv"
    subject_path = paper_root / "qc" / "relation_behavior" / "relation_behavior_subject_long.tsv"
    out_dir = paper_root / "figures_main"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_table(summary_path)
    subject = _read_table(subject_path)
    top = _select_top(summary, args.top_n)
    top.to_csv(out_dir / "table_relation_behavior_prediction_top.tsv", sep="\t", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    apply_publication_rcparams()
    n = len(top)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No primary relation-behavior rows", ha="center", va="center")
        ax.axis("off")
        save_png_pdf(fig, out_dir / "fig_relation_behavior_prediction.png")
        plt.close(fig)
        return

    n_cols = 3 if n > 2 else n
    n_rows = int((n + n_cols - 1) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, row in top.iterrows():
        ax = axes.ravel()[idx]
        ax.set_visible(True)
        subset = subject[
            subject["analysis_type"].eq(row["analysis_type"])
            & subject["roi_set"].eq(row["roi_set"])
            & subject["roi"].eq(row["roi"])
            & subject["condition_group"].eq(row["condition_group"])
            & subject["model"].eq(row["model"])
            & subject["predictor_name"].eq(row["predictor_name"])
            & subject["behavior_metric"].eq(row["behavior_metric"])
        ].copy()
        subset["predictor_value"] = pd.to_numeric(subset["predictor_value"], errors="coerce")
        subset["behavior_value"] = pd.to_numeric(subset["behavior_value"], errors="coerce")
        sns.regplot(
            data=subset,
            x="predictor_value",
            y="behavior_value",
            scatter_kws={"s": 34, "alpha": 0.85, "edgecolor": "white", "linewidths": 0.4},
            line_kws={"lw": 1.6, "color": "#333333"},
            ci=95,
            ax=ax,
        )
        ax.set_title(_panel_title(row), fontsize=10, fontweight="bold")
        ax.set_xlabel(PREDICTOR_LABELS.get(str(row["predictor_name"]), str(row["predictor_name"])))
        ax.set_ylabel(BEHAVIOR_LABELS.get(str(row["behavior_metric"]), str(row["behavior_metric"])))
        ax.text(
            0.03,
            0.96,
            f"rho={float(row['spearman_r']):.2f}, p={float(row['spearman_p']):.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        add_panel_label(ax, chr(ord("a") + idx))

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.90, wspace=0.34, hspace=0.48)
    save_png_pdf(fig, out_dir / "fig_relation_behavior_prediction.png")
    plt.close(fig)
    print(f"[relation-behavior-plot] wrote {out_dir / 'fig_relation_behavior_prediction.png'}")


if __name__ == "__main__":
    main()
