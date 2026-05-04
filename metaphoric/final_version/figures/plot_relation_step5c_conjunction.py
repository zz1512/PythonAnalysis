#!/usr/bin/env python3
"""Plot pair evidence by relation-vector contrast."""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from figure_utils import add_panel_label, apply_publication_rcparams, default_paper_output_root, save_png_pdf


CLASS_COLORS = {
    "yy_pair_and_yy_relation": "#2F6B4F",
    "yy_pair_kj_relation_dissociation": "#B1443C",
    "yy_pair_only": "#4C78A8",
    "yy_relation_only": "#6F4E9B",
    "kj_relation_only": "#8C6D31",
    "nonpositive_pair_effect": "#777777",
    "not_conjoined": "#CCCCCC",
}
CLASS_LABELS = {
    "yy_pair_and_yy_relation": "YY pair + YY relation",
    "yy_pair_kj_relation_dissociation": "YY pair + KJ relation",
    "yy_pair_only": "YY pair only",
    "yy_relation_only": "YY relation only",
    "kj_relation_only": "KJ relation only",
    "nonpositive_pair_effect": "Nonpositive pair",
    "not_conjoined": "Not conjoined",
}
SOURCE_MARKERS = {
    "edge_specificity": "o",
    "step5c_condition_drop": "s",
}


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _short(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _prepare(frame: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    for col in ["pair_mean_effect", "relation_mean_effect", "pair_q", "relation_q", "pair_p", "relation_p"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    interesting = data[data["conjunction_class"].ne("not_conjoined")].copy()
    if interesting.empty:
        interesting = data.sort_values(["pair_q", "relation_q", "pair_p"], na_position="last").head(top_n).copy()
    top = interesting.sort_values(
        ["conjunction_priority", "pair_q", "relation_q", "pair_p", "relation_p"],
        na_position="last",
    ).head(top_n)
    return data, top.reset_index(drop=True)


def _plot_scatter(ax: plt.Axes, data: pd.DataFrame) -> None:
    if data.empty:
        ax.text(0.5, 0.5, "No conjunction rows", ha="center", va="center")
        ax.axis("off")
        return
    for source, marker in SOURCE_MARKERS.items():
        subset = data[data["pair_evidence_source"].eq(source)]
        if subset.empty:
            continue
        colors = subset["conjunction_class"].map(CLASS_COLORS).fillna("#999999")
        ax.scatter(
            subset["pair_mean_effect"],
            subset["relation_mean_effect"],
            c=colors,
            s=56,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.88,
            label=source,
        )
    ax.axvline(0, color="#444444", ls="--", lw=0.9)
    ax.axhline(0, color="#444444", ls="--", lw=0.9)
    ax.set_xlabel("Pair differentiation evidence")
    ax.set_ylabel("YY - KJ relation-vector decoupling")
    ax.set_title("Direction-sensitive conjunction map", fontweight="bold")
    ax.grid(color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    class_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=7, label=CLASS_LABELS[key])
        for key, color in CLASS_COLORS.items()
        if key in set(data["conjunction_class"])
    ]
    source_handles = [
        Line2D([0], [0], marker=marker, color="#333333", linestyle="", markersize=7, label=source.replace("_", " "))
        for source, marker in SOURCE_MARKERS.items()
        if source in set(data["pair_evidence_source"])
    ]
    ax.legend(handles=class_handles + source_handles, frameon=False, loc="best", fontsize=7.5)


def _plot_top(ax: plt.Axes, top: pd.DataFrame) -> None:
    if top.empty:
        ax.text(0.5, 0.5, "No highlighted rows", ha="center", va="center")
        ax.axis("off")
        return
    frame = top.iloc[::-1].copy()
    labels = [
        _short(f"{row.roi_set} | {row.roi} | {row.pair_contrast_label} | {row.relation_model}", 46)
        for row in frame.itertuples(index=False)
    ]
    y = np.arange(len(frame))
    colors = frame["conjunction_class"].map(CLASS_COLORS).fillna("#999999")
    ax.barh(y - 0.18, frame["pair_mean_effect"], height=0.32, color=colors, alpha=0.9, label="Pair")
    ax.barh(y + 0.18, frame["relation_mean_effect"], height=0.32, color="#333333", alpha=0.72, label="Relation")
    ax.axvline(0, color="#444444", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Effect size in original metric")
    ax.set_title("Highlighted conjunction/dissociation rows", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step5C/A++ and relation-vector conjunction.")
    parser.add_argument("--paper-output-root", type=Path, default=default_paper_output_root())
    parser.add_argument("--top-n", type=int, default=18)
    args = parser.parse_args()

    paper_root = args.paper_output_root
    table_path = paper_root / "tables_main" / "table_relation_step5c_conjunction.tsv"
    frame = _read_table(table_path)
    data, top = _prepare(frame, args.top_n)

    figures_dir = paper_root / "figures_main"
    figures_dir.mkdir(parents=True, exist_ok=True)
    top_path = figures_dir / "table_relation_step5c_conjunction_top.tsv"
    top.to_csv(top_path, sep="\t", index=False)

    sns.set_theme(style="white", context="paper")
    apply_publication_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, max(5.8, len(top) * 0.42)), gridspec_kw={"width_ratios": [1.1, 1.55]})
    _plot_scatter(axes[0], data)
    _plot_top(axes[1], top)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.text(
        0.05,
        0.02,
        "Positive relation values indicate stronger YY than KJ decoupling; negative values indicate stronger KJ than YY decoupling.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.17, top=0.90, wspace=0.30)
    save_png_pdf(fig, figures_dir / "fig_relation_step5c_conjunction.png")
    plt.close(fig)
    print(f"[relation-step5c-conjunction-plot] wrote {figures_dir / 'fig_relation_step5c_conjunction.png'}")
    print(f"[relation-step5c-conjunction-plot] wrote {top_path}")


if __name__ == "__main__":
    main()
