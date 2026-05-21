#!/usr/bin/env python3
"""Draw compact result-final figures for analyses that lacked manuscript plots.

Outputs are written to:
    E:/python_metaphor/paper_outputs/figures_result_final

The figures intentionally summarize the bug-fixed convergence analyses rather
than replacing the existing main figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from figures.figure_utils import add_panel_label, apply_publication_rcparams, save_png_pdf  # noqa: E402


COLORS = {
    "yy": "#b23a48",
    "kj": "#3574b7",
    "post": "#b23a48",
    "retrieval": "#2f6f9f",
    "hpc": "#516b4f",
    "semantic": "#6f4e9a",
    "neutral": "#4b5563",
    "ns": "#9ca3af",
}


def _base_dir() -> Path:
    return Path("E:/python_metaphor")


def _qc_dir() -> Path:
    return _base_dir() / "paper_outputs" / "qc"


def _out_dir() -> Path:
    return _base_dir() / "paper_outputs" / "figures_result_final"


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _ci_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ci_low"] = out["estimate"] - 1.96 * out["se"]
    out["ci_high"] = out["estimate"] + 1.96 * out["se"]
    return out


def _q_label(q: float | int | None) -> str:
    try:
        val = float(q)
    except Exception:
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    if val < 0.001:
        return "q<.001"
    return f"q={val:.3f}"


def _term_label(term: str) -> str:
    labels = {
        "condition[T.yy]": "YY main",
        "condition[T.yy]:post_pair_similarity_network_z": "YY x post similarity",
        "condition[T.yy]:retrieval_pair_similarity_network_z": "YY x retrieval similarity",
        "post_pair_similarity_network_z": "Post similarity",
        "retrieval_pair_similarity_network_z": "Retrieval similarity",
        "pre_pair_similarity_z": "Pre similarity",
        "pre_pair_similarity_z:condition[T.yy]": "Pre similarity x YY",
        "global_similarity_change_z:condition[T.yy]": "Global change x YY",
        "condition[T.yy]:novelty_z": "Novelty x YY",
        "semantic_learning_specificity_z:condition[T.yy]": "Learning specificity x YY",
        "semantic_retrieval_rebound_z:condition[T.yy]": "Retrieval rebound x YY",
        "semantic_retrieval_rebound_z": "Semantic retrieval rebound",
        "semantic_retrieval_minus_post_pair_similarity_z:condition[T.yy]": "Retrieval rebound x YY",
        "semantic_retrieval_minus_post_pair_similarity_z": "Semantic retrieval rebound",
        "semantic_learning_specificity_z": "Semantic learning specificity",
        "axis_position:C(condition)[T.YY]": "AP gradient x YY",
        "C(condition)[T.YY]": "YY main",
    }
    return labels.get(str(term), str(term))


def _network_label(network: str) -> str:
    return {"hpc_spatial": "HPC-spatial", "semantic": "Semantic/metaphor"}.get(str(network), str(network))


def _save(fig: plt.Figure, out_path: Path, manifest: list[dict[str, str]], title: str, note: str) -> None:
    save_png_pdf(fig, out_path)
    manifest.append(
        {
            "figure": out_path.name,
            "png": str(out_path),
            "pdf": str(out_path.with_suffix(".pdf")),
            "title": title,
            "note": note,
        }
    )
    plt.close(fig)


def plot_a4_memory_components(out_dir: Path, manifest: list[dict[str, str]]) -> None:
    path = _qc_dir() / "nc_converge" / "a4_post_memory_component" / "memory_component_model.tsv"
    df = _ci_bounds(_read_tsv(path))
    terms = [
        "condition[T.yy]",
        "condition[T.yy]:post_pair_similarity_network_z",
        "condition[T.yy]:retrieval_pair_similarity_network_z",
    ]
    outcomes = ["memory_prop", "memory_strict", "memory_lenient"]
    keep = df[df["term"].isin(terms) & df["outcome"].isin(outcomes)].copy()
    keep["label"] = keep["network"].map(_network_label) + " | " + keep["outcome"].str.replace("memory_", "", regex=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), sharey=True)
    for ax, term, color, panel in zip(
        axes,
        terms,
        [COLORS["neutral"], COLORS["post"], COLORS["retrieval"]],
        ["a", "b", "c"],
    ):
        sub = keep[keep["term"] == term].copy()
        sub["order"] = sub["network"].map({"hpc_spatial": 0, "semantic": 1}) * 10 + sub["outcome"].map(
            {"memory_prop": 0, "memory_strict": 1, "memory_lenient": 2}
        )
        sub = sub.sort_values("order")
        y = np.arange(len(sub))
        ax.axvline(0, color="#9ca3af", linewidth=0.9, linestyle="--")
        ax.errorbar(
            sub["estimate"],
            y,
            xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
        for yi, row in zip(y, sub.itertuples(index=False)):
            ax.text(row.ci_high + 0.02 * max(1, abs(row.ci_high)), yi, _q_label(row.q_bh), va="center", fontsize=7)
        ax.set_title(_term_label(term), fontweight="bold", fontsize=10)
        ax.set_xlabel("Model estimate")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, panel)
    axes[0].invert_yaxis()
    fig.suptitle("A4 memory component model: YY advantage and post/retrieval components", fontweight="bold")
    fig.tight_layout()
    _save(
        fig,
        out_dir / "fig_result_final_a4_memory_components.png",
        manifest,
        "A4 memory component model",
        "Forest plot of YY condition effects and YY-specific post/retrieval similarity components.",
    )


def plot_b2_mahalanobis_proxy(out_dir: Path, manifest: list[dict[str, str]]) -> None:
    path = _qc_dir() / "nc_converge" / "b2_crossnobis_robustness" / "crossnobis_step5c.tsv"
    df = _read_tsv(path)
    sub = df[(df["roi_set"] == "network_composite") & df["condition"].isin(["Metaphor", "Spatial"])].copy()
    if sub.empty:
        raise ValueError("B2 network-composite rows are empty; check roi_set/roi filters.")
    sub["network_label"] = sub["network"].map(_network_label)
    sub["condition_label"] = sub["condition"].map({"Metaphor": "YY", "Spatial": "KJ"})
    sub = sub.sort_values(["network", "condition"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    metrics = [
        ("beta_corr", "Correlation delta"),
        ("beta_crossnobis", "Mahalanobis proxy delta"),
    ]
    labels = sub["network_label"] + " | " + sub["condition_label"]
    y = np.arange(len(sub))
    for ax, (metric, title), panel in zip(axes, metrics, ["a", "b"]):
        vals = sub[metric].to_numpy(dtype=float)
        bar_colors = [COLORS["yy"] if c == "YY" else COLORS["kj"] for c in sub["condition_label"]]
        ax.axvline(0, color="#9ca3af", linewidth=0.9, linestyle="--")
        ax.barh(y, vals, color=bar_colors, alpha=0.85, height=0.55)
        for yi, val in zip(y, vals):
            ha = "left" if val >= 0 else "right"
            offset = 0.004 if val >= 0 else -0.004
            ax.text(val + offset, yi, f"{val:+.3f}", ha=ha, va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-0.09, 0.09)
        ax.set_xlabel("Post - pre delta")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, panel)
    fig.suptitle("B2 network-composite robustness proxy", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Direction check only: diagonal-shrinkage Mahalanobis proxy is not strict crossnobis.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    _save(
        fig,
        out_dir / "fig_result_final_b2_mahalanobis_proxy.png",
        manifest,
        "B2 Mahalanobis robustness proxy",
        "Network-composite comparison of correlation deltas and diagonal-shrinkage Mahalanobis proxy deltas.",
    )


def plot_b3_network_coupling(out_dir: Path, manifest: list[dict[str, str]]) -> None:
    path = _qc_dir() / "nc_converge" / "b3_repr_coupling" / "repr_coupling.tsv"
    df = _ci_bounds(_read_tsv(path))
    keep_terms = [
        "condition[T.yy]",
        "semantic_learning_specificity_z",
        "semantic_learning_specificity_z:condition[T.yy]",
        "semantic_retrieval_minus_post_pair_similarity_z",
        "semantic_retrieval_minus_post_pair_similarity_z:condition[T.yy]",
    ]
    sub = df[df["term"].isin(keep_terms)].copy()
    sub["block"] = sub["coupling"].map(
        {
            "learning_semantic_to_post_hpc_spatial": "Learning semantic -> post HPC-spatial",
            "retrieval_semantic_to_retrieval_hpc_spatial": "Retrieval semantic -> retrieval HPC-spatial",
        }
    )
    sub["label"] = sub["block"] + "\n" + sub["term"].map(_term_label)
    sub = sub.sort_values(["coupling", "term"])

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(sub))
    colors = [COLORS["yy"] if "condition" in t else COLORS["semantic"] for t in sub["term"]]
    ax.axvline(0, color="#9ca3af", linewidth=0.9, linestyle="--")
    for yi, row, color in zip(y, sub.itertuples(index=False), colors):
        ax.errorbar(
            row.estimate,
            yi,
            xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
        ax.text(row.ci_high + 0.03, yi, _q_label(row.q_bh), va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Model estimate")
    ax.set_title("B3 predefined network-level representational coupling", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "a")
    fig.tight_layout()
    _save(
        fig,
        out_dir / "fig_result_final_b3_network_coupling.png",
        manifest,
        "B3 network-level coupling",
        "Planned semantic/metaphor to hpc-spatial coupling terms for learning-to-post and retrieval-to-retrieval models.",
    )


def plot_c_boundaries(out_dir: Path, manifest: list[dict[str, str]]) -> None:
    c1 = _ci_bounds(_read_tsv(_qc_dir() / "nc_converge" / "c1_input_output_transformation" / "input_output_transformation.tsv"))
    c2 = _ci_bounds(_read_tsv(_qc_dir() / "nc_converge" / "c2_global_similarity_centrality" / "global_similarity_models.tsv"))
    c3 = _ci_bounds(_read_tsv(_qc_dir() / "nc_converge" / "c3_novelty_familiarity_moderation" / "novelty_familiarity_moderation.tsv"))

    panels = [
        (
            "C1 input-output",
            c1[c1["term"].isin(["condition[T.yy]", "pre_pair_similarity_z", "pre_pair_similarity_z:condition[T.yy]"])],
            "a",
        ),
        (
            "C2 global centrality",
            c2[c2["term"].isin(["condition[T.yy]", "global_similarity_change_z:condition[T.yy]", "condition[T.yy]:memory_prop"])],
            "b",
        ),
        (
            "C3 novelty/materials",
            c3[c3["term"].isin(["condition[T.yy]", "novelty_z", "condition[T.yy]:novelty_z"])],
            "c",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5))
    for ax, (title, frame, panel) in zip(axes, panels):
        sub = frame.copy()
        sub["label"] = sub["network"].map(_network_label) + "\n" + sub["term"].map(_term_label)
        sub = sub.sort_values(["network", "term"])
        y = np.arange(len(sub))
        ax.axvline(0, color="#9ca3af", linewidth=0.9, linestyle="--")
        for yi, row in zip(y, sub.itertuples(index=False)):
            color = COLORS["ns"] if not (isinstance(row.q_bh, float) and np.isfinite(row.q_bh) and row.q_bh < 0.05) else COLORS["neutral"]
            if str(row.status) != "ok":
                color = "#c2410c"
            ax.errorbar(
                row.estimate,
                yi,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
            )
            label = _q_label(row.q_bh) if str(row.status) == "ok" else str(row.status).replace("_", " ")
            ax.text(row.ci_high + 0.04, yi, label, va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Model estimate")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, panel)
    fig.suptitle("Mechanism refinement and boundary checks", fontweight="bold")
    fig.tight_layout()
    _save(
        fig,
        out_dir / "fig_result_final_c1_c2_c3_boundaries.png",
        manifest,
        "C1-C3 mechanism boundary checks",
        "Forest plots for input-output, global centrality, and novelty/familiarity moderation checks.",
    )


def plot_hpc_long_axis(out_dir: Path, manifest: list[dict[str, str]]) -> None:
    s4 = _ci_bounds(
        _read_tsv(
            _qc_dir()
            / "hpc_subfield_three_axis"
            / "s4_edge_specificity_subfield"
            / "subfield_edge_specificity.tsv"
        ).rename(columns={"q": "q_bh"})
    )
    s5 = _ci_bounds(
        _read_tsv(
            _qc_dir()
            / "hpc_subfield_three_axis"
            / "s5_anterior_posterior_contrast"
            / "anterior_posterior_contrast.tsv"
        ).rename(columns={"q": "q_bh"})
    )
    order = ["hpc_L_head", "hpc_L_body", "hpc_L_tail", "hpc_R_head", "hpc_R_body", "hpc_R_tail"]
    label_map = {
        "hpc_L_head": "L head",
        "hpc_L_body": "L body",
        "hpc_L_tail": "L tail",
        "hpc_R_head": "R head",
        "hpc_R_body": "R body",
        "hpc_R_tail": "R tail",
    }
    contrasts = [
        ("C1_YY_trained_drop_vs_zero", "YY trained drop vs 0"),
        ("C2_YY_vs_KJ_trained_drop", "YY drop vs KJ drop"),
        ("C4_YY_trained_vs_non_edge_drop", "YY drop vs non-edge"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    for ax, (contrast, title), panel in zip(axes[:3], contrasts, ["a", "b", "c"]):
        sub = s4[s4["contrast"] == contrast].copy()
        if contrast == "C2_YY_vs_KJ_trained_drop":
            sub = sub[sub["term"] == "C(condition)[T.YY]"].copy()
        elif contrast == "C4_YY_trained_vs_non_edge_drop":
            sub = sub[sub["term"] == "C(edge_type)[T.trained]"].copy()
        sub["order"] = sub["subROI"].map({v: i for i, v in enumerate(order)})
        sub = sub.sort_values("order")
        y = np.arange(len(sub))
        colors = [COLORS["yy"] if q < 0.05 else COLORS["ns"] for q in sub["q_bh"]]
        ax.axvline(0, color="#9ca3af", linewidth=0.9, linestyle="--")
        for yi, row, color in zip(y, sub.itertuples(index=False), colors):
            ax.errorbar(
                row.estimate,
                yi,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
            )
            ax.text(row.ci_high + 0.01, yi, _q_label(row.q_bh), va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([label_map[x] for x in sub["subROI"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Drop estimate")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_panel_label(ax, panel)

    ax = axes[3]
    sub = s5[s5["term"].isin(["C(condition)[T.YY]", "axis_position:C(condition)[T.YY]"])].copy()
    sub["label"] = sub["hemisphere_set"].str.replace("hemi_", "", regex=False).str.replace("pooled_LR", "pooled LR", regex=False)
    sub["label"] = sub["label"] + "\n" + sub["term"].map(_term_label)
    sub = sub.sort_values(["term", "hemisphere_set"])
    y = np.arange(len(sub))
    ax.axvline(0, color="#9ca3af", linewidth=0.9, linestyle="--")
    for yi, row in zip(y, sub.itertuples(index=False)):
        color = COLORS["yy"] if row.q_bh < 0.05 else COLORS["ns"]
        ax.errorbar(
            row.estimate,
            yi,
            xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
        ax.text(row.ci_high + 0.01, yi, _q_label(row.q_bh), va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("AP model", fontweight="bold", fontsize=10)
    ax.set_xlabel("Model estimate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "d")
    fig.suptitle("Exploratory hippocampal long-axis extension", fontweight="bold")
    fig.tight_layout()
    _save(
        fig,
        out_dir / "fig_result_final_hpc_long_axis_summary.png",
        manifest,
        "Exploratory hippocampal long-axis summary",
        "S4 edge-specificity contrasts and S5 AP-gradient model terms for MNI-defined head/body/tail segments.",
    )


def write_manifest(out_dir: Path, manifest: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(manifest)
    frame.to_csv(out_dir / "result_final_figure_manifest.tsv", sep="\t", index=False)
    lines = ["# Result Final Figure Manifest", ""]
    for item in manifest:
        lines.extend(
            [
                f"## {item['title']}",
                "",
                f"- PNG: `{item['png']}`",
                f"- PDF: `{item['pdf']}`",
                f"- Note: {item['note']}",
                "",
            ]
        )
    (out_dir / "result_final_figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw missing result-final figures.")
    parser.add_argument("--out-dir", type=Path, default=_out_dir())
    args = parser.parse_args()

    apply_publication_rcparams()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    plot_a4_memory_components(args.out_dir, manifest)
    plot_b2_mahalanobis_proxy(args.out_dir, manifest)
    plot_b3_network_coupling(args.out_dir, manifest)
    plot_c_boundaries(args.out_dir, manifest)
    plot_hpc_long_axis(args.out_dir, manifest)
    write_manifest(args.out_dir, manifest)

    print(f"[result-final-figures] wrote {len(manifest)} figures to {args.out_dir}")
    for item in manifest:
        print(item["png"])


if __name__ == "__main__":
    main()
