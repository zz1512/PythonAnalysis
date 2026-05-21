#!/usr/bin/env python3
"""Supplementary figure: learning→pre/post/retrieval reinstatement + learning
within-item reliability (metaphor vs spatial).

Upstream inputs:
- Task 3: ``{BASE_DIR}/paper_outputs/qc/cross_phase_reinstatement_{roi_tag}/cross_phase_reinstatement_subject.tsv``
  and ``..._group_one_sample.tsv`` (with q column).
- Task 4: ``{BASE_DIR}/paper_outputs/qc/learning_within_item_reliability_{roi_tag}/learning_within_item_reliability_subject.tsv``,
  ``..._group_one_sample.tsv``, ``..._group_yy_vs_kj.tsv``.

Output:
- ``{BASE_DIR}/paper_outputs/figures_si/fig_learning_reinstatement_{roi_tag}.png`` (+ .pdf)
"""

from __future__ import annotations

import argparse
import math
import os
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

from common.final_utils import read_table  # noqa: E402
from common.roi_library import current_roi_set, sanitize_roi_tag  # noqa: E402
from figures.figure_utils import (  # noqa: E402
    abbreviate_roi_name,
    add_panel_label,
    apply_publication_rcparams,
    save_png_pdf,
)


TARGET_PHASES = ["pre", "post", "retrieval"]
CONDITIONS = ["yy", "kj"]
RUNS = [3, 4]
CONDITION_COLORS = {"yy": "#b23a48", "kj": "#3574b7"}
CONDITION_LABELS = {"yy": "Metaphor", "kj": "Spatial"}
Q_CANDIDATES = ("q_bh_primary_family", "q_bh_within_roi_set", "q_bh", "q")
REINSTATEMENT_VALUE_CANDIDATES = ("mean_reinstatement", "reinstatement", "value", "effect")
RELIABILITY_VALUE_CANDIDATES = ("mean_reliability", "reliability", "value", "effect")


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _pick_column(frame: pd.DataFrame, candidates) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _sig_marker(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "n.s."


def _mean_sem(vec: np.ndarray) -> tuple[float, float, int]:
    vec = np.asarray(vec, dtype=float)
    vec = vec[np.isfinite(vec)]
    n = int(vec.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(vec))
    sem = float(np.std(vec, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def _panel_reinstatement(
    axes_row,
    subject_df: pd.DataFrame,
    group_df: pd.DataFrame | None,
    rois: list[str],
) -> None:
    value_col = _pick_column(subject_df, REINSTATEMENT_VALUE_CANDIDATES)
    if value_col is None:
        for ax in axes_row:
            ax.text(0.5, 0.5, "No reinstatement value column", ha="center", va="center")
            ax.axis("off")
        return

    q_col = _pick_column(group_df, Q_CANDIDATES) if group_df is not None and not group_df.empty else None

    has_run = "run" in subject_df.columns
    x_groups = []
    for phase in TARGET_PHASES:
        for run in RUNS:
            x_groups.append((phase, run))
    x_positions = np.arange(len(x_groups), dtype=float)
    width = 0.35

    for col_idx, roi in enumerate(rois):
        ax = axes_row[col_idx]
        roi_rows = subject_df[subject_df["roi"].astype(str) == roi]
        for c_idx, cond in enumerate(CONDITIONS):
            offset = (c_idx - 0.5) * width
            means: list[float] = []
            sems: list[float] = []
            ns: list[int] = []
            for phase, run in x_groups:
                mask = (
                    (roi_rows["condition"].astype(str) == cond)
                    & (roi_rows["target_phase"].astype(str) == phase)
                )
                if has_run:
                    mask &= pd.to_numeric(roi_rows["run"], errors="coerce") == run
                vec = roi_rows.loc[mask, value_col].to_numpy(dtype=float)
                m, s, n = _mean_sem(vec)
                means.append(m)
                sems.append(s)
                ns.append(n)
            means_arr = np.asarray(means, dtype=float)
            sems_arr = np.asarray(sems, dtype=float)
            ax.errorbar(
                x_positions + offset,
                means_arr,
                yerr=sems_arr,
                fmt="o",
                color=CONDITION_COLORS[cond],
                ecolor=CONDITION_COLORS[cond],
                elinewidth=1.1,
                capsize=2.5,
                markersize=4,
                label=CONDITION_LABELS[cond] if col_idx == 0 else None,
            )

            # significance annotation per (phase, run, cond) if group_df has those columns
            if q_col is not None and group_df is not None:
                for j, (phase, run) in enumerate(x_groups):
                    if ns[j] < 3:
                        continue
                    match = group_df[
                        (group_df["roi"].astype(str) == roi)
                        & (group_df["condition"].astype(str) == cond)
                        & (group_df["target_phase"].astype(str) == phase)
                    ]
                    if "run" in group_df.columns:
                        match = match[pd.to_numeric(match["run"], errors="coerce") == run]
                    if match.empty:
                        continue
                    q_val = float(pd.to_numeric(match.iloc[0][q_col], errors="coerce"))
                    marker = _sig_marker(q_val)
                    if not marker or marker == "n.s.":
                        continue
                    y_val = means_arr[j]
                    err_val = sems_arr[j] if np.isfinite(sems_arr[j]) else 0.0
                    if not np.isfinite(y_val):
                        continue
                    y_annot = y_val + err_val + 0.02 * max(1.0, abs(y_val) + err_val)
                    ax.text(
                        x_positions[j] + offset,
                        y_annot,
                        marker,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=CONDITION_COLORS[cond],
                    )

        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax.set_title(abbreviate_roi_name(roi), fontsize=10, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{p}\nrun{r}" for p, r in x_groups], fontsize=7)
        ax.set_ylabel("Reinstatement", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if axes_row.size:
        handles, labels = axes_row[0].get_legend_handles_labels()
        if handles:
            axes_row[0].legend(handles, labels, frameon=False, fontsize=8, loc="best")


def _panel_reliability(
    axes_row,
    subject_df: pd.DataFrame,
    group_one_df: pd.DataFrame | None,
    group_contrast_df: pd.DataFrame | None,
    rois: list[str],
) -> None:
    value_col = _pick_column(subject_df, RELIABILITY_VALUE_CANDIDATES)
    if value_col is None:
        for ax in axes_row:
            ax.text(0.5, 0.5, "No reliability value column", ha="center", va="center")
            ax.axis("off")
        return

    q_col_one = (
        _pick_column(group_one_df, Q_CANDIDATES)
        if group_one_df is not None and not group_one_df.empty
        else None
    )
    q_col_contrast = (
        _pick_column(group_contrast_df, Q_CANDIDATES)
        if group_contrast_df is not None and not group_contrast_df.empty
        else None
    )

    x_positions = np.arange(len(CONDITIONS), dtype=float)
    bar_width = 0.55

    for col_idx, roi in enumerate(rois):
        ax = axes_row[col_idx]
        roi_rows = subject_df[subject_df["roi"].astype(str) == roi]

        means: list[float] = []
        sems: list[float] = []
        ns: list[int] = []
        for cond in CONDITIONS:
            vec = roi_rows.loc[
                roi_rows["condition"].astype(str) == cond, value_col
            ].to_numpy(dtype=float)
            m, s, n = _mean_sem(vec)
            means.append(m)
            sems.append(s)
            ns.append(n)
        means_arr = np.asarray(means, dtype=float)
        sems_arr = np.asarray(sems, dtype=float)

        bars = ax.bar(
            x_positions,
            means_arr,
            yerr=sems_arr,
            width=bar_width,
            color=[CONDITION_COLORS[c] for c in CONDITIONS],
            ecolor="#4b5563",
            capsize=3,
            edgecolor="white",
        )
        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax.set_title(abbreviate_roi_name(roi), fontsize=10, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=8)
        ax.set_ylabel("Within-item reliability", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # one-sample sig
        if q_col_one is not None and group_one_df is not None:
            for j, cond in enumerate(CONDITIONS):
                if ns[j] < 3:
                    continue
                match = group_one_df[
                    (group_one_df["roi"].astype(str) == roi)
                    & (group_one_df["condition"].astype(str) == cond)
                ]
                if match.empty:
                    continue
                q_val = float(pd.to_numeric(match.iloc[0][q_col_one], errors="coerce"))
                marker = _sig_marker(q_val)
                if not marker:
                    continue
                y_val = means_arr[j]
                err_val = sems_arr[j] if np.isfinite(sems_arr[j]) else 0.0
                if not np.isfinite(y_val):
                    continue
                y_annot = y_val + err_val + 0.015 * max(1.0, abs(y_val) + err_val)
                ax.text(
                    x_positions[j],
                    y_annot,
                    marker,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=CONDITION_COLORS[cond] if marker != "n.s." else "#6b7280",
                )

        # yy vs kj contrast bridge
        if q_col_contrast is not None and group_contrast_df is not None and min(ns) >= 3:
            match = group_contrast_df[group_contrast_df["roi"].astype(str) == roi]
            if not match.empty:
                q_val = float(pd.to_numeric(match.iloc[0][q_col_contrast], errors="coerce"))
                marker = _sig_marker(q_val)
                if marker:
                    tops = [
                        (means_arr[j] if np.isfinite(means_arr[j]) else 0.0)
                        + (sems_arr[j] if np.isfinite(sems_arr[j]) else 0.0)
                        for j in range(len(CONDITIONS))
                    ]
                    y_max = max(tops) if tops else 0.0
                    y_bridge = y_max + 0.08 * max(1.0, abs(y_max))
                    y_top = y_bridge + 0.04 * max(1.0, abs(y_max))
                    ax.plot(
                        [x_positions[0], x_positions[0], x_positions[-1], x_positions[-1]],
                        [y_bridge, y_top, y_top, y_bridge],
                        color="#374151",
                        linewidth=1.0,
                    )
                    ax.text(
                        x_positions.mean(),
                        y_top,
                        marker,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#374151" if marker != "n.s." else "#6b7280",
                    )


def _collect_rois(*frames: pd.DataFrame | None) -> list[str]:
    rois: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty or "roi" not in frame.columns:
            continue
        rois.update(frame["roi"].dropna().astype(str).unique().tolist())
    return sorted(rois)


def _load(path: Path, tag: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[learning-reinstatement-plot] WARN: {tag} not found: {path}")
        return None
    try:
        frame = read_table(path)
    except Exception as exc:
        print(f"[learning-reinstatement-plot] WARN: failed to read {tag} {path}: {exc}")
        return None
    if frame.empty:
        print(f"[learning-reinstatement-plot] WARN: {tag} is empty: {path}")
    return frame


def _plot(
    reinstatement_df: pd.DataFrame | None,
    reinstatement_group: pd.DataFrame | None,
    reliability_df: pd.DataFrame | None,
    reliability_group_one: pd.DataFrame | None,
    reliability_group_contrast: pd.DataFrame | None,
    out_path: Path,
    *,
    roi_tag: str,
) -> None:
    rois = _collect_rois(reinstatement_df, reliability_df)
    if not rois:
        print("[learning-reinstatement-plot] WARN: no ROI rows across inputs; exit.")
        sys.exit(0)

    n_cols = min(4, len(rois))
    n_rois = len(rois)
    # one row per panel (A: reinstatement, B: reliability); if too many ROIs, wrap into blocks
    cols = n_cols
    rows_per_panel = int(math.ceil(n_rois / cols))

    fig, axes = plt.subplots(
        2 * rows_per_panel,
        cols,
        figsize=(3.4 * cols, 2.9 * rows_per_panel * 2),
        squeeze=False,
    )

    # Panel A: reinstatement rows (top half)
    for block in range(rows_per_panel):
        block_rois = rois[block * cols : (block + 1) * cols]
        # pad with None placeholders not needed; just pass subset
        sub_axes = axes[block][: len(block_rois)]
        if reinstatement_df is None or reinstatement_df.empty:
            for ax in sub_axes:
                ax.text(0.5, 0.5, "No reinstatement data", ha="center", va="center")
                ax.axis("off")
        else:
            _panel_reinstatement(np.asarray(sub_axes), reinstatement_df, reinstatement_group, block_rois)
        for k in range(len(block_rois), cols):
            axes[block][k].axis("off")

    # Panel B: reliability rows (bottom half)
    for block in range(rows_per_panel):
        block_rois = rois[block * cols : (block + 1) * cols]
        row_idx = rows_per_panel + block
        sub_axes = axes[row_idx][: len(block_rois)]
        if reliability_df is None or reliability_df.empty:
            for ax in sub_axes:
                ax.text(0.5, 0.5, "No reliability data", ha="center", va="center")
                ax.axis("off")
        else:
            _panel_reliability(
                np.asarray(sub_axes),
                reliability_df,
                reliability_group_one,
                reliability_group_contrast,
                block_rois,
            )
        for k in range(len(block_rois), cols):
            axes[row_idx][k].axis("off")

    add_panel_label(axes[0][0], "a")
    add_panel_label(axes[rows_per_panel][0], "b")
    fig.suptitle(
        f"Learning reinstatement & within-item reliability (roi_set={roi_tag})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_png_pdf(fig, out_path)
    plt.close(fig)
    print(f"[learning-reinstatement-plot] wrote {out_path}")


def main() -> None:
    default_roi_set = current_roi_set("main_functional")
    parser = argparse.ArgumentParser(
        description="Plot learning reinstatement + within-item reliability SI figure."
    )
    parser.add_argument("--roi-set", default=default_roi_set)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reinstatement-subject", type=Path, default=None)
    parser.add_argument("--reinstatement-group", type=Path, default=None)
    parser.add_argument("--reliability-subject", type=Path, default=None)
    parser.add_argument("--reliability-group-one", type=Path, default=None)
    parser.add_argument("--reliability-group-contrast", type=Path, default=None)
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(args.roi_set)
    paper_root = (
        Path(args.paper_output_root)
        if args.paper_output_root
        else _default_base_dir() / "paper_outputs"
    )
    reinstatement_dir = paper_root / "qc" / f"cross_phase_reinstatement_{roi_tag}"
    reliability_dir = paper_root / "qc" / f"learning_within_item_reliability_{roi_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else paper_root / "figures_si"
    out_path = out_dir / f"fig_learning_reinstatement_{roi_tag}.png"

    reinstatement_subject = (
        Path(args.reinstatement_subject)
        if args.reinstatement_subject
        else reinstatement_dir / "cross_phase_reinstatement_subject.tsv"
    )
    reinstatement_group = (
        Path(args.reinstatement_group)
        if args.reinstatement_group
        else reinstatement_dir / "cross_phase_reinstatement_group_one_sample.tsv"
    )
    reliability_subject = (
        Path(args.reliability_subject)
        if args.reliability_subject
        else reliability_dir / "learning_within_item_reliability_subject.tsv"
    )
    reliability_group_one = (
        Path(args.reliability_group_one)
        if args.reliability_group_one
        else reliability_dir / "learning_within_item_reliability_group_one_sample.tsv"
    )
    reliability_group_contrast = (
        Path(args.reliability_group_contrast)
        if args.reliability_group_contrast
        else reliability_dir / "learning_within_item_reliability_group_yy_vs_kj.tsv"
    )

    reinstatement_df = _load(reinstatement_subject, "reinstatement subject TSV")
    reinstatement_group_df = _load(reinstatement_group, "reinstatement group TSV")
    reliability_df = _load(reliability_subject, "reliability subject TSV")
    reliability_group_one_df = _load(reliability_group_one, "reliability one-sample TSV")
    reliability_group_contrast_df = _load(
        reliability_group_contrast, "reliability yy-vs-kj TSV"
    )

    if (reinstatement_df is None or reinstatement_df.empty) and (
        reliability_df is None or reliability_df.empty
    ):
        print("[learning-reinstatement-plot] WARN: both reinstatement and reliability data missing; exit.")
        sys.exit(0)

    apply_publication_rcparams()
    _plot(
        reinstatement_df,
        reinstatement_group_df,
        reliability_df,
        reliability_group_one_df,
        reliability_group_contrast_df,
        out_path,
        roi_tag=roi_tag,
    )


if __name__ == "__main__":
    main()
