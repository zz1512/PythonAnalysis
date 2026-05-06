#!/usr/bin/env python3
"""Plot four/five-phase trajectory (pre / run3 / run4 / post / retrieval) of
condition-level RDM effect (`between - within`) for each ROI.

Upstream inputs (produced by Task 2):
- ``{BASE_DIR}/paper_outputs/qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_{roi_tag}.csv``
- ``{BASE_DIR}/paper_outputs/qc/learning_condition_rdm_{roi_tag}/four_phase_trajectory_group_one_sample.tsv``

Output:
- ``{BASE_DIR}/paper_outputs/figures_main/fig_four_phase_trajectory_{roi_tag}.png`` (+ .pdf)
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


PHASE_ORDER = ["pre", "run3", "run4", "post", "retrieval"]
Q_CANDIDATES = ("q_bh_primary_family", "q_bh_within_roi_set", "q_bh", "q")
EFFECT_CANDIDATES = ("between_minus_within", "effect", "mean_effect")


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


def _plot(
    subject_df: pd.DataFrame,
    group_df: pd.DataFrame | None,
    out_path: Path,
    *,
    roi_tag: str,
) -> None:
    effect_col = _pick_column(subject_df, EFFECT_CANDIDATES)
    if effect_col is None:
        print(
            f"[four-phase-plot] WARN: none of {EFFECT_CANDIDATES} present in subject CSV. "
            f"columns={list(subject_df.columns)}"
        )
        sys.exit(0)

    phases_present = [p for p in PHASE_ORDER if p in set(subject_df["phase"].astype(str))]
    if not phases_present:
        print("[four-phase-plot] WARN: no recognized phases in subject CSV; exit.")
        sys.exit(0)

    rois = sorted(subject_df["roi"].dropna().astype(str).unique().tolist())
    if not rois:
        print("[four-phase-plot] WARN: no ROI rows; exit.")
        sys.exit(0)

    q_col = None
    if group_df is not None and not group_df.empty:
        q_col = _pick_column(group_df, Q_CANDIDATES)

    n_rois = len(rois)
    n_cols = min(4, n_rois)
    n_rows = int(math.ceil(n_rois / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.3 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    x_positions = np.arange(len(phases_present), dtype=float)

    for idx, roi in enumerate(rois):
        ax = axes[idx // n_cols][idx % n_cols]
        roi_rows = subject_df[subject_df["roi"].astype(str) == roi]

        means: list[float] = []
        sems: list[float] = []
        ns: list[int] = []
        for phase in phases_present:
            vec = roi_rows.loc[
                roi_rows["phase"].astype(str) == phase, effect_col
            ].to_numpy(dtype=float)
            vec = vec[np.isfinite(vec)]
            ns.append(int(vec.size))
            if vec.size == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(float(np.mean(vec)))
                sems.append(float(np.std(vec, ddof=1) / np.sqrt(vec.size)) if vec.size > 1 else 0.0)

        means_arr = np.asarray(means, dtype=float)
        sems_arr = np.asarray(sems, dtype=float)

        ax.errorbar(
            x_positions,
            means_arr,
            yerr=sems_arr,
            marker="o",
            color="#2a4d8f",
            ecolor="#2a4d8f",
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            linewidth=1.4,
        )
        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax.set_title(abbreviate_roi_name(roi), fontsize=10, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(phases_present, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("between − within", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # annotate significance per phase
        low_n_flag = False
        for j, phase in enumerate(phases_present):
            if ns[j] < 3:
                low_n_flag = True
                continue
            if q_col is None or group_df is None:
                continue
            match = group_df[
                (group_df["roi"].astype(str) == roi)
                & (group_df["phase"].astype(str) == phase)
            ]
            if match.empty:
                continue
            q_val = float(pd.to_numeric(match.iloc[0][q_col], errors="coerce"))
            marker = _sig_marker(q_val)
            if not marker:
                continue
            y_val = means_arr[j]
            err_val = sems_arr[j] if np.isfinite(sems_arr[j]) else 0.0
            if not np.isfinite(y_val):
                continue
            y_annot = y_val + err_val + 0.02 * max(1.0, abs(y_val) + err_val)
            ax.text(
                x_positions[j],
                y_annot,
                marker,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#b23a48" if marker != "n.s." else "#6b7280",
            )

        if low_n_flag:
            ax.text(
                0.02,
                0.96,
                "n<3 in some phase",
                transform=ax.transAxes,
                fontsize=7,
                color="#6b7280",
                ha="left",
                va="top",
            )

    # hide unused axes
    for k in range(n_rois, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    add_panel_label(axes[0][0], "a")
    fig.suptitle(
        f"Four-phase trajectory of condition-level RDM (roi_set={roi_tag})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_pdf(fig, out_path)
    plt.close(fig)
    print(f"[four-phase-plot] wrote {out_path}")


def main() -> None:
    default_roi_set = current_roi_set("main_functional")
    parser = argparse.ArgumentParser(description="Plot four-phase RDM trajectory figure.")
    parser.add_argument("--roi-set", default=default_roi_set)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--group-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(args.roi_set)
    paper_root = (
        Path(args.paper_output_root)
        if args.paper_output_root
        else _default_base_dir() / "paper_outputs"
    )
    qc_dir = paper_root / "qc" / f"learning_condition_rdm_{roi_tag}"
    input_csv = Path(args.input_csv) if args.input_csv else qc_dir / f"four_phase_trajectory_{roi_tag}.csv"
    group_tsv = (
        Path(args.group_tsv)
        if args.group_tsv
        else qc_dir / "four_phase_trajectory_group_one_sample.tsv"
    )
    out_dir = Path(args.out_dir) if args.out_dir else paper_root / "figures_main"
    out_path = out_dir / f"fig_four_phase_trajectory_{roi_tag}.png"

    if not input_csv.exists():
        print(f"[four-phase-plot] WARN: subject CSV not found: {input_csv}; exit.")
        sys.exit(0)

    try:
        subject_df = read_table(input_csv)
    except Exception as exc:
        print(f"[four-phase-plot] WARN: failed to read {input_csv}: {exc}; exit.")
        sys.exit(0)
    if subject_df.empty:
        print(f"[four-phase-plot] WARN: subject CSV is empty: {input_csv}; exit.")
        sys.exit(0)
    required = {"subject", "roi", "phase"}
    missing = required - set(subject_df.columns)
    if missing:
        print(
            f"[four-phase-plot] WARN: subject CSV missing columns {missing}; exit. "
            f"columns={list(subject_df.columns)}"
        )
        sys.exit(0)

    group_df: pd.DataFrame | None = None
    if group_tsv.exists():
        try:
            group_df = read_table(group_tsv)
        except Exception as exc:
            print(f"[four-phase-plot] WARN: failed to read group TSV {group_tsv}: {exc}; continue without sig.")
            group_df = None
    else:
        print(f"[four-phase-plot] WARN: group TSV not found: {group_tsv}; continue without sig.")

    apply_publication_rcparams()
    _plot(subject_df, group_df, out_path, roi_tag=roi_tag)


if __name__ == "__main__":
    main()
