#!/usr/bin/env python3
"""Plot split phase profiles for condition-level RDM effect (`between - within`).

Main-text usage after P11 revision:
- word-level line: ``pre / post / retrieval`` (constituent-word geometry)
- learning line: ``run3 / run4`` (sentence-level geometry)

Upstream inputs (preferred, produced by ``learning_condition_rdm.py``):
- ``.../word_level_profile_{roi_tag}.csv``
- ``.../word_level_profile_group_one_sample.tsv``
- ``.../word_level_profile_group_pairwise.tsv``
- ``.../learning_profile_{roi_tag}.csv``
- ``.../learning_profile_group_one_sample.tsv``
- ``.../learning_profile_group_pairwise.tsv``

Backward compatibility:
- if the new split files are absent, this script falls back to
  ``four_phase_trajectory_{roi_tag}.csv`` and slices phases internally.

Output:
- ``{BASE_DIR}/paper_outputs/figures_main/fig_word_level_profile_{roi_tag}.png`` (+ .pdf)
- ``{BASE_DIR}/paper_outputs/figures_main/fig_learning_profile_{roi_tag}.png`` (+ .pdf)
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
WORD_LEVEL_PHASES = ["pre", "post", "retrieval"]
LEARNING_PHASES = ["run3", "run4"]
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


def _subset_phases(frame: pd.DataFrame, phases: list[str]) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    out = frame[frame["phase"].astype(str).isin(phases)].copy()
    if out.empty:
        return out
    out["phase"] = pd.Categorical(out["phase"], categories=phases, ordered=True)
    sort_cols = [c for c in ["roi_set", "roi", "subject", "phase"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _subset_pairwise(frame: pd.DataFrame | None, phases: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    needed = {"phase_a", "phase_b"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame()
    out = frame[
        frame["phase_a"].astype(str).isin(phases) & frame["phase_b"].astype(str).isin(phases)
    ].copy()
    if out.empty:
        return out
    phase_rank = {phase: idx for idx, phase in enumerate(phases)}
    out["phase_a_order"] = out["phase_a"].map(phase_rank)
    out["phase_b_order"] = out["phase_b"].map(phase_rank)
    return out.sort_values(
        ["roi_set", "roi", "phase_a_order", "phase_b_order"]
    ).drop(columns=["phase_a_order", "phase_b_order"])


def _draw_sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    text: str,
    *,
    h: float,
    color: str = "#7a1f2b",
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, linewidth=1.1)
    ax.text(
        (x1 + x2) / 2.0,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        fontweight="bold" if text != "n.s." else "normal",
    )


def _plot(
    subject_df: pd.DataFrame,
    group_df: pd.DataFrame | None,
    pairwise_df: pd.DataFrame | None,
    out_path: Path,
    *,
    roi_tag: str,
    phases: list[str],
    planned_contrasts: list[tuple[str, str]],
    figure_title: str,
    panel_label: str,
) -> None:
    effect_col = _pick_column(subject_df, EFFECT_CANDIDATES)
    if effect_col is None:
        print(
            f"[phase-profile] WARN: none of {EFFECT_CANDIDATES} present in subject CSV. "
            f"columns={list(subject_df.columns)}"
        )
        sys.exit(0)

    phases_present = [p for p in phases if p in set(subject_df["phase"].astype(str))]
    if not phases_present:
        print(f"[phase-profile] WARN: no recognized phases for {figure_title}; exit.")
        sys.exit(0)

    rois = sorted(subject_df["roi"].dropna().astype(str).unique().tolist())
    if not rois:
        print(f"[phase-profile] WARN: no ROI rows for {figure_title}; exit.")
        sys.exit(0)

    q_col = None
    if group_df is not None and not group_df.empty:
        q_col = _pick_column(group_df, Q_CANDIDATES)
    pairwise_q_col = None
    if pairwise_df is not None and not pairwise_df.empty:
        pairwise_q_col = _pick_column(pairwise_df, Q_CANDIDATES)

    n_rois = len(rois)
    n_cols = min(4, n_rois)
    n_rows = int(math.ceil(n_rois / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 3.1 * n_rows),
        squeeze=False,
    )

    x_positions = np.arange(len(phases_present), dtype=float)
    phase_to_x = {phase: x for x, phase in zip(x_positions, phases_present)}

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

        bar_colors = ["#8fb7e8" if p in {"pre", "post", "retrieval"} else "#f2b27c" for p in phases_present]
        ax.bar(
            x_positions,
            means_arr,
            color=bar_colors,
            edgecolor="#355070",
            linewidth=0.9,
            width=0.62,
            zorder=1,
        )
        ax.errorbar(
            x_positions,
            means_arr,
            yerr=sems_arr,
            fmt="none",
            ecolor="#355070",
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )

        # subject-level points to retain within-subject spread
        rng = np.random.default_rng(42 + idx)
        for j, phase in enumerate(phases_present):
            vec = roi_rows.loc[
                roi_rows["phase"].astype(str) == phase, effect_col
            ].to_numpy(dtype=float)
            vec = vec[np.isfinite(vec)]
            if vec.size == 0:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=vec.size)
            ax.scatter(
                np.full(vec.size, x_positions[j]) + jitter,
                vec,
                s=10,
                color="#4b5563",
                alpha=0.45,
                linewidths=0,
                zorder=2,
            )
        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax.set_title(abbreviate_roi_name(roi), fontsize=10, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(phases_present, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("between − within", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # annotate one-sample significance per phase (small marker above bar)
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
                fontsize=8,
                color="#b23a48" if marker != "n.s." else "#6b7280",
            )

        # annotate pairwise/significant changes between phases
        if pairwise_q_col is not None and pairwise_df is not None:
            roi_pairwise = pairwise_df[pairwise_df["roi"].astype(str) == roi].copy()
            if not roi_pairwise.empty:
                y_candidates = np.concatenate(
                    [
                        means_arr[np.isfinite(means_arr)],
                        (means_arr + np.nan_to_num(sems_arr, nan=0.0))[np.isfinite(means_arr)],
                    ]
                )
                base_y = float(np.nanmax(y_candidates)) if y_candidates.size else 0.0
                spread = float(np.nanmax(y_candidates) - np.nanmin(y_candidates)) if y_candidates.size else 0.2
                step = max(0.04, 0.12 * max(spread, 0.2))
                bracket_index = 0
                for phase_a, phase_b in planned_contrasts:
                    if phase_a not in phase_to_x or phase_b not in phase_to_x:
                        continue
                    match = roi_pairwise[
                        (roi_pairwise["phase_a"].astype(str) == phase_a)
                        & (roi_pairwise["phase_b"].astype(str) == phase_b)
                    ]
                    if match.empty:
                        continue
                    q_val = float(pd.to_numeric(match.iloc[0][pairwise_q_col], errors="coerce"))
                    marker = _sig_marker(q_val)
                    if not marker or marker == "n.s.":
                        continue
                    y = base_y + step * (1.2 + bracket_index)
                    _draw_sig_bracket(
                        ax,
                        phase_to_x[phase_b],
                        phase_to_x[phase_a],
                        y,
                        marker,
                        h=step * 0.35,
                    )
                    bracket_index += 1

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

    add_panel_label(axes[0][0], panel_label)
    fig.suptitle(
        f"{figure_title} (roi_set={roi_tag})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_pdf(fig, out_path)
    plt.close(fig)
    print(f"[phase-profile] wrote {out_path}")


def _read_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return read_table(path)
    except Exception as exc:
        print(f"[phase-profile] WARN: failed to read {path}: {exc}")
        return None


def main() -> None:
    default_roi_set = current_roi_set("main_functional")
    parser = argparse.ArgumentParser(
        description="Plot split word-level / learning-level RDM phase-profile figures."
    )
    parser.add_argument("--roi-set", default=default_roi_set)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--group-tsv", type=Path, default=None)
    parser.add_argument("--pairwise-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    roi_tag = sanitize_roi_tag(args.roi_set)
    paper_root = (
        Path(args.paper_output_root)
        if args.paper_output_root
        else _default_base_dir() / "paper_outputs"
    )
    qc_dir = paper_root / "qc" / f"learning_condition_rdm_{roi_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else paper_root / "figures_main"
    combined_input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else qc_dir / f"four_phase_trajectory_{roi_tag}.csv"
    )
    combined_group_tsv = (
        Path(args.group_tsv)
        if args.group_tsv
        else qc_dir / "four_phase_trajectory_group_one_sample.tsv"
    )
    combined_pairwise_tsv = (
        Path(args.pairwise_tsv)
        if args.pairwise_tsv
        else qc_dir / "four_phase_trajectory_group_pairwise.tsv"
    )

    # Prefer the new split outputs; fall back to combined 5-point table if needed.
    word_input_csv = qc_dir / f"word_level_profile_{roi_tag}.csv"
    word_group_tsv = qc_dir / "word_level_profile_group_one_sample.tsv"
    word_pairwise_tsv = qc_dir / "word_level_profile_group_pairwise.tsv"
    learn_input_csv = qc_dir / f"learning_profile_{roi_tag}.csv"
    learn_group_tsv = qc_dir / "learning_profile_group_one_sample.tsv"
    learn_pairwise_tsv = qc_dir / "learning_profile_group_pairwise.tsv"

    source_csv = combined_input_csv if combined_input_csv.exists() else word_input_csv
    if not source_csv.exists():
        print(f"[phase-profile] WARN: subject CSV not found: {source_csv}; exit.")
        sys.exit(0)

    try:
        source_df = read_table(source_csv)
    except Exception as exc:
        print(f"[phase-profile] WARN: failed to read {source_csv}: {exc}; exit.")
        sys.exit(0)
    if source_df.empty:
        print(f"[phase-profile] WARN: subject CSV is empty: {source_csv}; exit.")
        sys.exit(0)
    required = {"subject", "roi", "phase"}
    missing = required - set(source_df.columns)
    if missing:
        print(
            f"[phase-profile] WARN: subject CSV missing columns {missing}; exit. "
            f"columns={list(source_df.columns)}"
        )
        sys.exit(0)

    word_subject_df = (
        read_table(word_input_csv) if word_input_csv.exists() else _subset_phases(source_df, WORD_LEVEL_PHASES)
    )
    learning_subject_df = (
        read_table(learn_input_csv) if learn_input_csv.exists() else _subset_phases(source_df, LEARNING_PHASES)
    )
    combined_group_df = _read_optional_table(combined_group_tsv)
    combined_pairwise_df = _read_optional_table(combined_pairwise_tsv)
    word_group_df = (
        _read_optional_table(word_group_tsv)
        if word_group_tsv.exists()
        else _subset_phases(combined_group_df, WORD_LEVEL_PHASES)
    )
    learning_group_df = (
        _read_optional_table(learn_group_tsv)
        if learn_group_tsv.exists()
        else _subset_phases(combined_group_df, LEARNING_PHASES)
    )
    word_pairwise_df = (
        _read_optional_table(word_pairwise_tsv)
        if word_pairwise_tsv.exists()
        else _subset_pairwise(combined_pairwise_df, WORD_LEVEL_PHASES)
    )
    learning_pairwise_df = (
        _read_optional_table(learn_pairwise_tsv)
        if learn_pairwise_tsv.exists()
        else _subset_pairwise(combined_pairwise_df, LEARNING_PHASES)
    )

    apply_publication_rcparams()
    if word_subject_df is not None and not word_subject_df.empty:
        _plot(
            word_subject_df,
            word_group_df,
            word_pairwise_df,
            out_dir / f"fig_word_level_profile_{roi_tag}.png",
            roi_tag=roi_tag,
            phases=WORD_LEVEL_PHASES,
            planned_contrasts=[("post", "pre"), ("retrieval", "post"), ("retrieval", "pre")],
            figure_title="Word-level profile (pre / post / retrieval)",
            panel_label="a",
        )
    else:
        print("[phase-profile] WARN: word-level subject frame is empty; skip word-level figure.")

    if learning_subject_df is not None and not learning_subject_df.empty:
        _plot(
            learning_subject_df,
            learning_group_df,
            learning_pairwise_df,
            out_dir / f"fig_learning_profile_{roi_tag}.png",
            roi_tag=roi_tag,
            phases=LEARNING_PHASES,
            planned_contrasts=[("run4", "run3")],
            figure_title="Learning sentence-level profile (run3 / run4)",
            panel_label="b",
        )
    else:
        print("[phase-profile] WARN: learning subject frame is empty; skip learning figure.")


if __name__ == "__main__":
    main()
