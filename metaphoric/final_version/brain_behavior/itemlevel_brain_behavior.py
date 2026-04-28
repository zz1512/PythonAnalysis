#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
itemlevel_brain_behavior.py

Purpose
- Upgrade brain-behavior analysis from subject-level summary correlations to
  within-subject item-level coupling.
- Join run-7 trial-level behavior with pair-level neural similarity change via
  `pair_id`, then test whether items with stronger neural change are more likely
  to be remembered within the same subject and ROI.

Primary design
- Behavior DV: run-7 `memory` (0/1)
- Neural predictor: `delta_similarity_post_minus_pre = post - pre`
- Within-subject effect: item-level deviation from each subject's mean neural
  delta inside the same `roi × condition`
- Between-subject effect: the corresponding subject mean, added explicitly so
  the within-subject slope is not confounded with between-subject variation

Primary outputs
- `paper_outputs/tables_main/table_itemlevel_coupling.tsv`
- `paper_outputs/figures_main/fig_itemlevel_coupling.png`

Supporting outputs
- `paper_outputs/tables_si/table_itemlevel_coupling_full.tsv`
- `paper_outputs/tables_si/table_itemlevel_coupling_interaction.tsv`
- `paper_outputs/tables_si/table_itemlevel_coupling_rt.tsv`
- `paper_outputs/qc/itemlevel_behavior_neural_long.tsv`
- `paper_outputs/qc/itemlevel_neural_pair_delta.tsv`
- `paper_outputs/qc/itemlevel_behavior_join_qc.tsv`
- `paper_outputs/qc/itemlevel_coupling_manifest.json`
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.stimulus_text_mapping import normalize_stimulus_label  # noqa: E402


BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))

CONDITION_MAP = {
    "YY": "Metaphor",
    "KJ": "Spatial",
    "Metaphor": "Metaphor",
    "Spatial": "Spatial",
}

ROI_SET_SHORT_LABELS = {
    "main_functional": "Main functional",
    "literature": "Literature",
}


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _canonical_time(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"pre", "pretest"}:
        return "pre"
    if text in {"post", "posttest"}:
        return "post"
    return text


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric[np.isfinite(numeric)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    std = float(valid.std(ddof=1))
    if math.isclose(std, 0.0) or np.isnan(std):
        return pd.Series(0.0, index=series.index, dtype=float)
    mean = float(valid.mean())
    return (numeric - mean) / std


def _prepare_template(template_path: Path) -> pd.DataFrame:
    frame = _read_table(template_path).copy()
    frame["condition"] = frame["condition"].map(CONDITION_MAP).fillna(frame["condition"])
    frame["word_label"] = frame["word_label"].astype(str).str.strip()
    frame["normalized_word_label"] = frame["word_label"].map(normalize_stimulus_label)
    return frame[["condition", "pair_id", "type", "word_label", "normalized_word_label"]]


def _prepare_behavior_trials(behavior_trials_path: Path, template: pd.DataFrame) -> pd.DataFrame:
    frame = _read_table(behavior_trials_path).copy()
    frame = frame[frame["condition"].isin(["YY", "KJ"])].copy()
    frame["condition_neural"] = frame["condition"].map(CONDITION_MAP)
    frame["item_label"] = frame["pic_out"].fillna(frame["item"]).astype(str).str.strip()
    frame["normalized_word_label"] = frame["item_label"].map(normalize_stimulus_label)
    frame["memory"] = pd.to_numeric(frame["memory"], errors="coerce")
    frame["log_rt_correct"] = pd.to_numeric(frame["log_rt_correct"], errors="coerce")

    merged = frame.merge(
        template,
        how="left",
        on="normalized_word_label",
        suffixes=("", "_tpl"),
    )
    same_condition = merged["condition_neural"].eq(merged["condition_tpl"])
    merged = merged[same_condition | merged["condition_tpl"].isna()].copy()
    merged["pair_id"] = pd.to_numeric(merged["pair_id"], errors="coerce")
    merged = merged.dropna(subset=["pair_id", "memory"])
    merged["pair_id"] = merged["pair_id"].astype(int)
    # `word_label` after the merge is the canonical stimulus label from the template
    # (e.g., yyw_1 / yyew_1). We'll use it to join to the neural item-wise RSA rows.
    if "word_label" in merged.columns:
        merged["word_label"] = merged["word_label"].astype(str).str.strip()
    merged["word_label_template"] = merged.get("word_label")
    return merged


def _prepare_neural_pair_delta(rsa_path: Path, roi_set: str) -> pd.DataFrame:
    frame = _read_table(rsa_path).copy()
    if "time" not in frame.columns and "stage" in frame.columns:
        frame["time"] = frame["stage"]
    frame["time"] = frame["time"].map(_canonical_time)
    frame = frame[frame["condition"].isin(["Metaphor", "Spatial"])].copy()
    frame["pair_id"] = pd.to_numeric(frame["pair_id"], errors="coerce")
    frame = frame.dropna(subset=["pair_id", "similarity"])
    frame["pair_id"] = frame["pair_id"].astype(int)
    if "word_label" in frame.columns:
        frame["word_label"] = frame["word_label"].astype(str).str.strip()
    if "partner_label" in frame.columns:
        frame["partner_label"] = frame["partner_label"].astype(str).str.strip()

    pivot = (
        frame.pivot_table(
            index=["subject", "roi", "condition", "pair_id", "item", "word_label", "partner_label"],
            columns="time",
            values="similarity",
            aggfunc="mean",
        )
        .reset_index()
    )
    if "pre" not in pivot.columns or "post" not in pivot.columns:
        raise ValueError(f"Missing pre/post columns in {rsa_path}")

    pivot = pivot.rename(columns={"pre": "similarity_pre", "post": "similarity_post"})
    pivot["delta_similarity_post_minus_pre"] = pivot["similarity_post"] - pivot["similarity_pre"]
    pivot["differentiation_strength_pre_minus_post"] = pivot["similarity_pre"] - pivot["similarity_post"]
    pivot["roi_set"] = roi_set
    return pivot


def _build_itemlevel_long(
    behavior_trials: pd.DataFrame,
    neural_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []

    for roi_set, neural in neural_frames.items():
        # Avoid implicit cartesian products: join behavior trials to the matching
        # item-wise neural delta by `(subject, condition, pair_id, word_label)`.
        #
        # Context: for each learned pair_id, RSA item-wise tables usually contain
        # two rows (w and ew). If we join only on pair_id, each behavior row would
        # match both neural rows, duplicating observations and inflating significance.
        behavior_for_join = behavior_trials.copy()
        if "condition" in behavior_for_join.columns:
            behavior_for_join = behavior_for_join.rename(columns={"condition": "condition_behavior"})
        merged = behavior_for_join.merge(
            neural,
            how="inner",
            left_on=["subject", "condition_neural", "pair_id", "word_label"],
            right_on=["subject", "condition", "pair_id", "word_label"],
            suffixes=("", "_neural"),
        )
        merged = merged.drop(columns=["condition_neural"])
        joined_frames.append(merged)
        qc_rows.append(
            {
                "roi_set": roi_set,
                "n_behavior_rows": int(len(behavior_trials)),
                "n_joined_rows": int(len(merged)),
                "n_subjects": int(merged["subject"].nunique()) if not merged.empty else 0,
                "n_rois": int(merged["roi"].nunique()) if not merged.empty else 0,
            }
        )

    long_frame = pd.concat(joined_frames, ignore_index=True) if joined_frames else pd.DataFrame()
    if long_frame.empty:
        raise RuntimeError("No behavior/neural rows could be joined at item level.")

    long_frame["memory"] = pd.to_numeric(long_frame["memory"], errors="coerce")
    long_frame["log_rt_correct"] = pd.to_numeric(long_frame["log_rt_correct"], errors="coerce")
    long_frame["is_correct"] = long_frame["memory"].eq(1)
    long_frame["delta_similarity_subject_mean"] = long_frame.groupby(
        ["roi_set", "roi", "condition", "subject"]
    )["delta_similarity_post_minus_pre"].transform("mean")
    long_frame["delta_similarity_within"] = (
        long_frame["delta_similarity_post_minus_pre"] - long_frame["delta_similarity_subject_mean"]
    )
    long_frame["differentiation_subject_mean"] = long_frame.groupby(
        ["roi_set", "roi", "condition", "subject"]
    )["differentiation_strength_pre_minus_post"].transform("mean")
    long_frame["differentiation_within"] = (
        long_frame["differentiation_strength_pre_minus_post"] - long_frame["differentiation_subject_mean"]
    )

    qc = pd.DataFrame(qc_rows)
    return long_frame, qc


def _fit_gee(
    frame: pd.DataFrame,
    *,
    formula: str,
    response_family: str,
):
    family = Binomial() if response_family == "binomial" else Gaussian()
    model = GEE.from_formula(
        formula,
        groups="subject",
        cov_struct=Exchangeable(),
        family=family,
        data=frame,
    )
    return model.fit()


def _extract_param_rows(
    fit,
    *,
    roi_set: str,
    roi: str,
    response: str,
    model_label: str,
    condition: str | None,
    n_obs: int,
    n_subjects: int,
    n_pairs: int,
    primary_terms: list[str],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    conf = fit.conf_int()
    rows: list[dict[str, object]] = []
    primary_row: dict[str, object] | None = None
    for term in fit.params.index:
        row = {
            "roi_set": roi_set,
            "roi": roi,
            "response": response,
            "model_label": model_label,
            "condition": condition,
            "term": term,
            "estimate": float(fit.params[term]),
            "std_error": float(fit.bse[term]),
            "z_value": float(fit.tvalues[term]),
            "p_value": float(fit.pvalues[term]),
            "ci_low": float(conf.loc[term, 0]),
            "ci_high": float(conf.loc[term, 1]),
            "n_obs": int(n_obs),
            "n_subjects": int(n_subjects),
            "n_pairs": int(n_pairs),
        }
        rows.append(row)
        if term in primary_terms:
            primary_row = row.copy()
    return rows, primary_row


def _fit_condition_specific_models(long_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []

    for (roi_set, roi, condition), subset in long_frame.groupby(["roi_set", "roi", "condition"], sort=False):
        memory_frame = subset.dropna(subset=["memory", "delta_similarity_within", "delta_similarity_subject_mean"]).copy()
        if memory_frame["subject"].nunique() < 3 or memory_frame["pair_id"].nunique() < 5:
            continue
        memory_frame["delta_similarity_within_z"] = _zscore(memory_frame["delta_similarity_within"])
        memory_frame["delta_similarity_subject_mean_z"] = _zscore(memory_frame["delta_similarity_subject_mean"])
        fit = _fit_gee(
            memory_frame,
            formula="memory ~ delta_similarity_within_z + delta_similarity_subject_mean_z",
            response_family="binomial",
        )
        rows, primary = _extract_param_rows(
            fit,
            roi_set=roi_set,
            roi=roi,
            response="memory",
            model_label="condition_specific",
            condition=condition,
            n_obs=len(memory_frame),
            n_subjects=memory_frame["subject"].nunique(),
            n_pairs=memory_frame["pair_id"].nunique(),
            primary_terms=["delta_similarity_within_z"],
        )
        full_rows.extend(rows)
        if primary is not None:
            primary["effect_type"] = "within_subject_itemlevel"
            primary_rows.append(primary)

        rt_frame = subset[subset["is_correct"]].dropna(
            subset=["log_rt_correct", "delta_similarity_within", "delta_similarity_subject_mean"]
        ).copy()
        if rt_frame["subject"].nunique() < 3 or rt_frame["pair_id"].nunique() < 5:
            continue
        rt_frame["delta_similarity_within_z"] = _zscore(rt_frame["delta_similarity_within"])
        rt_frame["delta_similarity_subject_mean_z"] = _zscore(rt_frame["delta_similarity_subject_mean"])
        rt_fit = _fit_gee(
            rt_frame,
            formula="log_rt_correct ~ delta_similarity_within_z + delta_similarity_subject_mean_z",
            response_family="gaussian",
        )
        rt_rows, _ = _extract_param_rows(
            rt_fit,
            roi_set=roi_set,
            roi=roi,
            response="log_rt_correct",
            model_label="condition_specific",
            condition=condition,
            n_obs=len(rt_frame),
            n_subjects=rt_frame["subject"].nunique(),
            n_pairs=rt_frame["pair_id"].nunique(),
            primary_terms=["delta_similarity_within_z"],
        )
        full_rows.extend(rt_rows)

    return pd.DataFrame(primary_rows), pd.DataFrame(full_rows)


def _fit_interaction_models(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (roi_set, roi), subset in long_frame.groupby(["roi_set", "roi"], sort=False):
        memory_frame = subset.dropna(subset=["memory", "delta_similarity_within", "delta_similarity_subject_mean"]).copy()
        if memory_frame["subject"].nunique() < 3 or memory_frame["pair_id"].nunique() < 10:
            continue
        if memory_frame["condition"].nunique() < 2:
            continue
        memory_frame["delta_similarity_within_z"] = _zscore(memory_frame["delta_similarity_within"])
        memory_frame["delta_similarity_subject_mean_z"] = _zscore(memory_frame["delta_similarity_subject_mean"])
        fit = _fit_gee(
            memory_frame,
            formula="memory ~ C(condition) * delta_similarity_within_z + C(condition) * delta_similarity_subject_mean_z",
            response_family="binomial",
        )
        full_rows, _ = _extract_param_rows(
            fit,
            roi_set=roi_set,
            roi=roi,
            response="memory",
            model_label="condition_interaction",
            condition="Metaphor+Spatial",
            n_obs=len(memory_frame),
            n_subjects=memory_frame["subject"].nunique(),
            n_pairs=memory_frame["pair_id"].nunique(),
            primary_terms=["C(condition)[T.Spatial]:delta_similarity_within_z"],
        )
        rows.extend(full_rows)
    return pd.DataFrame(rows)


def _plot_primary_effects(
    plot_frame: pd.DataFrame,
    figure_path: Path,
    *,
    x_label: str,
    title: str,
) -> None:
    if plot_frame.empty:
        return

    plot_frame["condition"] = pd.Categorical(plot_frame["condition"], categories=["Metaphor", "Spatial"], ordered=True)
    plot_frame = plot_frame.sort_values(["condition", "estimate"], ascending=[True, True]).reset_index(drop=True)

    roi_order = (
        plot_frame.groupby("roi")["estimate"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    y_lookup = {roi: idx for idx, roi in enumerate(roi_order)}
    offsets = {"Metaphor": -0.14, "Spatial": 0.14}
    colors = {"Metaphor": "#b23a48", "Spatial": "#3574b7"}

    fig, ax = plt.subplots(figsize=(8.2, max(4.6, len(roi_order) * 0.48)))
    for condition in ["Metaphor", "Spatial"]:
        subset = plot_frame[plot_frame["condition"] == condition]
        y_values = [y_lookup[roi] + offsets[condition] for roi in subset["roi"]]
        ax.errorbar(
            subset["estimate"],
            y_values,
            xerr=[
                subset["estimate"] - subset["ci_low"],
                subset["ci_high"] - subset["estimate"],
            ],
            fmt="o",
            color=colors[condition],
            ecolor=colors[condition],
            elinewidth=1.6,
            capsize=3.5,
            markersize=5.5,
            label=condition,
        )

    ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(roi_order)))
    ax.set_yticklabels(roi_order, fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel("ROI")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-subject item-level brain-behavior coupling.")
    parser.add_argument(
        "--behavior-trials",
        type=Path,
        default=BASE_DIR / "paper_outputs" / "qc" / "behavior_results" / "refined" / "behavior_trials.tsv",
        help="Refined trial-level behavior table.",
    )
    parser.add_argument(
        "--stimuli-template",
        type=Path,
        default=BASE_DIR / "stimuli_template.csv",
        help="Stimulus template used to resolve pair_id.",
    )
    parser.add_argument(
        "--main-rsa",
        type=Path,
        default=BASE_DIR / "paper_outputs" / "qc" / "rsa_results_optimized_main_functional" / "rsa_itemwise_details.csv",
        help="Main functional item-wise RSA table.",
    )
    parser.add_argument(
        "--literature-rsa",
        type=Path,
        default=BASE_DIR / "paper_outputs" / "qc" / "rsa_results_optimized_literature" / "rsa_itemwise_details.csv",
        help="Literature item-wise RSA table.",
    )
    parser.add_argument(
        "--paper-output-root",
        type=Path,
        default=BASE_DIR / "paper_outputs",
        help="Unified output root for paper tables/figures/qc.",
    )
    args = parser.parse_args()

    paper_root = ensure_dir(args.paper_output_root)
    tables_main = ensure_dir(paper_root / "tables_main")
    tables_si = ensure_dir(paper_root / "tables_si")
    qc_dir = ensure_dir(paper_root / "qc")
    figures_main = ensure_dir(paper_root / "figures_main")

    template = _prepare_template(args.stimuli_template)
    behavior_trials = _prepare_behavior_trials(args.behavior_trials, template)

    neural_frames = {
        "main_functional": _prepare_neural_pair_delta(args.main_rsa, "main_functional"),
    }
    if args.literature_rsa.exists():
        neural_frames["literature"] = _prepare_neural_pair_delta(args.literature_rsa, "literature")

    long_frame, join_qc = _build_itemlevel_long(behavior_trials, neural_frames)
    write_table(long_frame, qc_dir / "itemlevel_behavior_neural_long.tsv")
    neural_pair_delta = pd.concat(neural_frames.values(), ignore_index=True)
    write_table(neural_pair_delta, qc_dir / "itemlevel_neural_pair_delta.tsv")
    write_table(join_qc, qc_dir / "itemlevel_behavior_join_qc.tsv")

    primary, full = _fit_condition_specific_models(long_frame)
    interaction = _fit_interaction_models(long_frame)

    if not primary.empty:
        primary = primary.sort_values(["roi_set", "condition", "p_value", "roi"]).reset_index(drop=True)
    if not full.empty:
        full = full.sort_values(["roi_set", "response", "condition", "roi", "term"]).reset_index(drop=True)
    if not interaction.empty:
        interaction = interaction.sort_values(["roi_set", "roi", "term"]).reset_index(drop=True)

    write_table(primary, tables_main / "table_itemlevel_coupling.tsv")
    write_table(full, tables_si / "table_itemlevel_coupling_full.tsv")
    write_table(interaction, tables_si / "table_itemlevel_coupling_interaction.tsv")

    rt_primary = full[
        (full["response"] == "log_rt_correct")
        & (full["term"] == "delta_similarity_within_z")
    ].copy() if not full.empty else pd.DataFrame()
    write_table(rt_primary, tables_si / "table_itemlevel_coupling_rt.tsv")

    memory_plot = primary[
        (primary["roi_set"] == "main_functional")
        & (primary["response"] == "memory")
        & (primary["term"] == "delta_similarity_within_z")
        & (primary["condition"].isin(["Metaphor", "Spatial"]))
    ].copy() if not primary.empty else pd.DataFrame()
    figure_path = figures_main / "fig_itemlevel_coupling.png"
    _plot_primary_effects(
        memory_plot,
        figure_path,
        x_label="Within-subject item-level coupling coefficient",
        title="Run-7 memory predicted by item-level neural change",
    )

    rt_plot = rt_primary[
        (rt_primary["roi_set"] == "main_functional")
        & (rt_primary["condition"].isin(["Metaphor", "Spatial"]))
    ].copy() if not rt_primary.empty else pd.DataFrame()
    rt_figure_path = figures_main / "fig_itemlevel_coupling_rt.png"
    _plot_primary_effects(
        rt_plot,
        rt_figure_path,
        x_label="Within-subject RT coupling coefficient",
        title="Correct-trial RT predicted by item-level neural change",
    )

    save_json(
        {
            "behavior_trials": str(args.behavior_trials),
            "stimuli_template": str(args.stimuli_template),
            "main_rsa": str(args.main_rsa),
            "literature_rsa": str(args.literature_rsa),
            "paper_output_root": str(args.paper_output_root),
            "n_behavior_rows": int(len(behavior_trials)),
            "n_itemlevel_joined_rows": int(len(long_frame)),
            "n_subjects": int(long_frame["subject"].nunique()),
            "n_main_rois": int(long_frame.loc[long_frame["roi_set"] == "main_functional", "roi"].nunique()),
            "n_literature_rois": int(long_frame.loc[long_frame["roi_set"] == "literature", "roi"].nunique()) if "literature" in long_frame["roi_set"].unique() else 0,
            "primary_table": str(tables_main / "table_itemlevel_coupling.tsv"),
            "figure_memory": str(figure_path),
            "figure_rt": str(rt_figure_path),
        },
        qc_dir / "itemlevel_coupling_manifest.json",
    )


if __name__ == "__main__":
    main()
