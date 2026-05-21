#!/usr/bin/env python3
"""M2: ROI mean-activation sanity check.

This script reads pattern NIfTIs and existing Step5C item tables, then writes only
under ``paper_outputs/qc/reviewer_supp/m2_univariate_sanity``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shared import (
    PHASE_TO_STAGE,
    add_common_args,
    bh_fdr,
    default_config,
    existing_table,
    fit_formula,
    ensure_condition_item_id,
    load_mask,
    load_stage,
    masked_samples,
    one_sample_summary,
    roi_masks,
    subject_dirs,
    write_standard_outputs,
    zscore,
)

MODULE = "m2_univariate_sanity"
CONDITIONS = ("yy", "kj", "baseline")


def build_activation_long(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, qc_rows = [], []
    masks_by_set = roi_masks(cfg)
    subjects = subject_dirs(cfg.pattern_root, PHASE_TO_STAGE.values())
    for subject_dir in subjects:
        for phase, stage in PHASE_TO_STAGE.items():
            for condition in CONDITIONS:
                try:
                    meta, data, image_path = load_stage(subject_dir, stage, condition)
                except Exception as exc:
                    qc_rows.append({"subject": subject_dir.name, "phase": phase, "condition": condition, "ok": False, "message": str(exc)})
                    continue
                for roi_set, masks in masks_by_set.items():
                    for roi, mask_path in masks.items():
                        try:
                            _, mask = load_mask(mask_path)
                            samples = masked_samples(data, mask, image_path, mask_path)
                            trial_mean = np.nanmean(samples, axis=1)
                            work = meta.copy()
                            work["roi_set"] = roi_set
                            work["roi"] = roi
                            work["phase"] = phase
                            work["stage"] = stage
                            work["roi_mean_activation"] = trial_mean
                            keep = [c for c in ["subject", "roi_set", "roi", "phase", "stage", "condition", "pair_id", "run", "roi_mean_activation"] if c in work.columns]
                            rows.extend(work[keep].to_dict("records"))
                            qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "phase": phase, "condition": condition, "ok": True, "n_trials": int(len(work))})
                        except Exception as exc:
                            qc_rows.append({"subject": subject_dir.name, "roi_set": roi_set, "roi": roi, "phase": phase, "condition": condition, "ok": False, "message": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def model_univariate(activation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if activation.empty:
        return pd.DataFrame()
    subject_level = activation.groupby(["subject", "roi_set", "roi", "phase", "condition"], dropna=False)["roi_mean_activation"].mean().reset_index()
    for (roi_set, roi), sub in subject_level.groupby(["roi_set", "roi"], dropna=False):
        model = fit_formula(
            sub,
            "roi_mean_activation ~ C(phase) * C(condition)",
            group_col="subject",
            prefer_mixed=True,
            allow_fallback=True,
        )
        model.insert(0, "roi", roi)
        model.insert(0, "roi_set", roi_set)
        rows.append(model)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def step5c_univariate_control(cfg, activation: pd.DataFrame, item_table: Path) -> pd.DataFrame:
    if activation.empty or not item_table.exists():
        return pd.DataFrame([{"status": "skipped", "reason": f"missing activation or item table: {item_table}"}])
    item = ensure_condition_item_id(existing_table(item_table))
    required = {"subject", "condition", "post_edge_specificity"}
    if not required.issubset(item.columns):
        return pd.DataFrame([{"status": "skipped", "reason": f"item table missing {sorted(required - set(item.columns))}"}])
    post = activation[activation["phase"].eq("post") & activation["condition"].isin(["yy", "kj"])].copy()
    post_subject = post.groupby(["subject", "roi_set", "roi", "condition"], dropna=False)["roi_mean_activation"].mean().reset_index()
    merge_cols = ["subject", "condition"]
    if {"roi_set", "roi"}.issubset(item.columns):
        merge_cols.extend(["roi_set", "roi"])
    else:
        post_subject = post_subject.rename(
            columns={
                "roi_set": "post_activation_roi_set",
                "roi": "post_activation_roi",
            }
        )
    merged = item.merge(post_subject, on=merge_cols, how="inner")
    if merged.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no rows after merge"}])
    if "pre_pair_similarity_z" not in merged and "pre_pair_similarity" in merged:
        merged["pre_pair_similarity_z"] = zscore(merged["pre_pair_similarity"])
    merged["roi_mean_activation_z"] = zscore(merged["roi_mean_activation"])
    model_specs = [("post_edge_specificity", "post_edge_specificity")]
    if "trained_edge_drop" in merged.columns:
        model_specs.append(("trained_edge_drop", "trained_edge_drop"))
    if "pseudo_edge_drop" in merged.columns:
        model_specs.append(("pseudo_edge_drop", "pseudo_edge_drop"))
    rows = []
    for (roi_set, roi), sub in merged.groupby(["roi_set", "roi"], dropna=False):
        for metric_name, response_col in model_specs:
            formula = f"{response_col} ~ C(condition, Treatment(reference='kj')) + roi_mean_activation_z"
            if "pre_pair_similarity_z" in sub:
                formula += " + pre_pair_similarity_z"
            model = fit_formula(
                sub.dropna(subset=[response_col, "condition", "roi_mean_activation_z"]),
                formula,
                group_col="subject",
                item_col="condition_item_id" if "condition_item_id" in sub.columns else None,
                prefer_mixed=True,
                allow_fallback=True,
            )
            model.insert(0, "response", metric_name)
            model.insert(0, "roi", roi)
            model.insert(0, "roi_set", roi_set)
            rows.append(model)
        if {"trained_edge_drop", "pseudo_edge_drop", "condition"}.issubset(sub.columns):
            yy = sub[sub["condition"].eq("yy")].copy()
            yy["trained_minus_pseudo"] = yy["trained_edge_drop"] - yy["pseudo_edge_drop"]
            if not yy.empty:
                contrast = fit_formula(
                    yy.dropna(subset=["trained_minus_pseudo", "roi_mean_activation_z"]),
                    "trained_minus_pseudo ~ roi_mean_activation_z",
                    group_col="subject",
                    item_col="condition_item_id" if "condition_item_id" in yy.columns else None,
                    prefer_mixed=True,
                    allow_fallback=True,
                )
                contrast.insert(0, "response", "yy_trained_minus_pseudo")
                contrast.insert(0, "roi", roi)
                contrast.insert(0, "roi_set", roi_set)
                rows.append(contrast)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty and "p" in out.columns:
        ok = out["status"].astype(str).str.startswith(("ok", "fallback"))
        out["q_bh"] = np.nan
        out.loc[ok, "q_bh"] = bh_fdr(out.loc[ok, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--step5c-item-table", type=Path, default=None)
    args = parser.parse_args()
    cfg = default_config(args)
    item_table = args.step5c_item_table or cfg.paper_output_root / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv"
    activation, qc = build_activation_long(cfg)
    subject_summary = activation.groupby(["subject", "roi_set", "roi", "phase", "condition"], dropna=False)["roi_mean_activation"].mean().reset_index() if not activation.empty else pd.DataFrame()
    group = one_sample_summary(subject_summary.assign(cell_mean=subject_summary.get("roi_mean_activation")), ["roi_set", "roi", "phase", "condition"], "cell_mean") if not subject_summary.empty else pd.DataFrame()
    outputs = {
        "roi_mean_activation_long.tsv": activation,
        "roi_mean_activation_subject.tsv": subject_summary,
        "roi_mean_activation_group.tsv": group,
        "roi_mean_activation_models.tsv": model_univariate(activation),
        "step5c_univariate_controlled.tsv": step5c_univariate_control(cfg, activation, item_table),
        "m2_univariate_sanity_qc.tsv": qc,
    }
    write_standard_outputs(cfg, MODULE, outputs)


if __name__ == "__main__":
    main()
