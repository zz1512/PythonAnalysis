#!/usr/bin/env python3
"""Stage 6 matched-control ERS and Stage 7 directional-mapping boundary tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
LR_DIR = FINAL_ROOT / "brain_behavior" / "learning_reactivation"
for path in [FINAL_ROOT, LR_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common.final_utils import ensure_dir, write_table  # noqa: E402
from common.roi_library import filter_roi_manifest, load_roi_manifest  # noqa: E402
from lr_utils import (  # noqa: E402
    CONDITIONS,
    CONDITION_LABELS,
    VALID_ORIGINAL_IDS,
    available_covariates,
    bh_fdr,
    condition_item_id,
    extract_original_id,
    fisher_corr,
    fit_gee_by_roi,
    load_mask,
    load_template_map,
    masked_samples,
    read_any,
    zscore_grouped,
)


TERM_YY = "C(condition, Treatment(reference='kj'))[T.yy]"
TERM_STAGE6_PRED = "constituent_ers_pair_mean_avg_z"
TERM_STAGE6_PRED_X_YY = f"{TERM_STAGE6_PRED}:{TERM_YY}"
PRIMARY_STAGE7_ROI_SET = "meta_metaphor"


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def md_table(frame: pd.DataFrame, floatfmt: str = ".4g") -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def _load_metadata(path: Path) -> pd.DataFrame:
    frame = read_any(path).reset_index(drop=True)
    if "word_label" in frame.columns:
        frame["word_label"] = frame["word_label"].astype(str).str.strip()
        frame["original_pair_id"] = frame["word_label"].map(extract_original_id).astype("Int64")
    if "pair_id" in frame.columns:
        frame["template_pair_id"] = pd.to_numeric(frame["pair_id"], errors="coerce").astype("Int64")
    return frame


def _word_index(meta: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, row in meta.iterrows():
        label = str(row.get("word_label", "")).strip()
        if label and label.lower() != "nan" and label not in out:
            out[label] = int(idx)
    return out


def _learning_index(meta: pd.DataFrame, run: int) -> dict[int, int]:
    sub = meta[pd.to_numeric(meta["run"], errors="coerce").eq(run)].copy()
    counts = sub.groupby("original_pair_id").size()
    dup = set(counts[counts > 1].index.tolist())
    out: dict[int, int] = {}
    for idx, row in sub.iterrows():
        original = row.get("original_pair_id")
        if pd.isna(original) or original in dup:
            continue
        out[int(original)] = int(idx)
    return out


def _stage55_ready(out_dir: Path) -> tuple[bool, str]:
    readiness_path = out_dir / "stage55_exploratory_readiness_review.md"
    matching_path = out_dir / "stage55_stage6_matching_manifest.tsv"
    role_path = out_dir / "stage55_stage7_role_coding_table.tsv"
    baseline_path = out_dir / "stage55_stage7_baseline_asymmetry.tsv"
    missing = [p.name for p in [readiness_path, matching_path, role_path, baseline_path] if not p.exists()]
    if missing:
        return False, "missing Stage5.5 outputs: " + ", ".join(missing)
    text = readiness_path.read_text(encoding="utf-8", errors="replace")
    if "Overall status: **pass_with_limitations**" not in text and "Overall status: **pass**" not in text:
        return False, "Stage5.5 readiness review is not pass/pass_with_limitations"
    return True, "Stage5.5 pass_with_limitations gate satisfied"


def _template_lookup(template: pd.DataFrame) -> dict[tuple[str, int, str], str]:
    lookup: dict[tuple[str, int, str], str] = {}
    for _, row in template.iterrows():
        condition = str(row["condition"])
        original = int(row["original_pair_id"])
        lookup[(condition, original, "role_a")] = str(row["role_a_label"])
        lookup[(condition, original, "role_b")] = str(row["role_b_label"])
    return lookup


def _stage6_subject_roi_condition(
    *,
    subject_dir: Path,
    roi_set: str,
    roi: str,
    mask: np.ndarray,
    mask_path: Path,
    condition: str,
    matching: pd.DataFrame,
    label_lookup: dict[tuple[str, int, str], str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    qcs: list[dict[str, object]] = []
    subject = subject_dir.name
    try:
        learn_meta = _load_metadata(subject_dir / f"learn_{condition}_metadata.tsv")
        pre_meta = _load_metadata(subject_dir / f"pre_{condition}_metadata.tsv")
        learn_samples = masked_samples(subject_dir / f"learn_{condition}.nii.gz", mask, mask_path)
        pre_samples = masked_samples(subject_dir / f"pre_{condition}.nii.gz", mask, mask_path)
        if len(learn_meta) != learn_samples.shape[0] or len(pre_meta) != pre_samples.shape[0]:
            raise ValueError("metadata/image row mismatch")
        pre_index = _word_index(pre_meta)
        run_indices = {3: _learning_index(learn_meta, 3), 4: _learning_index(learn_meta, 4)}
        cond_matching = matching[matching["condition"].eq(condition)].copy()
        for _, match in cond_matching.iterrows():
            original = int(match["true_original_pair_id"])
            control_original = int(match["control_original_pair_id"])
            role = str(match["role"])
            if original not in VALID_ORIGINAL_IDS[condition]:
                continue
            true_label = label_lookup.get((condition, original, role))
            control_label = label_lookup.get((condition, control_original, role))
            if not true_label or not control_label:
                continue
            true_idx = pre_index.get(true_label)
            control_idx = pre_index.get(control_label)
            if true_idx is None or control_idx is None:
                continue
            for run in [3, 4]:
                learn_idx = run_indices[run].get(original)
                if learn_idx is None:
                    continue
                learn_vec = learn_samples[learn_idx]
                true_z = fisher_corr(learn_vec, pre_samples[true_idx])
                control_z = fisher_corr(learn_vec, pre_samples[control_idx])
                rows.append(
                    {
                        "subject": subject,
                        "roi_set": roi_set,
                        "roi": roi,
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "run": run,
                        "role": role,
                        "original_pair_id": original,
                        "control_original_pair_id": control_original,
                        "condition_item_id": condition_item_id(condition, original),
                        "control_condition_item_id": condition_item_id(condition, control_original),
                        "true_word_label": true_label,
                        "control_word_label": control_label,
                        "true_word_text": match.get("true_word_text"),
                        "control_word_text": match.get("control_word_text"),
                        "true_similarity_z": true_z,
                        "control_similarity_z": control_z,
                        "constituent_ers": true_z - control_z if np.isfinite(true_z) and np.isfinite(control_z) else np.nan,
                        "match_distance": match.get("match_distance"),
                    }
                )
        qcs.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "ok": True,
                "fail_reason": "",
                "n_rows": len([r for r in rows if r["subject"] == subject and r["condition"] == condition]),
                "n_expected_role_matches": int(len(cond_matching)),
            }
        )
    except Exception as exc:
        qcs.append(
            {
                "subject": subject,
                "roi_set": roi_set,
                "roi": roi,
                "condition": condition,
                "ok": False,
                "fail_reason": str(exc),
                "n_rows": 0,
                "n_expected_role_matches": int(matching["condition"].eq(condition).sum()),
            }
        )
    return rows, qcs


def _stage6_wide(role_run: pd.DataFrame) -> pd.DataFrame:
    idx = [
        "subject",
        "roi_set",
        "roi",
        "condition",
        "condition_label",
        "original_pair_id",
        "condition_item_id",
    ]
    parts = []
    for run in [3, 4]:
        for role in ["role_a", "role_b"]:
            sub = role_run[role_run["run"].eq(run) & role_run["role"].eq(role)][idx + ["constituent_ers"]].copy()
            sub = sub.rename(columns={"constituent_ers": f"run{run}_{role}_constituent_ers"})
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=idx, how="outer")
    for run in [3, 4]:
        out[f"run{run}_pair_mean_constituent_ers"] = out[
            [f"run{run}_role_a_constituent_ers", f"run{run}_role_b_constituent_ers"]
        ].mean(axis=1)
    for role in ["role_a", "role_b"]:
        out[f"{role}_constituent_ers_avg"] = out[
            [f"run3_{role}_constituent_ers", f"run4_{role}_constituent_ers"]
        ].mean(axis=1)
    out["constituent_ers_pair_mean_avg"] = out[
        ["run3_pair_mean_constituent_ers", "run4_pair_mean_constituent_ers"]
    ].mean(axis=1)
    out["constituent_ers_delta_run4_minus_run3"] = (
        out["run4_pair_mean_constituent_ers"] - out["run3_pair_mean_constituent_ers"]
    )
    return out.sort_values(["subject", "roi_set", "roi", "condition", "original_pair_id"]).reset_index(drop=True)


def _one_sample_subject_tests(item: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    subject = (
        item.groupby(["subject", "roi_set", "roi", "condition"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for (roi_set, roi, condition), sub in subject.groupby(["roi_set", "roi", "condition"], dropna=False):
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if len(vals) >= 3:
                t, p = stats.ttest_1samp(vals, 0.0)
                mean = float(vals.mean())
                sd = float(vals.std(ddof=1))
            else:
                t, p, mean, sd = np.nan, np.nan, np.nan, np.nan
            rows.append(
                {
                    "test_type": "one_sample",
                    "metric": metric,
                    "roi_set": roi_set,
                    "roi": roi,
                    "condition": condition,
                    "contrast": f"{condition}>0",
                    "n_subjects": int(len(vals)),
                    "mean_effect": mean,
                    "sd": sd,
                    "t": float(t) if pd.notna(t) else np.nan,
                    "p": float(p) if pd.notna(p) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh_roi_set_metric"] = np.nan
        for _, idx in out.groupby(["test_type", "roi_set", "metric"], dropna=False).groups.items():
            out.loc[idx, "q_bh_roi_set_metric"] = bh_fdr(out.loc[idx, "p"])
    return out


def _condition_contrast_subject_tests(item: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    subject = (
        item.groupby(["subject", "roi_set", "roi", "condition"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for metric in metrics:
        wide = subject.pivot_table(index=["subject", "roi_set", "roi"], columns="condition", values=metric)
        for (subject_roi_set, subject_roi), sub in wide.groupby(level=["roi_set", "roi"], dropna=False):
            if not {"yy", "kj"}.issubset(set(sub.columns)):
                continue
            diff = (sub["yy"] - sub["kj"]).dropna()
            if len(diff) >= 3:
                t, p = stats.ttest_1samp(diff, 0.0)
                mean = float(diff.mean())
                sd = float(diff.std(ddof=1))
            else:
                t, p, mean, sd = np.nan, np.nan, np.nan, np.nan
            rows.append(
                {
                    "test_type": "condition_contrast",
                    "metric": metric,
                    "roi_set": subject_roi_set,
                    "roi": subject_roi,
                    "condition": "yy_minus_kj",
                    "contrast": "YY-KJ",
                    "n_subjects": int(len(diff)),
                    "mean_effect": mean,
                    "sd": sd,
                    "t": float(t) if pd.notna(t) else np.nan,
                    "p": float(p) if pd.notna(p) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh_roi_set_metric"] = np.nan
        for _, idx in out.groupby(["test_type", "roi_set", "metric"], dropna=False).groups.items():
            out.loc[idx, "q_bh_roi_set_metric"] = bh_fdr(out.loc[idx, "p"])
    return out


def run_stage6(args: argparse.Namespace, out_dir: Path, rois: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template = load_template_map(args.stimuli_template)
    label_lookup = _template_lookup(template)
    matching = _read(out_dir / "stage55_stage6_matching_manifest.tsv")
    matching["condition"] = matching["condition"].astype(str).str.lower()
    subject_dirs = sorted([p for p in args.pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if args.subjects:
        wanted = set(args.subjects)
        subject_dirs = [p for p in subject_dirs if p.name in wanted]

    rows: list[dict[str, object]] = []
    qcs: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        for roi_row in rois.itertuples(index=False):
            mask_path = Path(str(roi_row.mask_path))
            try:
                mask = load_mask(mask_path)
            except Exception as exc:
                for condition in CONDITIONS:
                    qcs.append(
                        {
                            "subject": subject_dir.name,
                            "roi_set": str(roi_row.roi_set),
                            "roi": str(roi_row.roi_name),
                            "condition": condition,
                            "ok": False,
                            "fail_reason": f"mask failed: {exc}",
                            "n_rows": 0,
                            "n_expected_role_matches": int(matching["condition"].eq(condition).sum()),
                        }
                    )
                continue
            for condition in CONDITIONS:
                item_rows, item_qc = _stage6_subject_roi_condition(
                    subject_dir=subject_dir,
                    roi_set=str(roi_row.roi_set),
                    roi=str(roi_row.roi_name),
                    mask=mask,
                    mask_path=mask_path,
                    condition=condition,
                    matching=matching,
                    label_lookup=label_lookup,
                )
                rows.extend(item_rows)
                qcs.extend(item_qc)

    role_run = pd.DataFrame(rows)
    item = _stage6_wide(role_run) if not role_run.empty else pd.DataFrame()
    qc = pd.DataFrame(qcs)
    metrics = [
        "role_a_constituent_ers_avg",
        "role_b_constituent_ers_avg",
        "constituent_ers_pair_mean_avg",
        "constituent_ers_delta_run4_minus_run3",
    ]
    group = pd.concat(
        [_one_sample_subject_tests(item, metrics), _condition_contrast_subject_tests(item, metrics)],
        ignore_index=True,
    ) if not item.empty else pd.DataFrame()

    analysis = item.copy()
    model = pd.DataFrame()
    mechanism_path = args.base_dir / "paper_outputs" / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv"
    if not analysis.empty and mechanism_path.exists():
        mech_cols = [
            "subject",
            "roi_set",
            "roi",
            "condition",
            "condition_item_id",
            "post_edge_specificity",
            "pre_pair_similarity_z",
            "sentence_char_len_z",
            "word_frequency_mean_z",
            "stroke_count_mean_z",
            "valence_mean_z",
            "arousal_mean_z",
        ]
        mech = _read(mechanism_path)
        keep = [c for c in mech_cols if c in mech.columns]
        analysis = analysis.merge(mech[keep], on=["subject", "roi_set", "roi", "condition", "condition_item_id"], how="left")
        analysis = zscore_grouped(
            analysis,
            "constituent_ers_pair_mean_avg",
            ["roi_set", "roi"],
            "constituent_ers_pair_mean_avg_z",
        )
        covariates = available_covariates(analysis)
        cov_expr = "".join([f" + {cov}" for cov in covariates])
        formula = (
            "post_edge_specificity ~ constituent_ers_pair_mean_avg_z * "
            "C(condition, Treatment(reference='kj')) + pre_pair_similarity_z"
            + cov_expr
        )
        model = fit_gee_by_roi(
            analysis,
            formula=formula,
            outcome="post_edge_specificity",
            model_name="stage6_ers_to_post_edge",
            min_rows=80,
        )
    write_table(role_run, out_dir / "stage6_matched_control_ers_role_run.tsv")
    write_table(item, out_dir / "stage6_matched_control_ers_item.tsv")
    write_table(qc, out_dir / "stage6_matched_control_ers_qc.tsv")
    write_table(group, out_dir / "stage6_matched_control_ers_group_tests.tsv")
    write_table(analysis, out_dir / "stage6_matched_control_ers_analysis_table.tsv")
    write_table(model, out_dir / "stage6_matched_control_ers_to_post_models.tsv")
    return role_run, item, qc, group, model


def run_stage7(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = _read(args.base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping" / "directional_mapping_item.tsv")
    role = _read(out_dir / "stage55_stage7_role_coding_table.tsv")
    baseline = _read(out_dir / "stage55_stage7_baseline_asymmetry.tsv")
    mapping = mapping.merge(
        role[["condition", "condition_item_id", "role_a_theory", "role_b_theory", "status"]].rename(columns={"status": "role_coding_status"}),
        on=["condition", "condition_item_id"],
        how="left",
    )
    metrics = ["target_source_over_self", "mapping_asymmetry_self_corrected"]
    one = _one_sample_subject_tests(mapping[mapping["condition"].eq("yy")].copy(), metrics)
    one = one[one["condition"].eq("yy")].copy()
    contrast = _condition_contrast_subject_tests(mapping, ["mapping_asymmetry_self_corrected"])
    primary = pd.concat([one, contrast], ignore_index=True)
    primary["primary_scope"] = np.where(primary["roi_set"].eq(PRIMARY_STAGE7_ROI_SET), "primary_meta_metaphor", "secondary_context")
    primary["primary_metric"] = np.select(
        [
            primary["test_type"].eq("one_sample") & primary["metric"].eq("target_source_over_self"),
            primary["test_type"].eq("one_sample") & primary["metric"].eq("mapping_asymmetry_self_corrected"),
            primary["test_type"].eq("condition_contrast") & primary["metric"].eq("mapping_asymmetry_self_corrected"),
        ],
        [
            "target_source_over_self_in_YY",
            "mapping_asymmetry_self_corrected_in_YY",
            "YY_minus_KJ_mapping_asymmetry_self_corrected",
        ],
        default="other",
    )
    primary["q_bh_primary_metric_scope"] = np.nan
    for _, idx in primary.groupby(["primary_metric", "primary_scope"], dropna=False).groups.items():
        primary.loc[idx, "q_bh_primary_metric_scope"] = bh_fdr(primary.loc[idx, "p"])
    readiness = pd.DataFrame(
        [
            {
                "check": "role_coding",
                "status": "pass" if role["status"].eq("ok").all() else "fail",
                "detail": f"{role['status'].eq('ok').sum()}/{len(role)} role rows ok",
            },
            {
                "check": "baseline_asymmetry",
                "status": "pass" if baseline["status"].eq("pass").all() else "fail",
                "detail": f"max abs baseline asymmetry = {_fmt(baseline['max_abs_baseline_asymmetry'].max())}",
            },
        ]
    )
    write_table(mapping, out_dir / "stage7_directional_mapping_boundary_item.tsv")
    write_table(primary, out_dir / "stage7_directional_mapping_primary_tests.tsv")
    write_table(readiness, out_dir / "stage7_directional_mapping_readiness.tsv")
    return mapping, primary, readiness


def _stage6_decision(group: pd.DataFrame, model: pd.DataFrame) -> str:
    if group.empty:
        return "blocked_no_stage6_group_results"
    pair = group[
        group["metric"].eq("constituent_ers_pair_mean_avg")
        & group["test_type"].eq("one_sample")
        & group["q_bh_roi_set_metric"].lt(0.05)
    ]
    yy_kj = group[
        group["metric"].eq("constituent_ers_pair_mean_avg")
        & group["test_type"].eq("condition_contrast")
        & group["q_bh_roi_set_metric"].lt(0.05)
    ]
    post_terms = pd.DataFrame()
    if not model.empty:
        post_terms = model[
            model["term"].isin([TERM_STAGE6_PRED, TERM_STAGE6_PRED_X_YY])
            & model["q"].lt(0.05)
            & model["status"].eq("ok")
        ]
    if not pair.empty and yy_kj.empty and post_terms.empty:
        return "general_constituent_reinstatement_only"
    if not yy_kj.empty and not post_terms.empty:
        return "exploratory_upstream_candidate_with_limitations"
    if yy_kj.empty and post_terms.empty:
        return "no_yy_specific_or_post_predictive_ers"
    return "mixed_exploratory_boundary"


def _stage7_decision(primary: pd.DataFrame) -> str:
    if primary.empty:
        return "blocked_no_stage7_primary_results"
    meta = primary[primary["primary_scope"].eq("primary_meta_metaphor")].copy()
    sig = meta[meta["q_bh_primary_metric_scope"].lt(0.05)]
    if sig.empty:
        return "no_stable_directional_mapping"
    directions = sig.groupby("primary_metric")["mean_effect"].apply(lambda s: set(np.sign(s.dropna()).astype(int))).to_dict()
    mixed = any(len(v) > 1 for v in directions.values())
    if mixed:
        return "significant_but_directionally_mixed_boundary"
    return "directional_mapping_signal_boundary_only"


def write_reviews(out_dir: Path, stage6_group: pd.DataFrame, stage6_model: pd.DataFrame, stage7_primary: pd.DataFrame) -> None:
    s6_decision = _stage6_decision(stage6_group, stage6_model)
    s7_decision = _stage7_decision(stage7_primary)
    s6_top = stage6_group[
        stage6_group["metric"].eq("constituent_ers_pair_mean_avg")
    ].sort_values(["q_bh_roi_set_metric", "p"], na_position="last").head(16)
    s6_post = pd.DataFrame()
    if not stage6_model.empty:
        s6_post = stage6_model[
            stage6_model["term"].isin([TERM_STAGE6_PRED, TERM_STAGE6_PRED_X_YY])
        ].sort_values(["q", "p"], na_position="last").head(16)
    s7_top = stage7_primary.sort_values(["primary_scope", "q_bh_primary_metric_scope", "p"], na_position="last").head(24)

    stage6_review = f"""# Stage 6 Matched-Control Constituent ERS

This is exploratory only and uses the Stage5.5 matched-control manifest.

Decision: **{s6_decision}**

## Primary ERS Group Tests

{md_table(s6_top[["test_type", "metric", "roi_set", "roi", "condition", "contrast", "n_subjects", "mean_effect", "p", "q_bh_roi_set_metric"]])}

## ERS -> Post Edge Models

{md_table(s6_post[["model", "roi_set", "roi", "term", "estimate", "p", "q", "status"]]) if not s6_post.empty else "_No post-edge model terms available._"}

Interpretation rule: a positive true-minus-control ERS effect is general constituent reinstatement unless it is YY-specific and predicts post-stage separation.
"""
    (out_dir / "stage6_matched_control_ers_review.md").write_text(stage6_review, encoding="utf-8")

    stage7_review = f"""# Stage 7 Directional-Mapping Boundary

This is exploratory/boundary only and uses corrected mapping metrics after Stage5.5 role/baseline checks.

Decision: **{s7_decision}**

## Primary Directional Tests

{md_table(s7_top[["primary_scope", "primary_metric", "test_type", "metric", "roi_set", "roi", "condition", "contrast", "n_subjects", "mean_effect", "p", "q_bh_primary_metric_scope"]])}

Interpretation rule: even a positive mapping signal remains boundary evidence unless source-to-target direction is stable, theory-aligned, and robust across meta_metaphor ROIs.
"""
    (out_dir / "stage7_directional_mapping_boundary_review.md").write_text(stage7_review, encoding="utf-8")

    combined = f"""# Stage 6/7 Exploratory Boundary Review

Stage6 decision: **{s6_decision}**

Stage7 decision: **{s7_decision}**

Overall conclusion: Stage6/7 may be reported as supplementary/boundary analyses only. They do not revise the Stage5 primary storyline.
"""
    (out_dir / "stage6_stage7_exploratory_boundary_review.md").write_text(combined, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage6/7 exploratory analyses.")
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--pattern-root", type=Path, default=base_dir / "pattern_root")
    parser.add_argument("--roi-manifest", type=Path, default=base_dir / "roi_library" / "manifest.tsv")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "stagewise_mechanism")
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    ready, ready_detail = _stage55_ready(out_dir)
    if not ready:
        raise RuntimeError(ready_detail)
    manifest = load_roi_manifest(args.roi_manifest)
    roi_frames = [filter_roi_manifest(manifest, roi_set=roi_set, include_flag="include_in_rsa") for roi_set in args.roi_sets]
    rois = pd.concat(roi_frames, ignore_index=True)

    stage6_role, stage6_item, stage6_qc, stage6_group, stage6_model = run_stage6(args, out_dir, rois)
    stage7_item, stage7_primary, stage7_ready = run_stage7(args, out_dir)
    write_reviews(out_dir, stage6_group, stage6_model, stage7_primary)

    summary = {
        "analysis_id": "stage6_stage7_exploratory",
        "stage55_gate": ready_detail,
        "n_stage6_role_run_rows": int(len(stage6_role)),
        "n_stage6_item_rows": int(len(stage6_item)),
        "stage6_qc_ok_rate": float(stage6_qc["ok"].mean()) if "ok" in stage6_qc else None,
        "n_stage6_group_rows": int(len(stage6_group)),
        "n_stage6_model_rows": int(len(stage6_model)),
        "stage6_decision": _stage6_decision(stage6_group, stage6_model),
        "n_stage7_item_rows": int(len(stage7_item)),
        "n_stage7_primary_rows": int(len(stage7_primary)),
        "stage7_decision": _stage7_decision(stage7_primary),
        "out_dir": str(out_dir),
    }
    (out_dir / "stage6_stage7_exploratory_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
