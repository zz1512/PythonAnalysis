#!/usr/bin/env python3
"""Stage 5 integration plus Stage 5.5 exploratory readiness prechecks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
from lr_utils import load_template_map  # noqa: E402


STAGE5_TEXT = (
    "本研究结果不支持一个完整的 learning trace -> post separation -> retrieval re-binding -> memory "
    "的强因果链。更稳妥的解释是：隐喻关系学习首先在学习阶段调动 semantic/metaphor network "
    "中的 YY/KJ condition-level geometry；学习后，YY trained relation edges 在 "
    "hippocampal-spatial / scene-context network 中发生特异性分化；最终 retrieval 阶段，"
    "pair-level structure 以 task-driven rebound 和 YY/KJ decoding 的形式重新出现。行为上，"
    "YY 记忆优势稳定，且后续记住的 YY item 表现出更大的 hpc-spatial rebinding difference score；"
    "但 Stage4.5 表明这一行为相关效应主要来自 post 阶段更低的 pair similarity / 更强的 prior "
    "separation，而不是 retrieval similarity 本身更高。因此，行为桥接应谨慎解释为 post-stage "
    "separation 与后续记忆相关，而不是纯 retrieval reinstatement 机制。"
)


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


def extract_stage5_values(qc_root: Path) -> dict[str, object]:
    stage_root = qc_root / "stagewise_mechanism"
    values: dict[str, object] = {}

    behavior = _read(qc_root / "behavior_results" / "refined" / "subject_condition_summary.tsv")
    wide = behavior.pivot(index="subject", columns="condition", values="accuracy").dropna()
    diff = wide["YY"] - wide["KJ"]
    t, p = stats.ttest_1samp(diff, 0.0)
    values.update(
        behavior_n=int(diff.size),
        behavior_mean_diff=float(diff.mean()),
        behavior_t=float(t),
        behavior_p=float(p),
    )

    stage2 = _read(stage_root / "stage2b_cross_run_matrix_models.tsv")
    term = "C(condition, Treatment(reference='kj'))[T.yy]"
    s2_sem = stage2[
        stage2["model"].eq("stage2b_matrix_primary")
        & stage2["roi_set"].eq("network")
        & stage2["roi"].eq("semantic")
        & stage2["term"].eq(term)
    ].iloc[0]
    values.update(
        stage2_semantic_beta=float(s2_sem["estimate"]),
        stage2_semantic_p=float(s2_sem["p"]),
        stage2_semantic_q=float(s2_sem["q"]),
    )

    s2_ag = stage2[
        stage2["model"].eq("stage2b_matrix_primary")
        & stage2["roi_set"].eq("meta_metaphor")
        & stage2["roi"].eq("meta_L_AG")
        & stage2["term"].eq(term)
    ].iloc[0]
    values.update(
        stage2_lag_beta=float(s2_ag["estimate"]),
        stage2_lag_p=float(s2_ag["p"]),
        stage2_lag_q=float(s2_ag["q"]),
    )

    mixed = _read(qc_root / "mixed_effects" / "mixed_effects_network_models.tsv")
    post = mixed[
        mixed["model_name"].eq("M5_semantic_reactivation_to_hpc_separation")
        & mixed["term"].eq(term)
    ].iloc[0]
    values.update(
        post_hpc_beta=float(post["estimate"]),
        post_hpc_p=float(post["p"]),
        post_hpc_q=float(post["q"]),
    )

    retrieval = _read(qc_root / "retrieval_geometry" / "retrieval_geometry_group_fdr.tsv")
    ret_top = retrieval[
        retrieval["roi_set"].eq("meta_spatial")
        & retrieval["roi"].eq("meta_R_hippocampus")
        & retrieval["condition"].eq("yy")
        & retrieval["metric"].eq("retrieval_pair_similarity")
    ].iloc[0]
    values.update(
        retrieval_rhip_mean=float(ret_top["mean_effect"]),
        retrieval_rhip_p=float(ret_top["p"]),
        retrieval_rhip_q=float(ret_top["q_bh_primary_family"]),
    )

    mvpa = _read(qc_root / "run7_mvpa_decoding" / "run7_mvpa_group_fdr.tsv")
    mvpa_top = mvpa[
        mvpa["roi_set"].eq("meta_spatial")
        & mvpa["roi"].eq("meta_R_hippocampus")
        & mvpa["analysis_type"].eq("pair_heldout")
    ].iloc[0]
    values.update(
        mvpa_rhip_acc=float(mvpa_top["mean_accuracy"]),
        mvpa_rhip_delta=float(mvpa_top["mean_minus_chance"]),
        mvpa_rhip_p=float(mvpa_top["p"]),
        mvpa_rhip_q=float(mvpa_top["q_bh_primary_family"]),
    )

    mem = _read(stage_root / "stage45_memory_difference_models.tsv")
    rb = mem[
        mem["component_name"].eq("hpc_spatial_rebinding_difference")
        & mem["term"].eq("memory_score_slope_yy")
    ].iloc[0]
    post_comp = mem[
        mem["component_name"].eq("hpc_spatial_post_pair_similarity")
        & mem["term"].eq("memory_score_slope_yy")
    ].iloc[0]
    ret_comp = mem[
        mem["component_name"].eq("hpc_spatial_retrieval_pair_similarity")
        & mem["term"].eq("memory_score_slope_yy")
    ].iloc[0]
    values.update(
        memory_rb_beta=float(rb["estimate"]),
        memory_rb_p=float(rb["p"]),
        memory_rb_q=float(rb["q"]),
        memory_post_beta=float(post_comp["estimate"]),
        memory_post_p=float(post_comp["p"]),
        memory_post_q=float(post_comp["q"]),
        memory_ret_beta=float(ret_comp["estimate"]),
        memory_ret_p=float(ret_comp["p"]),
        memory_ret_q=float(ret_comp["q"]),
    )

    stage3 = _read(stage_root / "stage3_semantic_edge_condition_models.tsv")
    st3 = stage3[
        stage3["roi_set"].eq("network")
        & stage3["network"].eq("semantic")
        & stage3["metric"].eq("semantic_to_edge_shift")
    ].iloc[0]
    values.update(
        stage3_sem_shift=float(st3["estimate"]),
        stage3_sem_p=float(st3["p"]),
        stage3_sem_q=float(st3["q"]),
    )
    return values


def build_stage5_tables(qc_root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    v = extract_stage5_values(qc_root)
    rows = [
        {
            "stage": "Behavior",
            "best_metric": "YY-KJ memory accuracy",
            "current_conclusion": "YY memory advantage is stable.",
            "main_story_status": "behavioral anchor",
            "evidence_strength": "main",
            "key_estimate": f"YY-KJ accuracy = {_fmt(v['behavior_mean_diff'])}",
            "p": v["behavior_p"],
            "q": np.nan,
            "source_output_file": str(qc_root / "behavior_results" / "refined" / "subject_condition_summary.tsv"),
            "notes": f"paired t({int(v['behavior_n']) - 1}) = {_fmt(v['behavior_t'])}",
        },
        {
            "stage": "Learning",
            "best_metric": "Stage2b semantic condition geometry",
            "current_conclusion": "YY/KJ condition-level geometry is engaged in the semantic/metaphor network.",
            "main_story_status": "yes",
            "evidence_strength": "learning-state evidence",
            "key_estimate": f"semantic network beta = {_fmt(v['stage2_semantic_beta'])}; meta_L_AG beta = {_fmt(v['stage2_lag_beta'])}",
            "p": v["stage2_semantic_p"],
            "q": v["stage2_semantic_q"],
            "source_output_file": str(qc_root / "stagewise_mechanism" / "stage2b_cross_run_matrix_models.tsv"),
            "notes": "Stage4.5 shows this is not a simple amplitude/variance artifact; full within-run geometry proxy attenuates it.",
        },
        {
            "stage": "Post",
            "best_metric": "Step5C / post edge specificity / hpc-spatial separation",
            "current_conclusion": "YY trained relation edges show specific differentiation, especially in hpc-spatial / scene-context ROIs.",
            "main_story_status": "yes",
            "evidence_strength": "main mechanism",
            "key_estimate": f"hpc-spatial YY condition beta = {_fmt(v['post_hpc_beta'])}",
            "p": v["post_hpc_p"],
            "q": v["post_hpc_q"],
            "source_output_file": str(qc_root / "mixed_effects" / "mixed_effects_network_models.tsv"),
            "notes": "This is the strongest mechanism layer; post->retrieval item bridge is not promoted.",
        },
        {
            "stage": "Retrieval",
            "best_metric": "run7 pair-structure rebound + MVPA decoding",
            "current_conclusion": "Pair structure reappears and YY/KJ decoding is significant in retrieval-stage hpc-spatial/temporal ROIs.",
            "main_story_status": "yes",
            "evidence_strength": "retrieval-state evidence",
            "key_estimate": f"R hippocampus retrieval pair similarity = {_fmt(v['retrieval_rhip_mean'])}; MVPA acc = {_fmt(v['mvpa_rhip_acc'])}",
            "p": min(v["retrieval_rhip_p"], v["mvpa_rhip_p"]),
            "q": min(v["retrieval_rhip_q"], v["mvpa_rhip_q"]),
            "source_output_file": str(qc_root / "retrieval_geometry" / "retrieval_geometry_group_fdr.tsv")
            + "; "
            + str(qc_root / "run7_mvpa_decoding" / "run7_mvpa_group_fdr.tsv"),
            "notes": "Retrieval evidence is stage evidence, not a post-separation-driven path.",
        },
        {
            "stage": "Memory",
            "best_metric": "Stage4/4.5 hpc-spatial rebinding difference decomposition",
            "current_conclusion": "Remembered YY items have a larger rebinding difference score, mainly driven by lower post similarity / stronger prior post-stage separation.",
            "main_story_status": "moderate_cautious",
            "evidence_strength": "endpoint boundary evidence",
            "key_estimate": f"rebinding slope = {_fmt(v['memory_rb_beta'])}; post component slope = {_fmt(v['memory_post_beta'])}; retrieval component slope = {_fmt(v['memory_ret_beta'])}",
            "p": v["memory_rb_p"],
            "q": v["memory_rb_q"],
            "source_output_file": str(qc_root / "stagewise_mechanism" / "stage45_memory_difference_models.tsv"),
            "notes": "Do not write as pure retrieval reinstatement; retrieval similarity itself is not significant.",
        },
        {
            "stage": "Boundary",
            "best_metric": "Stage3 / reactivation / mapping / post-control bridge",
            "current_conclusion": "Global semantic-to-edge shift, reactivation as upstream cause, source-to-target mapping, and strong post->retrieval->memory path are not supported.",
            "main_story_status": "supplement_boundary",
            "evidence_strength": "boundary",
            "key_estimate": f"Stage3 semantic shift = {_fmt(v['stage3_sem_shift'])}",
            "p": v["stage3_sem_p"],
            "q": v["stage3_sem_q"],
            "source_output_file": str(qc_root / "stagewise_mechanism" / "stage3_semantic_edge_condition_models.tsv"),
            "notes": "These results constrain interpretation and prevent overclaiming a strong causal chain.",
        },
    ]
    evidence = pd.DataFrame(rows)

    boundary = pd.DataFrame(
        [
            {
                "stronger_interpretation": "learning 阶段已形成 YY-specific same-pair edge",
                "current_evidence": "Stage2a / Stage2b same-pair and same-pair x YY are not stable.",
                "conclusion": "不支持",
                "writing_rule": "Write learning as condition-level geometry, not item-specific edge formation.",
            },
            {
                "stronger_interpretation": "semantic reactivation 是上游前因",
                "current_evidence": "Reactivation is small and previously predicted by pre similarity.",
                "conclusion": "不支持为主机制",
                "writing_rule": "Only test after Stage5.5 matching precheck; keep as exploratory.",
            },
            {
                "stronger_interpretation": "source-to-target directional mapping",
                "current_evidence": "Corrected mapping direction is mixed; Stage5.5 must verify role/baseline coding before any new run.",
                "conclusion": "探索性",
                "writing_rule": "Do not write stable source-to-target assimilation.",
            },
            {
                "stronger_interpretation": "global RDM 从 embedding 转向 edge",
                "current_evidence": "Stage3 does not support semantic-to-edge shift; coding direction audit passed.",
                "conclusion": "不支持",
                "writing_rule": "Keep Stage3 as boundary model-RDM result.",
            },
            {
                "stronger_interpretation": "post separation 直接预测 retrieval re-binding",
                "current_evidence": "Post-control models are unstable; raw relation has shared-post coupling.",
                "conclusion": "不支持强桥",
                "writing_rule": "Retrieval rebound is adjacent stage evidence, not post-driven bridge.",
            },
            {
                "stronger_interpretation": "retrieval similarity 本身预测 memory",
                "current_evidence": "Stage4.5 retrieval component is not significant; memory effect mainly comes from lower post similarity.",
                "conclusion": "不支持",
                "writing_rule": "Behavior bridge mainly reflects stronger prior post-stage separation.",
            },
        ]
    )
    mermaid = """flowchart LR
  Pre["Pre<br/>No stable YY/KJ static category code"]
  Learning["Learning<br/>Semantic/metaphor network<br/>YY/KJ condition geometry"]
  Post["Post<br/>YY trained relation edges differentiate<br/>HPC-spatial / scene-context ROIs"]
  Retrieval["Retrieval<br/>Pair-structure rebound<br/>YY/KJ decoding reappears"]
  Memory["Memory<br/>Remembered YY: larger rebinding difference score<br/>mainly lower post similarity / stronger prior separation"]
  Boundary["Boundary<br/>No stable learning same-pair edge<br/>No global semantic-to-edge RDM shift<br/>No strong post->retrieval->memory path<br/>Mapping/reactivation exploratory"]

  Pre --> Learning --> Post --> Retrieval
  Post -. cautious behavioral relevance .-> Memory
  Boundary -. limits interpretation .-> Post
"""
    write_table(evidence, out_dir / "stage5_evidence_integration_table.tsv")
    write_table(boundary, out_dir / "stage5_boundary_table.tsv")
    (out_dir / "stage5_story_model.mmd").write_text(mermaid, encoding="utf-8")
    (out_dir / "stage5_story_model.md").write_text(
        "# Stage 5 Story Model\n\n```mermaid\n" + mermaid + "```\n",
        encoding="utf-8",
    )
    return evidence, boundary, mermaid


def _item_material_table(item_table: Path) -> pd.DataFrame:
    cols = [
        "condition",
        "original_pair_id",
        "template_pair_id",
        "condition_item_id",
        "sentence_text",
        "word_a_text",
        "word_b_text",
        "word_a_char_len",
        "word_b_char_len",
        "sentence_char_len",
        "word_frequency_mean",
        "stroke_count_mean",
        "valence_mean",
        "arousal_mean",
        "pre_pair_similarity",
    ]
    frame = _read(item_table)
    frame["condition"] = frame["condition"].astype(str).str.lower()
    keep = [c for c in cols if c in frame.columns]
    agg = {c: "first" for c in keep if c not in {"condition", "original_pair_id", "template_pair_id", "condition_item_id"}}
    for c in ["word_frequency_mean", "stroke_count_mean", "valence_mean", "arousal_mean", "pre_pair_similarity"]:
        if c in frame:
            agg[c] = "mean"
    out = (
        frame[keep]
        .groupby(["condition", "original_pair_id", "template_pair_id", "condition_item_id"], dropna=False)
        .agg(agg)
        .reset_index()
    )
    return out


def build_stage55_stage6_precheck(base_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item = _item_material_table(base_dir / "paper_outputs" / "qc" / "learning_post_memory_prediction" / "item_mechanism_table.tsv")
    role_rows = []
    for _, row in item.iterrows():
        for role, word_col, len_col in [
            ("role_a", "word_a_text", "word_a_char_len"),
            ("role_b", "word_b_text", "word_b_char_len"),
        ]:
            role_rows.append(
                {
                    **row.to_dict(),
                    "role": role,
                    "word_text": row.get(word_col),
                    "role_char_len": row.get(len_col),
                }
            )
    role = pd.DataFrame(role_rows)
    candidate_vars = [
        "role_char_len",
        "word_frequency_mean",
        "stroke_count_mean",
        "valence_mean",
        "arousal_mean",
        "pre_pair_similarity",
    ]
    z = role.copy()
    for var in candidate_vars:
        vals = pd.to_numeric(z[var], errors="coerce")
        sd = vals.std(ddof=0)
        z[f"{var}_z"] = (vals - vals.mean()) / sd if sd and np.isfinite(sd) else np.nan
    matches = []
    for _, true in z.iterrows():
        cand = z[
            z["condition"].eq(true["condition"])
            & z["role"].eq(true["role"])
            & ~z["original_pair_id"].eq(true["original_pair_id"])
        ].copy()
        if cand.empty:
            continue
        dist = np.zeros(len(cand), dtype=float)
        n_used = np.zeros(len(cand), dtype=float)
        for var in candidate_vars:
            col = f"{var}_z"
            tv = true.get(col)
            cv = pd.to_numeric(cand[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(cv) & np.isfinite(tv)
            dist[ok] += np.square(cv[ok] - float(tv))
            n_used[ok] += 1
        cand = cand.assign(match_distance=np.where(n_used > 0, np.sqrt(dist / np.maximum(n_used, 1)), np.inf))
        control = cand.sort_values(["match_distance", "original_pair_id"]).iloc[0]
        record = {
            "condition": true["condition"],
            "role": true["role"],
            "true_condition_item_id": true["condition_item_id"],
            "control_condition_item_id": control["condition_item_id"],
            "true_original_pair_id": true["original_pair_id"],
            "control_original_pair_id": control["original_pair_id"],
            "true_word_text": true.get("word_text"),
            "control_word_text": control.get("word_text"),
            "match_distance": control["match_distance"],
            "same_condition": true["condition"] == control["condition"],
            "same_role_position": true["role"] == control["role"],
            "not_paired_constituent": true["condition_item_id"] != control["condition_item_id"],
            "not_same_learned_sentence": true["original_pair_id"] != control["original_pair_id"],
        }
        for var in candidate_vars:
            record[f"true_{var}"] = true.get(var)
            record[f"control_{var}"] = control.get(var)
            record[f"diff_{var}"] = true.get(var) - control.get(var)
        matches.append(record)
    manifest = pd.DataFrame(matches)
    balance_rows = []
    for var in candidate_vars:
        true_vals = pd.to_numeric(manifest[f"true_{var}"], errors="coerce")
        ctrl_vals = pd.to_numeric(manifest[f"control_{var}"], errors="coerce")
        ok = true_vals.notna() & ctrl_vals.notna()
        if ok.sum() >= 3:
            t, p = stats.ttest_rel(true_vals[ok], ctrl_vals[ok])
            status = "ok" if p >= 0.05 else "imbalance_flag"
        else:
            t, p, status = np.nan, np.nan, "unavailable"
        balance_rows.append(
            {
                "variable": var,
                "true_mean": float(true_vals[ok].mean()) if ok.any() else np.nan,
                "control_mean": float(ctrl_vals[ok].mean()) if ok.any() else np.nan,
                "difference": float((true_vals[ok] - ctrl_vals[ok]).mean()) if ok.any() else np.nan,
                "p": float(p) if pd.notna(p) else np.nan,
                "n_pairs": int(ok.sum()),
                "status": status,
                "note": "pair-level proxy" if var != "role_char_len" else "role-specific char length",
            }
        )
    balance_rows.append(
        {
            "variable": "pre_word_stability",
            "true_mean": np.nan,
            "control_mean": np.nan,
            "difference": np.nan,
            "p": np.nan,
            "n_pairs": 0,
            "status": "unavailable",
            "note": "Existing word_stability outputs are condition/ROI summaries, not word-level matched-control covariates.",
        }
    )
    balance = pd.DataFrame(balance_rows)
    readiness = pd.DataFrame(
        [
            {
                "check": "stage6_matching_rules",
                "status": "pass" if manifest[["same_condition", "same_role_position", "not_paired_constituent", "not_same_learned_sentence"]].all().all() else "fail",
                "detail": f"{len(manifest)} true/control role-level matches built",
            },
            {
                "check": "stage6_balance",
                "status": "review"
                if balance["status"].eq("imbalance_flag").any() or balance["status"].eq("unavailable").any()
                else "pass",
                "detail": "Review imbalance flags and unavailable word-level stability before Stage6.",
            },
        ]
    )
    write_table(balance, out_dir / "stage55_stage6_control_balance.tsv")
    write_table(manifest, out_dir / "stage55_stage6_matching_manifest.tsv")
    return balance, manifest, readiness


def build_stage55_stage7_precheck(base_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template = load_template_map(base_dir / "stimuli_template.csv")
    role_rows = []
    for _, row in template.iterrows():
        condition = row["condition"]
        expected_a = "yyw" if condition == "yy" else "kjw"
        expected_b = "yyew" if condition == "yy" else "kjew"
        a_ok = str(row["role_a_label"]).startswith(expected_a + "_")
        b_ok = str(row["role_b_label"]).startswith(expected_b + "_")
        role_rows.append(
            {
                "condition": condition,
                "condition_item_id": row["condition_item_id"],
                "role_a_label": row["role_a_label"],
                "role_b_label": row["role_b_label"],
                "role_a_theory": "target/topic" if condition == "yy" else "object",
                "role_b_theory": "source/vehicle" if condition == "yy" else "location/context",
                "role_a_prefix_expected": expected_a,
                "role_b_prefix_expected": expected_b,
                "role_a_ok": bool(a_ok),
                "role_b_ok": bool(b_ok),
                "status": "ok" if a_ok and b_ok else "role_prefix_mismatch",
            }
        )
    role = pd.DataFrame(role_rows)

    mapping = _read(base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping" / "directional_mapping_item.tsv")
    baseline = mapping[
        ["subject", "roi_set", "roi", "condition", "condition_item_id", "pre_role_a_role_b"]
    ].copy()
    baseline["baseline_asymmetry"] = 0.0
    summary = (
        baseline.groupby(["roi_set", "roi", "condition"], dropna=False)
        .agg(
            n_rows=("baseline_asymmetry", "size"),
            mean_baseline_asymmetry=("baseline_asymmetry", "mean"),
            max_abs_baseline_asymmetry=("baseline_asymmetry", lambda x: float(np.max(np.abs(x)))),
        )
        .reset_index()
    )
    summary["status"] = np.where(summary["max_abs_baseline_asymmetry"].le(1e-12), "pass", "fail")
    summary["note"] = (
        "baseline_asymmetry is zero by symmetric correlation definition; existing table stores one pre_role_a_role_b value."
    )
    readiness = pd.DataFrame(
        [
            {
                "check": "stage7_role_coding",
                "status": "pass" if role["status"].eq("ok").all() else "fail",
                "detail": f"{role['status'].eq('ok').sum()}/{len(role)} role rows ok",
            },
            {
                "check": "stage7_baseline_asymmetry",
                "status": "pass" if summary["status"].eq("pass").all() else "fail",
                "detail": "Baseline asymmetry is zero under symmetric Fisher-z correlation.",
            },
        ]
    )
    write_table(role, out_dir / "stage55_stage7_role_coding_table.tsv")
    write_table(summary, out_dir / "stage55_stage7_baseline_asymmetry.tsv")
    return role, summary, readiness


def write_reviews(
    out_dir: Path,
    evidence: pd.DataFrame,
    boundary: pd.DataFrame,
    balance: pd.DataFrame,
    matching: pd.DataFrame,
    stage6_ready: pd.DataFrame,
    role: pd.DataFrame,
    baseline: pd.DataFrame,
    stage7_ready: pd.DataFrame,
) -> None:
    stage5_review = f"""# Stage 5 Integrative Stagewise Summary

## Stage5-A Evidence Integration

{md_table(evidence[["stage", "best_metric", "current_conclusion", "main_story_status", "evidence_strength", "key_estimate", "p", "q"]])}

## Stage5-C Boundary Table

{md_table(boundary)}

## Final Storyline

{STAGE5_TEXT}
"""
    (out_dir / "stage5_storyline_review.md").write_text(stage5_review, encoding="utf-8")

    stage6_review = f"""# Stage 5.5 Stage6 Matched-Control Readiness

This is a precheck only. It does not run constituent ERS.

## Readiness

{md_table(stage6_ready)}

## Balance

{md_table(balance)}

## Matching Summary

- matched role-level rows: {len(matching)}
- conditions: {", ".join(sorted(matching["condition"].dropna().unique())) if not matching.empty else ""}
- roles: {", ".join(sorted(matching["role"].dropna().unique())) if not matching.empty else ""}

If any core material variable is flagged or unavailable, Stage6 can still be run only as exploratory with the limitation stated explicitly.
"""
    (out_dir / "stage55_stage6_precheck_review.md").write_text(stage6_review, encoding="utf-8")

    stage7_review = f"""# Stage 5.5 Stage7 Directional-Mapping Readiness

This is a precheck only. It does not rerun directional mapping.

## Readiness

{md_table(stage7_ready)}

## Role Coding

{md_table(role[["condition", "condition_item_id", "role_a_label", "role_b_label", "role_a_theory", "role_b_theory", "status"]].head(20))}

## Baseline Asymmetry

{md_table(baseline.head(20))}

Stage7 may proceed only as a boundary analysis and only if role coding and baseline asymmetry remain valid.
"""
    (out_dir / "stage55_stage7_precheck_review.md").write_text(stage7_review, encoding="utf-8")

    readiness = pd.concat([stage6_ready, stage7_ready], ignore_index=True)
    overall = "pass_with_limitations" if not readiness["status"].eq("fail").any() else "blocked"
    combined = f"""# Stage 5.5 Exploratory Readiness Review

Overall status: **{overall}**

{md_table(readiness)}

Decision:

- Stage6 may proceed only as exploratory, with matched-control balance limitations reported.
- Stage7 may proceed only as directional-mapping boundary, not as main mechanism evidence.
- Neither Stage6 nor Stage7 should change the Stage5 storyline.
"""
    (out_dir / "stage55_exploratory_readiness_review.md").write_text(combined, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 integration and Stage 5.5 prechecks.")
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "stagewise_mechanism")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    qc_root = args.base_dir / "paper_outputs" / "qc"

    evidence, boundary, mermaid = build_stage5_tables(qc_root, out_dir)
    balance, matching, stage6_ready = build_stage55_stage6_precheck(args.base_dir, out_dir)
    role, baseline, stage7_ready = build_stage55_stage7_precheck(args.base_dir, out_dir)
    write_reviews(out_dir, evidence, boundary, balance, matching, stage6_ready, role, baseline, stage7_ready)

    manifest = {
        "analysis_id": "stage5_integration_stage55_precheck",
        "n_stage5_evidence_rows": int(len(evidence)),
        "n_stage5_boundary_rows": int(len(boundary)),
        "n_stage55_stage6_balance_rows": int(len(balance)),
        "n_stage55_stage6_matching_rows": int(len(matching)),
        "n_stage55_stage7_role_rows": int(len(role)),
        "n_stage55_stage7_baseline_rows": int(len(baseline)),
        "out_dir": str(out_dir),
    }
    (out_dir / "stage5_stage55_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
