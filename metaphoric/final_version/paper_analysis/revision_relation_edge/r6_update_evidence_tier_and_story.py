#!/usr/bin/env python3
"""Task r6: update evidence tier and revised storyline."""

from __future__ import annotations

import pandas as pd

from shared_revision import out_path, read_table, write_text, write_tsv

MODULE = "r6_update_evidence_tier_and_story"
ROOT = out_path("r0_build_revision_master_table", "").parents[0]


def safe_read(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if path.exists():
        return read_table(path)
    return pd.DataFrame()


def get_term(df: pd.DataFrame, model: str | None, term: str) -> dict:
    if df.empty:
        return {"estimate": pd.NA, "p": pd.NA, "q": pd.NA, "status": "missing"}
    sub = df.copy()
    if model is not None and "model" in sub.columns:
        sub = sub[sub["model"].eq(model)]
    sub = sub[sub["term"].astype(str).eq(term)]
    if sub.empty:
        return {"estimate": pd.NA, "p": pd.NA, "q": pd.NA, "status": "missing_term"}
    row = sub.iloc[0]
    return {
        "estimate": row.get("estimate", pd.NA),
        "p": row.get("p", pd.NA),
        "q": row.get("q_bh_all_models", row.get("q_bh", pd.NA)),
        "status": row.get("status", "ok"),
    }


def tier_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r1_mem = safe_read("r1_material_distance_novelty_moderation/material_moderation_memory.tsv")
    r1_post = safe_read("r1_material_distance_novelty_moderation/material_moderation_post_separation.tsv")
    r3 = safe_read("r3_pre_post_retrieval_geometry/trajectory_geometry_models.tsv")
    r4_rebound = safe_read("r4_network_coupling_revision/network_coupling_retrieval_rebound_models.tsv")
    r4_stage = safe_read("r4_network_coupling_revision/network_coupling_stage_models.tsv")
    r4_mem = safe_read("r4_network_coupling_revision/network_coupling_memory_models.tsv")
    r5 = safe_read("r5_mvpa_stage_state_summary/mvpa_boundary_table.tsv")

    rows = [
        {"tier": "A", "analysis": "Behavior YY memory advantage", "claim": "existing core result", "support": "keep from result_new_meta_roi/result_final", "boundary": ""},
        {"tier": "A", "analysis": "Learning condition-level semantic geometry", "claim": "existing core result", "support": "keep from final_nc_converge", "boundary": "not item-specific causal trace"},
        {"tier": "A", "analysis": "Post Step5C / edge specificity", "claim": "existing core result", "support": "keep from final_nc_converge", "boundary": "do not call generic similarity drop strict pattern separation"},
        {"tier": "A", "analysis": "Memory component", "claim": "remembered YY = stronger prior post separation", "support": "keep from A4", "boundary": "not pure retrieval reinstatement"},
    ]
    for network in ["hpc_spatial", "semantic"]:
        sub_post = r1_post[r1_post["network"].eq(network)] if not r1_post.empty else pd.DataFrame()
        nov = get_term(sub_post, "post_separation_material_distance_novelty", "C(condition)[T.yy]:novelty_z")
        pre = get_term(sub_post, "post_separation_material_distance_novelty", "C(condition)[T.yy]:pre_pair_similarity_z")
        rows.append({"tier": "B", "analysis": f"Material moderation post separation {network}", "claim": "novelty/semantic distance boundary", "support": f"novelty interaction estimate={nov['estimate']}, q={nov['q']}; pre interaction estimate={pre['estimate']}, q={pre['q']}", "boundary": "support only if corrected; otherwise boundary"})
        sub_mem = r1_mem[r1_mem["network"].eq(network)] if not r1_mem.empty else pd.DataFrame()
        memnov = get_term(sub_mem, "memory_material_distance_novelty", "C(condition)[T.yy]:novelty_z")
        rows.append({"tier": "B", "analysis": f"Material moderation memory {network}", "claim": "material covariates do not explain memory advantage", "support": f"YY x novelty memory estimate={memnov['estimate']}, q={memnov['q']}", "boundary": "weak/non-significant interactions are boundary evidence"})
    tr = get_term(r3, "return_to_pre_index", "C(condition)[T.yy]")
    rows.append({"tier": "B", "analysis": "Trajectory geometry", "claim": "proxy analysis distinguishes return-to-pre vs reconstruction", "support": f"return_to_pre YY estimate={tr['estimate']}, q={tr['q']}", "boundary": "proxy from pair similarities; vector alignment unavailable"})
    rb = get_term(r4_rebound, "retrieval_rebound_coupling", "semantic_retrieval_rebound_z")
    rb_int = get_term(r4_rebound, "retrieval_rebound_coupling", "semantic_retrieval_rebound_z:C(condition)[T.yy]")
    rows.append({"tier": "A", "analysis": "Network coupling retrieval rebound", "claim": "retrieval-stage semantic-hpc coordination", "support": f"main coupling estimate={rb['estimate']}, q={rb['q']}; YY interaction estimate={rb_int['estimate']}, q={rb_int['q']}", "boundary": "not YY-specific unless interaction survives correction"})
    st = get_term(r4_stage, "stage_coupling", "C(condition)[T.yy]:C(stage)[T.retrieval]")
    rows.append({"tier": "B", "analysis": "Network coupling stage model", "claim": "retrieval stage coupling trend", "support": f"YY x retrieval estimate={st['estimate']}, q={st['q']}", "boundary": "stage coupling product proxy"})
    memc = get_term(r4_mem, "memory_coupling", "C(condition)[T.yy]:semantic_hpc_coupling_retrieval_z")
    rows.append({"tier": "B", "analysis": "Network coupling memory model", "claim": "coupling-memory bridge", "support": f"YY x retrieval coupling estimate={memc['estimate']}, q={memc['q']}", "boundary": "boundary if non-significant"})
    if not r5.empty:
        for _, row in r5.iterrows():
            tier = "B" if row["stage_state"] in {"learning", "post", "retrieval"} else "C"
            rows.append({"tier": tier, "analysis": f"MVPA {row['stage_state']}", "claim": row["claim"], "support": f"n_q_lt_05={row['n_q_lt_05']}; best_q={row['best_q']}", "boundary": row["interpretation"]})
    evidence = pd.DataFrame(rows)
    return evidence, evidence[evidence["tier"].eq("A")], evidence[evidence["tier"].eq("B")], evidence[evidence["tier"].eq("C")]


def storyline(evidence: pd.DataFrame) -> str:
    table = evidence.to_csv(sep="\t", index=False)
    return """# Storyline revision

## Core upgraded story

成功的隐喻学习不是简单增强词对相似性，而是在 post 阶段将 trained relation edge 从原有语义邻域中分化出来；这种分化后的 relation-edge 在 retrieval 阶段以任务依赖方式重新组织，并且后续被记住的 YY items 主要表现为 stronger prior post-stage separation。

English:

Successful metaphor learning does not simply increase similarity between associated concepts. Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood, creating relation-edge representations that can be task-dependently reconstructed during retrieval.

## What the new revision analyses add

1. Material moderation: the new r1 models do not support a simple novelty-only or semantic-distance-only account of memory. Post separation remains strongly tied to pre similarity, and YY-specific novelty moderation is at most a support/boundary result rather than a new main mechanism.
2. High/low item profile: high-separation YY items are mainly distinguished by stronger pre-pair similarity profiles; novelty and memory differences are not stable enough to become a main claim.
3. Trajectory geometry: the scalar trajectory proxy does not provide strong evidence that retrieval is simply a return to pre. Because voxel vector alignment is unavailable in this pass, this should be written as boundary evidence, not as proof of new-state reconstruction.
4. Network coupling: semantic retrieval rebound strongly predicts hpc-spatial retrieval rebound. YY-specific coupling and coupling-memory interactions are weaker, so the claim should remain retrieval-stage inter-network coordination rather than YY-specific causal communication.
5. MVPA: learning and retrieval decoding are useful stage-state evidence; MVPA-behavior bridge and cross-role claims remain supplementary/boundary.

## Revised main-text implication

The existing core story remains intact. The revision analyses sharpen the boundaries: YY differentiation is not trivially explained by novelty, material distance, or a simple retrieval-post difference score. The strongest wording is therefore stagewise representational reorganization, with post-stage trained-edge differentiation as the central mechanism and retrieval-stage semantic-hpc coordination as task-driven reconstruction support.

## Evidence tier table

```tsv
""" + table + "```\n"


def main() -> None:
    evidence, main, supp, boundary = tier_tables()
    write_tsv(evidence, out_path(MODULE, "revision_evidence_tier_table.tsv"))
    write_tsv(main, out_path(MODULE, "main_text_result_map.tsv"))
    write_tsv(supp, out_path(MODULE, "supplementary_result_map.tsv"))
    write_tsv(boundary, out_path(MODULE, "boundary_result_map.tsv"))
    write_text(storyline(evidence), out_path(MODULE, "storyline_revision.md"))
    print(f"Wrote r6 outputs to {out_path(MODULE, '').parent}")


if __name__ == "__main__":
    main()
