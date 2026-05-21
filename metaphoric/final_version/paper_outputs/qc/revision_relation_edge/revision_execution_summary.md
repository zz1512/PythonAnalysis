# relation_edge_story_revision execution summary

Date: 2026-05-16

## Completed scripts

```text
r0_build_revision_master_table.py
r1_material_distance_novelty_moderation.py
r2_high_low_post_separation_item_profile.py
r3_pre_post_retrieval_geometry.py
r4_network_coupling_revision.py
r5_mvpa_stage_state_summary.py
r6_update_evidence_tier_and_story.py
r7_make_revision_figures.py
r8_append_revision_story.py
```

## Core outputs

```text
paper_outputs/qc/revision_relation_edge/r0_build_revision_master_table/revision_master_network.tsv
paper_outputs/qc/revision_relation_edge/r1_material_distance_novelty_moderation/material_moderation_memory.tsv
paper_outputs/qc/revision_relation_edge/r1_material_distance_novelty_moderation/material_moderation_post_separation.tsv
paper_outputs/qc/revision_relation_edge/r2_high_low_post_separation_item_profile/high_low_post_separation_feature_tests.tsv
paper_outputs/qc/revision_relation_edge/r3_pre_post_retrieval_geometry/trajectory_geometry_review.md
paper_outputs/qc/revision_relation_edge/r4_network_coupling_revision/network_coupling_revision_review.md
paper_outputs/qc/revision_relation_edge/r5_mvpa_stage_state_summary/mvpa_boundary_table.tsv
paper_outputs/qc/revision_relation_edge/r6_update_evidence_tier_and_story/storyline_revision.md
paper_outputs/figures_revision/figure_revision_manifest.tsv
result_story_revision.md
result_final_revision.md
```

## Main findings from this execution pass

1. R0 schema audit passed. `YY_28` and `KJ_28` remain distinct, with 26 subjects, 70 condition items, and 3,640 network-level rows.
2. Run3/run4 behavioral proxy fields remain unavailable in the clean refined behavior table, so `run3_understand`, `run4_like`, `run3_rt_z`, and `run4_rt_z` are explicit missing placeholders.
3. R1 material moderation does not support a simple semantic-distance or novelty explanation of memory. YY x novelty for memory is not stable in either network.
4. R1 post-separation moderation shows strong dependence on pre-pair similarity overall. YY-specific novelty moderation for post separation is weak-to-borderline and should be treated as support/boundary, not a new main mechanism.
5. R2 high/low post-separation profiling shows high-separation YY items are mainly distinguished by higher pre-pair similarity. Novelty and memory differences are not stable enough to become main claims.
6. R3 trajectory geometry is a scalar pair-similarity proxy. It does not provide strong evidence for a clean return-to-pre or a strong new-state reconstruction claim; voxel vector alignment is unavailable in this pass.
7. R4 network coupling preserves the strongest B3-style result: semantic retrieval rebound strongly predicts hpc-spatial retrieval rebound. YY-specific interaction and coupling-memory bridge do not survive as main claims.
8. R5 MVPA should be retained as stage-state evidence: learning and retrieval decoding are robust; pre decoding and MVPA-behavior bridge remain boundary evidence.

## Story implication

The current main story should not be overturned. The revision analyses sharpen it:

```text
Successful metaphor learning does not simply increase similarity between associated concepts.
Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood,
creating relation-edge representations that can be task-dependently reconstructed during retrieval.
```

The safest Chinese version is:

```text
隐喻学习的关键不是让两个概念更相似，而是把 trained relation edge 从原有语义邻域中分化出来；
这种分化后的 relation-edge 能在 retrieval 阶段被任务依赖地重新组织。
后续被记住的 YY items 主要表现为 stronger prior post-stage separation。
```

## Writing boundary

Do not claim:

```text
learning item trace -> post separation -> retrieval rebinding -> memory
retrieval reinstatement explains memory
semantic-hpc coupling is YY-specific causal communication
trajectory geometry proves new-state reconstruction
novelty explains YY memory advantage
```
