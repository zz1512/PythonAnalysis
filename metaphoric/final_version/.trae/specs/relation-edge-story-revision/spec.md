# Spec: relation_edge_story_revision

## 1. Objective

This spec defines the next revision stage for the metaphor relation-edge manuscript. The goal is not to search for a new main effect, but to strengthen the current story around two review-facing questions:

```text
How does post-learning relation-edge differentiation relate to memory?
Is retrieval a return to pre-learning geometry, or a task-driven reconstruction?
```

Current main story remains locked:

```text
Behavior:
YY memory advantage

Learning:
YY/KJ condition-level semantic geometry emerges

Post:
YY trained relation edges show specific differentiation

Retrieval:
YY pair structure rebounds in a task-driven manner

Memory:
remembered YY items show stronger prior post-stage separation
```

The upgraded wording should be:

```text
Successful metaphor learning does not simply increase similarity between associated concepts.
Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood,
creating relation-edge representations that can be task-dependently reconstructed during retrieval.
```

Chinese manuscript wording:

```text
成功的隐喻学习不是简单增强词对相似性，而是在 post 阶段将 trained relation edge
从原有语义邻域中分化出来；这种分化后的 relation-edge 在 retrieval 阶段以任务依赖方式
重新组织，并且后续被记住的 YY items 主要表现为 stronger prior post-stage separation。
```

## 2. Story Fit Assessment

### High-fit tasks

These tasks directly answer the two revision questions and should be treated as the primary execution path:

```text
r0_build_revision_master_table.py
r1_material_distance_novelty_moderation.py
r3_pre_post_retrieval_geometry.py
r4_network_coupling_revision.py
r6_update_evidence_tier_and_story.py
r7_make_revision_figures.py
```

Rationale:

- `r0` prevents ID drift and is mandatory for every later model.
- `r1` tests whether material semantic distance / novelty explains memory and post separation.
- `r3` is the strongest conceptual upgrade because it distinguishes return-to-pre from task-driven reconstruction.
- `r4` refines the existing semantic-hpc coupling result into a constrained network-level mechanism.
- `r6` prevents result expansion from damaging the main story.
- `r7` makes the revised structure visible in figures.

### Medium-fit tasks

These tasks are valuable, but should not become the main evidence unless their results are unusually clean:

```text
r2_high_low_post_separation_item_profile.py
r5_mvpa_stage_state_summary.py
```

Rationale:

- `r2` can identify which YY items are more likely to separate, but it is descriptive and material-profile oriented.
- `r5` helps place MVPA back into the paper as stage-state evidence, but MVPA should not carry the edge-reorganization mechanism.

### Writing-only integration task

```text
r8_append_revision_story.py
```

This task should run only after `r6` and `r7`, and should produce revision drafts rather than overwrite the current final documents by default.

## 3. Scope

### Source-of-truth policy

All revision analyses should be source-first:

```text
Behavior:
E:/python_metaphor/data_events/sub-XX/sub-XX_run-3_events.csv
E:/python_metaphor/data_events/sub-XX/sub-XX_run-4_events.csv
E:/python_metaphor/data_events/sub-XX/sub-XX_run-7_events.csv/tsv

Neural patterns:
E:/python_metaphor/pattern_root/
E:/python_metaphor/lss_betas_final/
E:/python_metaphor/roi_library/

Materials:
E:/python_metaphor/materials_detail/
E:/python_metaphor/stimuli_template.csv
E:/python_metaphor/stimulus_embeddings/
```

Locked upstream tables under `E:/python_metaphor/paper_outputs/` may be used only when they are frozen outputs from these source-level pipelines and their provenance is recorded. New scripts must report whether a variable is `source`, `source-derived`, or `locked_upstream_neural_summary`.

The first revision audit script is:

```text
r0a_source_inventory.py
r0b_compare_source_recompute.py
```

`r0a` records source schemas and variable provenance before downstream model execution. `r0b` compares any source-recomputed neural table against the locked upstream table before the upstream table is trusted.

### In scope

- Build a single revision master table using `condition_item_id`.
- Restrict primary analyses to:

```text
semantic/metaphor network
hpc-spatial network
```

- Test material semantic distance and novelty as moderators of memory and post separation.
- Characterize pre-post-retrieval geometry.
- Revisit semantic-hpc network coupling at the predefined network level.
- Summarize MVPA as stage-state evidence.
- Re-tier evidence into Main text, Supplementary, and Boundary / negative evidence.
- Generate revision figures and draft revised story documents.

### Out of scope

- Adding a new main ROI family.
- Large ROI x ROI coupling scans.
- Large mediation or causal-chain models.
- Using bare `pair_id` as random item effect.
- Treating uncorrected p-values as main evidence.
- Calling B2 strict crossnobis.
- Treating KJ as a null control.
- Treating HPC long-axis as a hippocampal subfield mechanism.
- Treating retrieval-post difference score as direct retrieval reinstatement.

## 4. Required Directories

Scripts:

```text
paper_analysis/revision_relation_edge/
```

QC outputs:

```text
paper_outputs/qc/revision_relation_edge/
```

Figures:

```text
paper_outputs/figures_revision/
```

Final manuscript-facing figure copies may later be migrated into:

```text
figure_final/
```

## 5. Required Scripts

```text
r0a_source_inventory.py
r0b_compare_source_recompute.py
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

## 6. Evidence Hierarchy

### Tier A: main-text core

```text
Behavior YY memory advantage
Learning condition-level semantic geometry
Post Step5C / edge specificity
Single-word stability
Stage trajectory post dip + retrieval rebound
Memory component: remembered YY = stronger prior post separation
Network coupling: retrieval semantic-hpc coordination
```

### Tier B: main-text support or important supplement

```text
Mahalanobis proxy
MVPA stage-state summary
Material moderation
Trajectory geometry
Network coupling revision
HPC long-axis anatomical robustness
```

### Tier C: supplementary boundary

```text
relation-vector
global RDM shift
directional mapping
learning reactivation
ERS
MDS
novelty moderation if unstable
input-output slope if unstable
HPC AP gradient
MVPA-behavior bridge
```

## 7. Interpretation Rules

### Allowed claims

- YY memory advantage is the behavioral anchor.
- Learning shows condition-level semantic geometry.
- Post-learning trained-edge differentiation is the core mechanism evidence.
- Retrieval is task-driven reconstruction / rebound.
- Remembered YY items show stronger prior post-stage separation.
- KJ is an active spatial-relational comparison condition.
- MVPA indexes stage-state condition information.
- HPC long-axis is exploratory anatomical robustness.

### Disallowed claims

- YY simply increases pair similarity.
- Learning item trace causes post separation.
- Post separation causes memory.
- Retrieval reinstatement explains memory.
- Semantic-hpc coupling is YY-specific causal communication unless the revised model supports it.
- MDS proves geometry.
- HPC long-axis proves subfield mechanism.
- Uncorrected-only results are evidence.

## 8. Go / No-Go Criteria

### Minimum executable version

If time is limited, run only:

```text
r0_build_revision_master_table.py
r1_material_distance_novelty_moderation.py
r3_pre_post_retrieval_geometry.py
r6_update_evidence_tier_and_story.py
r7_make_revision_figures.py
```

This is sufficient for:

```text
material moderation
trajectory geometry
story rewrite
main figure reordering
evidence tiering
```

### Full execution version

Run all scripts from `r0` through `r8`.

### Story upgrade rule

The story can be upgraded if at least one of the following is true:

- Material moderation shows semantic distance or novelty explains which YY items separate or are remembered.
- Trajectory geometry shows retrieval is not merely returning to pre, but reconstructing a new relation state.
- Network coupling revision shows retrieval-stage semantic-hpc coordination is stronger or memory-relevant.

### Fallback rule

If new moderators are weak or unstable, keep the current story and use revision analyses as boundary evidence:

```text
YY post-stage differentiation remains the core result,
and the new analyses show it is not trivially explained by novelty, pre similarity,
or a simple return-to-pre retrieval account.
```
