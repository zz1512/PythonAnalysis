# Spec: NC Converge Stagewise Mechanism Plan

## 1. Objective

This spec converts the current paper strategy into a constrained execution plan for the final analysis phase. The goal is not to expand the search space, but to consolidate the strongest story into a robust, review-ready evidence package.

Core positioning:

```text
Metaphorical learning transforms pre-existing semantic geometry into
differentiated, task-retrievable relation-edge representations.
```

Chinese version:

```text
隐喻学习将既有语义几何转化为分化的、可在提取阶段重新调动的关系边表征。
```

## 2. Scope

### In scope

- Revise and execute `nc_converge` as the primary analysis suite.
- Split A4 into an explicit post/retrieval memory component model and a stage trajectory model.
- Rename or reframe B1 as staged trajectory characterization, not causal path analysis.
- Restrict B3 to pre-defined network-level coupling:

```text
semantic/metaphor network <-> hpc-spatial / scene-context network
```

- Add at most three mechanism upgrades after `nc_converge`:

```text
C1 input-output transformation
C2 global similarity / representational centrality
C3 novelty / familiarity moderation
```

- Use Go/No-Go criteria to decide whether to enter writing or run exploratory hippocampal long-axis analyses.

### Out of scope

- New large-scale searchlight analysis.
- Broad ROI x ROI coupling scans.
- Large mediation or causal-chain models.
- Treating KJ as a null control.
- Making relation-vector, source-to-target directional mapping, or run7 cross-role generalization part of the main story.
- Claiming CA3/DG or hippocampal subfield mechanisms from MNI y-axis segmentation.

## 3. Main Story Lock

The Results and Discussion should converge on the following structure:

```text
Behavior:
YY memory advantage

Learning:
YY/KJ condition-level geometry emerges in semantic/metaphor network

Post:
YY trained relation edges show specific differentiation,
especially in hippocampal-spatial / scene-context network

Retrieval:
YY pair-structure rebounds and YY/KJ decoding reappears

Memory:
remembered YY items show stronger prior post-stage separation,
not pure retrieval reinstatement
```

The manuscript should not claim a full causal chain:

```text
learning trace -> post separation -> retrieval re-binding -> memory
```

Preferred framing:

```text
learning condition geometry
-> post trained-edge differentiation
-> retrieval task-driven rebound
```

Behavioral bridge:

```text
remembered YY items show larger rebinding difference mainly because of
lower post-stage pair similarity / stronger prior separation,
not because retrieval similarity itself is globally higher.
```

## 4. Primary Analysis Suite: nc_converge

Priority modules:

```text
A2 schema lock
A3 shared-post review
A4 post/retrieval component model
A5 network composite
B2 crossnobis robustness
B3 representational coupling
A6 FDR tiering
Go/No-Go
```

Most critical modules:

```text
B2 crossnobis robustness
A4 post vs retrieval component memory model
B3 semantic/metaphor <-> hpc-spatial network coupling
```

Interpretation gate:

```text
If at least two of A4, B2, and B3 support the main direction,
the manuscript can move into Results/Discussion consolidation.
```

## 5. Required Script Revisions

### A4-1: Memory Component Model

Purpose:

```text
Test whether later memory is associated with lower post similarity,
higher retrieval similarity, or both.
```

Model:

```text
memory_successes ~ condition * post_pair_similarity_z
                 + condition * retrieval_pair_similarity_z
                 + covariates
                 + (1|subject)
                 + (1|condition_item_id)
```

Expected output:

```text
paper_outputs/qc/nc_converge/a4_post_memory_component/
  memory_component_model.tsv
  memory_component_roi_followup.tsv
  memory_component_input_manifest.tsv
```

Primary interpretation:

- YY remembered items should show stronger prior post-stage separation.
- Retrieval similarity should be interpreted cautiously unless it independently predicts memory.

### A4-2: Stage Trajectory Model

Purpose:

```text
Test whether YY follows a pre -> post drop and post -> retrieval rebound trajectory.
```

Model:

```text
pair_similarity ~ condition * stage
                + covariates
                + (1|subject)
                + (1|condition_item_id)
```

Required long-form stages:

```text
pre
post
retrieval
```

Expected output:

```text
paper_outputs/qc/nc_converge/a4_post_memory_component/
  stage_trajectory_model.tsv
  stage_trajectory_descriptives.tsv
```

Primary interpretation:

- YY post-stage reduction supports trained-edge differentiation.
- Retrieval rebound supports task-driven reactivation/rebinding.
- This is trajectory characterization, not a causal path.

### B1: Staged Trajectory Characterization

Rename or reframe:

```text
old: B1 staged path
new: B1 staged trajectory characterization
```

Allowed claims:

- Direct associations among stage-level metrics.
- Consistency of the learning/post/retrieval sequence.
- Boundary conditions for memory interpretation.

Disallowed claims:

- Mediation.
- Causal path.
- Full item-specific chain from learning trace to memory.

### B3: Network-Level Coupling

Restrict to:

```text
semantic/metaphor network <-> hpc-spatial network
```

Primary planned tests:

```text
learning semantic condition geometry <-> post hpc-spatial separation
retrieval semantic rebound <-> retrieval hpc-spatial rebound
```

Do not run:

```text
ROI x ROI exploratory scan
```

## 6. Mechanism Upgrade Analyses

These analyses should be run only after the primary `nc_converge` suite has produced usable outputs.

### C1: Input-Output Transformation

Purpose:

```text
Avoid equating any post similarity drop with pattern separation.
Test whether metaphor learning changes the transformation from pre-stage input
similarity to post-stage output similarity.
```

Model:

```text
post_pair_similarity ~ pre_pair_similarity * condition
                     + covariates
                     + (1|subject)
                     + (1|condition_item_id)
```

Key result:

```text
YY lower pre->post slope
```

Interpretation:

```text
Metaphor learning transforms existing semantic geometry into differentiated
relation-edge geometry.
```

### C2: Global Similarity / Representational Centrality

Purpose:

```text
Test whether remembered YY items become less globally central in the post-stage
representational space, not only less similar to their trained partner.
```

Metrics:

```text
pre_global_similarity_i =
  mean Sim(pre item_i, all other pre items)

post_global_similarity_i =
  mean Sim(post item_i, all other post items)

global_similarity_change =
  post_global_similarity_i - pre_global_similarity_i
```

Models:

```text
global_similarity_change ~ condition * memory + covariates
post_edge_differentiation ~ global_similarity_change * condition + covariates
```

Use only if item x item RDM outputs are complete enough for stable extraction.

### C3: Novelty / Familiarity Moderation

Purpose:

```text
Test whether YY edge differentiation is reducible to novelty or material-level
stimulus properties.
```

Model:

```text
post_edge_differentiation ~ condition
                          + novelty
                          + condition * novelty
                          + familiarity
                          + comprehensibility
                          + difficulty
                          + covariates
                          + (1|subject)
                          + (1|condition_item_id)
```

Interpretation:

- If condition remains stable: YY differentiation is not merely novelty.
- If condition x novelty is stable: higher novelty metaphors induce stronger edge differentiation.

## 7. Optional Analysis: Learning-Update Proxy

Run only if the primary suite is stable and time permits.

Candidate metric:

```text
learning_update = z(RT_run3) - z(RT_run4)
```

Model:

```text
post_edge_differentiation ~ learning_update * condition
                          + covariates
                          + (1|subject)
                          + (1|condition_item_id)
```

Interpret as a weak prediction-error / learning-update proxy, not as direct prediction-error evidence.

## 8. Exploratory Extension: hpc_subfield_three_axis

This suite is allowed only after `nc_converge` supports the hpc-spatial direction.

Required label:

```text
exploratory hippocampal long-axis extension
```

Do not label as:

```text
definitive hippocampal subfield mechanism
CA3/DG mechanism
```

Entry conditions:

```text
1. R hippocampus / hpc-spatial main direction remains stable in nc_converge.
2. B2 crossnobis supports the Step5C direction.
3. head/body/tail mask voxel counts and signal quality pass QC.
4. MNI y-axis thirds are described as exploratory long-axis segmentation.
5. MDS is used as visualization only, not statistical evidence.
```

## 9. Output Policy

All new analysis outputs must be sandboxed under:

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/
```

or, for the exploratory hippocampal extension:

```text
E:/python_metaphor/paper_outputs/qc/hpc_subfield_three_axis/
```

No existing upstream analysis outputs should be modified.

Main text updates should be appended only through the designated append scripts and only after dry-run review.

