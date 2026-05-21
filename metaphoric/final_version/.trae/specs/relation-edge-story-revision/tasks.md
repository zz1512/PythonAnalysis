# Tasks: relation_edge_story_revision

## Execution Status: 2026-05-16

- [x] Phase 0 preflight completed.
- [x] First batch completed: `r0`, `r1`, `r2`, `r3`.
- [x] Second batch completed: `r4`, `r5`.
- [x] Third batch completed: `r6`, `r7`, `r8`.
- [x] Revision figures written under `paper_outputs/figures_revision/`.
- [x] Revision drafts written to `result_story_revision.md` and `result_final_revision.md`.
- [x] Execution summary written to `paper_outputs/qc/revision_relation_edge/revision_execution_summary.md`.

Current interpretation:

```text
The revision analyses preserve the existing stagewise relation-edge story.
Material moderation and trajectory geometry are boundary/support analyses.
The strongest new support is retrieval-stage semantic-hpc coordination, not YY-specific causal coupling.
```

## Source-First Revision Addendum: 2026-05-17

- [x] Added `r0a_source_inventory.py` to audit original source files and variable provenance.
- [x] Added `r0b_compare_source_recompute.py` to compare source-recomputed neural tables against locked upstream outputs.
- [x] Confirmed run3/run4 behavior exists in raw E-Prime CSV files, not in compact event TSVs:

```text
run3: sentence3_RESP / sentence3_RT
run4: sentence4_RESP / sentence4_RT
```

- [x] Updated `r0_build_revision_master_table.py` so `run3_understand`, `run4_like`, `run3_rt_z`, `run4_rt_z`, and `learning_fluency_shift` are merged from source-derived `learning_behavior_item.tsv`.
- [x] Re-ran r0 and confirmed run3/run4 coverage is approximately 96-99%.
- [x] Re-ran r2 with the learning behavior fields included in high/low post-separation item profiling.
- [x] Recomputed retrieval geometry from `pattern_root/*.nii.gz` and `roi_library` masks into `paper_outputs/source_recompute/qc/retrieval_geometry/`.
- [x] Confirmed source-recomputed `retrieval_pair_metrics.tsv` exactly matches locked upstream `E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_pair_metrics.tsv` for audited rows and numeric columns.
- [ ] If a fully strict source-only rerun is required, repeat the same source-recompute comparison for MVPA and every upstream neural table used by nc_converge. Current revision now has source-verified retrieval geometry but still treats other neural summaries as locked upstream outputs.

New source audit outputs:

```text
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/source_event_schema.tsv
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/source_pattern_schema.tsv
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/source_material_schema.tsv
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/source_roi_schema.tsv
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/source_embedding_schema.tsv
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/revision_variable_provenance.tsv
paper_outputs/qc/revision_relation_edge/r0a_source_inventory/source_inventory_review.md
paper_outputs/qc/revision_relation_edge/r0b_compare_source_recompute/retrieval_geometry_source_compare.tsv
paper_outputs/qc/revision_relation_edge/r0b_compare_source_recompute/retrieval_geometry_source_compare_review.md
```

## Phase 0: Preflight

- [ ] Confirm active local project path:

```text
E:/PythonAnalysis/metaphoric/final_version
```

- [ ] Confirm script/output directories exist:

```text
paper_analysis/revision_relation_edge/
paper_outputs/qc/revision_relation_edge/
paper_outputs/figures_revision/
```

- [ ] Confirm current result documents exist:

```text
result_new_meta_roi.md
result_story.md
result_final.md
```

- [ ] Confirm prior outputs needed by the revision are discoverable under `paper_outputs/`.
- [ ] Confirm all new item-level mixed models use `condition_item_id`.
- [ ] Confirm no new analysis overwrites upstream results.

## Phase 1: P0 master table and material moderation

### Task 1.1: Build revision master table

Script:

```text
paper_analysis/revision_relation_edge/r0_build_revision_master_table.py
```

Purpose:

```text
Merge all item-level and network-level variables once, so later analyses do not repeat joins
or accidentally mix YY_28 and KJ_28.
```

Required inputs:

```text
post edge specificity / post separation
pre pair similarity
retrieval pair similarity
memory score: 0 / 0.5 / 1
memory_strict / memory_lenient
novelty / familiarity / comprehensibility / difficulty
word frequency / word length / valence / arousal
learning behavior: run3 understand, run4 liking, RT if available
network metrics: semantic, hpc-spatial
MVPA summary if available
```

Required fields:

```text
subject
condition
condition_item_id
pair_id_raw
network
pre_pair_similarity_z
post_pair_similarity_z
retrieval_pair_similarity_z
post_separation_z
retrieval_rebound_z
memory_ord
memory_strict
memory_lenient
novelty_z
familiarity_z
comprehensibility_z
difficulty_z
wordfreq_z
charlen_z
run3_understand
run4_like
run3_rt_z
run4_rt_z
```

Outputs:

```text
paper_outputs/qc/revision_relation_edge/revision_master_item.tsv
paper_outputs/qc/revision_relation_edge/revision_master_network.tsv
paper_outputs/qc/revision_relation_edge/revision_covariate_coverage.tsv
paper_outputs/qc/revision_relation_edge/revision_schema_audit.tsv
```

Acceptance:

- [ ] `condition_item_id` is unique by condition and item.
- [ ] `YY_28` and `KJ_28` cannot collapse into one item.
- [ ] Novelty / familiarity / comprehensibility / difficulty coverage is reported.
- [ ] Missingness is reported by condition.
- [ ] Sample size is reported by network x condition.

### Task 1.2: Material distance and novelty moderation

Script:

```text
paper_analysis/revision_relation_edge/r1_material_distance_novelty_moderation.py
```

Questions:

```text
Do semantically closer or farther word pairs differ in memory?
After controlling semantic distance, do more novel YY metaphors show better memory?
Do semantic distance / novelty explain post-stage separation?
```

Model A:

```text
memory_ord ~ condition * pre_pair_similarity_z
           + condition * novelty_z
           + familiarity_z
           + comprehensibility_z
           + difficulty_z
           + wordfreq_z
           + charlen_z
           + (1 | subject)
           + (1 | condition_item_id)
```

Model B:

```text
post_separation_z ~ condition * pre_pair_similarity_z
                  + condition * novelty_z
                  + familiarity_z
                  + comprehensibility_z
                  + difficulty_z
                  + wordfreq_z
                  + charlen_z
                  + (1 | subject)
                  + (1 | condition_item_id)
```

Model C:

```text
YY_memory_ord ~ pre_pair_similarity_z
              + novelty_z
              + familiarity_z
              + comprehensibility_z
              + difficulty_z
              + (1 | subject)
              + (1 | condition_item_id)

YY_post_separation_z ~ pre_pair_similarity_z
                     + novelty_z
                     + familiarity_z
                     + comprehensibility_z
                     + difficulty_z
                     + (1 | subject)
                     + (1 | condition_item_id)
```

Optional quadratic model:

```text
memory_ord ~ condition * pre_pair_similarity_z
           + condition * I(pre_pair_similarity_z^2)
           + covariates
```

Outputs:

```text
material_moderation_memory.tsv
material_moderation_post_separation.tsv
yy_only_material_models.tsv
material_covariate_coverage.tsv
fig_material_moderation_effects.png
fig_material_moderation_effects.pdf
```

Interpretation:

- [ ] If `pre_pair_similarity x YY` is stable, discuss semantic distance as a moderator.
- [ ] If `novelty x YY` is stable, discuss novelty as a moderator of metaphor learning benefit.
- [ ] If neither is stable, write this as boundary evidence showing the core YY effect is not trivially explained by material covariates.

### Task 1.3: High vs low post-separation item profile

Script:

```text
paper_analysis/revision_relation_edge/r2_high_low_post_separation_item_profile.py
```

Question:

```text
Which YY items show stronger post-stage separation?
```

Grouping:

```text
YY high_post_separation = top 30%
YY low_post_separation = bottom 30%
```

Sensitivity:

```text
median split
```

Variables:

```text
pre_pair_similarity
novelty
familiarity
comprehensibility
difficulty
word frequency
word length
valence
arousal
run3 understand rate
run4 liking rate
run3 RT
run4 RT
memory score
```

Outputs:

```text
high_low_post_separation_item_profile.tsv
high_low_post_separation_feature_tests.tsv
fig_high_low_separation_profile.png
fig_high_low_separation_profile.pdf
```

Interpretation:

- [ ] If high-separation items have a clear feature profile, use it as Discussion material.
- [ ] If no feature profile is stable, write that post separation is not determined by a single material dimension.

## Phase 2: P1 trajectory and network mechanisms

### Task 2.1: Pre-post-retrieval trajectory geometry

Script:

```text
paper_analysis/revision_relation_edge/r3_pre_post_retrieval_geometry.py
```

Core question:

```text
Is retrieval a return to pre-learning geometry, or a task-driven new-state reconstruction?
```

Metrics:

```text
d_pre_post
d_post_retrieval
d_pre_retrieval
return_to_pre_index = d_post_retrieval - d_pre_retrieval
```

If voxel patterns are available:

```text
v_pre_post = pattern_post - pattern_pre
v_post_retrieval = pattern_retrieval - pattern_post
trajectory_cosine = cosine(v_pre_post, v_post_retrieval)
```

Main model:

```text
trajectory_metric ~ condition
                  + memory_ord
                  + condition * memory_ord
                  + covariates
                  + (1 | subject)
                  + (1 | condition_item_id)
```

Outputs:

```text
trajectory_triangle_metrics.tsv
trajectory_vector_alignment.tsv
trajectory_geometry_models.tsv
fig_pre_post_retrieval_triangle.png
fig_pre_post_retrieval_triangle.pdf
fig_displacement_vector_alignment.png
fig_displacement_vector_alignment.pdf
trajectory_geometry_review.md
```

Interpretation:

- [ ] If YY retrieval is closer to pre, write partial reinstatement / return-to-pre.
- [ ] If YY retrieval is not close to pre but pair similarity rebounds, write task-driven reconstruction / new-state re-binding.
- [ ] If remembered YY items show stronger new-state reconstruction, upgrade the memory mechanism.

### Task 2.2: Semantic-hpc network coupling revision

Script:

```text
paper_analysis/revision_relation_edge/r4_network_coupling_revision.py
```

Scope:

```text
semantic/metaphor network
hpc-spatial network
```

No broad ROI x ROI scan.

Models:

```text
network_coupling_z ~ condition * stage
                   + (1 | subject)
                   + (1 | condition_item_id)

hpc_retrieval_rebound_z ~ semantic_retrieval_rebound_z * condition
                        + covariates
                        + (1 | subject)
                        + (1 | condition_item_id)

memory_ord ~ condition * semantic_hpc_retrieval_coupling_z
           + condition * semantic_hpc_post_coupling_z
           + covariates
           + (1 | subject)
           + (1 | condition_item_id)
```

Exploratory only:

```text
semantic_retrieval_rebound ~ hpc_head_rebound
semantic_retrieval_rebound ~ hpc_body_rebound
semantic_retrieval_rebound ~ hpc_tail_rebound
```

Outputs:

```text
network_coupling_stage_models.tsv
network_coupling_memory_models.tsv
hpc_long_axis_coupling_exploratory.tsv
fig_network_coupling_revision.png
fig_network_coupling_revision.pdf
network_coupling_revision_review.md
```

Interpretation:

- [ ] If retrieval coupling increases, write retrieval-stage inter-network coordination.
- [ ] If YY-specific interaction is stable, cautiously write YY-specific network coordination.
- [ ] If coupling predicts memory, use it as a stronger behavior bridge.
- [ ] If not stable, keep the previous B3 wording: retrieval-stage coordination, not YY-specific causal coupling.

### Task 2.3: MVPA stage-state summary

Script:

```text
paper_analysis/revision_relation_edge/r5_mvpa_stage_state_summary.py
```

Purpose:

```text
Put MVPA back into the paper as stage-state evidence, not as edge-reorganization mechanism evidence.
```

Expected structure:

```text
pre: weak / not stable
learning: strong YY/KJ decoding
post: partial retention, especially temporal pole
retrieval: robust YY/KJ decoding
cross-role: not stable
behavior bridge: not stable
```

Outputs:

```text
mvpa_stage_state_summary.tsv
mvpa_boundary_table.tsv
fig_mvpa_stage_state_summary.png
fig_mvpa_stage_state_summary.pdf
```

Interpretation:

- [ ] RSA explains pair / edge geometry.
- [ ] MVPA explains when YY/KJ condition information is available.

## Phase 3: P1 writing and figure integration

### Task 3.1: Update evidence tier and story

Script:

```text
paper_analysis/revision_relation_edge/r6_update_evidence_tier_and_story.py
```

Outputs:

```text
revision_evidence_tier_table.tsv
main_text_result_map.tsv
supplementary_result_map.tsv
boundary_result_map.tsv
storyline_revision.md
```

Required story update:

```text
Successful metaphor learning does not simply increase similarity between associated concepts.
Instead, it differentiates trained relation edges from their pre-existing semantic neighborhood,
creating relation-edge representations that can be task-dependently reconstructed during retrieval.
```

### Task 3.2: Make revision figures

Script:

```text
paper_analysis/revision_relation_edge/r7_make_revision_figures.py
```

Main figure plan:

```text
Fig 1. Design + behavior
Fig 2. Learning condition geometry + MVPA stage-state inset
Fig 3. Post trained-edge differentiation + word identity control
Fig 4. Stage trajectory + trajectory geometry
Fig 5. Subsequent-memory: prior post separation
Fig 6. Semantic-hpc network coupling + robustness summary
```

Supplementary figure plan:

```text
S1 full ROI Step5C
S2 edge specificity all ROI
S3 MVPA full results
S4 material moderation
S5 high/low post separation profile
S6 Mahalanobis proxy
S7 input-output / global centrality / novelty boundary
S8 relation-vector / global RDM shift
S9 directional mapping / ERS
S10 HPC long-axis / MDS
```

Outputs:

```text
paper_outputs/figures_revision/*.png
paper_outputs/figures_revision/*.pdf
paper_outputs/figures_revision/figure_revision_manifest.tsv
```

### Task 3.3: Append or replace revised story documents

Script:

```text
paper_analysis/revision_relation_edge/r8_append_revision_story.py
```

Required options:

```text
--dry-run
--replace-section
--skip-if-exists
```

Outputs:

```text
result_story_revision.md
result_final_revision.md
revision_story_preview.md
revision_append_log.txt
```

Default behavior:

- [ ] Do not overwrite `result_story.md` or `result_final.md` unless explicitly requested.
- [ ] Write revision drafts first.
- [ ] Copy final selected figures into `figure_final/` only after figure review.

## Phase 4: Execution order

### First batch: mandatory

```text
r0_build_revision_master_table.py
r1_material_distance_novelty_moderation.py
r2_high_low_post_separation_item_profile.py
r3_pre_post_retrieval_geometry.py
```

### Second batch: mechanism and network support

```text
r4_network_coupling_revision.py
r5_mvpa_stage_state_summary.py
```

### Third batch: writing integration

```text
r6_update_evidence_tier_and_story.py
r7_make_revision_figures.py
r8_append_revision_story.py
```

### Minimum executable version

```text
r0_build_revision_master_table.py
r1_material_distance_novelty_moderation.py
r3_pre_post_retrieval_geometry.py
r6_update_evidence_tier_and_story.py
r7_make_revision_figures.py
```
