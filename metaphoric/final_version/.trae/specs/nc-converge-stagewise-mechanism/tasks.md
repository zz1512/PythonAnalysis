# Tasks: NC Converge Stagewise Mechanism Plan

## Phase 0: Preflight

- [ ] Confirm the active workspace is:

```text
E:/PythonAnalysis/metaphoric/final_version
```

- [ ] Confirm upstream outputs exist:

```text
E:/python_metaphor/paper_outputs/
```

- [ ] Confirm the key item-level table exists:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv
```

- [ ] Confirm `nc_converge` scripts are present:

```text
E:/PythonAnalysis/metaphoric/final_version/nc_converge/
```

- [ ] Confirm no new analysis will overwrite upstream results.

## Phase 1: Revise nc_converge Scripts

### Task 1.1: Update A4 output design

Target file:

```text
nc_converge/a4_post_memory_component.py
```

Required changes:

- [ ] Keep current input discovery behavior.
- [ ] Preserve network composite support.
- [ ] Add `memory_component_model.tsv`.
- [ ] Add `stage_trajectory_model.tsv`.
- [ ] Add `stage_trajectory_descriptives.tsv`.
- [ ] Keep or rename ROI follow-up output as `memory_component_roi_followup.tsv`.
- [ ] Include covariates where present:

```text
sentence_char_len_z
word_frequency_mean_z
stroke_count_mean_z
valence_mean_z
arousal_mean_z
```

- [ ] Prefer strict mixed models through `shared_nc.fit_formula`.
- [ ] Fail visibly if required columns are missing, unless `--allow-empty` is passed.

### Task 1.2: Reframe B1 wording

Target files:

```text
nc_converge/b1_staged_path.py
nc_converge/README.md
nc_converge/run_all_nc_converge.py
nc_converge/a6_fdr_tiering.py
nc_converge/go_no_go.py
nc_converge/append_section28.py
```

Required changes:

- [ ] Replace causal/path language with staged trajectory characterization.
- [ ] Keep the script filename if changing it would cause unnecessary churn, but update the output text.
- [ ] Do not claim mediation.
- [ ] Do not claim learning causes post separation or retrieval rebound.

### Task 1.3: Verify B3 network restriction

Target file:

```text
nc_converge/b3_repr_coupling.py
```

Required checks:

- [ ] Confirm only `semantic` and `hpc_spatial` composites are modeled.
- [ ] Confirm no ROI x ROI scan is added.
- [ ] Add output fields that identify the planned coupling class when possible:

```text
learning_semantic_to_post_hpc_spatial
retrieval_semantic_to_retrieval_hpc_spatial
```

## Phase 2: Run Primary nc_converge Suite

Command:

```powershell
python E:\PythonAnalysis\metaphoric\final_version\nc_converge\run_all_nc_converge.py `
  --base-dir E:\python_metaphor `
  --paper-output-root E:\python_metaphor\paper_outputs `
  --skip-append-section28
```

Execution tasks:

- [ ] Run A2 schema lock.
- [ ] Run A3 shared-post review.
- [ ] Run revised A4 memory component and trajectory models.
- [ ] Run A5 network composite.
- [ ] Run B2 crossnobis robustness.
- [ ] Run B3 representational coupling.
- [ ] Run A6 FDR tiering.
- [ ] Run Go/No-Go.
- [ ] Run isolation check.

Expected output root:

```text
E:/python_metaphor/paper_outputs/qc/nc_converge/
```

## Phase 3: Review Primary Evidence

Priority review files:

```text
qc/nc_converge/a4_post_memory_component/memory_component_model.tsv
qc/nc_converge/a4_post_memory_component/stage_trajectory_model.tsv
qc/nc_converge/b2_crossnobis_robustness/crossnobis_step5c.tsv
qc/nc_converge/b3_repr_coupling/repr_coupling.tsv
qc/nc_converge/a6_fdr_tiering/fdr_tier_manifest.tsv
qc/nc_converge/go_no_go/go_no_go_stage6.md
```

Review questions:

- [ ] Does A4 show YY memory is better explained by prior post-stage separation than pure retrieval similarity?
- [ ] Does A4 stage trajectory show YY post drop and retrieval rebound?
- [ ] Does B2 preserve the main direction under crossnobis robustness?
- [ ] Does B3 show interpretable semantic/metaphor to hpc-spatial coupling?
- [ ] Does A6 preserve the important effects after the intended FDR tiering?

## Phase 4: Add Mechanism Upgrade Scripts

Only start this phase after Phase 3 has usable results.

### Task 4.1: C1 input-output transformation

Suggested file:

```text
nc_converge/c1_input_output_transformation.py
```

Primary input:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv
```

Model:

```text
post_pair_similarity ~ pre_pair_similarity * condition
                     + covariates
                     + (1|subject)
                     + (1|condition_item_id)
```

Expected outputs:

```text
input_output_transformation.tsv
input_output_descriptives.tsv
input_output_manifest.tsv
```

### Task 4.2: C2 global similarity / centrality

Suggested file:

```text
nc_converge/c2_global_similarity_centrality.py
```

Preflight:

- [ ] Identify usable item x item RDM sources.
- [ ] Confirm metadata has subject, ROI, phase, condition, item label, and pair id.
- [ ] Confirm item ordering matches the RDM matrix.

Candidate source pattern:

```text
E:/python_metaphor/paper_outputs/qc/model_rdm_results_*/model_rdm_audit/*/*/*_all_rdms.npz
```

Expected outputs:

```text
global_similarity_item.tsv
global_similarity_models.tsv
global_similarity_manifest.tsv
```

### Task 4.3: C3 novelty / familiarity moderation

Suggested file:

```text
nc_converge/c3_novelty_familiarity_moderation.py
```

Primary inputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv
E:/python_metaphor/paper_outputs/qc/behavior_results/refined/*.tsv
E:/python_metaphor/paper_outputs/tables_si/table_stimulus_control_s0.tsv
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

Expected outputs:

```text
novelty_familiarity_moderation.tsv
novelty_familiarity_input_coverage.tsv
novelty_familiarity_manifest.tsv
```

## Phase 5: Optional Learning-Update Proxy

Suggested file:

```text
nc_converge/c4_learning_update_proxy.py
```

Run only if:

- [ ] A4/B2/B3 are stable enough to justify a small mechanism add-on.
- [ ] run3/run4 behavioral RT and judgment data can be joined cleanly by subject and condition item.

Model:

```text
post_edge_differentiation ~ learning_update * condition
                          + covariates
                          + (1|subject)
                          + (1|condition_item_id)
```

## Phase 6: Decide on hpc_subfield_three_axis

Run only if all entry conditions pass:

- [ ] `nc_converge` supports R hippocampus / hpc-spatial direction.
- [ ] B2 crossnobis supports the Step5C direction.
- [ ] Mask voxel counts pass QC.
- [ ] Signal quality is acceptable.
- [ ] The write-up will label this as exploratory long-axis segmentation.

Command template:

```powershell
python E:\PythonAnalysis\metaphoric\final_version\hpc_subfield_three_axis\run_all_hpc_subfield.py `
  --base-dir E:\python_metaphor `
  --paper-output-root E:\python_metaphor\paper_outputs `
  --skip-append-section29
```

## Phase 7: Writing Consolidation

Start this phase after the primary Go/No-Go decision.

Writing tasks:

- [ ] Rewrite Results around the locked main story.
- [ ] Put A4/B2/B3 in the main robustness narrative.
- [ ] Put C1/C2/C3 in a mechanism refinement section if results are usable.
- [ ] Move weak or exploratory results to supplement.
- [ ] Describe KJ as an active spatial-relational learning comparison condition.
- [ ] Avoid the full causal-chain claim.
- [ ] Avoid definitive hippocampal subfield claims.

