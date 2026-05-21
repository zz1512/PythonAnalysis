# Mixed-Effects Model Extension Spec

## 0. Non-Destructive Rule

This is a parallel statistical re-estimation layer. It must not overwrite existing RSA, MVPA, trajectory, or learning-reactivation outputs.

Read-only inputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/
E:/python_metaphor/paper_outputs/qc/retrieval_geometry/
E:/python_metaphor/paper_outputs/qc/edge_specificity/
E:/python_metaphor/materials_detail/实验材料整理.xlsx
E:/python_metaphor/data_events/
```

New outputs only:

```text
E:/python_metaphor/paper_outputs/qc/mixed_effects/
E:/python_metaphor/paper_outputs/tables_exploratory/mixed_effects/
E:/python_metaphor/paper_outputs/figures_exploratory/mixed_effects/
```

New scripts only:

```text
metaphoric/final_version/brain_behavior/mixed_effects/
```

Do not modify or overwrite:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/
E:/python_metaphor/paper_outputs/qc/retrieval_geometry/
E:/python_metaphor/paper_outputs/qc/edge_specificity/
```

Result integration into `result_new_meta_roi.md` happens only after all mixed-model QC and convergence checks pass.

## 1. Goal

The mixed-effects extension is not a new fishing layer. Its purpose is to re-estimate the current key effects while preserving subject-level and item-level repeated-measure structure:

```text
learning-stage predictor / condition
        -> post-stage learned-edge differentiation
        -> retrieval-stage re-binding / memory
```

The main reasons to use mixed-effects models here are:

```text
avoid averaging away item-level information
handle run7 missing data without forcing complete-case means
avoid false YY/KJ pair matching
include item random effects for condition-specific item pools
control material-level covariates without collapsing observations
```

Mixed models should support the existing story only if they reproduce the same direction as the current analyses. They should not replace permutation / cross-validation logic for MVPA.

## 2. Design Principles

### 2.1 Unit of Analysis

Use item-level long tables wherever possible:

```text
subject_id x condition x condition_item_id x roi
```

For phase trajectory models:

```text
subject_id x condition x condition_item_id x roi x phase
```

For memory models:

```text
subject_id x condition x condition_item_id x roi
```

### 2.2 Item Identity Policy

YY and KJ are not paired by numeric `pair_id`.

Correct item identity:

```text
condition_item_id = condition + "_" + original_pair_id
```

Examples:

```text
yy_28 != kj_28
```

Valid item sets:

```text
YY valid original_pair_id: 1-40 except 20,25,30,31,34
KJ valid original_pair_id: 1-40 except 4,18,32,35,40
```

Do not:

```text
take numeric pair_id intersection between YY and KJ
construct yy_28 - kj_28 paired contrasts
use pair_id alone as item random effect
```

Do:

```text
use condition_item_id as item random effect / variance component
compare YY and KJ as two item pools in a hierarchical model
```

### 2.3 ROI Policy

Run all primary mixed models ROI-wise across the 18 external meta ROIs:

```text
meta_metaphor: 8 ROIs
meta_spatial: 10 ROIs
```

FDR correction:

```text
within meta_metaphor family
within meta_spatial family
```

Focus ROIs may be displayed but cannot be the only tested family.

### 2.4 Random-Effect Policy

Minimum random structure:

```text
(1 | subject_id) + (1 | condition_item_id)
```

Optional random slopes only if the model converges reliably:

```text
(1 + condition | subject_id)
(1 + phase | subject_id)
```

Do not use maximal random structures if they cause singular fits or unstable estimates. Record convergence status, singularity, and fallback formula.

### 2.5 Covariate Policy

Use covariates only when available and conceptually relevant:

```text
pre_pair_similarity_z
sentence_char_len_z
word_frequency_mean_z
stroke_count_mean_z
valence_mean_z
arousal_mean_z
run3_rt_z / run4_rt_z / learning_fluency_shift_z
```

Critical distinction:

```text
pre_pair_similarity_z = neural pre-stage pair similarity from brain pattern correlations
```

It is not BERT similarity, word-label similarity, or text embedding similarity.

Primary neural models should include `pre_pair_similarity_z` when the outcome is post-stage edge specificity or later-stage similarity, because this controls pre-existing item structure.

## 3. Primary Mixed-Effects Analyses

### 3.1 M1: Post Edge Differentiation Main Effect

Question:

```text
Does YY show stronger post-stage learned-edge differentiation than KJ after preserving subject/item structure?
```

Primary outcome:

```text
post_edge_specificity
```

Model:

```text
post_edge_specificity ~ condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject_id)
                      + (1 | condition_item_id)
```

Run ROI-wise.

Confirmatory expectation:

```text
YY condition beta > 0 in hippocampal-spatial ROIs / network
```

Sensitivity outcomes:

```text
trained_edge_drop
post_pair_similarity
```

`post_pair_similarity` is descriptive only. It should not become the primary outcome.

### 3.2 M2: Four-Phase Similarity Trajectory

Question:

```text
Does the trajectory follow pre -> learning -> post -> retrieval reorganization?
```

Phases:

```text
pre
learning
post
retrieval
```

Model:

```text
pair_similarity ~ phase * condition
                + material_covariates
                + (1 | subject_id)
                + (1 | condition_item_id)
```

Planned contrasts:

```text
post - pre
retrieval - post
retrieval - pre
learning - pre
```

Run7 missing data:

```text
Do not force complete-case subjects across all phases.
Mixed model may include available rows, but report n_subjects and n_items per phase.
For contrasts involving retrieval, use only rows with available retrieval data.
```

Interpretation rule:

```text
If post-pre drop and retrieval-post rebound are both present, this supports differentiation followed by task-driven re-binding.
If retrieval-pre remains below pre in hippocampus, write as partial rebound rather than full restoration.
```

### 3.3 M3: Learning Reactivation / Fluency -> Post Edge

Question:

```text
Do learning-stage reactivation or behavioral update predict post-stage edge differentiation?
```

Models:

```text
post_edge_specificity ~ react_pair_mean_avg_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject_id)
                      + (1 | condition_item_id)

post_edge_specificity ~ learning_fluency_shift_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject_id)
                      + (1 | condition_item_id)
```

Primary terms:

```text
react_pair_mean_avg_z
react_pair_mean_avg_z:conditionYY
learning_fluency_shift_z
learning_fluency_shift_z:conditionYY
```

Negative-control check:

```text
pre_pair_similarity ~ react_pair_mean_avg_z * condition
                    + material_covariates
                    + (1 | subject_id)
                    + (1 | condition_item_id)
```

If reactivation strongly predicts `pre_pair_similarity`, interpretation must remain exploratory because the predictor is contaminated by pre-existing structure.

### 3.4 M4: Corrected Directional Mapping -> Post Edge

Question:

```text
Does self-stability corrected directional mapping explain post edge differentiation?
```

Use corrected metrics only:

```text
target_source_over_self
mapping_asymmetry_self_corrected
```

Models:

```text
post_edge_specificity ~ target_source_over_self_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject_id)
                      + (1 | condition_item_id)

post_edge_specificity ~ mapping_asymmetry_self_corrected_z * condition
                      + pre_pair_similarity_z
                      + material_covariates
                      + (1 | subject_id)
                      + (1 | condition_item_id)
```

Primary terms:

```text
target_source_over_self_z:conditionYY
mapping_asymmetry_self_corrected_z:conditionYY
```

Interpretation rule:

```text
Positive effects in PPA/PHG can be described as spatial/scenario mapping-edge coupling.
Mixed signs across ROIs cannot be written as a unified source-to-target assimilation mechanism.
```

### 3.5 M5: Network-Level Bridge

Question:

```text
Do network-level post separation and retrieval re-binding remain linked under mixed models?
```

Network item table:

```text
semantic_reactivation
semantic_mapping_asymmetry_self_corrected
semantic_target_source_over_self
hpc_spatial_separation
retrieval_rebinding
```

Models:

```text
hpc_spatial_separation ~ condition
                       + semantic_reactivation_z
                       + semantic_reactivation_z:condition
                       + pre_pair_similarity_z
                       + (1 | subject_id)
                       + (1 | condition_item_id)

hpc_spatial_separation ~ condition
                       + semantic_target_source_over_self_z
                       + semantic_target_source_over_self_z:condition
                       + pre_pair_similarity_z
                       + (1 | subject_id)
                       + (1 | condition_item_id)

retrieval_rebinding ~ hpc_spatial_separation_z * condition
                    + pre_pair_similarity_z
                    + (1 | subject_id)
                    + (1 | condition_item_id)
```

Confirmatory expectation:

```text
hpc_spatial_separation predicts retrieval_rebinding
```

This is the strongest currently defensible bridge. Semantic reactivation / mapping predictors should remain exploratory unless they survive this model.

### 3.6 M6: Memory Logistic Mixed Model

Question:

```text
Do neural indices predict run7 memory at item level?
```

Only run if item-level memory success can be mapped reliably to `condition_item_id`.

Outcome:

```text
memory_success = 0 / 1
```

Models:

```text
memory_success ~ post_edge_specificity_z * condition
               + material_covariates
               + (1 | subject_id)
               + (1 | condition_item_id)

memory_success ~ retrieval_rebinding_z * condition
               + material_covariates
               + (1 | subject_id)
               + (1 | condition_item_id)

memory_success ~ target_source_over_self_z * condition
               + material_covariates
               + (1 | subject_id)
               + (1 | condition_item_id)
```

Use logistic mixed model:

```text
family = binomial
link = logit
```

Interpretation rule:

```text
If memory models fail or do not converge, do not force a behavioral bridge.
If only condition predicts memory, write that neural indices did not explain item-level memory beyond YY/KJ advantage.
```

## 4. Tooling Plan

Preferred implementation:

```text
Python for table assembly, QC, FDR, and result summaries
R lme4 / lmerTest / glmer for mixed models, if R is available
Python statsmodels MixedLM as fallback for Gaussian models
```

Logistic mixed models:

```text
prefer R glmer
fallback: Python BinomialBayesMixedGLM only if glmer is unavailable, label as exploratory
```

Each model script should write:

```text
model_results.tsv
model_convergence.tsv
model_manifest.json
model_qc.tsv
```

Required model fields:

```text
analysis_id
roi_set
roi
model_name
outcome
term
estimate
se
df_or_z
p
q
n_rows
n_subjects
n_items
formula
converged
singular
fallback_used
```

## 5. QC Requirements

Before modeling:

```text
confirm condition_item_id is unique within condition and never paired across YY/KJ
report n_subjects, n_items, n_rows by ROI x condition
report missingness by phase for trajectory models
report run7 available rows separately
check that pre_pair_similarity_z is neural, not textual
check covariate missingness before model fitting
```

After modeling:

```text
convergence rate >= 95% for each model family, or document failures
no primary conclusion from failed / singular models
FDR applied within ROI family and model family
effect direction checked against descriptive means
fixed-effect estimates exported with confidence intervals when possible
```

Sanity checks:

```text
M1 YY beta should broadly match existing post-edge direction
M2 retrieval-post contrast should broadly match existing retrieval rebound results
M3 reactivation negative control should reproduce pre-similarity concern
M4 corrected mapping should reproduce local L PPA/PHG and R PPC/SPL pattern if robust
M5 hpc_spatial_separation -> retrieval_rebinding should remain positive
M6 memory model should not be promoted unless neural terms survive
```

## 6. Reporting Rules

Use mixed-effects results as:

```text
robustness / hierarchy-aware re-estimation of existing evidence
```

Do not write:

```text
mixed model proves causality
YY and KJ matched item pairs
directional mapping is established if only mixed signs appear
memory bridge is established from condition-only effects
```

Preferred cautious language:

```text
Mixed-effects models confirmed that the YY post-stage edge effect is not driven only by subject/item averaging.
The strongest bridge remains hippocampal-spatial post separation predicting retrieval-stage re-binding.
Learning-stage reactivation and corrected mapping provide exploratory upstream candidates, but they do not yet fully explain the post separation.
```

## 7. Task Checklist

### Phase A: Preparation

- [ ] Create `brain_behavior/mixed_effects/` without touching existing scripts.
- [ ] Build a unified mixed-model item table from existing QC outputs.
- [ ] Validate `subject_id`, `condition`, `condition_item_id`, `roi`, and outcome columns.
- [ ] Confirm `pre_pair_similarity_z` is neural pattern similarity.
- [ ] Write table QC summary.

### Phase B: Model Execution

- [ ] Run M1 post edge differentiation model across all 18 meta ROIs.
- [ ] Run M2 four-phase trajectory model with retrieval missingness handled explicitly.
- [ ] Run M3 learning reactivation / fluency -> post edge models.
- [ ] Run M4 corrected directional mapping -> post edge models.
- [ ] Run M5 network-level bridge models.
- [ ] Run M6 logistic memory models only if item-level memory mapping is verified.

### Phase C: Checks

- [ ] Check convergence / singularity for every model.
- [ ] Compare fixed-effect direction with descriptive means.
- [ ] Apply FDR within ROI families.
- [ ] Flag exploratory-only results.
- [ ] Stop before result integration if primary models fail or IDs are ambiguous.

### Phase D: Reporting

- [ ] Write `mixed_effects_review.md`.
- [ ] Export compact result tables for M1-M6.
- [ ] Update `result_new_meta_roi.md` only after review.
- [ ] Add execution commands to `new_rerun_list.md` only after scripts are stable.

## 8. Stop Conditions

Stop and do not integrate results if:

```text
condition_item_id cannot be validated
YY/KJ are accidentally paired by numeric pair_id
pre_pair_similarity_z source cannot be confirmed as neural
more than 20% of primary ROI models fail convergence
memory item mapping is ambiguous
directional mapping signs are mixed but being interpreted as unified assimilation
```

