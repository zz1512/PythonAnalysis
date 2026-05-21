# Learning Reactivation / Mapping Mechanism Spec

## 0. Non-Destructive Rule

This is a parallel extension analysis. It must not overwrite existing scripts, existing QC tables, or existing result documents.

Read-only inputs:

```text
E:/python_metaphor/pattern_root/
E:/python_metaphor/data_events/
E:/python_metaphor/stimuli_template.csv
E:/python_metaphor/materials_detail/实验材料整理.xlsx
E:/python_metaphor/roi_library/manifest.tsv
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv
E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_pair_metrics.tsv
```

New outputs only:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/
E:/python_metaphor/paper_outputs/tables_exploratory/learning_reactivation_mapping/
E:/python_metaphor/paper_outputs/figures_exploratory/learning_reactivation_mapping/
```

New scripts only:

```text
metaphoric/final_version/brain_behavior/learning_reactivation/
```

Do not modify or overwrite:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/
E:/python_metaphor/paper_outputs/qc/edge_specificity/
E:/python_metaphor/paper_outputs/qc/retrieval_geometry/
metaphoric/final_version/result_new_meta_roi.md
metaphoric/final_version/new_rerun_list.md
```

Result integration into `result_new_meta_roi.md` happens only after the new analysis passes QC and after manual review.

## 1. Goal

The new mechanism story is:

```text
learning-stage semantic reactivation / behavioral update
        -> post-stage edge differentiation / directional mapping
        -> retrieval-stage re-binding / memory advantage
```

This replaces the overly generic predictor used in the previous spec:

```text
learning_specificity = run3/run4 sentence self-specificity
```

with theoretically sharper learning-stage predictors:

```text
LearningSentence_AB -> PreWord_A / PreWord_B reactivation
Run3 comprehension difficulty -> Run4 second-exposure fluency
Target/topic toward source/vehicle directional updating
Semantic-network reactivation -> hippocampal-spatial separation
```

## 2. Design Principles

### 2.1 ID Policy

Keep three identifiers separate:

```text
original_pair_id = material/picture ID, 1-40 with condition-specific missing IDs
template_pair_id = compact 1-35 pair ID from stimuli_template.csv
condition_item_id = condition + "_" + original_pair_id
```

Valid original IDs:

```text
YY valid original_pair_id: 1-40 except 20,25,30,31,34
KJ valid original_pair_id: 1-40 except 4,18,32,35,40
```

Never treat `yy_28` and `kj_28` as paired items.

### 2.2 Role Policy

For YY:

```text
yyw  = target/topic
yyew = source/vehicle
```

For KJ:

```text
kjw  = object/figure
kjew = location/context
```

Use role-neutral names in shared tables:

```text
role_a = target_or_object
role_b = source_or_location
```

Use metaphor-specific language only for YY interpretation.

### 2.3 Run3 / Run4 Behavior Policy

Run3 and Run4 are different tasks:

```text
Run3: comprehension judgment, 5 s sentence display
Run4: liking judgment, 3 s sentence display
```

Therefore:

- Do not compare raw RT directly as a pure learning-speed measure.
- Use within-subject/run/condition z-scored RT for item-level relative difficulty/fluency.
- Interpret `run3_rt_z` as first-exposure comprehension / semantic access load.
- Interpret `run4_rt_z` as second-exposure preference-decision fluency.
- Interpret `run3_rt_z - run4_rt_z` as a learning transition proxy, not as direct processing-speed improvement.

## 3. Primary Analyses

### 3.1 Analysis A: Learning-Stage Reactivation

Question:

```text
Does the learning sentence pattern reactivate its pre-stage component words?
```

For each subject x ROI x condition x item x learning run:

```text
react_role_a =
  Sim(LearningSentence_AB, PreWord_A)
  - mean Sim(LearningSentence_AB, OtherPreWords_same_condition_same_role)

react_role_b =
  Sim(LearningSentence_AB, PreWord_B)
  - mean Sim(LearningSentence_AB, OtherPreWords_same_condition_same_role)

react_pair_mean = mean(react_role_a, react_role_b)
react_asymmetry = react_role_b - react_role_a
```

Run-level outputs:

```text
run3_react_role_a
run3_react_role_b
run3_react_pair_mean
run4_react_role_a
run4_react_role_b
run4_react_pair_mean
react_pair_mean_avg
react_pair_mean_delta = run4_react_pair_mean - run3_react_pair_mean
```

Primary tests:

```text
react_pair_mean_avg ~ condition
react_pair_mean_delta ~ condition
```

Interpretation:

- YY > KJ supports stronger metaphor-related semantic reactivation.
- No YY > KJ does not invalidate the post edge result; it means reactivation is not condition-specific.

### 3.2 Analysis B: Learning Behavior Update

Use raw CSV files:

```text
E:/python_metaphor/data_events/sub-XX/sub-XX_run-3_events.csv
E:/python_metaphor/data_events/sub-XX/sub-XX_run-4_events.csv
```

Required raw columns:

```text
run3: juzi, sentence3_RT, sentence3_RESP, sentence3_ACC
run4: juzi_run, sentence4_RT, sentence4_RESP, sentence4_ACC
```

Derived item-level behavior metrics:

```text
run3_resp
run3_understand_yes = sentence3_RESP == 3
run3_rt
run3_rt_z_subject_condition
run4_resp
run4_like_yes = sentence4_RESP == 3
run4_rt
run4_rt_z_subject_condition
learning_fluency_shift = run3_rt_z_subject_condition - run4_rt_z_subject_condition
learning_response_profile = understand_yes / like_yes combination
```

Primary tests:

```text
run3_rt_z_subject_condition ~ condition
run4_rt_z_subject_condition ~ condition
learning_fluency_shift ~ condition
```

Interpretation:

- Higher run3 RT means higher initial comprehension / semantic access load.
- Higher learning_fluency_shift means stronger relative transition from initial access load to second-exposure fluency.
- Because tasks differ, this remains a proxy, not a pure learning-rate measure.

### 3.3 Analysis C: Reactivation / Behavior -> Post Edge Differentiation

Primary outcome from existing item table:

```text
post_edge_specificity
trained_edge_drop
pre_pair_similarity
```

Primary models:

```text
post_edge_specificity ~ react_pair_mean_avg * condition
                     + pre_pair_similarity
                     + material_covariates
                     + subject cluster / random effect

post_edge_specificity ~ learning_fluency_shift * condition
                     + pre_pair_similarity
                     + material_covariates
                     + subject cluster / random effect

post_edge_specificity ~ react_pair_mean_avg * learning_fluency_shift * condition
                     + pre_pair_similarity
                     + material_covariates
                     + subject cluster / random effect
```

Primary interpretation:

- A positive YY-specific reactivation effect supports learning-stage semantic reactivation as a precursor of edge differentiation.
- A positive YY-specific fluency-shift effect supports behavioral learning transition as a precursor of edge differentiation.
- If neither survives FDR, keep this analysis exploratory.

### 3.4 Analysis D: Directional Mapping / Assimilation

For each item:

```text
target_toward_source =
  Sim(PostRoleA, PreRoleB) - Sim(PreRoleA, PreRoleB)

source_toward_target =
  Sim(PostRoleB, PreRoleA) - Sim(PreRoleB, PreRoleA)

mapping_asymmetry =
  target_toward_source - source_toward_target
```

For YY, this means:

```text
target/topic toward source/vehicle
```

For KJ, this means:

```text
object toward location/context
```

Primary tests:

```text
mapping_asymmetry ~ condition
target_toward_source ~ condition
```

Bridge models:

```text
mapping_asymmetry ~ react_pair_mean_avg * condition
post_edge_specificity ~ mapping_asymmetry * condition + pre_pair_similarity
memory ~ mapping_asymmetry * condition
```

Interpretation:

- YY-specific positive `target_toward_source` / `mapping_asymmetry` supports metaphor directional mapping.
- If source and target both move symmetrically, write as assimilation/general updating rather than source-to-target mapping.

### 3.5 Analysis E: Network-Level Bridge

Define two predeclared networks.

Semantic network:

```text
meta_L_temporal_pole
meta_R_temporal_pole
meta_L_IFG
meta_R_IFG
meta_L_AG
meta_R_AG
meta_L_pMTG_pSTS
meta_R_pMTG_pSTS
```

Hippocampal-spatial network:

```text
meta_L_hippocampus
meta_R_hippocampus
meta_L_PPA_PHG
meta_R_PPA_PHG
meta_L_RSC_PCC
meta_R_RSC_PCC
meta_L_PPC_SPL
meta_R_PPC_SPL
meta_L_precuneus
meta_R_precuneus
```

Network indices:

```text
semantic_reactivation = mean reactivation across available semantic ROIs
hpc_spatial_separation = mean post_edge_specificity across available hpc-spatial ROIs
retrieval_rebinding = mean retrieval_minus_post_pair_similarity across available hpc-spatial ROIs
```

Primary bridge:

```text
hpc_spatial_separation ~ semantic_reactivation * condition + pre_pair_similarity
retrieval_rebinding ~ hpc_spatial_separation * condition
memory ~ retrieval_rebinding * condition
```

Subject-level behavior advantage:

```text
YY-KJ memory advantage ~ YY-KJ semantic_reactivation advantage
YY-KJ memory advantage ~ YY-KJ hpc_spatial_separation advantage
YY-KJ memory advantage ~ YY-KJ retrieval_rebinding advantage
```

Interpretation:

This is the most suitable level for brain-behavior bridge. It avoids over-interpreting noisy item-level memory models in single ROIs.

## 4. Secondary / Exploratory Analyses

### 4.1 Metaphor Tuning

Use:

```text
react_pair_mean_delta = run4 - run3
learning_fluency_shift = run3_rt_z - run4_rt_z
```

Models:

```text
react_pair_mean_delta ~ condition
post_edge_specificity ~ react_pair_mean_delta * condition + pre_pair_similarity
memory ~ react_pair_mean_delta * condition
```

Interpretation must acknowledge run3/run4 task differences.

### 4.2 Separation / Assimilation Dissociation

Network contrast:

```text
hpc_spatial_separation = post_edge_specificity in hpc-spatial network
semantic_assimilation = target_toward_source / mapping_asymmetry in semantic network
```

Model:

```text
network_index ~ network_type * condition
```

Interpretation:

- Spatial/hippocampal separation plus semantic assimilation would strongly match memory-integration literature.
- If both networks show separation, write as broader differentiation.

### 4.3 Schema Congruence

Not first-pass unless materials ratings or embeddings are ready.

Potential predictors:

```text
source-target embedding distance
metaphor familiarity
aptness
novelty
comprehensibility
shared-ground embedding similarity
```

Do not run as main analysis without a separate material-predictor QC.

## 5. Scripts

### Task 1: Extract Learning Behavior From Raw CSV

Create:

```text
metaphoric/final_version/brain_behavior/learning_reactivation/build_learning_behavior_table.py
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_behavior_item.tsv
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_behavior_qc.tsv
```

Stop if:

- Any subject has fewer than 60 usable run3 or run4 learning trials.
- `condition + original_pair_id` cannot be recovered from `juzi` / `juzi_run`.
- More than 15% RT missing in either run.

### Task 2: Compute Learning Reactivation

Create:

```text
metaphoric/final_version/brain_behavior/learning_reactivation/learning_stage_reactivation.py
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_reactivation_item.tsv
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_reactivation_qc.tsv
```

Computational note:

This script reads NIfTI files and will be slower. It must write only to the new output directory.

### Task 3: Directional Mapping / Assimilation

Create:

```text
metaphoric/final_version/brain_behavior/learning_reactivation/directional_mapping_assimilation.py
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/directional_mapping_item.tsv
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/directional_mapping_models.tsv
```

### Task 4: Reactivation-to-Post Models

Create:

```text
metaphoric/final_version/brain_behavior/learning_reactivation/reactivation_to_post_edge.py
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/reactivation_to_post_models.tsv
E:/python_metaphor/paper_outputs/tables_exploratory/learning_reactivation_mapping/table_reactivation_to_post.tsv
```

### Task 5: Network Bridge Models

Create:

```text
metaphoric/final_version/brain_behavior/learning_reactivation/network_bridge_models.py
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/network_bridge_item.tsv
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/network_bridge_models.tsv
E:/python_metaphor/paper_outputs/tables_exploratory/learning_reactivation_mapping/table_network_bridge.tsv
```

### Task 6: Review Summary

Create only after Tasks 1-5 pass QC:

```text
E:/python_metaphor/paper_outputs/qc/learning_reactivation_mapping/learning_reactivation_mapping_review.md
```

Do not edit `result_new_meta_roi.md` automatically.

## 6. Statistical Policy

Primary model level:

```text
ROI-wise GEE / mixed model with subject clustering
Network-level model for brain-behavior bridge
FDR within meta_metaphor / meta_spatial for ROI-wise tests
Network-level tests reported separately as predeclared index tests
```

Primary terms:

```text
react_pair_mean_avg
react_pair_mean_avg × condition
learning_fluency_shift
learning_fluency_shift × condition
mapping_asymmetry
mapping_asymmetry × condition
semantic_reactivation × condition
hpc_spatial_separation × condition
```

Controls:

```text
pre_pair_similarity
material covariates when available
run3/run4 RT missingness
subject-level mean RT
```

Negative controls:

```text
reactivation to wrong-role other item
shuffled condition_item_id reactivation
pre_pair_similarity as outcome
pseudo_edge_drop as outcome
```

## 7. Checklist

### 7.1 File Safety

- [ ] No existing output directory is overwritten.
- [ ] All new outputs are under `learning_reactivation_mapping`.
- [ ] Existing `learning_post_memory_prediction` tables are read-only.
- [ ] `result_new_meta_roi.md` is not edited until manual review.

### 7.2 Data Integrity

- [ ] `original_pair_id` and `template_pair_id` are both retained.
- [ ] YY and KJ valid item IDs match condition-specific missing-ID lists.
- [ ] Run3 raw CSV uses `sentence3_RT/RESP/ACC`.
- [ ] Run4 raw CSV uses `sentence4_RT/RESP/ACC`.
- [ ] RT is z-scored within subject x run x condition.
- [ ] No raw RT comparison is interpreted as pure learning speed.

### 7.3 Reactivation QC

- [ ] Learning volumes match metadata rows.
- [ ] Pre word volumes include both role A and role B for each pair.
- [ ] Reactivation baseline uses same-condition same-role other words.
- [ ] Fisher-z similarities are used before averaging.
- [ ] Missing cells are reported by subject, ROI, condition, run, and role.

### 7.4 Interpretation QC

- [ ] Reactivation is not equated with conscious retrieval.
- [ ] Run3/run4 behavior is interpreted according to task demands.
- [ ] Directional mapping language is used only when asymmetry supports it.
- [ ] Network bridge is described as predictive / mechanistic evidence, not causal mediation.
- [ ] Exploratory results are labeled exploratory unless they survive predeclared controls.

## 8. Decision Rules

Upgrade to main-story candidate only if at least one of the following survives the planned controls:

```text
YY-specific learning reactivation predicts post_edge_specificity.
YY-specific mapping_asymmetry predicts post_edge_specificity.
Semantic-network reactivation predicts hpc-spatial post separation.
Network-level neural advantage predicts YY-KJ memory advantage.
```

If none survive:

```text
Keep the current main story:
learning-stage condition geometry -> post edge differentiation -> retrieval re-binding.

Use this extension as a rigorous negative/boundary analysis.
```
