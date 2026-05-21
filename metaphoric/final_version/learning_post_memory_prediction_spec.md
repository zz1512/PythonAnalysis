# Learning-Post-Memory Predictive Mechanism Spec

## 1. Spec

### 1.1 Goal

当前论文主线不再依赖 ERS / reinstatement 类分析。新的核心问题是：

```text
learning-stage item trace
        -> post-stage learned-edge differentiation
        -> run7 behavioral memory
```

目标是检验学习阶段形成的 item-level neural trace 是否能解释后测阶段的 relation-edge reorganization，并进一步检验这种 post-stage edge specificity 是否预测最终记忆。

### 1.2 Design Principle

YY 和 KJ 是两个不同 item pool，不能把 `yy_28` 和 `kj_28` 当成 item-paired comparison。

正确的 item identity 是条件内 item 标识：

```text
condition_item_id = condition + "_" + pair_id
```

例如：

```text
yy_28 = 行为是性格的投影: 行为 / 投影
kj_28 = 简历在黄色的档案袋里: 简历 / 档案袋
```

`condition_item_id` 只用于 condition 内部匹配和 item random effect / cluster unit，不代表 YY 与 KJ 之间存在一一配对关系。

重要边界：

```text
YY valid pair_id: 1-40 except 20,25,30,31,34
KJ valid pair_id: 1-40 except 4,18,32,35,40
```

因此不能取 YY/KJ 的数字交集来做主分析。主表必须保留 YY 的 35 个 item 与 KJ 的 35 个 item。所有条件比较都是两个 item pool 的层级比较，不是同一数字编号下的配对比较。

### 1.3 Primary Hypotheses

H1: Learning-to-post transformation

```text
learning item trace predicts post-pre learned-edge differentiation
```

更具体地说，学习阶段同一 item 在 run3/run4 之间更稳定或更特异的 neural pattern，应该预测从 pre 到 post 的 trained-edge differentiation / specificity。这里的 outcome 必须是 `post-pre` 或基于 `pre` 的 specificity，不是单独的 post similarity。

H2: Post-to-memory prediction

```text
post learned-edge differentiation predicts run7 memory
```

如果 post-stage edge specificity 是核心机制，那么它应该比泛化的 ERS 指标更贴近后续记忆。

H3: Incremental predictive model comparison

```text
condition-only model
  < learning-trace model
  < post-edge-specificity model
  < retrieval-geometry model
```

比较 learning、post、retrieval 三类 neural predictor 对 memory 的增量解释力。这里不能写成 mediation 或 causal chain；最多写成 staged incremental prediction。只有在 learning -> post 与 post -> memory 两段都通过预设控制后，才可以谨慎使用 "bridge" 语言。

### 1.4 Primary ROIs

为避免循环验证，统计主 family 使用完整外部定义的 meta ROI，而不是只选已有显著 ROI：

```text
meta_metaphor: 8 ROIs
meta_spatial: 10 ROIs
```

以下 ROI 只作为 focus ROI，用于图示排序、结果解读和 power-aware 优先检查；不能作为唯一检验 family，也不能只报告这些 ROI：

```text
meta_R_hippocampus
meta_L_hippocampus
meta_R_PPA_PHG
meta_R_PPC_SPL
meta_R_temporal_pole
meta_L_temporal_pole
meta_R_IFG
meta_L_IFG
meta_R_AG
meta_L_pMTG_pSTS
```

其他 meta ROI 可作为 secondary / sensitivity，不进入第一轮主 family。

正式报告规则：

```text
Confirmatory table: all 18 meta ROIs, FDR within meta_metaphor / meta_spatial
Focus display: predeclared focus ROIs, clearly labeled as focus visualization
Sensitivity: literature / main-functional ROI, if needed
```

### 1.5 Core Metrics

Learning predictors:

```text
learning_self_similarity = corr(run3_sentence_pattern, run4_sentence_pattern)
learning_specificity = same_item_run3_run4 - mean(other_item_run3_run4 within same condition)
learning_decodability_optional = item-level classifier evidence, if available
```

Metric construction rules:

```text
compute similarity within subject x ROI x condition
use Fisher-z transformed correlations before averaging/modeling
use same-condition mismatches only for the null term
z-score predictors within ROI, and preferably within subject x condition for sensitivity
report split-half / run3-run4 reliability or finite-cell coverage for each ROI
```

`learning_specificity` 不能混入 YY/KJ 跨条件 mismatches，否则会把 condition-level category separability 写进 item-specific trace。

Post mechanism outcomes:

```text
trained_edge_drop = pre_pair_similarity - post_pair_similarity
pseudo_edge_drop = pre_pseudo_similarity - post_pseudo_similarity
post_edge_specificity = trained_edge_drop - same_condition_pseudo_edge_drop
```

Do not use raw `post_pair_similarity` as the primary outcome. The primary transformation metric must explicitly remove baseline/pre structure:

```text
primary_outcome = post_edge_specificity
secondary_outcome = trained_edge_drop
control_outcome = pre_pair_similarity
```

This guards against the ERS problem where a later-stage metric may look strong only because it shares stable pre-learning item structure.

Memory outcomes:

```text
memory = run7 item memory accuracy
log_rt_correct = log RT for correct trials only
```

Optional retrieval predictors:

```text
retrieval_pair_similarity
retrieval_minus_post_pair_similarity
retrieval_mvpa_evidence
```

Primary / secondary endpoint hierarchy:

```text
Primary neural predictor: learning_specificity
Primary post outcome: post_edge_specificity
Primary behavior outcome: run7 memory accuracy
Secondary: trained_edge_drop, learning_self_similarity, RT, retrieval predictors
Exploratory: MVPA evidence, alternative ROI families, material-specific subgroups
```

Do not promote a secondary metric to the main conclusion only because it is significant.

### 1.6 Confirmatory and Exploratory Balance

This spec should not eliminate exploratory analysis. It should separate exploratory discovery from confirmatory evidence.

Confirmatory analyses are used to support the main paper claim:

```text
all 18 meta ROIs
primary predictor = learning_specificity
primary post outcome = post_edge_specificity
primary behavior outcome = run7 memory accuracy
pre-baseline and negative-control checks
FDR within meta_metaphor / meta_spatial family
```

Exploratory analyses are allowed and useful when they are clearly labeled:

```text
alternative learning metrics: learning_self_similarity, decodability evidence
alternative post metrics: trained_edge_drop, post_pair_similarity as descriptive only
alternative behavior metrics: RT, confidence if available
alternative ROI views: focus ROIs, literature ROIs, previous-result ROIs
condition-specific models: YY only, KJ only
retrieval extensions: retrieval_pair_similarity, retrieval-minus-post, MVPA evidence
nonlinear / threshold effects if visual inspection suggests them
```

Exploratory results can be reported as hypothesis-generating if they pass basic sanity checks, but they should not replace the primary conclusion unless they are rerun under a declared sensitivity framework.

Recommended reporting language:

```text
Primary evidence:
  survives primary model, baseline control, negative controls, and ROI-family FDR.

Convergent exploratory evidence:
  same direction as primary result, biologically interpretable, but not required for the main claim.

Exploratory lead:
  interesting pattern that needs independent confirmation and should not be framed as decisive.
```

### 1.7 Model Family

Primary model A: learning -> post differentiation

```text
post_edge_specificity ~ learning_specificity_z * condition
                      + material_covariates
                      + subject random effect
                      + condition_item_id random effect
```

Primary model B: post differentiation -> memory

```text
memory ~ post_edge_specificity_z * condition
       + material_covariates
       + subject random effect
       + condition_item_id random effect
```

Because `memory` is binary, the primary implementation should be logistic mixed model or GEE. If convergence fails, report GEE with subject clustering plus item fixed/random sensitivity, not ordinary Gaussian regression as the main model.

Secondary model C: staged prediction comparison

```text
M0: memory ~ condition + covariates
M1: M0 + learning_specificity_z
M2: M1 + post_edge_specificity_z
M3: M2 + retrieval_pair_similarity_z
```

Control model D: pre-baseline confound check

```text
post_edge_specificity ~ learning_specificity_z * condition + pre_pair_similarity_z
memory ~ post_edge_specificity_z * condition + pre_pair_similarity_z
```

If the learning/post predictor disappears after adding `pre_pair_similarity_z`, the result should be interpreted as baseline item-structure driven rather than learning-induced transformation.

Control model E: residualized transformation check

```text
post_edge_specificity_resid = residual(post_edge_specificity ~ pre_pair_similarity + material_covariates)
post_edge_specificity_resid ~ learning_specificity_z * condition
memory ~ post_edge_specificity_resid_z * condition + material_covariates
```

Negative controls:

```text
pseudo_edge_drop ~ learning_specificity_z * condition
pre_pair_similarity ~ learning_specificity_z * condition
memory ~ shuffled_condition_item_predictor_z * condition
```

The main mechanism is more convincing if `learning_specificity` predicts post-pre trained-edge specificity, but not pseudo-edge change or shuffled item labels.

Material covariates should be limited and predeclared to avoid overfitting:

```text
word length / sentence length
word frequency if available
concreteness or imageability if available
semantic distance or baseline semantic association if available
run/order and trial count / missingness indicators when needed
```

Report covariate missingness. If a covariate is missing for many items, move it to sensitivity analysis rather than the primary model.

Report:

```text
beta
SE
p
BH-FDR within ROI family
AIC/BIC or cross-validated log loss
leave-one-subject-out or item-bootstrap robustness when feasible
N subjects
N condition-items
N trials
```

### 1.8 Interpretation Rules

- Significant `condition` effect means YY and KJ item pools differ on average, not item-paired YY-KJ difference.
- Significant `learning_specificity_z` means learning-stage item trace predicts post mechanism across both item pools.
- Significant `learning_specificity_z × condition` means the learning-to-post relation differs between YY and KJ.
- Significant `post_edge_specificity_z` in memory model is the desired brain-behavior mechanism bridge.
- Any learning-to-post claim must survive, or at least be reported alongside, the `pre_pair_similarity_z` baseline control.
- Do not claim causal mediation from staged model comparison. Write "incremental predictive evidence" unless a formal mediation design is added.
- If learning and post metrics from the same ROI are associated, discuss possible shared reliability / SNR contribution and report reliability or negative-control checks.
- If only retrieval predictors work, write retrieval as task-stage expression, not as evidence that learning trace monotonically increases.
- If none of the behavior models survive FDR, keep current main mechanism and state that behavioral prediction remains unresolved.

## 2. Tasks

### Task 1: Build Item-Level Mechanism Table

Create script:

```text
metaphoric/final_version/brain_behavior/build_learning_post_memory_item_table.py
```

Inputs:

```text
E:/python_metaphor/pattern_root/sub-XX/learn_yy.nii.gz
E:/python_metaphor/pattern_root/sub-XX/learn_kj.nii.gz
E:/python_metaphor/pattern_root/sub-XX/learn_yy_metadata.tsv
E:/python_metaphor/pattern_root/sub-XX/learn_kj_metadata.tsv
E:/python_metaphor/paper_outputs/qc/edge_specificity_*/edge_specificity_item.tsv or equivalent item table
E:/python_metaphor/paper_outputs/qc/retrieval_geometry/retrieval_pair_metrics.tsv
E:/python_metaphor/paper_outputs/qc/behavior_results/refined/behavior_trials.tsv
E:/python_metaphor/roi_library/manifest.tsv
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_table.tsv
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_qc.tsv
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/item_mechanism_manifest.json
```

Required columns:

```text
subject
roi_set
roi
condition
pair_id
condition_item_id
valid_condition_item
learning_self_similarity
learning_specificity
pre_pair_similarity
post_pair_similarity
trained_edge_drop
pseudo_edge_drop
post_edge_specificity
retrieval_pair_similarity
retrieval_minus_post_pair_similarity
memory
log_rt_correct
n_learning_run3_rows
n_learning_run4_rows
n_behavior_trials
learning_metric_reliability
material_covariate_missing
```

Coverage requirements:

```text
YY: expect 35 condition items per subject when data are complete
KJ: expect 35 condition items per subject when data are complete
Do not require YY/KJ pair_id intersection
Do not drop YY-only or KJ-only numeric ids
```

### Task 2: Run Learning-to-Post Models

Create script:

```text
metaphoric/final_version/brain_behavior/learning_to_post_prediction.py
```

Model:

```text
post_edge_specificity ~ learning_specificity_z * condition
post_edge_specificity ~ learning_specificity_z * condition + pre_pair_similarity_z
post_edge_specificity_resid ~ learning_specificity_z * condition
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/learning_to_post_models.tsv
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/learning_to_post_robustness.tsv
E:/python_metaphor/paper_outputs/tables_main/table_learning_to_post_prediction.tsv
```

### Task 3: Run Post-to-Memory Models

Create script:

```text
metaphoric/final_version/brain_behavior/post_edge_memory_prediction.py
```

Models:

```text
memory ~ post_edge_specificity_z * condition
memory ~ post_edge_specificity_z * condition + pre_pair_similarity_z
memory ~ post_edge_specificity_resid_z * condition
log_rt_correct ~ post_edge_specificity_z * condition
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/post_edge_memory_models.tsv
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/post_edge_memory_robustness.tsv
E:/python_metaphor/paper_outputs/tables_main/table_post_edge_memory_prediction.tsv
```

### Task 4: Negative Controls and Robustness

Create script:

```text
metaphoric/final_version/brain_behavior/negative_control_robustness.py
```

Run:

```text
pseudo_edge_drop ~ learning_specificity_z * condition
pre_pair_similarity ~ learning_specificity_z * condition
memory ~ shuffled_condition_item_predictor_z * condition
item bootstrap for primary slopes
leave-one-subject-out predictive check for memory model when feasible
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/negative_control_models.tsv
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/bootstrap_sensitivity.tsv
E:/python_metaphor/paper_outputs/tables_supp/table_learning_post_negative_controls.tsv
```

### Task 5: Staged Model Comparison

Create script:

```text
metaphoric/final_version/brain_behavior/staged_memory_model_comparison.py
```

Compare:

```text
M0 condition + covariates
M1 M0 + learning_specificity_z
M2 M1 + post_edge_specificity_z
M3 M2 + retrieval_pair_similarity_z
M4 M2 + retrieval_minus_post_pair_similarity_z
```

Outputs:

```text
E:/python_metaphor/paper_outputs/qc/learning_post_memory_prediction/staged_model_comparison.tsv
E:/python_metaphor/paper_outputs/tables_main/table_staged_memory_model_comparison.tsv
```

### Task 6: Figures

Create figure script only after model results pass QC:

```text
metaphoric/final_version/figures/plot_learning_post_memory_prediction.py
```

Candidate figures:

```text
fig_learning_to_post_prediction.png/pdf
fig_post_edge_memory_prediction.png/pdf
fig_staged_model_comparison.png/pdf
```

Do not create a figure if all predictive models fail FDR and show no coherent effect direction.

## 3. Checklist

### 3.1 Data Integrity

- [ ] `condition_item_id` is always `condition + "_" + pair_id`.
- [ ] No model treats `yy_28` and `kj_28` as paired items.
- [ ] YY and KJ each retain their own item pool.
- [ ] YY-only numeric ids `4,18,32,35,40` are retained.
- [ ] KJ-only numeric ids `20,25,30,31,34` are retained.
- [ ] No analysis takes the YY/KJ pair_id intersection as the default item set.
- [ ] All item-level joins use at least `subject + roi_set + roi + condition + pair_id`.
- [ ] Behavioral joins use `subject + condition + pair_id`; word-level joins additionally use `word_label` if trial-level rows are required.
- [ ] `pair_id` normalization handles `1` and `1.0` consistently.
- [ ] Missing run7 subjects are explicitly recorded.

### 3.2 Metric QC

- [ ] Learning run3/run4 volumes match metadata row counts.
- [ ] `learning_self_similarity` has finite values for most subject-condition-items.
- [ ] `learning_specificity` uses same-condition mismatches only.
- [ ] Similarities are Fisher-z transformed before averaging or modeling.
- [ ] Primary predictors are z-scored in a documented scope: ROI-level primary, subject x condition sensitivity if needed.
- [ ] Learning metric reliability / finite-cell coverage is reported by ROI.
- [ ] `post_edge_specificity` is computed from trained edge minus same-condition pseudo/null edge.
- [ ] Primary learning-to-post outcome is `post_edge_specificity` or `trained_edge_drop`, never raw post similarity alone.
- [ ] `pre_pair_similarity` is included as a baseline control in sensitivity models.
- [ ] Residualized `post_edge_specificity_resid` is available for baseline-robust checks.
- [ ] Retrieval predictors are optional and never required for primary post mechanism models.
- [ ] QC reports number of subjects, ROIs, items, trials, and missing cells by condition.
- [ ] Material covariate missingness is reported before model fitting.

### 3.3 Statistical QC

- [ ] Models are fit separately by ROI.
- [ ] Confirmatory FDR uses all 18 meta ROIs, split by `meta_metaphor` and `meta_spatial`.
- [ ] Focus ROI displays are labeled as focus visualization, not as a reduced confirmatory family.
- [ ] Subject clustering/random effect is included.
- [ ] Item-within-condition random effect or cluster-robust item SE is included when feasible.
- [ ] Memory accuracy is modeled with logistic mixed model or GEE, not primary Gaussian regression.
- [ ] If MixedLM/GLMM fails, fallback GEE must cluster by subject and include item fixed/random sensitivity.
- [ ] Negative controls include pseudo-edge, pre-similarity, and shuffled item labels.
- [ ] Item bootstrap or leave-one-subject-out robustness is reported when feasible.
- [ ] Report both KJ/reference slope and YY interaction slope when using `* condition`.
- [ ] Staged model comparison is labeled secondary and is not interpreted as mediation.

### 3.4 Interpretation QC

- [ ] YY-KJ differences are described as condition-level item-pool differences.
- [ ] Do not use “item-matched YY-KJ” language.
- [ ] Do not claim learning-stage progressive change unless run4-run3 evidence supports it.
- [ ] Do not revive ERS / reinstatement as a main mechanism.
- [ ] Do not claim learning-induced transformation from a model that ignores pre baseline similarity.
- [ ] Do not claim a learning mechanism if the same predictor mainly explains pseudo-edge or pre similarity.
- [ ] Discuss shared ROI reliability / SNR as a limitation if learning and post metrics are associated in the same ROI.
- [ ] If post-edge specificity predicts memory, make it the primary brain-behavior bridge.
- [ ] If prediction fails, state that neural reorganization is robust but its item-level behavioral consequence remains unresolved.

### 3.5 Stop Rules

- [ ] Stop after Task 1 if item table cannot recover `condition + pair_id` accurately.
- [ ] Stop after Task 1 if YY/KJ item counts imply accidental numeric intersection filtering.
- [ ] Stop after Task 2 if learning predictors cannot be aligned with post edge metrics for enough items.
- [ ] Stop after Task 4 if the primary effect is no stronger than pseudo-edge / shuffled-label controls.
- [ ] Stop before figures if all model effects are incoherent or fail basic QC.
- [ ] Do not add exploratory ROI families until primary ROI set has been evaluated.
