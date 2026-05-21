# Shared Difference-Score Artifact Audit

Date: 2026-05-10

Purpose: review previous analyses for the same risk found in Stage 1:

```text
X = A - B
Y = C - B
```

When `X` is used to predict/correlate with `Y`, the shared `B` term can mechanically induce positive coupling even if `A` and `C` are unrelated.

## Summary Judgment

The strongest risk is concentrated in analyses that link post-stage separation to run7 retrieval re-binding.

The following statement should be downgraded or rewritten:

```text
post-stage hippocampal-spatial separation predicts retrieval-stage re-binding
```

The safer statement is:

```text
YY shows both post-stage edge differentiation and retrieval-stage pair-structure rebound, but their item-level/network-level coupling is not robust after controlling post similarity.
```

## High-Risk Analyses

### 1. Network bridge: hpc separation -> retrieval rebinding

Files:

- `brain_behavior/learning_reactivation/network_bridge_models.py`
- `brain_behavior/mixed_effects/run_mixed_effects.py`
- `stagewise_mechanism_chain_spec.md`
- `result_new_meta_roi.md`
- `result_new_meta_roi_v1.md`

Problematic logic:

```text
hpc_spatial_separation = mean(post_edge_specificity)
post_edge_specificity  = trained_edge_drop - pseudo_edge_drop
trained_edge_drop      = pre_pair_similarity - post_pair_similarity

retrieval_rebinding    = retrieval_pair_similarity - post_pair_similarity
```

Even if `post_edge_specificity` additionally subtracts pseudo-edge drop, its core trained-edge component still contains `- post_pair_similarity`. `retrieval_rebinding` also contains `- post_pair_similarity`.

Therefore:

```text
retrieval_rebinding ~ hpc_spatial_separation
```

can be inflated by the shared post term.

Stage 1 post-controlled reanalysis confirmed this risk:

```text
raw hpc separation -> retrieval rebinding:
  strong positive

post-controlled / retrieval-similarity model:
  hpc_spatial_post_separation_z = +0.0094, p = .732
  YY interaction                = +0.0064, p = .465
```

Decision:

- Treat previous positive bridge as contaminated by shared post variance.
- Do not use it as a primary mechanism.
- Keep only as a deprecated/raw continuity check if needed.

### 2. Result document claims that should be revised

In `result_new_meta_roi.md`, sections 19.5 and 20.5 state that hpc-spatial separation strongly predicts retrieval re-binding.

These lines should be revised because they were based on the raw shared-post model.

Examples of claims to downgrade:

```text
hpc_spatial_separation 显著预测 retrieval rebinding
最强 bridge 是 hpc-spatial separation -> retrieval-stage re-binding
post separation 进一步预测 retrieval-stage re-binding
```

Recommended replacement:

```text
Raw models suggested a strong separation-rebinding coupling, but this association was not robust after controlling post-stage similarity. Therefore, post differentiation and retrieval rebound should be treated as adjacent stagewise signatures rather than a confirmed item-level bridge.
```

In `result_new_meta_roi_v1.md`, the sentence

```text
The strongest bridge remains hippocampal-spatial post separation predicting retrieval-stage re-binding.
```

should also be revised.

## Medium-Risk Analyses

### 3. Staged memory model comparison: M4 retrieval_minus_post

File:

- `brain_behavior/staged_memory_model_comparison.py`

Model:

```text
memory ~ condition + learning_specificity_z + post_edge_specificity_z + retrieval_minus_post_pair_similarity_z
```

This is not the same artifact because the outcome is memory, not another difference score sharing post.

However, `post_edge_specificity_z` and `retrieval_minus_post_pair_similarity_z` share post-stage information, so model comparison can be hard to interpret:

- M4 may improve log-loss because it carries retrieval information;
- or because it absorbs post-stage residual structure already related to post_edge_specificity.

Decision:

- Keep as exploratory/secondary only.
- Prefer a cleaner comparison:

```text
memory ~ retrieval_pair_similarity_z + post_pair_similarity_z + pre_pair_similarity_z + condition
```

or compare retrieval-pair models with explicit post/pre controls.

### 4. Directional mapping -> post edge specificity

Files:

- `brain_behavior/learning_reactivation/directional_mapping_assimilation.py`
- `brain_behavior/learning_reactivation/reactivation_to_post_edge.py`
- `brain_behavior/mixed_effects/run_mixed_effects.py`

This is not the exact shared `-post` scalar problem, because mapping metrics use cross-phase role similarities such as:

```text
post_role_a_pre_role_b - post_role_a_pre_role_a
```

while post edge specificity uses pair-level pre/post similarity.

But both variables are computed from overlapping pre/post neural patterns, so they are not fully independent.

Decision:

- Keep as exploratory.
- Interpret only if robust to pre similarity, self-stability, and material covariates.
- Do not make it a primary bridge.

## Low-Risk / Mostly Safe Analyses

### 5. Retrieval-minus-post rebound as a one-sample or YY-KJ contrast

Files/sections:

- `result_new_meta_roi.md`, run7 retrieval rebound section
- retrieval geometry outputs

This is mostly safe because it asks whether:

```text
retrieval_pair_similarity - post_pair_similarity > 0
```

or whether YY > KJ on that same contrast.

There is no second predictor/outcome difference score sharing `post` inside a regression. The interpretation should remain:

```text
run7 shows task-driven pair-structure rebound relative to post.
```

Not:

```text
post separation causes/predicts rebound.
```

### 6. Post learned-edge differentiation itself

Files/sections:

- edge specificity analyses
- mixed-effects M1 post edge condition

The core comparison:

```text
pre_pair_similarity - post_pair_similarity
```

and contrasts against KJ or pseudo/non-edge are valid as within-analysis stage contrasts.

This does not suffer the same problem unless it is used to predict another difference score sharing `post`.

### 7. Learning -> post models

Files:

- `brain_behavior/learning_to_post_prediction.py`
- `brain_behavior/learning_reactivation/reactivation_to_post_edge.py`
- mixed-effects M3 models

These use learning-stage predictors to predict post edge specificity. They do not share the same post term with the predictor, so they are not the same artifact.

Their weakness remains theoretical/statistical: learning predictors are small and not robust, but not because of shared post difference-score coupling.

### 8. Post edge specificity -> memory

File:

- `brain_behavior/post_edge_memory_prediction.py`

Outcome is behavior, so there is no shared neural post term in the outcome.

This analysis is not contaminated by the Stage 1 artifact. It remains weak because effects are not stable after correction/negative controls, not because of shared post algebra.

### 9. Four-phase trajectory

File:

- `brain_behavior/mixed_effects/run_mixed_effects.py`

Model:

```text
pair_similarity ~ phase * condition
```

This uses raw phase similarities rather than correlating difference scores. It is safe and useful for describing stagewise trajectories.

### 10. MVPA analyses

MVPA classification analyses do not use the shared difference-score structure and are not affected by this artifact.

## Practical Revision Checklist

- Rewrite all "post separation predicts retrieval re-binding" claims as raw-model-only or remove them from the main chain.
- Keep `retrieval - post` rebound as descriptive stage evidence.
- Keep post edge differentiation as a main mechanism.
- Keep four-phase trajectory and MVPA as state evidence.
- For future bridge models, avoid `A - B -> C - B`. Use one of:

```text
C ~ (A - B) + B
C - B ~ (A - B) + B
C ~ A + B + condition
```

and report whether the bridge survives the control.
