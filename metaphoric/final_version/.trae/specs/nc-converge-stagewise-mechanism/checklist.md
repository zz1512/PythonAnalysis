# Checklist: NC Converge Stagewise Mechanism Plan

## Story Lock

- [ ] The main story is stagewise, not causal.
- [ ] The final framing avoids a full mediation chain.
- [ ] YY is described as metaphorical relation learning.
- [ ] KJ is described as an active spatial-relational learning comparison condition.
- [ ] Memory interpretation emphasizes prior post-stage separation unless retrieval similarity independently supports reinstatement.

## Script Safety

- [ ] New outputs are written only under `paper_outputs/qc/nc_converge/`.
- [ ] Exploratory hippocampal outputs are written only under `paper_outputs/qc/hpc_subfield_three_axis/`.
- [ ] No upstream result tables are overwritten.
- [ ] Append scripts are run only after dry-run review.
- [ ] Existing user or upstream changes are not reverted.

## A2 Schema Lock

- [ ] `condition_item_id` is available or reconstructable.
- [ ] Duplicate audit is clean or clearly documented.
- [ ] Subject intersection is generated.
- [ ] run7 neural and behavior memory subjects are auditable.

Required outputs:

```text
id_audit.tsv
subject_intersection.tsv
schema_lock.log
```

## A3 Shared-Post Review

- [ ] Post-stage shared component is audited.
- [ ] YY/KJ/baseline post structure is reviewed.
- [ ] Main post-stage effects are not trivially explained by shared post structure.

Required outputs:

```text
shared_post_review.tsv
shared_post_review.md
shared_post_input_manifest.tsv
```

## A4 Memory Component Model

- [ ] Post and retrieval similarity are included as separate predictors.
- [ ] Model includes condition interactions.
- [ ] Subject and condition item grouping are included through `fit_formula` when possible.
- [ ] Available material covariates are included.
- [ ] YY memory bridge is interpreted as prior post-stage separation if post dominates retrieval.

Required outputs:

```text
memory_component_model.tsv
memory_component_roi_followup.tsv
memory_component_input_manifest.tsv
```

Pass condition:

```text
YY-related post_pair_similarity / post separation term is directionally interpretable
and survives the intended evidence threshold, or provides a clear boundary condition.
```

## A4 Stage Trajectory Model

- [ ] Long-form table includes pre, post, and retrieval stages.
- [ ] `pair_similarity ~ condition * stage` is fit.
- [ ] YY post drop is separately evaluated.
- [ ] YY retrieval rebound is separately evaluated.
- [ ] The text says trajectory characterization, not causal path.

Required outputs:

```text
stage_trajectory_model.tsv
stage_trajectory_descriptives.tsv
```

Pass condition:

```text
YY shows a directionally coherent post-stage drop and retrieval-stage rebound,
or the absence of this pattern is explicitly treated as a limit.
```

## A5 Network Composite

- [ ] ROI-level metrics are mapped into predefined networks.
- [ ] Semantic/metaphor and hpc-spatial composites are present.
- [ ] Network aggregation is z-scored in a defensible way.
- [ ] Composite outputs are traceable to input manifests.

Required outputs:

```text
network_composite_summary.tsv
network_trajectory_long.tsv
network_composite_input_manifest.tsv
```

## B2 Crossnobis Robustness

- [ ] Correlation-based Step5C effects have matching or interpretable crossnobis inputs.
- [ ] Direction consistency is evaluated.
- [ ] Magnitude divergence is documented.
- [ ] Missing crossnobis inputs do not get misread as negative evidence.

Required outputs:

```text
crossnobis_step5c.tsv
crossnobis_input_manifest.tsv
```

Pass condition:

```text
Main Step5C direction is preserved for the hpc-spatial / hippocampal direction,
or mismatches are clearly bounded.
```

## B3 Network-Level Coupling

- [ ] Coupling is restricted to semantic/metaphor and hpc-spatial networks.
- [ ] No exploratory ROI x ROI scan is introduced.
- [ ] Learning semantic geometry and post hpc-spatial separation are tested if inputs permit.
- [ ] Retrieval semantic rebound and retrieval hpc-spatial rebound are tested if inputs permit.

Required outputs:

```text
repr_coupling.tsv
repr_coupling_input_manifest.tsv
```

Pass condition:

```text
At least one predefined semantic/metaphor <-> hpc-spatial coupling is stable
or directionally supportive.
```

## A6 FDR Tiering

- [ ] Tier A contains primary planned tests.
- [ ] Tier B contains robustness or ROI follow-up tests.
- [ ] Tier C contains exploratory analyses only.
- [ ] Main-text claims rely on Tier A/B, not Tier C alone.

Required outputs:

```text
fdr_tier_manifest.tsv
fdr_tier_summary.md
```

## Go/No-Go

Primary GO rule:

```text
At least two of A4, B2, and B3 support the main direction.
```

Minimum fallback rule:

```text
At least one of A4/B1/B2/B3 is stable enough to preserve the stagewise
reorganization story, with weaker links written as boundary conditions.
```

Required outputs:

```text
go_no_go_stage6.md
go_no_go_evidence.tsv
```

Decision labels:

- [ ] `GO`: proceed to writing consolidation and optional mechanism upgrades.
- [ ] `HOLD`: do not expand; write weak links as limitations.
- [ ] `NO-GO`: stop mechanism expansion and return to the most stable descriptive story.

## C1 Input-Output Transformation

- [ ] `pre_pair_similarity` and `post_pair_similarity` exist.
- [ ] Condition is cleanly coded as YY/KJ.
- [ ] The model tests `pre_pair_similarity * condition`.
- [ ] The result is interpreted as input-output transformation, not generic pattern separation.

Pass condition:

```text
YY has a lower or otherwise meaningfully altered pre->post slope.
```

## C2 Global Similarity / Centrality

- [ ] Item x item RDM matrices are complete enough.
- [ ] Metadata ordering matches matrix ordering.
- [ ] Global similarity excludes self-comparisons.
- [ ] Pair-level similarity and global centrality are not conflated.

Pass condition:

```text
Remembered YY items show lower post global centrality or meaningful
post-pre centrality change, especially alongside lower pair similarity.
```

## C3 Novelty / Familiarity Moderation

- [ ] Novelty, familiarity, comprehensibility, and difficulty are available or missingness is audited.
- [ ] YY/KJ material imbalance is described honestly.
- [ ] Condition effect is tested with novelty covariates.
- [ ] Novelty x condition is interpreted as moderation if stable.

Pass condition:

```text
YY condition remains stable after novelty/familiarity controls, or novelty
moderates YY differentiation in a theoretically useful direction.
```

## Optional Learning-Update Proxy

- [ ] run3 and run4 behavioral metrics join cleanly.
- [ ] RT reduction is coded in the intended direction.
- [ ] The proxy is described as weak learning-update evidence.
- [ ] Non-significant results are not used to weaken the primary story.

## hpc_subfield_three_axis Entry Gate

- [ ] `nc_converge` supports hpc-spatial direction.
- [ ] B2 crossnobis supports Step5C direction.
- [ ] Mask voxel counts pass QC.
- [ ] Signal quality is acceptable.
- [ ] FreeSurfer segmentation availability is documented.
- [ ] MNI y-axis fallback is labeled exploratory.
- [ ] MDS is visualization only.
- [ ] No CA3/DG claim is made.

## Writing Checklist

- [ ] Results starts from behavior and stagewise neural evidence.
- [ ] Learning section emphasizes condition-level geometry.
- [ ] Post section emphasizes trained-edge differentiation.
- [ ] Retrieval section emphasizes task-driven rebound.
- [ ] Memory section emphasizes prior post-stage separation.
- [ ] Discussion avoids overclaiming causality.
- [ ] Limitations explicitly mention non-causal design and exploratory long-axis segmentation if used.
- [ ] Supplement houses weak, exploratory, or negative-control analyses.

