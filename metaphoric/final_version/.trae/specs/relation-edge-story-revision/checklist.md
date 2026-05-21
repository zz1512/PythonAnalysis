# Checklist: relation_edge_story_revision

## Story alignment

- [ ] The revision keeps the existing stagewise story.
- [ ] The revision does not introduce a full causal mediation chain.
- [ ] Behavior is treated as the anchor, not as proof of mechanism.
- [ ] Learning is described as condition-level semantic geometry.
- [ ] Post-learning trained-edge differentiation is the core mechanism evidence.
- [ ] Retrieval is described as task-driven reconstruction / rebound.
- [ ] Memory is interpreted as stronger prior post-stage separation unless new models show otherwise.
- [ ] KJ is described as active spatial-relational comparison.

## Global constraints

- [ ] No new main ROI family is introduced.
- [ ] Main-text network analyses use only semantic/metaphor and hpc-spatial networks.
- [ ] Item-level mixed models use `condition_item_id`.
- [ ] Bare `pair_id` is not used as the random item effect.
- [ ] No large mediation model is added.
- [ ] Uncorrected p-values are not used as main evidence.
- [ ] B2 is not described as strict crossnobis.
- [ ] HPC long-axis is not described as a subfield mechanism.
- [ ] Retrieval-post difference is not directly equated with retrieval reinstatement.

## Directory safety

- [ ] Scripts are written under `paper_analysis/revision_relation_edge/`.
- [ ] QC outputs are written under `paper_outputs/qc/revision_relation_edge/`.
- [ ] New figures are written under `paper_outputs/figures_revision/`.
- [ ] Existing upstream outputs are not overwritten.
- [ ] Existing `result_story.md` and `result_final.md` are not overwritten by `r8` unless explicitly requested.

## Source-first audit

- [x] `r0a_source_inventory.py` is created.
- [x] Original `E:/python_metaphor/data_events` CSV/TSV schemas are audited.
- [x] `pattern_root` metadata and NIfTI availability are audited.
- [x] `materials_detail` Excel sheets are audited.
- [x] `roi_library/manifest.tsv` is audited.
- [x] Variable provenance is written to `revision_variable_provenance.tsv`.
- [x] Run3/run4 behavior is sourced from raw CSV columns, not compact event TSVs.
- [x] Retrieval geometry is recomputed from source patterns into `paper_outputs/source_recompute/`.
- [x] Source-recomputed retrieval geometry is compared against the locked upstream table.
- [ ] MVPA and all other upstream neural summaries are source-recomputed if a fully strict source-only rerun is required.

## R0 master table

- [ ] `revision_master_item.tsv` is created.
- [ ] `revision_master_network.tsv` is created.
- [ ] `revision_covariate_coverage.tsv` is created.
- [ ] `revision_schema_audit.tsv` is created.
- [ ] `subject` is present.
- [ ] `condition` is present.
- [ ] `condition_item_id` is present.
- [ ] `pair_id_raw` is retained for audit only.
- [ ] `network` is present.
- [ ] `pre_pair_similarity_z` is present.
- [ ] `post_pair_similarity_z` is present.
- [ ] `retrieval_pair_similarity_z` is present.
- [ ] `post_separation_z` is present.
- [ ] `retrieval_rebound_z` is present.
- [ ] `memory_ord`, `memory_strict`, and `memory_lenient` are present.
- [ ] Material covariates are present when available.
- [ ] Learning behavior variables are present when available.
- [ ] `YY_28` and `KJ_28` remain distinct.
- [ ] Missingness is reported by condition and network.

## R1 material moderation

- [ ] Memory model is fit.
- [ ] Post-separation model is fit.
- [ ] YY-only memory model is fit.
- [ ] YY-only post-separation model is fit.
- [ ] Optional quadratic semantic-distance model is run only if coverage is sufficient.
- [ ] FDR or planned-test tier is documented.
- [ ] Results distinguish moderator evidence from covariate-boundary evidence.
- [ ] Figure is generated in PNG and PDF.

## R2 high/low post-separation profile

- [ ] YY top 30% and bottom 30% groups are created.
- [ ] Median split sensitivity is created.
- [ ] Material variables are compared.
- [ ] Learning behavior variables are compared.
- [ ] Memory score is compared descriptively.
- [ ] Item-mean and mixed-model versions are auditable.
- [ ] The result is written as descriptive profile, not main mechanism, unless very stable.

## R3 pre-post-retrieval geometry

- [ ] `d_pre_post` is computed.
- [ ] `d_post_retrieval` is computed.
- [ ] `d_pre_retrieval` is computed.
- [ ] `return_to_pre_index` is computed.
- [ ] Vector alignment is attempted only if voxel patterns are available.
- [ ] Missing vector alignment is documented as unavailable, not as negative evidence.
- [ ] Main models include condition, memory, and condition x memory.
- [ ] Interpretation distinguishes return-to-pre from new-state reconstruction.
- [ ] Review markdown is generated.

## R4 network coupling revision

- [ ] Only semantic/metaphor and hpc-spatial networks are modeled.
- [ ] Stage coupling model is fit.
- [ ] Retrieval rebound coupling model is fit.
- [ ] Memory coupling model is fit.
- [ ] HPC long-axis coupling is marked exploratory if run.
- [ ] YY-specific coupling is claimed only if interaction is stable.
- [ ] Coupling is not written as causal communication.

## R5 MVPA stage-state summary

- [ ] Pre decoding is summarized.
- [ ] Learning run3/run4 decoding is summarized.
- [ ] Post decoding is summarized.
- [ ] Run7 retrieval decoding is summarized.
- [ ] Cross-role generalization is summarized as boundary evidence.
- [ ] MVPA-behavior bridge is summarized as boundary evidence unless stable.
- [ ] Output clearly separates RSA geometry from MVPA stage-state information.

## R6 evidence tier and story

- [ ] Tier A table is generated.
- [ ] Tier B table is generated.
- [ ] Tier C table is generated.
- [ ] Main-text result map is generated.
- [ ] Supplementary result map is generated.
- [ ] Boundary result map is generated.
- [ ] `storyline_revision.md` uses the upgraded story wording.
- [ ] Claims-to-use and claims-to-avoid are explicit.

## R7 revision figures

- [ ] Main Fig 1 design + behavior is generated or mapped.
- [ ] Main Fig 2 learning condition geometry + MVPA inset is generated or mapped.
- [ ] Main Fig 3 post trained-edge differentiation + word identity control is generated or mapped.
- [ ] Main Fig 4 stage trajectory + trajectory geometry is generated or mapped.
- [ ] Main Fig 5 subsequent memory / prior post separation is generated or mapped.
- [ ] Main Fig 6 semantic-hpc coupling + robustness summary is generated or mapped.
- [ ] Supplementary figure manifest is generated.
- [ ] Empty panels are detected before finalizing figures.
- [ ] Figure paths are portable relative paths.

## R8 story documents

- [ ] `--dry-run` writes `revision_story_preview.md`.
- [ ] `--replace-section` can replace only marked sections.
- [ ] `--skip-if-exists` prevents accidental overwrite.
- [ ] `result_story_revision.md` is generated.
- [ ] `result_final_revision.md` is generated.
- [ ] `revision_append_log.txt` records changed sections and figures.

## Final acceptance

- [ ] The revised story answers: which metaphors separate more?
- [ ] The revised story answers: is retrieval return-to-pre or reconstruction?
- [ ] The revised story answers: whether semantic-hpc coupling supports reconstruction.
- [ ] New findings are integrated without losing prior core results.
- [ ] Weak or null findings are written as boundaries, not hidden.
- [ ] The final manuscript does not overclaim causality.
