# Source Inventory Review

## Summary

- Source root: `E:\python_metaphor`
- Event subjects detected: 28
- Pattern-root subjects detected: 28
- Run3 raw CSV files with `sentence3_RT`: 56
- Run4 raw CSV files with `sentence4_RT`: 56

## Interpretation

Run3/run4 behavior is available from original E-Prime CSV files and should be merged into the revision master table. The current r0 script now reads the source-derived `learning_behavior_item.tsv`, rebuilding it from raw CSVs if missing.

The following neural variables are still read from locked upstream neural summary tables, not recomputed inside the revision scripts:

- `pre/post/retrieval pair similarity`
- `post_separation/retrieval_rebound`
- `semantic/hpc-spatial network composites`
- `MVPA stage-state summary`

This is acceptable for fast manuscript revision only if those upstream tables are treated as frozen outputs from source-level RSA/MVPA pipelines. For a strict source-only rerun, add a separate neural recomputation step from `pattern_root/*.nii.gz` plus `roi_library` masks before r1-r5.
