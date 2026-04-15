# Dimension Story Python README

## 1. Goal

This folder now contains a fuller Python conversion of the MATLAB analysis story in `I:\FLXX1\Dimension`.

The code is organized by analysis stage instead of by many one-off scripts:

1. remembered / forgotten ROI analysis
2. memory-good / memory-poor ROI analysis
3. main ROI dimensionality and robustness analyses
4. pooled behavioral item analyses
5. whole-brain dimensionality searchlight
6. hippocampal seed PCA-connectivity searchlight


## 2. Main Python Files

- [dimension_analysis_core.py](/I:/FLXX1/Dimension/dimension_analysis_core.py:1)
  - shared functions for the two previously built ROI tasks
  - handles mask loading, NIfTI loading, event reading, RDM dimensionality, paired t-tests, `R^2`, CSV / JSON writing

- [dimension_story_utils.py](/I:/FLXX1/Dimension/dimension_story_utils.py:1)
  - shared functions for the full research-story conversion
  - adds pattern-PCA dimensionality, same-trial resampling, demean logic, cross-subject voxel filtering, behavioral pooling, searchlight generation, and group-level map statistics

- [run_remembered_forgotten_dimension.py](/I:/FLXX1/Dimension/run_remembered_forgotten_dimension.py:1)
  - task 1
  - extends the remembered-only MATLAB logic to forgotten trials

- [run_memory_group_dimension.py](/I:/FLXX1/Dimension/run_memory_group_dimension.py:1)
  - task 2
  - splits subjects into memory-good / memory-poor and compares `HSC` vs `LSC` within each group

- [run_story_roi_suite.py](/I:/FLXX1/Dimension/run_story_roi_suite.py:1)
  - main ROI suite
  - consolidates the core ROI MATLAB scripts and major robustness checks

- [run_story_behavior_suite.py](/I:/FLXX1/Dimension/run_story_behavior_suite.py:1)
  - converts the pooled item-score exploratory scripts

- [run_story_searchlight_suite.py](/I:/FLXX1/Dimension/run_story_searchlight_suite.py:1)
  - converts `searchlight_RD.m`, `my_var_measure.m`, and the paired group map step

- [run_story_connectivity_suite.py](/I:/FLXX1/Dimension/run_story_connectivity_suite.py:1)
  - converts `stage1_pca_conn_searchlight.m` and `my_measure.m`

- [memory_dimension_analysis.py](/I:/FLXX1/Dimension/memory_dimension_analysis.py:1)
  - lightweight launcher help
  - prints the full list of entry scripts and their MATLAB mapping


## 3. Fixed Data Paths

These defaults are written as absolute paths so the scripts still point to the current project data even if you move the Python files into another project folder.

- main pattern directory:
  - `I:\FLXX1\Dimension\pattern`

- remembered / forgotten single-trial directory:
  - `I:\FLXX1\First_level\patterns_hlrf`

- event directory:
  - `I:\FLXX1\events`

- left hippocampus mask:
  - `I:\FLXX1\mask\Hippocampus_L.nii`

- right hippocampus mask:
  - `I:\FLXX1\mask\Hippocampus_R.nii`

- subject-specific searchlight masks:
  - `L:\FLXX_1\mvpa\GLM_item_allexample\sub-XX\mask.nii`

- pooled item `glm_T_all.nii` directory:
  - `L:\FLXX_1\mvpa\GLM_item_allexample\rsa`

- alternative behavior event directory used in `Untitled2.m` / `Untitled3.m`:
  - `L:\FLXX_1\First_level\get_onset\newdata_to_use`

- output root:
  - `I:\FLXX1\Dimension\python_results`


## 4. MATLAB to Python Mapping

### 4.1 Remembered / forgotten

MATLAB source:
- [only_consider_remembered_items.m](/I:/FLXX1/Dimension/only_consider_remembered_items.m:1)

Python:
- [run_remembered_forgotten_dimension.py](/I:/FLXX1/Dimension/run_remembered_forgotten_dimension.py:1)

### 4.2 Memory-good / memory-poor

MATLAB source:
- no single clean MATLAB entry script existed; this Python task formalizes the grouped ROI comparison

Python:
- [run_memory_group_dimension.py](/I:/FLXX1/Dimension/run_memory_group_dimension.py:1)

### 4.3 Main ROI dimensionality and robustness

MATLAB sources:
- [hl_patternbutnotrdm.m](/I:/FLXX1/Dimension/hl_patternbutnotrdm.m:1)
- [hl.m](/I:/FLXX1/Dimension/hl.m:1)
- [demeantest.m](/I:/FLXX1/Dimension/demeantest.m:1)
- [demeantest2.m](/I:/FLXX1/Dimension/demeantest2.m:1)
- [get_voxel_mask.m](/I:/FLXX1/Dimension/get_voxel_mask.m:1)
- [sametrialnumber_maskvoxel.m](/I:/FLXX1/Dimension/sametrialnumber_maskvoxel.m:1)
- [sametrialnumber_first_3_pc.m](/I:/FLXX1/Dimension/sametrialnumber_first_3_pc.m:1)

Python:
- [run_story_roi_suite.py](/I:/FLXX1/Dimension/run_story_roi_suite.py:1)

### 4.4 Behavioral pooling

MATLAB sources:
- [across_sub_dim.m](/I:/FLXX1/Dimension/across_sub_dim.m:1)
- [Untitled2.m](/I:/FLXX1/Dimension/Untitled2.m:1)
- [Untitled3.m](/I:/FLXX1/Dimension/Untitled3.m:1)

Python:
- [run_story_behavior_suite.py](/I:/FLXX1/Dimension/run_story_behavior_suite.py:1)

### 4.5 Whole-brain searchlight

MATLAB sources:
- [searchlight_RD.m](/I:/FLXX1/Dimension/searchlight_RD.m:1)
- [my_var_measure.m](/I:/FLXX1/Dimension/my_var_measure.m:1)
- [pairedttest.m](/I:/FLXX1/Dimension/pairedttest.m:1)

Python:
- [run_story_searchlight_suite.py](/I:/FLXX1/Dimension/run_story_searchlight_suite.py:1)

### 4.6 Seed connectivity searchlight

MATLAB sources:
- [stage1_pca_conn_searchlight.m](/I:/FLXX1/Dimension/stage1_pca_conn_searchlight.m:1)
- [my_measure.m](/I:/FLXX1/Dimension/my_measure.m:1)

Python:
- [run_story_connectivity_suite.py](/I:/FLXX1/Dimension/run_story_connectivity_suite.py:1)


## 5. Core Logic Notes

### 5.1 Event reading

Memory accuracy uses:
- `sub-XX_run-1_events.txt`
- `sub-XX_run-2_events.txt`
- `sub-XX_run-3_events.txt`
- `sub-XX_run-4_events.txt`

Rules:
- only keep `trial_type == "e"`
- `memory == 1` means remembered
- `memory == -1` means forgotten

### 5.2 Dimensionality metrics

Two ROI dimensionality styles are used because the MATLAB code uses both:

1. `pattern` mode
   - direct PCA on the `voxel x trial` matrix
   - used for scripts like `hl_patternbutnotrdm.m`

2. `rdm` mode
   - build the trial-by-trial correlation-distance RDM
   - run PCA on the RDM
   - used for scripts like `hl.m`, `only_consider_remembered_items.m`, and the searchlight dimension story

### 5.3 Trial equalization

The ROI suite supports:
- `all`
  - use all trials
- `min`
  - equalize to the smaller condition count within subject
- `fixed`
  - equalize to a fixed trial count such as 20 or 25

Random equalization follows the MATLAB style:
- repeated subsampling without replacement
- average the metric across 1000 resamples by default

### 5.4 Effect size

All paired ROI summaries report:
- `n_subjects`
- `degrees_of_freedom`
- `t_stat`
- `p_value`
- `r_squared`
- `mean_hsc`
- `mean_lsc`

`r_squared` is computed as:

```text
R^2 = t^2 / (t^2 + df)
```


## 6. What `run_story_roi_suite.py` Includes

The ROI suite writes one subfolder per analysis stage:

- `hl_patternbutnotrdm`
  - left hippocampus
  - threshold sweep `70:80`
  - direct pattern PCA

- `hl_rdm_baseline`
  - right hippocampus
  - threshold `80`
  - RDM-based PCA

- `sametrial_rdm_fixed20`
  - left hippocampus
  - fixed `20` trials
  - threshold sweep `70:80`
  - RDM-based PCA

- `demean_pattern_min`
  - left hippocampus
  - demean after concatenating HSC and LSC
  - equalize to the smaller condition count
  - threshold sweep `70:90`

- `demean_pattern_fixed25`
  - left hippocampus
  - demean after concatenating HSC and LSC
  - fixed `25` trials
  - threshold sweep `70:90`

- `cross_subject_voxel_mask`
  - builds a voxel keep-mask using a paired HSC vs LSC voxel-wise test across subjects

- `voxelmask_pattern_fixed20`
  - left hippocampus
  - apply the cross-subject voxel mask
  - fixed `20` trials
  - threshold sweep `70:80`

- `pc_window_3_5_fixed20`
  - left hippocampus
  - fixed `20` trials
  - uses the summed explained variance from PCs `3:5`


## 7. Installation

```bash
pip install -r I:\FLXX1\Dimension\requirements_memory_dimension.txt
```

Dependencies:
- `numpy`
- `scipy`
- `nibabel`


## 8. How To Run

### 8.1 Custom tasks already built

```bash
python I:\FLXX1\Dimension\run_remembered_forgotten_dimension.py
python I:\FLXX1\Dimension\run_memory_group_dimension.py --good-memory-threshold 0.2
```

### 8.2 Full ROI story

```bash
python I:\FLXX1\Dimension\run_story_roi_suite.py
```

### 8.3 Behavioral pooling

```bash
python I:\FLXX1\Dimension\run_story_behavior_suite.py
```

### 8.4 Whole-brain searchlight

```bash
python I:\FLXX1\Dimension\run_story_searchlight_suite.py
```

### 8.5 Seed connectivity searchlight

```bash
python I:\FLXX1\Dimension\run_story_connectivity_suite.py
```


## 9. Outputs

Each entry script writes to its own subfolder under:

- `I:\FLXX1\Dimension\python_results`

Common output files:
- `summary.json`
- `subject_results.csv` or `subject_outputs.csv`
- `selection_details.csv` when subject inclusion / exclusion matters

Searchlight suites additionally write:
- subject-level NIfTI maps
- group-level paired `t`, `p`, `R^2`, mean-difference maps


## 10. Current Boundaries

This conversion is designed to preserve the main MATLAB logic while making the workflow easier to maintain.

What is preserved:
- main paths
- major masks
- main dimensionality definitions
- same-trial resampling logic
- demean logic
- voxel-filter logic
- searchlight / group-map workflow structure

What is intentionally cleaned up:
- many scratch scripts are merged into a smaller number of clearer Python entry points
- outputs are written as CSV / JSON instead of scattered `.mat` snapshots
- subject inclusion is logged explicitly

What still needs empirical verification:
- the Python code has been syntax-checked, but not every heavy analysis has been run end-to-end in this environment
- searchlight and connectivity steps are computationally expensive and should be validated on one subject first before a full batch run
