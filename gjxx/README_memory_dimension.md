# Memory Dimension Analysis README

## 1. Purpose

This Python codebase rebuilds the current MATLAB analysis logic in a cleaner form and splits it into two separate tasks.

Task 1:
- extend the existing remembered-only analysis to forgotten trials
- compare `HSC` vs `LSC` dimensionality within remembered trials
- compare `HSC` vs `LSC` dimensionality within forgotten trials

Task 2:
- compute memory accuracy for each subject
- split subjects into `memory-good` and `memory-poor`
- compare `HSC` vs `LSC` dimensionality within each subject group


## 2. Main Files

- [dimension_analysis_core.py](/I:/FLXX1/Dimension/dimension_analysis_core.py:1)
  - shared functions
  - contains data loading, ROI extraction, RDM computation, PCA-based dimensionality, memory accuracy calculation, subject filtering, statistics, and result writing

- [run_remembered_forgotten_dimension.py](/I:/FLXX1/Dimension/run_remembered_forgotten_dimension.py:1)
  - Task 1 entry script
  - remembered vs forgotten analysis

- [run_memory_group_dimension.py](/I:/FLXX1/Dimension/run_memory_group_dimension.py:1)
  - Task 2 entry script
  - memory-good vs memory-poor grouped analysis

- [memory_dimension_analysis.py](/I:/FLXX1/Dimension/memory_dimension_analysis.py:1)
  - lightweight wrapper
  - just reminds you which task scripts to run

- [requirements_memory_dimension.txt](/I:/FLXX1/Dimension/requirements_memory_dimension.txt:1)
  - minimal dependency list


## 3. Current Fixed Data Paths

These defaults are written as absolute paths on purpose. Even if you move the Python scripts into another project folder, they still point to the current project data unless you override them with command-line arguments.

- pattern directory:
  - `I:\FLXX1\Dimension\pattern`

- remembered/forgotten single-trial directory:
  - `I:\FLXX1\First_level\patterns_hlrf`

- event directory:
  - `I:\FLXX1\events`

- ROI mask:
  - `I:\FLXX1\mask\Hippocampus_L.nii`

- output root:
  - `I:\FLXX1\Dimension\python_results`


## 4. How Memory Results Are Read

The subject-level memory result is read exactly from:

- `I:\FLXX1\events\sub-XX\sub-XX_run-1_events.txt`
- `I:\FLXX1\events\sub-XX\sub-XX_run-2_events.txt`
- `I:\FLXX1\events\sub-XX\sub-XX_run-3_events.txt`
- `I:\FLXX1\events\sub-XX\sub-XX_run-4_events.txt`

Current logic:

1. match files using:
   - `sub-XX_run-*_events.txt`
2. only keep rows where:
   - `trial_type == "e"`
3. read the `memory` column:
   - `memory == 1` means remembered
   - `memory == -1` means forgotten
4. ignore all other rows and all other memory codes

The code also checks whether each subject has exactly 4 event files and prints a warning if not.


## 5. Dimensionality Logic

The dimensionality metric follows the main MATLAB logic as closely as possible.

For each condition:

1. extract a `trial x voxel` matrix within the ROI
2. compute the trial-by-trial correlation-distance RDM
3. run PCA on the RDM matrix
4. return the first dimension count whose cumulative explained variance exceeds the threshold

Default explained variance threshold:

- `80%`

This is meant to match the spirit of scripts such as:

- [only_consider_remembered_items.m](/I:/FLXX1/Dimension/only_consider_remembered_items.m:1)
- [hl.m](/I:/FLXX1/Dimension/hl.m:1)


## 6. Remembered / Forgotten Mapping

Task 1 uses the following file prefixes under:

- `I:\FLXX1\First_level\patterns_hlrf\sub-XX`

Remembered:
- `HSC_rspmT_*.nii`
- `LSC_rspmT_*.nii`

Forgotten:
- `HSC_fspmT_*.nii`
- `LSC_fspmT_*.nii`


## 7. Memory-Good / Memory-Poor Grouping

Task 2 computes each subject's memory accuracy from the event files.

Default rule:

- accuracy `> 0.5` -> `memory-good`
- accuracy `<= 0.5` -> `memory-poor`

Important:

- after grouping, dimensionality is still computed subject by subject
- then a paired `HSC vs LSC` comparison is performed within each group
- the code does not pool all subjects' trials into one huge matrix


## 8. Trial Count Filtering

Both Task 1 and Task 2 now use a `min_trials` filter.

Default:

- `min_trials = 8`

Meaning:

- if a subject has fewer than 8 trials in either `HSC` or `LSC` for the relevant analysis, that subject is excluded from that analysis

Important distinction:

1. `min_trials` is an analysis rule
2. there is also a mathematical lower bound:
   - at least 2 valid trials are required to build an RDM

So even if you set `min_trials=0`, the code can still skip a subject if fewer than 2 valid trials remain for dimensionality estimation.


## 9. Effect Size

The statistical summary includes:

- `n_subjects`
- `degrees_of_freedom`
- `t_stat`
- `p_value`
- `r_squared`
- `mean_hsc`
- `mean_lsc`

`r_squared` is computed from the paired t-test as:

```text
R² = t² / (t² + df)
```

This matches your current reporting preference that emphasizes `R²`.


## 10. Installation

Install dependencies first:

```bash
pip install -r I:\FLXX1\Dimension\requirements_memory_dimension.txt
```

Current dependencies:

- `numpy`
- `scipy`
- `nibabel`


## 11. How To Run

### Task 1

```bash
python run_remembered_forgotten_dimension.py ^
  --patterns-hlrf-dir I:\FLXX1\First_level\patterns_hlrf ^
  --mask-path I:\FLXX1\mask\Hippocampus_L.nii ^
  --output-dir I:\FLXX1\Dimension\python_results\remembered_forgotten_dimension ^
  --min-trials 8
```

### Task 2

```bash
python run_memory_group_dimension.py ^
  --pattern-dir I:\FLXX1\Dimension\pattern ^
  --events-dir I:\FLXX1\events ^
  --mask-path I:\FLXX1\mask\Hippocampus_L.nii ^
  --output-dir I:\FLXX1\Dimension\python_results\memory_group_dimension ^
  --min-trials 8
```

### Show available arguments

```bash
python run_remembered_forgotten_dimension.py --help
python run_memory_group_dimension.py --help
```


## 12. Outputs

### Task 1 output folder

- `I:\FLXX1\Dimension\python_results\remembered_forgotten_dimension`

Files:

- `remembered_forgotten_dimensions.csv`
  - subject-level dimensionality results for remembered and forgotten analyses

- `selection_details.csv`
  - inclusion/exclusion list for remembered and forgotten analyses
  - shows which subjects were included
  - shows which subjects were excluded due to trial count

- `summary.json`
  - task configuration
  - included subject IDs
  - excluded subject details
  - remembered `HSC vs LSC` statistics
  - forgotten `HSC vs LSC` statistics


### Task 2 output folder

- `I:\FLXX1\Dimension\python_results\memory_group_dimension`

Files:

- `subject_memory_accuracy.csv`
  - subject-level memory accuracy and memory-good / memory-poor assignment

- `memory_group_dimensions.csv`
  - subject-level dimensionality results within the two memory groups

- `selection_details.csv`
  - inclusion/exclusion list for `memory-good` and `memory-poor`
  - shows which subjects were included in each group analysis
  - shows which subjects were excluded due to trial count

- `summary.json`
  - task configuration
  - original memory-good / memory-poor subject IDs from the behavioral grouping step
  - final included subject IDs after dimensionality filtering
  - excluded subject details
  - `memory-good` group `HSC vs LSC` statistics
  - `memory-poor` group `HSC vs LSC` statistics


## 13. Logging

The scripts print:

- subject-level memory accuracy
- memory-good subject IDs
- memory-poor subject IDs
- final included subject IDs for each analysis
- subject IDs excluded because trial count is below `min_trials`
- subject IDs excluded because dimensionality estimation failed

This is meant to make subject inclusion transparent.


## 14. Current Scope

What the code currently does:

- reproduces the ROI-based dimensionality logic
- extends remembered-only analysis to forgotten trials
- supports subject grouping by memory ability
- reports paired t-tests and `R²`
- records inclusion/exclusion details

What it does not do yet:

- direct between-group comparison between memory-good and memory-poor
- searchlight analysis
- automatic Chinese report writing
- multi-ROI batch runs


## 15. Suggested Next Steps

If you continue extending this workflow, the most useful next upgrades would be:

1. add a small reporting script that converts `summary.json` into a human-readable result note
2. add multi-ROI support
3. add a command-line option to save subject-level HSC-LSC differences
4. add a compact QC report for trial counts before analysis

