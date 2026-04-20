#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning-stage LSS for run-3/run-4.

This version keeps the learning-stage-specific LSS event coding from the
 original script, but aligns the FirstLevelModel parameters and temp-cache
 cleanup behavior with glm_analysis/run_lss.py:

- same denoising path via glm_utils.robust_load_confounds(...)
- same sample_mask usage
- same run-level brain mask computation
- same GLM defaults: no smoothing, spm + derivative, 1/128 high-pass
- same lightweight temp-cache cleanup pattern
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

FINAL_ROOT = Path(__file__).resolve().parents[1]
GLM_DIR = FINAL_ROOT / "glm_analysis"
if str(GLM_DIR) not in sys.path:
    sys.path.append(str(GLM_DIR))

try:
    from glm_config import config
    import glm_utils
except ImportError as exc:
    print(f"Unable to import glm_config/glm_utils: {exc}")
    sys.exit(1)

SUBJECTS = list(config.SUBJECTS)
RUNS = [3, 4]
N_JOBS_PARALLEL = max(1, int(os.environ.get("LSS_N_JOBS", "1")))
BATCH_SIZE = max(1, int(os.environ.get("LSS_BATCH_SIZE", "20")))
SKIP_EXISTING_BETAS = os.environ.get("LSS_SKIP_EXISTING", "1") != "0"
LSS_TR = 2.0
LSS_SMOOTHING = None
OUTPUT_ROOT = getattr(config, "LSS_OUTPUT_ROOT", config.BASE_DIR / "lss_betas_final")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    stamp = pd.Timestamp.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def cleanup_temp_cache(path: Path) -> None:
    try:
        if path.exists() and "lss_cache_" in path.name:
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def cleanup_all_temp_caches() -> None:
    try:
        for path in Path(tempfile.gettempdir()).glob("lss_cache_*"):
            cleanup_temp_cache(path)
    except Exception:
        pass


def resolve_fmri_path(sub: str, run: int) -> Path | None:
    fmri_path = Path(str(config.FMRI_TPL).format(sub=sub, run=run))
    if fmri_path.exists():
        return fmri_path
    gz_path = fmri_path.with_suffix(".nii.gz")
    if gz_path.exists():
        return gz_path
    return None


def beta_file_is_readable(beta_path: Path) -> bool:
    try:
        beta_img = load_img(str(beta_path))
        # Force data access so truncated/corrupt NIfTI files fail fast.
        beta_img.get_fdata(dtype=np.float32, caching="unchanged")
        return True
    except Exception as exc:
        log(f"[WARN] Corrupt beta detected, recomputing: {beta_path.name} ({exc})")
        try:
            beta_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def load_and_prep_events(sub: str, run: int) -> pd.DataFrame | None:
    events_path = Path(str(config.EVENT_TPL).format(sub=sub, run=run))
    if not events_path.exists():
        return None

    frame = pd.read_csv(events_path, sep="\t")
    required = ["onset", "duration", "trial_type"]
    if not all(col in frame.columns for col in required):
        return None

    keep_cols = ["onset", "duration", "trial_type"]
    if "pic_num" in frame.columns:
        keep_cols.append("pic_num")

    frame = frame[keep_cols].copy()
    frame = frame.dropna(subset=["onset", "duration", "trial_type"])
    frame["onset"] = pd.to_numeric(frame["onset"], errors="coerce")
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
    frame = frame.dropna(subset=["onset", "duration"]).reset_index(drop=True)
    return frame if not frame.empty else None


def create_lss_events(trial_idx: int, base_events: pd.DataFrame) -> pd.DataFrame:
    events_lss = base_events[["onset", "duration", "trial_type"]].copy()
    # Keep learning-stage LSS consistent with glm_analysis/run_lss.py:
    # only rename the current target trial, leaving the other events unchanged.
    events_lss.at[trial_idx, "trial_type"] = "LSS_TARGET"
    return events_lss


def process_single_trial(args: tuple) -> dict | None:
    (
        trial_idx,
        fmri_img,
        events_df,
        confounds,
        sample_mask,
        base_kwargs,
        out_dir,
        unique_label,
        original_condition,
        run_mask,
    ) = args

    model = None
    beta_map = None

    try:
        beta_filename = out_dir / f"beta_trial-{trial_idx:03d}_{unique_label}.nii.gz"
        if (
            SKIP_EXISTING_BETAS
            and beta_filename.exists()
            and beta_file_is_readable(beta_filename)
        ):
            pic_num = events_df.iloc[trial_idx].get("pic_num", np.nan)
            return {
                "trial_index": trial_idx,
                "condition": original_condition,
                "original_condition": original_condition,
                "unique_label": unique_label,
                "pic_num": int(pic_num) if pd.notna(pic_num) else None,
                "beta_file": beta_filename.name,
                "onset": events_df.iloc[trial_idx]["onset"],
                "duration": events_df.iloc[trial_idx]["duration"],
            }

        lss_events = create_lss_events(trial_idx, events_df)

        model = FirstLevelModel(mask_img=run_mask, **base_kwargs)
        model.fit(
            fmri_img,
            events=lss_events,
            confounds=confounds,
            sample_masks=sample_mask,
        )

        design_matrix = model.design_matrices_[0]
        if "LSS_TARGET" not in design_matrix.columns:
            return None

        contrast_vec = np.zeros(len(design_matrix.columns), dtype=float)
        contrast_vec[list(design_matrix.columns).index("LSS_TARGET")] = 1.0
        beta_map = model.compute_contrast(contrast_vec, output_type="effect_size")

        beta_map.to_filename(beta_filename)

        pic_num = events_df.iloc[trial_idx].get("pic_num", np.nan)
        return {
            "trial_index": trial_idx,
            "condition": original_condition,
            "original_condition": original_condition,
            "unique_label": unique_label,
            "pic_num": int(pic_num) if pd.notna(pic_num) else None,
            "beta_file": beta_filename.name,
            "onset": events_df.iloc[trial_idx]["onset"],
            "duration": events_df.iloc[trial_idx]["duration"],
        }
    except Exception:
        return None
    finally:
        if beta_map is not None:
            del beta_map
        if model is not None:
            del model
        gc.collect()


def process_one_run(sub: str, run: int) -> list[dict]:
    fmri_path = resolve_fmri_path(sub, run)
    if fmri_path is None:
        log(f"[SKIP] Missing fMRI file: {sub}|run-{run}")
        return []

    events = load_and_prep_events(sub, run)
    if events is None:
        log(f"[SKIP] Missing/invalid events: {sub}|run-{run}")
        return []

    confounds, sample_mask = glm_utils.robust_load_confounds(
        str(config.CONFOUNDS_TPL).format(sub=sub, run=run),
        config.DENOISE_STRATEGY,
    )

    out_dir = OUTPUT_ROOT / sub / f"run-{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "mask.nii.gz"

    log(f"[{sub}|run-{run}] Loading fMRI image once for low-I/O processing...")
    fmri_img = load_img(str(fmri_path))

    if mask_path.exists():
        log(f"[{sub}|run-{run}] Reusing saved mask...")
        run_mask = load_img(str(mask_path))
    else:
        log(f"[{sub}|run-{run}] Computing mask...")
        run_mask = compute_brain_mask(fmri_img)
        try:
            run_mask.to_filename(mask_path)
        except Exception as exc:
            log(f"[WARN] Failed to save mask for {sub}|run-{run}: {exc}")

    base_kwargs = dict(
        t_r=LSS_TR,
        slice_time_ref=0.5,
        smoothing_fwhm=LSS_SMOOTHING,
        noise_model="ar1",
        standardize=True,
        hrf_model="spm + derivative",
        drift_model="cosine",
        high_pass=1 / 128.0,
        minimize_memory=True,
        verbose=0,
    )

    total_trials = len(events)
    log(f"[{sub}|run-{run}] Start LSS: {total_trials} trials")
    results_meta: list[dict] = []

    for batch_start in range(0, total_trials, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_trials)
        for trial_idx in range(batch_start, batch_end):
            row = events.iloc[trial_idx]
            pic_num = row.get("pic_num", np.nan)
            unique_label = (
                f"{row['trial_type']}_{int(pic_num)}"
                if pd.notna(pic_num)
                else f"{row['trial_type']}_{trial_idx}"
            )

            result = process_single_trial(
                (
                    trial_idx,
                    fmri_img,
                    events,
                    confounds,
                    sample_mask,
                    base_kwargs,
                    out_dir,
                    unique_label,
                    row["trial_type"],
                    run_mask,
                )
            )
            if result is not None:
                result.update({"subject": sub, "run": run, "stage": "Learning"})
                results_meta.append(result)

        gc.collect()
        log(f"[{sub}|run-{run}] Batch done: {len(results_meta)}/{total_trials}")

    if results_meta:
        pd.DataFrame(results_meta).to_csv(
            out_dir / "trial_info.csv",
            index=False,
            encoding="utf-8-sig",
        )
    del fmri_img
    del run_mask
    gc.collect()
    log(f"[{sub}|run-{run}] LSS complete: {len(results_meta)}/{total_trials}")
    return results_meta


def process_subject_run(task: tuple[str, int]) -> str:
    sub, run = task
    try:
        process_one_run(sub, run)
        return f"{sub}|run-{run}: success"
    except Exception as exc:
        cleanup_all_temp_caches()
        gc.collect()
        return f"{sub}|run-{run}: failed - {exc}"


def main() -> None:
    log("Start learning-stage LSS pipeline...")
    try:
        tasks = [(sub, run) for sub in SUBJECTS for run in RUNS]
        log(f"Total tasks: {len(tasks)}")
        log(
            "Low-I/O mode enabled: "
            f"n_jobs={N_JOBS_PARALLEL}, batch_size={BATCH_SIZE}, "
            f"skip_existing_betas={SKIP_EXISTING_BETAS}"
        )
        if N_JOBS_PARALLEL == 1:
            results = [process_subject_run(task) for task in tasks]
        else:
            results = Parallel(n_jobs=N_JOBS_PARALLEL)(
                delayed(process_subject_run)(task) for task in tasks
            )

        pd.DataFrame({"result": results}).to_csv(
            OUTPUT_ROOT / "processing_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        success_count = sum("success" in item for item in results)
        log(f"LSS done: {success_count}/{len(tasks)} tasks succeeded")
    finally:
        cleanup_all_temp_caches()
        gc.collect()


if __name__ == "__main__":
    main()
