#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-7 retrieval-stage LSS, without touching existing run1-6 indexes.

This script intentionally writes only run-7 products:
- lss_betas_final/sub-xx/run-7/beta_trial-*.nii.gz
- lss_betas_final/sub-xx/run-7/trial_info.csv
- lss_betas_final/lss_metadata_index_run7.csv
- lss_betas_final/processing_results_run7.csv

It does not rewrite lss_metadata_index_final.csv or
lss_metadata_index_learning.csv.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings(
    "ignore",
    message="Mean values of 0 observed.*",
    category=UserWarning,
)

FINAL_ROOT = Path(__file__).resolve().parents[1]
GLM_DIR = FINAL_ROOT / "glm_analysis"
if str(GLM_DIR) not in sys.path:
    sys.path.append(str(GLM_DIR))

try:
    from glm_config import config
    import glm_utils
except ImportError as exc:  # pragma: no cover
    print(f"Unable to import glm_config/glm_utils: {exc}", flush=True)
    sys.exit(1)


RUN = 7
STAGE = "Retrieval"
LSS_TR = 2.0
LSS_SMOOTHING = None
DEFAULT_OUTPUT_ROOT = getattr(config, "LSS_OUTPUT_ROOT", config.BASE_DIR / "lss_betas_final")


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


def parse_subjects(values: Iterable[str] | None) -> list[str]:
    if values:
        subjects: list[str] = []
        for value in values:
            for item in str(value).replace(",", " ").split():
                item = item.strip()
                if not item:
                    continue
                if item.isdigit():
                    item = f"sub-{int(item):02d}"
                subjects.append(item)
        return sorted(dict.fromkeys(subjects))
    return list(config.SUBJECTS)


def resolve_fmri_path(sub: str, run: int = RUN) -> Path | None:
    fmri_path = Path(str(config.FMRI_TPL).format(sub=sub, run=run))
    if fmri_path.exists():
        return fmri_path
    gz_path = fmri_path.with_suffix(".nii.gz")
    if gz_path.exists():
        return gz_path
    return None


def resolve_paths(sub: str) -> dict[str, Path | None]:
    return {
        "event": Path(str(config.EVENT_TPL).format(sub=sub, run=RUN)),
        "fmri": resolve_fmri_path(sub, RUN),
        "confounds": Path(str(config.CONFOUNDS_TPL).format(sub=sub, run=RUN)),
    }


def inspect_inputs(subjects: list[str], output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sub in subjects:
        paths = resolve_paths(sub)
        out_dir = output_root / sub / f"run-{RUN}"
        n_existing = len(list(out_dir.glob("beta_trial-*.nii.gz"))) if out_dir.exists() else 0
        n_events = 0
        event_path = paths["event"]
        if event_path is not None and event_path.exists():
            try:
                n_events = int(len(pd.read_csv(event_path, sep="\t")))
            except Exception:
                n_events = -1
        rows.append(
            {
                "subject": sub,
                "run": RUN,
                "event_exists": bool(event_path is not None and event_path.exists()),
                "fmri_exists": bool(paths["fmri"] is not None and paths["fmri"].exists()),
                "confounds_exists": bool(paths["confounds"] is not None and paths["confounds"].exists()),
                "n_event_rows": n_events,
                "n_existing_betas": n_existing,
                "ready": bool(
                    event_path is not None
                    and event_path.exists()
                    and paths["fmri"] is not None
                    and paths["fmri"].exists()
                    and paths["confounds"] is not None
                    and paths["confounds"].exists()
                ),
            }
        )
    return pd.DataFrame(rows)


def beta_file_is_readable(beta_path: Path) -> bool:
    try:
        beta_img = load_img(str(beta_path))
        beta_img.get_fdata(dtype=np.float32, caching="unchanged")
        return True
    except Exception as exc:
        log(f"[WARN] Corrupt beta detected, recomputing: {beta_path.name} ({exc})")
        try:
            beta_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def load_and_prep_events(sub: str) -> pd.DataFrame | None:
    events_path = Path(str(config.EVENT_TPL).format(sub=sub, run=RUN))
    if not events_path.exists():
        return None
    frame = pd.read_csv(events_path, sep="\t")
    required = ["onset", "duration", "trial_type"]
    if not all(col in frame.columns for col in required):
        return None
    keep_cols = [
        col
        for col in [
            "onset",
            "duration",
            "trial_type",
            "pic_num",
            "memory",
            "action",
            "action_time",
            "pic_out",
        ]
        if col in frame.columns
    ]
    frame = frame[keep_cols].copy()
    frame = frame.dropna(subset=["onset", "duration", "trial_type"])
    frame["onset"] = pd.to_numeric(frame["onset"], errors="coerce")
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
    frame = frame.dropna(subset=["onset", "duration"]).reset_index(drop=True)
    return frame if not frame.empty else None


def create_lss_events(trial_idx: int, base_events: pd.DataFrame) -> pd.DataFrame:
    events_lss = base_events[["onset", "duration", "trial_type"]].copy()
    events_lss.at[trial_idx, "trial_type"] = "LSS_TARGET"
    return events_lss


def safe_int(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


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
        skip_existing,
    ) = args

    model = None
    beta_map = None
    try:
        beta_filename = out_dir / f"beta_trial-{trial_idx:03d}_{unique_label}.nii.gz"
        row = events_df.iloc[trial_idx]
        if skip_existing and beta_filename.exists() and beta_file_is_readable(beta_filename):
            return {
                "trial_index": trial_idx,
                "condition": original_condition,
                "original_condition": original_condition,
                "unique_label": unique_label,
                "pic_num": safe_int(row.get("pic_num", np.nan)),
                "beta_file": beta_filename.name,
                "onset": row["onset"],
                "duration": row["duration"],
                "memory": safe_int(row.get("memory", np.nan)),
                "action": safe_int(row.get("action", np.nan)),
                "action_time": row.get("action_time", np.nan),
                "pic_out": row.get("pic_out", ""),
                "was_existing_beta": True,
            }

        lss_events = create_lss_events(trial_idx, events_df)
        model = FirstLevelModel(mask_img=run_mask, **base_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean values of 0 observed.*")
            model.fit(fmri_img, events=lss_events, confounds=confounds, sample_masks=sample_mask)
        design_matrix = model.design_matrices_[0]
        if "LSS_TARGET" not in design_matrix.columns:
            return None
        contrast_vec = np.zeros(len(design_matrix.columns), dtype=float)
        contrast_vec[list(design_matrix.columns).index("LSS_TARGET")] = 1.0
        beta_map = model.compute_contrast(contrast_vec, output_type="effect_size")
        beta_map.to_filename(beta_filename)
        return {
            "trial_index": trial_idx,
            "condition": original_condition,
            "original_condition": original_condition,
            "unique_label": unique_label,
            "pic_num": safe_int(row.get("pic_num", np.nan)),
            "beta_file": beta_filename.name,
            "onset": row["onset"],
            "duration": row["duration"],
            "memory": safe_int(row.get("memory", np.nan)),
            "action": safe_int(row.get("action", np.nan)),
            "action_time": row.get("action_time", np.nan),
            "pic_out": row.get("pic_out", ""),
            "was_existing_beta": False,
        }
    except Exception as exc:
        log(f"[WARN] Trial failed: {unique_label} ({exc})")
        return None
    finally:
        if beta_map is not None:
            del beta_map
        if model is not None:
            del model
        gc.collect()


def process_one_subject(
    sub: str,
    *,
    output_root: Path,
    batch_size: int,
    skip_existing: bool,
) -> tuple[str, list[dict]]:
    paths = resolve_paths(sub)
    if paths["fmri"] is None:
        return f"{sub}|run-{RUN}: skipped - missing fmri", []
    if paths["confounds"] is None or not paths["confounds"].exists():
        return f"{sub}|run-{RUN}: skipped - missing confounds", []

    events = load_and_prep_events(sub)
    if events is None:
        return f"{sub}|run-{RUN}: skipped - missing/invalid events", []

    confounds, sample_mask = glm_utils.robust_load_confounds(
        str(paths["confounds"]),
        config.DENOISE_STRATEGY,
    )

    out_dir = output_root / sub / f"run-{RUN}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "mask.nii.gz"

    log(f"[{sub}|run-{RUN}] Loading fMRI once...")
    fmri_img = load_img(str(paths["fmri"]))
    if mask_path.exists():
        log(f"[{sub}|run-{RUN}] Reusing saved mask")
        run_mask = load_img(str(mask_path))
    else:
        log(f"[{sub}|run-{RUN}] Computing mask")
        run_mask = compute_brain_mask(fmri_img)
        run_mask.to_filename(mask_path)

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

    results: list[dict] = []
    total_trials = len(events)
    log(f"[{sub}|run-{RUN}] Start run7 LSS: {total_trials} trials")
    for batch_start in range(0, total_trials, batch_size):
        batch_end = min(batch_start + batch_size, total_trials)
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
                    skip_existing,
                )
            )
            if result is not None:
                result.update({"subject": sub, "run": RUN, "stage": STAGE})
                results.append(result)
        gc.collect()
        log(f"[{sub}|run-{RUN}] Batch done: {len(results)}/{total_trials}")

    if results:
        pd.DataFrame(results).to_csv(out_dir / "trial_info.csv", index=False, encoding="utf-8-sig")
    del fmri_img
    del run_mask
    gc.collect()
    return f"{sub}|run-{RUN}: success {len(results)}/{total_trials}", results


def process_subject_task(args: tuple) -> tuple[str, list[dict]]:
    sub, output_root, batch_size, skip_existing = args
    try:
        return process_one_subject(
            sub,
            output_root=output_root,
            batch_size=batch_size,
            skip_existing=skip_existing,
        )
    except Exception as exc:
        cleanup_all_temp_caches()
        gc.collect()
        return f"{sub}|run-{RUN}: failed - {exc}", []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run run-7 retrieval LSS without rewriting run1-6 metadata.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject ids, e.g. sub-01 sub-02 or 1 2.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("LSS_N_JOBS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("LSS_BATCH_SIZE", "20")))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only inspect inputs and write no LSS outputs.")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = inspect_inputs(subjects, output_root)
    log("Run7 input inventory:")
    log(inventory.to_string(index=False))
    if args.dry_run:
        return

    ready_subjects = inventory.loc[inventory["ready"], "subject"].astype(str).tolist()
    if not ready_subjects:
        raise SystemExit("No run7 subjects have complete inputs.")

    log(
        f"Start run7 LSS: {len(ready_subjects)}/{len(subjects)} subjects ready, "
        f"n_jobs={args.n_jobs}, batch_size={args.batch_size}, skip_existing={args.skip_existing}"
    )
    tasks = [(sub, output_root, args.batch_size, args.skip_existing) for sub in ready_subjects]
    if args.n_jobs == 1:
        results = [process_subject_task(task) for task in tasks]
    else:
        results = Parallel(n_jobs=args.n_jobs)(delayed(process_subject_task)(task) for task in tasks)

    status_rows = [{"result": status} for status, _ in results]
    status_path = output_root / "processing_results_run7.csv"
    status_frame = pd.DataFrame(status_rows)
    if status_path.exists():
        try:
            previous_status = pd.read_csv(status_path, encoding="utf-8-sig")
            status_frame = pd.concat([previous_status, status_frame], ignore_index=True)
        except Exception:
            pass
    status_frame.to_csv(status_path, index=False, encoding="utf-8-sig")

    all_meta: list[dict] = []
    for _, rows in results:
        all_meta.extend(rows)
    if all_meta:
        meta = pd.DataFrame(all_meta)
        meta = meta.sort_values(["subject", "run", "trial_index"]).reset_index(drop=True)
        metadata_path = output_root / "lss_metadata_index_run7.csv"
        if metadata_path.exists():
            try:
                previous = pd.read_csv(metadata_path, encoding="utf-8-sig")
                current_subjects = set(meta["subject"].astype(str))
                previous = previous[~previous["subject"].astype(str).isin(current_subjects)].copy()
                meta = pd.concat([previous, meta], ignore_index=True, sort=False)
                meta = meta.sort_values(["subject", "run", "trial_index"]).reset_index(drop=True)
            except Exception as exc:
                log(f"[WARN] Could not merge existing run7 metadata, writing current batch only: {exc}")
        meta.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        log(f"Wrote run7 metadata index: {metadata_path} ({len(meta)} rows)")
    else:
        log("[WARN] No run7 metadata rows produced.")
    cleanup_all_temp_caches()
    log(f"Run7 LSS complete: {sum('success' in row['result'] for row in status_rows)}/{len(tasks)} ready tasks succeeded")


if __name__ == "__main__":
    main()
