from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .utils import ensure_dir, fisher_z_from_samples, mean_row_similarity, paired_t_summary, save_json, write_table


def _masked_samples(four_d_path: str | Path, mask_path: str | Path) -> np.ndarray:
    img = nib.load(str(four_d_path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image: {four_d_path}")
    mask = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
    reshaped = data[mask, :].T
    return reshaped


def compute_gps_subject_metrics(
    four_d_path: str | Path,
    metadata_path: str | Path,
    mask_path: str | Path,
    *,
    group_a: str = "HSC",
    group_b: str = "LSC",
) -> dict[str, float]:
    samples = _masked_samples(four_d_path, mask_path)
    metadata = pd.read_csv(metadata_path, sep="	")
    similarities = mean_row_similarity(samples)
    metadata = metadata.copy()
    metadata["mean_similarity"] = similarities
    a_values = metadata.loc[metadata["analysis_group"] == group_a, "mean_similarity"].to_numpy(dtype=float)
    b_values = metadata.loc[metadata["analysis_group"] == group_b, "mean_similarity"].to_numpy(dtype=float)
    return {
        "group_a_mean": float(np.mean(a_values)),
        "group_b_mean": float(np.mean(b_values)),
        "group_a_n": float(a_values.size),
        "group_b_n": float(b_values.size),
    }


def run_group_gps(
    pattern_root: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    pattern_name: str = "glm_T_gps.nii.gz",
    metadata_name: str = "glm_T_gps_metadata.tsv",
    group_a: str = "HSC",
    group_b: str = "LSC",
) -> pd.DataFrame:
    pattern_root = Path(pattern_root)
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, float | str]] = []
    for subject_dir in sorted([path for path in pattern_root.iterdir() if path.is_dir()]):
        pattern_path = subject_dir / pattern_name
        metadata_path = subject_dir / metadata_name
        if not pattern_path.exists() or not metadata_path.exists():
            continue
        metrics = compute_gps_subject_metrics(pattern_path, metadata_path, mask_path, group_a=group_a, group_b=group_b)
        rows.append({"subject": subject_dir.name, **metrics})

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "gps_subject_metrics.tsv")
    summary = paired_t_summary(frame["group_a_mean"], frame["group_b_mean"])
    save_json(summary, output_dir / "gps_group_summary.json")
    return frame


def _vectorized_dsm(samples: np.ndarray) -> np.ndarray:
    fisher = fisher_z_from_samples(samples)
    tri = np.triu_indices_from(fisher, k=1)
    return fisher[tri]


def compute_subject_roi_dsm_correlation(
    hsc_path: str | Path,
    lsc_path: str | Path,
    mask_a: str | Path,
    mask_b: str | Path,
) -> dict[str, float]:
    hsc_a = _masked_samples(hsc_path, mask_a)
    hsc_b = _masked_samples(hsc_path, mask_b)
    lsc_a = _masked_samples(lsc_path, mask_a)
    lsc_b = _masked_samples(lsc_path, mask_b)
    hsc_corr = np.corrcoef(_vectorized_dsm(hsc_a), _vectorized_dsm(hsc_b))[0, 1]
    lsc_corr = np.corrcoef(_vectorized_dsm(lsc_a), _vectorized_dsm(lsc_b))[0, 1]
    return {"hsc_roi_corr": float(hsc_corr), "lsc_roi_corr": float(lsc_corr)}


def run_group_roi_dsm_correlation(
    pattern_root: str | Path,
    mask_a: str | Path,
    mask_b: str | Path,
    output_dir: str | Path,
    *,
    hsc_name: str = "HSC.nii.gz",
    lsc_name: str = "LSC.nii.gz",
) -> pd.DataFrame:
    pattern_root = Path(pattern_root)
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, float | str]] = []
    for subject_dir in sorted([path for path in pattern_root.iterdir() if path.is_dir()]):
        hsc_path = subject_dir / hsc_name
        lsc_path = subject_dir / lsc_name
        if not hsc_path.exists() or not lsc_path.exists():
            continue
        metrics = compute_subject_roi_dsm_correlation(hsc_path, lsc_path, mask_a, mask_b)
        rows.append({"subject": subject_dir.name, **metrics})

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "roi_dsm_subject_metrics.tsv")
    summary = paired_t_summary(frame["hsc_roi_corr"], frame["lsc_roi_corr"])
    save_json(summary, output_dir / "roi_dsm_group_summary.json")
    return frame
