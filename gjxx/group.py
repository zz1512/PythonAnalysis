from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from .utils import ensure_dir, rank_transform, save_json, write_table, zscore


def get_subject_mean_scores(
    events_root: str | Path,
    output_path: str | Path,
    *,
    score_column: str,
    trial_type_column: str = "trial_type",
    include_trial_type: str = "e",
) -> pd.DataFrame:
    events_root = Path(events_root)
    rows = []
    for subject_dir in sorted([path for path in events_root.iterdir() if path.is_dir() and path.name.startswith("sub")]):
        run_files = sorted(subject_dir.glob("*run-*_events.tsv"))
        if not run_files:
            run_files = sorted(subject_dir.glob("*run-*_events.txt"))
        if not run_files:
            continue
        frame = pd.concat([pd.read_csv(path, sep="	") for path in run_files], ignore_index=True)
        values = frame.loc[frame[trial_type_column].astype(str) == include_trial_type, score_column].to_numpy(dtype=float)
        rows.append({"subject": subject_dir.name, "score": float(np.mean(values))})
    result = pd.DataFrame(rows)
    write_table(result, output_path)
    return result


def _load_group_maps(map_root: str | Path, image_name: str) -> tuple[list[str], np.ndarray, nib.Nifti1Image]:
    map_root = Path(map_root)
    subjects: list[str] = []
    images: list[np.ndarray] = []
    template: nib.Nifti1Image | None = None
    for subject_dir in sorted([path for path in map_root.iterdir() if path.is_dir()]):
        path = subject_dir / image_name
        if not path.exists():
            continue
        img = nib.load(str(path))
        if template is None:
            template = img
        subjects.append(subject_dir.name)
        images.append(np.asarray(img.get_fdata(), dtype=float))
    if template is None or not images:
        raise FileNotFoundError(f"No images named {image_name} found in {map_root}")
    stacked = np.stack(images, axis=0)
    return subjects, stacked, template


def _simple_voxelwise_regression(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean(axis=0, keepdims=True)
    ssx = np.sum(x_centered ** 2)
    slope = np.sum(x_centered[:, None] * y_centered, axis=0) / ssx
    intercept = y.mean(axis=0) - slope * x.mean()
    residuals = y - (intercept[None, :] + slope[None, :] * x[:, None])
    dof = max(y.shape[0] - 2, 1)
    sigma2 = np.sum(residuals ** 2, axis=0) / dof
    stderr = np.sqrt(sigma2 / ssx)
    t_values = np.divide(slope, stderr, out=np.zeros_like(slope), where=stderr > 0)
    p_values = stats.t.sf(np.abs(t_values), df=dof) * 2
    return t_values, p_values


def _threshold_by_cluster(masked_values: np.ndarray, brain_mask: np.ndarray, min_cluster_size: int) -> np.ndarray:
    binary = np.zeros(brain_mask.shape, dtype=bool)
    binary[brain_mask] = masked_values.astype(bool)
    labels, n_labels = ndimage.label(binary)
    keep = np.zeros(binary.shape, dtype=bool)
    for label in range(1, n_labels + 1):
        current = labels == label
        if int(current.sum()) >= min_cluster_size:
            keep |= current
    return keep[brain_mask]


def run_group_regression(
    map_root: str | Path,
    score_table: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    image_name: str,
    score_column: str = "score",
    alpha: float = 0.05,
    min_cluster_size: int = 5,
    rank_inputs: bool = True,
) -> None:
    output_dir = ensure_dir(output_dir)
    score_frame = pd.read_csv(score_table, sep="	")
    subjects, maps, template = _load_group_maps(map_root, image_name)
    score_frame = score_frame.copy()
    score_frame["subject_key"] = score_frame["subject"].str.replace("-", "", regex=False)
    subject_keys = [subject.replace("-", "") for subject in subjects]
    lookup = score_frame.set_index("subject_key")[score_column]
    x = np.asarray([lookup[key] for key in subject_keys], dtype=float)
    if rank_inputs:
        x = zscore(rank_transform(x))

    brain_mask = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
    y = maps[:, brain_mask]
    if rank_inputs:
        y = rank_transform(y, axis=0)
    t_values, p_values = _simple_voxelwise_regression(y, x)
    passed = p_values < alpha
    if min_cluster_size > 1:
        passed &= _threshold_by_cluster(passed, brain_mask, min_cluster_size)

    t_map = np.zeros(brain_mask.shape, dtype=float)
    p_map = np.ones(brain_mask.shape, dtype=float)
    thresholded = np.zeros(brain_mask.shape, dtype=float)
    t_map[brain_mask] = t_values
    p_map[brain_mask] = p_values
    thresholded[brain_mask] = np.where(passed, t_values, 0.0)

    nib.Nifti1Image(t_map, affine=template.affine, header=template.header).to_filename(
        str(output_dir / "t_map.nii.gz")
    )
    nib.Nifti1Image(p_map, affine=template.affine, header=template.header).to_filename(
        str(output_dir / "p_map.nii.gz")
    )
    nib.Nifti1Image(thresholded, affine=template.affine, header=template.header).to_filename(
        str(output_dir / "thresholded_t_map.nii.gz")
    )

    summary = {
        "n_subjects": len(subjects),
        "alpha": alpha,
        "min_cluster_size": min_cluster_size,
        "rank_inputs": rank_inputs,
        "num_significant_voxels": int(np.sum(passed)),
    }
    save_json(summary, output_dir / "group_regression_summary.json")
