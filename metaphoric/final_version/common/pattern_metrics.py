from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from common.final_utils import components_to_variance_threshold, ensure_dir


def load_mask(mask_path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.get_fdata()) > 0
    return mask_img, mask


def load_4d_data(image_path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(image_path))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D NIfTI image: {image_path}")
    return img, data


def load_masked_samples(image_path: str | Path, mask_path: str | Path) -> np.ndarray:
    _, data = load_4d_data(image_path)
    _, mask = load_mask(mask_path)
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")
    return data[mask, :].T.astype(np.float64, copy=False)


def save_scalar_map(reference_img: nib.Nifti1Image, mask: np.ndarray, values: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    full = np.zeros(mask.shape, dtype=np.float32)
    full[mask] = values.astype(np.float32)
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(full, reference_img.affine, header), str(output_path))


def concat_images(image_paths: Sequence[str | Path], output_path: str | Path) -> None:
    images = [nib.load(str(path)) for path in image_paths]
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    nib.concat_images(images, axis=3).to_filename(output_path)


def correlation_distance_rdm(samples: np.ndarray) -> np.ndarray:
    centered = samples - samples.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 0
    if valid.sum() < 2:
        raise ValueError("Need at least two non-constant trials to compute RDM.")
    centered = centered[valid]
    norms = norms[valid]
    corr = centered @ centered.T
    corr /= np.outer(norms, norms)
    corr = np.clip(corr, -1.0, 1.0)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def pca_explained_variance(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        raise ValueError("Centered matrix has zero variance.")
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    denom = max(centered.shape[0] - 1, 1)
    variances = (singular_values ** 2) / denom
    total = variances.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Total PCA variance is zero.")
    return variances / total * 100.0


def dimensionality_from_samples(samples: np.ndarray, explained_threshold: float = 80.0) -> float:
    if samples.shape[0] < 2:
        raise ValueError("Need at least two trials.")
    rdm = correlation_distance_rdm(samples)
    explained = pca_explained_variance(rdm)
    cumulative = np.cumsum(explained)
    return float(np.searchsorted(cumulative, explained_threshold, side="left") + 1)


def rd_from_covariance(samples: np.ndarray, explained_threshold: float = 80.0) -> float:
    cov = np.cov(samples, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    return float(components_to_variance_threshold(eigenvalues, explained_threshold))


def gps_from_samples(samples: np.ndarray) -> float:
    if samples.shape[0] < 2:
        return float("nan")
    corr = np.corrcoef(samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = np.arctanh(corr)
    fisher[~np.isfinite(fisher)] = np.nan
    np.fill_diagonal(fisher, np.nan)
    return float(np.nanmean(fisher))


def neural_rdm_vector(samples: np.ndarray, metric: str = "correlation") -> np.ndarray:
    if metric == "correlation":
        return pdist(samples, metric="correlation")
    return pdist(samples, metric=metric)


def dsm_correlation(samples_a: np.ndarray, samples_b: np.ndarray, metric: str = "correlation") -> float:
    vec_a = neural_rdm_vector(samples_a, metric=metric)
    vec_b = neural_rdm_vector(samples_b, metric=metric)
    if vec_a.size == 0 or vec_b.size == 0:
        return float("nan")
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def build_fixed_count_neighborhood(mask: np.ndarray, voxel_count: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] == 0:
        raise ValueError("Mask does not contain voxels.")
    tree = cKDTree(coordinates)
    k = min(voxel_count, coordinates.shape[0])
    _, indices = tree.query(coordinates, k=k)
    if k == 1:
        indices = indices[:, np.newaxis]
    return coordinates, np.asarray(indices, dtype=np.int32)


def compute_searchlight_dimension_map(image_path: str | Path, mask_path: str | Path, explained_threshold: float, voxel_count: int) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    img, data = load_4d_data(image_path)
    _, mask = load_mask(mask_path)
    coords, neighborhoods = build_fixed_count_neighborhood(mask, voxel_count)
    samples = data[mask, :].T
    values = np.full(coords.shape[0], np.nan, dtype=np.float32)
    for center_index, neighbor_indices in enumerate(neighborhoods):
        local_samples = samples[:, neighbor_indices]
        try:
            values[center_index] = dimensionality_from_samples(local_samples, explained_threshold)
        except ValueError:
            values[center_index] = np.nan
    return img, mask, values


def _corr_similarity_matrix(samples: np.ndarray) -> np.ndarray:
    return 1.0 - correlation_distance_rdm(samples)


def compute_seed_connectivity_map(image_path: str | Path, mask_path: str | Path, seed_samples: np.ndarray, explained_threshold: float, voxel_count: int) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    seed_similarity = _corr_similarity_matrix(seed_samples)
    seed_centered = seed_similarity - seed_similarity.mean(axis=0, keepdims=True)
    _, _, seed_vh = np.linalg.svd(seed_centered, full_matrices=False)
    seed_coeff = seed_vh.T
    seed_explained = pca_explained_variance(seed_similarity)
    seed_count = int(np.searchsorted(np.cumsum(seed_explained), explained_threshold, side="left") + 1)

    img, data = load_4d_data(image_path)
    _, mask = load_mask(mask_path)
    coords, neighborhoods = build_fixed_count_neighborhood(mask, voxel_count)
    samples = data[mask, :].T
    values = np.full(coords.shape[0], np.nan, dtype=np.float32)

    for center_index, neighbor_indices in enumerate(neighborhoods):
        local_samples = samples[:, neighbor_indices]
        try:
            local_similarity = _corr_similarity_matrix(local_samples)
            local_centered = local_similarity - local_similarity.mean(axis=0, keepdims=True)
            _, _, local_vh = np.linalg.svd(local_centered, full_matrices=False)
            local_coeff = local_vh.T
            local_explained = pca_explained_variance(local_similarity)
        except ValueError:
            continue
        local_count = int(np.searchsorted(np.cumsum(local_explained), explained_threshold, side="left") + 1)
        component_count = min(seed_count, local_count)
        if component_count < 1:
            continue
        similarities = []
        for component_index in range(component_count):
            vector_a = seed_coeff[:, component_index]
            vector_b = local_coeff[:, component_index]
            denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            if denominator == 0:
                continue
            similarities.append(float(np.dot(vector_a, vector_b) / denominator))
        if similarities:
            values[center_index] = float(np.mean(similarities))
    return img, mask, values


def compute_group_paired_map_statistics(map_a_paths: Sequence[str | Path], map_b_paths: Sequence[str | Path], output_dir: str | Path, prefix: str = "paired") -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    if len(map_a_paths) != len(map_b_paths):
        raise ValueError("Map counts do not match.")
    first_img = nib.load(str(map_a_paths[0]))
    shape = first_img.shape
    a_stack = np.vstack([np.asarray(nib.load(str(path)).get_fdata()).reshape(-1) for path in map_a_paths])
    b_stack = np.vstack([np.asarray(nib.load(str(path)).get_fdata()).reshape(-1) for path in map_b_paths])
    test = stats.ttest_rel(a_stack, b_stack, axis=0, nan_policy="omit")
    t_values = np.asarray(test.statistic, dtype=float)
    p_values = np.asarray(test.pvalue, dtype=float)
    diff_values = np.nanmean(a_stack - b_stack, axis=0)

    for name, values in {
        f"{prefix}_t_map.nii.gz": t_values,
        f"{prefix}_p_map.nii.gz": p_values,
        f"{prefix}_mean_diff_map.nii.gz": diff_values,
    }.items():
        nib.save(
            nib.Nifti1Image(values.reshape(shape).astype(np.float32), first_img.affine, first_img.header),
            str(output_dir / name),
        )

    return {
        "n_subjects": int(len(map_a_paths)),
        "t_map": str(output_dir / f"{prefix}_t_map.nii.gz"),
        "p_map": str(output_dir / f"{prefix}_p_map.nii.gz"),
        "mean_diff_map": str(output_dir / f"{prefix}_mean_diff_map.nii.gz"),
    }


def square_rdm(samples: np.ndarray, metric: str = "correlation") -> np.ndarray:
    return squareform(neural_rdm_vector(samples, metric=metric))
