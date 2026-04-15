from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    import nibabel as nib
    import numpy as np
    from scipy import stats
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise SystemExit(
        "Missing required dependency. Please install numpy, scipy, and nibabel. "
        f"Original error: {exc}"
    )

from dimension_analysis_core import (
    DEFAULT_EVENTS_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PATTERN_DIR,
    SUBJECTS,
    SubjectDimension,
    dimensionality_from_samples,
    load_mask,
    load_samples_from_nii,
    natural_sort_key,
    paired_t_test,
    pca_explained_variance,
    read_memory_accuracy,
    subject_dimension_rows,
    write_csv,
    write_json,
)


DEFAULT_HIPPOCAMPUS_LEFT_MASK = Path(r"I:\FLXX1\mask\Hippocampus_L.nii")
DEFAULT_HIPPOCAMPUS_RIGHT_MASK = Path(r"I:\FLXX1\mask\Hippocampus_R.nii")
DEFAULT_SEARCHLIGHT_MASK_ROOT = Path(r"L:\FLXX_1\mvpa\GLM_item_allexample")
DEFAULT_RSA_ALL_DIR = Path(r"L:\FLXX_1\mvpa\GLM_item_allexample\rsa")
DEFAULT_ALT_EVENTS_DIR = Path(r"L:\FLXX_1\First_level\get_onset\newdata_to_use")


def iter_thresholds(start: int, end: int) -> list[int]:
    if end < start:
        raise ValueError(f"Invalid threshold range: start={start}, end={end}")
    return list(range(start, end + 1))


def cumulative_threshold_dimension(explained: np.ndarray, threshold: float) -> float:
    cumulative = 0.0
    for index, value in enumerate(explained, start=1):
        cumulative += float(value)
        if cumulative > threshold:
            return float(index)
    return float(len(explained))


def dimensionality_from_pattern_samples(samples: np.ndarray, explained_threshold: float) -> float:
    # MATLAB scripts such as hl_patternbutnotrdm.m transpose the matrix to
    # voxel x trial before calling pca(...). We mimic that layout here.
    if samples.shape[0] < 2:
        raise ValueError("Need at least two trials to estimate dimensionality.")
    explained = pca_explained_variance(samples.T)
    return cumulative_threshold_dimension(explained, explained_threshold)


def explained_sum_for_pc_window(
    samples: np.ndarray,
    pc_start: int,
    pc_end: int,
) -> float:
    if pc_start < 1 or pc_end < pc_start:
        raise ValueError(f"Invalid PC window: {pc_start}-{pc_end}")
    explained = pca_explained_variance(samples.T)
    start = pc_start - 1
    end = min(pc_end, len(explained))
    if start >= end:
        raise ValueError("PC window exceeds available components.")
    return float(np.sum(explained[start:end]))


def demean_condition_samples(
    hsc_samples: np.ndarray,
    lsc_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([hsc_samples, lsc_samples])
    stacked = stacked - stacked.mean(axis=0, keepdims=True)
    return stacked[: hsc_samples.shape[0]], stacked[hsc_samples.shape[0] :]


def metric_factory(
    metric_mode: str,
    explained_threshold: float,
    pc_window: tuple[int, int] | None = None,
) -> Callable[[np.ndarray], float]:
    if metric_mode == "pattern":
        return lambda samples: dimensionality_from_pattern_samples(samples, explained_threshold)
    if metric_mode == "rdm":
        return lambda samples: dimensionality_from_samples(samples, explained_threshold)
    if metric_mode == "pc_window":
        if pc_window is None:
            raise ValueError("pc_window must be provided for metric_mode='pc_window'.")
        return lambda samples: explained_sum_for_pc_window(samples, pc_window[0], pc_window[1])
    raise ValueError(f"Unsupported metric_mode: {metric_mode}")


def compute_metric_with_optional_resampling(
    samples: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    target_trials: int | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    if target_trials is None:
        return metric_fn(samples)
    if target_trials < 2:
        raise ValueError("target_trials must be at least 2.")
    if samples.shape[0] < target_trials:
        raise ValueError(
            f"Need at least {target_trials} trials but only found {samples.shape[0]}."
        )
    if samples.shape[0] == target_trials:
        return metric_fn(samples)

    values = []
    for _ in range(n_resamples):
        indices = rng.choice(samples.shape[0], size=target_trials, replace=False)
        values.append(metric_fn(samples[indices]))
    return float(np.mean(values))


def save_roi_keep_vector(
    roi_mask_path: Path,
    keep_vector: np.ndarray,
    output_path: Path,
) -> None:
    roi_img = nib.load(str(roi_mask_path))
    roi_mask = np.asanyarray(roi_img.dataobj) != 0
    if keep_vector.shape[0] != int(roi_mask.sum()):
        raise ValueError("keep_vector length does not match ROI voxel count.")
    output = np.zeros(roi_mask.shape, dtype=np.float32)
    output[roi_mask] = keep_vector.astype(np.float32)
    header = roi_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(output, roi_img.affine, header), str(output_path))


def build_cross_subject_voxel_keep_vector(
    subjects: Sequence[str],
    pattern_dir: Path,
    roi_mask_path: Path,
    alpha: float = 0.05,
) -> dict[str, object]:
    roi_mask = load_mask(roi_mask_path)
    hsc_means = []
    lsc_means = []
    for subject_id in subjects:
        subject_dir = pattern_dir / subject_id
        hsc_samples = load_samples_from_nii(subject_dir / "glm_T_stats_HSC.nii", roi_mask)
        lsc_samples = load_samples_from_nii(subject_dir / "glm_T_stats_LSC.nii", roi_mask)
        hsc_means.append(np.mean(hsc_samples, axis=0))
        lsc_means.append(np.mean(lsc_samples, axis=0))

    hsc_matrix = np.vstack(hsc_means)
    lsc_matrix = np.vstack(lsc_means)
    test = stats.ttest_rel(hsc_matrix, lsc_matrix, axis=0, nan_policy="omit")
    p_values = np.asarray(test.pvalue)
    keep_vector = ~(p_values < alpha)
    return {
        "keep_vector": keep_vector,
        "n_voxels_total": int(keep_vector.size),
        "n_voxels_kept": int(np.sum(keep_vector)),
        "alpha": alpha,
    }


def run_roi_threshold_analysis(
    *,
    analysis_name: str,
    subjects: Sequence[str],
    pattern_dir: Path,
    roi_mask_path: Path,
    thresholds: Sequence[int],
    output_dir: Path,
    metric_mode: str,
    trial_strategy: str = "all",
    fixed_trials: int | None = None,
    demean: bool = False,
    voxel_keep_vector: np.ndarray | None = None,
    n_resamples: int = 1000,
    random_seed: int = 13,
    pc_window: tuple[int, int] | None = None,
) -> dict[str, object]:
    roi_mask = load_mask(roi_mask_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    threshold_summaries: list[dict[str, object]] = []

    for threshold in thresholds:
        metric_fn = metric_factory(metric_mode, threshold, pc_window=pc_window)
        records: list[SubjectDimension] = []
        included_subject_ids: list[str] = []
        excluded_subjects: list[dict[str, object]] = []

        for subject_index, subject_id in enumerate(subjects):
            subject_dir = pattern_dir / subject_id
            hsc_samples = load_samples_from_nii(subject_dir / "glm_T_stats_HSC.nii", roi_mask)
            lsc_samples = load_samples_from_nii(subject_dir / "glm_T_stats_LSC.nii", roi_mask)

            if voxel_keep_vector is not None:
                hsc_samples = hsc_samples[:, voxel_keep_vector]
                lsc_samples = lsc_samples[:, voxel_keep_vector]

            if demean:
                # Match demean scripts: concatenate both conditions first, subtract
                # the joint mean pattern, then split back into HSC / LSC.
                hsc_samples, lsc_samples = demean_condition_samples(hsc_samples, lsc_samples)

            if trial_strategy == "all":
                target_trials = None
            elif trial_strategy == "min":
                target_trials = min(hsc_samples.shape[0], lsc_samples.shape[0])
            elif trial_strategy == "fixed":
                if fixed_trials is None:
                    raise ValueError("fixed_trials must be provided when trial_strategy='fixed'.")
                target_trials = fixed_trials
            else:
                raise ValueError(f"Unsupported trial_strategy: {trial_strategy}")

            if target_trials is not None and (
                hsc_samples.shape[0] < target_trials or lsc_samples.shape[0] < target_trials
            ):
                excluded_subjects.append(
                    {
                        "analysis": analysis_name,
                        "threshold": threshold,
                        "subject_id": subject_id,
                        "status": "excluded",
                        "reason": "trial_count_below_target",
                        "hsc_trials": int(hsc_samples.shape[0]),
                        "lsc_trials": int(lsc_samples.shape[0]),
                        "target_trials": int(target_trials),
                    }
                )
                continue

            rng = np.random.default_rng(random_seed + (threshold * 100) + subject_index)
            try:
                # Same-trial MATLAB scripts repeatedly subsample without replacement
                # and then average the dimensionality estimate across repetitions.
                hsc_value = compute_metric_with_optional_resampling(
                    hsc_samples,
                    metric_fn,
                    target_trials=target_trials,
                    n_resamples=n_resamples,
                    rng=rng,
                )
                lsc_value = compute_metric_with_optional_resampling(
                    lsc_samples,
                    metric_fn,
                    target_trials=target_trials,
                    n_resamples=n_resamples,
                    rng=rng,
                )
            except ValueError as exc:
                excluded_subjects.append(
                    {
                        "analysis": analysis_name,
                        "threshold": threshold,
                        "subject_id": subject_id,
                        "status": "excluded",
                        "reason": f"metric_failed: {exc}",
                        "hsc_trials": int(hsc_samples.shape[0]),
                        "lsc_trials": int(lsc_samples.shape[0]),
                        "target_trials": None if target_trials is None else int(target_trials),
                    }
                )
                continue

            record = SubjectDimension(
                subject_id=subject_id,
                hsc_dimension=hsc_value,
                lsc_dimension=lsc_value,
                hsc_trials=int(hsc_samples.shape[0]),
                lsc_trials=int(lsc_samples.shape[0]),
            )
            records.append(record)
            included_subject_ids.append(subject_id)
            subject_rows.append(
                {
                    "analysis": analysis_name,
                    "threshold": threshold,
                    **subject_dimension_rows([record], "metric_mode", metric_mode)[0],
                }
            )

        stats_summary = paired_t_test(records)
        threshold_summaries.append(
            {
                "threshold": threshold,
                "included_subject_ids": included_subject_ids,
                "excluded_subjects": excluded_subjects,
                **stats_summary,
            }
        )
        selection_rows.extend(
            {
                "analysis": analysis_name,
                "threshold": threshold,
                "subject_id": subject_id,
                "status": "included",
                "reason": "included",
                "hsc_trials": None,
                "lsc_trials": None,
                "target_trials": None,
            }
            for subject_id in included_subject_ids
        )
        selection_rows.extend(excluded_subjects)

        logging.info(
            "%s threshold %s finished with %d included subjects.",
            analysis_name,
            threshold,
            len(included_subject_ids),
        )

    summary = {
        "task": analysis_name,
        "config": {
            "pattern_dir": str(pattern_dir),
            "roi_mask_path": str(roi_mask_path),
            "thresholds": list(thresholds),
            "metric_mode": metric_mode,
            "trial_strategy": trial_strategy,
            "fixed_trials": fixed_trials,
            "demean": demean,
            "n_resamples": n_resamples,
            "random_seed": random_seed,
            "pc_window": None if pc_window is None else list(pc_window),
            "voxel_mask_applied": voxel_keep_vector is not None,
        },
        "threshold_summaries": threshold_summaries,
    }
    write_csv(output_dir / "subject_results.csv", subject_rows)
    write_csv(output_dir / "selection_details.csv", selection_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def read_example_trial_scores(
    subject_id: str,
    events_dir: Path,
    score_column: str,
) -> list[float]:
    subject_dir = events_dir / subject_id
    event_paths = sorted(subject_dir.glob(f"{subject_id}_run-*_events.txt"), key=natural_sort_key)
    scores: list[float] = []
    for event_path in event_paths:
        with event_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if (row.get("trial_type") or "").strip() != "e":
                    continue
                raw_score = (row.get(score_column) or "").strip()
                if not raw_score:
                    continue
                try:
                    scores.append(float(raw_score))
                except ValueError:
                    continue
    return scores


def load_subject_glm_all_samples(
    subject_id: str,
    rsa_all_dir: Path,
    roi_mask_path: Path,
) -> np.ndarray:
    roi_mask = load_mask(roi_mask_path)
    return load_samples_from_nii(rsa_all_dir / subject_id / "glm_T_all.nii", roi_mask)


def paired_vector_stats(
    first: Sequence[float],
    second: Sequence[float],
    first_label: str,
    second_label: str,
) -> dict[str, object]:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.size < 2 or second_array.size < 2:
        return {
            "n_pairs": int(min(first_array.size, second_array.size)),
            "degrees_of_freedom": None,
            "t_stat": None,
            "p_value": None,
            "r_squared": None,
            f"mean_{first_label}": None,
            f"mean_{second_label}": None,
        }
    test = stats.ttest_rel(first_array, second_array, nan_policy="omit")
    degrees_of_freedom = int(first_array.size - 1)
    t_value = float(test.statistic)
    r_squared = (t_value * t_value) / ((t_value * t_value) + degrees_of_freedom)
    return {
        "n_pairs": int(first_array.size),
        "degrees_of_freedom": degrees_of_freedom,
        "t_stat": t_value,
        "p_value": float(test.pvalue),
        "r_squared": float(r_squared),
        f"mean_{first_label}": float(np.nanmean(first_array)),
        f"mean_{second_label}": float(np.nanmean(second_array)),
    }


def run_behavioral_pooling_analysis(
    *,
    analysis_name: str,
    subjects: Sequence[str],
    rsa_all_dir: Path,
    events_dir: Path,
    roi_mask_path: Path,
    score_column: str,
    top_fraction: float,
    bottom_fraction: float,
    top_sample_fraction: float | None,
    bottom_sample_fraction: float | None,
    n_resamples: int,
    explained_threshold: float,
    output_dir: Path,
    random_seed: int = 23,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[float] = []
    sample_blocks: list[np.ndarray] = []

    for subject_id in subjects:
        subject_scores = read_example_trial_scores(subject_id, events_dir, score_column)
        subject_samples = load_subject_glm_all_samples(subject_id, rsa_all_dir, roi_mask_path)
        if len(subject_scores) != subject_samples.shape[0]:
            logging.warning(
                "%s score/sample mismatch: %d scores vs %d samples.",
                subject_id,
                len(subject_scores),
                subject_samples.shape[0],
            )
        shared_length = min(len(subject_scores), subject_samples.shape[0])
        all_scores.extend(subject_scores[:shared_length])
        sample_blocks.append(subject_samples[:shared_length])

    stacked_samples = np.vstack(sample_blocks)
    scores = np.asarray(all_scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    sorted_samples = stacked_samples[order]
    sorted_scores = scores[order]

    # The exploratory MATLAB scripts sort all example trials by a behavioral
    # score, carve out top / bottom pools, and then resample from those pools.
    top_count = max(2, int(round(sorted_samples.shape[0] * top_fraction)))
    bottom_count = max(2, int(round(sorted_samples.shape[0] * bottom_fraction)))
    top_pool = sorted_samples[:top_count]
    bottom_pool = sorted_samples[-bottom_count:]

    top_values: list[float] = []
    bottom_values: list[float] = []
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)

    for repeat in range(1, n_resamples + 1):
        top_target = top_pool.shape[0] if top_sample_fraction is None else max(
            2, int(round(top_pool.shape[0] * top_sample_fraction))
        )
        bottom_target = bottom_pool.shape[0] if bottom_sample_fraction is None else max(
            2, int(round(bottom_pool.shape[0] * bottom_sample_fraction))
        )

        top_indices = rng.choice(top_pool.shape[0], size=top_target, replace=False)
        bottom_indices = rng.choice(bottom_pool.shape[0], size=bottom_target, replace=False)
        top_subset = top_pool[top_indices]
        bottom_subset = bottom_pool[bottom_indices]

        top_value = dimensionality_from_samples(top_subset, explained_threshold)
        bottom_value = dimensionality_from_samples(bottom_subset, explained_threshold)
        top_values.append(top_value)
        bottom_values.append(bottom_value)
        rows.append(
            {
                "analysis": analysis_name,
                "repeat": repeat,
                "top_value": top_value,
                "bottom_value": bottom_value,
                "top_mean_score": float(np.mean(sorted_scores[:top_count])),
                "bottom_mean_score": float(np.mean(sorted_scores[-bottom_count:])),
            }
        )

    summary = {
        "task": analysis_name,
        "config": {
            "rsa_all_dir": str(rsa_all_dir),
            "events_dir": str(events_dir),
            "roi_mask_path": str(roi_mask_path),
            "score_column": score_column,
            "top_fraction": top_fraction,
            "bottom_fraction": bottom_fraction,
            "top_sample_fraction": top_sample_fraction,
            "bottom_sample_fraction": bottom_sample_fraction,
            "n_resamples": n_resamples,
            "explained_threshold": explained_threshold,
        },
        "top_pool_size": int(top_count),
        "bottom_pool_size": int(bottom_count),
        "statistics": paired_vector_stats(top_values, bottom_values, "top", "bottom"),
    }
    write_csv(output_dir / "resample_results.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def load_binary_mask(mask_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    mask_img = nib.load(str(mask_path))
    mask = np.asanyarray(mask_img.dataobj) != 0
    return mask_img, mask


def build_fixed_count_neighborhood(mask: np.ndarray, voxel_count: int) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if coordinates.shape[0] == 0:
        raise ValueError("Mask does not contain any voxels.")
    # MATLAB used cosmo_spherical_neighborhood with a fixed voxel count.
    # Here we approximate that by querying each voxel's nearest masked neighbors.
    tree = cKDTree(coordinates)
    k = min(voxel_count, coordinates.shape[0])
    _, indices = tree.query(coordinates, k=k)
    if k == 1:
        indices = indices[:, np.newaxis]
    return np.asarray(indices, dtype=np.int32)


def save_masked_scalar_map(
    reference_mask_img: nib.Nifti1Image,
    mask: np.ndarray,
    values: np.ndarray,
    output_path: Path,
) -> None:
    output = np.zeros(mask.shape, dtype=np.float32)
    output[mask] = values.astype(np.float32)
    header = reference_mask_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(output, reference_mask_img.affine, header), str(output_path))


def compute_searchlight_dimension_map(
    data_path: Path,
    subject_mask_path: Path,
    explained_threshold: float,
    voxel_count: int,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    mask_img, mask = load_binary_mask(subject_mask_path)
    samples = load_samples_from_nii(data_path, mask)
    neighborhoods = build_fixed_count_neighborhood(mask, voxel_count)
    values = np.full(neighborhoods.shape[0], np.nan, dtype=np.float32)
    for center_index, neighbor_indices in enumerate(neighborhoods):
        local_samples = samples[:, neighbor_indices]
        try:
            values[center_index] = float(dimensionality_from_samples(local_samples, explained_threshold))
        except ValueError:
            values[center_index] = np.nan
    return mask_img, mask, values


def pca_components_and_explained(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        raise ValueError("Centered matrix has zero variance; cannot run PCA.")
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    denom = max(centered.shape[0] - 1, 1)
    variances = (singular_values**2) / denom
    total = np.sum(variances)
    if np.isclose(total, 0.0):
        raise ValueError("Total PCA variance is zero.")
    explained = variances / total * 100.0
    coeff = vh.T
    return coeff, explained


def first_component_count(explained: np.ndarray, explained_threshold: float) -> int:
    return int(cumulative_threshold_dimension(explained, explained_threshold))


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(np.dot(vector1, vector2) / denominator)


def correlation_similarity_matrix(samples: np.ndarray) -> np.ndarray:
    centered = samples - samples.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 0
    if np.sum(valid) < 2:
        raise ValueError("Need at least two non-constant trials to build similarity matrix.")
    centered = centered[valid]
    norms = norms[valid]
    correlation = centered @ centered.T
    correlation /= np.outer(norms, norms)
    correlation = np.clip(correlation, -1.0, 1.0)
    return correlation


def compute_seed_connectivity_map(
    data_path: Path,
    subject_mask_path: Path,
    seed_samples: np.ndarray,
    explained_threshold: float,
    voxel_count: int,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    seed_similarity = correlation_similarity_matrix(seed_samples)
    seed_coeff, seed_explained = pca_components_and_explained(seed_similarity)
    seed_component_count = first_component_count(seed_explained, explained_threshold)

    mask_img, mask = load_binary_mask(subject_mask_path)
    samples = load_samples_from_nii(data_path, mask)
    neighborhoods = build_fixed_count_neighborhood(mask, voxel_count)
    values = np.full(neighborhoods.shape[0], np.nan, dtype=np.float32)

    for center_index, neighbor_indices in enumerate(neighborhoods):
        local_samples = samples[:, neighbor_indices]
        try:
            local_similarity = correlation_similarity_matrix(local_samples)
            local_coeff, local_explained = pca_components_and_explained(local_similarity)
            local_component_count = first_component_count(local_explained, explained_threshold)
        except ValueError:
            values[center_index] = np.nan
            continue

        component_count = min(seed_component_count, local_component_count)
        if component_count < 1:
            values[center_index] = np.nan
            continue
        # This follows my_measure.m: compare matched PCA component vectors from
        # the hippocampal seed and the local searchlight via cosine similarity.
        similarities = [
            cosine_similarity(seed_coeff[:, component_index], local_coeff[:, component_index])
            for component_index in range(component_count)
        ]
        finite_values = [value for value in similarities if np.isfinite(value)]
        values[center_index] = float(np.mean(finite_values)) if finite_values else np.nan

    return mask_img, mask, values


def load_scalar_map(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D map, got {data.ndim}D: {path}")
    return img, data.astype(np.float64, copy=False)


def save_full_map(reference_img: nib.Nifti1Image, values: np.ndarray, output_path: Path) -> None:
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(values.astype(np.float32), reference_img.affine, header),
        str(output_path),
    )


def compute_group_paired_map_statistics(
    hsc_map_paths: Sequence[Path],
    lsc_map_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, object]:
    if len(hsc_map_paths) != len(lsc_map_paths):
        raise ValueError("Need matching numbers of HSC and LSC maps.")
    reference_img, reference_data = load_scalar_map(hsc_map_paths[0])
    flat_length = reference_data.size
    hsc_stack = np.vstack([load_scalar_map(path)[1].reshape(flat_length) for path in hsc_map_paths])
    lsc_stack = np.vstack([load_scalar_map(path)[1].reshape(flat_length) for path in lsc_map_paths])
    test = stats.ttest_rel(hsc_stack, lsc_stack, axis=0, nan_policy="omit")
    t_values = np.asarray(test.statistic, dtype=np.float64)
    p_values = np.asarray(test.pvalue, dtype=np.float64)
    valid_counts = np.sum(np.isfinite(hsc_stack) & np.isfinite(lsc_stack), axis=0)
    degrees_of_freedom = np.maximum(valid_counts - 1, 0)
    r_squared = np.full_like(t_values, np.nan, dtype=np.float64)
    valid_df = degrees_of_freedom > 0
    r_squared[valid_df] = (t_values[valid_df] ** 2) / (
        (t_values[valid_df] ** 2) + degrees_of_freedom[valid_df]
    )
    mean_diff = np.nanmean(hsc_stack - lsc_stack, axis=0)
    mean_hsc = np.nanmean(hsc_stack, axis=0)
    mean_lsc = np.nanmean(lsc_stack, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_full_map(reference_img, t_values.reshape(reference_data.shape), output_dir / "paired_t_map.nii")
    save_full_map(reference_img, p_values.reshape(reference_data.shape), output_dir / "paired_p_map.nii")
    save_full_map(reference_img, r_squared.reshape(reference_data.shape), output_dir / "paired_r_squared_map.nii")
    save_full_map(reference_img, mean_diff.reshape(reference_data.shape), output_dir / "mean_difference_map.nii")
    save_full_map(reference_img, mean_hsc.reshape(reference_data.shape), output_dir / "mean_hsc_map.nii")
    save_full_map(reference_img, mean_lsc.reshape(reference_data.shape), output_dir / "mean_lsc_map.nii")

    summary = {
        "n_subjects": int(len(hsc_map_paths)),
        "paired_t_map": str(output_dir / "paired_t_map.nii"),
        "paired_p_map": str(output_dir / "paired_p_map.nii"),
        "paired_r_squared_map": str(output_dir / "paired_r_squared_map.nii"),
        "mean_difference_map": str(output_dir / "mean_difference_map.nii"),
        "mean_hsc_map": str(output_dir / "mean_hsc_map.nii"),
        "mean_lsc_map": str(output_dir / "mean_lsc_map.nii"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_dimensionality_searchlight_suite(
    *,
    subjects: Sequence[str],
    pattern_dir: Path,
    subject_mask_root: Path,
    output_dir: Path,
    voxel_count: int,
    explained_threshold: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_rows: list[dict[str, object]] = []
    hsc_map_paths: list[Path] = []
    lsc_map_paths: list[Path] = []

    for subject_id in subjects:
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        subject_mask_path = subject_mask_root / subject_id / "mask.nii"
        subject_dir = pattern_dir / subject_id

        hsc_mask_img, hsc_mask, hsc_values = compute_searchlight_dimension_map(
            subject_dir / "glm_T_stats_HSC.nii",
            subject_mask_path,
            explained_threshold=explained_threshold,
            voxel_count=voxel_count,
        )
        lsc_mask_img, lsc_mask, lsc_values = compute_searchlight_dimension_map(
            subject_dir / "glm_T_stats_LSC.nii",
            subject_mask_path,
            explained_threshold=explained_threshold,
            voxel_count=voxel_count,
        )

        hsc_output_path = subject_output_dir / f"rdvar_hsc_{int(explained_threshold)}.nii"
        lsc_output_path = subject_output_dir / f"rdvar_lsc_{int(explained_threshold)}.nii"
        save_masked_scalar_map(hsc_mask_img, hsc_mask, hsc_values, hsc_output_path)
        save_masked_scalar_map(lsc_mask_img, lsc_mask, lsc_values, lsc_output_path)
        hsc_map_paths.append(hsc_output_path)
        lsc_map_paths.append(lsc_output_path)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "hsc_map": str(hsc_output_path),
                "lsc_map": str(lsc_output_path),
            }
        )

    group_summary = compute_group_paired_map_statistics(
        hsc_map_paths,
        lsc_map_paths,
        output_dir / f"second{int(explained_threshold)}",
    )
    summary = {
        "task": "searchlight_dimension",
        "config": {
            "pattern_dir": str(pattern_dir),
            "subject_mask_root": str(subject_mask_root),
            "voxel_count": voxel_count,
            "explained_threshold": explained_threshold,
        },
        "subject_outputs": subject_rows,
        "group_summary": group_summary,
    }
    write_csv(output_dir / "subject_outputs.csv", subject_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def run_seed_connectivity_searchlight_suite(
    *,
    subjects: Sequence[str],
    pattern_dir: Path,
    subject_mask_root: Path,
    seed_mask_path: Path,
    output_dir: Path,
    voxel_count: int,
    explained_threshold: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_mask = load_mask(seed_mask_path)
    subject_rows: list[dict[str, object]] = []
    hsc_map_paths: list[Path] = []
    lsc_map_paths: list[Path] = []

    for subject_id in subjects:
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        subject_mask_path = subject_mask_root / subject_id / "mask.nii"
        subject_dir = pattern_dir / subject_id

        hsc_seed_samples = load_samples_from_nii(subject_dir / "glm_T_stats_HSC.nii", seed_mask)
        lsc_seed_samples = load_samples_from_nii(subject_dir / "glm_T_stats_LSC.nii", seed_mask)

        hsc_mask_img, hsc_mask, hsc_values = compute_seed_connectivity_map(
            subject_dir / "glm_T_stats_HSC.nii",
            subject_mask_path,
            seed_samples=hsc_seed_samples,
            explained_threshold=explained_threshold,
            voxel_count=voxel_count,
        )
        lsc_mask_img, lsc_mask, lsc_values = compute_seed_connectivity_map(
            subject_dir / "glm_T_stats_LSC.nii",
            subject_mask_path,
            seed_samples=lsc_seed_samples,
            explained_threshold=explained_threshold,
            voxel_count=voxel_count,
        )

        hsc_output_path = subject_output_dir / "searchlight_lHIP_hsc.nii"
        lsc_output_path = subject_output_dir / "searchlight_lHIP_lsc.nii"
        save_masked_scalar_map(hsc_mask_img, hsc_mask, hsc_values, hsc_output_path)
        save_masked_scalar_map(lsc_mask_img, lsc_mask, lsc_values, lsc_output_path)
        hsc_map_paths.append(hsc_output_path)
        lsc_map_paths.append(lsc_output_path)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "hsc_map": str(hsc_output_path),
                "lsc_map": str(lsc_output_path),
            }
        )

    group_summary = compute_group_paired_map_statistics(
        hsc_map_paths,
        lsc_map_paths,
        output_dir / "second",
    )
    summary = {
        "task": "seed_connectivity_searchlight",
        "config": {
            "pattern_dir": str(pattern_dir),
            "subject_mask_root": str(subject_mask_root),
            "seed_mask_path": str(seed_mask_path),
            "voxel_count": voxel_count,
            "explained_threshold": explained_threshold,
        },
        "subject_outputs": subject_rows,
        "group_summary": group_summary,
    }
    write_csv(output_dir / "subject_outputs.csv", subject_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def collect_memory_accuracy_table(
    subjects: Iterable[str] = SUBJECTS,
    events_dir: Path = DEFAULT_EVENTS_DIR,
    good_memory_threshold: float = 0.5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_id in subjects:
        record = read_memory_accuracy(subject_id, events_dir, good_memory_threshold)
        rows.append(record.__dict__)
    return rows
