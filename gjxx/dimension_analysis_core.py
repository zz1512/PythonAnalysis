from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import nibabel as nib
    import numpy as np
    from scipy import stats
except ImportError as exc:
    raise SystemExit(
        "Missing required dependency. Please install numpy, scipy, and nibabel. "
        f"Original error: {exc}"
    )


SUBJECTS = [
    "sub-02",
    "sub-03",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-08",
    "sub-09",
    "sub-10",
    "sub-11",
    "sub-12",
    "sub-13",
    "sub-14",
    "sub-15",
    "sub-17",
    "sub-18",
    "sub-19",
    "sub-20",
    "sub-21",
    "sub-25",
    "sub-28",
    "sub-29",
    "sub-30",
]


DEFAULT_PATTERN_DIR = Path(r"I:\FLXX1\Dimension\pattern")
DEFAULT_PATTERNS_HLRF_DIR = Path(r"I:\FLXX1\First_level\patterns_hlrf")
DEFAULT_EVENTS_DIR = Path(r"I:\FLXX1\events")
DEFAULT_MASK_PATH = Path(r"I:\FLXX1\mask\Hippocampus_L.nii")
DEFAULT_OUTPUT_ROOT = Path(r"I:\FLXX1\Dimension\python_results")


@dataclass
class SubjectDimension:
    subject_id: str
    hsc_dimension: float
    lsc_dimension: float
    hsc_trials: int
    lsc_trials: int


@dataclass
class SubjectMemoryAccuracy:
    subject_id: str
    remembered_trials: int
    total_trials: int
    accuracy: float
    memory_group: str


@dataclass
class ExclusionRecord:
    subject_id: str
    hsc_trials: int
    lsc_trials: int
    reason: str


@dataclass
class AnalysisResult:
    records: list[SubjectDimension]
    included_subject_ids: list[str]
    excluded_due_to_trial_count: list[ExclusionRecord]
    excluded_due_to_estimation_failure: list[ExclusionRecord]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    key: list[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def load_mask(mask_path: Path) -> np.ndarray:
    mask_img = nib.load(str(mask_path))
    mask_data = np.asanyarray(mask_img.dataobj)
    if mask_data.ndim != 3:
        raise ValueError(f"Mask must be 3D, got {mask_data.ndim}D: {mask_path}")
    mask = mask_data != 0
    logging.info("Loaded mask %s with %d voxels.", mask_path, int(mask.sum()))
    return mask


def load_samples_from_nii(path: Path, mask: np.ndarray) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D NIfTI, got {data.ndim}D: {path}")
    if data.shape[:3] != mask.shape:
        raise ValueError(
            f"Spatial shape mismatch for {path}: image {data.shape[:3]}, mask {mask.shape}"
        )
    return data[mask, :].T.astype(np.float64, copy=False)


def load_samples_from_many(paths: Sequence[Path], mask: np.ndarray) -> np.ndarray:
    if not paths:
        return np.empty((0, int(mask.sum())), dtype=np.float64)
    blocks = [load_samples_from_nii(path, mask) for path in paths]
    return np.concatenate(blocks, axis=0)


def correlation_distance_rdm(samples: np.ndarray) -> np.ndarray:
    centered = samples - samples.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > 0
    if valid.sum() < 2:
        raise ValueError("Need at least two non-constant trials to compute an RDM.")
    if not np.all(valid):
        dropped = int((~valid).sum())
        logging.warning("Dropped %d constant trial(s) before RDM computation.", dropped)
        centered = centered[valid]
        norms = norms[valid]
    corr = centered @ centered.T
    corr /= np.outer(norms, norms)
    corr = np.clip(corr, -1.0, 1.0)
    rdm = 1.0 - corr
    np.fill_diagonal(rdm, 0.0)
    return rdm


def pca_explained_variance(matrix: np.ndarray) -> np.ndarray:
    # This reproduces MATLAB pca(matrix) behavior more closely than
    # eigen-decomposing the raw RDM. We center columns first and then run SVD.
    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
    if np.allclose(matrix_centered, 0.0):
        raise ValueError("Centered matrix has zero variance; cannot run PCA.")

    _, singular_values, _ = np.linalg.svd(matrix_centered, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("No singular values returned from PCA.")

    denom = max(matrix_centered.shape[0] - 1, 1)
    variances = (singular_values**2) / denom
    total = variances.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Total PCA variance is zero; cannot compute explained variance.")
    return variances / total * 100.0


def dimensionality_from_samples(samples: np.ndarray, explained_threshold: float) -> float:
    # MATLAB task logic:
    # 1) build trial-by-trial correlation-distance RDM
    # 2) run PCA on the RDM matrix
    # 3) return the first dimension count whose cumulative explained variance
    #    exceeds the threshold.
    if samples.shape[0] < 2:
        raise ValueError("Need at least two trials to estimate dimensionality.")
    rdm = correlation_distance_rdm(samples)
    explained = pca_explained_variance(rdm)
    cumulative = 0.0
    for index, value in enumerate(explained, start=1):
        cumulative += value
        if cumulative > explained_threshold:
            return float(index)
    return float(len(explained))


def collect_trial_files(subject_dir: Path, prefix: str) -> list[Path]:
    return sorted(subject_dir.glob(f"{prefix}*.nii"), key=natural_sort_key)


def analyze_trial_split(
    subjects: Iterable[str],
    patterns_hlrf_dir: Path,
    mask: np.ndarray,
    explained_threshold: float,
    min_trials: int,
    analysis_name: str,
    hsc_prefix: str,
    lsc_prefix: str,
) -> AnalysisResult:
    records: list[SubjectDimension] = []
    included_subject_ids: list[str] = []
    excluded_due_to_trial_count: list[ExclusionRecord] = []
    excluded_due_to_estimation_failure: list[ExclusionRecord] = []
    logging.info("Running %s analysis.", analysis_name)
    for subject_id in subjects:
        subject_dir = patterns_hlrf_dir / subject_id
        hsc_paths = collect_trial_files(subject_dir, hsc_prefix)
        lsc_paths = collect_trial_files(subject_dir, lsc_prefix)

        if len(hsc_paths) < min_trials or len(lsc_paths) < min_trials:
            logging.warning(
                "Skip %s %s because trial counts are too small (HSC=%d, LSC=%d).",
                analysis_name,
                subject_id,
                len(hsc_paths),
                len(lsc_paths),
            )
            excluded_due_to_trial_count.append(
                ExclusionRecord(
                    subject_id=subject_id,
                    hsc_trials=len(hsc_paths),
                    lsc_trials=len(lsc_paths),
                    reason="trial_count_below_min",
                )
            )
            continue

        hsc_samples = load_samples_from_many(hsc_paths, mask)
        lsc_samples = load_samples_from_many(lsc_paths, mask)
        try:
            hsc_dimension = dimensionality_from_samples(hsc_samples, explained_threshold)
            lsc_dimension = dimensionality_from_samples(lsc_samples, explained_threshold)
        except ValueError as exc:
            logging.warning(
                "Skip %s %s because dimensionality estimation failed: %s",
                analysis_name,
                subject_id,
                exc,
            )
            excluded_due_to_estimation_failure.append(
                ExclusionRecord(
                    subject_id=subject_id,
                    hsc_trials=hsc_samples.shape[0],
                    lsc_trials=lsc_samples.shape[0],
                    reason=f"dimensionality_failed: {exc}",
                )
            )
            continue

        records.append(
            SubjectDimension(
                subject_id=subject_id,
                hsc_dimension=hsc_dimension,
                lsc_dimension=lsc_dimension,
                hsc_trials=hsc_samples.shape[0],
                lsc_trials=lsc_samples.shape[0],
            )
        )
        included_subject_ids.append(subject_id)

    logging.info("%s finished with %d valid subjects.", analysis_name, len(records))
    logging.info("%s included subjects: %s", analysis_name, included_subject_ids)
    logging.info(
        "%s excluded due to trial count: %s",
        analysis_name,
        [record.subject_id for record in excluded_due_to_trial_count],
    )
    if excluded_due_to_estimation_failure:
        logging.info(
            "%s excluded due to dimensionality failure: %s",
            analysis_name,
            [record.subject_id for record in excluded_due_to_estimation_failure],
        )
    return AnalysisResult(
        records=records,
        included_subject_ids=included_subject_ids,
        excluded_due_to_trial_count=excluded_due_to_trial_count,
        excluded_due_to_estimation_failure=excluded_due_to_estimation_failure,
    )


def analyze_remembered_forgotten(
    subjects: Iterable[str],
    patterns_hlrf_dir: Path,
    mask: np.ndarray,
    explained_threshold: float,
    min_trials: int,
) -> dict[str, AnalysisResult]:
    return {
        "remembered": analyze_trial_split(
            subjects=subjects,
            patterns_hlrf_dir=patterns_hlrf_dir,
            mask=mask,
            explained_threshold=explained_threshold,
            min_trials=min_trials,
            analysis_name="remembered",
            hsc_prefix="HSC_rspmT_",
            lsc_prefix="LSC_rspmT_",
        ),
        "forgotten": analyze_trial_split(
            subjects=subjects,
            patterns_hlrf_dir=patterns_hlrf_dir,
            mask=mask,
            explained_threshold=explained_threshold,
            min_trials=min_trials,
            analysis_name="forgotten",
            hsc_prefix="HSC_fspmT_",
            lsc_prefix="LSC_fspmT_",
        ),
    }


def read_memory_accuracy(
    subject_id: str,
    events_dir: Path,
    good_memory_threshold: float,
) -> SubjectMemoryAccuracy:
    subject_dir = events_dir / subject_id
    event_paths = sorted(subject_dir.glob(f"{subject_id}_run-*_events.txt"), key=natural_sort_key)
    if len(event_paths) != 4:
        logging.warning(
            "%s expected 4 event files but found %d under %s",
            subject_id,
            len(event_paths),
            subject_dir,
        )

    remembered_trials = 0
    total_trials = 0
    for event_path in event_paths:
        with event_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                # Only count example trials.
                trial_type = (row.get("trial_type") or "").strip()
                if trial_type != "e":
                    continue

                # Project rule:
                # memory == 1  -> remembered
                # memory == -1 -> forgotten
                memory_raw = (row.get("memory") or "").strip()
                try:
                    memory_value = int(float(memory_raw))
                except ValueError:
                    continue
                if memory_value not in (-1, 1):
                    continue

                total_trials += 1
                if memory_value == 1:
                    remembered_trials += 1

    accuracy = remembered_trials / total_trials if total_trials else float("nan")
    memory_group = "good" if accuracy > good_memory_threshold else "poor"
    return SubjectMemoryAccuracy(
        subject_id=subject_id,
        remembered_trials=remembered_trials,
        total_trials=total_trials,
        accuracy=accuracy,
        memory_group=memory_group,
    )


def compute_memory_accuracies(
    subjects: Iterable[str],
    events_dir: Path,
    good_memory_threshold: float,
) -> list[SubjectMemoryAccuracy]:
    return [
        read_memory_accuracy(
            subject_id=subject_id,
            events_dir=events_dir,
            good_memory_threshold=good_memory_threshold,
        )
        for subject_id in subjects
    ]


def analyze_memory_groups(
    subjects: Iterable[str],
    pattern_dir: Path,
    mask: np.ndarray,
    explained_threshold: float,
    min_trials: int,
    accuracy_by_subject: dict[str, SubjectMemoryAccuracy],
) -> dict[str, AnalysisResult]:
    grouped_results: dict[str, AnalysisResult] = {
        "good": AnalysisResult([], [], [], []),
        "poor": AnalysisResult([], [], [], []),
    }

    for subject_id in subjects:
        subject_info = accuracy_by_subject[subject_id]
        group_name = subject_info.memory_group
        group_result = grouped_results[group_name]
        subject_dir = pattern_dir / subject_id
        hsc_path = subject_dir / "glm_T_stats_HSC.nii"
        lsc_path = subject_dir / "glm_T_stats_LSC.nii"

        if not hsc_path.exists() or not lsc_path.exists():
            logging.warning("Skip %s because required pattern files are missing.", subject_id)
            continue

        hsc_samples = load_samples_from_nii(hsc_path, mask)
        lsc_samples = load_samples_from_nii(lsc_path, mask)
        if hsc_samples.shape[0] < min_trials or lsc_samples.shape[0] < min_trials:
            logging.warning(
                "Skip memory-group analysis for %s because trial counts are too small "
                "(HSC=%d, LSC=%d; min_trials=%d).",
                subject_id,
                hsc_samples.shape[0],
                lsc_samples.shape[0],
                min_trials,
            )
            group_result.excluded_due_to_trial_count.append(
                ExclusionRecord(
                    subject_id=subject_id,
                    hsc_trials=hsc_samples.shape[0],
                    lsc_trials=lsc_samples.shape[0],
                    reason="trial_count_below_min",
                )
            )
            continue
        try:
            hsc_dimension = dimensionality_from_samples(hsc_samples, explained_threshold)
            lsc_dimension = dimensionality_from_samples(lsc_samples, explained_threshold)
        except ValueError as exc:
            logging.warning(
                "Skip memory-group analysis for %s because dimensionality estimation failed: %s",
                subject_id,
                exc,
            )
            group_result.excluded_due_to_estimation_failure.append(
                ExclusionRecord(
                    subject_id=subject_id,
                    hsc_trials=hsc_samples.shape[0],
                    lsc_trials=lsc_samples.shape[0],
                    reason=f"dimensionality_failed: {exc}",
                )
            )
            continue

        group_result.records.append(
            SubjectDimension(
                subject_id=subject_id,
                hsc_dimension=hsc_dimension,
                lsc_dimension=lsc_dimension,
                hsc_trials=hsc_samples.shape[0],
                lsc_trials=lsc_samples.shape[0],
            )
        )
        group_result.included_subject_ids.append(subject_id)

    for group_name, group_result in grouped_results.items():
        logging.info("%s final included subjects: %s", group_name, group_result.included_subject_ids)
        logging.info(
            "%s excluded due to trial count: %s",
            group_name,
            [record.subject_id for record in group_result.excluded_due_to_trial_count],
        )
        if group_result.excluded_due_to_estimation_failure:
            logging.info(
                "%s excluded due to dimensionality failure: %s",
                group_name,
                [record.subject_id for record in group_result.excluded_due_to_estimation_failure],
            )
    return grouped_results


def paired_t_test(records: Sequence[SubjectDimension]) -> dict[str, object]:
    if len(records) < 2:
        return {
            "n_subjects": len(records),
            "degrees_of_freedom": None,
            "t_stat": None,
            "p_value": None,
            "r_squared": None,
            "mean_hsc": None,
            "mean_lsc": None,
        }
    hsc = np.array([record.hsc_dimension for record in records], dtype=np.float64)
    lsc = np.array([record.lsc_dimension for record in records], dtype=np.float64)
    test = stats.ttest_rel(hsc, lsc, nan_policy="omit")
    degrees_of_freedom = len(records) - 1
    t_value = float(test.statistic)
    r_squared = (t_value * t_value) / ((t_value * t_value) + degrees_of_freedom)
    return {
        "n_subjects": int(len(records)),
        "degrees_of_freedom": int(degrees_of_freedom),
        "t_stat": t_value,
        "p_value": float(test.pvalue),
        "r_squared": float(r_squared),
        "mean_hsc": float(np.nanmean(hsc)),
        "mean_lsc": float(np.nanmean(lsc)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def subject_dimension_rows(
    records: Sequence[SubjectDimension],
    extra_key: str,
    extra_value: str,
) -> list[dict[str, object]]:
    return [{extra_key: extra_value, **asdict(record)} for record in records]


def exclusion_rows(
    records: Sequence[ExclusionRecord],
    extra_key: str,
    extra_value: str,
    status: str,
) -> list[dict[str, object]]:
    return [
        {
            extra_key: extra_value,
            "status": status,
            **asdict(record),
        }
        for record in records
    ]
