from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from gjxx.utils import components_to_variance_threshold, ensure_dir, paired_t_summary, save_json, write_table


def _masked_samples(four_d_path: str | Path, mask_path: str | Path) -> np.ndarray:
    img = nib.load(str(four_d_path))
    data = np.asarray(img.get_fdata(), dtype=float)
    mask = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
    return data[mask, :].T


def _rd_from_rdm(samples: np.ndarray, threshold: float) -> int:
    corr = np.corrcoef(samples)
    rdm = 1.0 - corr
    eigenvalues = np.linalg.eigvalsh(rdm)
    return components_to_variance_threshold(eigenvalues, threshold)


def _rd_from_equalized_patterns(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    threshold: float,
    n_iter: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    target = min(samples_a.shape[0], samples_b.shape[0])

    def _estimate(samples: np.ndarray) -> float:
        if samples.shape[0] == target:
            eigenvalues = np.linalg.eigvalsh(np.cov(samples, rowvar=False))
            return float(components_to_variance_threshold(eigenvalues, threshold))
        values = []
        for _ in range(n_iter):
            choice = rng.choice(samples.shape[0], size=target, replace=False)
            subset = samples[choice]
            eigenvalues = np.linalg.eigvalsh(np.cov(subset, rowvar=False))
            values.append(components_to_variance_threshold(eigenvalues, threshold))
        return float(np.mean(values))

    return _estimate(samples_a), _estimate(samples_b)


def run_group_rd(
    pattern_root: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 80.0,
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
        samples_h = _masked_samples(hsc_path, mask_path)
        samples_l = _masked_samples(lsc_path, mask_path)
        rows.append(
            {
                "subject": subject_dir.name,
                "rd_hsc": float(_rd_from_rdm(samples_h, threshold)),
                "rd_lsc": float(_rd_from_rdm(samples_l, threshold)),
            }
        )

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "rd_subject_metrics.tsv")
    save_json(paired_t_summary(frame["rd_hsc"], frame["rd_lsc"]), output_dir / "rd_group_summary.json")
    return frame


def run_group_equalized_rd(
    pattern_root: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 80.0,
    n_iter: int = 1000,
    seed: int = 0,
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
        samples_h = _masked_samples(hsc_path, mask_path)
        samples_l = _masked_samples(lsc_path, mask_path)
        rd_hsc, rd_lsc = _rd_from_equalized_patterns(
            samples_h,
            samples_l,
            threshold=threshold,
            n_iter=n_iter,
            seed=seed,
        )
        rows.append({"subject": subject_dir.name, "rd_hsc": rd_hsc, "rd_lsc": rd_lsc})

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "rd_equalized_subject_metrics.tsv")
    save_json(
        paired_t_summary(frame["rd_hsc"], frame["rd_lsc"]),
        output_dir / "rd_equalized_group_summary.json",
    )
    return frame


def _neighborhood_matrix(mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    distances = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    return coords, neighbors


def run_rd_searchlight(
    input_img: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float = 80.0,
    neighbors: int = 100,
) -> None:
    img = nib.load(str(input_img))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D input image: {input_img}")
    mask = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
    coords, neighborhood = _neighborhood_matrix(mask, neighbors)
    flattened = data[mask, :].T
    values = np.zeros(coords.shape[0], dtype=float)
    for center_index, feature_ids in enumerate(neighborhood):
        samples = flattened[:, feature_ids]
        values[center_index] = _rd_from_rdm(samples, threshold)

    out = np.zeros(mask.shape, dtype=float)
    out[mask] = values
    ensure_dir(Path(output_path).parent)
    nib.Nifti1Image(out, affine=img.affine, header=img.header).to_filename(str(output_path))
