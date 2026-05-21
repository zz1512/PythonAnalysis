#!/usr/bin/env python3
"""Network ROI overlap QC (Task 1.6).

For every peak in ``language_network`` / ``tom_network`` and every peak in
``meta_metaphor`` / ``meta_spatial`` (all with a fixed 6mm sphere), compute:

- Euclidean distance in MNI space between the two peaks (mm).
- Voxel-level intersection / union ratio between the two 6mm spheres sampled
  on the reference image grid (same grid used by ``build_meta_roi_library.py``).

Output table goes to ``$paper/qc/roi_library/network_overlap_qc.tsv``.

Usage::

    python metaphoric/final_version/roi_management/network_overlap_qc.py \
        --primary-peaks metaphoric/final_version/roi_management/meta_roi_peaks_curated.tsv \
        --network-peaks metaphoric/final_version/roi_management/language_network_peaks.tsv \
                        metaphoric/final_version/roi_management/tom_network_peaks.tsv \
        --reference-image $patternRoot/sub-01/pre_yy.nii.gz \
        --output $paper/qc/roi_library/network_overlap_qc.tsv

The sphere geometry matches ``build_meta_roi_library.py`` so overlap ratios
here are directly comparable to the primary meta ROI masks.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _read_peaks(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"roi_set", "roi_name", "x", "y", "z"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "radius_mm" not in frame.columns:
        frame["radius_mm"] = 6.0
    frame["radius_mm"] = pd.to_numeric(frame["radius_mm"], errors="coerce").fillna(6.0)
    for col in ("x", "y", "z"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[["x", "y", "z"]].isna().any().any():
        raise ValueError(f"{path}: non-numeric MNI coordinates detected")
    return frame


def _voxel_sizes_mm(affine: np.ndarray) -> np.ndarray:
    # Column norms of the affine upper-left 3x3 approximate voxel sizes (mm).
    A = np.asarray(affine[:3, :3], dtype=float)
    return np.array([np.linalg.norm(A[:, i]) for i in range(3)], dtype=float)


def _sphere_linear_indices(
    center_mni: np.ndarray,
    radius_mm: float,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    inv_affine: np.ndarray,
    voxel_sizes: np.ndarray,
) -> np.ndarray:
    """
    Return 1D linear voxel indices inside a sphere.

    This avoids building a full-brain world grid (performance). We only scan a
    tight bounding box around the sphere center.
    """
    center_mni = np.asarray(center_mni, dtype=float).reshape(3)
    hom = np.array([center_mni[0], center_mni[1], center_mni[2], 1.0], dtype=float)
    center_ijk = (inv_affine @ hom)[:3]
    center_ijk = np.asarray(center_ijk, dtype=float)

    # Conservative bbox in voxel space (handles anisotropy).
    deltas = np.ceil(float(radius_mm) / np.maximum(voxel_sizes, 1e-6)).astype(int)
    i0 = max(0, int(np.floor(center_ijk[0] - deltas[0])))
    i1 = min(shape[0] - 1, int(np.ceil(center_ijk[0] + deltas[0])))
    j0 = max(0, int(np.floor(center_ijk[1] - deltas[1])))
    j1 = min(shape[1] - 1, int(np.ceil(center_ijk[1] + deltas[1])))
    k0 = max(0, int(np.floor(center_ijk[2] - deltas[2])))
    k1 = min(shape[2] - 1, int(np.ceil(center_ijk[2] + deltas[2])))

    if i1 < i0 or j1 < j0 or k1 < k0:
        return np.array([], dtype=np.int64)

    ii, jj, kk = np.meshgrid(
        np.arange(i0, i1 + 1, dtype=float),
        np.arange(j0, j1 + 1, dtype=float),
        np.arange(k0, k1 + 1, dtype=float),
        indexing="ij",
    )
    ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
    hom_ijk = np.c_[ijk, np.ones((ijk.shape[0], 1), dtype=float)]
    world = hom_ijk @ np.asarray(affine, dtype=float).T
    displacement = world[:, :3] - center_mni.reshape(1, 3)
    distance = np.linalg.norm(displacement, axis=1)
    inside = distance <= float(radius_mm)

    if not np.any(inside):
        return np.array([], dtype=np.int64)

    ijk_inside = ijk[inside]
    lin = (
        ijk_inside[:, 0].astype(np.int64) * (shape[1] * shape[2])
        + ijk_inside[:, 1].astype(np.int64) * shape[2]
        + ijk_inside[:, 2].astype(np.int64)
    )
    return np.unique(lin)


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Compute network ROI overlap QC.")
    parser.add_argument(
        "--primary-peaks",
        type=Path,
        default=FINAL_ROOT / "roi_management" / "meta_roi_peaks_curated.tsv",
        help="TSV of primary meta ROI peaks (meta_metaphor + meta_spatial).",
    )
    parser.add_argument(
        "--network-peaks",
        type=Path,
        nargs="+",
        default=[
            FINAL_ROOT / "roi_management" / "language_network_peaks.tsv",
            FINAL_ROOT / "roi_management" / "tom_network_peaks.tsv",
        ],
        help="TSVs of network ROI peaks (language_network / tom_network).",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=base_dir / "pattern_root" / "sub-01" / "pre_yy.nii.gz",
        help="Reference 3D/4D image used to define the voxel grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "paper_outputs" / "qc" / "roi_library" / "network_overlap_qc.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest. Defaults to sibling .json of --output.",
    )
    args = parser.parse_args()

    primary = _read_peaks(args.primary_peaks)
    network_frames = [_read_peaks(path) for path in args.network_peaks]
    network = pd.concat(network_frames, ignore_index=True, sort=False)

    primary_sets = sorted(primary["roi_set"].astype(str).unique().tolist())
    network_sets = sorted(network["roi_set"].astype(str).unique().tolist())

    if not args.reference_image.exists():
        raise FileNotFoundError(
            f"Reference image not found: {args.reference_image}. "
            "Pass --reference-image to a 3D/4D NIfTI on the target machine."
        )

    img = nib.load(str(args.reference_image))
    shape = tuple(int(v) for v in img.shape[:3])
    affine = np.asarray(img.affine, dtype=float)
    inv_affine = np.linalg.inv(affine)
    voxel_sizes = _voxel_sizes_mm(affine)

    rows: list[dict[str, object]] = []
    # Precompute sphere voxel index sets for speed.
    primary_spheres: dict[tuple[str, str], np.ndarray] = {}
    network_spheres: dict[tuple[str, str], np.ndarray] = {}

    for _, pri_row in primary.iterrows():
        pri_center = np.asarray([pri_row["x"], pri_row["y"], pri_row["z"]], dtype=float)
        pri_radius = float(pri_row["radius_mm"])
        key = (str(pri_row["roi_set"]), str(pri_row["roi_name"]))
        primary_spheres[key] = _sphere_linear_indices(
            pri_center, pri_radius, affine, shape, inv_affine, voxel_sizes
        )

    for _, net_row in network.iterrows():
        net_center = np.asarray([net_row["x"], net_row["y"], net_row["z"]], dtype=float)
        net_radius = float(net_row["radius_mm"])
        key = (str(net_row["roi_set"]), str(net_row["roi_name"]))
        network_spheres[key] = _sphere_linear_indices(
            net_center, net_radius, affine, shape, inv_affine, voxel_sizes
        )

    for _, net_row in network.iterrows():
        net_center = np.asarray(
            [net_row["x"], net_row["y"], net_row["z"]], dtype=float
        )
        net_key = (str(net_row["roi_set"]), str(net_row["roi_name"]))
        net_lin = network_spheres.get(net_key, np.array([], dtype=np.int64))
        net_voxels = int(net_lin.size)

        for _, pri_row in primary.iterrows():
            pri_center = np.asarray(
                [pri_row["x"], pri_row["y"], pri_row["z"]], dtype=float
            )
            pri_key = (str(pri_row["roi_set"]), str(pri_row["roi_name"]))
            pri_lin = primary_spheres.get(pri_key, np.array([], dtype=np.int64))
            pri_voxels = int(pri_lin.size)

            if net_voxels == 0 or pri_voxels == 0:
                inter = 0
                union = int(net_voxels + pri_voxels)
            else:
                inter = int(np.intersect1d(net_lin, pri_lin, assume_unique=True).size)
                union = int(net_voxels + pri_voxels - inter)
            distance_mm = float(np.linalg.norm(net_center - pri_center))

            rows.append(
                {
                    "network_roi_set": net_row["roi_set"],
                    "network_roi_name": net_row["roi_name"],
                    "network_x": float(net_center[0]),
                    "network_y": float(net_center[1]),
                    "network_z": float(net_center[2]),
                    "primary_roi_set": pri_row["roi_set"],
                    "primary_roi_name": pri_row["roi_name"],
                    "primary_x": float(pri_center[0]),
                    "primary_y": float(pri_center[1]),
                    "primary_z": float(pri_center[2]),
                    "euclidean_distance_mm": distance_mm,
                    "network_n_voxels": net_voxels,
                    "primary_n_voxels": pri_voxels,
                    "intersection_voxels": inter,
                    "union_voxels": union,
                    "iou_ratio": float(inter / union) if union > 0 else float("nan"),
                    "overlap_ratio_to_network": (
                        float(inter / net_voxels) if net_voxels > 0 else float("nan")
                    ),
                    "overlap_ratio_to_primary": (
                        float(inter / pri_voxels) if pri_voxels > 0 else float("nan")
                    ),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["network_roi_set", "network_roi_name", "primary_roi_set", "primary_roi_name"]
    )
    ensure_dir(args.output.parent)
    write_table(frame, args.output)

    manifest_path = args.manifest or args.output.with_suffix(".json")
    save_json(
        {
            "primary_peaks": str(args.primary_peaks),
            "network_peaks": [str(p) for p in args.network_peaks],
            "reference_image": str(args.reference_image),
            "primary_roi_sets": primary_sets,
            "network_roi_sets": network_sets,
            "n_rows": int(len(frame)),
            "n_voxels_reference_grid": int(math.prod(shape)),
            "output": str(args.output),
            "note": "Sphere geometry matches build_meta_roi_library.py (radius_mm from peak table).",
        },
        manifest_path,
    )


if __name__ == "__main__":
    main()
