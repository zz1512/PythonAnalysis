#!/usr/bin/env python3
"""
Build externally defined meta-analysis ROI masks from MNI peak coordinates.

The input is a TSV table curated from external meta-analysis sources such as
Neurosynth, NeuroQuery, or published ALE/MACM papers. Each row specifies one
MNI peak. Rows with the same roi_set + roi_name are merged into one mask.
"""

from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import sys

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


BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
DEFAULT_OUTPUT_DIR = BASE_DIR / "roi_library"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.tsv"
DEFAULT_PEAKS = DEFAULT_OUTPUT_DIR / "meta_sources" / "meta_roi_peaks.tsv"
DEFAULT_REFERENCE = BASE_DIR / "pattern_root" / "sub-01" / "pre_yy.nii.gz"
DEFAULT_RADIUS_MM = 6.0
REQUIRED_COLUMNS = {
    "roi_set",
    "roi_name",
    "domain",
    "meta_source",
    "term_or_query",
    "map_type",
    "hemisphere",
    "x",
    "y",
    "z",
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_reference(path: Path) -> tuple[tuple[int, int, int], np.ndarray, nib.Nifti1Header]:
    if not path.exists():
        raise FileNotFoundError(f"Missing reference image: {path}")
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape[:3])
    header = img.header.copy()
    header.set_data_shape(shape)
    return shape, np.asarray(img.affine, dtype=float), header


def _world_grid(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    ijk = np.indices(shape, dtype=float).reshape(3, -1).T
    hom = np.c_[ijk, np.ones(len(ijk), dtype=float)]
    return hom @ affine.T


def _sphere_mask(
    xyz_world: np.ndarray,
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    center_arr = np.asarray(center, dtype=float)
    distances = np.linalg.norm(xyz_world[:, :3] - center_arr[None, :], axis=1)
    return (distances <= float(radius_mm)).reshape(shape)


def _normalize_peaks(peaks: pd.DataFrame, roi_sets: list[str] | None) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(peaks.columns)
    if missing:
        raise ValueError(f"Meta ROI peaks table missing columns: {sorted(missing)}")
    out = peaks.copy()
    for col in ["roi_set", "roi_name", "domain", "meta_source", "term_or_query", "map_type", "hemisphere"]:
        out[col] = out[col].astype(str).str.strip()
    for col in ["x", "y", "z"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "radius_mm" not in out.columns:
        out["radius_mm"] = DEFAULT_RADIUS_MM
    out["radius_mm"] = pd.to_numeric(out["radius_mm"], errors="coerce").fillna(DEFAULT_RADIUS_MM)
    out = out.dropna(subset=["x", "y", "z"]).copy()
    if roi_sets:
        wanted = {str(item).strip() for item in roi_sets}
        out = out[out["roi_set"].isin(wanted)].copy()
    if out.empty:
        raise RuntimeError("No usable meta ROI peak rows after filtering.")
    return out.reset_index(drop=True)


def _join_unique(values: pd.Series) -> str:
    items = [str(v).strip() for v in values.dropna().tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
    return "|".join(dict.fromkeys(items))


def build_masks(
    peaks: pd.DataFrame,
    *,
    output_dir: Path,
    reference_image: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape, affine, header = _load_reference(reference_image)
    xyz_world = _world_grid(shape, affine)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for (roi_set, roi_name), group in peaks.groupby(["roi_set", "roi_name"], sort=False):
        roi_dir = ensure_dir(output_dir / "masks" / str(roi_set))
        mask = np.zeros(shape, dtype=bool)
        peak_strings: list[str] = []
        for row in group.itertuples(index=False):
            center = (float(row.x), float(row.y), float(row.z))
            radius = float(row.radius_mm)
            component = _sphere_mask(xyz_world, shape, center, radius)
            mask |= component
            peak_strings.append(f"{center[0]:.1f},{center[1]:.1f},{center[2]:.1f};r={radius:.1f}")
            audit_rows.append(
                {
                    "roi_set": roi_set,
                    "roi_name": roi_name,
                    "peak_mni": peak_strings[-1],
                    "peak_voxels": int(component.sum()),
                    "status": "ok" if int(component.sum()) > 0 else "empty",
                    "meta_source": getattr(row, "meta_source", ""),
                    "term_or_query": getattr(row, "term_or_query", ""),
                    "map_type": getattr(row, "map_type", ""),
                    "source_url": getattr(row, "source_url", ""),
                }
            )
        mask_path = roi_dir / f"{roi_name}.nii.gz"
        image = nib.Nifti1Image(mask.astype(np.uint8), affine, header=header)
        nib.save(image, str(mask_path))
        voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
        rows.append(
            {
                "roi_name": roi_name,
                "roi_set": roi_set,
                "source_type": "meta_analysis_peak_sphere",
                "mask_path": str(mask_path),
                "hemisphere": _join_unique(group["hemisphere"]),
                "base_contrast": "",
                "atlas_name": "",
                "atlas_label": "",
                "theory_role": _join_unique(group["theory_role"]) if "theory_role" in group.columns else _join_unique(group["domain"]),
                "include_in_main": True,
                "include_in_rsa": True,
                "include_in_mvpa": True,
                "include_in_rd": True,
                "include_in_gps": True,
                "n_voxels": int(mask.sum()),
                "volume_mm3": float(mask.sum() * voxel_volume),
                "notes": (
                    "Externally defined meta-analysis ROI; "
                    f"source={_join_unique(group['meta_source'])}; "
                    f"term={_join_unique(group['term_or_query'])}; "
                    f"map={_join_unique(group['map_type'])}; "
                    f"peaks={';'.join(peak_strings)}"
                ),
                "meta_source": _join_unique(group["meta_source"]),
                "term_or_query": _join_unique(group["term_or_query"]),
                "map_type": _join_unique(group["map_type"]),
                "source_url": _join_unique(group["source_url"]) if "source_url" in group.columns else "",
                "download_date": _join_unique(group["download_date"]) if "download_date" in group.columns else str(date.today()),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def update_manifest(manifest_path: Path, new_rows: pd.DataFrame, replace_roi_sets: list[str]) -> pd.DataFrame:
    if manifest_path.exists():
        old = _read_table(manifest_path)
    else:
        old = pd.DataFrame()
    if not old.empty and replace_roi_sets:
        old = old[~old["roi_set"].astype(str).isin(set(replace_roi_sets))].copy()
    all_columns = list(dict.fromkeys([*old.columns.tolist(), *new_rows.columns.tolist()]))
    if old.empty:
        out = new_rows.reindex(columns=all_columns)
    else:
        out = pd.concat([old.reindex(columns=all_columns), new_rows.reindex(columns=all_columns)], ignore_index=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build meta-analysis ROI masks from MNI peak coordinates.")
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roi-sets", nargs="+", default=["meta_metaphor", "meta_spatial"])
    parser.add_argument("--no-replace-existing", action="store_true")
    args = parser.parse_args()

    peaks = _normalize_peaks(_read_table(args.peaks), args.roi_sets)
    output_dir = ensure_dir(args.output_dir)
    meta_dir = ensure_dir(output_dir / "meta_sources")
    new_rows, audit = build_masks(peaks, output_dir=output_dir, reference_image=args.reference_image)
    if (new_rows["n_voxels"] <= 0).any():
        bad = new_rows.loc[new_rows["n_voxels"] <= 0, ["roi_set", "roi_name"]].to_dict("records")
        raise RuntimeError(f"Some meta ROI masks are empty: {bad}")

    replace_sets = [] if args.no_replace_existing else list(args.roi_sets)
    manifest = update_manifest(args.manifest, new_rows, replace_sets)
    write_table(manifest, args.manifest)
    write_table(new_rows, meta_dir / "meta_roi_manifest_rows.tsv")
    write_table(audit, meta_dir / "meta_roi_source_audit.tsv")
    save_json(
        {
            "peaks": str(args.peaks),
            "reference_image": str(args.reference_image),
            "output_dir": str(args.output_dir),
            "manifest": str(args.manifest),
            "roi_sets": list(args.roi_sets),
            "n_input_peaks": int(len(peaks)),
            "n_roi_masks": int(len(new_rows)),
        },
        meta_dir / "meta_roi_build_manifest.json",
    )
    print(f"[meta-roi] wrote {len(new_rows)} masks and updated {args.manifest}")


if __name__ == "__main__":
    main()
