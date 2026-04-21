from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import load_img
from scipy import ndimage


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table


BASE_DIR = Path(r"E:\python_metaphor")
DEFAULT_OUTPUT_DIR = BASE_DIR / "roi_library"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.tsv"


FUNCTIONAL_SPECS = [
    {
        "contrast": "Metaphor_gt_Spatial",
        "mask_path": BASE_DIR / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Metaphor_gt_Spatial/Metaphor_gt_Spatial_roi_mask.nii.gz",
        "cluster_report": BASE_DIR / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Metaphor_gt_Spatial/cluster_report.csv",
    },
    {
        "contrast": "Spatial_gt_Metaphor",
        "mask_path": BASE_DIR / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Spatial_gt_Metaphor/Spatial_gt_Metaphor_roi_mask.nii.gz",
        "cluster_report": BASE_DIR / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Spatial_gt_Metaphor/cluster_report.csv",
    },
]


LITERATURE_ROIS = [
    {
        "roi_name": "lit_L_temporal_pole",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Temporal Pole"],
        "hemisphere": "L",
        "theory_role": "core_semantic_hub",
    },
    {
        "roi_name": "lit_L_IFG",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": [
            "Inferior Frontal Gyrus, pars triangularis",
            "Inferior Frontal Gyrus, pars opercularis",
        ],
        "hemisphere": "L",
        "theory_role": "semantic_control",
    },
    {
        "roi_name": "lit_L_AG",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Angular Gyrus"],
        "hemisphere": "L",
        "theory_role": "combinatorial_semantics",
    },
    {
        "roi_name": "lit_L_pMTG",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Middle Temporal Gyrus, posterior division"],
        "hemisphere": "L",
        "theory_role": "lexical_semantics",
    },
    {
        "roi_name": "lit_L_pSTS",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Superior Temporal Gyrus, posterior division"],
        "hemisphere": "L",
        "theory_role": "comprehension_interface",
    },
    {
        "roi_name": "lit_R_pSTS",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Superior Temporal Gyrus, posterior division"],
        "hemisphere": "R",
        "theory_role": "novelty_sensitivity",
    },
    {
        "roi_name": "lit_L_precuneus",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Precuneous Cortex"],
        "hemisphere": "L",
        "theory_role": "imagery_context",
    },
]


ATLAS_ROBUSTNESS_ROIS = [
    {
        "roi_name": "atlas_L_temporal_pole",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Temporal_Pole_Mid_L"],
        "hemisphere": "L",
        "theory_role": "robustness_semantic_hub",
    },
    {
        "roi_name": "atlas_L_IFGop",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Frontal_Inf_Oper_L"],
        "hemisphere": "L",
        "theory_role": "robustness_semantic_control",
    },
    {
        "roi_name": "atlas_L_IFGtri",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Frontal_Inf_Tri_L"],
        "hemisphere": "L",
        "theory_role": "robustness_semantic_control",
    },
    {
        "roi_name": "atlas_L_AG",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Angular_L"],
        "hemisphere": "L",
        "theory_role": "robustness_combinatorial_semantics",
    },
    {
        "roi_name": "atlas_L_pMTG",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Temporal_Mid_L"],
        "hemisphere": "L",
        "theory_role": "robustness_lexical_semantics",
    },
    {
        "roi_name": "atlas_L_precuneus",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Precuneus_L"],
        "hemisphere": "L",
        "theory_role": "robustness_spatial_imagery",
    },
    {
        "roi_name": "atlas_R_precuneus",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Precuneus_R"],
        "hemisphere": "R",
        "theory_role": "robustness_spatial_imagery",
    },
    {
        "roi_name": "atlas_R_parahippocampal",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["ParaHippocampal_R"],
        "hemisphere": "R",
        "theory_role": "robustness_scene_memory",
    },
    {
        "roi_name": "atlas_L_fusiform",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Fusiform_L"],
        "hemisphere": "L",
        "theory_role": "robustness_visual_form",
    },
]


def sanitize_name(text: object) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "roi"


def infer_hemisphere_from_x(x_coord: float) -> str:
    if x_coord < 0:
        return "L"
    if x_coord > 0:
        return "R"
    return "M"


def save_mask(mask: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> int:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    data = mask.astype(np.uint8)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(data, reference_img.affine, header), str(output_path))
    return int(data.sum())


def world_to_voxel(affine: np.ndarray, xyz: tuple[float, float, float]) -> tuple[int, int, int]:
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), np.asarray(xyz, dtype=float))
    rounded = np.rint(ijk).astype(int)
    return int(rounded[0]), int(rounded[1]), int(rounded[2])


def nearest_component_id(
    labeled: np.ndarray,
    affine: np.ndarray,
    xyz: tuple[float, float, float],
) -> int:
    coords = np.argwhere(labeled > 0)
    if coords.size == 0:
        return 0
    xyz_arr = np.asarray(xyz, dtype=float)
    world_coords = nib.affines.apply_affine(affine, coords)
    distances = np.linalg.norm(world_coords - xyz_arr[None, :], axis=1)
    closest = coords[int(np.argmin(distances))]
    return int(labeled[tuple(closest)])


def build_functional_rois(mask_root: Path) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []
    functional_dir = ensure_dir(mask_root / "main_functional")
    qc_dir = ensure_dir(mask_root.parent / "qc")

    functional_qc = []
    for spec in FUNCTIONAL_SPECS:
        mask_path = spec["mask_path"]
        if not mask_path.exists():
            continue
        mask_img = nib.load(str(mask_path))
        mask_data = np.asarray(mask_img.get_fdata()) > 0
        labeled, _ = ndimage.label(mask_data)

        used_components: set[int] = set()
        cluster_report_path = spec["cluster_report"]
        if cluster_report_path.exists():
            cluster_frame = pd.read_csv(cluster_report_path)
            for row in cluster_frame.to_dict(orient="records"):
                cluster_id = int(row["Cluster ID"])
                xyz = (float(row["X"]), float(row["Y"]), float(row["Z"]))
                voxel = world_to_voxel(mask_img.affine, xyz)
                component_id = 0
                if all(0 <= voxel[idx] < labeled.shape[idx] for idx in range(3)):
                    component_id = int(labeled[voxel])
                if component_id <= 0:
                    component_id = nearest_component_id(labeled, mask_img.affine, xyz)
                if component_id <= 0 or component_id in used_components:
                    continue

                component_mask = labeled == component_id
                region_label = sanitize_name(row.get("Region", f"cluster_{cluster_id:02d}"))
                roi_name = f"func_{sanitize_name(spec['contrast'])}_c{cluster_id:02d}_{region_label}"
                output_path = functional_dir / f"{roi_name}.nii.gz"
                voxel_count = save_mask(component_mask, mask_img, output_path)
                used_components.add(component_id)
                hemisphere = infer_hemisphere_from_x(float(row["X"]))

                functional_qc.append(
                    {
                        "roi_name": roi_name,
                        "contrast": spec["contrast"],
                        "cluster_id": cluster_id,
                        "region": row.get("Region", ""),
                        "x": float(row["X"]),
                        "y": float(row["Y"]),
                        "z": float(row["Z"]),
                        "voxel_count": voxel_count,
                    }
                )
                manifest_rows.append(
                    {
                        "roi_name": roi_name,
                        "roi_set": "main_functional",
                        "source_type": "glm_cluster",
                        "mask_path": str(output_path.resolve()),
                        "hemisphere": hemisphere,
                        "base_contrast": spec["contrast"],
                        "atlas_name": "",
                        "atlas_label": row.get("Region", ""),
                        "theory_role": "data_driven_main_roi",
                        "include_in_main": True,
                        "include_in_rsa": True,
                        "include_in_mvpa": True,
                        "include_in_rd": True,
                        "include_in_gps": True,
                        "notes": f"Derived from {spec['contrast']} cluster {cluster_id}.",
                    }
                )

        remaining_components = sorted(set(np.unique(labeled)) - {0} - used_components)
        for component_index, component_id in enumerate(remaining_components, start=1):
            component_mask = labeled == component_id
            roi_name = f"func_{sanitize_name(spec['contrast'])}_extra_{component_index:02d}"
            output_path = functional_dir / f"{roi_name}.nii.gz"
            voxel_count = save_mask(component_mask, mask_img, output_path)
            manifest_rows.append(
                {
                    "roi_name": roi_name,
                    "roi_set": "main_functional",
                    "source_type": "glm_cluster",
                    "mask_path": str(output_path.resolve()),
                    "hemisphere": "U",
                    "base_contrast": spec["contrast"],
                    "atlas_name": "",
                    "atlas_label": "",
                    "theory_role": "data_driven_main_roi",
                    "include_in_main": False,
                    "include_in_rsa": True,
                    "include_in_mvpa": True,
                    "include_in_rd": True,
                    "include_in_gps": True,
                    "notes": f"Unmatched connected component from {spec['contrast']}; kept for completeness.",
                }
            )
            functional_qc.append(
                {
                    "roi_name": roi_name,
                    "contrast": spec["contrast"],
                    "cluster_id": "",
                    "region": "extra_component",
                    "x": np.nan,
                    "y": np.nan,
                    "z": np.nan,
                    "voxel_count": voxel_count,
                }
            )

    if functional_qc:
        write_table(pd.DataFrame(functional_qc), qc_dir / "functional_roi_qc.tsv")
    return manifest_rows


def atlas_label_lookup(labels: list[str] | tuple[str, ...]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, label in enumerate(labels):
        lookup[str(label)] = index
    return lookup


def match_label_indices(
    label_to_index: dict[str, int],
    requested_labels: list[str],
    *,
    hemisphere: str | None = None,
) -> list[int]:
    matched: list[int] = []
    for requested in requested_labels:
        exact = label_to_index.get(requested)
        if exact is not None:
            matched.append(int(exact))
            continue

        requested_lower = requested.lower()
        candidates = []
        for label, index in label_to_index.items():
            label_lower = label.lower()
            if requested_lower in label_lower:
                if hemisphere == "L" and "left" not in label_lower and not label_lower.endswith("_l"):
                    continue
                if hemisphere == "R" and "right" not in label_lower and not label_lower.endswith("_r"):
                    continue
                candidates.append(int(index))
        if not candidates:
            raise ValueError(f"Atlas label not found: {requested}")
        matched.extend(candidates)
    return sorted(set(matched))


def create_atlas_mask(
    atlas_img: nib.Nifti1Image,
    atlas_data: np.ndarray,
    label_to_index: dict[str, int],
    labels: list[str],
    *,
    hemisphere: str | None = None,
) -> np.ndarray:
    indices = match_label_indices(label_to_index, labels, hemisphere=hemisphere)
    return np.isin(atlas_data, indices)


def build_harvard_oxford_rois(mask_root: Path) -> list[dict[str, object]]:
    output_dir = ensure_dir(mask_root / "literature")
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = load_img(atlas["maps"])
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=int)
    label_to_index = atlas_label_lookup(atlas["labels"])

    manifest_rows: list[dict[str, object]] = []
    for spec in LITERATURE_ROIS:
        mask = create_atlas_mask(
            atlas_img,
            atlas_data,
            label_to_index,
            spec["labels"],
            hemisphere=spec["hemisphere"],
        )
        output_path = output_dir / f"{spec['roi_name']}.nii.gz"
        save_mask(mask, atlas_img, output_path)
        manifest_rows.append(
            {
                "roi_name": spec["roi_name"],
                "roi_set": "literature",
                "source_type": spec["atlas_type"],
                "mask_path": str(output_path.resolve()),
                "hemisphere": spec["hemisphere"],
                "base_contrast": "",
                "atlas_name": spec["atlas_name"],
                "atlas_label": "|".join(spec["labels"]),
                "theory_role": spec["theory_role"],
                "include_in_main": True,
                "include_in_rsa": True,
                "include_in_mvpa": True,
                "include_in_rd": True,
                "include_in_gps": True,
                "notes": "Literature-guided ROI built from Harvard-Oxford atlas labels.",
            }
        )
    return manifest_rows


def build_aal_rois(mask_root: Path) -> list[dict[str, object]]:
    output_dir = ensure_dir(mask_root / "atlas_robustness")
    atlas = datasets.fetch_atlas_aal()
    atlas_img = load_img(atlas["maps"])
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=int)
    label_to_index = {str(label): int(index) for label, index in zip(atlas["labels"], atlas["indices"])}

    manifest_rows: list[dict[str, object]] = []
    for spec in ATLAS_ROBUSTNESS_ROIS:
        indices = match_label_indices(label_to_index, spec["labels"], hemisphere=spec["hemisphere"])
        mask = np.isin(atlas_data, indices)
        output_path = output_dir / f"{spec['roi_name']}.nii.gz"
        save_mask(mask, atlas_img, output_path)
        manifest_rows.append(
            {
                "roi_name": spec["roi_name"],
                "roi_set": "atlas_robustness",
                "source_type": spec["atlas_type"],
                "mask_path": str(output_path.resolve()),
                "hemisphere": spec["hemisphere"],
                "base_contrast": "",
                "atlas_name": spec["atlas_name"],
                "atlas_label": "|".join(spec["labels"]),
                "theory_role": spec["theory_role"],
                "include_in_main": False,
                "include_in_rsa": True,
                "include_in_mvpa": True,
                "include_in_rd": True,
                "include_in_gps": True,
                "notes": "Atlas robustness ROI built from AAL atlas labels.",
            }
        )
    return manifest_rows


def summarize_manifest(manifest: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        manifest.groupby("roi_set")["roi_name"]
        .count()
        .rename("n_roi")
        .reset_index()
    )
    write_table(summary, output_dir / "qc" / "roi_set_summary.tsv")
    payload = {
        "output_dir": str(output_dir.resolve()),
        "n_total_roi": int(len(manifest)),
        "roi_sets": summary.to_dict(orient="records"),
    }
    save_json(payload, output_dir / "qc" / "roi_library_summary.json")


def build_roi_library(output_dir: Path, manifest_path: Path) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "masks")
    ensure_dir(output_dir / "qc")

    rows: list[dict[str, object]] = []
    rows.extend(build_functional_rois(output_dir / "masks"))
    rows.extend(build_harvard_oxford_rois(output_dir / "masks"))
    rows.extend(build_aal_rois(output_dir / "masks"))

    manifest = pd.DataFrame(rows).sort_values(["roi_set", "roi_name"]).reset_index(drop=True)
    write_table(manifest, manifest_path)
    summarize_manifest(manifest, output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable ROI library for RSA/RD/GPS/MVPA.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"ROI library root (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, help=f"Manifest output path (default: {DEFAULT_MANIFEST_PATH}).")
    args = parser.parse_args()

    manifest = build_roi_library(args.output_dir, args.manifest_path)
    print(f"Built ROI library with {len(manifest)} ROI masks -> {args.manifest_path}")


if __name__ == "__main__":
    main()
