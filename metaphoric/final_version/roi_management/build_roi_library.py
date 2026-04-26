from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import load_img, resample_to_img
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


# 数据根目录：与 glm_config.py 保持一致，从 PYTHON_METAPHOR_ROOT 读，兜底 Windows 原路径。
# 例：macOS/Linux  export PYTHON_METAPHOR_ROOT=/path/to/python_metaphor
#    Windows      $env:PYTHON_METAPHOR_ROOT="E:/python_metaphor"
BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", r"E:\python_metaphor"))
DEFAULT_OUTPUT_DIR = BASE_DIR / "roi_library"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.tsv"

# 功能 ROI：peak 映射到连通分量时，如果峰值体素落到背景，
# 允许用最近非零 component 兜底，但若最近距离 > 该阈值（mm）则判为孤儿峰并跳过。
PEAK_MAX_DISTANCE_MM = 8.0

# Atlas ROI 最小体素数阈值，避免过小 ROI 混入主分析。
MIN_ATLAS_VOXELS = 50

# pattern_root 用于 affine/shape 对齐 QC（默认读环境变量，找不到则跳过）。
DEFAULT_PATTERN_ROOT = Path(os.environ.get("PATTERN_ROOT", str(BASE_DIR / "pattern_root")))


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
    # --- 右半球镜像 ROI（新增）：为 lateralization / hemisphere × condition 交互提供对称锚点 ---
    {
        "roi_name": "lit_R_temporal_pole",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Temporal Pole"],
        "hemisphere": "R",
        "theory_role": "lateralization_mirror_semantic_hub",
    },
    {
        "roi_name": "lit_R_IFG",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": [
            "Inferior Frontal Gyrus, pars triangularis",
            "Inferior Frontal Gyrus, pars opercularis",
        ],
        "hemisphere": "R",
        "theory_role": "lateralization_mirror_semantic_control",
    },
    {
        "roi_name": "lit_R_AG",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Angular Gyrus"],
        "hemisphere": "R",
        "theory_role": "lateralization_mirror_combinatorial_semantics",
    },
    {
        "roi_name": "lit_R_pMTG",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Middle Temporal Gyrus, posterior division"],
        "hemisphere": "R",
        "theory_role": "lateralization_mirror_lexical_semantics",
    },
    {
        "roi_name": "lit_R_precuneus",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Precuneous Cortex"],
        "hemisphere": "R",
        "theory_role": "lateralization_mirror_imagery_context",
    },
]


LITERATURE_SPATIAL_ROIS = [
    {
        "roi_name": "litspat_L_PPA",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Parahippocampal Gyrus, posterior division"],
        "hemisphere": "L",
        "theory_role": "scene_landmark_processing",
        "notes": "PPA proxy from Harvard-Oxford parahippocampal posterior division.",
    },
    {
        "roi_name": "litspat_R_PPA",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Parahippocampal Gyrus, posterior division"],
        "hemisphere": "R",
        "theory_role": "scene_landmark_processing",
        "notes": "PPA proxy from Harvard-Oxford parahippocampal posterior division.",
    },
    {
        "roi_name": "litspat_L_OPA",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Lateral Occipital Cortex, superior division"],
        "hemisphere": "L",
        "theory_role": "environmental_boundaries_proxy",
        "notes": "OPA proxy from lateral occipital superior division; atlas lacks a dedicated OPA parcel.",
    },
    {
        "roi_name": "litspat_R_OPA",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Lateral Occipital Cortex, superior division"],
        "hemisphere": "R",
        "theory_role": "environmental_boundaries_proxy",
        "notes": "OPA proxy from lateral occipital superior division; atlas lacks a dedicated OPA parcel.",
    },
    {
        "roi_name": "litspat_L_precuneus",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Precuneous Cortex"],
        "hemisphere": "L",
        "theory_role": "egocentric_spatial_imagery",
        "notes": "Medial parietal spatial imagery / self-centered spatial representation ROI.",
    },
    {
        "roi_name": "litspat_R_precuneus",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Precuneous Cortex"],
        "hemisphere": "R",
        "theory_role": "egocentric_spatial_imagery",
        "notes": "Medial parietal spatial imagery / self-centered spatial representation ROI.",
    },
    {
        "roi_name": "litspat_L_PPC",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Superior Parietal Lobule"],
        "hemisphere": "L",
        "theory_role": "posterior_parietal_spatial_attention_proxy",
        "notes": "Posterior parietal / IPS-adjacent proxy from superior parietal lobule.",
    },
    {
        "roi_name": "litspat_R_PPC",
        "atlas_type": "harvard_oxford",
        "atlas_name": "cort-maxprob-thr25-2mm",
        "labels": ["Superior Parietal Lobule"],
        "hemisphere": "R",
        "theory_role": "posterior_parietal_spatial_attention_proxy",
        "notes": "Posterior parietal / IPS-adjacent proxy from superior parietal lobule.",
    },
    {
        "roi_name": "litspat_L_RSC",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Cingulum_Post_L"],
        "hemisphere": "L",
        "theory_role": "reference_frame_transformation_proxy",
        "notes": "RSC proxy from AAL posterior cingulate/cingulum parcel.",
    },
    {
        "roi_name": "litspat_R_RSC",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Cingulum_Post_R"],
        "hemisphere": "R",
        "theory_role": "reference_frame_transformation_proxy",
        "notes": "RSC proxy from AAL posterior cingulate/cingulum parcel.",
    },
    {
        "roi_name": "litspat_L_hippocampus",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Hippocampus_L"],
        "hemisphere": "L",
        "theory_role": "cognitive_map_memory",
        "notes": "Anatomical hippocampus ROI for spatial cognitive-map coding.",
    },
    {
        "roi_name": "litspat_R_hippocampus",
        "atlas_type": "aal",
        "atlas_name": "aal",
        "labels": ["Hippocampus_R"],
        "hemisphere": "R",
        "theory_role": "cognitive_map_memory",
        "notes": "Anatomical hippocampus ROI for spatial cognitive-map coding.",
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


def save_mask(mask: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> tuple[int, float]:
    """
    保存二值 mask，并返回 (voxel_count, volume_mm3)。
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    data = mask.astype(np.uint8)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(data, reference_img.affine, header), str(output_path))
    voxel_count = int(data.sum())
    zooms = reference_img.header.get_zooms()[:3]
    volume_mm3 = float(voxel_count * float(np.prod(zooms)))
    return voxel_count, volume_mm3


def save_mask_image(mask_img: nib.Nifti1Image, output_path: Path) -> tuple[int, float]:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    data = (np.asarray(mask_img.get_fdata()) > 0).astype(np.uint8)
    header = mask_img.header.copy()
    header.set_data_dtype(np.uint8)
    clean_img = nib.Nifti1Image(data, mask_img.affine, header)
    nib.save(clean_img, str(output_path))
    voxel_count = int(data.sum())
    zooms = clean_img.header.get_zooms()[:3]
    volume_mm3 = float(voxel_count * float(np.prod(zooms)))
    return voxel_count, volume_mm3


def world_to_voxel(affine: np.ndarray, xyz: tuple[float, float, float]) -> tuple[int, int, int]:
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), np.asarray(xyz, dtype=float))
    rounded = np.rint(ijk).astype(int)
    return int(rounded[0]), int(rounded[1]), int(rounded[2])


def nearest_component_id(
    labeled: np.ndarray,
    affine: np.ndarray,
    xyz: tuple[float, float, float],
) -> tuple[int, float]:
    """返回最近非零连通分量的 (component_id, distance_mm)；若不存在则 (0, inf)。"""
    coords = np.argwhere(labeled > 0)
    if coords.size == 0:
        return 0, float("inf")
    xyz_arr = np.asarray(xyz, dtype=float)
    world_coords = nib.affines.apply_affine(affine, coords)
    distances = np.linalg.norm(world_coords - xyz_arr[None, :], axis=1)
    min_idx = int(np.argmin(distances))
    closest = coords[min_idx]
    return int(labeled[tuple(closest)]), float(distances[min_idx])


def build_functional_rois(mask_root: Path) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []
    functional_dir = ensure_dir(mask_root / "main_functional")
    qc_dir = ensure_dir(mask_root.parent / "qc")

    functional_qc: list[dict[str, object]] = []
    orphan_peaks: list[dict[str, object]] = []
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
                peak_distance_mm = 0.0
                peak_in_component = False
                if all(0 <= voxel[idx] < labeled.shape[idx] for idx in range(3)):
                    component_id = int(labeled[voxel])
                    if component_id > 0:
                        peak_in_component = True
                if component_id <= 0:
                    # 峰值落到背景：用最近 component 兜底，但超过距离阈值判为孤儿峰
                    nearest_id, nearest_dist = nearest_component_id(labeled, mask_img.affine, xyz)
                    if nearest_id > 0 and nearest_dist <= PEAK_MAX_DISTANCE_MM:
                        component_id = nearest_id
                        peak_distance_mm = nearest_dist
                    else:
                        orphan_peaks.append({
                            "contrast": spec["contrast"],
                            "cluster_id": cluster_id,
                            "x": xyz[0], "y": xyz[1], "z": xyz[2],
                            "reason": "peak_outside_mask_and_no_nearby_component"
                                     if nearest_id == 0
                                     else f"peak_too_far_from_nearest_component_{nearest_dist:.2f}mm",
                            "nearest_component_distance_mm": nearest_dist,
                        })
                        continue
                if component_id <= 0 or component_id in used_components:
                    if component_id in used_components:
                        orphan_peaks.append({
                            "contrast": spec["contrast"],
                            "cluster_id": cluster_id,
                            "x": xyz[0], "y": xyz[1], "z": xyz[2],
                            "reason": "component_already_claimed_by_another_peak",
                            "nearest_component_distance_mm": peak_distance_mm,
                        })
                    continue

                component_mask = labeled == component_id
                region_label = sanitize_name(row.get("Region", f"cluster_{cluster_id:02d}"))
                roi_name = f"func_{sanitize_name(spec['contrast'])}_c{cluster_id:02d}_{region_label}"
                output_path = functional_dir / f"{roi_name}.nii.gz"
                voxel_count, volume_mm3 = save_mask(component_mask, mask_img, output_path)
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
                        "volume_mm3": volume_mm3,
                        "peak_in_component": peak_in_component,
                        "peak_to_component_distance_mm": peak_distance_mm,
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
                        "n_voxels": voxel_count,
                        "volume_mm3": volume_mm3,
                        "notes": f"Derived from {spec['contrast']} cluster {cluster_id}.",
                    }
                )

        remaining_components = sorted(set(np.unique(labeled)) - {0} - used_components)
        for component_index, component_id in enumerate(remaining_components, start=1):
            component_mask = labeled == component_id
            roi_name = f"func_{sanitize_name(spec['contrast'])}_extra_{component_index:02d}"
            output_path = functional_dir / f"{roi_name}.nii.gz"
            voxel_count, volume_mm3 = save_mask(component_mask, mask_img, output_path)
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
                    "n_voxels": voxel_count,
                    "volume_mm3": volume_mm3,
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
                    "volume_mm3": volume_mm3,
                    "peak_in_component": False,
                    "peak_to_component_distance_mm": np.nan,
                }
            )

    if functional_qc:
        write_table(pd.DataFrame(functional_qc), qc_dir / "functional_roi_qc.tsv")
    if orphan_peaks:
        write_table(pd.DataFrame(orphan_peaks), qc_dir / "functional_roi_orphan_peaks.tsv")
    return manifest_rows


def atlas_label_lookup(labels: list[str] | tuple[str, ...]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, label in enumerate(labels):
        lookup[str(label)] = index
    return lookup


def has_explicit_hemisphere(labels: list[str]) -> bool:
    for label in labels:
        label_lower = str(label).lower()
        if "left" in label_lower or "right" in label_lower or label_lower.endswith("_l") or label_lower.endswith("_r"):
            return True
    return False


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
    mask = np.isin(atlas_data, indices)
    if hemisphere in {"L", "R"} and not has_explicit_hemisphere(labels):
        coords = np.argwhere(mask)
        hemi_mask = np.zeros_like(mask, dtype=bool)
        if coords.size == 0:
            return hemi_mask
        world = nib.affines.apply_affine(atlas_img.affine, coords)
        if hemisphere == "L":
            keep = world[:, 0] < 0
        else:
            keep = world[:, 0] > 0
        kept_coords = coords[keep]
        if kept_coords.size > 0:
            hemi_mask[tuple(kept_coords.T)] = True
        return hemi_mask
    return mask


def get_reference_pattern_image(pattern_root: Path) -> tuple[nib.Nifti1Image | None, str | None]:
    if pattern_root.exists():
        candidates = sorted(pattern_root.glob("sub-*/pre_yy.nii.gz"))
        if not candidates:
            candidates = sorted(pattern_root.glob("sub-*/*.nii.gz"))
        if candidates:
            return nib.load(str(candidates[0])), str(candidates[0])
    return None, None


def maybe_resample_mask_to_reference(mask: np.ndarray, atlas_img: nib.Nifti1Image, reference_img: nib.Nifti1Image | None) -> nib.Nifti1Image:
    header = atlas_img.header.copy()
    header.set_data_dtype(np.uint8)
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), atlas_img.affine, header)
    if reference_img is None:
        return mask_img
    if tuple(mask_img.shape[:3]) == tuple(reference_img.shape[:3]) and np.allclose(mask_img.affine, reference_img.affine, atol=1e-4):
        return mask_img
    return resample_to_img(mask_img, reference_img, interpolation="nearest", force_resample=True, copy_header=True)


def build_harvard_oxford_rois(mask_root: Path, reference_img: nib.Nifti1Image | None = None) -> list[dict[str, object]]:
    output_dir = ensure_dir(mask_root / "literature")
    qc_dir = ensure_dir(mask_root.parent / "qc")
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = load_img(atlas["maps"])
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=int)
    label_to_index = atlas_label_lookup(atlas["labels"])

    manifest_rows: list[dict[str, object]] = []
    atlas_qc: list[dict[str, object]] = []
    for spec in LITERATURE_ROIS:
        mask = create_atlas_mask(
            atlas_img,
            atlas_data,
            label_to_index,
            spec["labels"],
            hemisphere=spec["hemisphere"],
        )
        output_path = output_dir / f"{spec['roi_name']}.nii.gz"
        mask_img = maybe_resample_mask_to_reference(mask, atlas_img, reference_img)
        voxel_count, volume_mm3 = save_mask_image(mask_img, output_path)
        is_valid = voxel_count >= MIN_ATLAS_VOXELS
        atlas_qc.append({
            "roi_name": spec["roi_name"],
            "roi_set": "literature",
            "atlas_name": spec["atlas_name"],
            "labels": "|".join(spec["labels"]),
            "hemisphere": spec["hemisphere"],
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
            "is_valid": is_valid,
            "reason": "" if is_valid else f"n_voxels<{MIN_ATLAS_VOXELS}",
        })
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
                "include_in_main": bool(is_valid),
                "include_in_rsa": bool(is_valid),
                "include_in_mvpa": bool(is_valid),
                "include_in_rd": bool(is_valid),
                "include_in_gps": bool(is_valid),
                "n_voxels": voxel_count,
                "volume_mm3": volume_mm3,
                "notes": "Literature-guided ROI built from Harvard-Oxford atlas labels."
                         + ("" if is_valid else f" Excluded from analyses: n_voxels<{MIN_ATLAS_VOXELS}."),
            }
        )
    if atlas_qc:
        write_table(pd.DataFrame(atlas_qc), qc_dir / "literature_roi_qc.tsv")
    return manifest_rows


def build_literature_spatial_rois(mask_root: Path, reference_img: nib.Nifti1Image | None = None) -> list[dict[str, object]]:
    output_dir = ensure_dir(mask_root / "literature_spatial")
    qc_dir = ensure_dir(mask_root.parent / "qc")

    ho_atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_img = load_img(ho_atlas["maps"])
    ho_data = np.asarray(ho_img.get_fdata(), dtype=int)
    ho_lookup = atlas_label_lookup(ho_atlas["labels"])

    aal_atlas = datasets.fetch_atlas_aal()
    aal_img = load_img(aal_atlas["maps"])
    aal_data = np.asarray(aal_img.get_fdata(), dtype=int)
    aal_lookup = {str(label): int(index) for label, index in zip(aal_atlas["labels"], aal_atlas["indices"])}

    manifest_rows: list[dict[str, object]] = []
    atlas_qc: list[dict[str, object]] = []

    for spec in LITERATURE_SPATIAL_ROIS:
        if spec["atlas_type"] == "harvard_oxford":
            atlas_img = ho_img
            atlas_data = ho_data
            label_to_index = ho_lookup
        elif spec["atlas_type"] == "aal":
            atlas_img = aal_img
            atlas_data = aal_data
            label_to_index = aal_lookup
        else:
            raise ValueError(f"Unsupported atlas_type for literature_spatial: {spec['atlas_type']}")

        mask = create_atlas_mask(
            atlas_img,
            atlas_data,
            label_to_index,
            spec["labels"],
            hemisphere=spec["hemisphere"],
        )
        output_path = output_dir / f"{spec['roi_name']}.nii.gz"
        mask_img = maybe_resample_mask_to_reference(mask, atlas_img, reference_img)
        voxel_count, volume_mm3 = save_mask_image(mask_img, output_path)
        is_valid = voxel_count >= MIN_ATLAS_VOXELS
        atlas_qc.append({
            "roi_name": spec["roi_name"],
            "roi_set": "literature_spatial",
            "atlas_name": spec["atlas_name"],
            "labels": "|".join(spec["labels"]),
            "hemisphere": spec["hemisphere"],
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
            "is_valid": is_valid,
            "reason": "" if is_valid else f"n_voxels<{MIN_ATLAS_VOXELS}",
        })
        note = spec.get("notes", "Literature-guided spatial ROI built from atlas labels.")
        manifest_rows.append(
            {
                "roi_name": spec["roi_name"],
                "roi_set": "literature_spatial",
                "source_type": spec["atlas_type"],
                "mask_path": str(output_path.resolve()),
                "hemisphere": spec["hemisphere"],
                "base_contrast": "",
                "atlas_name": spec["atlas_name"],
                "atlas_label": "|".join(spec["labels"]),
                "theory_role": spec["theory_role"],
                "include_in_main": False,
                "include_in_rsa": bool(is_valid),
                "include_in_mvpa": bool(is_valid),
                "include_in_rd": bool(is_valid),
                "include_in_gps": bool(is_valid),
                "n_voxels": voxel_count,
                "volume_mm3": volume_mm3,
                "notes": str(note) + ("" if is_valid else f" Excluded from analyses: n_voxels<{MIN_ATLAS_VOXELS}."),
            }
        )
    if atlas_qc:
        write_table(pd.DataFrame(atlas_qc), qc_dir / "literature_spatial_roi_qc.tsv")
    return manifest_rows


def build_aal_rois(mask_root: Path, reference_img: nib.Nifti1Image | None = None) -> list[dict[str, object]]:
    output_dir = ensure_dir(mask_root / "atlas_robustness")
    qc_dir = ensure_dir(mask_root.parent / "qc")
    atlas = datasets.fetch_atlas_aal()
    atlas_img = load_img(atlas["maps"])
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=int)
    label_to_index = {str(label): int(index) for label, index in zip(atlas["labels"], atlas["indices"])}

    manifest_rows: list[dict[str, object]] = []
    atlas_qc: list[dict[str, object]] = []
    for spec in ATLAS_ROBUSTNESS_ROIS:
        indices = match_label_indices(label_to_index, spec["labels"], hemisphere=spec["hemisphere"])
        mask = np.isin(atlas_data, indices)
        output_path = output_dir / f"{spec['roi_name']}.nii.gz"
        mask_img = maybe_resample_mask_to_reference(mask, atlas_img, reference_img)
        voxel_count, volume_mm3 = save_mask_image(mask_img, output_path)
        is_valid = voxel_count >= MIN_ATLAS_VOXELS
        atlas_qc.append({
            "roi_name": spec["roi_name"],
            "roi_set": "atlas_robustness",
            "atlas_name": spec["atlas_name"],
            "labels": "|".join(spec["labels"]),
            "hemisphere": spec["hemisphere"],
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
            "is_valid": is_valid,
            "reason": "" if is_valid else f"n_voxels<{MIN_ATLAS_VOXELS}",
        })
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
                "include_in_rsa": bool(is_valid),
                "include_in_mvpa": bool(is_valid),
                "include_in_rd": bool(is_valid),
                "include_in_gps": bool(is_valid),
                "n_voxels": voxel_count,
                "volume_mm3": volume_mm3,
                "notes": "Atlas robustness ROI built from AAL atlas labels."
                         + ("" if is_valid else f" Excluded from analyses: n_voxels<{MIN_ATLAS_VOXELS}."),
            }
        )
    if atlas_qc:
        write_table(pd.DataFrame(atlas_qc), qc_dir / "atlas_robustness_roi_qc.tsv")
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


def run_roi_alignment_qc(manifest: pd.DataFrame, output_dir: Path, pattern_root: Path) -> None:
    """
    QC: 检查每个 ROI 与 pattern_root 里代表性 pattern 文件的 affine/shape 是否一致。

    思路
    - 找 pattern_root 下第一个存在的 `sub-XX/pre_yy.nii.gz`（或任一 nii.gz）作为参考。
    - 对 manifest 中每个 ROI，比较 affine（按 1e-4 容差）与 shape。
    - 不一致的 ROI 不会被自动重采样；仅在 `roi_alignment_qc.tsv` 里标红，提示下游需要显式 resample。
    """
    qc_dir = ensure_dir(output_dir / "qc")
    ref_img, ref_source = get_reference_pattern_image(pattern_root)

    if ref_img is None:
        save_json(
            {
                "status": "skipped",
                "reason": f"No reference pattern found under {pattern_root}",
                "pattern_root": str(pattern_root),
            },
            qc_dir / "roi_alignment_status.json",
        )
        return

    ref_affine = ref_img.affine
    ref_shape = tuple(ref_img.shape[:3])

    rows: list[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        mask_path = Path(str(row["mask_path"]))
        if not mask_path.exists():
            rows.append({
                "roi_name": row["roi_name"],
                "roi_set": row["roi_set"],
                "mask_path": str(mask_path),
                "affine_match": False,
                "shape_match": False,
                "note": "missing_mask_file",
            })
            continue
        m = nib.load(str(mask_path))
        affine_match = bool(np.allclose(m.affine, ref_affine, atol=1e-4))
        shape_match = tuple(m.shape[:3]) == ref_shape
        rows.append({
            "roi_name": row["roi_name"],
            "roi_set": row["roi_set"],
            "mask_path": str(mask_path),
            "affine_match": affine_match,
            "shape_match": shape_match,
            "mask_shape": str(tuple(m.shape[:3])),
            "ref_shape": str(ref_shape),
            "note": "ok" if (affine_match and shape_match) else "needs_resample_before_downstream",
        })

    write_table(pd.DataFrame(rows), qc_dir / "roi_alignment_qc.tsv")
    save_json(
        {
            "status": "done",
            "reference_file": ref_source,
            "ref_shape": list(ref_shape),
            "n_roi_ok": int(sum(1 for r in rows if r["affine_match"] and r["shape_match"])),
            "n_roi_misaligned": int(sum(1 for r in rows if not (r["affine_match"] and r["shape_match"]))),
        },
        qc_dir / "roi_alignment_status.json",
    )


def build_roi_library(
    output_dir: Path,
    manifest_path: Path,
    pattern_root: Path = DEFAULT_PATTERN_ROOT,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    ensure_dir(output_dir / "masks")
    ensure_dir(output_dir / "qc")
    reference_img, _ = get_reference_pattern_image(pattern_root)

    rows: list[dict[str, object]] = []
    rows.extend(build_functional_rois(output_dir / "masks"))
    rows.extend(build_harvard_oxford_rois(output_dir / "masks", reference_img=reference_img))
    rows.extend(build_literature_spatial_rois(output_dir / "masks", reference_img=reference_img))
    rows.extend(build_aal_rois(output_dir / "masks", reference_img=reference_img))

    manifest = pd.DataFrame(rows).sort_values(["roi_set", "roi_name"]).reset_index(drop=True)
    write_table(manifest, manifest_path)
    summarize_manifest(manifest, output_dir)
    run_roi_alignment_qc(manifest, output_dir, pattern_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable ROI library for RSA/RD/GPS/MVPA.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"ROI library root (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, help=f"Manifest output path (default: {DEFAULT_MANIFEST_PATH}).")
    parser.add_argument(
        "--pattern-root",
        type=Path,
        default=DEFAULT_PATTERN_ROOT,
        help=f"Pattern root for affine/shape QC (default: {DEFAULT_PATTERN_ROOT}).",
    )
    args = parser.parse_args()

    manifest = build_roi_library(args.output_dir, args.manifest_path, args.pattern_root)
    print(f"Built ROI library with {len(manifest)} ROI masks -> {args.manifest_path}")


if __name__ == "__main__":
    main()
