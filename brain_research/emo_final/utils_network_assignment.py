#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_network_assignment.py

Schaefer 200 7Networks ROI → network mapping & Tian S2 32 subcortical ROI → structure mapping.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SCHAEFER_7NETWORKS: Dict[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience/Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode Network",
}

_LEFT_NETWORK_RANGES: List[tuple[str, int, int]] = [
    ("Vis", 1, 13),
    ("SomMot", 14, 30),
    ("DorsAttn", 31, 43),
    ("SalVentAttn", 44, 52),
    ("Limbic", 53, 59),
    ("Cont", 60, 72),
    ("Default", 73, 100),
]

_RIGHT_NETWORK_RANGES: List[tuple[str, int, int]] = [
    ("Vis", 1, 13),
    ("SomMot", 14, 30),
    ("DorsAttn", 31, 43),
    ("SalVentAttn", 44, 53),
    ("Limbic", 54, 59),
    ("Cont", 60, 72),
    ("Default", 73, 100),
]


def get_schaefer200_network_labels() -> pd.DataFrame:
    rows: list[dict] = []
    for hemi, prefix, ranges in [
        ("L", "L", _LEFT_NETWORK_RANGES),
        ("R", "R", _RIGHT_NETWORK_RANGES),
    ]:
        idx_to_net: Dict[int, str] = {}
        for net, start, end in ranges:
            for i in range(start, end + 1):
                idx_to_net[i] = net
        for i in range(1, 101):
            roi = f"{prefix}_{i}"
            net = idx_to_net[i]
            rows.append(
                {
                    "roi": roi,
                    "roi_id": i,
                    "hemi": hemi,
                    "network": net,
                    "network_full": SCHAEFER_7NETWORKS[net],
                }
            )
    return pd.DataFrame(rows)


TIAN_S2_LABELS: Dict[str, str] = {
    "V_1": "HIP-head",
    "V_2": "HIP-body",
    "V_3": "AMY",
    "V_4": "THA-anterior",
    "V_5": "THA-posterior",
    "V_6": "THA-lateral",
    "V_7": "THA-medial",
    "V_8": "CAU-head",
    "V_9": "CAU-body",
    "V_10": "PUT-anterior",
    "V_11": "PUT-posterior",
    "V_12": "NAc",
    "V_13": "GP-external",
    "V_14": "GP-internal",
    "V_15": "THA-ventral",
    "V_16": "THA-intralaminar",
    "V_17": "HIP-head",
    "V_18": "HIP-body",
    "V_19": "AMY",
    "V_20": "THA-anterior",
    "V_21": "THA-posterior",
    "V_22": "THA-lateral",
    "V_23": "THA-medial",
    "V_24": "CAU-head",
    "V_25": "CAU-body",
    "V_26": "PUT-anterior",
    "V_27": "PUT-posterior",
    "V_28": "NAc",
    "V_29": "GP-external",
    "V_30": "GP-internal",
    "V_31": "THA-ventral",
    "V_32": "THA-intralaminar",
}

_STRUCTURE_TO_GROUP: Dict[str, str] = {
    "HIP-head": "Hippocampus",
    "HIP-body": "Hippocampus",
    "AMY": "Amygdala",
    "THA-anterior": "Thalamus",
    "THA-posterior": "Thalamus",
    "THA-lateral": "Thalamus",
    "THA-medial": "Thalamus",
    "THA-ventral": "Thalamus",
    "THA-intralaminar": "Thalamus",
    "CAU-head": "Caudate",
    "CAU-body": "Caudate",
    "PUT-anterior": "Putamen",
    "PUT-posterior": "Putamen",
    "NAc": "NAc",
    "GP-external": "GP",
    "GP-internal": "GP",
}


def get_tian_s2_labels() -> pd.DataFrame:
    rows: list[dict] = []
    for vid in range(1, 33):
        roi = f"V_{vid}"
        structure = TIAN_S2_LABELS[roi]
        hemi = "L" if vid <= 16 else "R"
        rows.append(
            {
                "roi": roi,
                "structure": structure,
                "hemi": hemi,
                "structure_group": _STRUCTURE_TO_GROUP[structure],
            }
        )
    return pd.DataFrame(rows)


def get_all_roi_info(roi_set: int = 232) -> pd.DataFrame:
    cortical = get_schaefer200_network_labels()
    cortical_out = pd.DataFrame(
        {
            "roi": cortical["roi"],
            "region_type": "cortical",
            "network_or_structure": cortical["network"],
            "hemi": cortical["hemi"],
        }
    )

    if roi_set == 200:
        return cortical_out.reset_index(drop=True)

    subcortical = get_tian_s2_labels()
    subcortical_out = pd.DataFrame(
        {
            "roi": subcortical["roi"],
            "region_type": "subcortical",
            "network_or_structure": subcortical["structure_group"],
            "hemi": subcortical["hemi"],
        }
    )
    return pd.concat([cortical_out, subcortical_out], ignore_index=True)


def get_roi_network_map(roi_set: int = 232) -> dict[str, str]:
    df = get_all_roi_info(roi_set=roi_set)
    return dict(zip(df["roi"], df["network_or_structure"]))


def get_sensorimotor_association_order() -> list[str]:
    return ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


SCAN_REGIONS: Dict[str, tuple[float, float, float]] = {
    "IE_superior_L": (-19.0, -34.0, 59.0),
    "IE_superior_R": (20.0, -31.0, 58.0),
    "IE_middle_L": (-38.0, -18.0, 44.0),
    "IE_middle_R": (40.0, -15.0, 43.0),
    "IE_inferior_L": (-54.0, -3.0, 14.0),
    "IE_inferior_R": (56.0, -1.0, 16.0),
    "SMA_L": (-5.0, -7.0, 52.0),
    "SMA_R": (5.0, -5.0, 49.0),
    "preSMA_dACC_L": (-7.0, 1.0, 36.0),
    "preSMA_dACC_R": (6.0, 3.0, 36.0),
    "Putamen_post_L": (-28.0, -5.0, -1.0),
    "Putamen_post_R": (28.0, -9.0, 3.0),
    "Thalamus_CM_L": (-10.0, -21.0, 2.0),
    "Thalamus_CM_R": (12.0, -20.0, 3.0),
}

SCAN_FUNCTIONAL_GROUPS: Dict[str, List[str]] = {
    "inter_effector": [
        "IE_superior_L", "IE_superior_R",
        "IE_middle_L", "IE_middle_R",
        "IE_inferior_L", "IE_inferior_R",
    ],
    "medial_motor": [
        "SMA_L", "SMA_R",
        "preSMA_dACC_L", "preSMA_dACC_R",
    ],
    "subcortical": [
        "Putamen_post_L", "Putamen_post_R",
        "Thalamus_CM_L", "Thalamus_CM_R",
    ],
}

_SCAN_SUBCORTICAL_ROIS: Dict[str, List[str]] = {
    "Putamen_post_L": ["V_11"],
    "Putamen_post_R": ["V_27"],
    "Thalamus_CM_L": ["V_7", "V_16"],
    "Thalamus_CM_R": ["V_23", "V_32"],
}


def _schaefer200_centroids_nilearn() -> Dict[str, tuple[float, float, float]]:
    try:
        from nilearn import datasets, image
    except ImportError:
        raise ImportError("nilearn is required: pip install nilearn")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200, yeo_networks=7, resolution_mm=1
    )
    atlas_img = image.load_img(atlas["maps"])
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    centroids: Dict[str, tuple[float, float, float]] = {}
    for parcel_id in range(1, 201):
        voxels = np.argwhere(atlas_data == parcel_id)
        if len(voxels) == 0:
            continue
        centroid_vox = voxels.mean(axis=0)
        mni = affine @ np.append(centroid_vox, 1.0)
        if parcel_id <= 100:
            roi = f"L_{parcel_id}"
        else:
            roi = f"R_{parcel_id}"
        centroids[roi] = (float(mni[0]), float(mni[1]), float(mni[2]))
    return centroids


def get_scan_roi_mapping(
    atlas_source: str = "nilearn",
    distance_threshold: float = 20.0,
) -> pd.DataFrame:
    if atlas_source == "nilearn":
        centroids = _schaefer200_centroids_nilearn()
    else:
        raise ValueError(f"Unsupported atlas_source: {atlas_source}")

    roi_info = get_all_roi_info(roi_set=232)
    roi_net_map = get_roi_network_map(roi_set=232)

    rows: list[dict] = []
    for scan_name, (sx, sy, sz) in SCAN_REGIONS.items():
        best_dist = float("inf")
        best_roi = ""
        for roi, (cx, cy, cz) in centroids.items():
            d = float(np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2))
            if d < best_dist:
                best_dist = d
                best_roi = roi

        if scan_name in _SCAN_SUBCORTICAL_ROIS:
            sub_rois = _SCAN_SUBCORTICAL_ROIS[scan_name]
            for sr in sub_rois:
                rows.append({
                    "scan_region": scan_name,
                    "nearest_roi": sr,
                    "distance_mm": 0.0,
                    "roi_network": roi_net_map.get(sr, ""),
                    "is_cortical": False,
                    "within_threshold": True,
                })

        rows.append({
            "scan_region": scan_name,
            "nearest_roi": best_roi,
            "distance_mm": round(best_dist, 1),
            "roi_network": roi_net_map.get(best_roi, ""),
            "is_cortical": True,
            "within_threshold": best_dist <= distance_threshold,
        })

    return pd.DataFrame(rows)


def get_scan_cortical_rois(distance_threshold: float = 20.0) -> List[str]:
    df = get_scan_roi_mapping(distance_threshold=distance_threshold)
    df = df[(df["is_cortical"]) & (df["within_threshold"])].copy()
    return sorted(set(df["nearest_roi"].tolist()))


def get_scan_subcortical_rois() -> List[str]:
    rois: List[str] = []
    for sub_list in _SCAN_SUBCORTICAL_ROIS.values():
        rois.extend(sub_list)
    return sorted(set(rois))


def get_scan_all_rois(distance_threshold: float = 20.0) -> List[str]:
    return sorted(set(get_scan_cortical_rois(distance_threshold) + get_scan_subcortical_rois()))
