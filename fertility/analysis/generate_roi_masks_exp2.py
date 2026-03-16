#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成实验 1/2/3 可直接使用的 ROI masks。

输出命名方式示例：
ROI_MASKS = {
    "ACC": BASE_DIR / "roi_masks/ACC_mask.nii.gz",
    "Amygdala": BASE_DIR / "roi_masks/Amygdala_mask.nii.gz",
    "Insula": BASE_DIR / "roi_masks/Insula_mask.nii.gz",
}

设计原则：
1. threat/RSA 主要 ROI：ACC, Amygdala, Insula
2. 生育/婴儿图片/奖赏与照护补充 ROI：NAcc, OFC, Fusiform, Hippocampus
3. 决策/价值整合补充 ROI：vmPFC, dlPFC, dmPFC, TPJ, PCC_Precuneus
4. 尽量优先使用 Harvard-Oxford atlas；atlas 中没有清晰标签的区域用 MNI 球形 ROI

依赖：nilearn, nibabel, numpy, pandas
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import new_img_like

# =========================
# 基础路径
# =========================
BASE_DIR = Path("C:/python_fertility")
OUTPUT_DIR = BASE_DIR / "roi_masks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 模板与 atlas
# =========================
print("[INFO] Loading MNI template and Harvard-Oxford atlases...")
template_img = datasets.load_mni152_template(resolution=2)
if not hasattr(template_img, "shape"):
    template_img = image.load_img(template_img)

template_img = image.resample_img(template_img, target_affine=template_img.affine, target_shape=template_img.shape)

a_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
a_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

cort_img = image.resample_to_img(image.load_img(a_cort.maps), template_img, interpolation="nearest")
sub_img = image.resample_to_img(image.load_img(a_sub.maps), template_img, interpolation="nearest")

cort_data = np.asarray(cort_img.get_fdata(), dtype=np.int16)
sub_data = np.asarray(sub_img.get_fdata(), dtype=np.int16)

# Harvard-Oxford labels: index 0 is background
cort_labels = list(a_cort.labels)
sub_labels = list(a_sub.labels)

# =========================
# 工具函数
# =========================
def save_mask(mask_data: np.ndarray, out_path: Path):
    mask_img = new_img_like(template_img, mask_data.astype(np.uint8))
    mask_img.to_filename(str(out_path))
    n_vox = int(mask_data.sum())
    print(f"[OK] Saved {out_path.name} | voxels={n_vox}")
    return n_vox


def find_label_indices(labels, patterns):
    patterns = [p.lower() for p in patterns]
    idx = []
    matched_labels = []
    for i, lab in enumerate(labels):
        lab_l = str(lab).lower()
        if any(p in lab_l for p in patterns):
            idx.append(i)
            matched_labels.append(str(lab))
    return idx, matched_labels


def atlas_mask_from_patterns(atlas_data, labels, patterns, out_name):
    idx, matched = find_label_indices(labels, patterns)
    if len(idx) == 0:
        raise ValueError(f"No labels matched for {out_name}: {patterns}")
    mask = np.isin(atlas_data, idx)
    out_path = OUTPUT_DIR / f"{out_name}_mask.nii.gz"
    n_vox = save_mask(mask, out_path)
    return {
        "roi": out_name,
        "method": "HarvardOxford",
        "patterns": patterns,
        "matched_labels": matched,
        "file": str(out_path),
        "n_voxels": n_vox,
    }


def sphere_mask_from_centers(centers_mni, radius_mm, out_name):
    shape = template_img.shape
    affine = template_img.affine
    ijk = np.indices(shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk)
    mask_flat = np.zeros(xyz.shape[0], dtype=bool)
    for center in centers_mni:
        center = np.asarray(center, dtype=float)
        dist2 = np.sum((xyz - center) ** 2, axis=1)
        mask_flat |= dist2 <= (radius_mm ** 2)
    mask = mask_flat.reshape(shape)
    out_path = OUTPUT_DIR / f"{out_name}_mask.nii.gz"
    n_vox = save_mask(mask, out_path)
    return {
        "roi": out_name,
        "method": "Sphere",
        "centers_mni": centers_mni,
        "radius_mm": radius_mm,
        "file": str(out_path),
        "n_voxels": n_vox,
    }


def try_build_atlas_roi(atlas_data, labels, patterns, out_name, fallback_centers=None, fallback_radius=8):
    try:
        return atlas_mask_from_patterns(atlas_data, labels, patterns, out_name)
    except Exception as e:
        print(f"[WARN] Atlas ROI failed for {out_name}: {e}")
        if fallback_centers is None:
            raise
        print(f"[INFO] Falling back to spherical ROI for {out_name}...")
        return sphere_mask_from_centers(fallback_centers, fallback_radius, out_name)


# =========================
# ROI 方案
# =========================
# 1) Threat / 情绪加工主 ROI
# ACC：优先 atlas 的 anterior cingulate/paracingulate；若匹配失败则回退到球形 ROI
# Insula / Amygdala：atlas
#
# 2) 生育/婴儿图片/奖赏补充 ROI
# NAcc, OFC, Fusiform, Hippocampus, LOC
#
# 3) 决策/价值整合补充 ROI
# vmPFC, dlPFC, dmPFC, TPJ, PCC_Precuneus

manifest = []

# ---------- threat core ----------
manifest.append(
    try_build_atlas_roi(
        cort_data,
        cort_labels,
        patterns=["cingulate gyrus, anterior division", "paracingulate gyrus"],
        out_name="ACC",
        fallback_centers=[(0, 24, 30)],
        fallback_radius=8,
    )
)

manifest.append(
    atlas_mask_from_patterns(
        sub_data,
        sub_labels,
        patterns=["amygdala"],
        out_name="Amygdala",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        cort_data,
        cort_labels,
        patterns=["insular cortex"],
        out_name="Insula",
    )
)

# ---------- fertility / infant-picture / caregiving reward ----------
manifest.append(
    atlas_mask_from_patterns(
        sub_data,
        sub_labels,
        patterns=["accumbens"],
        out_name="NAcc",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        cort_data,
        cort_labels,
        patterns=["frontal orbital cortex"],
        out_name="OFC",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        cort_data,
        cort_labels,
        patterns=["fusiform"],
        out_name="Fusiform",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        sub_data,
        sub_labels,
        patterns=["hippocampus"],
        out_name="Hippocampus",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        cort_data,
        cort_labels,
        patterns=["lateral occipital cortex"],
        out_name="LOC",
    )
)

# ---------- decision / valuation / control ----------
manifest.append(
    sphere_mask_from_centers(
        centers_mni=[(0, 46, -8)],
        radius_mm=8,
        out_name="vmPFC",
    )
)

manifest.append(
    sphere_mask_from_centers(
        centers_mni=[(-44, 36, 20), (44, 36, 20)],
        radius_mm=8,
        out_name="dlPFC",
    )
)

manifest.append(
    sphere_mask_from_centers(
        centers_mni=[(0, 52, 30)],
        radius_mm=8,
        out_name="dmPFC",
    )
)

manifest.append(
    sphere_mask_from_centers(
        centers_mni=[(-54, -54, 22), (54, -54, 22)],
        radius_mm=8,
        out_name="TPJ",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        cort_data,
        cort_labels,
        patterns=["precuneous cortex", "cingulate gyrus, posterior division"],
        out_name="PCC_Precuneus",
    )
)

# ---------- optional striatal masks ----------
manifest.append(
    atlas_mask_from_patterns(
        sub_data,
        sub_labels,
        patterns=["caudate"],
        out_name="Caudate",
    )
)

manifest.append(
    atlas_mask_from_patterns(
        sub_data,
        sub_labels,
        patterns=["putamen"],
        out_name="Putamen",
    )
)

# =========================
# 保存 manifest 和配置片段
# =========================
manifest_df = pd.DataFrame(manifest)
manifest_csv = OUTPUT_DIR / "roi_manifest.csv"
manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
print(f"[OK] Saved manifest CSV: {manifest_csv}")

manifest_json = OUTPUT_DIR / "roi_manifest.json"
with open(manifest_json, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"[OK] Saved manifest JSON: {manifest_json}")

roi_masks_py = OUTPUT_DIR / "roi_masks_config_snippet.py"
with open(roi_masks_py, "w", encoding="utf-8") as f:
    f.write('from pathlib import Path\n')
    f.write('BASE_DIR = Path("C:/python_fertility")\n\n')
    f.write('ROI_MASKS = {\n')
    for row in manifest:
        roi_name = row["roi"]
        f.write(f'    "{roi_name}": BASE_DIR / "roi_masks/{roi_name}_mask.nii.gz",\n')
    f.write('}\n')
print(f"[OK] Saved config snippet: {roi_masks_py}")

print("\nDone. Suggested primary masks for experiment 2 RSA:")
print("  - ACC_mask.nii.gz")
print("  - Amygdala_mask.nii.gz")
print("  - Insula_mask.nii.gz")
