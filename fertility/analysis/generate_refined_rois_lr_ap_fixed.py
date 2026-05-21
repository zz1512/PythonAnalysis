from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import new_img_like

"""
Generate refined ROI masks:
- ACC_L, ACC_R
- Amygdala_L, Amygdala_R
- Insula_L_Anterior, Insula_L_Posterior
- Insula_R_Anterior, Insula_R_Posterior

Fixes included:
1. Load atlas images explicitly and resample to template space
2. Robust label matching with printout and error checks
3. Split bilateral ACC / Insula by x-coordinate after selecting whole-region label
4. Save QC csv with voxel counts and label matches
"""

# ======================
# PATH CONFIG
# ======================
BASE_DIR = Path("C:/python_fertility")
OUT_DIR = BASE_DIR / "roi_masks_refined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_PATH = BASE_DIR / "fmriprep_core_data" / "sub-004" / "run-4" / "sub-004_run-4_MNI_preproc_bold.nii.gz"

# ======================
# LOAD TEMPLATE
# ======================
if not TEMPLATE_PATH.exists():
    raise FileNotFoundError(f"Template image not found: {TEMPLATE_PATH}")

template_img = nib.load(str(TEMPLATE_PATH))
template_img = image.index_img(template_img, 0)

# ======================
# LOAD & RESAMPLE ATLAS
# ======================
atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
sub_atlas = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

atlas_img = image.load_img(atlas.maps)
sub_atlas_img = image.load_img(sub_atlas.maps)

atlas_img = image.resample_to_img(atlas_img, template_img, interpolation="nearest", force_resample=True)
sub_atlas_img = image.resample_to_img(sub_atlas_img, template_img, interpolation="nearest", force_resample=True)

atlas_data = np.asarray(atlas_img.get_fdata())
sub_data = np.asarray(sub_atlas_img.get_fdata())

labels = list(atlas.labels)
sub_labels = list(sub_atlas.labels)

print("=== Cortical labels ===")
for i, lab in enumerate(labels):
    print(i, lab)

print("=== Subcortical labels ===")
for i, lab in enumerate(sub_labels):
    print(i, lab)


def find_first_contains(label_list, keyword: str):
    keyword = keyword.lower()
    for i, name in enumerate(label_list):
        if keyword in str(name).lower():
            return i, str(name)
    return None, None


def save_mask(mask_data: np.ndarray, name: str, matched_label: str = ""):
    n_vox = int(mask_data.sum())
    print(f"{name}: {n_vox} voxels")
    if n_vox == 0:
        print(f"[WARNING] {name} is empty!")
    mask_img = new_img_like(template_img, mask_data.astype(np.uint8))
    out_path = OUT_DIR / f"{name}.nii.gz"
    mask_img.to_filename(str(out_path))
    return {
        "roi": name,
        "matched_label": matched_label,
        "n_voxels": n_vox,
        "file": str(out_path),
    }


qc_rows = []

# ======================
# ACC LEFT / RIGHT
# ======================
# Prefer anterior cingulate label
acc_index, acc_label = find_first_contains(labels, "anterior division")
if acc_index is None:
    acc_index, acc_label = find_first_contains(labels, "cingulate gyrus")

print("ACC label:", acc_index, acc_label)
if acc_index is None:
    raise ValueError("Cannot find ACC-related label in Harvard-Oxford cortical atlas.")

acc_mask = atlas_data == acc_index
coords = np.indices(acc_mask.shape)
xyz = nib.affines.apply_affine(atlas_img.affine, coords.reshape(3, -1).T)
x_coords = xyz[:, 0].reshape(acc_mask.shape)

acc_L = acc_mask & (x_coords < 0)
acc_R = acc_mask & (x_coords > 0)

qc_rows.append(save_mask(acc_L, "ACC_L", acc_label))
qc_rows.append(save_mask(acc_R, "ACC_R", acc_label))

# ======================
# AMYGDALA LEFT / RIGHT
# ======================
amy_L_index, amy_L_label = find_first_contains(sub_labels, "left amygdala")
amy_R_index, amy_R_label = find_first_contains(sub_labels, "right amygdala")

# Backward-compatible fallbacks
if amy_L_index is None:
    amy_L_index, amy_L_label = find_first_contains(sub_labels, "amygdala (l)")
if amy_R_index is None:
    amy_R_index, amy_R_label = find_first_contains(sub_labels, "amygdala (r)")

print("Amy L label:", amy_L_index, amy_L_label)
print("Amy R label:", amy_R_index, amy_R_label)

if amy_L_index is None or amy_R_index is None:
    raise ValueError("Cannot find left/right amygdala labels in Harvard-Oxford subcortical atlas.")

amy_L = sub_data == amy_L_index
amy_R = sub_data == amy_R_index

qc_rows.append(save_mask(amy_L, "Amygdala_L", amy_L_label))
qc_rows.append(save_mask(amy_R, "Amygdala_R", amy_R_label))

# ======================
# INSULA ANTERIOR / POSTERIOR × LEFT / RIGHT
# ======================
insula_index, insula_label = find_first_contains(labels, "insular cortex")
if insula_index is None:
    insula_index, insula_label = find_first_contains(labels, "insula")

print("Insula label:", insula_index, insula_label)
if insula_index is None:
    raise ValueError("Cannot find insula label in Harvard-Oxford cortical atlas.")

insula_mask = atlas_data == insula_index
coords = np.indices(insula_mask.shape)
xyz = nib.affines.apply_affine(atlas_img.affine, coords.reshape(3, -1).T)
x_coords = xyz[:, 0].reshape(insula_mask.shape)
y_coords = xyz[:, 1].reshape(insula_mask.shape)

insula_L = insula_mask & (x_coords < 0)
insula_R = insula_mask & (x_coords > 0)

# y > 0 anterior, y <= 0 posterior
insula_L_A = insula_L & (y_coords > 0)
insula_L_P = insula_L & (y_coords <= 0)
insula_R_A = insula_R & (y_coords > 0)
insula_R_P = insula_R & (y_coords <= 0)

qc_rows.append(save_mask(insula_L_A, "Insula_L_Anterior", insula_label))
qc_rows.append(save_mask(insula_L_P, "Insula_L_Posterior", insula_label))
qc_rows.append(save_mask(insula_R_A, "Insula_R_Anterior", insula_label))
qc_rows.append(save_mask(insula_R_P, "Insula_R_Posterior", insula_label))

# ======================
# SAVE QC + CONFIG SNIPPET
# ======================
qc_df = pd.DataFrame(qc_rows)
qc_csv = OUT_DIR / "roi_mask_qc.csv"
qc_df.to_csv(qc_csv, index=False, encoding="utf-8-sig")
print(f"Saved QC: {qc_csv}")

cfg_path = OUT_DIR / "roi_masks_config_snippet.py"
with open(cfg_path, "w", encoding="utf-8") as f:
    f.write('from pathlib import Path\n')
    f.write('BASE_DIR = Path("C:/python_fertility")\n\n')
    f.write('ROI_MASKS = {\n')
    for roi in [
        "ACC_L", "ACC_R",
        "Amygdala_L", "Amygdala_R",
        "Insula_L_Anterior", "Insula_L_Posterior",
        "Insula_R_Anterior", "Insula_R_Posterior",
    ]:
        f.write(f'    "{roi}": BASE_DIR / "roi_masks_refined/{roi}.nii.gz",\n')
    f.write('}\n')
print(f"Saved config snippet: {cfg_path}")

print("All refined masks generated successfully.")
