# ================================
# 二阶组分析（SPM风格，对齐你的路径）
# 读取：H:\PythonAnalysis\learn_batch\sub-XX\run-3_spm_style\con_0001.nii.gz
# 输出：K:\metaphor\2nd_level\yy_vs_kj\  (con_0001.nii / spmT_0001.nii / ResMS.nii / mask.nii)
# ================================
import os
from pathlib import Path
from glob import glob
import json
import numpy as np
import nibabel as nib
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import new_img_like, resample_to_img
import pandas as pd

# ===== 路径参数（按你的环境）=====
FIRST_LEVEL_ROOT = Path(r"H:\PythonAnalysis\learn_batch")     # 你的一阶输出根目录
RUN_ID = 4                                                     # run3
CON_INDEX = 1                                                  # 对应 con_0001（yy - kj）
SECOND_LEVEL_DIR = Path(r"H:\PythonAnalysis\learn_batch\2nd_level\run4\yy_vs_kj")     # 与 MATLAB 一致
SECOND_LEVEL_DIR.mkdir(parents=True, exist_ok=True)

# ===== 收集所有 con_0001.nii.gz 路径 =====
pattern = str(FIRST_LEVEL_ROOT / f"sub-*" / f"run-{RUN_ID}_spm_style" / f"con_{CON_INDEX:04d}.nii.gz")
con_paths = sorted(glob(pattern))
if len(con_paths) < 3:
    raise RuntimeError(f"仅找到 {len(con_paths)} 个被试的 con_{CON_INDEX:04d}，需要 >= 3")

# ===== 生成 SPM 风格组掩膜：隐式 0/NaN 排除 + 被试间交集 =====
def nonzero_finite_mask(img_path):
    img = nib.load(img_path)
    dat = img.get_fdata()
    return np.isfinite(dat) & (dat != 0), img

group_mask = None
ref_img = None
for p in con_paths:
    m, ref_img = nonzero_finite_mask(p)
    group_mask = m if group_mask is None else (group_mask & m)

mask_img = new_img_like(ref_img, group_mask.astype(np.uint8), copy_header=True)
# 保存为 .nii（更贴近 SPM）
nib.save(mask_img, str(SECOND_LEVEL_DIR / "mask.nii"))

# ===== 二阶模型：仅截距；不做二阶平滑（与 SPM 一致）=====
n = len(con_paths)
design_matrix = pd.DataFrame({"intercept": np.ones(n, dtype=float)})
slm = SecondLevelModel(mask_img=mask_img, smoothing_fwhm=None, minimize_memory=True)
slm = slm.fit(con_paths, design_matrix=design_matrix)

# ===== 输出 con_0001.nii / spmT_0001.nii（未阈值）=====
con_img = slm.compute_contrast("intercept", output_type="effect_size")
t_img   = slm.compute_contrast("intercept", output_type="stat")

# 转存为 .nii（去掉 .gz，以贴近 SPM 文件形态）
con_path = SECOND_LEVEL_DIR / f"con_{CON_INDEX:04d}.nii"
t_path   = SECOND_LEVEL_DIR / f"spmT_{CON_INDEX:04d}.nii"
nib.save(nib.Nifti1Image(con_img.get_fdata(), con_img.affine, con_img.header), str(con_path))
nib.save(nib.Nifti1Image(t_img.get_fdata(),   t_img.affine,   t_img.header),   str(t_path))

# ===== 计算 ResMS：RSS/(n-1)，在组掩膜内逐体素 =====
n = len(con_paths)
df = n - 1
stack = []
# 为了与 mask 空间完全一致，先把每个被试的 con 重采样到 mask 参考（如有必要）
for p in con_paths:
    img = nib.load(p)
    # 如果各图像空间一致，可以直接用 get_fdata()；稳妥起见，做一次重采样对齐
    if img.shape != mask_img.shape:
        from nilearn import image
        img = image.resample_to_img(img, mask_img, interpolation="continuous")
    stack.append(img.get_fdata())

stack = np.stack(stack, axis=-1)         # (X,Y,Z,n)
mean_ = np.nanmean(stack, axis=-1, keepdims=True)
rss = np.nansum((stack - mean_)**2, axis=-1)
rss[~group_mask] = np.nan
resms = rss / df
resms_img = new_img_like(mask_img, resms, copy_header=True)
nib.save(resms_img, str(SECOND_LEVEL_DIR / "ResMS.nii"))

# ===== 记录元信息（便于核查）=====
meta = {
    "software": "nilearn second-level (SPM-compatible layout)",
    "first_level_root": str(FIRST_LEVEL_ROOT),
    "run_id": RUN_ID,
    "contrast_index": CON_INDEX,
    "contrast_name": "yy > kj",
    "n_subjects": n,
    "df": int(df),
    "outputs": {
        "mask": "mask.nii",
        "con":  f"con_{CON_INDEX:04d}.nii",
        "t":    f"spmT_{CON_INDEX:04d}.nii",
        "ResMS":"ResMS.nii"
    },
    "subjects": con_paths
}
with open(SECOND_LEVEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"二阶完成：{SECOND_LEVEL_DIR}")
