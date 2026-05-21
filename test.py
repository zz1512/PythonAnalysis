import nibabel as nib
from pathlib import Path
import pandas as pd

mask_dir = Path("../../learn_LSS/sub-01/run-3_LSS")

# 找到所有 .nii.gz 文件
mask_files = sorted(mask_dir.glob("*.nii.gz"))

records = []
for f in mask_files:
    img = nib.load(str(f))
    shape = img.shape
    voxel_size = nib.affines.voxel_sizes(img.affine)
    records.append({
        "文件名": f.name,
        "维度": shape,
        "体素大小(mm)": tuple(round(v,2) for v in voxel_size)
    })

# 转为DataFrame并打印
df = pd.DataFrame(records)
print(df.to_string(index=False))
