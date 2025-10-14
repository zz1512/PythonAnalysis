# ===== 二阶结果可视化 & 激活簇表 =====
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting, datasets
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
import matplotlib
matplotlib.use("Agg")  # 无界面安全

# ==== 配置：改这里 ====
SECOND_LEVEL_DIR = Path(r"H:\PythonAnalysis\learn_batch\2nd_level\run4\yy_vs_kj")
STAT_BASENAME    = "spmT_0001.nii"   # 也可以换成 z_map，比如 "z_map_0001.nii"

# 阈值二选一
USE_FDR = True        # True: FDR q<ALPHA；False: p< P_UNC 未校正 + 聚类
ALPHA   = 0.05
P_UNC   = 0.001
CL_K    = 10          # 聚类体素阈（未校正方案常用）
TWO_SIDED = True

stat_img_path = SECOND_LEVEL_DIR / STAT_BASENAME
mask_img_path = SECOND_LEVEL_DIR / "mask.nii"
assert stat_img_path.exists(), f"找不到统计图：{stat_img_path}"
mask_img = nib.load(str(mask_img_path)) if mask_img_path.exists() else None

# 读统计图（若空间不一致，重采样到 mask）
stat_img = image.load_img(str(stat_img_path))
if mask_img is not None and stat_img.shape != mask_img.shape:
    stat_img = image.resample_to_img(stat_img, mask_img, interpolation="continuous")

# 阈值
if USE_FDR:
    thr_img, thr_val = threshold_stats_img(
        stat_img, alpha=ALPHA, height_control="fdr",
        cluster_threshold=0, two_sided=TWO_SIDED
    )
    thr_label = f"FDR q<{ALPHA}"
else:
    thr_img, thr_val = threshold_stats_img(
        stat_img, alpha=P_UNC, height_control="fpr",
        cluster_threshold=CL_K, two_sided=TWO_SIDED
    )
    thr_label = f"p<{P_UNC} unc., k≥{CL_K}"

print(f"[INFO] threshold (stat units) = {thr_val:.3f} | {thr_label}")

# 图：正交切片 & 玻璃脑
out_png1 = SECOND_LEVEL_DIR / "stat_map_ortho.png"
out_png2 = SECOND_LEVEL_DIR / "glass_brain.png"

# 若阈后图像为空，用原始图取切点以避免报错
has_data = np.nanmax(np.abs(thr_img.get_fdata())) > 0
coords = plotting.find_xyz_cut_coords(thr_img if has_data else stat_img)

disp = plotting.plot_stat_map(
    thr_img,
    threshold=thr_val,
    display_mode="ortho",
    cut_coords=coords,   # <- 三个坐标，而不是 7
    annotate=True,
    title=f"Second-level ({thr_label})"
)
disp.savefig(str(SECOND_LEVEL_DIR / "stat_map_ortho.png"), dpi=220)
disp.close()

disp2 = plotting.plot_glass_brain(
    thr_img, threshold=thr_val, display_mode="lyrz",
    colorbar=True, plot_abs=False, title=f"Glass brain ({thr_label})"
)
disp2.savefig(str(out_png2), dpi=220); disp2.close()

# 聚类/峰值表
# 新版 nilearn 推荐传原始 stat_img 和数值阈值
try:
    clusters_df = get_clusters_table(
        stat_img=stat_img, stat_threshold=thr_val,
        cluster_threshold=(0 if USE_FDR else CL_K),
        two_sided=TWO_SIDED
    )
except TypeError:
    # 向后兼容：传阈后图像
    clusters_df = get_clusters_table(thr_img, min_distance=8)

# 脑区标注（Harvard-Oxford 皮层+皮下）
def label_with_harvard_oxford(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    ho_subc = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    atlas_imgs = [image.load_img(ho_cort.maps), image.load_img(ho_subc.maps)]
    labels    = [ho_cort.labels, ho_subc.labels]

    # 取列名（兼容 X/Y/Z 或 x/y/z）
    xyz_cols = None
    for cols in [('X','Y','Z'), ('x','y','z')]:
        if set(cols).issubset(df.columns):
            xyz_cols = cols; break
    if xyz_cols is None:
        return df

    def query_labels(x, y, z):
        names = []
        for atlas_img, lab in zip(atlas_imgs, labels):
            ijk = image.coord_transform(x, y, z, np.linalg.inv(atlas_img.affine))
            ijk = tuple(int(round(v)) for v in ijk)
            data = atlas_img.get_fdata()
            if all(0 <= idx < dim for idx, dim in zip(ijk, data.shape)):
                idx = int(data[ijk])
                if 0 <= idx < len(lab) and lab[idx]:
                    names.append(lab[idx])
        return "; ".join(dict.fromkeys(names))  # 去重保序

    df = df.copy()
    df["Label(HO)"] = df.apply(
        lambda r: query_labels(float(r[xyz_cols[0]]),
                               float(r[xyz_cols[1]]),
                               float(r[xyz_cols[2]])), axis=1)
    return df

clusters_df = label_with_harvard_oxford(clusters_df)

# 保存表格
out_csv = SECOND_LEVEL_DIR / "clusters_table.csv"
if clusters_df is None or clusters_df.empty:
    print("[INFO] No suprathreshold clusters at current threshold.")
    pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
else:
    clusters_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"[OK] Saved:\n - {out_png1}\n - {out_png2}\n - {out_csv}")
