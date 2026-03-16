import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from nilearn import plotting, datasets, surface
from nilearn.image import load_img, math_img

# =========================================================
# 1. 路径设置
# =========================================================
roi_dir = r"C:\python_fertility\roi_masks_refined"

out_single_dir = os.path.join(roi_dir, "single_masks")
out_combined_dir = os.path.join(roi_dir, "combined_figures")
out_surface_dir = os.path.join(roi_dir, "surface_figures")

os.makedirs(out_single_dir, exist_ok=True)
os.makedirs(out_combined_dir, exist_ok=True)
os.makedirs(out_surface_dir, exist_ok=True)

# =========================================================
# 2. ROI 文件名
# =========================================================
roi_files = {
    "ACC_L": "ACC_L.nii.gz",
    "ACC_R": "ACC_R.nii.gz",
    "Amygdala_L": "Amygdala_L.nii.gz",
    "Amygdala_R": "Amygdala_R.nii.gz",
    "Insula_L_Anterior": "Insula_L_Anterior.nii.gz",
    "Insula_L_Posterior": "Insula_L_Posterior.nii.gz",
    "Insula_R_Anterior": "Insula_R_Anterior.nii.gz",
    "Insula_R_Posterior": "Insula_R_Posterior.nii.gz",
}

# =========================================================
# 3. 标签名称
# =========================================================
roi_labels = {
    "ACC_L": "ACC (Left)",
    "ACC_R": "ACC (Right)",
    "Amygdala_L": "Amygdala (Left)",
    "Amygdala_R": "Amygdala (Right)",
    "Insula_L_Anterior": "Anterior Insula (Left)",
    "Insula_L_Posterior": "Posterior Insula (Left)",
    "Insula_R_Anterior": "Anterior Insula (Right)",
    "Insula_R_Posterior": "Posterior Insula (Right)",
}

# =========================================================
# 4. 推荐配色
# =========================================================
roi_colors = {
    "ACC_L": "#D62728",               # 深红
    "ACC_R": "#D62728",
    "Amygdala_L": "#1F77B4",          # 深蓝
    "Amygdala_R": "#1F77B4",
    "Insula_L_Anterior": "#2CA02C",   # 深绿
    "Insula_L_Posterior": "#17BECF",  # 青绿
    "Insula_R_Anterior": "#2CA02C",
    "Insula_R_Posterior": "#17BECF",
}

# =========================================================
# 5. 全局参数
# =========================================================
DPI = 600
BLACK_BG = False
ALPHA_SINGLE = 1.0
ALPHA_COMBINED = 0.95
THRESHOLD = 0.5
DIM_VALUE = -0.35

# =========================================================
# 6. 加载模板
# =========================================================
mni_bg = datasets.load_mni152_template()
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

# =========================================================
# 7. 工具函数
# =========================================================
def make_single_color_cmap(hex_color):
    return ListedColormap([hex_color])

def make_surface_cmap(hex_color):
    """
    surface 图使用更稳定的单色渐变 colormap，
    避免 plot_surf_roi + 单色映射导致串色/显示异常
    """
    return LinearSegmentedColormap.from_list(
        "custom_surface_cmap",
        ["#f7f7f7", hex_color]
    )

def safe_exists(path):
    return os.path.exists(path) and os.path.isfile(path)

# =========================================================
# 8. 单个 ROI：ortho 图
# =========================================================
def save_single_ortho(roi_name, roi_path, hex_color, output_path):
    cmap = make_single_color_cmap(hex_color)

    display = plotting.plot_roi(
        roi_img=roi_path,
        bg_img=mni_bg,
        display_mode="ortho",
        draw_cross=False,
        black_bg=BLACK_BG,
        cmap=cmap,
        alpha=ALPHA_SINGLE,
        annotate=False,
        colorbar=False,
        threshold=THRESHOLD,
        dim=DIM_VALUE
    )

    display.title(
        roi_labels[roi_name],
        size=18,
        color="black",
        bgcolor="white"
    )

    display.savefig(output_path, dpi=DPI, transparent=True)
    display.close()

# =========================================================
# 9. 单个 ROI：glass brain 图
# =========================================================
def save_single_glass(roi_name, roi_path, hex_color, output_path):
    cmap = make_single_color_cmap(hex_color)

    display = plotting.plot_glass_brain(
        stat_map_img=None,
        display_mode="lyrz",
        colorbar=False,
        black_bg=BLACK_BG,
        annotate=False,
        plot_abs=False
    )

    display.add_overlay(
        roi_path,
        cmap=cmap,
        alpha=ALPHA_SINGLE,
        threshold=THRESHOLD
    )

    display.title(
        roi_labels[roi_name],
        size=18,
        color="black",
        bgcolor="white"
    )

    display.savefig(output_path, dpi=DPI, transparent=True)
    display.close()

# =========================================================
# 10. 单个 ROI：surface 图
# 只推荐 ACC / Insula 使用
# =========================================================
def save_single_surface(roi_name, roi_path, hex_color, output_prefix):
    img = load_img(roi_path)

    if "_L" in roi_name:
        surf_mesh = fsaverage.infl_left
        surf_bg = fsaverage.sulc_left
        hemi = "left"
    else:
        surf_mesh = fsaverage.infl_right
        surf_bg = fsaverage.sulc_right
        hemi = "right"

    # ROI 投影到表面
    texture = surface.vol_to_surf(
        img,
        surf_mesh,
        interpolation="nearest"
    )

    if np.all(np.isnan(texture)) or np.nanmax(texture) <= 0:
        print(f"[警告] {roi_name} 投影到表面后没有有效信号，已跳过")
        return

    cmap = make_surface_cmap(hex_color)

    # lateral
    fig = plt.figure(figsize=(8, 6))
    plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh,
        stat_map=texture,
        hemi=hemi,
        view="lateral",
        bg_map=surf_bg,
        bg_on_data=True,
        darkness=0.6,
        cmap=cmap,
        colorbar=False,
        threshold=0.01,
        figure=fig
    )
    plt.title(f"{roi_labels[roi_name]} - lateral", fontsize=16)
    plt.savefig(
        f"{output_prefix}_surface_lateral.png",
        dpi=DPI,
        transparent=True,
        bbox_inches="tight"
    )
    plt.close(fig)

    # medial
    fig = plt.figure(figsize=(8, 6))
    plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh,
        stat_map=texture,
        hemi=hemi,
        view="medial",
        bg_map=surf_bg,
        bg_on_data=True,
        darkness=0.6,
        cmap=cmap,
        colorbar=False,
        threshold=0.01,
        figure=fig
    )
    plt.title(f"{roi_labels[roi_name]} - medial", fontsize=16)
    plt.savefig(
        f"{output_prefix}_surface_medial.png",
        dpi=DPI,
        transparent=True,
        bbox_inches="tight"
    )
    plt.close(fig)

# =========================================================
# 11. surface 分组图
# 注意：一个 group 图只用一个颜色，适合作为总览
# anterior / posterior 的区分请看单个 surface 图
# =========================================================
def save_surface_group(group_name, roi_list, hemi, output_path, hex_color):
    if hemi == "left":
        surf_mesh = fsaverage.infl_left
        surf_bg = fsaverage.sulc_left
    else:
        surf_mesh = fsaverage.infl_right
        surf_bg = fsaverage.sulc_right

    merged_img = None
    for roi_name in roi_list:
        roi_path = os.path.join(roi_dir, roi_files[roi_name])
        if safe_exists(roi_path):
            img = load_img(roi_path)
            if merged_img is None:
                merged_img = img
            else:
                merged_img = math_img("img1 + img2", img1=merged_img, img2=img)

    if merged_img is None:
        print(f"[警告] {group_name} ({hemi}) 没有可用 ROI")
        return

    merged_img = math_img("img > 0", img=merged_img)

    texture = surface.vol_to_surf(
        merged_img,
        surf_mesh,
        interpolation="nearest"
    )

    if np.all(np.isnan(texture)) or np.nanmax(texture) <= 0:
        print(f"[警告] {group_name} ({hemi}) 投影到表面后无有效信号")
        return

    cmap = make_surface_cmap(hex_color)

    fig = plt.figure(figsize=(8, 6))
    plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh,
        stat_map=texture,
        hemi=hemi,
        view="lateral",
        bg_map=surf_bg,
        bg_on_data=True,
        darkness=0.6,
        cmap=cmap,
        colorbar=False,
        threshold=0.01,
        figure=fig
    )
    plt.title(f"{group_name} ({hemi})", fontsize=16)
    plt.savefig(
        output_path,
        dpi=DPI,
        transparent=True,
        bbox_inches="tight"
    )
    plt.close(fig)

# =========================================================
# 12. 生成每个 ROI 的体空间图
# =========================================================
print("=" * 70)
print("开始生成单个 ROI 图（ortho + glass）...")
print("=" * 70)

for roi_name, file_name in roi_files.items():
    roi_path = os.path.join(roi_dir, file_name)

    if not safe_exists(roi_path):
        print(f"[跳过] 文件不存在: {roi_path}")
        continue

    color = roi_colors[roi_name]

    ortho_out = os.path.join(out_single_dir, f"{roi_name}_ortho.png")
    glass_out = os.path.join(out_single_dir, f"{roi_name}_glass.png")

    save_single_ortho(roi_name, roi_path, color, ortho_out)
    save_single_glass(roi_name, roi_path, color, glass_out)

    print(f"[完成] {roi_name} -> ortho / glass")

# =========================================================
# 13. 生成单个 surface 图
# 只对 ACC / Insula 生成
# =========================================================
surface_roi_names = [
    "ACC_L",
    "ACC_R",
    "Insula_L_Anterior",
    "Insula_L_Posterior",
    "Insula_R_Anterior",
    "Insula_R_Posterior",
]

print("=" * 70)
print("开始生成单个 surface 图（ACC / Insula）...")
print("=" * 70)

for roi_name in surface_roi_names:
    roi_path = os.path.join(roi_dir, roi_files[roi_name])

    if not safe_exists(roi_path):
        print(f"[跳过] 文件不存在: {roi_path}")
        continue

    color = roi_colors[roi_name]
    output_prefix = os.path.join(out_surface_dir, roi_name)

    try:
        save_single_surface(roi_name, roi_path, color, output_prefix)
        print(f"[完成] {roi_name} -> surface lateral / medial")
    except Exception as e:
        print(f"[失败] {roi_name} surface 绘图失败: {e}")

# =========================================================
# 14. 全部 ROI 总览图
# =========================================================
print("=" * 70)
print("开始生成全部 ROI 总览图...")
print("=" * 70)

display = plotting.plot_glass_brain(
    stat_map_img=None,
    display_mode="lyrz",
    colorbar=False,
    black_bg=BLACK_BG,
    annotate=False,
    plot_abs=False
)

for roi_name, file_name in roi_files.items():
    roi_path = os.path.join(roi_dir, file_name)
    if safe_exists(roi_path):
        display.add_overlay(
            roi_path,
            cmap=make_single_color_cmap(roi_colors[roi_name]),
            alpha=ALPHA_COMBINED,
            threshold=THRESHOLD
        )

display.title(
    "All ROI Masks",
    size=20,
    color="black",
    bgcolor="white"
)

all_rois_out = os.path.join(out_combined_dir, "all_rois_glass.png")
display.savefig(all_rois_out, dpi=DPI, transparent=True)
display.close()

print(f"[完成] 全部 ROI 总览图: {all_rois_out}")

# =========================================================
# 15. 分组 glass 图
# =========================================================
groups = {
    "ACC": ["ACC_L", "ACC_R"],
    "Amygdala": ["Amygdala_L", "Amygdala_R"],
    "Insula": [
        "Insula_L_Anterior",
        "Insula_L_Posterior",
        "Insula_R_Anterior",
        "Insula_R_Posterior",
    ],
}

print("=" * 70)
print("开始生成分组 glass 图...")
print("=" * 70)

for group_name, roi_list in groups.items():
    display = plotting.plot_glass_brain(
        stat_map_img=None,
        display_mode="lyrz",
        colorbar=False,
        black_bg=BLACK_BG,
        annotate=False,
        plot_abs=False
    )

    for roi_name in roi_list:
        roi_path = os.path.join(roi_dir, roi_files[roi_name])
        if safe_exists(roi_path):
            display.add_overlay(
                roi_path,
                cmap=make_single_color_cmap(roi_colors[roi_name]),
                alpha=ALPHA_COMBINED,
                threshold=THRESHOLD
            )

    display.title(
        group_name,
        size=20,
        color="black",
        bgcolor="white"
    )

    out_path = os.path.join(out_combined_dir, f"{group_name}_group_glass.png")
    display.savefig(out_path, dpi=DPI, transparent=True)
    display.close()

    print(f"[完成] {group_name} 分组图: {out_path}")

# =========================================================
# 16. 双侧合并图
# =========================================================
bilateral_groups = {
    "ACC_bilateral": ["ACC_L", "ACC_R"],
    "Amygdala_bilateral": ["Amygdala_L", "Amygdala_R"],
    "Insula_Anterior_bilateral": ["Insula_L_Anterior", "Insula_R_Anterior"],
    "Insula_Posterior_bilateral": ["Insula_L_Posterior", "Insula_R_Posterior"],
}

print("=" * 70)
print("开始生成双侧合并 ortho 图...")
print("=" * 70)

for pair_name, roi_list in bilateral_groups.items():
    imgs = []

    for roi_name in roi_list:
        roi_path = os.path.join(roi_dir, roi_files[roi_name])
        if safe_exists(roi_path):
            imgs.append(load_img(roi_path))

    if len(imgs) == 0:
        print(f"[跳过] {pair_name} 无可用文件")
        continue

    merged = imgs[0]
    for img in imgs[1:]:
        merged = math_img("img1 + img2", img1=merged, img2=img)

    merged = math_img("img > 0", img=merged)

    if "ACC" in pair_name:
        color = "#D62728"
    elif "Amygdala" in pair_name:
        color = "#1F77B4"
    elif "Anterior" in pair_name:
        color = "#2CA02C"
    else:
        color = "#17BECF"

    display = plotting.plot_roi(
        roi_img=merged,
        bg_img=mni_bg,
        display_mode="ortho",
        draw_cross=False,
        black_bg=BLACK_BG,
        cmap=make_single_color_cmap(color),
        alpha=ALPHA_SINGLE,
        annotate=False,
        colorbar=False,
        threshold=THRESHOLD,
        dim=DIM_VALUE
    )

    display.title(
        pair_name.replace("_", " "),
        size=18,
        color="black",
        bgcolor="white"
    )

    out_path = os.path.join(out_combined_dir, f"{pair_name}_ortho.png")
    display.savefig(out_path, dpi=DPI, transparent=True)
    display.close()

    print(f"[完成] 双侧图: {out_path}")

# =========================================================
# 17. surface 分组图
# =========================================================
print("=" * 70)
print("开始生成 surface 分组图...")
print("=" * 70)

try:
    save_surface_group(
        group_name="ACC",
        roi_list=["ACC_L"],
        hemi="left",
        output_path=os.path.join(out_surface_dir, "ACC_left_surface_group.png"),
        hex_color="#D62728"
    )
    save_surface_group(
        group_name="ACC",
        roi_list=["ACC_R"],
        hemi="right",
        output_path=os.path.join(out_surface_dir, "ACC_right_surface_group.png"),
        hex_color="#D62728"
    )
    print("[完成] ACC surface 分组图")
except Exception as e:
    print(f"[失败] ACC surface 分组图失败: {e}")

try:
    save_surface_group(
        group_name="Insula",
        roi_list=["Insula_L_Anterior", "Insula_L_Posterior"],
        hemi="left",
        output_path=os.path.join(out_surface_dir, "Insula_left_surface_group.png"),
        hex_color="#2CA02C"
    )
    save_surface_group(
        group_name="Insula",
        roi_list=["Insula_R_Anterior", "Insula_R_Posterior"],
        hemi="right",
        output_path=os.path.join(out_surface_dir, "Insula_right_surface_group.png"),
        hex_color="#2CA02C"
    )
    print("[完成] Insula surface 分组图")
except Exception as e:
    print(f"[失败] Insula surface 分组图失败: {e}")

# =========================================================
# 18. 完成提示
# =========================================================
print("=" * 70)
print("全部绘图完成")
print(f"单个 ROI 图片保存在: {out_single_dir}")
print(f"组合图片保存在: {out_combined_dir}")
print(f"立体 surface 图片保存在: {out_surface_dir}")
print("=" * 70)
print("说明：")
print("1) Amygdala 为深部脑区，建议优先使用 ortho / glass 图，不建议作为 surface 主展示。")
print("2) ACC / Insula 的 surface 图更适合放到论文主文的 ROI 示意图中。")
print("3) surface 分组图使用单一颜色做总览；anterior / posterior 的颜色区分请看单个 surface 图。")
print("=" * 70)