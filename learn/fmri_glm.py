import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import image, plotting, surface, datasets
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.thresholding import threshold_stats_img

# =========================
# 1) 与 SPM 输出对应的导出函数
# =========================
def save_spm_style_outputs(model: FirstLevelModel,
                           design_matrix: pd.DataFrame,
                           fmri_img,
                           out_dir: Path,
                           contrast_expr: str = None,
                           con_index: int = 1,
                           save_mat_info: bool = True):
    """
    导出与 SPM 一阶分析相对应的文件：
      - beta_0001.nii.gz, beta_0002.nii.gz, ...
      - con_0001.nii.gz（若指定 contrast_expr）
      - spmT_0001.nii.gz（若指定 contrast_expr）
      - mask.nii.gz
      - ResMS.nii.gz（残差均方）
      - model_info.npz（等价 SPM.mat 的关键信息，若 save_mat_info=True）

    参数
    ----
    model : 已 fit 的 FirstLevelModel
    design_matrix : make_first_level_design_matrix 生成的设计矩阵
    fmri_img : 4D fMRI NIfTI
    out_dir : 输出目录
    contrast_expr : 例如 "yyw - jx"
    con_index : 对比编号（用于命名 con_XXXX 与 spmT_XXXX）
    save_mat_info : 是否额外保存 model_info.npz
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) beta_*：每个设计矩阵列的effect size
    for i, name in enumerate(design_matrix.columns, start=1):
        beta_map = model.compute_contrast(name, output_type="effect_size")
        beta_map.to_filename(out_dir / f"beta_{i:04d}.nii.gz")

    # 2) mask.nii.gz
    if hasattr(model, "masker_") and getattr(model.masker_, "mask_img_", None) is not None:
        model.masker_.mask_img_.to_filename(out_dir / "mask.nii.gz")

    # 3) ResMS.nii.gz（残差均方）
    #   ResMS ≈ mean(residuals^2, axis=time)
    pred = model.predicted  # 与 fmri_img 同形状的 4D 预测
    resid_img = image.math_img("img - fit", img=fmri_img, fit=pred)
    resms_img = image.math_img("np.mean(img**2, axis=-1)", img=resid_img)
    resms_img.to_filename(out_dir / "ResMS.nii.gz")

    # 4) con_* 与 spmT_*（若提供对比）
    if contrast_expr is not None:
        con_map = model.compute_contrast(contrast_expr, output_type="effect_size")
        con_map.to_filename(out_dir / f"con_{con_index:04d}.nii.gz")

        t_map = model.compute_contrast(contrast_expr, output_type="stat")
        t_map.to_filename(out_dir / f"spmT_{con_index:04d}.nii.gz")

    # 5) 保存等价 SPM.mat 的关键信息
    if save_mat_info:
        np.savez(out_dir / "model_info.npz",
                 design_matrix=design_matrix.values,
                 column_names=np.array(design_matrix.columns, dtype=object),
                 tr=model.t_r,
                 noise_model=getattr(model, "noise_model", "ar1"),
                 hrf_model=getattr(model, "hrf_model", "spm"),
                 contrast_expr=np.array(contrast_expr if contrast_expr else "", dtype=object))


# =======================================
# 2) 指定条件对比 + 阈值化 + 可视化（3D）
# =======================================
def run_contrast_and_visualize(model: FirstLevelModel,
                               design_matrix: pd.DataFrame,
                               fmri_img,
                               contrast_expr: str,
                               alpha: float = 0.05,
                               height_control: str = "fdr",  # 'fdr' 或 'fpr'（未校正）
                               cluster_k: int = 0,
                               out_dir: Path = None,
                               make_surface: bool = True,
                               title_prefix: str = ""):
    """
    计算指定对比的 t-map，做阈值化，并输出体积+表面可视化。

    返回
    ----
    dict，包含 t_map、t_map_thr、thr_value、paths 等
    """
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # t-map
    t_map = model.compute_contrast(contrast_expr, output_type="stat")

    # 阈值化
    t_map_thr, thr_value = threshold_stats_img(
        t_map,
        alpha=alpha,
        height_control=height_control,  # 'fdr' or 'fpr' (uncorrected)
        two_sided=True,
        cluster_threshold=cluster_k
    )

    # 清 NaN/Inf
    t_map_thr_clean = image.math_img(
        "np.nan_to_num(img, nan=0, posinf=0, neginf=0)", img=t_map_thr
    )
    
    # 检查显著体素数量
    nonzero = np.sum(t_map_thr_clean.get_fdata() != 0)
    print(f"🔢 Significant voxels: {nonzero}")

    if nonzero == 0:
        print("⚠️ Warning: no significant voxels found at current threshold.")

    # 保存阈值图
    saved_paths = {}
    if out_dir is not None:
        postfix = f"_{height_control}{alpha}".replace(" ", "")
        tpath = out_dir / f"tmap_{contrast_expr.replace(' ', '')}{postfix}.nii.gz"
        t_map_thr_clean.to_filename(tpath)
        saved_paths["tmap_thresholded"] = tpath

    # 体积 3D 概览
    plotting.plot_stat_map(
        t_map_thr_clean,
        threshold=thr_value,
        display_mode="ortho",
        title=f"{title_prefix}{contrast_expr} ({height_control} α<{alpha})",
        colorbar=True
    )

    # 表面可视化（需标准空间）
    if make_surface:
        fsaverage = datasets.fetch_surf_fsaverage()
        mni_template = datasets.load_mni152_template()
        t_map_resampled = image.resample_to_img(t_map_thr_clean, mni_template)

        for hemi in ["left", "right"]:
            surf_mesh = getattr(fsaverage, f"infl_{hemi}")
            pial_mesh = getattr(fsaverage, f"pial_{hemi}")
            sulc_map = getattr(fsaverage, f"sulc_{hemi}")
            surf_data = surface.vol_to_surf(t_map_resampled, pial_mesh)

            plotting.plot_surf_stat_map(
                surf_mesh,
                surf_data,
                hemi=hemi,
                title=f"{hemi.capitalize()} hemisphere ({contrast_expr})",
                colorbar=True,
                bg_map=sulc_map,
                threshold=thr_value
            )

    # 返回信息
    return {
        "t_map": t_map,
        "t_map_thr": t_map_thr_clean,
        "thr_value": thr_value,
        "saved_paths": saved_paths
    }


# ======================
# 下面是你的原始流程示例
# ======================
if __name__ == "__main__":
    # ========= 参数配置 =========
    sub_id = "sub-01"
    tr = 2.0
    contrast_expr = "yyw - jc"   # 对比表达式
    alpha = 0.05                 # 显著性阈值
    output_dir = Path(r"H:\PythonAnalysis\learn")
    output_dir.mkdir(exist_ok=True)

    fmri_path = Path(rf"K:\metaphor\out_fmri\{sub_id}\func\{sub_id}_task-yy_run-1_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
    events_path = Path(rf"K:\metaphor\events\{sub_id}\{sub_id}_run-1_events.tsv")

    # ========= Step 1: 加载数据 =========
    print("🧠 Loading fMRI data_events...")
    fmri_img = image.load_img(fmri_path)
    n_scans = fmri_img.shape[-1]
    frame_times = np.arange(n_scans) * tr

    print("📑 Loading events...")
    events = pd.read_csv(events_path, sep="\t")

    # ========= Step 2: 构建设计矩阵 =========
    print("🧩 Building design matrix...")
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model="spm",
        drift_model="cosine"
    )
    print("✅ Design matrix columns:", design_matrix.columns.tolist())

    # ========= Step 3: 构建 & 拟合 GLM =========
    print("🔍 Fitting GLM model...")
    first_level_model = FirstLevelModel(
        t_r=tr,
        noise_model="ar1",
        hrf_model="spm",
        standardize=True,
        minimize_memory=False
    ).fit(fmri_img, events=events)

    # ========= 用函数 1：导出 SPM 风格文件 =========
    save_spm_style_outputs(
        model=first_level_model,
        design_matrix=design_matrix,
        fmri_img=fmri_img,
        out_dir=output_dir / f"{sub_id}_spm_style",
        contrast_expr=contrast_expr,   # 若不传则只导出 beta/mask/ResMS
        con_index=1,
        save_mat_info=True
    )

    # ========= 用函数 2：做对比 + 阈值化 + 可视化 =========
    res = run_contrast_and_visualize(
        model=first_level_model,
        design_matrix=design_matrix,
        fmri_img=fmri_img,
        contrast_expr=contrast_expr,
        alpha=alpha,
        height_control="fdr",     # 或 'fpr'（未校正 p），例如 alpha=0.001
        cluster_k=0,
        out_dir=output_dir / f"{sub_id}_contrasts",
        make_surface=True,
        title_prefix=f"{sub_id}: "
    )

    print("Saved paths:", res["saved_paths"])
    print("✅ Done.")
