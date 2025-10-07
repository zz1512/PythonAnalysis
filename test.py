import os
# USE_MULTIPROCESS = True, 需要使用下面代码避免外层多进程 × 内层GLM × BLAS三层并发导致过载
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# 无界面后端，防止并行绘图崩溃/假卡
import matplotlib
matplotlib.use("Agg")
import gc
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from nilearn import image, plotting, surface, datasets
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.thresholding import threshold_stats_img

# =========================
# 参数区（请按实际情况修改）
# =========================
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]  # sub-01 ... sub-28
RUNS = [3]  # 如需多 run，写成 [1,2,3]
TR_FALLBACK = 2.0

# 数据路径模板（按你的命名来）
FMRI_TPL = r"H:\PythonAnalysis\Pro_proc_data\{sub}\run{run}\smooth{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
EVENT_TPL = r"H:\PythonAnalysis\data_events\{sub}\{sub}_run-{run}_events.tsv"

# 输出
OUTPUT_ROOT = Path(r"H:\PythonAnalysis\learn_batch")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# 对比（可写多个）
CONTRAST_LIST = [
    "yy - kj",
    # "kjw - jc",
    # "yyew - kjew",
]

# 阈值参数
ALPHA = 0.05
HEIGHT_CONTROL = "fdr"  # 或 "fpr"
CLUSTER_K = 0
MAKE_SURFACE = True
SAVE_RESMS = False  # 建议先关掉；若你的 nilearn 版本可稳定获取 residuals 再打开

# 并行设置
USE_MULTIPROCESS = True  # True 开多进程，False关闭多进程
MAX_WORKERS = 2 # 并行线程数
# N_JOBS_GLM 传给FirstLevelModel的并行（GLM内部），如果并行开启，这个设置为2，否则-1（用满核心）
N_JOBS_GLM = 2 if USE_MULTIPROCESS else -1


# =========================
# 工具函数
# =========================
def save_and_close(display_or_fig, out_png, dpi=200):
    """
    兼容 nilearn 不同版本的保存与关闭：
    - 若是 Display：优先用 display.savefig()；关闭用 display.close()
    - 若是 matplotlib Figure：用 fig.savefig()；关闭用 plt.close(fig)
    - 最后兜底 plt.close('all')
    """
    import matplotlib.pyplot as plt

    fig = None

    # ===== 保存 =====
    if hasattr(display_or_fig, "savefig"):
        # nilearn Display 常见
        display_or_fig.savefig(str(out_png), dpi=dpi)
    else:
        # 取底层 figure（不同版本路径略有差异）
        fig = getattr(display_or_fig, "figure", None)
        if fig is None:
            fa = getattr(display_or_fig, "frame_axes", None)
            if fa is not None:
                fig = getattr(fa, "figure", None)
        if fig is not None and hasattr(fig, "savefig"):
            fig.savefig(str(out_png), dpi=dpi)
        else:
            # 实在拿不到，兜底：关掉以免内存泄漏
            plt.savefig(str(out_png), dpi=dpi)

    # ===== 关闭 =====
    try:
        if hasattr(display_or_fig, "close"):
            display_or_fig.close()
            return
    except Exception:
        pass

    if fig is None:
        fig = getattr(display_or_fig, "figure", None)
        if fig is None:
            fa = getattr(display_or_fig, "frame_axes", None)
            if fa is not None:
                fig = getattr(fa, "figure", None)

    if fig is not None:
        plt.close(fig)
    else:
        plt.close('all')

# 读取运动参数并传给 GLM
def load_motion_confounds(sub: str, run: int) -> Optional[pd.DataFrame]:
    path = Path(fr"H:\PythonAnalysis\Pro_proc_data\{sub}\multi_reg\multi_reg_run{run}.txt")
    if not path.exists():
        print(f"confounds missing: {path}")
        return None
    try:
        # SPM 的 multi_reg 通常是空格分隔、无表头；根据你的文件实际格式调整
        df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
        # 给列起名（6 参假设）：RP1..RP6；如果更多列，自动命名
        df.columns = [f"RP{i+1}" for i in range(df.shape[1])]
        return df
    except Exception as e:
        print(f"Read confounds failed: {path} | {e}")
        return None

def safe_load_events(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Events missing: {path}")
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as e:
        print(f"Read events failed: {path} | {e}")
        return None

def infer_tr_from_img(img) -> float:
    try:
        z = img.header.get_zooms()
        tr = float(z[-1])
        if not math.isfinite(tr) or tr <= 0:
            raise ValueError
        return tr
    except Exception:
        return TR_FALLBACK

def compile_contrast_weights(design_cols: List[str], expr: str) -> Optional[np.ndarray]:
    tokens = expr.replace('+',' + ').replace('-',' - ').split()
    w = np.zeros(len(design_cols), dtype=float)
    sign = +1.0
    idx = {c:i for i,c in enumerate(design_cols)}
    for tok in tokens:
        if tok == '+': sign = +1.0
        elif tok == '-': sign = -1.0
        elif tok.strip():
            if tok not in idx:
                print(f"[WARN] Contrast term '{tok}' not in design columns; skip.")
                return None
            w[idx[tok]] += sign
            sign = +1.0
    return w

def save_spm_style_outputs(model: FirstLevelModel,
                           design_matrix: pd.DataFrame,
                           fmri_img,
                           out_dir: Path,
                           contrast_expr: Optional[object] = None,
                           con_index: int = 1,
                           save_mat_info: bool = True,
                           try_resms: bool = SAVE_RESMS):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) beta_*：按设计矩阵列顺序导出
    model_dm = model.design_matrices_[0]
    cols = list(model_dm.columns)
    p = len(cols)
    for i, name in enumerate(cols, start=1):
        w = np.zeros(p, dtype=float)
        w[i - 1] = 1.0
        beta_map = model.compute_contrast(w, output_type="effect_size")
        beta_map.to_filename(out_dir / f"beta_{i:04d}.nii.gz")

    # 2) mask.nii.gz
    if hasattr(model, "masker_") and getattr(model.masker_, "mask_img_", None) is not None:
        model.masker_.mask_img_.to_filename(out_dir / "mask.nii.gz")

    # 3) ResMS（谨慎）
    if try_resms:
        try:
            # 优先尝试以预测-残差方式估计（不同版本适配）
            # A) 若有 residuals 接口
            if hasattr(model, "residuals"):
                resid_img = model.residuals[0]  # 有的版本按run索引
            else:
                # B) 回退：尝试以 t-map 的设计矩阵重建预测（可能不可用，按版本兼容）
                # 警告：不同 nilearn 版本可能不存在或形态不同
                pred = getattr(model, "predicted", None)
                if pred is None:
                    raise AttributeError("No 'predicted' or 'residuals' attribute on FirstLevelModel.")
                resid_img = image.math_img("img - fit", img=fmri_img, fit=pred)

            resms_img = image.math_img("np.mean(img**2, axis=-1)", img=resid_img)
            resms_img.to_filename(out_dir / "ResMS.nii.gz")
        except Exception as e:
            print(f"ResMS export skipped (version-limited): {e}")

    # 4) con_* 与 spmT_*（若提供对比）
    if contrast_expr is not None:
        con_map = model.compute_contrast(contrast_expr, output_type="effect_size")
        con_map.to_filename(out_dir / f"con_{con_index:04d}.nii.gz")

        t_map = model.compute_contrast(contrast_expr, output_type="stat")
        t_map.to_filename(out_dir / f"spmT_{con_index:04d}.nii.gz")

    # 5) “SPM.mat 等价信息”存储
    if save_mat_info:
        np.savez(out_dir / "model_info.npz",
                 design_matrix=design_matrix.values,
                 column_names=np.array(cols, dtype=object),
                 tr=model.t_r,
                 noise_model=getattr(model, "noise_model", "ar1"),
                 hrf_model=getattr(model, "hrf_model", "spm"),
                 contrast_expr=np.array(contrast_expr if contrast_expr else "", dtype=object))

def process_one_subject(sub: str,
                        runs: List[int],
                        contrasts: List[str],
                        fsaverage=None,
                        mni_template=None) -> Dict:
    print(f"\n===== {sub} =====")
    sub_out = OUTPUT_ROOT / sub
    sub_out.mkdir(exist_ok=True)

    results_summary = []

    for run in runs:
        # 准备路径
        fmri_path = Path(FMRI_TPL.format(sub=sub, run=run))
        events_path = Path(EVENT_TPL.format(sub=sub, run=run))
        run_tag = f"run-{run}"

        if not fmri_path.exists():
            print(f"fMRI missing: {fmri_path}")
            continue

        events = safe_load_events(events_path)
        if events is None or events.empty:
            print(f"events empty: {events_path}")
            continue

        # 加载 fMRI
        print(f"Loading: {fmri_path.name}")
        fmri_img = image.load_img(fmri_path)
        n_scans = fmri_img.shape[-1]
        tr = infer_tr_from_img(fmri_img)
        frame_times = np.arange(n_scans) * tr

        # 设计矩阵
        print("Building design matrix...")
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=1/128
        )
        print("DM cols:", list(design_matrix.columns))

        # 拟合 GLM
        print("Fitting GLM...")
        motion_df = load_motion_confounds(sub, run)
        first_level_model = FirstLevelModel(
            t_r=tr,
            noise_model="ar1",
            hrf_model="spm",
            drift_model="cosine",
            high_pass=1 / 128,
            standardize=False,  # ← 关闭标准化
            signal_scaling=False,  # ← 关闭全局缩放
            minimize_memory=False,
            n_jobs=N_JOBS_GLM
        ).fit(fmri_img, events=events, confounds=motion_df)
        model_dm = first_level_model.design_matrices_[0]
        dm_cols = list(model_dm.columns)
        print("MODEL DM cols:", dm_cols)
        # 导出 SPM 风格
        spm_out_dir = sub_out / f"{run_tag}_spm_style"
        save_spm_style_outputs(
            model=first_level_model,
            design_matrix=design_matrix,
            fmri_img=fmri_img,
            out_dir=spm_out_dir,
            contrast_expr=None,  # beta/mask/ResMS（ResMS 视 SAVE_RESMS 决定）
            con_index=1,
            save_mat_info=True,
            try_resms=SAVE_RESMS
        )

        # 每个对比：阈值/图像
        for i, contrast_expr in enumerate(contrasts, start=1):
            print(f"Contrast: {contrast_expr}")
            w = compile_contrast_weights(dm_cols, contrast_expr)
            if w is None:
                # 该被试缺列（比如没有某个条件），跳过这个对比
                continue

            # 单独导出 con/spmT —— 现在传“权重向量”
            save_spm_style_outputs(
                model=first_level_model,
                design_matrix=design_matrix,
                fmri_img=fmri_img,
                out_dir=spm_out_dir,
                contrast_expr=w,
                con_index=i,
                save_mat_info=False,
                try_resms=False
            )

            results_summary.append({
                "subject": sub,
                "run": run,
                "contrast": contrast_expr
            })

        # 清理内存
        del first_level_model, fmri_img, design_matrix
        gc.collect()

    return {"subject": sub, "summary": results_summary}


def main():
    print("Prefetching fsaverage & MNI template (once)...")
    # 仅预热缓存；真用时子进程会各自load（从缓存读）
    try:
        datasets.load_mni152_template()
        if MAKE_SURFACE:
            datasets.fetch_surf_fsaverage()
    except Exception:
        pass

    all_rows = []
    if USE_MULTIPROCESS:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(process_one_subject, sub, RUNS, CONTRAST_LIST, None, None): sub
                for sub in SUBJECTS
            }
            for fut in as_completed(futures):
                try:
                    out = fut.result()
                    all_rows.extend(out["summary"])
                except Exception as e:
                    print(f"Subject crashed: {futures[fut]} | {e}")
    else:
        for sub in SUBJECTS:
            try:
                out = process_one_subject(sub, RUNS, CONTRAST_LIST, None, None)
                all_rows.extend(out["summary"])
            except Exception as e:
                print(f"Subject crashed: {sub} | {e}")

    # 汇总表（主进程写即可）
    df = pd.DataFrame(all_rows)
    sum_path = OUTPUT_ROOT / "batch_summary.csv"
    df.to_csv(sum_path, index=False, encoding="utf-8-sig")
    print(f"\nBatch done. Summary → {sum_path}")


if __name__ == "__main__":
    main()
