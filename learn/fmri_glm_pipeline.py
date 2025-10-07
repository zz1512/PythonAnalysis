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
from nilearn import image, datasets
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

# =========================
# 参数区（请按实际情况修改）
# =========================
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]  # sub-01 ... sub-28
RUNS = [3]  # 如需多 run，写成 [1,2,3]
TR_FALLBACK = 2.0

# 数据路径模板（按你的命名来）
FMRI_TPL = r"H:\PythonAnalysis\Pro_proc_data\{sub}\run{run}\smooth{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
EVENT_TPL = r"H:\PythonAnalysis\data_events\{sub}\{sub}_run-{run}_events.tsv"
MOTION_TPL = r"H:\PythonAnalysis\Pro_proc_data\{sub}\multi_reg\multi_reg_run{run}.txt"

# 输出
OUTPUT_ROOT = Path(r"H:\PythonAnalysis\learn_batch")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# 对比（可写多个）
CONTRASTS: Dict[str, Dict[str, float]] = {
    "yy > kj": {"yy": +1.0, "kj": -1.0},
    # "kj > yy": {"yy": -1, "kj": +1},
}

# 并行设置
USE_MULTIPROCESS = True  # True 开多进程，False关闭多进程
MAX_WORKERS = 2 # 并行线程数
# N_JOBS_GLM 传给FirstLevelModel的并行（GLM内部），如果并行开启，这个设置为2，否则-1（用满核心）
N_JOBS_GLM = 2 if USE_MULTIPROCESS else -1


# =========================
# 工具函数
# =========================
def log(msg: str):
    print(msg, flush=True)

# 读取运动参数并传给 GLM
def load_motion_confounds(sub: str, run: int) -> Optional[pd.DataFrame]:
    path = Path(MOTION_TPL.format(sub=sub, run=run))
    if not path.exists():
        log(f"[INFO] confounds missing: {path}")
        return None
    try:
        df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
        df.columns = [f"RP{i+1}" for i in range(df.shape[1])]
        return df
    except Exception as e:
        log(f"[WARN] read confounds failed: {path} | {e}")
        return None

def load_events_from_table(path: Path) -> Optional[pd.DataFrame]:
    """
    以 events 表为主：自动识别 onset/duration/trial_type（兼容 condition/name），
    不改动 duration；做基础清洗与排序。
    """
    if not path.exists():
        log(f"[WARN] events missing: {path}")
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        # 列名兼容
        lower = {c.lower(): c for c in df.columns}
        onset_col = lower.get("onset")
        dur_col = lower.get("duration")
        tt_col = lower.get("trial_type")
        if onset_col is None or dur_col is None or tt_col is None:
            log(f"[WARN] events columns incomplete (need onset/duration/trial_type-like): {path}")
            return None

        ev = df[[onset_col, dur_col, tt_col]].copy()
        ev.columns = ["onset", "duration", "trial_type"]  # 规范化列名
        # 基础清洗
        ev = ev.dropna(subset=["onset", "duration", "trial_type"])
        ev["onset"] = ev["onset"].astype(float)
        ev["duration"] = ev["duration"].astype(float)
        ev["trial_type"] = ev["trial_type"].astype(str)
        return ev
    except Exception as e:
        log(f"[WARN] read events failed: {path} | {e}")
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
        tr = float(img.header.get_zooms()[-1])
        return tr if math.isfinite(tr) and tr > 0 else TR_FALLBACK
    except Exception:
        return TR_FALLBACK

def contrast_vector(name2weight: Dict[str, float], dm_cols: List[str]) -> np.ndarray:
    """根据列名映射生成与设计矩阵等长的向量；未提到的列自动为 0。"""
    w = np.zeros(len(dm_cols), dtype=float)
    idx = {c: i for i, c in enumerate(dm_cols)}
    for k, v in name2weight.items():
        if k in idx:
            w[idx[k]] = v
        else:
            log(f"[INFO] design has no column '{k}' (subject/run may lack this condition)")
    return w


def build_design_matrix(fmri_img, events_df: pd.DataFrame) -> pd.DataFrame:
    tr = infer_tr_from_img(fmri_img)
    n_scans = fmri_img.shape[-1]
    frame_times = np.arange(n_scans) * tr
    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=1/128.0
    )
    return dm

def export_spm_style(model: FirstLevelModel,
                     fmri_img,
                     design_matrix: pd.DataFrame,
                     out_dir: Path,
                     contrast_w: Optional[np.ndarray],
                     con_index: int,
                     save_model_info: bool = True,
                     try_resms: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) beta_*（按设计矩阵列序）
    dm_cols = list(model.design_matrices_[0].columns)
    p = len(dm_cols)
    for i in range(p):
        w = np.zeros(p); w[i] = 1.0
        beta_map = model.compute_contrast(w, output_type="effect_size")
        beta_map.to_filename(out_dir / f"beta_{i+1:04d}.nii.gz")

    # 2) mask
    if hasattr(model, "masker_") and getattr(model.masker_, "mask_img_", None) is not None:
        model.masker_.mask_img_.to_filename(out_dir / "mask.nii.gz")

    # 3) con/spmT
    if contrast_w is not None:
        con_map = model.compute_contrast(contrast_w, output_type="effect_size")
        t_map  = model.compute_contrast(contrast_w, output_type="stat")
        con_map.to_filename(out_dir / f"con_{con_index:04d}.nii.gz")
        t_map.to_filename(out_dir / f"spmT_{con_index:04d}.nii.gz")

    # 4) 可选保存关键信息（便于复现/二阶）
    if save_model_info:
        np.savez(out_dir / "model_info.npz",
                 design_matrix=design_matrix.values,
                 column_names=np.array(dm_cols, dtype=object),
                 tr=model.t_r,
                 slice_time_ref=getattr(model, "slice_time_ref", None),
                 noise_model="ar1",
                 hrf_model="spm",
                 high_pass=1/128.0)

# =========================
# 单被试处理
# =========================
def process_one_subject(sub: str, runs: List[int], contrasts: Dict[str, Dict[str, float]]) -> Dict:
    log(f"\n===== {sub} =====")
    sub_out = OUTPUT_ROOT / sub
    sub_out.mkdir(exist_ok=True)

    summary_rows = []

    for run in runs:
        fmri_path = Path(FMRI_TPL.format(sub=sub, run=run))
        events_path = Path(EVENT_TPL.format(sub=sub, run=run))
        run_tag = f"run-{run}"

        if not fmri_path.exists():
            log(f"[WARN] fMRI missing: {fmri_path}")
            continue

        events = load_events_from_table(events_path)
        if events is None or events.empty:
            log(f"[WARN] events empty/bad: {events_path}")
            continue

        # 加载数据
        log(f"[{sub}|{run_tag}] loading fMRI ...")
        fmri_img = image.load_img(fmri_path)

        # 设计矩阵（仅 yy/kj + 漂移 + 常数）
        dm_events_only = build_design_matrix(fmri_img, events)
        log(f"[{sub}|{run_tag}] DM (events-only) columns: {list(dm_events_only.columns)}")

        # 运动参数
        motion_df = load_motion_confounds(sub, run)

        # 掩膜
        # mask_img = compute_mask(fmri_img)

        # 拟合（严格对齐 SPM）
        tr = infer_tr_from_img(fmri_img)
        model = FirstLevelModel(
            t_r=tr,
            slice_time_ref=0.5,      # TR 中点 = SPM fmri_t0/fmri_t
            # mask_img=mask_img,
            noise_model="ar1",
            hrf_model="spm",
            drift_model="cosine",
            high_pass=1/128.0,
            standardize=False,
            signal_scaling=False,
            minimize_memory=False,
            n_jobs=N_JOBS_GLM,
            verbose=0               # 批量运行建议 0；单个调试可改 1/2
        ).fit(fmri_img, events=events, confounds=motion_df)

        # >>> 关键：以后都用“模型内部的完整设计矩阵”（含头动/漂移/常数）
        dm_full = model.design_matrices_[0]
        dm_cols = list(dm_full.columns)
        log(f"[{sub}|{run_tag}] DM (FULL) columns: {dm_cols}")

        # 导出 SPM 风格：beta/mask
        spm_out_dir = sub_out / f"{run_tag}_spm_style"
        export_spm_style(
            model=model,
            fmri_img=fmri_img,
            design_matrix=dm_full,
            out_dir=spm_out_dir,
            contrast_w=None,
            con_index=1,
            save_model_info=True,
            try_resms=False
        )

        # 逐对比导出 con/spmT
        for idx, (cname, spec) in enumerate(contrasts.items(), start=1):
            w = contrast_vector(spec, dm_cols)
            export_spm_style(
                model=model,
                fmri_img=fmri_img,
                design_matrix=dm_full,
                out_dir=spm_out_dir,
                contrast_w=w,
                con_index=idx,
                save_model_info=False,
                try_resms=False
            )
            summary_rows.append({"subject": sub, "run": run, "contrast": cname})

        # 清理
        del model, fmri_img, dm_full
        gc.collect()

    return {"subject": sub, "summary": summary_rows}


def main():
    # 预热（可选）
    try:
        datasets.load_mni152_template()
    except Exception:
        pass

    all_rows = []
    if USE_MULTIPROCESS:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut2sub = {ex.submit(process_one_subject, sub, RUNS, CONTRASTS): sub
                       for sub in SUBJECTS}
            for fut in as_completed(fut2sub):
                sub = fut2sub[fut]
                try:
                    out = fut.result()
                    all_rows.extend(out["summary"])
                except Exception as e:
                    log(f"[ERROR] subject crashed: {sub} | {e}")
    else:
        for sub in SUBJECTS:
            try:
                out = process_one_subject(sub, RUNS, CONTRASTS)
                all_rows.extend(out["summary"])
            except Exception as e:
                log(f"[ERROR] subject crashed: {sub} | {e}")

    # 汇总
    df = pd.DataFrame(all_rows)
    sum_path = OUTPUT_ROOT / "batch_summary.csv"
    df.to_csv(sum_path, index=False, encoding="utf-8-sig")
    log(f"\n[OK] Batch done. Summary → {sum_path}")

if __name__ == "__main__":
    main()
