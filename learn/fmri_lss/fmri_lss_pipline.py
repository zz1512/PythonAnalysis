import os, math, gc
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel

# ========== 基本配置 ==========
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
RUNS = [3]
TR_FALLBACK = 2.0
N_JOBS_GLM = 2         # GLM内部并行；CPU紧张可改为1
LSS_COLLAPSE_BY_COND = False  # True: other按条件折叠；False: 所有other合并一列

# 路径模板（按你的文件命名改）
FMRI_TPL   = r"H:\PythonAnalysis\Pro_proc_data\{sub}\run{run}\smooth{sub}_task-yy_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii"
EVENT_TPL  = r"H:\PythonAnalysis\data_events\{sub}\{sub}_run-{run}_events.tsv"
MOTION_TPL = r"H:\PythonAnalysis\Pro_proc_data\{sub}\multi_reg\{sub}_task-yy_run-{run}_desc-confounds_timeseries.tsv"
OUTPUT_ROOT = Path(r"H:\PythonAnalysis\learn_LSS")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ========== 小工具 ==========
def log(s: str): print(s, flush=True)

def infer_tr_from_img(img) -> float:
    try:
        tr = float(img.header.get_zooms()[-1])
        return tr if math.isfinite(tr) and tr > 0 else TR_FALLBACK
    except Exception:
        return TR_FALLBACK

def load_events(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        log(f"[WARN] events missing: {path}")
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        lower = {c.lower(): c for c in df.columns}
        onset = lower.get("onset")
        dur = lower.get("duration")
        ttype = lower.get("trial_type")
        if onset is None or dur is None or ttype is None:
            log(f"[WARN] bad columns in events: {path}")
            return None
        ev = df[[onset, dur, ttype]].copy()
        ev.columns = ["onset", "duration", "trial_type"]
        ev = ev.dropna(subset=["onset", "duration", "trial_type"])
        ev["onset"] = ev["onset"].astype(float)
        ev["duration"] = ev["duration"].astype(float)
        ev["trial_type"] = ev["trial_type"].astype(str)
        # 保留原condition便于LSS按条件折叠
        ev["condition"] = ev["trial_type"]
        ev = ev.sort_values("onset").reset_index(drop=True)
        return ev
    except Exception as e:
        log(f"[WARN] read events failed: {path} | {e}")
        return None

def load_motion_confounds(sub: str, run: int) -> Optional[pd.DataFrame]:
    path = Path(MOTION_TPL.format(sub=sub, run=run))
    if not path.exists():
        log(f"[INFO] confounds missing: {path}")
        return None
    try:
        df = pd.read_csv(path, delimiter='\t')
        confound_glm = df[['white_matter','global_signal','framewise_displacement', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].replace(np.nan, 0)
        return confound_glm
    except Exception as e:
        log(f"[WARN] read confounds failed: {path} | {e}")
        return None

def iter_events_LSS(events: pd.DataFrame):
    """
    为每个trial构造一个小events：
      - 目标trial：trial_type='trial_trXXX'
      - 其它trial：trial_type='other' 或 'other_<cond>'
    """
    base = events.copy()
    n = len(base)
    for i in range(n):
        this = base.iloc[[i]][["onset", "duration"]].copy()
        this["trial_type"] = f"trial_tr{str(i+1).zfill(3)}"

        others = base.drop(base.index[i])[["onset", "duration", "condition"]].copy()
        if LSS_COLLAPSE_BY_COND:
            others["trial_type"] = "other_" + others["condition"].astype(str)
        else:
            others["trial_type"] = "other"

        ev_i = pd.concat([this, others[["onset","duration","trial_type"]]], ignore_index=True)
        yield (i+1, ev_i, this["trial_type"].iloc[0])  # trial_idx, events_i, trial_col_name

# ========== 主流程 ==========
def process_one_run(sub: str, run: int):
    fmri_path = Path(FMRI_TPL.format(sub=sub, run=run))
    events_path = Path(EVENT_TPL.format(sub=sub, run=run))
    if not fmri_path.exists():
        log(f"[WARN] fMRI missing: {fmri_path}"); return

    ev = load_events(events_path)
    if ev is None or ev.empty:
        log(f"[WARN] events empty/bad: {events_path}"); return

    log(f"[{sub}|run-{run}] load fMRI ...")
    fmri_img = image.load_img(str(fmri_path))
    tr = infer_tr_from_img(fmri_img)
    conf = load_motion_confounds(sub, run)

    out_dir = OUTPUT_ROOT / sub / f"run-{run}_LSS"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一的GLM参数（与SPM尽量对齐）
    base_kwargs = dict(
        t_r=tr, slice_time_ref=0.5,
        noise_model="ar1", hrf_model="spm",
        drift_model="cosine", high_pass=1/128.0,
        standardize=False, signal_scaling=False,
        minimize_memory=True,
        n_jobs=N_JOBS_GLM, verbose=0
    )

    trial_rows = []
    saved_mask = False
    total = len(ev)
    log(f"[{sub}|run-{run}] LSS start, trials={total} ...")

    for tidx, ev_i, trial_col in iter_events_LSS(ev):
        # 拟合该trial的小GLM
        model_i = FirstLevelModel(**base_kwargs).fit(
            fmri_img, events=ev_i, confounds=conf
        )
        dm_i = model_i.design_matrices_[0]
        cols = list(dm_i.columns)

        # 取目标trial的β和t
        w = np.zeros(len(cols)); w[cols.index(trial_col)] = 1.0
        beta_i = model_i.compute_contrast(w, output_type="effect_size")
        t_i    = model_i.compute_contrast(w, output_type="stat")

        beta_i.to_filename(out_dir / f"beta_trial_{tidx:03d}.nii.gz")
        t_i.to_filename(out_dir / f"spmT_trial_{tidx:03d}.nii.gz")

        # 导出一次mask便于后续二阶/可视化
        if (not saved_mask) and hasattr(model_i, "masker_") and getattr(model_i.masker_, "mask_img_", None) is not None:
            model_i.masker_.mask_img_.to_filename(out_dir / "mask.nii.gz")
            saved_mask = True

        trial_rows.append({
            "subject": sub, "run": run,
            "trial_index": tidx,
            "trial_col": trial_col,
            "trial_condition": ev.loc[tidx-1, "condition"]
        })

        # 释放
        del model_i, dm_i, beta_i, t_i
        gc.collect()

        if tidx % 10 == 0 or tidx == total:
            log(f"[{sub}|run-{run}] done {tidx}/{total}")

    pd.DataFrame(trial_rows).to_csv(out_dir / "trial_map.csv", index=False, encoding="utf-8-sig")
    del fmri_img
    gc.collect()
    log(f"[{sub}|run-{run}] LSS finished → {out_dir}")

def main():
    for sub in SUBJECTS:
        log(f"\n===== {sub} =====")
        for run in RUNS:
            try:
                process_one_run(sub, run)
            except Exception as e:
                log(f"[ERROR] {sub}|run-{run} crashed: {e}")

if __name__ == "__main__":
    main()
