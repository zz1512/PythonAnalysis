"""
gPPI_analysis.py

用途
- ROI-to-ROI 的 gPPI（广义心理生理交互）简化实现：检验学习阶段 yy vs kj 条件是否调制
  seed ROI 与 target ROI 的功能连接强度。

当前实现
- 心理变量（psych）：yy=+1、kj=-1（仅学习阶段 run3-4）
- 生理变量（phys）：seed ROI 的标准化时间序列
- 交互项（ppi）：phys * psych（再标准化）
- 对每个 target ROI 做 OLS：target_ts ~ seed + psych + ppi + confounds
  关注交互项系数（gPPI beta）

输入
- seed_mask: 种子 ROI mask（NIfTI）
- target_roi_dir: target ROI masks 目录（NIfTI）
- BOLD / events / confounds 路径从 `glm_config.py` 读取（支持 `PYTHON_METAPHOR_ROOT` 覆盖）

输出（output_dir）
- `gppi_subject_metrics.tsv`：subject×run×target 的 gPPI beta
- `gppi_group_summary.tsv`：按 ROI 聚合后的组水平 one-sample t
- `gppi_meta.json`
"""

from __future__ import annotations



import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from nilearn.maskers import NiftiMasker
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from glm_config import config  # noqa: E402
import glm_utils  # noqa: E402


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=1)
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(values)
    return (values - mean) / std


def build_psych_regressor(events: pd.DataFrame, frame_times: np.ndarray, *, tr: float) -> np.ndarray:
    """
    Psychological regressor: +1 for yy, -1 for kj during learning runs.
    Convolved with HRF via compute_regressor.
    """
    work = events[events["trial_type"].isin(["yy", "kj"])].copy()
    if work.empty:
        return np.zeros((frame_times.size, 1), dtype=float)
    amplitudes = np.where(work["trial_type"].astype(str).str.lower() == "yy", 1.0, -1.0)
    exp = np.column_stack([work["onset"].to_numpy(float), work["duration"].to_numpy(float), amplitudes.astype(float)])
    reg, _ = compute_regressor(exp, hrf_model="spm", frame_times=frame_times, con_id="psych")
    return reg.reshape(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-to-ROI gPPI (seed x psych) during learning runs (run3-4).")
    parser.add_argument("seed_mask", type=Path, help="Seed ROI mask NIfTI.")
    parser.add_argument("target_roi_dir", type=Path, help="Target ROI directory (NIfTI masks).")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subjects", nargs="*", default=None, help="Default: glm_config.SUBJECTS")
    parser.add_argument("--runs", nargs="*", type=int, default=[3, 4])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    base_dir = Path(os.environ.get("PYTHON_METAPHOR_ROOT", str(getattr(config, "BASE_DIR", "E:/python_metaphor"))))

    subjects = args.subjects or getattr(config, "SUBJECTS", [f"sub-{i:02d}" for i in range(1, 29)])
    target_rois = sorted(args.target_roi_dir.glob("*.nii*"))
    if not target_rois:
        raise ValueError(f"No ROI masks in {args.target_roi_dir}")

    rows: list[dict[str, Any]] = []
    for sub in subjects:
        for run in args.runs:
            fmri_path = Path(str(config.FMRI_TPL).format(sub=sub, run=run))
            if not fmri_path.exists():
                fmri_path = fmri_path.with_suffix(".nii.gz")
            events_path = Path(str(config.EVENT_TPL).format(sub=sub, run=run))
            confounds_path = Path(str(config.CONFOUNDS_TPL).format(sub=sub, run=run))
            if not fmri_path.exists() or not events_path.exists() or not confounds_path.exists():
                continue

            events = pd.read_csv(events_path, sep="\t")
            if not {"onset", "duration", "trial_type"}.issubset(events.columns):
                continue

            confounds, sample_mask = glm_utils.robust_load_confounds(str(confounds_path), config.DENOISE_STRATEGY)
            tr = float(getattr(config, "TR", 2.0))

            seed_masker = NiftiMasker(mask_img=str(args.seed_mask), standardize=True, t_r=tr)
            seed_ts = seed_masker.fit_transform(str(fmri_path), confounds=confounds, sample_mask=sample_mask).ravel()
            n_scans = seed_ts.size
            frame_times = np.arange(n_scans, dtype=float) * tr

            psych = build_psych_regressor(events, frame_times, tr=tr).ravel()
            seed_z = zscore(seed_ts)
            psych_z = zscore(psych)
            interaction = zscore(seed_z * psych_z)

            # Design: intercept + seed + psych + interaction + confounds
            base = np.column_stack([np.ones(n_scans), seed_z, psych_z, interaction])
            if confounds is not None and len(confounds) == n_scans:
                design = np.column_stack([base, np.asarray(confounds, dtype=float)])
            else:
                design = base

            for roi_path in target_rois:
                roi_name = roi_path.stem.replace(".nii", "")
                target_masker = NiftiMasker(mask_img=str(roi_path), standardize=True, t_r=tr)
                y = target_masker.fit_transform(str(fmri_path), confounds=confounds, sample_mask=sample_mask).ravel()
                if y.size != n_scans:
                    continue

                # OLS coef for interaction term
                coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
                beta_int = float(coef[3])
                rows.append({"subject": sub, "run": int(run), "target_roi": roi_name, "gppi_beta": beta_int})

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "gppi_subject_metrics.tsv")

    summaries = []
    for roi_name, sub in frame.groupby("target_roi") if not frame.empty else []:
        # Average across runs within subject first
        pivot = sub.groupby("subject", as_index=False)["gppi_beta"].mean()
        t_stat, p_val = stats.ttest_1samp(pivot["gppi_beta"].to_numpy(), popmean=0.0, nan_policy="omit")
        summaries.append({"target_roi": roi_name, "n": int(len(pivot)), "mean": float(pivot["gppi_beta"].mean()), "t": float(t_stat), "p": float(p_val)})
    summary_df = pd.DataFrame(summaries)
    write_table(summary_df, output_dir / "gppi_group_summary.tsv")
    save_json({"n_rows": int(len(frame)), "n_rois": int(frame["target_roi"].nunique()) if not frame.empty else 0}, output_dir / "gppi_meta.json")


if __name__ == "__main__":
    main()
