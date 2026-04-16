"""
parametric_modulation.py

用途
- 在学习阶段（默认 run3-4）做“参数调制 GLM”（parametric modulation）：
  给每个 trial 一个连续变量（例如暴露次数/熟悉度/新颖性评分），检验哪些脑区的激活随该变量系统变化。
- 这是把 pre/post 的离散对比，提升为连续“熟悉度梯度/学习曲线”的方法（常见审稿加分项）。

输入
- events.tsv 必须包含：onset / duration / trial_type 以及你指定的 `--param-col`
- BOLD / confounds 路径从 `glm_config.py` 读取（支持 `PYTHON_METAPHOR_ROOT` 覆盖）

输出（默认 `${PYTHON_METAPHOR_ROOT}/glm_parametric_modulation`）
- `${out_root}/sub-xx/run-3/pm_{cond}_pm_{param_col}_zmap.nii.gz`：每个条件的参数效应 z-map
- `${out_root}/sub-xx/run-3/design_matrix.tsv`：一阶设计矩阵（便于复核）
- `parametric_modulation_summary.json`：输出概览
"""

from __future__ import annotations



import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from nilearn.masking import compute_brain_mask
from nilearn import image


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json  # noqa: E402
from glm_config import config  # noqa: E402
import glm_utils  # noqa: E402


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.nanmean(values)
    std = np.nanstd(values, ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.zeros_like(values)
    return values / std


def build_parametric_regressor(
    events: pd.DataFrame,
    param_col: str,
    frame_times: np.ndarray,
    *,
    hrf_model: str,
    condition_filter: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build one parametric regressor per condition:
    - amplitude-modulated stick function: amplitude = zscored(param_col)
    - convolved with HRF via nilearn.compute_regressor
    """
    if param_col not in events.columns:
        raise ValueError(f"Missing parametric column in events: {param_col}")

    regressors = []
    names = []

    working = events.copy()
    if condition_filter:
        working = working[working["trial_type"].isin(condition_filter)].copy()
    if working.empty:
        raise ValueError("No events left after filtering; cannot build parametric regressor.")

    for cond, cond_frame in working.groupby("trial_type"):
        amplitudes = zscore(cond_frame[param_col].to_numpy(dtype=float))
        exp = np.column_stack(
            [
                cond_frame["onset"].to_numpy(dtype=float),
                cond_frame["duration"].to_numpy(dtype=float),
                amplitudes,
            ]
        )
        reg, reg_names = compute_regressor(exp, hrf_model=hrf_model, frame_times=frame_times, con_id=str(cond))
        regressors.append(reg)
        names.extend([f"{cond}_pm_{param_col}"])

    x = np.hstack(regressors)
    if x.ndim == 1:
        x = x[:, None]
    return x, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametric modulation GLM (learning stage run3-4).")
    parser.add_argument("--runs", nargs="*", type=int, default=[3, 4])
    parser.add_argument("--subjects", nargs="*", default=None, help="Default: glm_config.SUBJECTS")
    parser.add_argument("--param-col", required=True, help="Column in events.tsv used for parametric modulation.")
    parser.add_argument("--conditions", nargs="*", default=["yy", "kj"], help="trial_type values to keep (default: yy kj).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: ${PYTHON_METAPHOR_ROOT}/glm_parametric_modulation")
    args = parser.parse_args()

    base_dir = Path(os.environ.get("PYTHON_METAPHOR_ROOT", str(getattr(config, "BASE_DIR", "E:/python_metaphor"))))
    out_root = args.output_dir or (base_dir / "glm_parametric_modulation")
    out_root = ensure_dir(out_root)

    subjects = args.subjects or getattr(config, "SUBJECTS", [f"sub-{i:02d}" for i in range(1, 29)])

    results: list[dict[str, Any]] = []
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

            img = image.load_img(str(fmri_path))
            n_scans = img.shape[-1]
            tr = float(getattr(config, "TR", 2.0))
            frame_times = np.arange(n_scans) * tr

            # Build parametric regressors (one per condition).
            add_regs, add_reg_names = build_parametric_regressor(
                events,
                args.param_col,
                frame_times,
                hrf_model="spm",
                condition_filter=args.conditions,
            )

            mask = compute_brain_mask(str(fmri_path))
            model = FirstLevelModel(
                t_r=tr,
                slice_time_ref=0.5,
                smoothing_fwhm=getattr(config, "SMOOTHING_FWHM", 6.0),
                noise_model="ar1",
                standardize=True,
                hrf_model="spm + derivative",
                drift_model="cosine",
                high_pass=1 / 128.0,
                mask_img=mask,
                minimize_memory=True,
            )
            model.fit(str(fmri_path), events=events[["onset", "duration", "trial_type"]], confounds=confounds, sample_masks=sample_mask, add_regs=add_regs, add_reg_names=add_reg_names)

            design = model.design_matrices_[0]
            out_dir = ensure_dir(out_root / sub / f"run-{run}")
            design.to_csv(out_dir / "design_matrix.tsv", sep="\t", index=False)

            for reg_name in add_reg_names:
                if reg_name not in design.columns:
                    continue
                con = np.zeros(len(design.columns), dtype=float)
                con[list(design.columns).index(reg_name)] = 1.0
                zmap = model.compute_contrast(con, output_type="z_score")
                out_path = out_dir / f"pm_{reg_name}_zmap.nii.gz"
                zmap.to_filename(out_path)
                results.append({"subject": sub, "run": int(run), "regressor": reg_name, "zmap": str(out_path)})

    save_json({"n_maps": int(len(results)), "output_root": str(out_root)}, out_root / "parametric_modulation_summary.json")


if __name__ == "__main__":
    main()
