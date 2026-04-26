"""
motion_qc_reliability.py

用途
- 为 Step 5C/5D 提供最小“反驳头动/SNR/测量变差”的证据包。
- 复用 GLM/LSS 主线使用的 confounds 路径与 scrubbing 逻辑，提取 pre/post 的 motion/QC 指标。
- 基于 voxel split-half reliability，在每个 ROI 内评估 pre/post 阶段的表征稳定性，
  检验 post 是否系统性更不稳定。
- 将 QC / reliability 与现有 RSA `Δsimilarity` 对接，便于后续做相关或写 SI。

默认输入
- confounds: `glm_analysis.glm_config.config.CONFOUNDS_TPL`
- pattern_root: `${PYTHON_METAPHOR_ROOT}/pattern_root`
- rsa_itemwise: `rsa_analysis.rsa_config.OUTPUT_DIR / rsa_itemwise_details.csv`

默认输出
- `${PYTHON_METAPHOR_ROOT}/qc_reliability_{ROI_SET}/`

为什么 reliability 不是直接做 post vs pre
- 这里要区分两个不同问题：
  1. `QC` 是否在阶段层面变差：这部分确实直接比较 `pre vs post`
     （例如 mean FD、DVARS、scrubbing 保留比例）。
  2. `pattern reliability` 是否在 post 阶段变差：这部分不能直接算 `pre vs post`，
     因为 pre/post 之间本来就可能包含真实学习效应。
- 若直接把 pre 和 post 的 neural RDM 做相关，相关下降既可能表示“post 更不稳定”，
  也可能只是表示“学习后表征真的变了”，两者无法区分。
- 因此 reliability 的正确定义应是“同一阶段内部的重复一致性”，
  然后再比较 `pre` 与 `post` 的 reliability 指标是否存在 `post < pre`。
- 这样才能真正回答替代解释：
  `yy` 的 pre->post similarity 下降，是否只是因为 post 阶段整体更不可靠。

为什么这里采用 voxel split-half reliability
- 当前实验里，同一阶段的刺激总集会随机拆分到两个 run，因此 `run1` 和 `run2`
  本来就没有完全相同的 item 标签。
- 所以基于“跨 run 找共同 item”的 reliability 定义并不适用，也不再保留。
- 更合适的内部一致性指标是 voxel split-half reliability：
  用相同 trial 集合、但彼此独立的体素子集，各自构建 neural RDM，再相关两套 RDM。
- 这样仍然可以比较 `pre` 与 `post` 的表征稳定性，同时避免把真实学习效应误当成 reliability 下降。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
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
GLM_ROOT = FINAL_ROOT / "glm_analysis"
if str(GLM_ROOT) not in sys.path:
    sys.path.append(str(GLM_ROOT))
RSA_ROOT = FINAL_ROOT / "rsa_analysis"
if str(RSA_ROOT) not in sys.path:
    sys.path.append(str(RSA_ROOT))

from common.final_utils import ensure_dir, paired_t_summary, read_table, save_json, write_table  # noqa: E402
from common.pattern_metrics import load_masked_samples, neural_rdm_vector  # noqa: E402
from common.roi_library import default_roi_tagged_out_dir, select_roi_masks  # noqa: E402
from glm_config import config as glm_config  # noqa: E402
import glm_utils  # noqa: E402
import rsa_config as rsa_cfg  # noqa: E402


STAGE_TO_RUNS = {
    "pre": [1, 2],
    "post": [5, 6],
}

RSA_CONDITION_MAP = {
    "metaphor": "yy",
    "spatial": "kj",
    "baseline": "baseline",
    "yy": "yy",
    "kj": "kj",
}


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    return RSA_CONDITION_MAP.get(text, text)


def _zscore_trials(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean(axis=1, keepdims=True)
    std = samples.std(axis=1, keepdims=True)
    return (samples - mean) / (std + 1e-8)


def _default_output_dir(roi_set: str | None = None) -> Path:
    return default_roi_tagged_out_dir(
        glm_config.BASE_DIR,
        "qc_reliability",
        override_env="METAPHOR_QC_RELIABILITY_OUT_DIR",
        roi_set=roi_set or getattr(rsa_cfg, "ROI_SET", "main_functional"),
    )


def _confounds_path(subject: str, run: int) -> Path:
    return Path(str(glm_config.CONFOUNDS_TPL).format(sub=subject, run=run))


def _collect_run_qc(subject: str, run: int) -> dict[str, object]:
    confounds_path = _confounds_path(subject, run)
    if not confounds_path.exists():
        return {
            "subject": subject,
            "run": run,
            "stage": "pre" if run in STAGE_TO_RUNS["pre"] else "post",
            "confounds_path": str(confounds_path),
            "missing_confounds": True,
        }

    raw = pd.read_csv(confounds_path, sep="\t")
    _, sample_mask = glm_utils.robust_load_confounds(str(confounds_path), glm_config.DENOISE_STRATEGY)
    n_volumes = int(len(raw))
    n_retained = int(len(sample_mask)) if sample_mask is not None else n_volumes
    n_scrubbed = int(n_volumes - n_retained)
    retain_ratio = float(n_retained / n_volumes) if n_volumes else float("nan")

    motion_outlier_cols = [col for col in raw.columns if "motion_outlier" in col]
    motion_outlier_rows = (
        raw[motion_outlier_cols].fillna(0).sum(axis=1).gt(0).sum() if motion_outlier_cols else 0
    )
    motion_outlier_ratio = float(motion_outlier_rows / n_volumes) if n_volumes else float("nan")

    fd = pd.to_numeric(raw.get("framewise_displacement"), errors="coerce")
    std_dvars = pd.to_numeric(raw.get("std_dvars"), errors="coerce")
    rmsd = pd.to_numeric(raw.get("rmsd"), errors="coerce")
    fd_threshold = float(glm_config.DENOISE_STRATEGY.get("fd_threshold", np.nan))
    dvars_threshold = float(glm_config.DENOISE_STRATEGY.get("std_dvars_threshold", np.nan))

    fd_gt = int(fd.gt(fd_threshold).sum()) if np.isfinite(fd_threshold) else 0
    dvars_gt = int(std_dvars.gt(dvars_threshold).sum()) if np.isfinite(dvars_threshold) else 0

    return {
        "subject": subject,
        "run": run,
        "stage": "pre" if run in STAGE_TO_RUNS["pre"] else "post",
        "confounds_path": str(confounds_path),
        "missing_confounds": False,
        "n_volumes": n_volumes,
        "n_retained": n_retained,
        "n_scrubbed": n_scrubbed,
        "retain_ratio": retain_ratio,
        "n_motion_outlier_rows": int(motion_outlier_rows),
        "motion_outlier_ratio": motion_outlier_ratio,
        "mean_fd": _safe_float(fd.mean()),
        "median_fd": _safe_float(fd.median()),
        "max_fd": _safe_float(fd.max()),
        "fd_gt_threshold_count": fd_gt,
        "fd_gt_threshold_ratio": float(fd_gt / n_volumes) if n_volumes else float("nan"),
        "mean_std_dvars": _safe_float(std_dvars.mean()),
        "max_std_dvars": _safe_float(std_dvars.max()),
        "std_dvars_gt_threshold_count": dvars_gt,
        "std_dvars_gt_threshold_ratio": float(dvars_gt / n_volumes) if n_volumes else float("nan"),
        "mean_rmsd": _safe_float(rmsd.mean()),
        "max_rmsd": _safe_float(rmsd.max()),
    }


def _aggregate_stage_qc(run_metrics: pd.DataFrame) -> pd.DataFrame:
    if run_metrics.empty:
        return pd.DataFrame()
    valid = run_metrics[run_metrics["missing_confounds"] == False].copy()  # noqa: E712
    if valid.empty:
        return pd.DataFrame()

    rows = []
    weighted_cols = [
        "mean_fd",
        "median_fd",
        "mean_std_dvars",
        "mean_rmsd",
        "retain_ratio",
        "motion_outlier_ratio",
        "fd_gt_threshold_ratio",
        "std_dvars_gt_threshold_ratio",
    ]
    max_cols = ["max_fd", "max_std_dvars", "max_rmsd"]
    sum_cols = ["n_volumes", "n_retained", "n_scrubbed", "n_motion_outlier_rows", "fd_gt_threshold_count", "std_dvars_gt_threshold_count"]

    for (subject, stage), cell in valid.groupby(["subject", "stage"]):
        weights = cell["n_volumes"].astype(float).to_numpy()
        row = {"subject": subject, "stage": stage, "n_runs": int(len(cell))}
        for column in sum_cols:
            row[column] = int(cell[column].sum())
        row["retain_ratio"] = float(row["n_retained"] / row["n_volumes"]) if row["n_volumes"] else float("nan")
        row["motion_outlier_ratio"] = (
            float(row["n_motion_outlier_rows"] / row["n_volumes"]) if row["n_volumes"] else float("nan")
        )
        row["fd_gt_threshold_ratio"] = (
            float(row["fd_gt_threshold_count"] / row["n_volumes"]) if row["n_volumes"] else float("nan")
        )
        row["std_dvars_gt_threshold_ratio"] = (
            float(row["std_dvars_gt_threshold_count"] / row["n_volumes"]) if row["n_volumes"] else float("nan")
        )
        for column in weighted_cols:
            if column in {"retain_ratio", "motion_outlier_ratio", "fd_gt_threshold_ratio", "std_dvars_gt_threshold_ratio"}:
                continue
            values = pd.to_numeric(cell[column], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values) & np.isfinite(weights)
            row[column] = (
                float(np.average(values[mask], weights=weights[mask])) if mask.any() else float("nan")
            )
        for column in max_cols:
            row[column] = _safe_float(pd.to_numeric(cell[column], errors="coerce").max())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_stage_paired(frame: pd.DataFrame, value_cols: list[str], *, group_col: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups = [(None, frame)] if group_col is None else list(frame.groupby(group_col))
    for group_value, cell in groups:
        for value_col in value_cols:
            pivot = cell.pivot_table(index="subject", columns="stage", values=value_col, aggfunc="first").dropna()
            if "pre" not in pivot.columns or "post" not in pivot.columns or pivot.empty:
                continue
            row = {"metric": value_col, **paired_t_summary(pivot["post"], pivot["pre"])}
            if group_col is not None:
                row[group_col] = group_value
            rows.append(row)
    return pd.DataFrame(rows)


def _wide_stage_delta(stage_frame: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    if stage_frame.empty:
        return pd.DataFrame()
    pivot = stage_frame.pivot_table(index="subject", columns="stage", values=value_cols, aggfunc="first")
    if pivot.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=pivot.index)
    for column in value_cols:
        pre = pivot[(column, "pre")] if (column, "pre") in pivot.columns else pd.Series(index=pivot.index, dtype=float)
        post = pivot[(column, "post")] if (column, "post") in pivot.columns else pd.Series(index=pivot.index, dtype=float)
        out[f"{prefix}_{column}_pre"] = pre
        out[f"{prefix}_{column}_post"] = post
        out[f"{prefix}_{column}_delta_post_minus_pre"] = post - pre
    return out.reset_index()


def _load_similarity_deltas(rsa_itemwise_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_table(rsa_itemwise_path).copy()
    if "stage" not in frame.columns and "time" in frame.columns:
        frame["stage"] = frame["time"]
    frame["stage"] = frame["stage"].astype(str).str.strip().str.lower()
    frame["condition_group"] = frame["condition"].map(_normalize_condition)
    grouped = (
        frame.groupby(["subject", "roi", "condition_group", "stage"], as_index=False)
        .agg(similarity=("similarity", "mean"))
    )
    roi_pivot = grouped.pivot_table(
        index=["subject", "roi", "condition_group"],
        columns="stage",
        values="similarity",
        aggfunc="first",
    ).reset_index()
    if "pre" not in roi_pivot.columns:
        roi_pivot["pre"] = np.nan
    if "post" not in roi_pivot.columns:
        roi_pivot["post"] = np.nan
    roi_pivot["delta_post_minus_pre"] = roi_pivot["post"] - roi_pivot["pre"]
    roi_pivot["delta_pre_minus_post"] = roi_pivot["pre"] - roi_pivot["post"]

    overall = (
        roi_pivot.groupby(["subject", "condition_group"], as_index=False)
        .agg(
            pre=("pre", "mean"),
            post=("post", "mean"),
            delta_post_minus_pre=("delta_post_minus_pre", "mean"),
            delta_pre_minus_post=("delta_pre_minus_post", "mean"),
        )
    )
    return roi_pivot, overall


def _compute_voxel_split_reliability(
    pattern_root: Path,
    roi_masks: dict[str, Path],
    subjects: list[str],
    conditions: list[str],
    *,
    n_splits: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for subject in subjects:
        subject_dir = pattern_root / subject
        if not subject_dir.exists():
            continue
        for stage in STAGE_TO_RUNS:
            for condition in conditions:
                image_path = subject_dir / f"{stage}_{condition}.nii.gz"
                meta_path = subject_dir / f"{stage}_{condition}_metadata.tsv"
                if not image_path.exists() or not meta_path.exists():
                    rows.append({
                        "subject": subject,
                        "roi": None,
                        "stage": stage,
                        "condition_group": condition,
                        "mode": "voxel_split",
                        "n_items": 0,
                        "n_voxels": 0,
                        "n_splits": n_splits,
                        "n_valid_splits": 0,
                        "rho": float("nan"),
                        "rho_sd": float("nan"),
                        "p_value": float("nan"),
                        "skip_reason": "missing_pattern_or_metadata",
                    })
                    continue
                metadata = pd.read_csv(meta_path, sep="\t")
                n_items = int(len(metadata))
                for roi_name, roi_path in roi_masks.items():
                    samples = load_masked_samples(image_path, roi_path)
                    samples = _zscore_trials(samples)
                    n_voxels = int(samples.shape[1]) if samples.ndim == 2 else 0
                    if n_items < 3:
                        rows.append({
                            "subject": subject,
                            "roi": roi_name,
                            "stage": stage,
                            "condition_group": condition,
                            "mode": "voxel_split",
                            "n_items": n_items,
                            "n_voxels": n_voxels,
                            "n_splits": n_splits,
                            "n_valid_splits": 0,
                            "rho": float("nan"),
                            "rho_sd": float("nan"),
                            "p_value": float("nan"),
                            "skip_reason": "fewer_than_three_items",
                        })
                        continue
                    if n_voxels < 6:
                        rows.append({
                            "subject": subject,
                            "roi": roi_name,
                            "stage": stage,
                            "condition_group": condition,
                            "mode": "voxel_split",
                            "n_items": n_items,
                            "n_voxels": n_voxels,
                            "n_splits": n_splits,
                            "n_valid_splits": 0,
                            "rho": float("nan"),
                            "rho_sd": float("nan"),
                            "p_value": float("nan"),
                            "skip_reason": "too_few_voxels",
                        })
                        continue

                    split_rhos: list[float] = []
                    half = n_voxels // 2
                    for _ in range(n_splits):
                        perm = rng.permutation(n_voxels)
                        left_idx = perm[:half]
                        right_idx = perm[half : half + half]
                        if len(left_idx) < 3 or len(right_idx) < 3:
                            continue
                        vec_left = neural_rdm_vector(samples[:, left_idx], metric="correlation")
                        vec_right = neural_rdm_vector(samples[:, right_idx], metric="correlation")
                        mask = np.isfinite(vec_left) & np.isfinite(vec_right)
                        if int(mask.sum()) < 3:
                            continue
                        rho, _ = stats.spearmanr(vec_left[mask], vec_right[mask])
                        rho = _safe_float(rho)
                        if math.isfinite(rho):
                            split_rhos.append(rho)

                    if not split_rhos:
                        rows.append({
                            "subject": subject,
                            "roi": roi_name,
                            "stage": stage,
                            "condition_group": condition,
                            "mode": "voxel_split",
                            "n_items": n_items,
                            "n_voxels": n_voxels,
                            "n_splits": n_splits,
                            "n_valid_splits": 0,
                            "rho": float("nan"),
                            "rho_sd": float("nan"),
                            "p_value": float("nan"),
                            "skip_reason": "no_valid_voxel_splits",
                        })
                        continue

                    rows.append({
                        "subject": subject,
                        "roi": roi_name,
                        "stage": stage,
                        "condition_group": condition,
                        "mode": "voxel_split",
                        "n_items": n_items,
                        "n_voxels": n_voxels,
                        "n_splits": n_splits,
                        "n_valid_splits": int(len(split_rhos)),
                        "rho": float(np.mean(split_rhos)),
                        "rho_sd": float(np.std(split_rhos, ddof=1)) if len(split_rhos) > 1 else 0.0,
                        "p_value": float("nan"),
                        "skip_reason": "",
                    })
    return pd.DataFrame(rows)


def _aggregate_reliability(reliability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reliability.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid = reliability[(reliability["roi"].notna()) & (reliability["skip_reason"] == "")].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    overall = (
        valid.groupby(["subject", "stage", "condition_group"], as_index=False)
        .agg(
            mean_rho=("rho", "mean"),
            min_rho=("rho", "min"),
            max_rho=("rho", "max"),
            n_roi=("roi", "nunique"),
        )
    )

    summary_rows = []
    for condition, cell in overall.groupby("condition_group"):
        pivot = cell.pivot_table(index="subject", columns="stage", values="mean_rho").dropna()
        if "pre" in pivot.columns and "post" in pivot.columns and not pivot.empty:
            summary_rows.append({
                "condition_group": condition,
                **paired_t_summary(pivot["post"], pivot["pre"]),
            })
    return overall, pd.DataFrame(summary_rows)


def _summarize_correlations(
    frame: pd.DataFrame,
    *,
    value_col: str,
    predictors: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, cell in frame.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for predictor in predictors:
            x = pd.to_numeric(cell[predictor], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(cell[value_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 6:
                rows.append({**base, "predictor": predictor, "n": int(mask.sum()), "rho": float("nan"), "p_value": float("nan")})
                continue
            rho, pvalue = stats.spearmanr(x[mask], y[mask])
            rows.append({
                **base,
                "predictor": predictor,
                "n": int(mask.sum()),
                "rho": _safe_float(rho),
                "p_value": _safe_float(pvalue),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build motion/QC and voxel split-half reliability support tables for RSA.")
    parser.add_argument("--pattern-root", type=Path, default=glm_config.BASE_DIR / "pattern_root")
    parser.add_argument("--rsa-itemwise", type=Path, default=rsa_cfg.OUTPUT_DIR / "rsa_itemwise_details.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--roi-set", default=getattr(rsa_cfg, "ROI_SET", "main_functional"))
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset like sub-01 sub-02")
    parser.add_argument("--conditions", nargs="*", default=["yy", "kj", "baseline"])
    parser.add_argument("--rois", nargs="*", default=None, help="Optional subset of ROI names.")
    parser.add_argument("--voxel-split-repeats", type=int, default=25)
    parser.add_argument("--voxel-split-seed", type=int, default=20260425)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir or _default_output_dir(args.roi_set))
    subjects = args.subjects or list(glm_config.SUBJECTS)

    roi_masks: dict[str, Path]
    manifest_path = getattr(rsa_cfg, "ROI_MANIFEST", None)
    if manifest_path and Path(manifest_path).exists():
        roi_masks = select_roi_masks(manifest_path, roi_set=args.roi_set, include_flag="include_in_rsa")
    else:
        roi_masks = dict(rsa_cfg.ROI_MASKS)
    if args.rois:
        wanted = {str(item) for item in args.rois}
        roi_masks = {name: path for name, path in roi_masks.items() if name in wanted}
    if not roi_masks:
        raise ValueError("No ROI masks available for reliability analysis.")

    run_qc = pd.DataFrame([_collect_run_qc(subject, run) for subject in subjects for run in sum(STAGE_TO_RUNS.values(), [])])
    write_table(run_qc, output_dir / "motion_qc_run_metrics.tsv")

    stage_qc = _aggregate_stage_qc(run_qc)
    if not stage_qc.empty:
        write_table(stage_qc, output_dir / "motion_qc_subject_stage.tsv")
        qc_group_summary = _summarize_stage_paired(
            stage_qc,
            [
                "mean_fd",
                "mean_std_dvars",
                "mean_rmsd",
                "retain_ratio",
                "motion_outlier_ratio",
                "fd_gt_threshold_ratio",
                "std_dvars_gt_threshold_ratio",
            ],
        )
        if not qc_group_summary.empty:
            write_table(qc_group_summary, output_dir / "motion_qc_group_summary.tsv")
        qc_value_cols = [col for col in stage_qc.columns if col not in {"subject", "stage", "n_runs"}]
        qc_wide = _wide_stage_delta(stage_qc, qc_value_cols, prefix="qc")
        write_table(qc_wide, output_dir / "motion_qc_subject_summary.tsv")
    else:
        qc_wide = pd.DataFrame()

    similarity_by_roi, similarity_overall = _load_similarity_deltas(args.rsa_itemwise)
    write_table(similarity_by_roi, output_dir / "similarity_delta_subject_roi_condition.tsv")
    write_table(similarity_overall, output_dir / "similarity_delta_subject_condition_overall.tsv")

    reliability = _compute_voxel_split_reliability(
        args.pattern_root,
        roi_masks,
        subjects,
        args.conditions,
        n_splits=max(int(args.voxel_split_repeats), 1),
        random_seed=int(args.voxel_split_seed),
    )
    write_table(reliability, output_dir / "reliability_by_roi.tsv")
    write_table(reliability, output_dir / "voxel_split_reliability_by_roi.tsv")

    reliability_overall, reliability_summary = _aggregate_reliability(reliability)
    if not reliability_overall.empty:
        write_table(reliability_overall, output_dir / "reliability_subject_stage_condition.tsv")
        write_table(reliability_overall, output_dir / "voxel_split_reliability_subject_stage_condition.tsv")
    if not reliability_summary.empty:
        write_table(reliability_summary, output_dir / "reliability_group_summary.tsv")
        write_table(reliability_summary, output_dir / "voxel_split_reliability_group_summary.tsv")

    reliability_wide = pd.DataFrame()
    if not reliability_overall.empty:
        reliability_wide = _wide_stage_delta(
            reliability_overall[["subject", "stage", "condition_group", "mean_rho"]],
            ["mean_rho"],
            prefix="reliability",
        )
        if not reliability_wide.empty:
            cond_frame = reliability_overall[["subject", "stage", "condition_group", "mean_rho"]].pivot_table(
                index="subject",
                columns=["condition_group", "stage"],
                values="mean_rho",
                aggfunc="first",
            )
            if not cond_frame.empty:
                flat = cond_frame.copy()
                flat.columns = [f"reliability_{cond}_{stage}" for cond, stage in flat.columns.to_flat_index()]
                flat = flat.reset_index()
                for condition in args.conditions:
                    pre_col = f"reliability_{condition}_pre"
                    post_col = f"reliability_{condition}_post"
                    if pre_col in flat.columns and post_col in flat.columns:
                        flat[f"reliability_{condition}_delta_post_minus_pre"] = flat[post_col] - flat[pre_col]
                reliability_wide = flat
                write_table(reliability_wide, output_dir / "reliability_subject_summary.tsv")
                write_table(reliability_wide, output_dir / "voxel_split_reliability_subject_summary.tsv")

    if not qc_wide.empty:
        qc_predictors = [col for col in qc_wide.columns if col != "subject"]
        merged_overall = similarity_overall.merge(qc_wide, on="subject", how="left")
        qc_corr = _summarize_correlations(
            merged_overall,
            value_col="delta_post_minus_pre",
            predictors=qc_predictors,
            group_cols=["condition_group"],
        )
        write_table(qc_corr, output_dir / "qc_similarity_correlations.tsv")

    if not reliability_wide.empty:
        merged_reliability = similarity_overall.merge(reliability_wide, on="subject", how="left")
        rel_predictors = [col for col in reliability_wide.columns if col != "subject"]
        rel_corr = _summarize_correlations(
            merged_reliability,
            value_col="delta_post_minus_pre",
            predictors=rel_predictors,
            group_cols=["condition_group"],
        )
        write_table(rel_corr, output_dir / "reliability_similarity_correlations.tsv")
        write_table(rel_corr, output_dir / "voxel_split_reliability_similarity_correlations.tsv")

    save_json(
        {
            "output_dir": str(output_dir),
            "rsa_itemwise": str(args.rsa_itemwise),
            "pattern_root": str(args.pattern_root),
            "n_subjects_requested": int(len(subjects)),
            "n_roi": int(len(roi_masks)),
            "roi_set": str(args.roi_set),
            "conditions": [str(item) for item in args.conditions],
            "reliability_mode": "voxel_split",
            "voxel_split_repeats": int(args.voxel_split_repeats),
        },
        output_dir / "analysis_manifest.json",
    )

    print(f"[motion_qc_reliability] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
