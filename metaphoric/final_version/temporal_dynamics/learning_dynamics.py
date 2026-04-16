"""
learning_dynamics.py

用途
- 在学习阶段（通常是 learn_yy / learn_kj）按固定窗口（例如每 20 trials）追踪：
  - RD（表征维度）
  - GPS（表征紧凑度）
  - 可选 MVPA（yy vs kj 的可解码性）
- 输出窗口级明细表，以及把“每个被试的趋势斜率”做组水平检验（斜率是否显著偏离 0）。

输入
- pattern_root: `${PATTERN_ROOT}`，每个被试目录下应包含 `learn_yy.nii.gz`、`learn_kj.nii.gz`（由 stack_patterns 生成）
- roi_dir: ROI mask 目录（`*.nii` 或 `*.nii.gz`）
- --window-size: 每窗 trials 数（默认 20）
- --threshold: RD 的累计解释方差阈值（默认 80）
- --compute-mvpa: 是否计算每窗 MVPA 准确率

输出（output_dir）
- `learning_dynamics_windows.tsv`：窗口级（subject × roi × window）指标
- `learning_dynamics_group_summary.tsv`：趋势斜率的组水平统计（one-sample t；以及 yy vs kj 斜率配对检验）
- `learning_dynamics_meta.json`：元信息
"""

from __future__ import annotations



import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, paired_t_summary, save_json, write_table  # noqa: E402
from common.pattern_metrics import gps_from_samples, load_masked_samples, dimensionality_from_samples  # noqa: E402


def classify_binary(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    x = np.vstack([samples_a, samples_b])
    y = np.array([0] * len(samples_a) + [1] * len(samples_b))
    if np.bincount(y).min() < 2:
        return float("nan")
    n_splits = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=1.0))])
    return float(np.mean(cross_val_score(clf, x, y, cv=cv)))


def slice_trials(samples: np.ndarray, start: int, end: int) -> np.ndarray:
    # samples shape: trials x voxels
    start = max(0, int(start))
    end = min(int(end), int(samples.shape[0]))
    return samples[start:end]


def per_subject_windows(n_trials: int, window_size: int) -> list[tuple[int, int, int]]:
    windows = []
    window_index = 0
    for start in range(0, n_trials, window_size):
        end = min(start + window_size, n_trials)
        if end - start < 2:
            continue
        windows.append((window_index, start, end))
        window_index += 1
    return windows


def slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return float("nan")
    x = np.arange(y.size, dtype=float)
    return float(stats.linregress(x, y).slope)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learning-stage dynamics over time windows (learn phase): RD/GPS and optional MVPA in ROIs."
    )
    parser.add_argument("pattern_root", type=Path, help="Pattern root with per-subject folders.")
    parser.add_argument("roi_dir", type=Path, help="ROI mask directory.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument("--window-size", type=int, default=20, help="Trials per time window (default: 20).")
    parser.add_argument("--threshold", type=float, default=80.0, help="RD explained-variance threshold (default: 80).")
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--time", default="learn", help="Time label to use (default: learn).")
    parser.add_argument("--compute-mvpa", action="store_true", help="Also compute yy-vs-kj MVPA accuracy per window.")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([p for p in args.pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    rows: list[dict[str, Any]] = []

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            yy_path = subject_dir / args.filename_template.format(time=args.time, condition="yy")
            kj_path = subject_dir / args.filename_template.format(time=args.time, condition="kj")
            if not yy_path.exists() or not kj_path.exists():
                continue

            yy_samples = load_masked_samples(yy_path, roi_path)
            kj_samples = load_masked_samples(kj_path, roi_path)
            windows = per_subject_windows(min(yy_samples.shape[0], kj_samples.shape[0]), args.window_size)
            if not windows:
                continue

            for win_idx, start, end in windows:
                yy_win = slice_trials(yy_samples, start, end)
                kj_win = slice_trials(kj_samples, start, end)
                if yy_win.shape[0] < 2 or kj_win.shape[0] < 2:
                    continue
                try:
                    rd_yy = float(dimensionality_from_samples(yy_win, args.threshold))
                    rd_kj = float(dimensionality_from_samples(kj_win, args.threshold))
                except Exception:
                    rd_yy = float("nan")
                    rd_kj = float("nan")
                gps_yy = float(gps_from_samples(yy_win))
                gps_kj = float(gps_from_samples(kj_win))
                acc = float("nan")
                if args.compute_mvpa:
                    acc = classify_binary(yy_win, kj_win)

                rows.append(
                    {
                        "subject": subject_dir.name,
                        "roi": roi_name,
                        "window": int(win_idx),
                        "start": int(start),
                        "end": int(end),
                        "n_trials": int(end - start),
                        "rd_yy": rd_yy,
                        "rd_kj": rd_kj,
                        "gps_yy": gps_yy,
                        "gps_kj": gps_kj,
                        "mvpa_acc": acc,
                    }
                )

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "learning_dynamics_windows.tsv")

    # Group summaries: slope across windows (per subject) then one-sample t-test.
    summaries: list[dict[str, Any]] = []
    for roi_name, roi_frame in frame.groupby("roi") if not frame.empty else []:
        for metric in ["rd_yy", "rd_kj", "gps_yy", "gps_kj", "mvpa_acc"]:
            sub_slopes = []
            for _, sf in roi_frame.groupby("subject"):
                series = sf.sort_values("window")[metric].to_numpy()
                sub_slopes.append(slope(series))
            sub_slopes = np.asarray(sub_slopes, dtype=float)
            sub_slopes = sub_slopes[np.isfinite(sub_slopes)]
            if sub_slopes.size < 3:
                continue
            t_stat, p_val = stats.ttest_1samp(sub_slopes, popmean=0.0, nan_policy="omit")
            summaries.append(
                {
                    "roi": roi_name,
                    "metric": metric,
                    "n_subjects": int(sub_slopes.size),
                    "mean_slope": float(np.mean(sub_slopes)),
                    "t": float(t_stat),
                    "p": float(p_val),
                }
            )

        # yy vs kj slope comparison for RD and GPS
        for pair in [("rd_yy", "rd_kj"), ("gps_yy", "gps_kj")]:
            a_slopes = []
            b_slopes = []
            for _, sf in roi_frame.groupby("subject"):
                sf_sorted = sf.sort_values("window")
                a_slopes.append(slope(sf_sorted[pair[0]].to_numpy()))
                b_slopes.append(slope(sf_sorted[pair[1]].to_numpy()))
            summary = {"roi": roi_name, "metric": f"slope_{pair[0]}_vs_{pair[1]}", **paired_t_summary(a_slopes, b_slopes)}
            summaries.append(summary)

    out_summary = pd.DataFrame(summaries)
    write_table(out_summary, output_dir / "learning_dynamics_group_summary.tsv")
    save_json({"n_rows": int(len(frame)), "n_group_summaries": int(len(out_summary))}, output_dir / "learning_dynamics_meta.json")


if __name__ == "__main__":
    main()
