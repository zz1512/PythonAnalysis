"""
effective_connectivity.py

用途
- 用“简化有效连接”思路估计方向性信息流（探索性分析）：
  对每个 ROI 的 trial-wise beta-series（每 trial 的 ROI 均值）做多元回归，
  把 `from_roi` 作为自变量、`to_roi` 作为因变量、其余 ROI 作为协变量，
  回归系数可被解释为“控制其他 ROI 后 from 对 to 的独立预测力”。

输入
- pattern_root: `${PATTERN_ROOT}`（需要 `{time}_{condition}.nii.gz`）
- roi_dir: ROI masks（多个 ROI）
- --time: pre/post/learn（默认 learn）
- --condition: 默认 yy

输出（output_dir）
- `effective_connectivity_edges.tsv`：subject-level 有向边系数（from→to）
- `effective_connectivity_group_summary.tsv`：组水平 one-sample t 检验（beta 是否偏离 0）
- `effective_connectivity_meta.json`

注意
- 这是探索性近似方法，不等价于 DCM；适合作为补充分析或候选机制线索。
"""

from __future__ import annotations



import argparse
from itertools import permutations
from pathlib import Path
import sys
from typing import Any

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

from common.final_utils import ensure_dir, one_sample_t_summary, save_json, write_table  # noqa: E402
from common.pattern_metrics import load_masked_samples  # noqa: E402


def beta_series_mean(samples: np.ndarray) -> np.ndarray:
    # samples: trials x voxels -> trial-wise mean beta
    if samples.size == 0:
        return np.array([], dtype=float)
    return np.nanmean(samples, axis=1)


def partial_regression_coef(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    coef for x in y ~ x + z (z can be 2D: n x k).
    Implemented via least squares.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    valid = np.isfinite(x[:, 0]) & np.isfinite(y[:, 0]) & np.all(np.isfinite(z), axis=1)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    if x.shape[0] < 5:
        return float("nan")
    design = np.hstack([np.ones((x.shape[0], 1)), x, z])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified effective connectivity via multivariate regression on beta-series.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--time", default="learn", choices=["pre", "post", "learn"])
    parser.add_argument("--condition", default="yy", help="Which condition file to use (default: yy).")
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([p for p in args.pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    rows: list[dict[str, Any]] = []
    roi_names = [p.stem.replace(".nii", "") for p in roi_paths]

    for subject_dir in subject_dirs:
        img_path = subject_dir / args.filename_template.format(time=args.time, condition=args.condition)
        if not img_path.exists():
            continue

        series = []
        for roi_path in roi_paths:
            samples = load_masked_samples(img_path, roi_path)
            series.append(beta_series_mean(samples))
        min_len = min((len(s) for s in series if len(s) > 0), default=0)
        if min_len < 10:
            continue
        series = [s[:min_len] for s in series]
        mat = np.column_stack(series)  # n_trials x n_rois

        for i, j in permutations(range(len(roi_names)), 2):
            y = mat[:, j]
            x = mat[:, i]
            others = np.delete(mat, [j, i], axis=1)
            coef = partial_regression_coef(x, y, others)
            rows.append(
                {
                    "subject": subject_dir.name,
                    "from_roi": roi_names[i],
                    "to_roi": roi_names[j],
                    "beta": coef,
                    "time": args.time,
                    "condition": args.condition,
                }
            )

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "effective_connectivity_edges.tsv")

    summaries = []
    for (from_roi, to_roi), sub in frame.groupby(["from_roi", "to_roi"]) if not frame.empty else []:
        summary = {"from_roi": from_roi, "to_roi": to_roi, **one_sample_t_summary(sub["beta"].to_numpy(), popmean=0.0)}
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    write_table(summary_df, output_dir / "effective_connectivity_group_summary.tsv")
    save_json({"n_edges": int(len(frame)), "n_pairs": int(len(summary_df))}, output_dir / "effective_connectivity_meta.json")


if __name__ == "__main__":
    main()
