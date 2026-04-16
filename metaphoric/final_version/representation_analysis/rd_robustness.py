"""
rd_robustness.py

用途
- RD（Representational Dimensionality）稳健性检验套件（ROI-level）：
  通过改变解释方差阈值、是否做 demean、是否做 trial 数等量化重采样，
  检验 RD 结论是否对这些实现选择敏感。

输入
- pattern_root: `${PATTERN_ROOT}`（包含 pre/post 的 4D patterns）
- roi_dir: ROI masks
- --thresholds: 解释方差阈值列表（默认 70/75/80/85/90）
- --demean: 是否对 yy/kj 样本做合并去均值（消除均值差异驱动）
- --equalize: 是否把两条件 trial 数等量化（避免 trial 数差异驱动）

输出（output_dir）
- `rd_robustness_metrics.tsv`：逐 subject×roi×time×threshold×condition 的 RD 值
"""

from __future__ import annotations



import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, write_table
from common.pattern_metrics import dimensionality_from_samples, load_masked_samples


def maybe_demean(samples_a: np.ndarray, samples_b: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return samples_a, samples_b
    stacked = np.vstack([samples_a, samples_b])
    stacked = stacked - stacked.mean(axis=0, keepdims=True)
    return stacked[: samples_a.shape[0]], stacked[samples_a.shape[0] :]


def equalized_rd(samples: np.ndarray, threshold: float, target_trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if samples.shape[0] == target_trials:
        return dimensionality_from_samples(samples, threshold)
    values = []
    for _ in range(200):
        choice = rng.choice(samples.shape[0], size=target_trials, replace=False)
        values.append(dimensionality_from_samples(samples[choice], threshold))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="RD robustness analyses across thresholds and equalized resampling.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[70, 75, 80, 85, 90])
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--demean", action="store_true")
    parser.add_argument("--equalize", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rows = []
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            for time in ["pre", "post"]:
                yy_path = subject_dir / args.filename_template.format(time=time, condition="yy")
                kj_path = subject_dir / args.filename_template.format(time=time, condition="kj")
                if not yy_path.exists() or not kj_path.exists():
                    continue
                yy_samples = load_masked_samples(yy_path, roi_path)
                kj_samples = load_masked_samples(kj_path, roi_path)
                yy_samples, kj_samples = maybe_demean(yy_samples, kj_samples, args.demean)
                target_trials = min(yy_samples.shape[0], kj_samples.shape[0])
                for threshold in args.thresholds:
                    if args.equalize:
                        yy_value = equalized_rd(yy_samples, threshold, target_trials, seed=42)
                        kj_value = equalized_rd(kj_samples, threshold, target_trials, seed=84)
                    else:
                        yy_value = dimensionality_from_samples(yy_samples, threshold)
                        kj_value = dimensionality_from_samples(kj_samples, threshold)
                    rows.append({"subject": subject_dir.name, "roi": roi_name, "time": time, "threshold": threshold, "condition": "yy", "value": yy_value})
                    rows.append({"subject": subject_dir.name, "roi": roi_name, "time": time, "threshold": threshold, "condition": "kj", "value": kj_value})

    write_table(pd.DataFrame(rows), output_dir / "rd_robustness_metrics.tsv")


if __name__ == "__main__":
    main()
