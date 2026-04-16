"""
lateralization_analysis.py

用途
- 计算偏侧化指数 LI = (L - R) / (L + R)，用于回答“左右半球贡献是否随学习改变”的争议点。
- 当前实现基于 patterns（pre/post 的 4D beta-series）在 ROI 内的均值激活计算 LI。

输入
- pattern_root: `${PATTERN_ROOT}`（应有 `pre_yy.nii.gz` 等文件）
- roi_dir: ROI masks（要求能推断 L/R 配对；支持 L-/R-、Left/Right、lh_/rh_ 命名）
- --time / --conditions / --filename-template

输出（output_dir）
- `lateralization_subject_metrics.tsv`：逐被试的 LI 指标
- `lateralization_group_summary.tsv`：pre vs post 的配对检验摘要
- `lateralization_meta.json`
"""

from __future__ import annotations



import argparse
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

from common.final_utils import ensure_dir, paired_t_summary, save_json, write_table  # noqa: E402
from common.pattern_metrics import load_masked_samples  # noqa: E402


def mean_trial_activation(samples: np.ndarray) -> float:
    # samples: trials x voxels
    if samples.size == 0:
        return float("nan")
    return float(np.nanmean(samples))


def lateralization_index(left: float, right: float) -> float:
    denom = left + right
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float((left - right) / denom)


def infer_pairs(roi_paths: list[Path]) -> list[tuple[Path, Path, str]]:
    """
    Try to infer L/R ROI pairs from filenames.
    Supported patterns:
    - L-XXX vs R-XXX
    - LeftXXX vs RightXXX
    - lh_XXX vs rh_XXX
    """
    by_name = {p.stem.replace(".nii", ""): p for p in roi_paths}
    pairs = []
    used = set()
    for name, path in by_name.items():
        if name in used:
            continue
        candidates = []
        if name.startswith("L-"):
            candidates.append(("R-" + name[2:], f"{name[2:]}"))
        if "Left" in name:
            candidates.append((name.replace("Left", "Right"), name.replace("Left", "")))
        if name.startswith("lh_"):
            candidates.append(("rh_" + name[3:], name[3:]))
        for other_name, base in candidates:
            if other_name in by_name:
                pairs.append((path, by_name[other_name], base))
                used.add(name)
                used.add(other_name)
                break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute hemispheric lateralization index (LI) from ROI mean activations.")
    parser.add_argument("pattern_root", type=Path, help="Pattern root with per-subject phase_condition 4D files.")
    parser.add_argument("roi_dir", type=Path, help="ROI directory containing L/R masks.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--time", nargs="*", default=["pre", "post"], help="Time labels (default: pre post).")
    parser.add_argument("--conditions", nargs="*", default=["yy", "kj"], help="Condition labels (default: yy kj).")
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([p for p in args.pattern_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    roi_pairs = infer_pairs(roi_paths)
    if not roi_pairs:
        raise RuntimeError("No L/R ROI pairs inferred. Please name masks with L-/R- or Left/Right or lh_/rh_.")

    rows: list[dict[str, Any]] = []
    for left_roi, right_roi, base_name in roi_pairs:
        for subject_dir in subject_dirs:
            for time in args.time:
                for condition in args.conditions:
                    img_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    if not img_path.exists():
                        continue
                    left_samples = load_masked_samples(img_path, left_roi)
                    right_samples = load_masked_samples(img_path, right_roi)
                    l_mean = mean_trial_activation(left_samples)
                    r_mean = mean_trial_activation(right_samples)
                    rows.append(
                        {
                            "subject": subject_dir.name,
                            "roi_pair": base_name,
                            "time": time,
                            "condition": condition,
                            "left_roi": left_roi.name,
                            "right_roi": right_roi.name,
                            "left_mean": l_mean,
                            "right_mean": r_mean,
                            "li": lateralization_index(l_mean, r_mean),
                        }
                    )

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "lateralization_subject_metrics.tsv")

    summaries: list[dict[str, Any]] = []
    for (roi_pair, condition), sub in frame.groupby(["roi_pair", "condition"]):
        pivot = sub.pivot(index="subject", columns="time", values="li").dropna()
        if {"pre", "post"}.issubset(pivot.columns):
            summary = {"roi_pair": roi_pair, "condition": condition, **paired_t_summary(pivot["post"], pivot["pre"])}
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    write_table(summary_df, output_dir / "lateralization_group_summary.tsv")
    save_json({"n_rows": int(len(frame)), "n_pairs": int(len(roi_pairs))}, output_dir / "lateralization_meta.json")


if __name__ == "__main__":
    main()
