"""
gps_analysis.py

用途
- 在 ROI 内计算 GPS（Global Pattern Similarity，全局模式相似性/紧凑度）：
  对 trials×voxels 样本计算 trial-by-trial 相关（Fisher-Z），每个 trial 与其他 trials 的平均相似度即 GPS。
  GPS 越高，说明同条件 trials 的神经模式越一致/越紧凑。

输入
- pattern_root: `${PATTERN_ROOT}`（每个 sub-xx 一个目录，包含 `pre_yy.nii.gz` 等）
- roi_dir: ROI masks 目录
- output_dir: 输出目录

输出（output_dir）
- `gps_subject_metrics.tsv`：subject×roi×time×condition 的 GPS 值
- `gps_group_summary.tsv`：组水平统计（post vs pre；yy vs kj；以及差异中的差异）
- `gps_summary.json`：摘要统计
"""

from __future__ import annotations



import argparse
import os
from pathlib import Path
import sys

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

from common.final_utils import difference_in_differences, ensure_dir, paired_t_summary, save_json, write_table
from common.pattern_metrics import gps_from_samples, load_masked_samples
from common.roi_library import default_roi_tagged_out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-level global pattern similarity analysis.")
    parser.add_argument("pattern_root", type=Path)
    parser.add_argument("roi_dir", type=Path)
    # output_dir 为可选；默认使用 `${PYTHON_METAPHOR_ROOT}/gps_results_<ROI_SET>`，
    # 可通过环境变量 METAPHOR_GPS_OUT_DIR 强制覆盖。
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    args = parser.parse_args()

    if args.output_dir is None:
        base_dir = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
        args.output_dir = default_roi_tagged_out_dir(
            base_dir, "gps_results", override_env="METAPHOR_GPS_OUT_DIR"
        )
    output_dir = ensure_dir(args.output_dir)
    rows: list[dict[str, object]] = []
    roi_paths = sorted(args.roi_dir.glob("*.nii*"))
    subject_dirs = sorted([path for path in args.pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

    for roi_path in roi_paths:
        roi_name = roi_path.stem.replace(".nii", "")
        for subject_dir in subject_dirs:
            for time in ["pre", "post"]:
                for condition in ["yy", "kj"]:
                    image_path = subject_dir / args.filename_template.format(time=time, condition=condition)
                    if not image_path.exists():
                        continue
                    samples = load_masked_samples(image_path, roi_path)
                    rows.append({
                        "subject": subject_dir.name,
                        "roi": roi_name,
                        "time": time,
                        "condition": condition,
                        "value": gps_from_samples(samples),
                    })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "gps_subject_metrics.tsv")

    summaries = []
    for roi_name, roi_frame in frame.groupby("roi"):
        interaction = difference_in_differences(roi_frame)
        summary = {"roi": roi_name, **paired_t_summary(interaction["yy_delta"], interaction["kj_delta"])}
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_name}_gps_summary.json")

    write_table(pd.DataFrame(summaries), output_dir / "gps_group_summary.tsv")


if __name__ == "__main__":
    main()
