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
from common.roi_library import current_roi_set, default_roi_tagged_out_dir, sanitize_roi_tag, select_roi_masks


def _default_base_dir() -> Path:
    try:
        from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

        return Path(BASE_DIR)
    except Exception:
        return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _default_pattern_root(base_dir: Path) -> Path:
    return base_dir / "pattern_root"


def _resolve_roi_paths(roi_dir: Path | None, *, roi_set: str) -> list[Path]:
    if roi_dir is not None:
        if roi_dir.is_file():
            return [roi_dir]
        return sorted(roi_dir.glob("*.nii*"))

    try:
        from rsa_analysis.rsa_config import ROI_MANIFEST, ROI_MASKS  # noqa: E402

        manifest = Path(ROI_MANIFEST) if ROI_MANIFEST is not None else None
        if manifest is not None and manifest.exists():
            masks = select_roi_masks(manifest, roi_set=roi_set, include_flag="include_in_gps")
            return sorted(masks.values())
        return sorted([Path(p) for p in dict(ROI_MASKS).values() if Path(p).exists()])
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-level global pattern similarity analysis.")
    parser.add_argument("pattern_root", type=Path, nargs="?", default=None,
                        help="Defaults to {BASE_DIR}/pattern_root.")
    parser.add_argument("roi_dir", type=Path, nargs="?", default=None,
                        help="Optional ROI directory. If omitted, uses roi_library manifest + METAPHOR_ROI_SET.")
    # output_dir 为可选；默认使用 `${BASE_DIR}/paper_outputs/qc/gps_results_<ROI_SET>`，
    # 可通过环境变量 METAPHOR_GPS_OUT_DIR 强制覆盖。
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None,
                        help="Unified output root. Default: {BASE_DIR}/paper_outputs.")
    parser.add_argument("--roi-set", default=None,
                        help="Override ROI set (default: METAPHOR_ROI_SET).")
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    args = parser.parse_args()

    base_dir = _default_base_dir()
    roi_set = (args.roi_set or current_roi_set()).strip() or "main_functional"
    roi_tag = sanitize_roi_tag(roi_set)
    pattern_root = args.pattern_root or _default_pattern_root(base_dir)
    paper_root = args.paper_output_root or (base_dir / "paper_outputs")
    if args.output_dir is None:
        args.output_dir = default_roi_tagged_out_dir(
            paper_root / "qc",
            "gps_results",
            override_env="METAPHOR_GPS_OUT_DIR",
            roi_set=roi_set,
        )
    output_dir = ensure_dir(args.output_dir)
    tables_si = ensure_dir(paper_root / "tables_si")
    rows: list[dict[str, object]] = []
    roi_paths = _resolve_roi_paths(args.roi_dir, roi_set=roi_set)
    if not roi_paths:
        raise RuntimeError("No ROI masks resolved for GPS. Provide roi_dir or check roi_library/manifest.tsv include_in_gps.")
    subject_dirs = sorted([path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])

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

    summary_df = pd.DataFrame(summaries)
    write_table(summary_df, output_dir / "gps_group_summary.tsv")

    if not summary_df.empty:
        paper_df = summary_df.copy()
        rename = {
            "n": "n_subjects",
            "mean_a": "mean_yy_delta",
            "mean_b": "mean_kj_delta",
            "t": "t_yy_vs_kj_delta",
            "p": "p_yy_vs_kj_delta",
            "cohens_dz": "cohens_dz_yy_vs_kj_delta",
        }
        for src, dst in rename.items():
            if src in paper_df.columns:
                paper_df = paper_df.rename(columns={src: dst})
        paper_df.insert(1, "roi_set", roi_set)
        write_table(paper_df, tables_si / f"table_gps_{roi_tag}.tsv")
        save_json(
            {
                "roi_set": roi_set,
                "pattern_root": str(pattern_root),
                "n_subjects": int(frame["subject"].nunique()) if not frame.empty else 0,
                "n_rois": int(frame["roi"].nunique()) if not frame.empty else 0,
                "output_dir": str(output_dir),
                "paper_table": str(tables_si / f"table_gps_{roi_tag}.tsv"),
            },
            output_dir / "gps_meta.json",
        )


if __name__ == "__main__":
    main()
