"""
rd_analysis.py

用途
- 在 ROI 内计算表征维度 RD（Representational Dimensionality）：
  对每个 subject、每个 time（pre/post）、每个 condition（yy/kj）的 4D patterns 提取 trials×voxels 样本，
  默认使用 participation ratio（基于 voxel covariance eigenspectrum 的软维度）；
  也保留达到给定解释方差阈值所需的 PCA 成分数版本作为敏感性分析。

输入
- pattern_root: `${PATTERN_ROOT}`（每个 sub-xx 一个目录，包含 `pre_yy.nii.gz` 等）
- roi_dir: ROI masks 目录（`*.nii` 或 `*.nii.gz`）
- output_dir: 输出目录

输出（output_dir）
- `rd_subject_metrics.tsv`：subject×roi×time×condition 的 RD 值
- `rd_group_summary.tsv`：组水平统计（post vs pre；yy vs kj；以及差异中的差异）
- `rd_summary.json`：包含 ANOVA（若 statsmodels 可用）与摘要统计
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
from common.pattern_metrics import (
    dimensionality_from_samples,
    load_masked_samples,
    participation_ratio_from_samples,
    rd_from_covariance,
)
from common.roi_library import current_roi_set, default_roi_tagged_out_dir, sanitize_roi_tag, select_roi_masks


def run_anova(frame: pd.DataFrame) -> dict[str, float]:
    try:
        from statsmodels.stats.anova import AnovaRM
    except Exception:
        return {}
    renamed = frame.rename(columns={"subject": "subject", "condition": "condition", "time": "time", "value": "value"})
    result = AnovaRM(renamed, depvar="value", subject="subject", within=["condition", "time"]).fit()
    table = result.anova_table.reset_index().rename(columns={"index": "effect"})
    if "Pr > F" not in table.columns:
        return {}
    output = {}
    for _, row in table.iterrows():
        output[f"anova_{row['effect']}_F"] = float(row["F Value"])
        output[f"anova_{row['effect']}_p"] = float(row["Pr > F"])
    return output


def _default_base_dir() -> Path:
    try:
        from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

        return Path(BASE_DIR)
    except Exception:
        return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _default_pattern_root(base_dir: Path) -> Path:
    return base_dir / "pattern_root"


def _resolve_roi_paths(roi_dir: Path | None, *, roi_set: str) -> list[Path]:
    """
    Prefer ROI library manifest when available so ROI sets remain consistent across analyses.
    Fallback to the provided roi_dir directory.
    """
    if roi_dir is not None:
        if roi_dir.is_file():
            return [roi_dir]
        return sorted(roi_dir.glob("*.nii*"))

    try:
        from rsa_analysis.rsa_config import ROI_MANIFEST, ROI_MASKS  # noqa: E402

        manifest = Path(ROI_MANIFEST) if ROI_MANIFEST is not None else None
        if manifest is not None and manifest.exists():
            masks = select_roi_masks(manifest, roi_set=roi_set, include_flag="include_in_rd")
            return sorted(masks.values())
        return sorted([Path(p) for p in dict(ROI_MASKS).values() if Path(p).exists()])
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-level representational dimensionality analysis.")
    parser.add_argument("pattern_root", type=Path, nargs="?", default=None,
                        help="Defaults to {BASE_DIR}/pattern_root.")
    parser.add_argument("roi_dir", type=Path, nargs="?", default=None,
                        help="Optional ROI directory. If omitted, uses roi_library manifest + METAPHOR_ROI_SET.")
    # output_dir 为可选；默认使用 `${BASE_DIR}/paper_outputs/qc/rd_results_<ROI_SET>`，
    # 可通过环境变量 METAPHOR_RD_OUT_DIR 强制覆盖。
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None,
                        help="Unified output root. Default: {BASE_DIR}/paper_outputs.")
    parser.add_argument("--roi-set", default=None,
                        help="Override ROI set (default: METAPHOR_ROI_SET).")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument(
        "--rd-mode",
        choices=["participation_ratio", "covariance", "rdm"],
        default="participation_ratio",
    )
    args = parser.parse_args()

    base_dir = _default_base_dir()
    roi_set = (args.roi_set or current_roi_set()).strip() or "main_functional"
    roi_tag = sanitize_roi_tag(roi_set)
    pattern_root = args.pattern_root or _default_pattern_root(base_dir)
    paper_root = args.paper_output_root or (base_dir / "paper_outputs")

    if args.output_dir is None:
        # Default to paper_outputs/qc and keep ROI-tagged subfolders to avoid collisions.
        args.output_dir = default_roi_tagged_out_dir(
            paper_root / "qc",
            "rd_results",
            override_env="METAPHOR_RD_OUT_DIR",
            roi_set=roi_set,
        )
    output_dir = ensure_dir(args.output_dir)
    tables_si = ensure_dir(paper_root / "tables_si")

    rows: list[dict[str, object]] = []
    roi_paths = _resolve_roi_paths(args.roi_dir, roi_set=roi_set)
    if not roi_paths:
        raise RuntimeError("No ROI masks resolved for RD. Provide roi_dir or check roi_library/manifest.tsv include_in_rd.")
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
                    if args.rd_mode == "participation_ratio":
                        value = participation_ratio_from_samples(samples)
                    elif args.rd_mode == "covariance":
                        value = rd_from_covariance(samples, args.threshold)
                    else:
                        value = dimensionality_from_samples(samples, args.threshold)
                    rows.append({
                        "subject": subject_dir.name,
                        "roi": roi_name,
                        "time": time,
                        "condition": condition,
                        "value": value,
                    })

    frame = pd.DataFrame(rows)
    write_table(frame, output_dir / "rd_subject_metrics.tsv")

    summaries = []
    for roi_name, roi_frame in frame.groupby("roi"):
        interaction = difference_in_differences(roi_frame)
        summary = {
            "roi": roi_name,
            **paired_t_summary(interaction["yy_delta"], interaction["kj_delta"]),
            **run_anova(roi_frame),
        }
        summaries.append(summary)
        save_json(summary, output_dir / f"{roi_name}_rd_summary.json")

    summary_df = pd.DataFrame(summaries)
    write_table(summary_df, output_dir / "rd_group_summary.tsv")

    # Paper/SI table with clear column names and ROI set tag.
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
        write_table(paper_df, tables_si / f"table_rd_{roi_tag}.tsv")
        save_json(
            {
                "roi_set": roi_set,
                "pattern_root": str(pattern_root),
                "n_subjects": int(frame["subject"].nunique()) if not frame.empty else 0,
                "n_rois": int(frame["roi"].nunique()) if not frame.empty else 0,
                "threshold": float(args.threshold),
                "rd_mode": args.rd_mode,
                "output_dir": str(output_dir),
                "paper_table": str(tables_si / f"table_rd_{roi_tag}.tsv"),
            },
            output_dir / "rd_meta.json",
        )


if __name__ == "__main__":
    main()
