#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_volume_betas.py
用于校验 LSS volume 模式输出的 beta NIfTI 文件格式，定位异常文件。

主要检查：
1) NIfTI 可读性与维度 (应为 3D)
2) 数据类型与 NaN/Inf
3) 与常见（众数）shape 是否一致
4) 可选：与提供的 mask shape 是否一致
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd


@dataclass
class BetaCheckResult:
    file: str
    status: str
    issues: str
    shape: str
    ndim: int
    dtype: str
    zooms: str
    nan_count: int
    inf_count: int
    zero_fraction: float
    mask_shape_match: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 volume beta NIfTI 文件格式，定位异常输出。",
    )
    parser.add_argument(
        "--beta-root",
        default=None,
        help="beta 输出根目录（将递归查找 beta_*.nii.gz）。默认读取 lss_main 的 OUTPUT_ROOT。",
    )
    parser.add_argument(
        "--pattern",
        default="beta_*.nii.gz",
        help="匹配 beta 文件的 glob 模式（默认: beta_*.nii.gz）",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="可选：MNI mask 路径，用于检测 shape 是否匹配。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选：输出 CSV 路径（默认: beta_root/beta_validation_report.csv）",
    )
    return parser.parse_args()


def collect_beta_files(beta_root: Path, pattern: str) -> list[Path]:
    return sorted(beta_root.rglob(pattern))


def compute_mode_shape(shapes: list[tuple[int, ...]]) -> tuple[int, ...] | None:
    if not shapes:
        return None
    return Counter(shapes).most_common(1)[0][0]


def check_beta_file(
    fpath: Path,
    mode_shape: tuple[int, ...] | None,
    mask_shape: tuple[int, ...] | None,
) -> BetaCheckResult:
    issues = []
    shape = ""
    ndim = 0
    dtype = ""
    zooms = ""
    nan_count = 0
    inf_count = 0
    zero_fraction = 0.0
    mask_shape_match = "n/a"

    try:
        img = nb.load(str(fpath))
        data = img.get_fdata(dtype=np.float32)
        shape = str(img.shape)
        ndim = data.ndim
        dtype = str(data.dtype)
        zooms = str(img.header.get_zooms()[:3])

        if data.ndim != 3:
            issues.append("ndim_not_3")

        if mode_shape and img.shape != mode_shape:
            issues.append("shape_mismatch_mode")

        if mask_shape is not None:
            mask_shape_match = "yes" if img.shape == mask_shape else "no"
            if img.shape != mask_shape:
                issues.append("shape_mismatch_mask")

        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        total = data.size
        zero_fraction = float(np.sum(data == 0) / total) if total else 0.0

        if nan_count > 0:
            issues.append("has_nan")
        if inf_count > 0:
            issues.append("has_inf")
    except Exception as exc:
        issues.append(f"load_error:{exc}")

    status = "ok" if not issues else "issue"

    return BetaCheckResult(
        file=str(fpath),
        status=status,
        issues=";".join(issues) if issues else "",
        shape=shape,
        ndim=ndim,
        dtype=dtype,
        zooms=zooms,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_fraction=zero_fraction,
        mask_shape_match=mask_shape_match,
    )


def main() -> None:
    args = parse_args()
    if args.beta_root:
        beta_root = Path(args.beta_root)
    else:
        from emo.glm_config import Config

        config = Config(data_space="volume")
        beta_root = config.OUTPUT_ROOT
    if not beta_root.exists():
        raise FileNotFoundError(f"beta_root not found: {beta_root}")

    beta_files = collect_beta_files(beta_root, args.pattern)
    if not beta_files:
        raise FileNotFoundError(f"No beta files found in {beta_root} with {args.pattern}")

    mask_shape = None
    if args.mask:
        mask_img = nb.load(args.mask)
        mask_shape = mask_img.shape

    shapes = []
    for fpath in beta_files:
        try:
            shapes.append(nb.load(str(fpath)).shape)
        except Exception:
            continue

    mode_shape = compute_mode_shape(shapes)
    if mode_shape:
        print(f"Detected mode shape: {mode_shape}")
    else:
        print("No valid shapes found to compute mode shape.")

    results = [
        asdict(check_beta_file(fpath, mode_shape, mask_shape))
        for fpath in beta_files
    ]
    df = pd.DataFrame(results)

    output_path = Path(args.output) if args.output else beta_root / "beta_validation_report.csv"
    df.to_csv(output_path, index=False)

    total = len(df)
    issue_count = int((df["status"] == "issue").sum())
    print(f"Checked {total} files. Issues found: {issue_count}.")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
