#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch render significant cortical activation maps with surfplot.

输入目录下每个 L/R `.func.gii` 配对会生成一张 surfplot PNG：
- 红色 = 正效应（高 ISC / 高激活）
- 蓝色 = 负效应
- 仅渲染 FWER/FDR 命名的图（默认）
- 仅显示显著顶点（默认通过阈值把接近 0 的值透明）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets
from surfplot import Plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 surfplot 批量渲染 surface 激活图")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="包含 *_surf_L.func.gii 和 *_surf_R.func.gii 的目录",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出 PNG 目录，默认 input-dir/surfplot_png",
    )
    p.add_argument(
        "--pattern-mode",
        choices=["corrected", "all"],
        default="corrected",
        help="corrected=只处理文件名包含 fdr/fwer 的图；all=处理所有配对",
    )
    p.add_argument(
        "--min-abs",
        type=float,
        default=1e-12,
        help="把 |value| < min-abs 视作不显著并透明显示",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="固定颜色范围上界；默认按当前图的 99 分位自动估计",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def is_corrected_name(stem: str) -> bool:
    low = stem.lower()
    return "fdr" in low or "fwer" in low


def get_prefix(left_file: Path) -> str:
    suffix = "_surf_L.func.gii"
    name = left_file.name
    if not name.endswith(suffix):
        raise ValueError(f"非法左半球文件命名: {name}")
    return name[: -len(suffix)]


def robust_vmax(lh: np.ndarray, rh: np.ndarray) -> float:
    values = np.concatenate([lh, rh])
    nz = values[np.abs(values) > 0]
    if nz.size == 0:
        return 0.05
    return max(0.05, float(np.quantile(np.abs(nz), 0.99)))


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir = args.output_dir or (input_dir / "surfplot_png")
    output_dir.mkdir(parents=True, exist_ok=True)

    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    left_files = sorted(input_dir.glob("*_surf_L.func.gii"))
    if not left_files:
        raise FileNotFoundError(f"在 {input_dir} 未找到 *_surf_L.func.gii")

    rendered = 0
    skipped = 0

    for lf in left_files:
        prefix = get_prefix(lf)
        if args.pattern_mode == "corrected" and not is_corrected_name(prefix):
            skipped += 1
            continue

        rf = input_dir / f"{prefix}_surf_R.func.gii"
        if not rf.exists():
            print(f"[SKIP] 缺少右半球配对: {rf.name}")
            skipped += 1
            continue

        lh = np.asarray(nib.load(str(lf)).darrays[0].data, dtype=float)
        rh = np.asarray(nib.load(str(rf)).darrays[0].data, dtype=float)

        lh[np.abs(lh) < args.min_abs] = 0.0
        rh[np.abs(rh) < args.min_abs] = 0.0

        vmax = float(args.vmax) if args.vmax is not None else robust_vmax(lh, rh)

        p = Plot(
            fsavg.infl_left,
            fsavg.infl_right,
            views=["lateral", "medial"],
            layout="grid",
            size=(1200, 900),
        )
        p.add_layer(
            {"left": lh, "right": rh},
            cmap="RdBu_r",
            color_range=(-vmax, vmax),
            cbar=True,
            cbar_label="Effect size (ISC / activation)",
            zero_transparent=True,
        )

        fig = p.build()
        out_png = output_dir / f"{prefix}_surfplot.png"
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        rendered += 1
        print(f"[OK] {out_png}")

    print(f"完成：渲染 {rendered} 张，跳过 {skipped} 张。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
