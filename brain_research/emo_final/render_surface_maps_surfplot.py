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
from nilearn.surface import load_surf_mesh
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
    p.add_argument(
        "--surface-type",
        choices=["auto", "fsaverage", "fslr"],
        default="auto",
        help="surface 模板类型；auto 会按输入顶点数自动判断",
    )
    p.add_argument(
        "--mesh",
        choices=["auto", "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fslr32k"],
        default="auto",
        help="surface 网格分辨率；auto 会按输入顶点数自动匹配",
    )
    p.add_argument(
        "--fslr-left-surf",
        type=Path,
        default=None,
        help="fslr 左半球几何表面文件（如 *.surf.gii）",
    )
    p.add_argument(
        "--fslr-right-surf",
        type=Path,
        default=None,
        help="fslr 右半球几何表面文件（如 *.surf.gii）",
    )
    p.add_argument(
        "--online-fslr",
        choices=["auto", "never"],
        default="auto",
        help="当本地未找到 fslr 几何表面时，是否尝试在线下载标准 32k 网格（默认 auto）",
    )
    return p.parse_args()


def is_corrected_name(stem: str) -> bool:
    low = stem.lower()
    return "fdr" in low or "fwer" in low


HEMI_MARKERS = [
    ("_surf_L", "left"),
    ("_surf_R", "right"),
    ("_hemi-L", "left"),
    ("_hemi-R", "right"),
    ("hemi-L", "left"),
    ("hemi-R", "right"),
    ("_lh", "left"),
    ("_rh", "right"),
    (".lh", "left"),
    (".rh", "right"),
    ("-lh", "left"),
    ("-rh", "right"),
    ("_L", "left"),
    ("_R", "right"),
    (".L", "left"),
    (".R", "right"),
    ("-L", "left"),
    ("-R", "right"),
]


def discover_func_pairs(input_dir: Path) -> list[tuple[str, Path, Path]]:
    """自动发现并配对 L/R .func.gii 文件。"""
    pairs: dict[str, dict[str, Path]] = {}
    files = sorted(input_dir.glob("*.func.gii"))

    for f in files:
        base = f.name[: -len(".func.gii")]
        matched = False
        for marker, side in HEMI_MARKERS:
            if base.endswith(marker):
                prefix = base[: -len(marker)]
                if not prefix:
                    continue
                pairs.setdefault(prefix, {})[side] = f
                matched = True
                break
        if not matched:
            continue

    out: list[tuple[str, Path, Path]] = []
    for prefix in sorted(pairs.keys()):
        lr = pairs[prefix]
        if "left" in lr and "right" in lr:
            out.append((prefix, lr["left"], lr["right"]))
        elif "left" in lr:
            print(f"[SKIP] 缺少右半球配对: {lr['left'].name}")
        elif "right" in lr:
            print(f"[SKIP] 缺少左半球配对: {lr['right'].name}")
    return out


def robust_vmax(lh: np.ndarray, rh: np.ndarray) -> float:
    values = np.concatenate([lh, rh])
    nz = values[np.abs(values) > 0]
    if nz.size == 0:
        return 0.05
    return max(0.05, float(np.quantile(np.abs(nz), 0.99)))


MESH_VERTICES = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fslr32k": 32492,
}


def infer_mesh_from_vertices(n_vertices: int, surface_type: str) -> str:
    if surface_type == "fslr":
        if n_vertices == MESH_VERTICES["fslr32k"]:
            return "fslr32k"
        raise ValueError(
            f"surface-type=fslr 时仅支持每半球 {MESH_VERTICES['fslr32k']} 顶点，当前为 {n_vertices}"
        )

    for mesh, n in MESH_VERTICES.items():
        if mesh.startswith("fsaverage") and n_vertices == n:
            return mesh

    if surface_type == "auto" and n_vertices == MESH_VERTICES["fslr32k"]:
        return "fslr32k"

    supported = ", ".join(f"{k}={v}" for k, v in MESH_VERTICES.items())
    raise ValueError(
        f"无法根据顶点数 {n_vertices} 匹配网格。支持: {supported}"
    )


def _is_left_surface_name(name: str) -> bool:
    low = name.lower()
    return any(x in low for x in ["hemi-l", "_l.", ".l.", "-l.", "_lh", ".lh", "-lh"])


def _is_right_surface_name(name: str) -> bool:
    low = name.lower()
    return any(x in low for x in ["hemi-r", "_r.", ".r.", "-r.", "_rh", ".rh", "-rh"])


def auto_discover_fslr_surfaces(input_dir: Path) -> tuple[Path | None, Path | None]:
    """按输入路径自动查找 fslr 左右半球几何表面文件。"""
    # 优先在当前目录与上级目录查找；每层做受限递归，覆盖常见目录组织
    candidate_dirs = [input_dir]
    for p in input_dir.parents:
        candidate_dirs.append(p)
        if len(candidate_dirs) >= 6:
            break

    # 常见几何文件关键字
    geom_keywords = ("midthickness", "inflated", "very_inflated", "pial", "white")
    exts = (".surf.gii", ".gii")
    likely_subdirs = (
        "surf",
        "surface",
        "surfaces",
        "mesh",
        "meshes",
        "fslr",
        "fs_lr",
        "conte69",
    )

    left_candidates: list[Path] = []
    right_candidates: list[Path] = []

    def _iter_gii_files(base: Path):
        # 1) 先看当前目录
        for f in base.glob("*.gii"):
            yield f
        # 2) 再看明显相关的子目录（最多向下 2 层）
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            sub_name = sub.name.lower()
            if not any(k in sub_name for k in likely_subdirs):
                continue
            for f in sub.rglob("*.gii"):
                rel_depth = len(f.relative_to(sub).parts)
                if rel_depth <= 3:
                    yield f

    for d in candidate_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for f in _iter_gii_files(d):
            low = f.name.lower()
            if ".func." in low:
                continue
            if not any(low.endswith(ext) for ext in exts):
                continue
            if not any(k in low for k in geom_keywords):
                continue
            if _is_left_surface_name(low):
                left_candidates.append(f)
            elif _is_right_surface_name(low):
                right_candidates.append(f)

    def _pick_best(cands: list[Path]) -> Path | None:
        if not cands:
            return None
        # 先选 midthickness，再选 inflated，再选 very_inflated
        def score(p: Path) -> tuple[int, int]:
            n = p.name.lower()
            if "midthickness" in n:
                pri = 0
            elif "inflated" in n and "very_inflated" not in n:
                pri = 1
            elif "very_inflated" in n:
                pri = 2
            else:
                pri = 3
            # 更靠近 input_dir 的路径优先
            try:
                dist = len(p.relative_to(input_dir).parts)
            except ValueError:
                dist = 99
            return (pri, dist)

        return sorted(cands, key=score)[0]

    return _pick_best(left_candidates), _pick_best(right_candidates)


def load_fslr_surfaces(
    input_dir: Path,
    left_surf: Path | None,
    right_surf: Path | None,
    online_fslr: str = "auto",
):
    if left_surf is None or right_surf is None:
        auto_left, auto_right = auto_discover_fslr_surfaces(input_dir)
        left_surf = left_surf or auto_left
        right_surf = right_surf or auto_right

    if left_surf is None or right_surf is None:
        if online_fslr == "auto":
            try:
                # 仅在需要时导入，避免对非 fslr 场景引入硬依赖
                from neuromaps import datasets as nm_datasets

                fslr = nm_datasets.fetch_fslr(density="32k")
                # neuromaps 返回结构在不同版本可能略有差异，做兼容处理
                if isinstance(fslr, dict) and "midthickness" in fslr:
                    mid = fslr["midthickness"]
                    if isinstance(mid, (list, tuple)) and len(mid) >= 2:
                        left_surf = Path(mid[0])
                        right_surf = Path(mid[1])
                if left_surf is not None and right_surf is not None:
                    print(
                        "[INFO] 本地未找到 fslr 几何表面，已自动下载并使用标准 fsLR-32k midthickness。"
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 在线获取标准 fslr32k 失败：{e}")

    if left_surf is None or right_surf is None:
        raise ValueError(
            "fslr 自动查找失败：请显式提供 --fslr-left-surf 与 --fslr-right-surf，"
            "或把几何 .gii（midthickness/inflated/pial/white）放在输入目录、上级目录"
            "或它们的 surface/surf/mesh/fslr 子目录。"
            "也可安装 neuromaps 并使用默认 --online-fslr auto 在线获取标准 fsLR-32k 网格。"
        )
    if not left_surf.exists() or not right_surf.exists():
        raise FileNotFoundError(
            f"fslr 表面文件不存在: left={left_surf}, right={right_surf}"
        )
    print(f"[INFO] fslr 几何表面: left={left_surf}, right={right_surf}")
    left_mesh = load_surf_mesh(str(left_surf))
    right_mesh = load_surf_mesh(str(right_surf))
    return left_mesh, right_mesh


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir = args.output_dir or (input_dir / "surfplot_png")
    output_dir.mkdir(parents=True, exist_ok=True)

    surf_mesh = None
    current_mesh = None

    func_pairs = discover_func_pairs(input_dir)
    if not func_pairs:
        raise FileNotFoundError(
            f"在 {input_dir} 未找到可配对的 L/R .func.gii 文件（支持 *_surf_L/_R 等常见命名）"
        )

    rendered = 0
    skipped = 0

    for prefix, lf, rf in func_pairs:
        if args.pattern_mode == "corrected" and not is_corrected_name(prefix):
            skipped += 1
            continue

        lh = np.asarray(nib.load(str(lf)).darrays[0].data, dtype=float)
        rh = np.asarray(nib.load(str(rf)).darrays[0].data, dtype=float)
        if lh.shape != rh.shape:
            raise ValueError(
                f"左右半球顶点数不一致: {lf.name}={lh.shape}, {rf.name}={rh.shape}"
            )

        mesh = args.mesh if args.mesh != "auto" else infer_mesh_from_vertices(
            lh.size, args.surface_type
        )
        if args.surface_type == "fsaverage" and mesh == "fslr32k":
            raise ValueError(
                "surface-type=fsaverage 不能使用 fslr32k 顶点数，请改为 --surface-type fslr"
            )
        if args.surface_type == "fslr" and mesh != "fslr32k":
            raise ValueError("surface-type=fslr 仅支持 fslr32k 网格")

        if mesh != current_mesh:
            print(f"[INFO] 使用网格: {mesh}（每半球 {lh.size} 顶点）")
            if mesh == "fslr32k":
                surf_mesh = load_fslr_surfaces(
                    input_dir,
                    args.fslr_left_surf,
                    args.fslr_right_surf,
                    args.online_fslr,
                )
            else:
                fsavg = datasets.fetch_surf_fsaverage(mesh=mesh)
                surf_mesh = (fsavg.infl_left, fsavg.infl_right)
            current_mesh = mesh

        lh[np.abs(lh) < args.min_abs] = 0.0
        rh[np.abs(rh) < args.min_abs] = 0.0

        vmax = float(args.vmax) if args.vmax is not None else robust_vmax(lh, rh)

        p = Plot(
            surf_mesh[0],
            surf_mesh[1],
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
