#!/usr/bin/env python3
"""S1: 海马亚区分割（head/body/tail × 左右半球 = 6 个 subROI）。

优先使用 FreeSurfer FS60 hipposubfields 输出；若某被试 FS60 不可用，
则按 MNI y 轴三等分 meta_L/R_hippocampus mask 做 fallback。

输出：
    s1_segmentation/masks/<subject>/<subfield>.nii.gz
    s1_segmentation/subfield_manifest.tsv
    s1_segmentation/s1_log.txt

本脚本只做代码编写 + 规格落地；分析在数据机执行。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_subfield import (
    add_common_args,
    assemble_subfield_label,
    default_config,
    discover_subjects,
    iter_subfield_labels,
    log_text,
    module_dir,
    safe_output_path,
    save_segment_mask,
    split_mask_along_y,
    write_outputs,
)

MODULE = "s1_segmentation"
MIN_VOXEL_WARN = 30
MIN_VOXEL_FAIL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="显式被试列表；缺省时从 fs60-root 或 mask-root 扫描 sub-* 目录。")
    parser.add_argument("--mask-pattern", default="meta_{hemi}_hippocampus.nii.gz",
                        help="整体海马 mask 文件名模板，占位符 {hemi} 为 L/R。")
    parser.add_argument("--fs60-subfield-pattern", default="{subject}/mri/lh.hippoSfLabels-T1.v10.mgz",
                        help="FS60 subfield 标签文件路径模板（相对 fs60-root），占位符 {subject} 与 {hemi}。")
    return parser.parse_args()


def qc_status_for(n_voxels: int) -> str:
    if n_voxels < MIN_VOXEL_FAIL:
        return "qc_fail"
    if n_voxels < MIN_VOXEL_WARN:
        return "qc_warn"
    return "ok"


def fs60_mask_available(cfg, subject: str, hemi: str, args: argparse.Namespace) -> Path | None:
    if cfg.fs60_root is None:
        return None
    hemi_lower = "lh" if hemi == "L" else "rh"
    pattern = args.fs60_subfield_pattern.replace("lh.", f"{hemi_lower}.").format(subject=subject, hemi=hemi_lower)
    candidate = Path(cfg.fs60_root) / pattern
    return candidate if candidate.exists() else None


def split_fs60_subfield(path: Path) -> list[dict]:
    """基于 FS60 掩膜执行 head/body/tail 切分。

    当前实现使用 FS60 标签体积的非零体素作为海马整体掩膜，
    再沿 MNI y 轴三等分为 tail/body/head。这样可确保 FS60 可用时
    优先走 FS60 体素空间，而不是无条件回退到 meta 掩膜。
    """
    return split_mask_along_y(path, n_segments=3)


def segment_for_subject(cfg, subject: str, args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    masks_dir = module_dir(cfg, MODULE) / "masks" / subject
    log_lines: list[str] = []
    for hemi in ("L", "R"):
        provenance = None
        segments: list[dict] = []
        fs60_path = fs60_mask_available(cfg, subject, hemi, args)
        if fs60_path is not None:
            try:
                segments = split_fs60_subfield(fs60_path)
                provenance = "fs60"
            except Exception as exc:
                log_lines.append(f"{subject} hemi={hemi} fs60_failed={fs60_path} err={exc}")
                segments = []
        if not segments:
            mask_file = Path(cfg.mask_root) / args.mask_pattern.format(hemi=hemi)
            if not mask_file.exists():
                log_lines.append(f"{subject} hemi={hemi} mask_missing={mask_file}")
                continue
            segments = split_mask_along_y(mask_file, n_segments=3)
            provenance = "mni_split"
        for seg in segments:
            label = assemble_subfield_label(hemi, seg["segment"])
            out_path = masks_dir / f"{label}.nii.gz"
            if out_path.exists():
                raise FileExistsError(f"Output exists; refusing to overwrite: {out_path}")
            save_segment_mask(seg, out_path)
            rows.append({
                "subject": subject,
                "hemisphere": hemi,
                "subROI": label,
                "n_voxels": seg["n_voxels"],
                "provenance": provenance,
                "qc_status": qc_status_for(seg["n_voxels"]),
                "y_min": seg.get("y_min"),
                "y_max": seg.get("y_max"),
                "mask_path": str(out_path),
            })
            log_lines.append(
                f"{subject} hemi={hemi} segment={seg['segment']} n_voxels={seg['n_voxels']} "
                f"provenance={provenance} qc={qc_status_for(seg['n_voxels'])}"
            )
    return rows, log_lines


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    subjects = (
        args.subjects
        or discover_subjects(cfg.fs60_root or cfg.mask_root)
        or discover_subjects(cfg.base_dir / "pattern_root")
        or discover_subjects(cfg.base_dir / "lss_betas_final")
        or discover_subjects(cfg.base_dir / "derivatives")
    )
    if not subjects:
        if not args.allow_empty:
            raise RuntimeError("未能发现任何被试；请检查 --fs60-root / --mask-root / --base-dir / --subjects。")
        subjects = []
    all_rows: list[dict] = []
    all_log: list[str] = []
    for sub in subjects:
        rows, log_lines = segment_for_subject(cfg, sub, args)
        all_rows.extend(rows)
        all_log.extend(log_lines)
    manifest = pd.DataFrame(all_rows, columns=[
        "subject", "hemisphere", "subROI", "n_voxels", "provenance", "qc_status",
        "y_min", "y_max", "mask_path",
    ])
    expected_labels = iter_subfield_labels()
    missing_summary = []
    if not manifest.empty:
        for sub in sorted(manifest["subject"].unique()):
            actual = set(manifest.loc[manifest["subject"] == sub, "subROI"])
            missing = [lbl for lbl in expected_labels if lbl not in actual]
            if missing:
                missing_summary.append(f"{sub}: missing={','.join(missing)}")
    write_outputs(cfg, MODULE, {
        "subfield_manifest.tsv": manifest,
        "s1_log.txt": log_text("S1 subfield segmentation log", all_log + missing_summary),
    })


if __name__ == "__main__":
    main()
