#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于已有 LSS beta（不重算 LSS），按刺激类型生成“被试平均 beta 图”。

输出：
1) averaged_betas/<scenario>/<stimulus_type>/<subject>/mean_beta.*
2) averaged_betas/avg_beta_index_<scenario>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, surface

DEFAULT_LSS_ROOT = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results")
DEFAULT_OUT_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/averaged_betas_by_stimulus")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按刺激类型计算被试平均 beta")
    p.add_argument("--lss-root", type=Path, default=DEFAULT_LSS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stim-col", type=str, default="raw_emotion", help="刺激类型列名（默认 raw_emotion）")
    p.add_argument("--task", type=str, default="EMO", help="仅使用该 task（默认 EMO）；传 ALL 表示不过滤")
    return p.parse_args()


def sanitize_token(x: object) -> str:
    s = str(x).strip().replace(" ", "_")
    return s.replace("/", "-")


def infer_file_path(row: pd.Series, lss_root: Path) -> Path:
    raw_file = str(row["file"])
    if raw_file.startswith("/"):
        return Path(raw_file)

    task_folder = str(row.get("task", "")).lower()
    if not task_folder:
        task_folder = "emo" if "EMO" in str(row.get("stimulus_content", "")) else "soc"

    p1 = lss_root / task_folder / str(row["subject"]) / raw_file
    if p1.exists():
        return p1

    run = row.get("run", None)
    if pd.notna(run):
        p2 = lss_root / task_folder / str(row["subject"]) / f"run-{run}" / raw_file
        if p2.exists():
            return p2

    return p1


def load_beta(path: Path, is_volume: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if is_volume:
        img = image.load_img(str(path))
        return img.get_fdata(dtype=np.float32), np.asarray(img.affine)
    return surface.load_surf_data(str(path)).astype(np.float32), None


def save_mean_beta(arr: np.ndarray, out_path: Path, is_volume: bool, affine: Optional[np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_volume:
        aff = affine if affine is not None else np.eye(4)
        nib.Nifti1Image(arr.astype(np.float32), affine=aff).to_filename(str(out_path))
        return
    da = nib.gifti.GiftiDataArray(arr.reshape(-1).astype(np.float32), intent="NIFTI_INTENT_ESTIMATE")
    nib.save(nib.gifti.GiftiImage(darrays=[da]), str(out_path))


def process_one_index(index_file: Path, lss_root: Path, out_dir: Path, stim_col: str, task_filter: str) -> None:
    scenario = index_file.stem.replace("lss_index_", "").replace("_aligned", "")
    is_volume = "volume" in scenario.lower()
    ext = ".nii.gz" if is_volume else ".gii"

    df = pd.read_csv(index_file)
    required = {"subject", "file", "task", stim_col}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"[跳过] {index_file.name} 缺少列: {sorted(missing)}")
        return

    audit_file = lss_root / "lss_audit_surface_L.csv"
    if not audit_file.exists():
        print(f"[跳过] {index_file.name} 未找到条件文件: {audit_file}")
        return

    audit_df = pd.read_csv(audit_file)
    condition_col = "trial_type"
    if condition_col not in audit_df.columns:
        print(f"[跳过] {index_file.name} 的条件文件缺少列: {condition_col}")
        return

    condition_series = audit_df[condition_col].astype("string").str.strip()
    valid_conditions = sorted({c for c in condition_series.dropna().tolist() if c})
    if not valid_conditions:
        print(f"[跳过] {index_file.name} 的条件文件 {condition_col} 无有效条件")
        return

    preview = ", ".join(valid_conditions[:10])
    if len(valid_conditions) > 10:
        preview += ", ..."
    print(f"[条件] {index_file.name} 从 {audit_file.name}:{condition_col} 提取 {len(valid_conditions)} 个条件: {preview}")

    df = df.dropna(subset=[stim_col]).copy()
    stim_values = df[stim_col].astype("string").str.strip()
    df = df[stim_values.isin(valid_conditions)].copy()
    if task_filter.upper() != "ALL":
        df = df[df["task"].astype(str).str.upper() == task_filter.upper()].copy()

    if df.empty:
        print(f"[跳过] {index_file.name} 过滤后为空")
        return

    records: List[Dict[str, object]] = []
    grouped = df.groupby(["subject", stim_col], dropna=True)
    for (sub, stim_value), g in grouped:
        stim_key = sanitize_token(stim_value)
        arrs = []
        affine = None
        for _, row in g.iterrows():
            p = infer_file_path(row, lss_root)
            if not p.exists():
                continue
            try:
                arr, aff = load_beta(p, is_volume=is_volume)
                arrs.append(arr)
                if affine is None and aff is not None:
                    affine = aff
            except Exception:
                continue

        if not arrs:
            continue

        mean_beta = np.mean(np.stack(arrs, axis=0), axis=0)
        out_file = out_dir / scenario / stim_key / str(sub) / f"mean_beta{ext}"
        save_mean_beta(mean_beta, out_file, is_volume=is_volume, affine=affine)

        records.append(
            {
                "scenario": scenario,
                "space": "volume" if is_volume else scenario,
                "subject": str(sub),
                "stimulus_type": stim_key,
                "task_filter": task_filter,
                "n_trials": int(len(arrs)),
                "file": str(out_file),
            }
        )

    if records:
        out_csv = out_dir / f"avg_beta_index_{scenario}.csv"
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"[完成] {scenario}: {len(records)} 条记录 -> {out_csv}")
    else:
        print(f"[提示] {scenario}: 无可保存记录")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(args.lss_root.glob("lss_index_*_aligned.csv"))
    if not aligned_files:
        raise FileNotFoundError(f"未找到索引文件: {args.lss_root}/lss_index_*_aligned.csv")

    for f in aligned_files:
        process_one_index(f, args.lss_root, args.out_dir, args.stim_col, args.task)


if __name__ == "__main__":
    main()
