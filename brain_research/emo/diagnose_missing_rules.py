#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_missing_rules.py

用途
-
对既有 aligned 索引与硬盘文件进行校验，输出：
- 可能需要剔除的刺激（缺失人数多）
- 可能需要剔除的被试（缺失刺激多）

默认策略
-
1) 先按 min_coverage 选公共刺激
2) 统计每个刺激缺失的被试数
3) 统计每个被试缺失的刺激数
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MissingPreset:
    name: str
    aligned_csv: Path
    lss_root: Path
    task: str
    min_coverage: float
    top_n: int
    out_dir: Path


ACTIVE_PRESET = "EMO_surface_L"


PRESETS: Dict[str, MissingPreset] = {
    "EMO_surface_L": MissingPreset(
        name="EMO_surface_L",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="EMO",
        min_coverage=0.9,
        top_n=10,
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/missing_diagnose_out/emo_surface_L"),
    ),
    "EMO_surface_R": MissingPreset(
        name="EMO_surface_R",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_R_aligned.csv"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="EMO",
        min_coverage=0.9,
        top_n=10,
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/missing_diagnose_out/emo_surface_R"),
    ),
    "EMO_volume": MissingPreset(
        name="EMO_volume",
        aligned_csv=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_volume_aligned.csv"),
        lss_root=Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"),
        task="EMO",
        min_coverage=0.9,
        top_n=10,
        out_dir=Path("/public/home/dingrui/fmri_analysis/zz_analysis/missing_diagnose_out/emo_volume"),
    ),
}


def choose_preset(name: str, presets: Dict[str, MissingPreset]) -> MissingPreset:
    if name in presets:
        return presets[name]
    keys = sorted(presets.keys())
    if not keys:
        raise ValueError("未配置任何 preset。")
    raise ValueError(f"未找到 preset: {name}。可用 presets: {', '.join(keys)}")


def resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", ""))
    if f.startswith("/"):
        return Path(f)
    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))
    p = lss_root / task / sub / f
    if p.exists():
        return p
    run = str(row.get("run", ""))
    return lss_root / task / sub / f"run-{run}" / f


def common_stimulus_keys(df: pd.DataFrame, subjects: List[str], min_coverage: float) -> List[str]:
    df = df[df["subject"].isin(set(subjects))].copy()
    counts = df["stimulus_content"].value_counts()
    thr = int(np.floor(float(min_coverage) * len(subjects)))
    thr = max(thr, 1)
    keys = counts[counts >= thr].index.astype(str).tolist()
    keys.sort()
    return keys


def build_lookup(df: pd.DataFrame, lss_root: Path, subjects: List[str], keys: List[str]) -> Dict[Tuple[str, str], Path]:
    sub_set = set(subjects)
    key_set = set(keys)
    df = df[df["subject"].astype(str).isin(sub_set) & df["stimulus_content"].astype(str).isin(key_set)].copy()
    lookup: Dict[Tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        s = str(row.get("subject", ""))
        k = str(row.get("stimulus_content", ""))
        lookup[(s, k)] = resolve_beta_path(lss_root, row)
    return lookup


def main() -> None:
    p = argparse.ArgumentParser(description="检查缺失刺激/被试情况")
    p.add_argument("--preset", type=str, default=ACTIVE_PRESET, help="预置配置名称")
    p.add_argument("--aligned-csv", type=Path, default=None, help="lss_index_*_aligned.csv")
    p.add_argument("--lss-root", type=Path, default=None, help="LSS 根目录")
    p.add_argument("--task", type=str, default=None, help="仅分析某个 task（EMO/SOC），为空表示不过滤")
    p.add_argument("--min-coverage", type=float, default=None, help="公共刺激阈值")
    p.add_argument("--top-n", type=int, default=None, help="输出 top N 缺失统计")
    p.add_argument("--out-dir", type=Path, default=None, help="输出目录")
    args = p.parse_args()

    preset = choose_preset(args.preset, PRESETS)
    aligned_csv = args.aligned_csv or preset.aligned_csv
    lss_root = args.lss_root or preset.lss_root
    task = args.task if args.task is not None else preset.task
    min_coverage = args.min_coverage if args.min_coverage is not None else preset.min_coverage
    top_n = args.top_n if args.top_n is not None else preset.top_n
    out_dir = args.out_dir or preset.out_dir

    df = pd.read_csv(aligned_csv)
    df = df.dropna(subset=["subject", "stimulus_content", "file", "task"]).copy()
    df["subject"] = df["subject"].astype(str)
    df["task"] = df["task"].astype(str)
    df["stimulus_content"] = df["stimulus_content"].astype(str)

    if task:
        df = df[df["task"] == task].copy()
    if df.empty:
        raise ValueError("过滤后 aligned 数据为空。")

    subjects = sorted(df["subject"].unique().astype(str).tolist())
    keys = common_stimulus_keys(df, subjects, min_coverage)
    if not keys:
        raise ValueError("没有满足 coverage 阈值的公共刺激。")

    lookup = build_lookup(df, lss_root, subjects, keys)
    missing_by_key: Dict[str, List[str]] = {k: [] for k in keys}
    missing_by_sub: Dict[str, List[str]] = {s: [] for s in subjects}

    for sub in subjects:
        for key in keys:
            fpath = lookup.get((sub, key))
            if fpath is None or not fpath.exists():
                missing_by_key[key].append(sub)
                missing_by_sub[sub].append(key)

    missing_key_counts = sorted(
        [(k, len(v)) for k, v in missing_by_key.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    missing_sub_counts = sorted(
        [(s, len(v)) for s, v in missing_by_sub.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"总被试数: {len(subjects)}")
    print(f"公共刺激数: {len(keys)} (min_coverage={min_coverage})")
    print("\n--- 缺失最严重的刺激 (Top) ---")
    for k, n in missing_key_counts[:top_n]:
        print(f"  {k}: 缺失 {n} 人")
    print("\n--- 缺失最严重的被试 (Top) ---")
    for s, n in missing_sub_counts[:top_n]:
        print(f"  {s}: 缺失 {n} / {len(keys)} 刺激")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stimulus_content": [k for k, _ in missing_key_counts],
            "n_missing_subjects": [n for _, n in missing_key_counts],
        }
    ).to_csv(out_dir / "missing_stimuli_rank.csv", index=False)
    pd.DataFrame(
        {
            "subject": [s for s, _ in missing_sub_counts],
            "n_missing_stimuli": [n for _, n in missing_sub_counts],
        }
    ).to_csv(out_dir / "missing_subjects_rank.csv", index=False)


if __name__ == "__main__":
    main()
