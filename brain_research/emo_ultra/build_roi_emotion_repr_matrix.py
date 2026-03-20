#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_roi_emotion_repr_matrix.py

将 Step1 输出的 ROI 表征矩阵（subject x stimulus x stimulus）
按刺激情绪聚合为（subject x emotion x emotion）。

默认仅聚合四类情绪关键词：Anger / Disgust / Fear / Sad。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_ultra")
DEFAULT_EMOTIONS = ("Anger", "Disgust", "Fear", "Sad")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将 ROI 刺激矩阵聚合为刺激情绪矩阵")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus")
    p.add_argument("--repr-prefix", type=str, default="roi_repr_matrix_200")
    p.add_argument("--out-prefix", type=str, default=None)
    p.add_argument("--emotions", nargs="+", default=list(DEFAULT_EMOTIONS), help="按该顺序输出情绪维度")
    return p.parse_args()


def detect_emotion(stimulus_name: str, emotions: List[str]) -> str:
    s = str(stimulus_name)
    s_lower = s.lower()
    for emo in emotions:
        if str(emo).lower() in s_lower:
            return str(emo)
    return ""


def aggregate_roi(arr: np.ndarray, emo_idx: Dict[str, np.ndarray], emotions: List[str]) -> np.ndarray:
    n_sub = int(arr.shape[0])
    n_emo = len(emotions)
    out = np.full((n_sub, n_emo, n_emo), np.nan, dtype=np.float32)

    for ei, e1 in enumerate(emotions):
        i1 = emo_idx[e1]
        for ej, e2 in enumerate(emotions):
            i2 = emo_idx[e2]
            block = arr[:, i1[:, None], i2[None, :]]
            out[:, ei, ej] = np.nanmean(block, axis=(1, 2)).astype(np.float32)
    return out


def run_one(stim_dir: Path, repr_prefix: str, out_prefix: str, emotions: List[str]) -> Dict[str, object] | None:
    npz_path = stim_dir / f"{repr_prefix}.npz"
    rois_path = stim_dir / f"{repr_prefix}_rois.csv"
    subs_path = stim_dir / f"{repr_prefix}_subjects.csv"
    order_path = stim_dir / "stimulus_order.csv"

    for p in (npz_path, rois_path, subs_path, order_path):
        if not p.exists():
            raise FileNotFoundError(f"缺少文件: {p}")

    order_df = pd.read_csv(order_path)
    if "stimulus_order" not in order_df.columns:
        raise ValueError(f"{order_path} 缺少 stimulus_order 列")

    stim_order = order_df["stimulus_order"].astype(str).tolist()
    emo_labels = [detect_emotion(s, emotions) for s in stim_order]
    keep_idx = [i for i, e in enumerate(emo_labels) if e != ""]
    if not keep_idx:
        print(f"[SKIP] {stim_dir.name}: 没有命中情绪关键词的刺激，跳过该 stimulus_type。")
        return None

    stim_order_kept = [stim_order[i] for i in keep_idx]
    emo_labels_kept = [emo_labels[i] for i in keep_idx]
    emo_idx: Dict[str, np.ndarray] = {}
    for emo in emotions:
        idx = np.where(np.array(emo_labels_kept, dtype=object) == emo)[0]
        if int(idx.size) == 0:
            raise ValueError(f"{stim_dir} 中缺少情绪 {emo} 的刺激，无法生成固定 {len(emotions)}x{len(emotions)} 矩阵")
        emo_idx[emo] = idx

    npz = np.load(npz_path)
    rois = pd.read_csv(rois_path)["roi"].astype(str).tolist()
    subjects = pd.read_csv(subs_path)["subject"].astype(str).tolist()

    out_npz: Dict[str, np.ndarray] = {}
    for roi in rois:
        arr = np.asarray(npz[roi], dtype=np.float32)
        arr = arr[:, keep_idx, :][:, :, keep_idx]
        out_npz[roi] = aggregate_roi(arr, emo_idx=emo_idx, emotions=emotions)

    np.savez_compressed(stim_dir / f"{out_prefix}.npz", **out_npz)
    pd.DataFrame({"roi": rois}).to_csv(stim_dir / f"{out_prefix}_rois.csv", index=False)
    pd.DataFrame({"subject": subjects}).to_csv(stim_dir / f"{out_prefix}_subjects.csv", index=False)
    pd.DataFrame({"emotion_order": emotions}).to_csv(stim_dir / "emotion_order.csv", index=False)
    pd.DataFrame({"stimulus_order": stim_order_kept, "emotion": emo_labels_kept}).to_csv(
        stim_dir / "stimulus_to_emotion.csv", index=False
    )
    pd.DataFrame(
        [{"emotion": emo, "n_stimuli": int(emo_idx[emo].size)} for emo in emotions]
    ).to_csv(stim_dir / "emotion_stimulus_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "repr_prefix_input": str(repr_prefix),
                "repr_prefix_output": str(out_prefix),
                "n_subjects": int(len(subjects)),
                "n_rois": int(len(rois)),
                "n_stimuli_kept": int(len(stim_order_kept)),
                "n_emotions": int(len(emotions)),
            }
        ]
    ).to_csv(stim_dir / f"{out_prefix}_meta.csv", index=False)

    return {
        "stimulus_type": stim_dir.name,
        "n_subjects": int(len(subjects)),
        "n_rois": int(len(rois)),
        "n_stimuli_kept": int(len(stim_order_kept)),
        "n_emotions": int(len(emotions)),
    }


def run(matrix_dir: Path, stimulus_dir_name: str, repr_prefix: str, out_prefix: str, emotions: List[str]) -> None:
    by_stim = matrix_dir / str(stimulus_dir_name)
    if not by_stim.exists():
        raise FileNotFoundError(f"未找到目录: {by_stim}")

    rows = []
    for stim_dir in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
        row = run_one(stim_dir, repr_prefix=repr_prefix, out_prefix=out_prefix, emotions=emotions)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stimulus_type")
    out.to_csv(matrix_dir / f"{out_prefix}_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    out_prefix = str(args.out_prefix) if args.out_prefix is not None else f"{str(args.repr_prefix)}_emotion4"
    run(
        matrix_dir=Path(args.matrix_dir),
        stimulus_dir_name=str(args.stimulus_dir_name),
        repr_prefix=str(args.repr_prefix),
        out_prefix=str(out_prefix),
        emotions=[str(x) for x in args.emotions],
    )


if __name__ == "__main__":
    main()
