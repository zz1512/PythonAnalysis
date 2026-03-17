#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按刺激类型执行：被试相似性矩阵 vs 发育模型 的置换检验 + FWER 校正。

默认读取 calc_roi_similarity_by_stimulus_type.py 输出的矩阵文件：
- similarity_surface_L_<stim>_AgeSorted.csv
- similarity_surface_R_<stim>_AgeSorted.csv
- similarity_surface_[L/R]_<ROI>_<stim>_AgeSorted.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/similarity_matrices_by_stimulus")
DEFAULT_SUBJECT_INFO_PATH = Path("/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")
DEFAULT_ROIS = ("Visual", "Salience_Insula", "Control_PFC")


@dataclass(frozen=True)
class SimilarityInput:
    name: str
    path: Path


def parse_age(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return np.nan
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y = re.search(r"(\d+)\s*岁", s)
    m = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d = re.search(r"(\d+)\s*天", s)
    yy = int(y.group(1)) if y else 0
    mm = int(m.group(1)) if m else 0
    dd = int(d.group(1)) if d else 0
    return yy + mm / 12.0 + dd / 365.0


def load_subject_ages_map(path: Path) -> Dict[str, float]:
    if str(path).endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t")
    elif str(path).endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if COL_SUB_ID in df.columns and COL_AGE in df.columns:
        sid_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in df.columns and LEGACY_COL_AGE in df.columns:
        sid_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError("年龄表缺少必要列")

    out = {}
    for _, r in df.iterrows():
        sid = str(r[sid_col]).strip()
        sid = sid if sid.startswith("sub-") else f"sub-{sid}"
        out[sid] = parse_age(r[age_col])
    return out


def load_similarity_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 10:
        return np.full_like(x, np.nan, dtype=np.float32)
    mu = float(x[mask].mean())
    sd = float(x[mask].std(ddof=0))
    if sd <= 0:
        return np.full_like(x, np.nan, dtype=np.float32)
    out = (x - mu) / sd
    out[~mask] = np.nan
    return out.astype(np.float32, copy=False)


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def build_models(ages: np.ndarray, iu: np.ndarray, ju: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    amax = float(ages.max())
    ai = ages[iu]
    aj = ages[ju]
    m_nn = zscore_1d((amax - np.abs(ai - aj)).astype(np.float32))
    m_conv = zscore_1d(np.minimum(ai, aj).astype(np.float32))
    m_div = zscore_1d((amax - 0.5 * (ai + aj)).astype(np.float32))
    return m_nn, m_conv, m_div


def collect_stimulus_types(matrix_dir: Path) -> List[str]:
    stims = set()
    for p in matrix_dir.glob("similarity_*_AgeSorted.csv"):
        name = p.stem.replace("similarity_", "").replace("_AgeSorted", "")
        parts = name.split("_")
        if len(parts) >= 3 and parts[0] == "surface" and parts[1] in {"L", "R"}:
            stims.add(parts[-1])
    return sorted(stims)


def expected_inputs_for_stim(matrix_dir: Path, stim: str, rois: Sequence[str]) -> List[SimilarityInput]:
    items: List[SimilarityInput] = [
        SimilarityInput("surface_L", matrix_dir / f"similarity_surface_L_{stim}_AgeSorted.csv"),
        SimilarityInput("surface_R", matrix_dir / f"similarity_surface_R_{stim}_AgeSorted.csv"),
    ]
    for roi in rois:
        items.append(SimilarityInput(f"surface_L_{roi}", matrix_dir / f"similarity_surface_L_{roi}_{stim}_AgeSorted.csv"))
        items.append(SimilarityInput(f"surface_R_{roi}", matrix_dir / f"similarity_surface_R_{roi}_{stim}_AgeSorted.csv"))

    return [x for x in items if x.path.exists()]


def run_one_stim(
    stim: str,
    inputs: Sequence[SimilarityInput],
    age_map: Dict[str, float],
    out_dir: Path,
    n_perm: int,
    seed: int,
) -> None:
    if len(inputs) < 2:
        print(f"[跳过] {stim}: 可用矩阵不足")
        return

    dfs = [load_similarity_df(x.path) for x in inputs]
    common_subs = set(dfs[0].index.tolist())
    for df in dfs[1:]:
        common_subs &= set(df.index.tolist())

    subjects = [s for s in sorted(common_subs) if np.isfinite(age_map.get(s, np.nan))]
    subjects.sort(key=lambda s: (age_map[s], s))
    if len(subjects) < 10:
        print(f"[跳过] {stim}: 共同被试 < 10")
        return

    ages = np.array([age_map[s] for s in subjects], dtype=np.float32)
    iu, ju = triu_indices(len(subjects))
    n_pairs = int(iu.size)

    S = []
    names = []
    for item in inputs:
        mat = load_similarity_df(item.path).loc[subjects, subjects].to_numpy(dtype=np.float32)
        vec = zscore_1d(mat[iu, ju])
        if np.isfinite(vec).any():
            S.append(vec)
            names.append(item.name)

    if len(S) < 2:
        print(f"[跳过] {stim}: 有效矩阵不足")
        return

    S = np.stack(S, axis=0).astype(np.float32)
    models = build_models(ages, iu, ju)
    model_names = ["M_nn", "M_conv", "M_div"]

    obs = []
    tests = []
    for mi, m in enumerate(models):
        r = (S @ m) / float(n_pairs)
        for si, name in enumerate(names):
            tests.append((name, model_names[mi]))
            obs.append(float(r[si]))

    obs = np.asarray(obs, dtype=np.float32)
    abs_obs = np.abs(obs)
    count_raw = np.zeros_like(abs_obs, dtype=np.int64)
    count_fwer = np.zeros_like(abs_obs, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for _ in range(int(n_perm)):
        ages_p = ages[rng.permutation(len(ages))]
        models_p = build_models(ages_p, iu, ju)
        all_r = []
        for m in models_p:
            all_r.append((S @ m) / float(n_pairs))
        all_r = np.concatenate(all_r).astype(np.float32)

        abs_perm = np.abs(all_r)
        count_raw += (abs_perm >= abs_obs)
        count_fwer += (abs_perm.max(initial=0.0) >= abs_obs)

    p_perm = (count_raw + 1.0) / (float(n_perm) + 1.0)
    p_fwer = (count_fwer + 1.0) / (float(n_perm) + 1.0)

    out_rows = []
    for i, (mat_name, model_name) in enumerate(tests):
        out_rows.append(
            {
                "stimulus_type": stim,
                "matrix": mat_name,
                "model": model_name,
                "r_obs": float(obs[i]),
                "p_perm": float(p_perm[i]),
                "p_fwer": float(p_fwer[i]),
                "n_subjects": int(len(subjects)),
                "n_pairs": int(n_pairs),
                "n_perm": int(n_perm),
                "seed": int(seed),
            }
        )

    stim_out_dir = out_dir / stim
    stim_out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).sort_values(["matrix", "model"]).to_csv(
        stim_out_dir / "joint_analysis_dev_models_perm_fwer.csv", index=False
    )
    pd.DataFrame({"subject": subjects, "age": ages}).to_csv(stim_out_dir / "subjects_common_age_sorted.csv", index=False)
    print(f"[完成] {stim}: {stim_out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按刺激类型进行模型相似度 + FWER 置换分析")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--subject-info", type=Path, default=DEFAULT_SUBJECT_INFO_PATH)
    p.add_argument("--out-dir", type=Path, default=Path("./joint_perm_fwer_by_stimulus"))
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rois", type=str, default=",".join(DEFAULT_ROIS))
    p.add_argument("--stimuli", type=str, default="", help="手动指定刺激类型，逗号分隔；留空则自动扫描")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    age_map = load_subject_ages_map(args.subject_info)
    rois = [x.strip() for x in args.rois.split(",") if x.strip()]

    if args.stimuli.strip():
        stim_list = [x.strip() for x in args.stimuli.split(",") if x.strip()]
    else:
        stim_list = collect_stimulus_types(args.matrix_dir)

    if not stim_list:
        raise ValueError("未发现任何刺激类型，请检查 matrix-dir 中文件命名")

    for stim in stim_list:
        inputs = expected_inputs_for_stim(args.matrix_dir, stim, rois)
        run_one_stim(stim, inputs, age_map, args.out_dir, args.n_perm, args.seed)


if __name__ == "__main__":
    main()
