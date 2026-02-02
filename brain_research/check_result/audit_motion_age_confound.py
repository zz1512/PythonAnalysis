#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_motion_age_confound.py

论文级审计：Motion 是否与 Age 混杂（发育队列常见审稿问题）。

输出：
- audit_motion_age_confound_by_subject.csv：每被试 motion 指标 + 年龄
- audit_motion_age_confound_summary.csv：各指标与年龄的 Pearson/Spearman 相关及置换 p 值
- audit_motion_age_high_motion_groups.csv：按高 motion 阈值分组的年龄摘要

默认路径与 emo 分析输出保持一致：
- BIDS_DATA_DIR：BIDS 数据根目录（默认 /public/home/dingrui/BIDS_DATA）
- LSS_OUTPUT_ROOT：LSS 输出根目录（默认 /public/home/dingrui/fmri_analysis/zz_analysis/lss_results）
- QC_AUDIT_OUT_DIR：QC 输出目录（默认 <LSS_OUTPUT_ROOT 的上一级>/qc_audit_out）
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

COL_SUB_ID = "sub_id"
COL_AGE = "age"
LEGACY_COL_SUB_ID = "被试编号"
LEGACY_COL_AGE = "采集年龄"


def _default_lss_output_root() -> Path:
    return Path(os.environ.get("LSS_OUTPUT_ROOT", "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"))


def _default_qc_out_dir() -> Path:
    p = os.environ.get("QC_AUDIT_OUT_DIR", "")
    if p:
        return Path(p)
    return _default_lss_output_root().parent / "qc_audit_out"


def parse_chinese_age_exact(age_str: object) -> float:
    if pd.isna(age_str) or str(age_str).strip() == "":
        return float("nan")
    if isinstance(age_str, (int, float, np.integer, np.floating)):
        return float(age_str)
    s = str(age_str).strip()
    try:
        return float(s)
    except Exception:
        pass
    y_match = re.search(r"(\d+)\s*岁", s)
    m_match = re.search(r"(\d+)\s*个月", s) or re.search(r"(\d+)\s*月", s)
    d_match = re.search(r"(\d+)\s*天", s)
    years = int(y_match.group(1)) if y_match else 0
    months = int(m_match.group(1)) if m_match else 0
    days = int(d_match.group(1)) if d_match else 0
    return years + (months / 12.0) + (days / 365.0)


def load_subject_ages_map(subject_info_path: Path) -> Dict[str, float]:
    p = str(subject_info_path)
    if p.endswith(".tsv") or p.endswith(".txt"):
        info_df = pd.read_csv(p, sep="\t")
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        info_df = pd.read_excel(p)
    elif p.endswith(".csv"):
        info_df = pd.read_csv(p)
    else:
        raise ValueError("不支持的年龄表格式，请使用 .tsv/.csv/.xlsx")

    if COL_SUB_ID in info_df.columns and COL_AGE in info_df.columns:
        sub_col, age_col = COL_SUB_ID, COL_AGE
    elif LEGACY_COL_SUB_ID in info_df.columns and LEGACY_COL_AGE in info_df.columns:
        sub_col, age_col = LEGACY_COL_SUB_ID, LEGACY_COL_AGE
    else:
        raise ValueError(f"年龄表缺少列: {COL_SUB_ID}/{COL_AGE} 或 {LEGACY_COL_SUB_ID}/{LEGACY_COL_AGE}")

    age_map: Dict[str, float] = {}
    for _, row in info_df.iterrows():
        raw_id = str(row[sub_col]).strip()
        sub_id = raw_id if raw_id.startswith("sub-") else f"sub-{raw_id}"
        age_map[sub_id] = parse_chinese_age_exact(row[age_col])
    return age_map


def _find_subjects(root: Path) -> List[str]:
    return sorted([p.name for p in root.glob("sub-*") if p.is_dir()])


def _first_or_none(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def _count_regex_columns(cols: List[str], pattern: str) -> int:
    rx = re.compile(pattern)
    return int(sum(1 for c in cols if rx.search(c)))


def _read_confounds(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _audit_one_subject(task: str, sub_dir: Path, conf_glob: str) -> Dict[str, object]:
    conf_path = _first_or_none(list(sub_dir.glob(conf_glob)))
    if conf_path is None:
        return {
            "task": task,
            "subject": sub_dir.name,
            "confounds_found": False,
            "confounds_read_ok": False,
            "n_tp": np.nan,
            "fd_mean": np.nan,
            "fd_p95": np.nan,
            "dvars_mean": np.nan,
            "n_motion_outlier_cols": np.nan,
            "outlier_frac": np.nan,
        }

    df = _read_confounds(conf_path)
    if df is None:
        return {
            "task": task,
            "subject": sub_dir.name,
            "confounds_found": True,
            "confounds_read_ok": False,
            "n_tp": np.nan,
            "fd_mean": np.nan,
            "fd_p95": np.nan,
            "dvars_mean": np.nan,
            "n_motion_outlier_cols": np.nan,
            "outlier_frac": np.nan,
        }

    cols = [str(c) for c in df.columns]
    n_tp = int(df.shape[0])

    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce") if "framewise_displacement" in df.columns else None
    dvars = pd.to_numeric(df["dvars"], errors="coerce") if "dvars" in df.columns else None
    fd_mean = float(np.nanmean(fd)) if fd is not None else float("nan")
    fd_p95 = float(np.nanpercentile(fd, 95)) if fd is not None and np.isfinite(fd).any() else float("nan")
    dvars_mean = float(np.nanmean(dvars)) if dvars is not None else float("nan")

    n_out = (
        _count_regex_columns(cols, r"^motion_outlier")
        + _count_regex_columns(cols, r"^non_steady_state_outlier")
        + _count_regex_columns(cols, r"^outlier")
        + _count_regex_columns(cols, r"^scrub")
    )
    outlier_frac = float(n_out / n_tp) if n_tp > 0 else float("nan")

    return {
        "task": task,
        "subject": sub_dir.name,
        "confounds_found": True,
        "confounds_read_ok": True,
        "n_tp": n_tp,
        "fd_mean": fd_mean,
        "fd_p95": fd_p95,
        "dvars_mean": dvars_mean,
        "n_motion_outlier_cols": int(n_out),
        "outlier_frac": outlier_frac,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return float("nan")
    xx = x[m] - float(np.mean(x[m]))
    yy = y[m] - float(np.mean(y[m]))
    denom = float(np.linalg.norm(xx) * np.linalg.norm(yy))
    if denom <= 0:
        return float("nan")
    return float(np.dot(xx, yy) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(x).rank(method="average", na_option="keep").to_numpy(dtype=float)
    y = pd.Series(y).rank(method="average", na_option="keep").to_numpy(dtype=float)
    return _pearson(x, y)


def _perm_p_greater(x: np.ndarray, y: np.ndarray, stat_obs: float, seed: int, n_perm: int, stat_fn) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 5 or not np.isfinite(stat_obs):
        return float("nan")
    rng = np.random.default_rng(int(seed))
    cnt = 0
    for _ in range(int(n_perm)):
        yp = y[rng.permutation(y.size)]
        s = float(stat_fn(x, yp))
        if np.isfinite(s) and s >= float(stat_obs):
            cnt += 1
    return float((cnt + 1.0) / (float(n_perm) + 1.0))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="审计 motion 是否与 age 混杂（FD/outlier_frac/dvars vs age）")
    p.add_argument("--bids-dir", type=Path, default=Path(os.environ.get("BIDS_DATA_DIR", "/public/home/dingrui/BIDS_DATA")), help="BIDS_DATA 根目录")
    p.add_argument("--subject-info", type=Path, default=Path(os.environ.get("SUBJECT_AGE_TABLE", "/public/home/dingrui/fmri_analysis/data/beh/beh_indices_mri_exp_ER_TG.csv")), help="年龄表路径")
    p.add_argument("--out-dir", type=Path, default=_default_qc_out_dir(), help="输出目录")
    p.add_argument("--n-perm", type=int, default=5000, help="置换次数（用于相关性的经验 p 值）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--limit-subjects", type=int, default=0, help="仅审计前 N 个被试（0=全部）")
    p.add_argument("--max-outlier-frac", type=float, default=0.2, help="outlier_frac 超过阈值标记为高 motion")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    age_map = load_subject_ages_map(Path(args.subject_info))

    specs = [
        ("EMO", Path(args.bids_dir) / "emo_20250623/v1", "func/emo_miniprep/*desc-confounds_timeseries.tsv"),
        ("SOC", Path(args.bids_dir) / "soc_20250623/v1", "func/soc_miniprep/*desc-confounds_timeseries.tsv"),
    ]

    rows: List[Dict[str, object]] = []
    for task, root, glob_pat in specs:
        if not root.exists():
            continue
        subs = _find_subjects(root)
        if args.limit_subjects and int(args.limit_subjects) > 0:
            subs = subs[: int(args.limit_subjects)]
        for sub in subs:
            rows.append(_audit_one_subject(task, root / sub, glob_pat))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("未产生任何 motion 审计记录，请检查 bids-dir 与目录结构")

    df["age"] = df["subject"].astype(str).map(lambda s: age_map.get(s, np.nan))
    out_by_sub = out_dir / "audit_motion_age_confound_by_subject.csv"
    df.to_csv(out_by_sub, index=False)

    df_ok = df[(df["confounds_found"] == True) & (df["confounds_read_ok"] == True)].copy()
    df_ok["age"] = pd.to_numeric(df_ok["age"], errors="coerce")
    metrics = ["outlier_frac", "fd_mean", "fd_p95", "dvars_mean"]

    summary_rows: List[Dict[str, object]] = []
    for task in sorted(df_ok["task"].astype(str).unique().tolist()):
        sub = df_ok[df_ok["task"] == task].copy()
        age = sub["age"].to_numpy(dtype=float)
        for m in metrics:
            y = pd.to_numeric(sub.get(m, np.nan), errors="coerce").to_numpy(dtype=float)
            r_p = _pearson(age, y)
            r_s = _spearman(age, y)
            p_p = _perm_p_greater(age, y, r_p, seed=int(args.seed), n_perm=int(args.n_perm), stat_fn=_pearson)
            p_s = _perm_p_greater(age, y, r_s, seed=int(args.seed) + 7, n_perm=int(args.n_perm), stat_fn=_spearman)
            summary_rows.append(
                {
                    "task": task,
                    "metric": m,
                    "n": int(np.isfinite(age).sum() if np.isfinite(y).any() else 0),
                    "pearson_r": float(r_p),
                    "pearson_p_perm_greater": float(p_p),
                    "spearman_r": float(r_s),
                    "spearman_p_perm_greater": float(p_s),
                }
            )

    summary = pd.DataFrame(summary_rows)
    out_summary = out_dir / "audit_motion_age_confound_summary.csv"
    summary.to_csv(out_summary, index=False)

    df_ok["high_motion"] = pd.to_numeric(df_ok["outlier_frac"], errors="coerce") > float(args.max_outlier_frac)
    grp = (
        df_ok.groupby(["task", "high_motion"], as_index=False)
        .agg(
            n=("subject", "nunique"),
            age_mean=("age", "mean"),
            age_std=("age", "std"),
            outlier_frac_mean=("outlier_frac", "mean"),
            fd_p95_mean=("fd_p95", "mean"),
        )
    )
    out_grp = out_dir / "audit_motion_age_high_motion_groups.csv"
    grp.to_csv(out_grp, index=False)

    print(f"Saved: {out_by_sub}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_grp}")


if __name__ == "__main__":
    main()
