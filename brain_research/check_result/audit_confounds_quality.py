#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_confounds_quality.py

论文级审计（confounds 质量与 scrubbing 强度）：
1) confounds 文件是否存在、是否可读、行数是否为正
2) 关键列/列组是否存在（FD/DVARS/Friston/aCompCor/运动参数等）
3) scrubbing regressors 强度（motion_outlier/non_steady_state/outlier 等列数量与占比）
4) 生成按 task×subject 的汇总表，并在日志给出 PASS/FAIL + Top-N 失败样例

输出：
- audit_confounds_quality.csv

扩展建议（未来）：
- 结合 fMRI n_tp 与模型设计矩阵，估计有效自由度（dof）与是否过拟合
- 对 FD/DVARS 做更细粒度的阈值审计与分层（儿童队列常见年龄相关 motion 混杂）
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfoundSpec:
    name: str
    root: Path
    conf_glob: str


def _find_subjects(root: Path) -> List[str]:
    return sorted([p.name for p in root.glob("sub-*") if p.is_dir()])


def _first_or_none(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths[-1]


def _read_confounds(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _count_regex_columns(cols: List[str], pattern: str) -> int:
    rx = re.compile(pattern)
    return int(sum(1 for c in cols if rx.search(c)))


def _has_any(cols: List[str], candidates: List[str]) -> bool:
    s = set(cols)
    return any(c in s for c in candidates)


def audit_one_subject(task: str, sub_dir: Path, conf_glob: str) -> Dict[str, object]:
    conf_path = _first_or_none(list(sub_dir.glob(conf_glob)))
    if conf_path is None:
        return {
            "task": task,
            "subject": sub_dir.name,
            "confounds_path": "",
            "confounds_found": False,
            "confounds_read_ok": False,
            "n_tp": np.nan,
            "n_cols": np.nan,
            "fd_mean": np.nan,
            "fd_p95": np.nan,
            "dvars_mean": np.nan,
            "n_motion_outlier_cols": np.nan,
            "outlier_frac": np.nan,
            "has_basic_motion": False,
            "has_fd": False,
            "has_dvars": False,
            "has_compcor": False,
        }

    df = _read_confounds(conf_path)
    if df is None:
        return {
            "task": task,
            "subject": sub_dir.name,
            "confounds_path": str(conf_path),
            "confounds_found": True,
            "confounds_read_ok": False,
            "n_tp": np.nan,
            "n_cols": np.nan,
            "fd_mean": np.nan,
            "fd_p95": np.nan,
            "dvars_mean": np.nan,
            "n_motion_outlier_cols": np.nan,
            "outlier_frac": np.nan,
            "has_basic_motion": False,
            "has_fd": False,
            "has_dvars": False,
            "has_compcor": False,
        }

    cols = [str(c) for c in df.columns]
    n_tp = int(df.shape[0])
    n_cols = int(df.shape[1])

    has_basic_motion = _has_any(cols, ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"])
    has_fd = _has_any(cols, ["framewise_displacement"])
    has_dvars = _has_any(cols, ["dvars"])
    has_compcor = any(c.startswith("a_comp_cor_") or c.startswith("t_comp_cor_") for c in cols)

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
        "confounds_path": str(conf_path),
        "confounds_found": True,
        "confounds_read_ok": True,
        "n_tp": n_tp,
        "n_cols": n_cols,
        "fd_mean": fd_mean,
        "fd_p95": fd_p95,
        "dvars_mean": dvars_mean,
        "n_motion_outlier_cols": int(n_out),
        "outlier_frac": outlier_frac,
        "has_basic_motion": bool(has_basic_motion),
        "has_fd": bool(has_fd),
        "has_dvars": bool(has_dvars),
        "has_compcor": bool(has_compcor),
    }


def build_default_specs(base_dir: Path) -> List[ConfoundSpec]:
    emo_root = base_dir / "emo_20250623/v1"
    soc_root = base_dir / "soc_20250623/v1"
    return [
        ConfoundSpec(name="EMO", root=emo_root, conf_glob="func/emo_miniprep/*desc-confounds_timeseries.tsv"),
        ConfoundSpec(name="SOC", root=soc_root, conf_glob="func/soc_miniprep/*desc-confounds_timeseries.tsv"),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="审计 confounds 质量与 scrubbing 强度")
    p.add_argument("--bids-dir", type=Path, default=Path("/public/home/dingrui/BIDS_DATA"), help="BIDS_DATA 根目录")
    p.add_argument("--out-dir", type=Path, default=Path("./qc_audit_out"), help="输出目录")
    p.add_argument("--max-outlier-frac", type=float, default=0.2, help="outlier_frac 超过该阈值判为未通过")
    p.add_argument("--require-basic-motion", action="store_true", help="要求存在基本运动参数列（trans/rot）")
    p.add_argument("--log-top", type=int, default=10, help="FAIL 时打印示例条目数量")
    p.add_argument("--limit-subjects", type=int, default=0, help="仅审计前 N 个被试（0=全部）")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_default_specs(args.bids_dir)
    rows: List[Dict[str, object]] = []

    for spec in specs:
        if not spec.root.exists():
            continue
        subs = _find_subjects(spec.root)
        if args.limit_subjects and args.limit_subjects > 0:
            subs = subs[: int(args.limit_subjects)]
        for sub in subs:
            rows.append(audit_one_subject(spec.name, spec.root / sub, spec.conf_glob))

    df = pd.DataFrame(rows)
    out_path = out_dir / "audit_confounds_quality.csv"
    df.to_csv(out_path, index=False)

    if df.empty:
        print("[QC] audit_confounds_quality: FAIL")
        print("[QC] No rows produced.")
        return

    fail_mask = (df["confounds_found"] == False) | (df["confounds_read_ok"] == False)
    fail_mask = fail_mask | (df["outlier_frac"].fillna(1.0) > float(args.max_outlier_frac))
    if bool(args.require_basic_motion):
        fail_mask = fail_mask | (df["has_basic_motion"] == False)

    n_fail = int(df[fail_mask].shape[0])
    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"[QC] audit_confounds_quality: {status}")
    print(f"[QC] Saved: {out_path}")
    print(f"[QC] Rows={len(df)} | Fail={n_fail} | MaxOutlierFrac={float(args.max_outlier_frac):.3f}")

    if status == "FAIL":
        top_n = int(args.log_top)
        if top_n > 0:
            shown = df[fail_mask].sort_values(["outlier_frac", "n_motion_outlier_cols"], ascending=False).head(top_n)
            cols = [
                "task",
                "subject",
                "confounds_read_ok",
                "n_tp",
                "n_cols",
                "n_motion_outlier_cols",
                "outlier_frac",
                "fd_mean",
                "fd_p95",
            ]
            cols = [c for c in cols if c in shown.columns]
            print(f"[QC] Fail examples (showing up to {top_n}):")
            print(shown[cols].to_string(index=False))


if __name__ == "__main__":
    main()

