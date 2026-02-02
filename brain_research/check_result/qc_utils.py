#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import nibabel as nib


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


def resolve_beta_path_from_index_row(lss_root: Path, row: pd.Series) -> Path:
    rel = str(row.get("file", ""))
    if rel.startswith("/"):
        return Path(rel)
    task = str(row.get("task", "")).lower()
    sub = str(row.get("subject", ""))
    run = row.get("run", None)
    if run is None or (isinstance(run, float) and np.isnan(run)):
        return lss_root / task / sub / rel
    run_str = f"run-{int(run)}"
    p1 = lss_root / task / sub / rel
    if p1.exists():
        return p1
    return lss_root / task / sub / run_str / rel


def resolve_beta_path(lss_root: Path, task: str, subject: str, run: object, rel_file: str) -> Optional[Path]:
    root = Path(lss_root)
    task_s = str(task)
    sub_s = str(subject)
    rel = str(rel_file)
    run_str = None
    try:
        if run is not None and not (isinstance(run, float) and np.isnan(run)):
            run_str = f"run-{int(run)}"
    except Exception:
        run_str = None

    candidates = [
        root / task_s.lower() / sub_s / rel,
        root / task_s.upper() / sub_s / rel,
    ]
    if run_str is not None:
        candidates.extend(
            [
                root / task_s.lower() / sub_s / run_str / rel,
                root / task_s.upper() / sub_s / run_str / rel,
            ]
        )
    for p in candidates:
        if p.exists():
            return p
    return None


def load_gifti_flatten(path: Path) -> Optional[np.ndarray]:
    try:
        g = nib.load(str(path))
        return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32, copy=False)
    except Exception:
        return None


def safe_corr(a: np.ndarray, b: np.ndarray, min_valid: int = 10) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < int(min_valid):
        return float("nan")
    aa = a[m].astype(np.float64, copy=False)
    bb = b[m].astype(np.float64, copy=False)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)
