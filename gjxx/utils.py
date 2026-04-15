from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import json
import math

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class SubjectRun:
    subject: str
    run: int
    path: Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_subject_id(name: str) -> str:
    return name.replace("-", "").strip()


def discover_subject_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("sub")]
    )


def infer_subject_mapping(left_root: str | Path, right_root: str | Path) -> dict[str, tuple[Path, Path]]:
    left = {normalize_subject_id(path.name): path for path in discover_subject_dirs(left_root)}
    right = {normalize_subject_id(path.name): path for path in discover_subject_dirs(right_root)}
    common = sorted(set(left) & set(right))
    return {subject: (left[subject], right[subject]) for subject in common}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="	")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        frame.to_csv(path, sep="	", index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path}")


def list_run_event_files(subject_dir: str | Path) -> list[SubjectRun]:
    subject_dir = Path(subject_dir)
    files = sorted(subject_dir.glob("*run-*_events.tsv"))
    if not files:
        files = sorted(subject_dir.glob("*run-*_events.txt"))
    runs: list[SubjectRun] = []
    for path in files:
        stem = path.stem
        marker = "run-"
        if marker not in stem:
            continue
        run_part = stem.split(marker, maxsplit=1)[1]
        run = int(run_part.split("_", maxsplit=1)[0])
        runs.append(SubjectRun(subject=subject_dir.name, run=run, path=path))
    return sorted(runs, key=lambda item: item.run)


def list_bold_runs(subject_dir: str | Path, pattern: str = "sub-*.nii*") -> list[Path]:
    subject_dir = Path(subject_dir)
    run_dirs = sorted([path for path in subject_dir.iterdir() if path.is_dir() and path.name.startswith("run")])
    bolds: list[Path] = []
    for run_dir in run_dirs:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            bolds.append(matches[0])
    return bolds


def read_motion_confounds(subject_dir: str | Path) -> list[pd.DataFrame]:
    subject_dir = Path(subject_dir)
    motion_dir = subject_dir / "multi_reg"
    if not motion_dir.exists():
        return []
    frames: list[pd.DataFrame] = []
    for path in sorted(motion_dir.glob("sub*.tsv")):
        frame = pd.read_csv(path, sep="	")
        columns = [
            column
            for column in ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
            if column in frame.columns
        ]
        frames.append(frame.loc[:, columns].copy())
    return frames


def fisher_z_from_samples(samples: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = np.arctanh(corr)
    fisher[~np.isfinite(fisher)] = 0.0
    return fisher


def mean_row_similarity(samples: np.ndarray) -> np.ndarray:
    fisher = fisher_z_from_samples(samples)
    return fisher.mean(axis=1)


def rank_transform(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return stats.rankdata(values, axis=axis)


def paired_t_summary(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    stat, pvalue = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")
    return {
        "n": float(np.sum(np.isfinite(arr_a) & np.isfinite(arr_b))),
        "mean_a": float(np.nanmean(arr_a)),
        "mean_b": float(np.nanmean(arr_b)),
        "t": float(stat),
        "p": float(pvalue),
    }


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def components_to_variance_threshold(eigenvalues: np.ndarray, threshold: float) -> int:
    positive = np.asarray(eigenvalues, dtype=float)
    positive = positive[np.isfinite(positive)]
    positive = positive[positive > 0]
    if positive.size == 0:
        return 0
    positive = np.sort(positive)[::-1]
    explained = positive / positive.sum() * 100.0
    cumulative = np.cumsum(explained)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def flatten(iterables: Iterable[Iterable[str]]) -> list[str]:
    return [item for iterable in iterables for item in iterable]


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1)
    if math.isclose(std, 0.0):
        return np.zeros_like(values)
    return (values - values.mean()) / std
