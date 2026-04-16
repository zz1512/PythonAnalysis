"""
final_utils.py

用途
- final_version 各分析脚本共享的通用工具函数（文件读写、统计摘要、差异中的差异等）。
- 尽量保持“轻依赖、可复用、可读”，让上层脚本只关注业务逻辑。

常见输入/输出
- 输入多为 TSV/CSV（`read_table()`），输出多为 TSV/CSV/JSON（`write_table()` / `save_json()`）。

论文意义
- 这类工具不直接产生“显著性结论”，但它们决定你的结果表是否可复现、统计摘要是否一致、
  以及各模块输出能否被后续步骤可靠读取（避免因为 IO/格式问题造成的隐性错误）。

结果解读
- `paired_t_summary()` / `one_sample_t_summary()` 等返回的是“快速 sanity check”的统计摘要。
  若要写入论文，仍应回到主分析脚本中确认模型设定、数据过滤与多重比较策略。

常见坑
- TSV/CSV 分隔符混用导致列读错（建议统一使用 `read_table()`/`write_table()`）。
- `save_json()` 默认 `ensure_ascii=False`，便于中文元信息；但跨语言环境读取时要确保 UTF-8。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        frame.to_csv(path, sep="\t", index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path}")


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def list_subject_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("sub-")])


def fisher_z_from_samples(samples: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(samples, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = np.arctanh(corr)
    fisher[~np.isfinite(fisher)] = 0.0
    return fisher


def mean_row_similarity(samples: np.ndarray) -> np.ndarray:
    fisher = fisher_z_from_samples(samples)
    if fisher.shape[0] <= 1:
        return np.zeros(fisher.shape[0], dtype=float)
    sums = fisher.sum(axis=1) - np.diag(fisher)
    return sums / max(fisher.shape[0] - 1, 1)


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    if diff.size == 0 or np.allclose(diff.std(ddof=1), 0.0):
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def paired_t_summary(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[valid]
    arr_b = arr_b[valid]
    if arr_a.size == 0:
        return {
            "n": 0.0,
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "cohens_dz": float("nan"),
        }
    stat, pvalue = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")
    return {
        "n": float(arr_a.size),
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "t": float(stat),
        "p": float(pvalue),
        "cohens_dz": cohens_dz(arr_a, arr_b),
    }


def one_sample_t_summary(values: Sequence[float], popmean: float = 0.0) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "t": float("nan"), "p": float("nan")}
    stat, pvalue = stats.ttest_1samp(arr, popmean=popmean, nan_policy="omit")
    return {"n": float(arr.size), "mean": float(np.mean(arr)), "t": float(stat), "p": float(pvalue)}


def percentile_bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats_boot[index] = float(np.mean(sample))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats_boot, [alpha, 1.0 - alpha])
    return {"low": float(low), "high": float(high)}


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


def rank_transform(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return stats.rankdata(values, axis=axis)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1)
    if math.isclose(std, 0.0):
        return np.zeros_like(values)
    return (values - values.mean()) / std


def interaction_rows_to_matrix(
    frame: pd.DataFrame,
    subject_col: str = "subject",
    condition_col: str = "condition",
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    return frame.pivot_table(
        index=subject_col,
        columns=[condition_col, time_col],
        values=value_col,
        aggfunc="mean",
    )


def difference_in_differences(
    frame: pd.DataFrame,
    *,
    subject_col: str = "subject",
    condition_col: str = "condition",
    time_col: str = "time",
    value_col: str = "value",
    metaphor_label: str = "yy",
    control_label: str = "kj",
    pre_label: str = "pre",
    post_label: str = "post",
) -> pd.DataFrame:
    pivot = interaction_rows_to_matrix(
        frame,
        subject_col=subject_col,
        condition_col=condition_col,
        time_col=time_col,
        value_col=value_col,
    )
    required = [
        (metaphor_label, pre_label),
        (metaphor_label, post_label),
        (control_label, pre_label),
        (control_label, post_label),
    ]
    for key in required:
        if key not in pivot.columns:
            raise ValueError(f"Missing cell in interaction table: {key}")
    result = pivot.copy()
    result["yy_delta"] = result[(metaphor_label, post_label)] - result[(metaphor_label, pre_label)]
    result["kj_delta"] = result[(control_label, post_label)] - result[(control_label, pre_label)]
    result["interaction_delta"] = result["yy_delta"] - result["kj_delta"]
    return result.reset_index()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part) != "")
        for column in frame.columns.to_flat_index()
    ]
    return frame


def safe_merge(frames: Iterable[pd.DataFrame], on: str = "subject") -> pd.DataFrame:
    frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=[on])
    output = frames[0]
    for frame in frames[1:]:
        output = output.merge(frame, on=on, how="outer")
    return output
