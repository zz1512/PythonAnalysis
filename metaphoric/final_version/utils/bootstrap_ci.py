"""
bootstrap_ci.py

用途
- 对一个数值列做 percentile bootstrap 的均值置信区间（CI），可按分组列分别计算。
- 主要用于论文中的稳健性呈现：给 RD/GPS/MVPA/RSA 等指标补充 95% CI。

输入
- input_path: TSV/CSV
- --value-col: 要 bootstrap 的数值列
- --group-cols: 可选分组列（例如 roi/time/condition）
- --confidence / --n-boot / --seed

输出（output_dir）
- `bootstrap_mean_ci.tsv`：每组的 mean + CI
- `bootstrap_mean_ci_meta.json`：参数与元信息

论文意义
- 在样本量不大或分布偏态时，bootstrap CI 可作为对点估计（均值）不确定性的直观补充，
  让“效应方向/幅度”比“是否过阈值”更可解释。

结果解读
- CI 宽：不确定性大，常见原因包括被试少、组内方差大、指标不稳定。
- CI 若跨过 0 并不等价于“零假设为真”，它只是说明在当前数据与抽样假设下证据不够集中。

常见坑
- 把 bootstrap CI 当作多重比较校正的替代品：CI 解决的是不确定性呈现，不自动控制全管道假阳性率。
- `group-cols` 过细导致每组 n 很小，CI 会非常不稳定（甚至全是 NaN）。
"""

from __future__ import annotations



import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, percentile_bootstrap_ci, read_table, save_json, write_table  # noqa: E402


def bootstrap_mean_ci(values: np.ndarray, confidence: float, n_boot: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    ci = percentile_bootstrap_ci(values, confidence=confidence, n_boot=n_boot, seed=seed)
    return {"mean": float(values.mean()), "ci_low": float(ci["low"]), "ci_high": float(ci["high"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute percentile bootstrap CI for the mean of a column.")
    parser.add_argument("input_path", type=Path, help="Input TSV/CSV.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument("--value-col", required=True, help="Numeric column to bootstrap.")
    parser.add_argument("--group-cols", nargs="*", default=[], help="Optional grouping columns.")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = read_table(args.input_path)
    if args.value_col not in frame.columns:
        raise ValueError(f"Missing value column: {args.value_col}")

    output_dir = ensure_dir(args.output_dir)
    group_cols = [col for col in args.group_cols if col]

    groups = [("ALL", frame)]
    if group_cols:
        for col in group_cols:
            if col not in frame.columns:
                raise ValueError(f"Missing group column: {col}")
        groups = []
        for keys, sub in frame.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = "|".join(str(k) for k in keys)
            groups.append((label, sub))

    rows: list[dict[str, Any]] = []
    for group_label, sub in groups:
        summary = bootstrap_mean_ci(sub[args.value_col].to_numpy(), args.confidence, args.n_boot, args.seed)
        rows.append({"group": group_label, "n": int(sub[args.value_col].notna().sum()), **summary})

    out = pd.DataFrame(rows)
    write_table(out, output_dir / "bootstrap_mean_ci.tsv")
    save_json(
        {
            "input": str(args.input_path),
            "value_col": args.value_col,
            "group_cols": group_cols,
            "confidence": float(args.confidence),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
        output_dir / "bootstrap_mean_ci_meta.json",
    )


if __name__ == "__main__":
    main()
