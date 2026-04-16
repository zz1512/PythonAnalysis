"""
mediation_analysis.py

用途
- 对 subject-level 指标表做 bootstrap 中介分析（简化版）：X -> M -> Y
- 用于把“多层次并列证据”提升为“机制路径假设”（PNAS 加分项）。

输入
- input_path: 合并后的 subject-level TSV/CSV（通常来自 brain_behavior_correlation 的合并表）
- --x-col / --m-col / --y-col: 指定 X/M/Y 三列
- --n-boot / --seed

输出（output_dir）
- `mediation_summary.json`：包含 a/b/indirect 与 bootstrap CI
- `mediation_summary.tsv`：同内容的表格版

说明
- 这里的 b 路径使用了对 X 的残差化来实现“控制 X 后 M 对 Y 的贡献”。
- 样本量较小（~26）时，建议把中介作为补充分析，并报告 CI 与 bootstrap p。
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

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402


def _ols_coef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y - y.mean()) / denom)


def bootstrap_mediation(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(m) & np.isfinite(y)
    x = x[valid]
    m = m[valid]
    y = y[valid]

    n = x.size
    if n < 10:
        return {"n": int(n), "error": "Need at least 10 complete cases for bootstrap mediation."}

    # Path a: M ~ X
    a = _ols_coef(x, m)
    # Path b: Y ~ M (controlling for X) via residualization
    # b = coef(resid(M|X), resid(Y|X))
    mx = _ols_coef(x, m)
    yx = _ols_coef(x, y)
    m_res = m - (mx * (x - x.mean()) + m.mean())
    y_res = y - (yx * (x - x.mean()) + y.mean())
    b = _ols_coef(m_res, y_res)
    indirect = a * b

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        mb = m[idx]
        yb = y[idx]
        a_b = _ols_coef(xb, mb)
        mx_b = _ols_coef(xb, mb)
        yx_b = _ols_coef(xb, yb)
        m_res_b = mb - (mx_b * (xb - xb.mean()) + mb.mean())
        y_res_b = yb - (yx_b * (xb - xb.mean()) + yb.mean())
        b_b = _ols_coef(m_res_b, y_res_b)
        boots[i] = a_b * b_b

    low, high = np.quantile(boots[np.isfinite(boots)], [0.025, 0.975])
    p_boot = float(np.mean((boots <= 0) if indirect > 0 else (boots >= 0)))
    return {
        "n": int(n),
        "a": float(a),
        "b": float(b),
        "indirect": float(indirect),
        "indirect_ci_low": float(low),
        "indirect_ci_high": float(high),
        "indirect_p_boot": float(p_boot),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap mediation analysis on subject-level metrics.")
    parser.add_argument("input_path", type=Path, help="Input merged TSV/CSV with columns for X, M, Y.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--x-col", required=True)
    parser.add_argument("--m-col", required=True)
    parser.add_argument("--y-col", required=True)
    parser.add_argument("--subject-col", default="subject")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = read_table(args.input_path)
    for col in [args.x_col, args.m_col, args.y_col]:
        if col not in frame.columns:
            raise ValueError(f"Missing column: {col}")

    output_dir = ensure_dir(args.output_dir)
    result = bootstrap_mediation(
        frame[args.x_col].to_numpy(),
        frame[args.m_col].to_numpy(),
        frame[args.y_col].to_numpy(),
        n_boot=args.n_boot,
        seed=args.seed,
    )
    save_json(result, output_dir / "mediation_summary.json")
    write_table(pd.DataFrame([{"x": args.x_col, "m": args.m_col, "y": args.y_col, **result}]), output_dir / "mediation_summary.tsv")


if __name__ == "__main__":
    main()
