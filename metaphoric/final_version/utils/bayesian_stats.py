"""
bayesian_stats.py

用途
- 为 t 检验输出 Bayes Factor（BF10），用于论文中“支持零假设/连续证据强度”的稳健性呈现。
- 实现策略：优先使用 `pingouin.ttest()` 的 BF10；若未安装 pingouin，则仍输出频率派 t/p，
  并将 BF10 标记为 NaN，同时提示如何安装（保证脚本可运行，不会因缺依赖直接崩溃）。

输入
- input_path: TSV/CSV
- --mode: paired / one-sample
- --a-col / --b-col: 列名
- --group-col: 可选按组计算（例如 roi）
- --alternative / --r: pingouin 的先验参数（若可用）

输出（output_dir）
- `bayesfactor_ttest.tsv`：每组的 n/t/p/BF10
- `bayesfactor_meta.json`：元信息

论文意义
- BF10 提供“连续证据强度”，在 p 值边缘或你希望表达“更支持零假设”的情形下尤其有用。
- 与频率派检验互补：p 值回答“在零假设下观察到当前或更极端数据的概率”，BF10 更像“数据支持 H1 的程度”。

结果解读（经验口径）
- BF10 > 1 倾向支持备择，BF10 < 1 倾向支持零假设；阈值解释应结合领域惯例与先验设定。
- 若 BF10 为 NaN，通常是因为未安装 `pingouin`，此时脚本只给出 t/p 作为最低限度输出。

常见坑
- 不同实现/版本对 BF 的先验（例如 Cauchy r）默认值不同；写论文时要记录 `--r` 与 `--alternative`。
- 把 BF10 当成“显著/不显著”二元判断：更推荐报告为证据强度并与效应量/CI 一起呈现。
"""

from __future__ import annotations



import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


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


def _try_pingouin_bf(
    a: np.ndarray,
    b: np.ndarray | None,
    *,
    paired: bool,
    alternative: str,
    r: float,
) -> float | None:
    """
    Use pingouin's implementation if available.
    Returns None if pingouin is not installed.
    """
    try:
        import pingouin as pg  # type: ignore
    except Exception:
        return None

    if paired:
        if b is None:
            raise ValueError("paired=True requires b.")
        t_res = pg.ttest(a, b, paired=True, alternative=alternative)
    else:
        # one-sample t-test around 0 for now
        t_res = pg.ttest(a, 0.0, paired=False, alternative=alternative)
    # pingouin returns BF10 column as string or float depending on version
    bf10 = t_res.loc[0, "BF10"]
    try:
        return float(bf10)
    except Exception:
        return None


def bayes_factor_ttest(
    a: np.ndarray,
    b: np.ndarray | None = None,
    *,
    paired: bool = True,
    alternative: str = "two-sided",
    r: float = 0.707,
) -> dict[str, Any]:
    """
    Compute BF10 for a t-test.

    Implementation note:
    - If pingouin is installed, use pg.ttest() -> BF10 directly (recommended).
    - Otherwise, we still compute and return the frequentist t-test summary,
      and set BF10 to NaN with an explicit reason.
    """
    a = np.asarray(a, dtype=float)
    if b is not None:
        b = np.asarray(b, dtype=float)

    if paired:
        if b is None:
            raise ValueError("paired=True requires b.")
        valid = np.isfinite(a) & np.isfinite(b)
        a2 = a[valid]
        b2 = b[valid]
        t_stat, p_val = stats.ttest_rel(a2, b2, nan_policy="omit")
        n = int(np.sum(valid))
    else:
        valid = np.isfinite(a)
        a2 = a[valid]
        t_stat, p_val = stats.ttest_1samp(a2, popmean=0.0, nan_policy="omit")
        n = int(np.sum(valid))

    bf10 = _try_pingouin_bf(a2, b2 if paired else None, paired=paired, alternative=alternative, r=r)
    if bf10 is None:
        bf10 = float("nan")
        bf_note = (
            "BF10 unavailable because pingouin is not installed. "
            "Install with `pip install pingouin` to enable Bayes factors."
        )
    else:
        bf_note = "BF10 computed via pingouin."

    return {
        "n": n,
        "t": float(t_stat) if np.isfinite(t_stat) else float("nan"),
        "p": float(p_val) if np.isfinite(p_val) else float("nan"),
        "bf10": float(bf10),
        "bf_note": bf_note,
        "paired": bool(paired),
        "alternative": str(alternative),
        "r": float(r),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Bayes factor (BF10) for t-tests (uses pingouin if installed).")
    parser.add_argument("input_path", type=Path, help="Input TSV/CSV.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument("--mode", choices=["paired", "one-sample"], default="paired")
    parser.add_argument("--a-col", required=True, help="Column name for sample A (or the one-sample values).")
    parser.add_argument("--b-col", help="Column name for sample B (paired mode only).")
    parser.add_argument("--group-col", default=None, help="Optional: compute BF per group (e.g., roi).")
    parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")
    parser.add_argument("--r", type=float, default=0.707, help="Cauchy prior scale (used by pingouin).")
    args = parser.parse_args()

    frame = read_table(args.input_path)
    output_dir = ensure_dir(args.output_dir)

    if args.a_col not in frame.columns:
        raise ValueError(f"Missing column: {args.a_col}")
    if args.mode == "paired" and (not args.b_col or args.b_col not in frame.columns):
        raise ValueError("paired mode requires --b-col and the column must exist.")

    groups = [("ALL", frame)]
    if args.group_col:
        if args.group_col not in frame.columns:
            raise ValueError(f"Missing group column: {args.group_col}")
        groups = [(str(k), v) for k, v in frame.groupby(args.group_col)]

    rows: list[dict[str, Any]] = []
    for group_value, group_frame in groups:
        if args.mode == "paired":
            summary = bayes_factor_ttest(
                group_frame[args.a_col].to_numpy(),
                group_frame[args.b_col].to_numpy(),  # type: ignore[arg-type]
                paired=True,
                alternative=args.alternative,
                r=args.r,
            )
        else:
            summary = bayes_factor_ttest(
                group_frame[args.a_col].to_numpy(),
                paired=False,
                alternative=args.alternative,
                r=args.r,
            )

        row = {"group": group_value, **summary}
        rows.append(row)

    out_frame = pd.DataFrame(rows)
    write_table(out_frame, output_dir / "bayesfactor_ttest.tsv")
    save_json({"n_groups": int(len(rows)), "mode": args.mode}, output_dir / "bayesfactor_meta.json")


if __name__ == "__main__":
    main()
