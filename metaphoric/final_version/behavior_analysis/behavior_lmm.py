"""
behavior_lmm.py

用途
- 对行为数据做线性混合模型（LMM / mixed-effects model），以更稳健地处理：
  - 被试内重复测量（subject 随机效应）
  - item 级别差异（item 作为方差成分 vc_formula）

输入
- input_path: TSV/CSV（至少包含 subject、condition、response；item 可选）
- --response: 因变量列名（默认 accuracy）
- --subject-col / --item-col / --condition-col

输出（output_dir）
- `behavior_lmm_summary.txt`：statsmodels 的模型摘要
- `behavior_lmm_params.tsv`：固定效应与随机效应参数
- `behavior_lmm_model.json`：AIC/BIC/样本量等元信息
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

from common.final_utils import ensure_dir, save_json, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-effects modeling for behavior data.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--response", default="accuracy")
    parser.add_argument("--subject-col", default="subject")
    parser.add_argument("--item-col", default="item")
    parser.add_argument("--condition-col", default="condition")
    args = parser.parse_args()

    import statsmodels.formula.api as smf

    sep = "\t" if args.input_path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(args.input_path, sep=sep)
    if args.item_col not in frame.columns:
        frame[args.item_col] = range(1, len(frame) + 1)

    # Fixed effect: condition；Random effect: subject intercept；Variance component: item intercept
    formula = f"{args.response} ~ C({args.condition_col})"
    model = smf.mixedlm(
        formula,
        data=frame,
        groups=frame[args.subject_col],
        vc_formula={"item": f"0 + C({args.item_col})"},
        re_formula="1",
    )
    fit = model.fit(method="lbfgs", maxiter=200, disp=False)

    output_dir = ensure_dir(args.output_dir)
    (output_dir / "behavior_lmm_summary.txt").write_text(str(fit.summary()), encoding="utf-8")
    write_table(pd.DataFrame({"term": fit.params.index, "estimate": fit.params.values}), output_dir / "behavior_lmm_params.tsv")
    save_json({"aic": float(fit.aic), "bic": float(fit.bic), "n_obs": int(fit.nobs)}, output_dir / "behavior_lmm_model.json")


if __name__ == "__main__":
    main()
