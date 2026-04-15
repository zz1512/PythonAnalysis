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


def fit_model(frame: pd.DataFrame):
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        raise RuntimeError(f"statsmodels is required: {exc}")

    working = frame.copy()
    if "time" not in working.columns and "stage" in working.columns:
        working["time"] = working["stage"]
    if "item" not in working.columns:
        if "pair_id" in working.columns:
            working["item"] = working["pair_id"]
        elif "word_label" in working.columns:
            working["item"] = working["word_label"]
        else:
            working["item"] = range(1, len(working) + 1)

    model = smf.mixedlm(
        "similarity ~ C(condition) * C(time)",
        data=working,
        groups=working["subject"],
        vc_formula={"item": "0 + C(item)"},
        re_formula="1",
    )
    return model.fit(method="lbfgs", maxiter=200, disp=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-effects model for item-wise RSA results.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    sep = "\t" if args.input_path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(args.input_path, sep=sep)
    fit = fit_model(frame)

    summary_text = str(fit.summary())
    (output_dir / "rsa_lmm_summary.txt").write_text(summary_text, encoding="utf-8")
    params = pd.DataFrame({"term": fit.params.index, "estimate": fit.params.values})
    write_table(params, output_dir / "rsa_lmm_params.tsv")
    save_json({"aic": float(fit.aic), "bic": float(fit.bic), "n_obs": int(fit.nobs)}, output_dir / "rsa_lmm_model.json")


if __name__ == "__main__":
    main()
