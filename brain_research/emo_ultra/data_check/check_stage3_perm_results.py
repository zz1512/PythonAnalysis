#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Step3 置换检验结果")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for d in sorted([p for p in (args.matrix_dir / "by_stimulus").iterdir() if p.is_dir()]):
        p = d / "roi_isc_dev_models_perm_fwer.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sig = df[(df["r_obs"] > 0) & (df["p_fwer_model_wise"] <= float(args.alpha))]
        rows.append({"stimulus_type": d.name, "n_tests": int(df.shape[0]), "n_sig_pos_fwer": int(sig.shape[0])})
    out = pd.DataFrame(rows).sort_values("stimulus_type")
    out.to_csv(args.matrix_dir / "data_check_stage3.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
