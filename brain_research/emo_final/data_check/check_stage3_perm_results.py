#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Step3 置换检验结果")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--trial-dir-name", type=str, default="by_stimulus")
    p.add_argument("--emotion-dir-name", type=str, default="by_emotion")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--check-figures", action="store_true")
    return p.parse_args()


def _count_files(glob_root: Path, pattern: str) -> int:
    return int(len(list(glob_root.glob(pattern)))) if glob_root.exists() else 0


def main() -> None:
    args = parse_args()
    rows = []

    def scan_branch(stimulus_dir_name: str) -> None:
        by_stim = args.matrix_dir / str(stimulus_dir_name)
        if not by_stim.exists():
            return
        for d in sorted([p for p in by_stim.iterdir() if p.is_dir()]):
            row = {"branch": str(stimulus_dir_name), "stimulus_type": d.name}
            p = d / "roi_isc_dev_models_perm_fwer.csv"
            row["result_csv_ok"] = bool(p.exists())
            if not p.exists():
                rows.append(row)
                continue
            try:
                df = pd.read_csv(p)
                need = {"roi", "model", "r_obs", "p_perm_one_tailed", "p_fwer_model_wise", "p_fdr_bh_model_wise", "p_fdr_bh_global"}
                miss = need - set(df.columns)
                if miss:
                    raise ValueError(f"{p} 缺少列: {sorted(miss)}")
                row["n_tests"] = int(df.shape[0])
                row["n_rois"] = int(df["roi"].astype(str).nunique())
                row["n_models"] = int(df["model"].astype(str).nunique())
                row["r_obs_finite_ratio"] = float(df["r_obs"].apply(lambda x: pd.notna(x)).mean())
                sig_fwer = df[(df["r_obs"] > 0) & (df["p_fwer_model_wise"] <= float(args.alpha))]
                row["n_sig_pos_fwer"] = int(sig_fwer.shape[0])
                sig_fdr = df[(df["r_obs"] > 0) & (df["p_fdr_bh_model_wise"] <= float(args.alpha))]
                row["n_sig_pos_fdr_model_wise"] = int(sig_fdr.shape[0])

                if bool(args.check_figures):
                    fig_root = args.matrix_dir / "figures"
                    heat_root = fig_root / str(stimulus_dir_name)
                    brain_root = fig_root / f"{str(stimulus_dir_name)}_brain_maps"
                    traj_root = fig_root / f"{str(stimulus_dir_name)}_pair_age_traj"

                    row["heatmap_png_count"] = _count_files(heat_root, "*sig_heatmap_*.png")
                    row["heatmap_csv_count"] = _count_files(heat_root, "*sig_results_*.csv")

                    row["brain_gii_count"] = _count_files(brain_root, f"brain_map_{stimulus_dir_name}_{d.name}_*_a*_surf_*.func.gii")
                    row["brain_vol_count"] = _count_files(brain_root, f"brain_map_{stimulus_dir_name}_{d.name}_*_a*_volume.nii.gz")

                    if int(row.get("n_sig_pos_fwer", 0)) > 0:
                        row["traj_png_count"] = _count_files(traj_root, f"*__{d.name}__*.png")
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)

    scan_branch(args.trial_dir_name)
    scan_branch(args.emotion_dir_name)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["branch", "stimulus_type"])
    out.to_csv(args.matrix_dir / "data_check_stage3.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
