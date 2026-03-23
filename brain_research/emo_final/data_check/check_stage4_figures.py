from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Stage4-6 绘图输出（热图/脑图/轨迹）")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--trial-dir-name", type=str, default="by_stimulus")
    p.add_argument("--emotion-dir-name", type=str, default="by_emotion")
    p.add_argument("--stimulus-types", type=str, default=None, help="可选：逗号分隔，仅检查这些 stimulus_type；默认检查所有")
    return p.parse_args()


def _parse_list(x: Optional[str]) -> Optional[List[str]]:
    if x is None:
        return None
    xs = [s.strip() for s in str(x).split(",") if s.strip()]
    return xs if xs else None


def _count(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return int(len(list(root.glob(pattern))))


def _iter_stimulus_types(matrix_dir: Path, stimulus_dir_name: str, subset: Optional[List[str]]) -> List[str]:
    by_stim = matrix_dir / stimulus_dir_name
    if not by_stim.exists():
        return []
    if subset is not None:
        return subset
    return sorted([p.name for p in by_stim.iterdir() if p.is_dir()])


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    subset = _parse_list(args.stimulus_types)

    fig_root = matrix_dir / "figures"
    rows: List[Dict[str, object]] = []

    for branch, stim_dir_name in [("trial", str(args.trial_dir_name)), ("emotion", str(args.emotion_dir_name))]:
        stypes = _iter_stimulus_types(matrix_dir, stimulus_dir_name=stim_dir_name, subset=subset)

        heat_root = fig_root / stim_dir_name
        brain_root = fig_root / f"{stim_dir_name}_brain_maps"
        traj_root = fig_root / f"{stim_dir_name}_pair_age_traj"

        base = {
            "branch": branch,
            "stimulus_dir_name": stim_dir_name,
            "heatmap_png_total": _count(heat_root, "*sig_heatmap_*.png"),
            "heatmap_csv_total": _count(heat_root, "*sig_results_*.csv"),
            "brain_func_gii_total": _count(brain_root, "*.func.gii"),
            "brain_vol_nii_total": _count(brain_root, "*.nii.gz"),
            "traj_png_total": _count(traj_root, "*.png"),
        }
        rows.append(base)

        for st in stypes:
            rows.append(
                {
                    "branch": branch,
                    "stimulus_dir_name": stim_dir_name,
                    "stimulus_type": st,
                    "brain_func_gii_for_stimulus": _count(brain_root, f"brain_map_{stim_dir_name}_{st}_*_a*_surf_*.func.gii"),
                    "brain_vol_nii_for_stimulus": _count(brain_root, f"brain_map_{stim_dir_name}_{st}_*_a*_volume.nii.gz"),
                    "traj_png_for_stimulus": _count(traj_root, f"*__{st}__*.png"),
                }
            )

    out = pd.DataFrame(rows)
    out_path = matrix_dir / "data_check_stage4_figures.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

