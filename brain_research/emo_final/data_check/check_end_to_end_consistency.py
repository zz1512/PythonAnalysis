from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="端到端一致性检查：RSM/pattern/ISC/perm 的 subjects/rois/prefix 对齐")
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--trial-dir-name", type=str, default="by_stimulus")
    p.add_argument("--emotion-dir-name", type=str, default="by_emotion")
    p.add_argument("--trial-repr-prefix", type=str, default="roi_repr_matrix_232")
    p.add_argument("--emotion-repr-prefix", type=str, default="roi_repr_matrix_232_emotion4")
    p.add_argument("--pattern-prefix", type=str, default="roi_beta_patterns_232")
    p.add_argument("--isc-prefix", type=str, default="roi_isc_mahalanobis_by_age")
    p.add_argument("--perm-file", type=str, default="roi_isc_dev_models_perm_fwer.csv")
    return p.parse_args()


def _safe_read_csv(path: Path, col: str) -> List[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"{path} 缺少列: {col}")
    return df[col].astype(str).tolist()


def _exists_all(paths: List[Path]) -> bool:
    return all(p.exists() for p in paths)


def _load_rois_subjects(stim_dir: Path, prefix: str) -> Tuple[List[str], List[str]]:
    rois = _safe_read_csv(stim_dir / f"{prefix}_rois.csv", "roi")
    subs = _safe_read_csv(stim_dir / f"{prefix}_subjects.csv", "subject")
    return rois, subs


def _load_isc_rois_subjects(stim_dir: Path, isc_prefix: str) -> Tuple[List[str], List[str]]:
    rois = _safe_read_csv(stim_dir / f"{isc_prefix}_rois.csv", "roi")
    subs = _safe_read_csv(stim_dir / f"{isc_prefix}_subjects_sorted.csv", "subject")
    return rois, subs


def _check_set_equal(a: List[str], b: List[str]) -> bool:
    return set(a) == set(b)


def _check_order_equal(a: List[str], b: List[str]) -> bool:
    return list(a) == list(b)


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    rows: List[Dict[str, object]] = []

    branches = [
        ("trial", str(args.trial_dir_name), str(args.trial_repr_prefix), True),
        ("emotion", str(args.emotion_dir_name), str(args.emotion_repr_prefix), False),
    ]

    for branch_name, stim_root_name, repr_prefix, expect_patterns in branches:
        stim_root = matrix_dir / stim_root_name
        if not stim_root.exists():
            continue
        for stim_dir in sorted([p for p in stim_root.iterdir() if p.is_dir()]):
            row: Dict[str, object] = {
                "branch": branch_name,
                "stimulus_type": stim_dir.name,
            }

            repr_npz = stim_dir / f"{repr_prefix}.npz"
            repr_rois = stim_dir / f"{repr_prefix}_rois.csv"
            repr_subs = stim_dir / f"{repr_prefix}_subjects.csv"
            order_csv = stim_dir / "stimulus_order.csv"
            row["repr_files_ok"] = bool(_exists_all([repr_npz, repr_rois, repr_subs, order_csv]))

            isc_npy = stim_dir / f"{args.isc_prefix}.npy"
            isc_rois = stim_dir / f"{args.isc_prefix}_rois.csv"
            isc_subs = stim_dir / f"{args.isc_prefix}_subjects_sorted.csv"
            row["isc_files_ok"] = bool(_exists_all([isc_npy, isc_rois, isc_subs]))

            perm_csv = stim_dir / str(args.perm_file)
            row["perm_file_ok"] = bool(perm_csv.exists())

            if not row["repr_files_ok"]:
                rows.append(row)
                continue

            try:
                rois_repr, subs_repr = _load_rois_subjects(stim_dir, prefix=repr_prefix)
                row["n_rois_repr"] = int(len(rois_repr))
                row["n_subjects_repr"] = int(len(subs_repr))

                if bool(expect_patterns):
                    pat_npz = stim_dir / f"{args.pattern_prefix}.npz"
                    pat_rois = stim_dir / f"{args.pattern_prefix}_rois.csv"
                    pat_subs = stim_dir / f"{args.pattern_prefix}_subjects.csv"
                    pat_sizes = stim_dir / f"{args.pattern_prefix}_roi_feat_sizes.csv"
                    row["pattern_files_ok"] = bool(_exists_all([pat_npz, pat_rois, pat_subs, pat_sizes]))
                    if row["pattern_files_ok"]:
                        rois_pat = _safe_read_csv(pat_rois, "roi")
                        subs_pat = _safe_read_csv(pat_subs, "subject")
                        row["pattern_subjects_equal"] = bool(_check_order_equal(subs_repr, subs_pat))
                        row["pattern_rois_equal"] = bool(_check_order_equal(rois_repr, rois_pat))
                        npz = np.load(pat_npz)
                        missing_keys = [s for s in subs_pat if s not in npz]
                        row["pattern_missing_subject_keys"] = int(len(missing_keys))

                if row["isc_files_ok"]:
                    rois_isc, subs_isc = _load_isc_rois_subjects(stim_dir, isc_prefix=str(args.isc_prefix))
                    row["n_rois_isc"] = int(len(rois_isc))
                    row["n_subjects_isc"] = int(len(subs_isc))
                    row["rois_equal_repr_isc"] = bool(_check_order_equal(rois_repr, rois_isc))
                    row["subjects_set_equal_repr_isc"] = bool(_check_set_equal(subs_repr, subs_isc))
                    row["subjects_order_equal_repr_isc"] = bool(_check_order_equal(subs_repr, subs_isc))

                if row["perm_file_ok"]:
                    dfp = pd.read_csv(perm_csv)
                    if "roi" in dfp.columns:
                        row["perm_n_rois"] = int(dfp["roi"].astype(str).nunique())
                    if "model" in dfp.columns:
                        row["perm_n_models"] = int(dfp["model"].astype(str).nunique())
                    if "p_fwer_model_wise" in dfp.columns:
                        row["perm_min_p_fwer"] = float(np.nanmin(dfp["p_fwer_model_wise"].to_numpy(dtype=float)))
            except Exception as e:
                row["error"] = str(e)

            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["branch", "stimulus_type"])
    out_path = matrix_dir / "data_check_end_to_end.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

