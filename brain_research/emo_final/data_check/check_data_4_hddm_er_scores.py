#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_BEHAVIOR_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/behavior")
SUBJECT_COL_CANDIDATES = ("participant_id", "participant", "subject", "sub_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare subject-wise score sequences between data_4_hddm_ER.csv and data_4_hddm_ER_base.csv")
    p.add_argument("--behavior-dir", type=Path, default=DEFAULT_BEHAVIOR_DIR)
    p.add_argument("--main-file", type=str, default="data_4_hddm_ER.csv")
    p.add_argument("--base-file", type=str, default="data_4_hddm_ER_base.csv")
    p.add_argument("--score-col", type=str, default="emot_rating")
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=1e-8)
    return p.parse_args()


def _find_subject_col(df: pd.DataFrame, path: Path) -> str:
    for col in SUBJECT_COL_CANDIDATES:
        if col in df.columns:
            return str(col)
    raise ValueError(f"Cannot find subject column in {path}. Candidates: {list(SUBJECT_COL_CANDIDATES)}")


def _normalize_subject_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("^sub-", "", regex=True)
    return s


def _values_equal(a: float, b: float, rtol: float, atol: float) -> bool:
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    if np.isfinite(a) != np.isfinite(b):
        return False
    return bool(np.isclose(a, b, rtol=float(rtol), atol=float(atol), equal_nan=True))


def compare_subject_scores(
    df_main: pd.DataFrame,
    df_base: pd.DataFrame,
    main_subject_col: str,
    base_subject_col: str,
    score_col: str,
    rtol: float,
    atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_main = df_main.copy()
    df_base = df_base.copy()
    df_main["_subject_norm"] = _normalize_subject_series(df_main[main_subject_col])
    df_base["_subject_norm"] = _normalize_subject_series(df_base[base_subject_col])

    subject_order = df_main["_subject_norm"].drop_duplicates().astype(str).tolist()
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for subject in subject_order:
        sub_main = df_main[df_main["_subject_norm"] == str(subject)].copy()
        sub_base = df_base[df_base["_subject_norm"] == str(subject)].copy()

        n_main = int(sub_main.shape[0])
        n_base = int(sub_base.shape[0])
        if n_base == 0:
            summary_rows.append(
                {
                    "subject": str(subject),
                    "status": "missing_in_base",
                    "n_rows_main": n_main,
                    "n_rows_base": 0,
                    "n_equal_rows": 0,
                    "n_diff_rows": n_main,
                    "first_diff_index_1based": 1 if n_main > 0 else np.nan,
                    "main_first_diff_value": float(pd.to_numeric(sub_main[score_col], errors="coerce").iloc[0]) if n_main > 0 else np.nan,
                    "base_first_diff_value": np.nan,
                }
            )
            for i, val in enumerate(pd.to_numeric(sub_main[score_col], errors="coerce").tolist(), start=1):
                detail_rows.append(
                    {
                        "subject": str(subject),
                        "row_index_1based": int(i),
                        "status": "missing_in_base",
                        "main_value": float(val) if np.isfinite(val) else np.nan,
                        "base_value": np.nan,
                    }
                )
            continue

        vals_main = pd.to_numeric(sub_main[score_col], errors="coerce").to_numpy(dtype=float)
        vals_base = pd.to_numeric(sub_base[score_col], errors="coerce").to_numpy(dtype=float)
        n_min = int(min(len(vals_main), len(vals_base)))
        equal_mask = np.array([_values_equal(vals_main[i], vals_base[i], rtol=rtol, atol=atol) for i in range(n_min)], dtype=bool)

        if n_main != n_base:
            status = "row_count_mismatch"
        elif bool(equal_mask.all()) and n_main == n_base:
            status = "pass"
        else:
            status = "score_mismatch"

        diff_positions = [i + 1 for i, ok in enumerate(equal_mask.tolist()) if not bool(ok)]
        if n_main != n_base:
            diff_positions.extend(range(n_min + 1, max(n_main, n_base) + 1))
        diff_positions = sorted(set(diff_positions))

        first_idx = diff_positions[0] if diff_positions else np.nan
        main_first = vals_main[int(first_idx) - 1] if np.isfinite(first_idx) and int(first_idx) <= n_main else np.nan
        base_first = vals_base[int(first_idx) - 1] if np.isfinite(first_idx) and int(first_idx) <= n_base else np.nan

        summary_rows.append(
            {
                "subject": str(subject),
                "status": str(status),
                "n_rows_main": n_main,
                "n_rows_base": n_base,
                "n_equal_rows": int(equal_mask.sum()),
                "n_diff_rows": int(len(diff_positions)),
                "first_diff_index_1based": first_idx,
                "main_first_diff_value": float(main_first) if np.isfinite(main_first) else np.nan,
                "base_first_diff_value": float(base_first) if np.isfinite(base_first) else np.nan,
            }
        )

        if status != "pass":
            for idx in diff_positions:
                i0 = int(idx) - 1
                main_val = vals_main[i0] if i0 < n_main else np.nan
                base_val = vals_base[i0] if i0 < n_base else np.nan
                detail_rows.append(
                    {
                        "subject": str(subject),
                        "row_index_1based": int(idx),
                        "status": str(status),
                        "main_value": float(main_val) if np.isfinite(main_val) else np.nan,
                        "base_value": float(base_val) if np.isfinite(base_val) else np.nan,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def main() -> None:
    args = parse_args()
    behavior_dir = Path(args.behavior_dir)
    main_path = behavior_dir / str(args.main_file)
    base_path = behavior_dir / str(args.base_file)

    if not main_path.exists():
        raise FileNotFoundError(f"Cannot find main file: {main_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"Cannot find base file: {base_path}")

    df_main = pd.read_csv(main_path)
    df_base = pd.read_csv(base_path)
    if str(args.score_col) not in df_main.columns:
        raise ValueError(f"{main_path} missing score column: {args.score_col}")
    if str(args.score_col) not in df_base.columns:
        raise ValueError(f"{base_path} missing score column: {args.score_col}")

    main_subject_col = _find_subject_col(df_main, main_path)
    base_subject_col = _find_subject_col(df_base, base_path)

    summary_df, detail_df = compare_subject_scores(
        df_main=df_main,
        df_base=df_base,
        main_subject_col=main_subject_col,
        base_subject_col=base_subject_col,
        score_col=str(args.score_col),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    out_prefix = f"check_{main_path.stem}_vs_{base_path.stem}_{str(args.score_col)}"
    summary_path = behavior_dir / f"{out_prefix}_summary.csv"
    detail_path = behavior_dir / f"{out_prefix}_details.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    counts = summary_df["status"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
    print(f"Main file: {main_path}")
    print(f"Base file: {base_path}")
    print(f"Score column: {args.score_col}")
    print(f"Compared subjects: {int(summary_df.shape[0])}")
    print(f"Status counts: {counts}")
    print(f"Summary saved to: {summary_path}")
    print(f"Details saved to: {detail_path}")


if __name__ == "__main__":
    main()
