from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="检查 Stage0 (LSS) 输出：aligned index + audit + beta 文件存在性")
    p.add_argument("--lss-root", type=Path, required=True, help="LSS 输出根目录（包含 lss_index_*_aligned.csv）")
    p.add_argument("--check-files", action="store_true", help="检查 index 里每条记录对应的 beta 文件是否存在")
    p.add_argument("--max-missing-report", type=int, default=50, help="最多展示多少条缺失文件样例")
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_subject(x: object) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s.startswith("sub-") else f"sub-{s}"


def _resolve_beta_path(lss_root: Path, row: pd.Series) -> Path:
    f = str(row.get("file", "")).strip()
    if f.startswith("/"):
        return Path(f)
    task = str(row.get("task", "")).strip()
    sub = str(row.get("subject", "")).strip()
    p1 = lss_root / task / sub / f
    if p1.exists():
        return p1
    run = str(row.get("run", "")).strip()
    if run:
        p2 = lss_root / task / sub / f"run-{run}" / f
        return p2
    return p1


def _check_required_cols(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    miss = [c for c in required if c not in df.columns]
    return (len(miss) == 0), miss


def _task_coverage(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    g = df.groupby("subject")["task"].agg(lambda x: set([str(t).strip().upper() for t in x.dropna().tolist()]))
    out["n_subjects"] = int(g.shape[0])
    out["n_subjects_has_emo"] = int(sum("EMO" in s for s in g))
    out["n_subjects_has_soc"] = int(sum("SOC" in s for s in g))
    out["n_subjects_has_both_emo_soc"] = int(sum({"EMO", "SOC"}.issubset(s) for s in g))
    return out


def _stimulus_content_stats(df: pd.DataFrame) -> Dict[str, object]:
    if "stimulus_content" not in df.columns:
        return {"has_stimulus_content": False}
    sc = df["stimulus_content"].astype(str).str.strip()
    ok = sc.str.contains("_").mean() if sc.shape[0] > 0 else 0.0
    return {
        "has_stimulus_content": True,
        "stimulus_content_unique": int(sc.nunique(dropna=True)),
        "stimulus_content_underscore_ratio": float(ok),
    }


def main() -> None:
    args = parse_args()
    lss_root = Path(args.lss_root)
    index_files = sorted(lss_root.glob("lss_index_*_aligned.csv"))
    if not index_files:
        raise FileNotFoundError(f"未找到 aligned index: {lss_root}/lss_index_*_aligned.csv")

    rows = []
    missing_examples: List[Dict[str, str]] = []

    for f in index_files:
        df = _read_csv(f)
        if "subject" in df.columns:
            df["subject"] = df["subject"].apply(_normalize_subject)
        if "task" in df.columns:
            df["task"] = df["task"].astype(str).str.strip().str.upper()

        required = ["subject", "task", "file"]
        ok_cols, miss = _check_required_cols(df, required)
        row: Dict[str, object] = {
            "index_file": str(f.name),
            "rows": int(df.shape[0]),
            "cols_ok": bool(ok_cols),
            "missing_cols": ",".join(miss),
        }
        row.update(_task_coverage(df) if "subject" in df.columns and "task" in df.columns else {})
        row.update(_stimulus_content_stats(df))

        if bool(args.check_files) and ok_cols:
            exists_flags = []
            for _, r in df.iterrows():
                p = _resolve_beta_path(lss_root, r)
                ok = p.exists()
                exists_flags.append(ok)
                if (not ok) and len(missing_examples) < int(args.max_missing_report):
                    missing_examples.append(
                        {
                            "index_file": str(f.name),
                            "subject": str(r.get("subject", "")),
                            "task": str(r.get("task", "")),
                            "file": str(r.get("file", "")),
                            "resolved_path": str(p),
                        }
                    )
            row["beta_exists_ratio"] = float(np.mean(exists_flags)) if exists_flags else float("nan")
            row["beta_missing_count"] = int(np.sum([not x for x in exists_flags])) if exists_flags else 0

        audit_guess = lss_root / str(f.name).replace("lss_index_", "lss_audit_").replace("_aligned.csv", ".csv")
        row["audit_file_guess"] = str(audit_guess.name)
        if audit_guess.exists():
            adf = _read_csv(audit_guess)
            row["audit_rows"] = int(adf.shape[0])
            if "status" in adf.columns:
                vc = adf["status"].astype(str).value_counts(dropna=False)
                row["audit_status_counts"] = ";".join([f"{k}:{int(v)}" for k, v in vc.items()])
            if "trial_type" in adf.columns:
                tv = adf["trial_type"].astype(str).value_counts(dropna=False)
                row["audit_trial_type_counts"] = ";".join([f"{k}:{int(v)}" for k, v in tv.items()])
        else:
            row["audit_rows"] = ""
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("index_file") if not out.empty else out
    out_path = lss_root / "data_check_stage0_lss.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))

    if missing_examples:
        ex = pd.DataFrame(missing_examples)
        ex_path = lss_root / "data_check_stage0_missing_beta_examples.csv"
        ex.to_csv(ex_path, index=False)
        print(f"\n缺失 beta 样例已写入: {ex_path}")


if __name__ == "__main__":
    main()

