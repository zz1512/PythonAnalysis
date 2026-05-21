#!/usr/bin/env python3
"""Extract run3/run4 learning behavior from raw E-Prime CSV files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [FINAL_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lr_utils import (  # noqa: E402
    CONDITIONS,
    VALID_ORIGINAL_IDS,
    condition_item_id,
    extract_original_id,
    load_template_map,
    normalize_condition,
    write_tsv,
    zscore_grouped,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _extract_condition_from_stim(stim: object) -> str:
    text = str(stim).strip().lower()
    if text.startswith("yy"):
        return "yy"
    if text.startswith("kj"):
        return "kj"
    return ""


def _read_run_csv(path: Path, subject: str, run: int, template_map: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path)
    if run == 3:
        stim_col = "juzi"
        rt_col = "sentence3_RT"
        resp_col = "sentence3_RESP"
        acc_col = "sentence3_ACC"
        task = "comprehension"
    elif run == 4:
        stim_col = "juzi_run"
        rt_col = "sentence4_RT"
        resp_col = "sentence4_RESP"
        acc_col = "sentence4_ACC"
        task = "liking"
    else:
        raise ValueError(f"Unsupported learning run: {run}")

    required = [stim_col, rt_col, resp_col, acc_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "subject": subject,
            "run": run,
            "learning_task": task,
            "stimulus": frame[stim_col],
            "rt": pd.to_numeric(frame[rt_col], errors="coerce"),
            "resp": pd.to_numeric(frame[resp_col], errors="coerce"),
            "acc": pd.to_numeric(frame[acc_col], errors="coerce"),
        }
    )
    out["condition"] = out["stimulus"].map(_extract_condition_from_stim)
    out["original_pair_id"] = out["stimulus"].map(extract_original_id).astype("Int64")
    out = out[out["condition"].isin(CONDITIONS)].copy()
    out = out[out.apply(lambda row: int(row["original_pair_id"]) in VALID_ORIGINAL_IDS[row["condition"]], axis=1)]
    out["condition_item_id"] = out.apply(lambda row: condition_item_id(row["condition"], row["original_pair_id"]), axis=1)
    out["resp_missing"] = out["resp"].isna()
    out["rt_valid"] = out["rt"].notna() & out["rt"].gt(0) & out["resp"].notna()
    out.loc[~out["rt_valid"], "rt"] = np.nan
    if run == 3:
        out["run3_understand_yes"] = out["resp"].eq(3).astype("object")
        out.loc[out["resp"].isna(), "run3_understand_yes"] = np.nan
        out["run4_like_yes"] = np.nan
    else:
        out["run3_understand_yes"] = np.nan
        out["run4_like_yes"] = out["resp"].eq(3).astype("object")
        out.loc[out["resp"].isna(), "run4_like_yes"] = np.nan
    out = out.merge(
        template_map[["condition", "original_pair_id", "template_pair_id", "role_a_label", "role_b_label"]],
        on=["condition", "original_pair_id"],
        how="left",
    )
    out = out.sort_values(["condition", "original_pair_id"]).reset_index(drop=True)
    qc = {
        "subject": subject,
        "run": run,
        "file": str(path),
        "n_raw_rows": int(len(frame)),
        "n_learning_rows": int(len(out)),
        "n_valid_rt": int(out["rt"].notna().sum()),
        "n_missing_resp": int(out["resp_missing"].sum()),
        "n_conditions": int(out["condition"].nunique()),
        "ok": bool(len(out) >= 60 and out["rt"].notna().mean() >= 0.85),
    }
    return out, qc


def _wide_table(long: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "subject",
        "condition",
        "original_pair_id",
        "template_pair_id",
        "condition_item_id",
        "role_a_label",
        "role_b_label",
    ]
    keep = long.copy()
    keep = zscore_grouped(keep, "rt", ["subject", "run", "condition"], "rt_z_subject_run_condition")
    run3 = keep[keep["run"].eq(3)].copy()
    run4 = keep[keep["run"].eq(4)].copy()
    run3 = run3.rename(
        columns={
            "rt": "run3_rt",
            "rt_z_subject_run_condition": "run3_rt_z_subject_condition",
            "resp": "run3_resp",
            "acc": "run3_acc",
            "resp_missing": "run3_resp_missing",
        }
    )
    run4 = run4.rename(
        columns={
            "rt": "run4_rt",
            "rt_z_subject_run_condition": "run4_rt_z_subject_condition",
            "resp": "run4_resp",
            "acc": "run4_acc",
            "resp_missing": "run4_resp_missing",
        }
    )
    run3_cols = index_cols + [
        "run3_rt",
        "run3_rt_z_subject_condition",
        "run3_resp",
        "run3_acc",
        "run3_resp_missing",
        "run3_understand_yes",
    ]
    run4_cols = index_cols + [
        "run4_rt",
        "run4_rt_z_subject_condition",
        "run4_resp",
        "run4_acc",
        "run4_resp_missing",
        "run4_like_yes",
    ]
    out = run3[run3_cols].merge(run4[run4_cols], on=index_cols, how="outer")
    out["learning_fluency_shift"] = out["run3_rt_z_subject_condition"] - out["run4_rt_z_subject_condition"]
    def _flag(value: object, yes: str, no: str, missing: str) -> str:
        if pd.isna(value):
            return missing
        return yes if bool(value) else no
    out["learning_response_profile"] = (
        out["run3_understand_yes"].map(lambda v: _flag(v, "understand_yes", "understand_no", "understand_missing"))
        + "__"
        + out["run4_like_yes"].map(lambda v: _flag(v, "like_yes", "like_no", "like_missing"))
    )
    return out.sort_values(["subject", "condition", "original_pair_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = _default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--data-events", type=Path, default=base_dir / "data_events")
    parser.add_argument("--stimuli-template", type=Path, default=base_dir / "stimuli_template.csv")
    parser.add_argument("--out-dir", type=Path, default=base_dir / "paper_outputs" / "qc" / "learning_reactivation_mapping")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    template_map = load_template_map(args.stimuli_template)
    rows = []
    qcs = []
    for subject_dir in sorted(args.data_events.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        for run in [3, 4]:
            path = subject_dir / f"{subject}_run-{run}_events.csv"
            if not path.exists():
                qcs.append({"subject": subject, "run": run, "file": str(path), "ok": False, "fail_reason": "missing file"})
                continue
            try:
                run_frame, qc = _read_run_csv(path, subject, run, template_map)
                rows.append(run_frame)
                qc["fail_reason"] = "" if qc["ok"] else "coverage below threshold"
                qcs.append(qc)
            except Exception as exc:
                qcs.append({"subject": subject, "run": run, "file": str(path), "ok": False, "fail_reason": str(exc)})
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    wide = _wide_table(long) if not long.empty else pd.DataFrame()
    qc = pd.DataFrame(qcs)
    write_tsv(long, out_dir / "learning_behavior_long.tsv")
    write_tsv(wide, out_dir / "learning_behavior_item.tsv")
    write_tsv(qc, out_dir / "learning_behavior_qc.tsv")
    summary = {
        "script": str(Path(__file__).resolve()),
        "n_long_rows": int(len(long)),
        "n_item_rows": int(len(wide)),
        "n_subjects": int(wide["subject"].nunique()) if not wide.empty else 0,
        "mean_run3_rt": float(wide["run3_rt"].mean()) if "run3_rt" in wide else None,
        "mean_run4_rt": float(wide["run4_rt"].mean()) if "run4_rt" in wide else None,
        "all_qc_ok": bool(qc["ok"].all()) if "ok" in qc else False,
    }
    (out_dir / "learning_behavior_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
