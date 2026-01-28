#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_data_alignment.py

论文级数据审计（输入一致性）：
1) events 表存在性与格式识别（BIDS onset/duration 或 E-Prime 宽表）
2) events 基本合法性（负时间、关键列缺失、最大事件时间）
3) fMRI 时间点数（volume / surface L / surface R）
4) confounds 行数与 fMRI timepoints 对齐检查

输出：
- audit_data_alignment.csv：按 task×subject 汇总的审计结果，便于写入论文 QC / Exclusion criteria。

扩展建议（未来）：
- 加入对 confounds 列完整性审计（Friston-24/aCompCor/outliers 等）
- 加入对 TR 的自动读取（从 json sidecar 或 NIfTI header）
- 对 events 单位与扫描时长一致性做强制判定（当前只输出扫描时长与 max_event_time）
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import nibabel as nib


@dataclass(frozen=True)
class TaskSpec:
    """任务审计配置（用 glob 定位 events/fMRI/confounds）。"""
    name: str
    root: Path
    events_glob: str
    fmri_glob_volume: str
    fmri_glob_surf_l: str
    fmri_glob_surf_r: str
    conf_glob: str


def _read_events_table(path: Path) -> pd.DataFrame:
    """读取 events.tsv。默认按 TSV 读取（sep='\\t'）。"""
    return pd.read_csv(path, sep="\t")


def _infer_events_schema(df: pd.DataFrame) -> str:
    """
    粗略识别 events 表结构：
    - bids: 具有 onset/duration
    - eprime_wide: 具有 *.OnsetTime 或 Choice.OnsetTime
    - unknown: 未识别
    """
    cols = set(df.columns.astype(str).tolist())
    if {"onset", "duration"}.issubset(cols):
        return "bids"
    if "Choice.OnsetTime" in cols or any(c.endswith(".OnsetTime") for c in cols):
        return "eprime_wide"
    return "unknown"


def _events_required_columns(schema: str) -> List[str]:
    """返回不同 schema 下，用于基本时间审计的必要列名。"""
    if schema == "bids":
        return ["onset", "duration"]
    if schema == "eprime_wide":
        return ["Choice.OnsetTime", "Choice.RTTime"]
    return []


def _events_max_time_sec(df: pd.DataFrame, schema: str) -> float:
    """
    估算 events 的最大时间点（秒）：
    - bids: max(onset + duration)
    - eprime_wide: max((OnsetTime + RTTime) / 1000)
    """
    if df.empty:
        return float("nan")
    if schema == "bids":
        onset = pd.to_numeric(df["onset"], errors="coerce")
        dur = pd.to_numeric(df["duration"], errors="coerce")
        return float(np.nanmax(onset + dur))
    if schema == "eprime_wide":
        onset_ms = pd.to_numeric(df.get("Choice.OnsetTime"), errors="coerce")
        dur_ms = pd.to_numeric(df.get("Choice.RTTime"), errors="coerce")
        return float(np.nanmax((onset_ms + dur_ms) / 1000.0))
    return float("nan")


def _events_negative_time_count(df: pd.DataFrame, schema: str) -> int:
    """统计负 onset 或负 duration 的行数（若无法识别 schema 则返回 0）。"""
    if df.empty:
        return 0
    if schema == "bids":
        onset = pd.to_numeric(df["onset"], errors="coerce")
        dur = pd.to_numeric(df["duration"], errors="coerce")
        return int(((onset < 0) | (dur < 0)).sum())
    if schema == "eprime_wide":
        onset_ms = pd.to_numeric(df.get("Choice.OnsetTime"), errors="coerce")
        dur_ms = pd.to_numeric(df.get("Choice.RTTime"), errors="coerce")
        return int(((onset_ms < 0) | (dur_ms < 0)).sum())
    return 0


def _events_nonpositive_duration_count(df: pd.DataFrame) -> int:
    if df.empty or "duration" not in df.columns:
        return 0
    dur = pd.to_numeric(df["duration"], errors="coerce")
    return int((dur <= 0).sum())


def _events_nonmonotonic_onset_count(df: pd.DataFrame) -> int:
    if df.empty or "onset" not in df.columns:
        return 0
    onset = pd.to_numeric(df["onset"], errors="coerce")
    onset = onset[np.isfinite(onset)]
    if onset.size < 2:
        return 0
    return int((np.diff(onset.to_numpy()) < 0).sum())


def _events_trial_type_other_frac(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    if "trial_type" not in df.columns:
        return float("nan")
    tt = df["trial_type"].astype(str)
    if tt.size == 0:
        return float("nan")
    return float((tt.str.lower() == "other").mean())


def _load_n_tp_volume(path: Path) -> Optional[int]:
    """读取 4D NIfTI 的 timepoints（shape[3]）。"""
    try:
        img = nib.load(str(path))
        shp = img.shape
        if len(shp) < 4:
            return None
        return int(shp[3])
    except Exception:
        return None


def _load_n_tp_gifti(path: Path) -> Optional[int]:
    """
    读取 Gifti 的 timepoints：
    - 常见的 surface time-series: (n_vertices, n_tp)
    - 若为 1D beta（n_vertices,）则返回 1
    """
    try:
        g = nib.load(str(path))
        if not hasattr(g, "darrays") or len(g.darrays) == 0:
            return None
        data = np.asarray(g.darrays[0].data)
        if data.ndim == 1:
            return 1
        return int(data.shape[1])
    except Exception:
        return None


def _read_confounds_n_rows(path: Path) -> Optional[int]:
    """读取 confounds TSV 的行数（通常应与 fMRI timepoints 相等）。"""
    try:
        df = pd.read_csv(path, sep="\t")
        return int(df.shape[0])
    except Exception:
        return None


def _first_or_none(paths: List[Path]) -> Optional[Path]:
    """若多个候选文件存在，默认取 mtime 最新的一个。"""
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths[-1]


def _find_subjects(root: Path) -> List[str]:
    """扫描 root/sub-* 目录得到被试列表。"""
    return sorted([p.name for p in root.glob("sub-*") if p.is_dir()])


def audit_one_subject(
    sub_dir: Path,
    spec: TaskSpec,
    tr: float,
) -> Dict[str, object]:
    """对单个被试（在某任务目录下）输出一行审计记录。"""
    events_files = list(sub_dir.glob(spec.events_glob))
    events_path = _first_or_none(events_files)

    fmri_vol = _first_or_none(list(sub_dir.glob(spec.fmri_glob_volume)))
    fmri_l = _first_or_none(list(sub_dir.glob(spec.fmri_glob_surf_l)))
    fmri_r = _first_or_none(list(sub_dir.glob(spec.fmri_glob_surf_r)))

    conf_path = _first_or_none(list(sub_dir.glob(spec.conf_glob)))

    schema = "missing"
    missing_cols = ""
    n_rows_events = None
    max_event_time_sec = float("nan")
    neg_event_rows = 0
    nonpos_dur_rows = 0
    nonmono_onset_rows = 0
    other_frac = float("nan")

    if events_path is not None and events_path.exists():
        try:
            ev = _read_events_table(events_path)
            schema = _infer_events_schema(ev)
            n_rows_events = int(ev.shape[0])
            req = _events_required_columns(schema)
            missing = [c for c in req if c not in ev.columns]
            missing_cols = ",".join(missing)
            max_event_time_sec = _events_max_time_sec(ev, schema)
            neg_event_rows = _events_negative_time_count(ev, schema)
            if schema == "bids":
                nonpos_dur_rows = _events_nonpositive_duration_count(ev)
                nonmono_onset_rows = _events_nonmonotonic_onset_count(ev)
                other_frac = _events_trial_type_other_frac(ev)
        except Exception:
            schema = "read_error"

    n_tp_vol = _load_n_tp_volume(fmri_vol) if fmri_vol is not None else None
    n_tp_l = _load_n_tp_gifti(fmri_l) if fmri_l is not None else None
    n_tp_r = _load_n_tp_gifti(fmri_r) if fmri_r is not None else None

    n_rows_conf = _read_confounds_n_rows(conf_path) if conf_path is not None else None

    scan_dur_vol = float(n_tp_vol * tr) if n_tp_vol is not None else float("nan")
    scan_dur_l = float(n_tp_l * tr) if n_tp_l is not None else float("nan")
    scan_dur_r = float(n_tp_r * tr) if n_tp_r is not None else float("nan")

    def _align_flag(n_tp: Optional[int]) -> Optional[bool]:
        """timepoints 与 confounds 行数对齐：None 表示缺失无法判定。"""
        if n_tp is None or n_rows_conf is None:
            return None
        return bool(n_tp == n_rows_conf)

    return {
        "task": spec.name,
        "subject": sub_dir.name,
        "events_path": str(events_path) if events_path is not None else "",
        "events_found": bool(events_path is not None),
        "events_schema": schema,
        "events_rows": n_rows_events if n_rows_events is not None else np.nan,
        "events_missing_required_cols": missing_cols,
        "events_max_time_sec": max_event_time_sec,
        "events_negative_rows": int(neg_event_rows),
        "events_nonpositive_duration_rows": int(nonpos_dur_rows),
        "events_nonmonotonic_onset_steps": int(nonmono_onset_rows),
        "events_other_trial_type_frac": other_frac,
        "fmri_volume_path": str(fmri_vol) if fmri_vol is not None else "",
        "fmri_surf_l_path": str(fmri_l) if fmri_l is not None else "",
        "fmri_surf_r_path": str(fmri_r) if fmri_r is not None else "",
        "n_tp_volume": n_tp_vol if n_tp_vol is not None else np.nan,
        "n_tp_surf_l": n_tp_l if n_tp_l is not None else np.nan,
        "n_tp_surf_r": n_tp_r if n_tp_r is not None else np.nan,
        "confounds_path": str(conf_path) if conf_path is not None else "",
        "confounds_rows": n_rows_conf if n_rows_conf is not None else np.nan,
        "align_volume_confounds": _align_flag(n_tp_vol),
        "align_surf_l_confounds": _align_flag(n_tp_l),
        "align_surf_r_confounds": _align_flag(n_tp_r),
        "scan_dur_volume_sec": scan_dur_vol,
        "scan_dur_surf_l_sec": scan_dur_l,
        "scan_dur_surf_r_sec": scan_dur_r,
    }


def build_default_specs(base_dir: Path) -> List[TaskSpec]:
    emo_root = base_dir / "emo_20250623/v1"
    soc_root = base_dir / "soc_20250623/v1"
    return [
        TaskSpec(
            name="EMO",
            root=emo_root,
            events_glob="beh/func_task_emo/run-1/*task-EMO_events.tsv",
            fmri_glob_volume="func/emo_midprep/*space-MNI152NLin2009cAsym*desc-taskFriston_custom-smooth*.nii.gz",
            fmri_glob_surf_l="func/emo_midprep/*hemi-L*space-fsLR*desc-taskFriston_custom-smooth*.shape.gii",
            fmri_glob_surf_r="func/emo_midprep/*hemi-R*space-fsLR*desc-taskFriston_custom-smooth*.shape.gii",
            conf_glob="func/emo_miniprep/*desc-confounds_timeseries.tsv",
        ),
        TaskSpec(
            name="SOC",
            root=soc_root,
            events_glob="beh/func_task_soc/run-1/*task-SOC_events.tsv",
            fmri_glob_volume="func/soc_midprep/*space-MNI152NLin2009cAsym*desc-taskFriston_custom-smooth*.nii.gz",
            fmri_glob_surf_l="func/soc_midprep/*hemi-L*space-fsLR*desc-taskFriston_custom-smooth*.shape.gii",
            fmri_glob_surf_r="func/soc_midprep/*hemi-R*space-fsLR*desc-taskFriston_custom-smooth*.shape.gii",
            conf_glob="func/soc_miniprep/*desc-confounds_timeseries.tsv",
        ),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="审计 events/fMRI/confounds 对齐与时间覆盖")
    p.add_argument("--bids-dir", type=Path, default=Path("/public/home/dingrui/BIDS_DATA"), help="BIDS_DATA 根目录")
    p.add_argument("--tr", type=float, default=2.0, help="TR（秒）")
    p.add_argument("--out-dir", type=Path, default=Path("./qc_audit_out"), help="输出目录")
    p.add_argument("--limit-subjects", type=int, default=0, help="仅审计前 N 个被试（0=全部）")
    p.add_argument("--event-scan-margin-sec", type=float, default=5.0, help="允许 events_max_time 超出 scan_dur 的容忍秒数")
    p.add_argument("--log-top", type=int, default=10, help="FAIL 时在日志中打印的示例条目数量")
    p.add_argument("--max-other-trial-type-frac", type=float, default=0.5, help="BIDS events 中 Other 的比例超过则判为未通过")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    specs = build_default_specs(args.bids_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for spec in specs:
        if not spec.root.exists():
            continue
        subs = _find_subjects(spec.root)
        if args.limit_subjects and args.limit_subjects > 0:
            subs = subs[: int(args.limit_subjects)]
        for sub in subs:
            sub_dir = spec.root / sub
            rows.append(audit_one_subject(sub_dir, spec, tr=float(args.tr)))

    df = pd.DataFrame(rows)
    out_path = out_dir / "audit_data_alignment.csv"
    df.to_csv(out_path, index=False)

    if not df.empty:
        df2 = df.copy()
        df2["has_confounds"] = df2["confounds_path"].astype(str).str.len() > 0
        df2["has_vol"] = df2["fmri_volume_path"].astype(str).str.len() > 0
        df2["has_l"] = df2["fmri_surf_l_path"].astype(str).str.len() > 0
        df2["has_r"] = df2["fmri_surf_r_path"].astype(str).str.len() > 0

        bad_events = df2[
            (df2["events_found"] == False)
            | (df2["events_schema"].isin(["missing", "unknown", "read_error"]))
            | (df2["events_missing_required_cols"].astype(str) != "")
            | (df2["events_negative_rows"].fillna(0).astype(int) > 0)
            | (df2["events_nonpositive_duration_rows"].fillna(0).astype(int) > 0)
            | (df2["events_nonmonotonic_onset_steps"].fillna(0).astype(int) > 0)
            | (
                np.isfinite(df2["events_other_trial_type_frac"].to_numpy(dtype=float))
                & (df2["events_other_trial_type_frac"].to_numpy(dtype=float) > float(args.max_other_trial_type_frac))
            )
        ]

        def _align_bad(row, key: str, has_key: str) -> bool:
            if not bool(row[has_key]):
                return False
            v = row.get(key, None)
            return v is not True

        bad_align_mask = df2.apply(
            lambda r: (
                _align_bad(r, "align_volume_confounds", "has_vol")
                | _align_bad(r, "align_surf_l_confounds", "has_l")
                | _align_bad(r, "align_surf_r_confounds", "has_r")
            )
            and bool(r["has_confounds"]),
            axis=1,
        )
        bad_align = df2[bad_align_mask]

        scan_dur = np.nanmax(
            np.vstack(
                [
                    df2["scan_dur_volume_sec"].to_numpy(dtype=float),
                    df2["scan_dur_surf_l_sec"].to_numpy(dtype=float),
                    df2["scan_dur_surf_r_sec"].to_numpy(dtype=float),
                ]
            ),
            axis=0,
        )
        df2["scan_dur_any_sec"] = scan_dur
        overrun = df2[
            np.isfinite(df2["events_max_time_sec"].to_numpy(dtype=float))
            & np.isfinite(df2["scan_dur_any_sec"].to_numpy(dtype=float))
            & (df2["events_max_time_sec"].to_numpy(dtype=float) > (df2["scan_dur_any_sec"].to_numpy(dtype=float) + float(args.event_scan_margin_sec)))
        ]

        ok = (len(bad_events) == 0) and (len(bad_align) == 0) and (len(overrun) == 0)
        status = "PASS" if ok else "FAIL"
        print(f"[QC] audit_data_alignment: {status}")
        print(f"[QC] Saved: {out_path}")
        print(f"[QC] Rows={len(df2)} | BadEvents={len(bad_events)} | BadAlign={len(bad_align)} | EventOverrun={len(overrun)}")

        if status == "FAIL":
            top_n = int(args.log_top)
            if top_n <= 0:
                return

            def _print_group(title: str, dff: pd.DataFrame, cols: List[str]) -> None:
                if dff.empty:
                    return
                sample = dff[cols].head(top_n)
                print(f"[QC] {title}: {len(dff)} (showing up to {top_n})")
                print(sample.to_string(index=False))

            cols_events = [
                "task",
                "subject",
                "events_schema",
                "events_missing_required_cols",
                "events_negative_rows",
                "events_nonpositive_duration_rows",
                "events_nonmonotonic_onset_steps",
                "events_other_trial_type_frac",
                "events_max_time_sec",
            ]
            _print_group("BadEvents", bad_events.sort_values(["task", "subject"]), cols_events)

            cols_align = [
                "task",
                "subject",
                "confounds_rows",
                "n_tp_volume",
                "align_volume_confounds",
                "n_tp_surf_l",
                "align_surf_l_confounds",
                "n_tp_surf_r",
                "align_surf_r_confounds",
            ]
            _print_group("BadAlign", bad_align.sort_values(["task", "subject"]), cols_align)

            cols_overrun = [
                "task",
                "subject",
                "events_max_time_sec",
                "scan_dur_any_sec",
                "scan_dur_volume_sec",
                "scan_dur_surf_l_sec",
                "scan_dur_surf_r_sec",
            ]
            _print_group("EventOverrun", overrun.sort_values(["task", "subject"]), cols_overrun)


if __name__ == "__main__":
    main()
