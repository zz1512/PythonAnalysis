#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_emo_stage_outputs.py

用途：
- 对 emo / emo_new 链路中关键 .npy/.npz 结果做阶段化质检。
- 重点检查：文件存在性、维度一致性、矩阵基本数学属性、跨阶段元数据一致性。
- 提供“多维度抽检”：随机抽取 ROI 与被试对，检查取值范围/对称性/对角线。

默认检查对象：
1) emo（老链路，单目录）
   - roi_beta_matrix_232.npy + 对应 *_rois.csv / *_subjects.csv / *_stimuli.csv
   - roi_isc_spearman_by_age.npy + 对应 *_subjects_sorted.csv / *_rois.csv
2) emo_new（按条件目录）
   - by_stimulus/<stim>/roi_beta_matrix_200.npz + 对应 *_rois.csv / *_subjects.csv
   - by_stimulus/<stim>/roi_isc_spearman_by_age.npy + 对应 *_subjects_sorted.csv / *_rois.csv

输出：
- 控制台打印 PASS/FAIL 汇总
- out_dir/check_emo_stage_outputs.csv（逐项检查结果）
- out_dir/check_emo_stage_outputs_summary.json（总览）
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CheckRow:
    stage: str
    scope: str
    item: str
    ok: bool
    detail: str


def _read_list_csv(path: Path, preferred_cols: Sequence[str]) -> List[str]:
    df = pd.read_csv(path)
    for c in preferred_cols:
        if c in df.columns:
            return df[c].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def _safe_ages(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "age" not in df.columns:
        return np.array([], dtype=np.float64)
    return pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=np.float64)


def _is_non_decreasing(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return True
    return bool(np.all(np.diff(x) >= -1e-12))


def _check_corr_cube_properties(
    cube: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    diag_tol: float,
    sym_tol: float,
) -> Tuple[bool, str]:
    if cube.ndim != 3:
        return False, f"期望 3D 相关立方体，实际 ndim={cube.ndim}"

    n_roi, n_sub, n_sub2 = cube.shape
    if n_sub != n_sub2:
        return False, f"相关矩阵不是方阵: shape={cube.shape}"

    sample_roi = min(sample_size, n_roi)
    roi_idx = rng.choice(n_roi, size=sample_roi, replace=False)

    max_sym = 0.0
    max_diag_dev = 0.0
    min_val = np.inf
    max_val = -np.inf

    for ri in roi_idx:
        m = np.asarray(cube[ri], dtype=np.float64)
        sym = np.nanmax(np.abs(m - m.T))
        diag_dev = np.nanmax(np.abs(np.diag(m) - 1.0))
        vmax = np.nanmax(m)
        vmin = np.nanmin(m)

        max_sym = max(max_sym, float(sym))
        max_diag_dev = max(max_diag_dev, float(diag_dev))
        max_val = max(max_val, float(vmax))
        min_val = min(min_val, float(vmin))

    ok = (max_sym <= sym_tol) and (max_diag_dev <= diag_tol) and (min_val >= -1.0001) and (max_val <= 1.0001)
    detail = (
        f"抽检ROI={sample_roi}/{n_roi}, max|A-A^T|={max_sym:.3e}, "
        f"max|diag-1|={max_diag_dev:.3e}, value_range=[{min_val:.4f},{max_val:.4f}]"
    )
    return ok, detail


def _summarize_npz_feature_dims(npz: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Tuple[int, int, float]:
    widths = []
    for k in keys:
        arr = np.asarray(npz[k])
        if arr.ndim != 2:
            continue
        widths.append(int(arr.shape[1]))
    if not widths:
        return 0, 0, 0.0
    return int(min(widths)), int(max(widths)), float(np.median(np.asarray(widths, dtype=np.float64)))


def check_emo(root: Path, rows: List[CheckRow], sample_size: int, rng: np.random.Generator, diag_tol: float, sym_tol: float) -> None:
    scope = "emo"

    beta = root / "roi_beta_matrix_232.npy"
    rois_csv = root / "roi_beta_matrix_232_rois.csv"
    subs_csv = root / "roi_beta_matrix_232_subjects.csv"
    stims_csv = root / "roi_beta_matrix_232_stimuli.csv"

    required = [beta, rois_csv, subs_csv, stims_csv]
    miss = [str(p) for p in required if not p.exists()]
    rows.append(CheckRow("emo_step1_beta", scope, "required_files", len(miss) == 0, "missing=" + (", ".join(miss) if miss else "None")))
    if miss:
        return

    mat = np.load(beta)
    rois = _read_list_csv(rois_csv, ["roi"])
    subs = _read_list_csv(subs_csv, ["subject", "sub_id"])
    stims = _read_list_csv(stims_csv, ["stimulus_content", "stimulus_type", "stimulus"])

    ok_shape = (mat.ndim == 3)
    rows.append(CheckRow("emo_step1_beta", scope, "beta_ndim", ok_shape, f"shape={tuple(mat.shape)}"))
    if not ok_shape:
        return

    n_roi, n_sub, n_stim = mat.shape
    rows.append(CheckRow("emo_step1_beta", scope, "shape_vs_csv", (n_roi == len(rois)) and (n_sub == len(subs)) and (n_stim == len(stims)), f"shape=({n_roi},{n_sub},{n_stim}), csv=({len(rois)},{len(subs)},{len(stims)})"))
    rows.append(CheckRow("emo_step1_beta", scope, "expected_roi_count_232", n_roi == 232, f"n_roi={n_roi}"))

    finite_ratio = float(np.isfinite(mat).mean())
    rows.append(CheckRow("emo_step1_beta", scope, "finite_ratio", finite_ratio >= 0.999, f"finite_ratio={finite_ratio:.6f}"))

    isc = root / "roi_isc_spearman_by_age.npy"
    isc_rois = root / "roi_isc_spearman_by_age_rois.csv"
    isc_subs = root / "roi_isc_spearman_by_age_subjects_sorted.csv"
    required2 = [isc, isc_rois, isc_subs]
    miss2 = [str(p) for p in required2 if not p.exists()]
    rows.append(CheckRow("emo_step2_isc", scope, "required_files", len(miss2) == 0, "missing=" + (", ".join(miss2) if miss2 else "None")))
    if miss2:
        return

    cube = np.load(isc)
    rois2 = _read_list_csv(isc_rois, ["roi"])
    subs2 = _read_list_csv(isc_subs, ["subject", "sub_id"])

    ok3d = cube.ndim == 3
    rows.append(CheckRow("emo_step2_isc", scope, "isc_ndim", ok3d, f"shape={tuple(cube.shape)}"))
    if not ok3d:
        return

    r2, s2, s3 = cube.shape
    rows.append(CheckRow("emo_step2_isc", scope, "shape_vs_csv", (r2 == len(rois2)) and (s2 == len(subs2)) and (s2 == s3), f"shape=({r2},{s2},{s3}), csv=({len(rois2)},{len(subs2)})"))
    rows.append(CheckRow("emo_step2_isc", scope, "roi_count_consistent_with_step1", r2 == n_roi, f"step1={n_roi}, step2={r2}"))

    ok_corr, detail_corr = _check_corr_cube_properties(cube, sample_size=sample_size, rng=rng, diag_tol=diag_tol, sym_tol=sym_tol)
    rows.append(CheckRow("emo_step2_isc", scope, "corr_matrix_property_sampling", ok_corr, detail_corr))

    ages = _safe_ages(isc_subs)
    rows.append(CheckRow("emo_step2_isc", scope, "age_non_decreasing", _is_non_decreasing(ages), f"n_age={ages.size}"))


def check_emo_new(root: Path, rows: List[CheckRow], sample_size: int, rng: np.random.Generator, diag_tol: float, sym_tol: float, stimuli: Optional[List[str]]) -> None:
    base = root / "by_stimulus"
    if not base.exists():
        rows.append(CheckRow("emo_new", "emo_new", "by_stimulus_dir", False, f"目录不存在: {base}"))
        return

    stim_dirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if stimuli:
        sel = set(stimuli)
        stim_dirs = [p for p in stim_dirs if p.name in sel]

    rows.append(CheckRow("emo_new", "emo_new", "stimulus_dir_count", len(stim_dirs) > 0, f"n={len(stim_dirs)}"))
    for stim_dir in stim_dirs:
        stim = stim_dir.name
        scope = f"emo_new/{stim}"

        beta_npz = stim_dir / "roi_beta_matrix_200.npz"
        beta_rois = stim_dir / "roi_beta_matrix_200_rois.csv"
        beta_subs = stim_dir / "roi_beta_matrix_200_subjects.csv"
        req1 = [beta_npz, beta_rois, beta_subs]
        miss1 = [str(p) for p in req1 if not p.exists()]
        rows.append(CheckRow("emo_new_step1_beta", scope, "required_files", len(miss1) == 0, "missing=" + (", ".join(miss1) if miss1 else "None")))
        if miss1:
            continue

        rois = _read_list_csv(beta_rois, ["roi"])
        subs = _read_list_csv(beta_subs, ["subject", "sub_id"])
        with np.load(beta_npz) as z:
            keys = list(z.files)
            rows.append(CheckRow("emo_new_step1_beta", scope, "roi_keys_match_csv", set(keys) == set(rois), f"npz_keys={len(keys)}, csv_rois={len(rois)}"))
            rows.append(CheckRow("emo_new_step1_beta", scope, "expected_roi_count_200", len(keys) == 200, f"n_roi={len(keys)}"))

            bad_shapes = []
            finite_ratios = []
            for k in rois:
                if k not in z.files:
                    continue
                arr = np.asarray(z[k])
                if arr.ndim != 2 or arr.shape[0] != len(subs):
                    bad_shapes.append((k, tuple(arr.shape)))
                    continue
                finite_ratios.append(float(np.isfinite(arr).mean()))

            ok_shapes = len(bad_shapes) == 0
            rows.append(CheckRow("emo_new_step1_beta", scope, "each_roi_shape_2d_and_subject_dim", ok_shapes, f"bad_shape_count={len(bad_shapes)}" + (f", example={bad_shapes[0]}" if bad_shapes else "")))

            min_w, max_w, med_w = _summarize_npz_feature_dims(z, rois)
            rows.append(CheckRow("emo_new_step1_beta", scope, "roi_feature_width_summary", max_w > 0, f"min={min_w}, median={med_w:.1f}, max={max_w}"))

            fr = float(np.mean(finite_ratios)) if finite_ratios else 0.0
            rows.append(CheckRow("emo_new_step1_beta", scope, "mean_finite_ratio_over_rois", fr >= 0.999, f"mean_finite_ratio={fr:.6f}"))

        isc = stim_dir / "roi_isc_spearman_by_age.npy"
        isc_rois = stim_dir / "roi_isc_spearman_by_age_rois.csv"
        isc_subs = stim_dir / "roi_isc_spearman_by_age_subjects_sorted.csv"
        req2 = [isc, isc_rois, isc_subs]
        miss2 = [str(p) for p in req2 if not p.exists()]
        rows.append(CheckRow("emo_new_step2_isc", scope, "required_files", len(miss2) == 0, "missing=" + (", ".join(miss2) if miss2 else "None")))
        if miss2:
            continue

        cube = np.load(isc)
        rois2 = _read_list_csv(isc_rois, ["roi"])
        subs2 = _read_list_csv(isc_subs, ["subject", "sub_id"])

        ok3d = cube.ndim == 3
        rows.append(CheckRow("emo_new_step2_isc", scope, "isc_ndim", ok3d, f"shape={tuple(cube.shape)}"))
        if not ok3d:
            continue

        r2, s2, s3 = cube.shape
        rows.append(CheckRow("emo_new_step2_isc", scope, "shape_vs_csv", (r2 == len(rois2)) and (s2 == len(subs2)) and (s2 == s3), f"shape=({r2},{s2},{s3}), csv=({len(rois2)},{len(subs2)})"))
        rows.append(CheckRow("emo_new_step2_isc", scope, "roi_count_consistent_with_step1", len(rois) == r2, f"step1={len(rois)}, step2={r2}"))
        rows.append(CheckRow("emo_new_step2_isc", scope, "subject_set_consistent_with_step1", set(subs) == set(subs2), f"step1_n={len(subs)}, step2_n={len(subs2)}"))

        ok_corr, detail_corr = _check_corr_cube_properties(cube, sample_size=sample_size, rng=rng, diag_tol=diag_tol, sym_tol=sym_tol)
        rows.append(CheckRow("emo_new_step2_isc", scope, "corr_matrix_property_sampling", ok_corr, detail_corr))

        ages = _safe_ages(isc_subs)
        rows.append(CheckRow("emo_new_step2_isc", scope, "age_non_decreasing", _is_non_decreasing(ages), f"n_age={ages.size}"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="质检 emo / emo_new 阶段输出（.npy/.npz）")
    p.add_argument("--roi-results-dir", type=Path, default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results"), help="结果根目录")
    p.add_argument("--skip-emo", action="store_true", help="跳过 emo 链路结果检查")
    p.add_argument("--skip-emo-new", action="store_true", help="跳过 emo_new 链路结果检查")
    p.add_argument("--stimuli", type=str, default="", help="仅检查指定条件，逗号分隔（emo_new）")
    p.add_argument("--sample-size", type=int, default=8, help="每个 ISC 立方体抽检 ROI 数")
    p.add_argument("--seed", type=int, default=42, help="随机抽检种子")
    p.add_argument("--diag-tol", type=float, default=1e-5, help="对角线接近 1 的容忍误差")
    p.add_argument("--sym-tol", type=float, default=1e-5, help="对称性容忍误差")
    p.add_argument("--out-dir", type=Path, default=Path("./qc_audit_out"), help="输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli = [x.strip() for x in str(args.stimuli).split(",") if x.strip()]
    rng = np.random.default_rng(int(args.seed))

    rows: List[CheckRow] = []

    if not bool(args.skip_emo):
        check_emo(
            root=Path(args.roi_results_dir),
            rows=rows,
            sample_size=int(args.sample_size),
            rng=rng,
            diag_tol=float(args.diag_tol),
            sym_tol=float(args.sym_tol),
        )

    if not bool(args.skip_emo_new):
        check_emo_new(
            root=Path(args.roi_results_dir),
            rows=rows,
            sample_size=int(args.sample_size),
            rng=rng,
            diag_tol=float(args.diag_tol),
            sym_tol=float(args.sym_tol),
            stimuli=stimuli if stimuli else None,
        )

    if not rows:
        raise RuntimeError("未产生任何质检记录，请检查参数。")

    df = pd.DataFrame([r.__dict__ for r in rows])
    out_csv = out_dir / "check_emo_stage_outputs.csv"
    df.to_csv(out_csv, index=False)

    n_total = int(len(df))
    n_fail = int((~df["ok"]).sum())
    n_pass = int(df["ok"].sum())
    status = "PASS" if n_fail == 0 else "FAIL"

    summary = {
        "status": status,
        "n_total_checks": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "roi_results_dir": str(Path(args.roi_results_dir)),
        "stimuli_filter": stimuli,
        "sample_size": int(args.sample_size),
        "diag_tol": float(args.diag_tol),
        "sym_tol": float(args.sym_tol),
        "output_csv": str(out_csv),
    }
    out_json = out_dir / "check_emo_stage_outputs_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[QC] check_emo_stage_outputs: {status}")
    print(f"[QC] 总检查数={n_total} | 通过={n_pass} | 失败={n_fail}")
    print(f"[QC] 明细: {out_csv}")
    print(f"[QC] 汇总: {out_json}")

    if n_fail > 0:
        top = df[~df["ok"]].head(20)
        print("[QC] 失败示例（最多20条）:")
        print(top[["stage", "scope", "item", "detail"]].to_string(index=False))


if __name__ == "__main__":
    main()
