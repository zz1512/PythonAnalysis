#!/usr/bin/env python3
"""HPC ↔ vmPFC post-encoding connectivity (C2, gated by G2).

理论动机
- System consolidation 文献（Preston & Eichenbaum 2013；Schlichting & Preston
  2015）预测：encoding 结束后的"post-encoding rest"期间，海马与 vmPFC 的
  功能连接上升程度，与后续记忆 / 学习效应相关。
- 这里把"post-encoding rest"定义为每个 learning run 结束后的 BOLD 段
  （由 ``--bold-manifest`` 内的 ``pre_encoding_tr_*`` / ``post_encoding_tr_*`` 窗口指定）。

脚本做什么
- 对每个 subject × (HPC seed × vmPFC target) pair：在 pre-encoding 与
  post-encoding 两个窗口里分别提取 seed / target 平均时间序列，去趋势，
  控制 motion/WM/CSF confounds，计算 Pearson r，Fisher-z 转换，取
  Δ = z_post − z_pre。
- 组水平：对每个 seed × target pair 做 one-sample t（Δ > 0），BH-FDR
  within (seed × target) family。
- 可选：把 Δ 回归到 subject-level 学习效应（post−pre pair similarity）或
  retrieval memory score，作为 brain-behavior bridge。

Gate G2（由主 gPPI 结果触发）：
- 需要 ``$paper/qc/gppi_results_<seed>_<roi_tag>/gppi_group_summary.tsv``
  里至少存在一行 ROI (HPC seed → vmPFC target，任意 learning condition)
  p < 0.1。不满足则脚本仅写 ``gate_status.tsv`` 并退出。

输入（由 CLI 指定，与本机文件布局对齐）
- ``--bold-manifest``：TSV，列 = ``subject, run_label, bold_path, confounds_path,
  pre_encoding_tr_start, pre_encoding_tr_end, post_encoding_tr_start,
  post_encoding_tr_end, tr``；``run_label`` 区分 learning run (run3 / run4)。
- ``--hpc-masks``、``--vmpfc-masks``：NIfTI mask 文件列表，命名用于输出表。

输出：``$paper/qc/hpc_vmpfc_post_encoding_ppi_<roi_tag>/`` 与
``$paper/tables_main/table_hpc_vmpfc_post_encoding_ppi_<roi_tag>.tsv``。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, read_table, save_json, write_table  # noqa: E402
from common.roi_library import sanitize_roi_tag  # noqa: E402


DEFAULT_CONFOUND_COLS = (
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "csf", "white_matter",
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric.astype(float))
    if not valid.any():
        return out
    values = numeric.loc[valid].astype(float)
    order = values.sort_values().index
    ranked = values.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _find_gate_source(paper_root: Path) -> list[Path]:
    return sorted(paper_root.glob("qc/gppi_results_*/gppi_group_summary.tsv"))


def _check_gate_g2(paper_root: Path, p_threshold: float) -> tuple[bool, pd.DataFrame]:
    sources = _find_gate_source(paper_root)
    if not sources:
        return False, pd.DataFrame(
            [
                {
                    "gate": "G2_condition_ppi",
                    "gate_pass": False,
                    "reason": "No gppi_group_summary.tsv found under qc/gppi_results_*.",
                    "p_threshold": p_threshold,
                }
            ]
        )
    frames = []
    for path in sources:
        try:
            frame = read_table(path)
            frame["_source_path"] = str(path)
            frames.append(frame)
        except Exception as exc:
            frames.append(
                pd.DataFrame(
                    {"_load_error": [str(exc)], "_source_path": [str(path)]}
                )
            )
    merged = pd.concat(frames, ignore_index=True, sort=False)
    regex = r"(?i)hippocamp|HPC"
    target_regex = r"(?i)vmPFC|ventromedial|vMPFC|MMPFC"
    mask = (
        merged.get("seed_roi", pd.Series(dtype=str))
        .astype(str)
        .str.contains(regex, na=False, regex=True)
        & merged.get("target_roi", pd.Series(dtype=str))
        .astype(str)
        .str.contains(target_regex, na=False, regex=True)
        & pd.to_numeric(merged.get("p", pd.Series(dtype=float)), errors="coerce").lt(
            p_threshold
        )
    )
    passing = merged[mask]
    report = pd.DataFrame(
        [
            {
                "gate": "G2_condition_ppi",
                "gate_pass": bool(not passing.empty),
                "p_threshold": p_threshold,
                "n_passing_rows": int(len(passing)),
                "source_files_scanned": ";".join(str(p) for p in sources),
                "passing_pairs": ";".join(
                    (
                        passing.get("seed_roi", pd.Series(dtype=str)).astype(str)
                        + "->"
                        + passing.get("target_roi", pd.Series(dtype=str)).astype(str)
                    ).tolist()
                ),
            }
        ]
    )
    return bool(not passing.empty), report


def _load_mean_series(
    bold_path: Path, mask_path: Path
) -> np.ndarray:
    img = nib.load(str(bold_path))
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.get_fdata(), dtype=bool)
    if mask.shape != data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask.shape} != BOLD spatial shape {data.shape[:3]} "
            f"({bold_path} vs {mask_path})"
        )
    voxels = data[mask, :]  # (V, T)
    if voxels.size == 0:
        raise ValueError(f"Empty mask for {mask_path}")
    return np.nanmean(voxels, axis=0)


def _residualize(
    series: np.ndarray, confounds: np.ndarray | None
) -> np.ndarray:
    series = np.asarray(series, dtype=float).reshape(-1)
    t = np.arange(series.size, dtype=float)
    X_list = [np.ones_like(t), t, t**2]
    if confounds is not None and confounds.size > 0:
        conf = np.asarray(confounds, dtype=float)
        if conf.ndim == 1:
            conf = conf.reshape(-1, 1)
        mean_conf = np.nanmean(conf, axis=0, keepdims=True)
        conf = np.where(np.isnan(conf), mean_conf, conf)
        X_list.extend(conf.T)
    X = np.column_stack(X_list)
    finite = np.isfinite(series) & np.isfinite(X).all(axis=1)
    if finite.sum() < X.shape[1] + 2:
        return series - np.nanmean(series)
    beta, *_ = np.linalg.lstsq(X[finite], series[finite], rcond=None)
    residuals = series - X @ beta
    return residuals


def _fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return float("nan")
    r = float(max(min(r, 0.999999), -0.999999))
    return float(np.arctanh(r))


def _window_slice(
    series: np.ndarray, start: int | float, end: int | float
) -> np.ndarray:
    n = series.size
    if pd.isna(start) or pd.isna(end):
        return np.array([], dtype=float)
    start_i = max(0, int(start))
    end_i = min(n, int(end))
    if end_i - start_i < 10:
        return np.array([], dtype=float)
    return series[start_i:end_i]


def _load_confounds(path: Path, columns: tuple[str, ...]) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return None
    return frame[cols].to_numpy(dtype=float)


def _connectivity_delta(
    bold_path: Path,
    seed_mask_path: Path,
    target_mask_path: Path,
    pre_start: int | float,
    pre_end: int | float,
    post_start: int | float,
    post_end: int | float,
    confounds_path: Path | None,
    confound_cols: tuple[str, ...],
) -> dict[str, float]:
    seed_series = _load_mean_series(bold_path, seed_mask_path)
    target_series = _load_mean_series(bold_path, target_mask_path)
    confounds = _load_confounds(confounds_path, confound_cols)
    seed_resid = _residualize(seed_series, confounds)
    target_resid = _residualize(target_series, confounds)

    results: dict[str, float] = {}
    for label, (start, end) in (
        ("pre", (pre_start, pre_end)),
        ("post", (post_start, post_end)),
    ):
        seed_w = _window_slice(seed_resid, start, end)
        target_w = _window_slice(target_resid, start, end)
        if seed_w.size == 0 or target_w.size != seed_w.size:
            results[f"{label}_r"] = float("nan")
            results[f"{label}_n_tr"] = int(seed_w.size)
            continue
        with np.errstate(invalid="ignore"):
            r = float(np.corrcoef(seed_w, target_w)[0, 1])
        results[f"{label}_r"] = r
        results[f"{label}_n_tr"] = int(seed_w.size)
    results["pre_z"] = _fisher_z(results.get("pre_r", float("nan")))
    results["post_z"] = _fisher_z(results.get("post_r", float("nan")))
    results["delta_z"] = (
        results["post_z"] - results["pre_z"]
        if np.isfinite(results["post_z"]) and np.isfinite(results["pre_z"])
        else float("nan")
    )
    return results


def main() -> None:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(
        description="HPC ↔ vmPFC post-encoding connectivity (C2, gated by G2).",
    )
    parser.add_argument(
        "--paper-output-root", type=Path, default=base_dir / "paper_outputs"
    )
    parser.add_argument(
        "--bold-manifest",
        type=Path,
        default=None,
        help=(
            "TSV with columns: subject, run_label, bold_path, confounds_path, "
            "pre_encoding_tr_start, pre_encoding_tr_end, post_encoding_tr_start, "
            "post_encoding_tr_end, tr."
        ),
    )
    parser.add_argument(
        "--hpc-masks",
        type=Path,
        nargs="+",
        default=None,
        help="HPC seed NIfTI masks (e.g. meta_L_hippocampus, meta_R_hippocampus).",
    )
    parser.add_argument(
        "--vmpfc-masks",
        type=Path,
        nargs="+",
        default=None,
        help="vmPFC target NIfTI masks (e.g. tom_VMPFC / tom_MMPFC or manifest-resolved).",
    )
    parser.add_argument(
        "--confound-cols",
        nargs="+",
        default=list(DEFAULT_CONFOUND_COLS),
        help="Columns from confounds TSV to regress out (default: fMRIPrep 6 motion + WM/CSF).",
    )
    parser.add_argument("--gate-p-threshold", type=float, default=0.1)
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="忽略 Gate G2 直接运行（不建议）。",
    )
    args = parser.parse_args()

    combined_tag = sanitize_roi_tag("meta_hpc_vmpfc_ppi")
    out_dir = ensure_dir(
        args.paper_output_root / "qc" / f"hpc_vmpfc_post_encoding_ppi_{combined_tag}"
    )
    tables_main = ensure_dir(args.paper_output_root / "tables_main")

    gate_pass, gate_report = _check_gate_g2(
        args.paper_output_root, args.gate_p_threshold
    )
    write_table(gate_report, out_dir / "gate_status.tsv")

    if not gate_pass and not args.force_run:
        save_json(
            {
                "gate": "G2_condition_ppi",
                "gate_pass": False,
                "message": (
                    "Gate G2 not met (no HPC→vmPFC condition PPI row with p < "
                    f"{args.gate_p_threshold}). Skipped. Pass --force-run to override."
                ),
                "output_dir": str(out_dir),
            },
            out_dir / "manifest.json",
        )
        return

    if args.bold_manifest is None or args.hpc_masks is None or args.vmpfc_masks is None:
        save_json(
            {
                "gate": "G2_condition_ppi",
                "gate_pass": True,
                "message": (
                    "Gate passed but one of --bold-manifest / --hpc-masks / --vmpfc-masks "
                    "is missing; cannot run connectivity. Provide all three and rerun."
                ),
                "output_dir": str(out_dir),
            },
            out_dir / "manifest.json",
        )
        return

    manifest = read_table(args.bold_manifest)
    required_cols = {
        "subject", "run_label", "bold_path", "confounds_path",
        "pre_encoding_tr_start", "pre_encoding_tr_end",
        "post_encoding_tr_start", "post_encoding_tr_end",
    }
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"--bold-manifest missing columns: {sorted(missing)}")

    subject_rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        bold_path = Path(str(row["bold_path"]))
        conf_path = (
            Path(str(row["confounds_path"]))
            if str(row["confounds_path"]) and not pd.isna(row["confounds_path"])
            else None
        )
        if not bold_path.exists():
            subject_rows.append(
                {
                    "subject": row["subject"],
                    "run_label": row["run_label"],
                    "status": "missing_bold",
                    "bold_path": str(bold_path),
                }
            )
            continue
        for hpc_path in args.hpc_masks:
            if not Path(hpc_path).exists():
                continue
            for vmpfc_path in args.vmpfc_masks:
                if not Path(vmpfc_path).exists():
                    continue
                try:
                    metrics = _connectivity_delta(
                        bold_path=bold_path,
                        seed_mask_path=Path(hpc_path),
                        target_mask_path=Path(vmpfc_path),
                        pre_start=row["pre_encoding_tr_start"],
                        pre_end=row["pre_encoding_tr_end"],
                        post_start=row["post_encoding_tr_start"],
                        post_end=row["post_encoding_tr_end"],
                        confounds_path=conf_path,
                        confound_cols=tuple(args.confound_cols),
                    )
                    status = "ok"
                    error = ""
                except Exception as exc:
                    metrics = {
                        "pre_r": float("nan"),
                        "post_r": float("nan"),
                        "pre_z": float("nan"),
                        "post_z": float("nan"),
                        "delta_z": float("nan"),
                        "pre_n_tr": 0,
                        "post_n_tr": 0,
                    }
                    status = "failed"
                    error = str(exc)
                subject_rows.append(
                    {
                        "subject": row["subject"],
                        "run_label": row["run_label"],
                        "seed_roi": Path(hpc_path).stem,
                        "target_roi": Path(vmpfc_path).stem,
                        "seed_mask": str(hpc_path),
                        "target_mask": str(vmpfc_path),
                        "status": status,
                        "error": error,
                        **metrics,
                    }
                )

    subject_frame = pd.DataFrame(subject_rows)
    write_table(subject_frame, out_dir / "hpc_vmpfc_subject_delta.tsv")

    ok_frame = subject_frame[subject_frame["status"] == "ok"]
    group_rows: list[dict[str, object]] = []
    if not ok_frame.empty:
        # Average across runs within subject first.
        per_subject = (
            ok_frame.groupby(
                ["subject", "seed_roi", "target_roi"], dropna=False
            )["delta_z"]
            .mean()
            .reset_index()
        )
        for keys, subset in per_subject.groupby(
            ["seed_roi", "target_roi"], sort=False
        ):
            seed_roi, target_roi = keys
            values = pd.to_numeric(subset["delta_z"], errors="coerce").dropna()
            if values.size < 2:
                continue
            t_val, p_val = stats.ttest_1samp(
                values.to_numpy(dtype=float), 0.0, nan_policy="omit"
            )
            group_rows.append(
                {
                    "seed_roi": seed_roi,
                    "target_roi": target_roi,
                    "n_subjects": int(values.size),
                    "mean_delta_z": float(values.mean()),
                    "t": float(t_val) if np.isfinite(t_val) else float("nan"),
                    "p": float(p_val) if np.isfinite(p_val) else float("nan"),
                }
            )
    group = pd.DataFrame(group_rows)
    if not group.empty:
        group["q_bh_within_family"] = _bh_fdr(group["p"])
        group = group.sort_values(["q_bh_within_family", "p"], na_position="last")
    write_table(group, out_dir / "hpc_vmpfc_group_delta.tsv")
    write_table(
        group,
        tables_main / f"table_hpc_vmpfc_post_encoding_ppi_{combined_tag}.tsv",
    )

    save_json(
        {
            "gate": "G2_condition_ppi",
            "gate_pass": True,
            "p_threshold": args.gate_p_threshold,
            "n_subject_rows": int(len(subject_frame)),
            "n_group_rows": int(len(group)),
            "confound_cols": list(args.confound_cols),
            "output_dir": str(out_dir),
            "force_run": bool(args.force_run),
        },
        out_dir / "manifest.json",
    )


if __name__ == "__main__":
    main()
