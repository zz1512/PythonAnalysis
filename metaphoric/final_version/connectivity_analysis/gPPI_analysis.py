"""
gPPI_analysis.py

用途
- ROI-to-ROI 的 gPPI（广义心理生理交互）简化实现：检验学习阶段 yy vs kj 条件是否调制
  seed ROI 与 target ROI 的功能连接强度。

当前实现
- 支持 generalized PPI（gPPI）多条件建模：为每个条件单独构造 psych 与 ppi 项。
- 支持可选 HRF deconvolution（ridge）以在“神经层面”构造交互项（更接近 SPM 路线）。
- 对每个 target ROI 做 OLS：
    target_ts ~ seed_phys + sum(psych_cond) + sum(ppi_cond) + confounds
  关注每个条件的 ppi beta，并可对比 `ppi_yy - ppi_kj`（二阶检验的常用对比）。

输入
- seed_mask: 种子 ROI mask（NIfTI）
- target_roi_dir: target ROI masks 目录（NIfTI）
- BOLD / events / confounds 路径从 `glm_config.py` 读取（支持 `PYTHON_METAPHOR_ROOT` 覆盖）

输出（output_dir）
- `gppi_subject_metrics.tsv`：subject×run×target 的 gPPI beta
- `gppi_group_summary.tsv`：按 ROI 聚合后的组水平 one-sample t
- `gppi_meta.json`
"""

from __future__ import annotations



import argparse
import os
from pathlib import Path
import sys
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from nilearn.maskers import NiftiMasker
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

from common.final_utils import ensure_dir, save_json, write_table  # noqa: E402
from common.final_utils import paired_t_summary  # noqa: E402
from common.roi_library import current_roi_set, default_roi_tagged_out_dir, sanitize_roi_tag  # noqa: E402
from glm_config import config  # noqa: E402
import glm_utils  # noqa: E402


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=1)
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(values)
    return (values - mean) / std


def _trial_type_lower(events: pd.DataFrame) -> pd.Series:
    raw = events["trial_type"].astype(str).str.strip().str.lower()
    aliases = {
        # metaphor / yy
        "metaphor": "yy",
        "yy": "yy",
        "yyw": "yy",
        "yyew": "yy",
        # spatial / kj
        "spatial": "kj",
        "kj": "kj",
        "kjw": "kj",
        "kjew": "kj",
        # baseline (optional)
        "baseline": "baseline",
        "base": "baseline",
        "bl": "baseline",
        "jx": "baseline",
        "nonlink": "baseline",
        "no_link": "baseline",
        "unlinked": "baseline",
    }
    return raw.map(lambda x: aliases.get(x, x))


def _build_neural_boxcar(events: pd.DataFrame, frame_times: np.ndarray, condition: str) -> np.ndarray:
    """
    Build a neural-level psychological boxcar regressor at TR resolution.

    - For each event matching `condition`, set frames within [onset, onset+duration) to 1.0.
    - If duration is missing/zero, treat as impulse spanning one TR.
    """
    cond = str(condition).strip().lower()
    work = events.loc[_trial_type_lower(events) == cond].copy()
    out = np.zeros(frame_times.size, dtype=float)
    if work.empty:
        return out
    for row in work.itertuples(index=False):
        onset = float(getattr(row, "onset"))
        duration = float(getattr(row, "duration", 0.0) or 0.0)
        start = int(np.searchsorted(frame_times, onset, side="left"))
        end_time = onset + (duration if duration > 0 else (frame_times[1] - frame_times[0]))
        end = int(np.searchsorted(frame_times, end_time, side="left"))
        end = max(end, start + 1)
        out[start:end] = 1.0
    return out


def _convolve_hrf(neural: np.ndarray, *, tr: float) -> np.ndarray:
    h = np.asarray(spm_hrf(tr), dtype=float).ravel()
    y = np.convolve(np.asarray(neural, dtype=float).ravel(), h, mode="full")[: len(neural)]
    return y


def _deconvolve_ridge(y: np.ndarray, *, tr: float, l2: float) -> np.ndarray:
    """
    Ridge deconvolution under a simple LTI HRF model:
      y ≈ x (*) h
    Solve for x with (X^T X + l2 I)x = X^T y where X is the convolution matrix of h.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    h = np.asarray(spm_hrf(tr), dtype=float).ravel()
    k = min(h.size, n)
    h = h[:k]
    # Build a Toeplitz-like convolution matrix: X[i, j] = h[i-j] if i>=j else 0
    X = np.zeros((n, n), dtype=float)
    for i in range(n):
        j0 = max(0, i - k + 1)
        seg = h[: i - j0 + 1][::-1]
        X[i, j0:i + 1] = seg
    XtX = X.T @ X
    XtX.flat[:: n + 1] += float(l2)
    Xty = X.T @ y
    try:
        x = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    return np.asarray(x, dtype=float)


def apply_sample_mask(values, sample_mask):
    if sample_mask is None:
        return values
    keep = np.asarray(sample_mask, dtype=int)
    if isinstance(values, pd.DataFrame):
        return values.iloc[keep].reset_index(drop=True)
    array = np.asarray(values)
    return array[keep]


def _resolve_roi_manifest() -> Path:
    """
    Keep ROI sourcing consistent with model_rdm_comparison.py:
    prefer rsa_config.ROI_MANIFEST under PYTHON_METAPHOR_ROOT.
    """
    try:
        from rsa_analysis.rsa_config import ROI_MANIFEST  # noqa: E402

        return Path(ROI_MANIFEST)
    except Exception:
        base_dir = Path(os.environ.get("PYTHON_METAPHOR_ROOT", str(getattr(config, "BASE_DIR", "E:/python_metaphor"))))
        return base_dir / "roi_library" / "manifest.tsv"


def _load_roi_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"roi_name", "roi_set", "mask_path"}
    if not required.issubset(frame.columns):
        raise ValueError(f"ROI manifest missing columns {required - set(frame.columns)}: {path}")
    return frame


def _resolve_seeds_from_manifest(seed_roi_names: list[str]) -> list[tuple[str, Path]]:
    manifest_path = _resolve_roi_manifest()
    if not manifest_path.exists():
        raise FileNotFoundError(f"ROI manifest not found: {manifest_path}")
    manifest = _load_roi_manifest(manifest_path)
    wanted = {str(name).strip() for name in seed_roi_names if str(name).strip()}
    subset = manifest[manifest["roi_name"].astype(str).isin(wanted)].copy()
    if subset.empty:
        raise ValueError(f"No seed ROIs found in manifest for: {sorted(wanted)}")
    rows = []
    for row in subset.itertuples(index=False):
        roi_name = str(getattr(row, "roi_name"))
        mask_path = Path(str(getattr(row, "mask_path")))
        rows.append((roi_name, mask_path))
    return rows


def _resolve_targets_from_manifest(target_roi_sets: list[str]) -> list[tuple[str, str, Path]]:
    manifest_path = _resolve_roi_manifest()
    if not manifest_path.exists():
        raise FileNotFoundError(f"ROI manifest not found: {manifest_path}")
    manifest = _load_roi_manifest(manifest_path)
    wanted_sets = {str(item).strip().lower() for item in target_roi_sets if str(item).strip()}
    subset = manifest[manifest["roi_set"].astype(str).str.strip().str.lower().isin(wanted_sets)].copy()
    if subset.empty:
        raise ValueError(f"No target ROIs found in manifest for roi_set in: {sorted(wanted_sets)}")
    # If include flags exist, prefer include_in_rsa to keep selection consistent with RSA ROI pools.
    if "include_in_rsa" in subset.columns:
        subset = subset[subset["include_in_rsa"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})]
    rows = []
    for row in subset.itertuples(index=False):
        roi_name = str(getattr(row, "roi_name"))
        roi_set = str(getattr(row, "roi_set"))
        mask_path = Path(str(getattr(row, "mask_path")))
        rows.append((roi_name, roi_set, mask_path))
    return rows


def _target_scope(target_roi_set: str) -> str:
    norm = str(target_roi_set).strip().lower()
    if norm == "literature":
        return "within_literature"
    if norm == "literature_spatial":
        return "within_literature_spatial"
    return f"within_{norm or 'unknown'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-to-ROI gPPI (seed x psych) during learning runs (run3-4).")
    parser.add_argument("seed_mask", type=Path, nargs="?", default=None,
                        help="Optional seed ROI mask NIfTI. If omitted, use --seed-roi-names with roi_library manifest.")
    parser.add_argument("target_roi_dir", type=Path, nargs="?", default=None,
                        help="Optional target ROI directory (NIfTI masks). If omitted, use --target-roi-sets with roi_library manifest.")
    # output_dir 为可选；默认 `${BASE_DIR}/paper_outputs/qc/gppi_results_<seed_stem>_<ROI_SET>`，
    # 可通过环境变量 METAPHOR_GPPI_OUT_DIR 强制覆盖。
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--paper-output-root", type=Path, default=None,
                        help="Unified output root. Default: {BASE_DIR}/paper_outputs.")
    parser.add_argument("--roi-set", default=None,
                        help="Override ROI set tag in output folder naming (default: METAPHOR_ROI_SET).")
    parser.add_argument("--seed-roi-names", nargs="*", default=None,
                        help="Seed ROI names from roi_library/manifest.tsv (recommended for reproducibility). "
                             "Example: func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort")
    parser.add_argument("--target-roi-sets", nargs="*", default=None,
                        help="One or more target ROI sets from roi_library/manifest.tsv. "
                             "Example: literature literature_spatial")
    parser.add_argument("--stage", choices=["prepost", "learning"], default="prepost",
                        help="Which stage to run gPPI on. "
                             "prepost: run1/2 vs run5/6 (recommended for your main story); "
                             "learning: run3/4.")
    parser.add_argument("--conditions", nargs="*", default=["yy", "kj"],
                        help="Conditions to include in the gPPI model (default: yy kj).")
    parser.add_argument("--deconvolution", choices=["none", "ridge"], default="ridge",
                        help="How to construct the interaction term. "
                             "'ridge' approximates SPM-style deconvolution at TR resolution; "
                             "'none' uses observed seed time series directly (faster, less standard).")
    parser.add_argument("--deconv-l2", type=float, default=10.0,
                        help="Ridge penalty for deconvolution (only for --deconvolution ridge).")
    parser.add_argument("--subjects", nargs="*", default=None, help="Default: glm_config.SUBJECTS")
    parser.add_argument("--runs", nargs="*", type=int, default=None,
                        help="Optional explicit run list. If omitted: prepost uses 1 2 5 6; learning uses 3 4.")
    args = parser.parse_args()

    base_dir = Path(os.environ.get("PYTHON_METAPHOR_ROOT", str(getattr(config, "BASE_DIR", "E:/python_metaphor"))))
    paper_root = args.paper_output_root or (base_dir / "paper_outputs")
    # Output tag: default to METAPHOR_ROI_SET, but if user passes target ROI sets and no explicit roi-set,
    # use a stable combined tag to avoid confusion/collisions.
    combined_target_tag = None
    if args.target_roi_sets and not args.roi_set:
        combined_target_tag = "plus_".join(sorted({sanitize_roi_tag(x) for x in args.target_roi_sets if str(x).strip()}))
    roi_set = (args.roi_set or combined_target_tag or current_roi_set()).strip() or "main_functional"
    roi_tag = sanitize_roi_tag(roi_set)
    tables_si = ensure_dir(paper_root / "tables_si")

    subjects = args.subjects or getattr(config, "SUBJECTS", [f"sub-{i:02d}" for i in range(1, 29)])
    if args.runs is None:
        args.runs = [1, 2, 5, 6] if args.stage == "prepost" else [3, 4]

    # Resolve seeds
    seed_specs: list[tuple[str, Path]] = []
    if args.seed_roi_names:
        seed_specs = _resolve_seeds_from_manifest(list(args.seed_roi_names))
    elif args.seed_mask is not None:
        seed_specs = [(args.seed_mask.stem.replace(".nii", "") or "seed", Path(args.seed_mask))]
    else:
        raise ValueError("Missing seed specification: provide seed_mask or --seed-roi-names.")

    # Resolve targets
    target_specs: list[tuple[str, str, Path]] = []
    if args.target_roi_sets:
        target_specs = _resolve_targets_from_manifest(list(args.target_roi_sets))
    elif args.target_roi_dir is not None:
        roi_paths = sorted(Path(args.target_roi_dir).glob("*.nii*"))
        if not roi_paths:
            raise ValueError(f"No ROI masks in {args.target_roi_dir}")
        target_specs = [(p.stem.replace(".nii", ""), "unknown", p) for p in roi_paths]
    else:
        raise ValueError("Missing target specification: provide target_roi_dir or --target-roi-sets.")

    # Conditions
    conditions = [str(c).strip().lower() for c in (args.conditions or []) if str(c).strip()]
    conditions = [c for c in conditions if c in {"yy", "kj", "baseline"}] or ["yy", "kj"]

    # Run each seed separately to keep outputs and interpretations clean.
    all_rows: list[dict[str, Any]] = []
    for seed_roi_name, seed_mask_path in seed_specs:
        seed_tag = sanitize_roi_tag(seed_roi_name)
        if args.output_dir is not None:
            explicit_root = Path(args.output_dir)
            output_dir = ensure_dir(explicit_root / seed_tag) if len(seed_specs) > 1 else ensure_dir(explicit_root)
        else:
            output_dir = ensure_dir(
                default_roi_tagged_out_dir(
                    paper_root / "qc",
                    f"gppi_results_{seed_tag}",
                    override_env="METAPHOR_GPPI_OUT_DIR",
                    roi_set=roi_set,
                )
            )

        rows: list[dict[str, Any]] = []
        for sub in subjects:
            for run in args.runs:
                fmri_path = Path(str(config.FMRI_TPL).format(sub=sub, run=run))
                if not fmri_path.exists():
                    fmri_path = fmri_path.with_suffix(".nii.gz")
                events_path = Path(str(config.EVENT_TPL).format(sub=sub, run=run))
                confounds_path = Path(str(config.CONFOUNDS_TPL).format(sub=sub, run=run))
                if not fmri_path.exists() or not events_path.exists() or not confounds_path.exists():
                    continue

                events = pd.read_csv(events_path, sep="\t")
                if not {"onset", "duration", "trial_type"}.issubset(events.columns):
                    continue

                confounds, sample_mask = glm_utils.robust_load_confounds(str(confounds_path), config.DENOISE_STRATEGY)
                tr = float(getattr(config, "TR", 2.0))
                full_n_scans = int(nib.load(str(fmri_path)).shape[3])

                seed_masker = NiftiMasker(mask_img=str(seed_mask_path), standardize=True, t_r=tr)
                seed_ts = seed_masker.fit_transform(str(fmri_path), confounds=confounds, sample_mask=sample_mask).ravel()
                frame_times_full = np.arange(full_n_scans, dtype=float) * tr

                confounds_kept = apply_sample_mask(confounds, sample_mask)
                seed_phys = zscore(seed_ts)
                n_scans = seed_phys.size

                if args.deconvolution == "ridge":
                    seed_neural = _deconvolve_ridge(seed_phys, tr=tr, l2=float(args.deconv_l2))
                else:
                    seed_neural = seed_phys

                psych_regs = []
                ppi_regs = []
                psych_names = []
                ppi_names = []
                for cond in conditions:
                    psych_neural_full = _build_neural_boxcar(events, frame_times_full, cond)
                    psych_neural = np.asarray(apply_sample_mask(psych_neural_full, sample_mask), dtype=float).ravel()
                    if psych_neural.size != n_scans:
                        continue
                    psych_conv = zscore(_convolve_hrf(psych_neural, tr=tr))
                    ppi_neural = seed_neural * zscore(psych_neural)
                    ppi_conv = zscore(_convolve_hrf(ppi_neural, tr=tr))
                    psych_regs.append(psych_conv)
                    ppi_regs.append(ppi_conv)
                    psych_names.append(f"psych_{cond}")
                    ppi_names.append(f"ppi_{cond}")

                base_cols = [np.ones(n_scans), seed_phys]
                base_names = ["intercept", "seed_phys"]
                if psych_regs:
                    base_cols.extend(psych_regs)
                    base_names.extend(psych_names)
                if ppi_regs:
                    base_cols.extend(ppi_regs)
                    base_names.extend(ppi_names)
                base = np.column_stack(base_cols)
                if confounds_kept is not None and len(confounds_kept) == n_scans:
                    design = np.column_stack([base, np.asarray(confounds_kept, dtype=float)])
                else:
                    design = base

                for target_roi_name, target_roi_set, target_mask_path in target_specs:
                    target_masker = NiftiMasker(mask_img=str(target_mask_path), standardize=True, t_r=tr)
                    y = target_masker.fit_transform(str(fmri_path), confounds=confounds, sample_mask=sample_mask).ravel()
                    if y.size != n_scans:
                        continue

                    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
                    coef_map = {name: float(val) for name, val in zip(base_names, coef[: len(base_names)])}

                    if args.stage == "prepost":
                        phase = "pre" if int(run) in {1, 2} else ("post" if int(run) in {5, 6} else "other")
                    else:
                        phase = "learning"

                    for cond in conditions:
                        term = f"ppi_{cond}"
                        if term not in coef_map:
                            continue
                        rows.append(
                            {
                                "seed_roi": seed_roi_name,
                                "subject": sub,
                                "run": int(run),
                                "phase": phase,
                                "target_roi": target_roi_name,
                                "target_roi_set": target_roi_set,
                                "target_scope": _target_scope(target_roi_set),
                                "ppi_condition": cond,
                                "gppi_beta": float(coef_map[term]),
                                "deconvolution": args.deconvolution,
                            }
                        )
                    if "ppi_yy" in coef_map and "ppi_kj" in coef_map:
                        rows.append(
                            {
                                "seed_roi": seed_roi_name,
                                "subject": sub,
                                "run": int(run),
                                "phase": phase,
                                "target_roi": target_roi_name,
                                "target_roi_set": target_roi_set,
                                "target_scope": _target_scope(target_roi_set),
                                "ppi_condition": "yy_minus_kj",
                                "gppi_beta": float(coef_map["ppi_yy"] - coef_map["ppi_kj"]),
                                "deconvolution": args.deconvolution,
                            }
                        )

        frame = pd.DataFrame(rows)
        write_table(frame, output_dir / "gppi_subject_metrics.tsv")

        summaries: list[dict[str, Any]] = []
        if not frame.empty:
            # 1) Per-target ROI results (post vs pre)
            if args.stage == "prepost":
                subj_phase = (
                    frame[frame["phase"].isin(["pre", "post"])]
                    .groupby(["subject", "seed_roi", "target_roi", "target_roi_set", "target_scope", "ppi_condition", "phase"], as_index=False)["gppi_beta"]
                    .mean()
                )
                for keys, sub_frame in subj_phase.groupby(["seed_roi", "target_roi", "target_roi_set", "target_scope", "ppi_condition"]):
                    seed_name, target_name, target_set, target_scope, ppi_cond = keys
                    pivot = sub_frame.pivot(index="subject", columns="phase", values="gppi_beta").dropna()
                    if not {"pre", "post"}.issubset(pivot.columns) or pivot.empty:
                        continue
                    paired = paired_t_summary(pivot["post"].to_numpy(dtype=float), pivot["pre"].to_numpy(dtype=float))
                    summaries.append(
                        {
                            "seed_roi": seed_name,
                            "target_roi": target_name,
                            "target_roi_set": target_set,
                            "target_scope": target_scope,
                            "ppi_condition": ppi_cond,
                            "test_type": "post_vs_pre",
                            "n": int(paired["n"]),
                            "mean_post": paired["mean_a"],
                            "mean_pre": paired["mean_b"],
                            "t": paired["t"],
                            "p": paired["p"],
                            "cohens_dz": paired["cohens_dz"],
                            "mean_diff": float(np.mean((pivot["post"] - pivot["pre"]).to_numpy(dtype=float))),
                        }
                    )

                # 2) Scope-aggregated results (within_literature / within_literature_spatial)
                subj_scope = (
                    frame[frame["phase"].isin(["pre", "post"]) & frame["target_scope"].isin(["within_literature", "within_literature_spatial"])]
                    .groupby(["subject", "seed_roi", "target_scope", "ppi_condition", "phase"], as_index=False)["gppi_beta"]
                    .mean()
                )
                for (seed_name, target_scope, ppi_cond), sub_frame in subj_scope.groupby(["seed_roi", "target_scope", "ppi_condition"]):
                    pivot = sub_frame.pivot(index="subject", columns="phase", values="gppi_beta").dropna()
                    if not {"pre", "post"}.issubset(pivot.columns) or pivot.empty:
                        continue
                    paired = paired_t_summary(pivot["post"].to_numpy(dtype=float), pivot["pre"].to_numpy(dtype=float))
                    summaries.append(
                        {
                            "seed_roi": seed_name,
                            "target_roi": "__scope_mean__",
                            "target_roi_set": "__mixed__",
                            "target_scope": target_scope,
                            "ppi_condition": ppi_cond,
                            "test_type": "post_vs_pre",
                            "n": int(paired["n"]),
                            "mean_post": paired["mean_a"],
                            "mean_pre": paired["mean_b"],
                            "t": paired["t"],
                            "p": paired["p"],
                            "cohens_dz": paired["cohens_dz"],
                            "mean_diff": float(np.mean((pivot["post"] - pivot["pre"]).to_numpy(dtype=float))),
                        }
                    )

                # 3) cross_sets: (delta within_literature) - (delta within_literature_spatial)
                wide = subj_scope.pivot_table(
                    index=["subject", "seed_roi", "ppi_condition", "phase"],
                    columns="target_scope",
                    values="gppi_beta",
                    aggfunc="mean",
                ).reset_index()
                if not wide.empty and {"within_literature", "within_literature_spatial"}.issubset(wide.columns):
                    for (seed_name, ppi_cond), sub_frame in wide.groupby(["seed_roi", "ppi_condition"]):
                        pre = sub_frame[sub_frame["phase"] == "pre"].set_index("subject")
                        post = sub_frame[sub_frame["phase"] == "post"].set_index("subject")
                        common = sorted(set(pre.index) & set(post.index))
                        if not common:
                            continue
                        pre_delta = (pre.loc[common, "within_literature"] - pre.loc[common, "within_literature_spatial"]).to_numpy(dtype=float)
                        post_delta = (post.loc[common, "within_literature"] - post.loc[common, "within_literature_spatial"]).to_numpy(dtype=float)
                        paired = paired_t_summary(post_delta, pre_delta)
                        summaries.append(
                            {
                                "seed_roi": seed_name,
                                "target_roi": "__cross_sets__",
                                "target_roi_set": "__mixed__",
                                "target_scope": "cross_sets",
                                "ppi_condition": ppi_cond,
                                "test_type": "post_vs_pre",
                                "n": int(paired["n"]),
                                "mean_post": paired["mean_a"],
                                "mean_pre": paired["mean_b"],
                                "t": paired["t"],
                                "p": paired["p"],
                                "cohens_dz": paired["cohens_dz"],
                                "mean_diff": float(np.nanmean(post_delta - pre_delta)),
                                "note": "cross_sets = (within_literature mean) - (within_literature_spatial mean)",
                            }
                        )
            else:
                # learning summary: one-sample test against 0 on subject-level run-averaged betas.
                for keys, sub_frame in frame.groupby(["seed_roi", "target_roi", "target_roi_set", "target_scope", "ppi_condition"]):
                    seed_name, target_name, target_set, target_scope, ppi_cond = keys
                    pivot = sub_frame.groupby("subject", as_index=False)["gppi_beta"].mean()
                    t_stat, p_val = stats.ttest_1samp(pivot["gppi_beta"].to_numpy(), popmean=0.0, nan_policy="omit")
                    summaries.append(
                        {
                            "seed_roi": seed_name,
                            "target_roi": target_name,
                            "target_roi_set": target_set,
                            "target_scope": target_scope,
                            "ppi_condition": ppi_cond,
                            "test_type": "vs_zero",
                            "n": int(len(pivot)),
                            "mean": float(pivot["gppi_beta"].mean()),
                            "t": float(t_stat),
                            "p": float(p_val),
                        }
                    )

        summary_df = pd.DataFrame(summaries)
        write_table(summary_df, output_dir / "gppi_group_summary.tsv")

        if not summary_df.empty:
            paper_df = summary_df.copy()
            paper_df.insert(0, "seed", seed_roi_name)
            paper_df.insert(1, "roi_set", roi_set)
            paper_df.insert(2, "deconvolution", args.deconvolution)
            write_table(paper_df, tables_si / f"table_gppi_summary_{seed_tag}_{roi_tag}.tsv")

        save_json(
            {
                "seed_roi": seed_roi_name,
                "seed_mask": str(seed_mask_path),
                "target_spec": {
                    "target_roi_dir": str(args.target_roi_dir) if args.target_roi_dir is not None else None,
                    "target_roi_sets": list(args.target_roi_sets) if args.target_roi_sets is not None else None,
                    "n_targets": int(len(target_specs)),
                },
                "roi_set_tag_for_output": roi_set,
                "conditions": conditions,
                "deconvolution": args.deconvolution,
                "deconv_l2": float(args.deconv_l2),
                "stage": args.stage,
                "runs": [int(r) for r in args.runs],
                "n_rows": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()) if not frame.empty else 0,
                "output_dir": str(output_dir),
                "paper_table": str(tables_si / f"table_gppi_summary_{seed_tag}_{roi_tag}.tsv"),
            },
            output_dir / "gppi_meta.json",
        )
        all_rows.extend(rows)

    # Also write a combined long table across all seeds when running manifest mode.
    if all_rows:
        combined_dir = ensure_dir(paper_root / "qc" / f"gppi_combined_{roi_tag}")
        write_table(pd.DataFrame(all_rows), combined_dir / "gppi_subject_metrics_all_seeds.tsv")


if __name__ == "__main__":
    main()
