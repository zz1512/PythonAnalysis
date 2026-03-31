#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata

DEFAULT_MATRIX_DIR = Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final")
DEFAULT_MODELS = ("M_nn", "M_conv", "M_div")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sliding-window Brain_ISC ~ Behavior_ISC coupling along an age model")
    p.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    p.add_argument("--stimulus-dir-name", type=str, default="by_stimulus", choices=("by_stimulus", "by_emotion"))
    p.add_argument("--stimulus-type", type=str, required=True)
    p.add_argument("--model", type=str, default="M_conv", choices=DEFAULT_MODELS)
    p.add_argument("--brain-isc-method", type=str, default="mahalanobis")
    p.add_argument("--brain-isc-prefix", type=str, default=None)
    p.add_argument("--behavior-isc-method", type=str, default="mahalanobis")
    p.add_argument("--behavior-isc-prefix", type=str, default=None)
    p.add_argument("--result-file", type=str, default="roi_isc_behavior_age_regression.csv")
    p.add_argument("--subject-file", type=str, default="roi_isc_behavior_age_regression_subjects_sorted.csv")
    p.add_argument("--rois", nargs="+", default=None, help="Optional ROI list. If omitted, auto-select from the result table.")
    p.add_argument(
        "--effect-col",
        type=str,
        default="beta_interaction",
        choices=("beta_interaction", "beta_behavior", "beta_age", "t_interaction", "t_behavior", "t_age"),
    )
    p.add_argument(
        "--p-col",
        type=str,
        default="p_fdr_perm_interaction_model_wise",
        choices=(
            "p_fdr_perm_interaction_model_wise",
            "p_fwer_interaction_model_wise",
            "p_perm_interaction",
            "p_fdr_perm_behavior_model_wise",
            "p_fwer_behavior_model_wise",
            "p_perm_behavior",
            "p_fdr_perm_age_model_wise",
            "p_fwer_age_model_wise",
            "p_perm_age",
        ),
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--negative-only", action="store_true")
    p.add_argument("--show-all-if-no-sig", action="store_true", default=False)
    p.add_argument("--rank-transform", action="store_true", help="Rank + zscore Behavior_ISC and Brain_ISC before plotting.")
    p.add_argument("--window-frac", type=float, default=0.15, help="Fraction of pairwise observations used in each sliding window.")
    p.add_argument("--step-frac", type=float, default=0.03, help="Sliding step size as a fraction of all pairwise observations.")
    p.add_argument("--min-pairs", type=int, default=5000, help="Minimum number of pairwise observations per window.")
    p.add_argument("--show-density", action="store_true", help="Add the distribution of age-model values below the slope curve.")
    p.add_argument("--model-value-scale", type=str, default="raw", choices=("raw", "zscore"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _branch_tag(stimulus_dir_name: str) -> str:
    return "trial" if str(stimulus_dir_name) == "by_stimulus" else "emotion"


def _resolve_prefix(prefix: Optional[str], method: str, fallback_template: str) -> str:
    return str(prefix) if prefix is not None else fallback_template.format(method=str(method).strip().lower())


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.full(xv.shape, np.nan, dtype=np.float64)
    m = np.isfinite(xv)
    if not bool(m.any()):
        return out
    vals = xv[m]
    mu = float(vals.mean())
    sd = float(vals.std(ddof=0))
    if sd <= 0:
        out[m] = 0.0
    else:
        out[m] = (vals - mu) / sd
    return out


def _rank_z(x: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.full(xv.shape, np.nan, dtype=np.float64)
    m = np.isfinite(xv)
    if not bool(m.any()):
        return out
    out[m] = rankdata(xv[m])
    return _zscore_1d(out)


def _load_overlap_subjects(stim_dir: Path, subject_file: str) -> pd.DataFrame:
    path = stim_dir / str(subject_file)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find overlap subject file: {path}")
    sub = pd.read_csv(path)
    need = {"subject", "age"}
    miss = need - set(sub.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    sub = sub.copy()
    sub["subject"] = sub["subject"].astype(str)
    sub["age"] = pd.to_numeric(sub["age"], errors="coerce")
    sub = sub[np.isfinite(sub["age"].to_numpy(dtype=float))].copy()
    if sub.empty:
        raise ValueError(f"No usable subjects in {path}")
    return sub.reset_index(drop=True)


def _subset_matrix_by_subjects(mat: np.ndarray, subjects_df: pd.DataFrame, overlap_subjects: Sequence[str]) -> np.ndarray:
    subj = subjects_df["subject"].astype(str).tolist()
    idx_map = {s: i for i, s in enumerate(subj)}
    idx = [idx_map[s] for s in overlap_subjects]
    arr = np.asarray(mat)
    if arr.ndim == 3:
        return arr[:, idx, :][:, :, idx]
    if arr.ndim == 2:
        return arr[np.ix_(idx, idx)]
    raise ValueError(f"Unsupported matrix ndim: {arr.ndim}")


def _select_rois(
    stim_dir: Path,
    result_file: str,
    model: str,
    effect_col: str,
    p_col: str,
    alpha: float,
    top_k: int,
    positive_only: bool,
    negative_only: bool,
    show_all_if_no_sig: bool,
    explicit_rois: Optional[Sequence[str]],
    brain_isc_prefix: str,
) -> List[str]:
    roi_list = pd.read_csv(stim_dir / f"{brain_isc_prefix}_rois.csv")["roi"].astype(str).tolist()
    roi_set = set(roi_list)
    if explicit_rois:
        keep = [str(r) for r in explicit_rois if str(r) in roi_set]
        missing = [str(r) for r in explicit_rois if str(r) not in roi_set]
        if missing:
            raise ValueError(f"ROIs not found in {stim_dir.name}: {missing}")
        return keep

    res_path = stim_dir / str(result_file)
    if not res_path.exists():
        raise FileNotFoundError(f"Cannot find result file: {res_path}")
    df = pd.read_csv(res_path)
    need = {"roi", "model", str(effect_col), str(p_col)}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{res_path} missing columns: {sorted(miss)}")
    sub = df[df["model"].astype(str) == str(model)].copy()
    sub = sub[
        np.isfinite(pd.to_numeric(sub[str(effect_col)], errors="coerce").to_numpy(dtype=float))
        & np.isfinite(pd.to_numeric(sub[str(p_col)], errors="coerce").to_numpy(dtype=float))
    ].copy()
    if bool(positive_only) and bool(negative_only):
        raise ValueError("Cannot use both --positive-only and --negative-only")
    if bool(positive_only):
        sub = sub[pd.to_numeric(sub[str(effect_col)], errors="coerce") > 0].copy()
    if bool(negative_only):
        sub = sub[pd.to_numeric(sub[str(effect_col)], errors="coerce") < 0].copy()
    sig = sub[pd.to_numeric(sub[str(p_col)], errors="coerce") <= float(alpha)].copy()
    chosen = sig if not sig.empty else (sub if bool(show_all_if_no_sig) else pd.DataFrame(columns=sub.columns))
    if chosen.empty:
        raise ValueError(f"No ROI passed the selection rule for {stim_dir.name} | {model}")
    chosen = chosen.reindex(pd.to_numeric(chosen[str(effect_col)], errors="coerce").abs().sort_values(ascending=False).index)
    return chosen["roi"].astype(str).head(int(top_k)).tolist()


def _build_model_values(ages: np.ndarray, model: str, scale: str) -> np.ndarray:
    a = np.asarray(ages, dtype=np.float64).reshape(-1)
    iu, ju = np.triu_indices(int(a.size), k=1)
    amax = float(np.nanmax(a))
    ai = a[iu]
    aj = a[ju]
    model_key = str(model)
    if model_key == "M_nn":
        vals = amax - np.abs(ai - aj)
    elif model_key == "M_conv":
        vals = np.minimum(ai, aj)
    elif model_key == "M_div":
        vals = amax - 0.5 * (ai + aj)
    else:
        raise ValueError(f"Unknown model: {model}")
    if str(scale) == "zscore":
        return _zscore_1d(vals)
    return vals.astype(np.float64)


def _fit_simple_slope(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[valid]
    yv = yv[valid]
    n = int(xv.size)
    if n < 4:
        return {
            "n_pairs": float(n),
            "slope": np.nan,
            "intercept": np.nan,
            "corr": np.nan,
            "slope_se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    sxx = float(np.sum((xv - x_mean) ** 2))
    if sxx <= 0:
        return {
            "n_pairs": float(n),
            "slope": np.nan,
            "intercept": np.nan,
            "corr": np.nan,
            "slope_se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    sxy = float(np.sum((xv - x_mean) * (yv - y_mean)))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    resid = yv - (intercept + slope * xv)
    dof = max(1, n - 2)
    mse = float(np.sum(resid ** 2) / dof)
    slope_se = math.sqrt(max(0.0, mse / sxx))
    ci = 1.96 * slope_se
    corr = float(np.corrcoef(xv, yv)[0, 1]) if np.nanstd(xv) > 0 and np.nanstd(yv) > 0 else np.nan
    return {
        "n_pairs": float(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "corr": corr,
        "slope_se": float(slope_se),
        "ci_low": float(slope - ci),
        "ci_high": float(slope + ci),
    }


def _windowed_coupling(
    df: pd.DataFrame,
    model_col: str,
    x_col: str,
    y_col: str,
    window_frac: float,
    step_frac: float,
    min_pairs: int,
) -> pd.DataFrame:
    sub = df[[str(model_col), str(x_col), str(y_col)]].copy()
    sub = sub[
        np.isfinite(pd.to_numeric(sub[str(model_col)], errors="coerce").to_numpy(dtype=float))
        & np.isfinite(pd.to_numeric(sub[str(x_col)], errors="coerce").to_numpy(dtype=float))
        & np.isfinite(pd.to_numeric(sub[str(y_col)], errors="coerce").to_numpy(dtype=float))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values(str(model_col)).reset_index(drop=True)
    n = int(sub.shape[0])
    win = max(int(math.ceil(float(window_frac) * n)), int(min_pairs))
    win = min(win, n)
    step = max(1, int(math.ceil(float(step_frac) * n)))
    rows: List[Dict[str, float]] = []
    starts = list(range(0, max(1, n - win + 1), step))
    if starts[-1] != n - win:
        starts.append(max(0, n - win))
    starts = sorted(set(starts))
    model_vals = sub[str(model_col)].to_numpy(dtype=float)
    x_vals = sub[str(x_col)].to_numpy(dtype=float)
    y_vals = sub[str(y_col)].to_numpy(dtype=float)
    for start in starts:
        end = min(n, start + win)
        model_slice = model_vals[start:end]
        x_slice = x_vals[start:end]
        y_slice = y_vals[start:end]
        fit = _fit_simple_slope(x_slice, y_slice)
        fit.update(
            {
                "window_start": float(start),
                "window_end": float(end),
                "model_min": float(np.nanmin(model_slice)),
                "model_max": float(np.nanmax(model_slice)),
                "model_center": float(np.nanmean(model_slice)),
                "model_median": float(np.nanmedian(model_slice)),
                "behavior_mean": float(np.nanmean(x_slice)),
                "brain_mean": float(np.nanmean(y_slice)),
            }
        )
        rows.append(fit)
    return pd.DataFrame(rows)


def load_pair_frame(
    stim_dir: Path,
    roi: str,
    model: str,
    brain_isc_prefix: str,
    behavior_isc_prefix: str,
    subject_file: str,
    rank_transform: bool,
    model_value_scale: str,
) -> pd.DataFrame:
    overlap_df = _load_overlap_subjects(stim_dir, subject_file=str(subject_file))
    overlap_subjects = overlap_df["subject"].astype(str).tolist()
    ages = overlap_df["age"].to_numpy(dtype=float)

    brain = np.load(stim_dir / f"{brain_isc_prefix}.npy")
    beh = np.load(stim_dir / f"{behavior_isc_prefix}.npy")
    brain_sub_df = pd.read_csv(stim_dir / f"{brain_isc_prefix}_subjects_sorted.csv")
    beh_sub_df = pd.read_csv(stim_dir / f"{behavior_isc_prefix}_subjects_sorted.csv")
    roi_df = pd.read_csv(stim_dir / f"{brain_isc_prefix}_rois.csv")
    roi_list = roi_df["roi"].astype(str).tolist()
    if roi not in roi_list:
        raise ValueError(f"ROI not found in {stim_dir.name}: {roi}")
    ri = int(roi_list.index(roi))

    brain = _subset_matrix_by_subjects(brain, brain_sub_df, overlap_subjects)
    beh = _subset_matrix_by_subjects(beh, beh_sub_df, overlap_subjects)
    iu, ju = np.triu_indices(len(overlap_subjects), k=1)
    pair_mean_age = 0.5 * (ages[iu] + ages[ju])
    model_value = _build_model_values(ages=ages, model=str(model), scale=str(model_value_scale))
    brain_vec = np.asarray(brain[ri][iu, ju], dtype=np.float64)
    beh_vec = np.asarray(beh[iu, ju], dtype=np.float64)
    if bool(rank_transform):
        brain_vec = _rank_z(brain_vec)
        beh_vec = _rank_z(beh_vec)
    valid = np.isfinite(pair_mean_age) & np.isfinite(model_value) & np.isfinite(brain_vec) & np.isfinite(beh_vec)
    pair_mean_age = pair_mean_age[valid]
    model_value = model_value[valid]
    brain_vec = brain_vec[valid]
    beh_vec = beh_vec[valid]
    out = pd.DataFrame(
        {
            "pair_mean_age": pair_mean_age,
            "model_value": model_value,
            "behavior_isc": beh_vec,
            "brain_isc": brain_vec,
        }
    )
    return out


def plot_one(
    df: pd.DataFrame,
    out_png: Path,
    out_window_csv: Path,
    title: str,
    model_name: str,
    window_frac: float,
    step_frac: float,
    min_pairs: int,
    show_density: bool,
    dpi: int,
) -> pd.DataFrame:
    win_df = _windowed_coupling(
        df=df,
        model_col="model_value",
        x_col="behavior_isc",
        y_col="brain_isc",
        window_frac=float(window_frac),
        step_frac=float(step_frac),
        min_pairs=int(min_pairs),
    )
    if win_df.empty:
        return pd.DataFrame()
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks")
    if bool(show_density):
        fig, (ax, ax_den) = plt.subplots(
            2,
            1,
            figsize=(7.0, 6.2),
            gridspec_kw={"height_ratios": [4.0, 1.2], "hspace": 0.08},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax_den = None
    x_center = win_df["model_center"].to_numpy(dtype=float)
    slope = win_df["slope"].to_numpy(dtype=float)
    ci_low = win_df["ci_low"].to_numpy(dtype=float)
    ci_high = win_df["ci_high"].to_numpy(dtype=float)
    corr = win_df["corr"].to_numpy(dtype=float)
    ax.fill_between(x_center, ci_low, ci_high, color="#8FB9E3", alpha=0.35, linewidth=0)
    ax.plot(x_center, slope, color="#1F5A99", lw=2.4, label="Slope: Brain_ISC ~ Behavior_ISC")
    if np.isfinite(corr).any():
        ax.plot(x_center, corr, color="#D97904", lw=1.8, ls="--", label="Pairwise corr")
    ax.axhline(0.0, color="black", lw=1.0, ls=":")
    ax.set_title(title)
    ax.set_ylabel("Windowed coupling slope")
    ax.legend(frameon=True, loc="best")
    if ax_den is not None:
        sns.histplot(
            data=df,
            x="model_value",
            bins=40,
            stat="density",
            color="#7A7A7A",
            alpha=0.45,
            edgecolor=None,
            ax=ax_den,
        )
        ax_den.set_ylabel("Density")
        ax_den.set_xlabel(str(model_name))
    else:
        ax.set_xlabel(str(model_name))
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    win_df.to_csv(out_window_csv, index=False)
    return win_df


def main() -> None:
    args = parse_args()
    stim_dir = Path(args.matrix_dir) / str(args.stimulus_dir_name) / str(args.stimulus_type)
    if not stim_dir.exists():
        raise FileNotFoundError(f"Cannot find stimulus directory: {stim_dir}")

    brain_isc_prefix = _resolve_prefix(args.brain_isc_prefix, method=str(args.brain_isc_method), fallback_template="roi_isc_{method}_by_age")
    behavior_isc_prefix = _resolve_prefix(args.behavior_isc_prefix, method=str(args.behavior_isc_method), fallback_template="behavior_isc_{method}_by_age")
    rois = _select_rois(
        stim_dir=stim_dir,
        result_file=str(args.result_file),
        model=str(args.model),
        effect_col=str(args.effect_col),
        p_col=str(args.p_col),
        alpha=float(args.alpha),
        top_k=int(args.top_k),
        positive_only=bool(args.positive_only),
        negative_only=bool(args.negative_only),
        show_all_if_no_sig=bool(args.show_all_if_no_sig),
        explicit_rois=args.rois,
        brain_isc_prefix=str(brain_isc_prefix),
    )

    branch_tag = _branch_tag(str(args.stimulus_dir_name))
    out_dir = Path(args.matrix_dir) / "figures" / f"{str(args.stimulus_dir_name)}_brain_behavior_age_coupling"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows: List[Dict[str, object]] = []
    for roi in rois:
        pair_df = load_pair_frame(
            stim_dir=stim_dir,
            roi=str(roi),
            model=str(args.model),
            brain_isc_prefix=str(brain_isc_prefix),
            behavior_isc_prefix=str(behavior_isc_prefix),
            subject_file=str(args.subject_file),
            rank_transform=bool(args.rank_transform),
            model_value_scale=str(args.model_value_scale),
        )
        if int(pair_df.shape[0]) < 10:
            continue
        stem = (
            f"brain_behavior_age_coupling_{branch_tag}_{args.stimulus_type}_{args.model}_{roi}"
        )
        png_path = out_dir / f"{stem}.png"
        csv_path = out_dir / f"{stem}_pairs.csv"
        window_csv_path = out_dir / f"{stem}_window_summary.csv"
        win_df = plot_one(
            df=pair_df,
            out_png=png_path,
            out_window_csv=window_csv_path,
            title=f"{args.stimulus_type} | {args.model} | {roi}",
            model_name=str(args.model),
            window_frac=float(args.window_frac),
            step_frac=float(args.step_frac),
            min_pairs=int(args.min_pairs),
            show_density=bool(args.show_density),
            dpi=int(args.dpi),
        )
        pair_df.to_csv(csv_path, index=False)
        meta_rows.append(
            {
                "stimulus_type": str(args.stimulus_type),
                "model": str(args.model),
                "roi": str(roi),
                "n_pairs": int(pair_df.shape[0]),
                "n_windows": int(win_df.shape[0]),
                "figure": str(png_path),
                "pairs_csv": str(csv_path),
                "window_summary_csv": str(window_csv_path),
            }
        )
        print(f"saved: {png_path}")

    meta_df = pd.DataFrame(meta_rows)
    if not meta_df.empty:
        meta_df.to_csv(
            out_dir / f"brain_behavior_age_coupling_{branch_tag}_{args.stimulus_type}_{args.model}_summary.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
