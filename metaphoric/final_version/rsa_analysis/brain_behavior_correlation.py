#!/usr/bin/env python3
"""
brain_behavior_correlation.py

Purpose
- Compute ROI-wise brain-behavior association:
  subject-level delta similarity (post - pre) vs Run-7 accuracy.
- Default uses partial Spearman correlation, controlling kj accuracy.

Outputs
- brain_behavior_correlation.tsv (roi, r, p_raw, p_fdr, n_subjects)
- brain_behavior_scatter.png (simple panel)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR (no dependency)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out
    p0 = p[mask]
    order = np.argsort(p0)
    ranked = p0[order]
    m = float(len(ranked))
    q = ranked * (m / (np.arange(1, len(ranked) + 1)))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(ranked)
    restored[order] = q
    out[mask] = restored
    return out


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residualize y on x with intercept (OLS)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - (X @ beta)


def _partial_spearman(y: np.ndarray, x: np.ndarray, cov: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(y) & np.isfinite(x) & np.isfinite(cov)
    n = int(valid.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    y_r = stats.rankdata(y[valid])
    x_r = stats.rankdata(x[valid])
    cov_r = stats.rankdata(cov[valid])
    y_res = _residualize(y_r, cov_r)
    x_res = _residualize(x_r, cov_r)
    if np.allclose(np.std(y_res), 0) or np.allclose(np.std(x_res), 0):
        return float("nan"), float("nan"), n
    r, p = stats.pearsonr(x_res, y_res)
    return float(r), float(p), n


def _default_out_dir() -> Path:
    override = os.environ.get("METAPHOR_BRAIN_BEHAVIOR_OUT_DIR", "").strip()
    if override:
        return Path(override)
    base = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "."))
    roi_set = os.environ.get("METAPHOR_ROI_SET", "unknown")
    return base / f"brain_behavior_{roi_set}"


def _load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _prepare_rsa_delta(itemwise: pd.DataFrame, *, condition: str) -> pd.DataFrame:
    work = itemwise.copy()
    if "time" not in work.columns and "stage" in work.columns:
        work["time"] = work["stage"]
    work["time"] = work["time"].astype(str).str.strip().str.lower()
    work["time"] = work["time"].replace({"pre": "pre", "post": "post", "pre-test": "pre", "post-test": "post"})
    required = {"subject", "roi", "condition", "time", "similarity"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"RSA item-wise table missing columns: {sorted(missing)}")
    work = work[work["condition"].astype(str).eq(condition)].copy()
    grouped = (
        work[work["time"].isin(["pre", "post"])]
        .groupby(["subject", "roi", "time"], as_index=False)["similarity"]
        .mean()
        .rename(columns={"similarity": "mean_similarity"})
    )
    pivot = grouped.pivot_table(index=["subject", "roi"], columns="time", values="mean_similarity", aggfunc="mean")
    pivot = pivot.dropna(subset=["pre", "post"]).copy()
    pivot["delta_similarity"] = pivot["post"] - pivot["pre"]
    out = pivot.reset_index()[["subject", "roi", "delta_similarity"]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-wise brain-behavior correlation (partial Spearman).")
    parser.add_argument("--rsa-itemwise", type=Path, required=True,
                        help="rsa_itemwise_details.csv from run_rsa_optimized.py.")
    parser.add_argument("--behavior", type=Path, required=True,
                        help="Behavior table with subject and accuracy columns.")
    parser.add_argument("--condition", default="yy", help="Condition for delta similarity (default: yy).")
    parser.add_argument("--behavior-col", default="accuracy_yy_run7",
                        help="Behavior column for target accuracy (default: accuracy_yy_run7).")
    parser.add_argument("--covariate-col", default="accuracy_kj_run7",
                        help="Covariate column to control (default: accuracy_kj_run7).")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rsa_item = _load_table(args.rsa_itemwise)
    delta = _prepare_rsa_delta(rsa_item, condition=str(args.condition))

    beh = _load_table(args.behavior)
    if "subject" not in beh.columns:
        raise ValueError("Behavior table must contain `subject` column.")
    for col in [args.behavior_col, args.covariate_col]:
        if col not in beh.columns:
            raise ValueError(f"Behavior table missing column: {col}")
    beh = beh[["subject", args.behavior_col, args.covariate_col]].copy()
    beh[args.behavior_col] = pd.to_numeric(beh[args.behavior_col], errors="coerce")
    beh[args.covariate_col] = pd.to_numeric(beh[args.covariate_col], errors="coerce")

    merged = delta.merge(beh, on="subject", how="inner")

    rows = []
    for roi, sub in merged.groupby("roi"):
        r, p, n = _partial_spearman(
            y=sub[args.behavior_col].to_numpy(),
            x=sub["delta_similarity"].to_numpy(),
            cov=sub[args.covariate_col].to_numpy(),
        )
        rows.append({"roi": str(roi), "r": r, "p_raw": p, "n_subjects": n})

    result = pd.DataFrame(rows)
    result["p_fdr"] = _bh_fdr(result["p_raw"].to_numpy(dtype=float))
    result = result.sort_values(["p_fdr", "p_raw"], na_position="last").reset_index(drop=True)
    out_tsv = out_dir / "brain_behavior_correlation.tsv"
    result.to_csv(out_tsv, sep="\t", index=False)

    # Simple scatter panel for top ROIs by p_fdr
    import matplotlib.pyplot as plt

    top = result.dropna(subset=["r", "p_raw"]).head(12)
    n_panels = len(top)
    if n_panels:
        n_cols = 4
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for i, row in enumerate(top.itertuples(index=False)):
            ax = axes.ravel()[i]
            ax.set_visible(True)
            roi = row.roi
            sub = merged[merged["roi"].astype(str).eq(str(roi))].copy()
            ax.scatter(sub["delta_similarity"], sub[args.behavior_col], alpha=0.8)
            ax.set_title(f"{roi}\nr={row.r:.3f}, p={row.p_raw:.3g}, q={row.p_fdr:.3g}")
            ax.set_xlabel("delta_similarity (post-pre)")
            ax.set_ylabel(args.behavior_col)
        plt.tight_layout()
        fig_path = out_dir / "brain_behavior_scatter.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()

    print(f"[brain-behavior] wrote {out_tsv}")


if __name__ == "__main__":
    main()

