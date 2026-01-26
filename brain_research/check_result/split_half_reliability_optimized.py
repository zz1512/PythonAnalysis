# split_half_reliability_optimized.py
"""
split_half_reliability_optimized.py
数据校验 - 针对LSS分析产出的 beta 图，检验被试在不同 trials 之间的神经响应是否稳定一致
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm


###############################################################################
# Utilities
###############################################################################

def load_gifti_flatten(gii_path: str) -> np.ndarray:
    """Load a .func.gii beta file and flatten to 1D float32."""
    try:
        gii = nib.load(gii_path)
        data = gii.darrays[0].data
        return np.asarray(data).reshape(-1).astype(np.float32, copy=False)
    except Exception as e:
        print(f"Error loading {gii_path}: {e}")
        return None


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with NaN/constant guards."""
    if a.shape != b.shape:
        return np.nan
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return np.nan
    aa, bb = a[mask], b[mask]

    # Check for zero variance
    if aa.std() == 0 or bb.std() == 0:
        return np.nan

    return float(np.corrcoef(aa, bb)[0, 1])


def fisher_mean(rs: List[float]) -> float:
    """Compute mean of correlations using Fisher z-transformation."""
    rs = np.array(rs)
    rs = rs[np.isfinite(rs)]  # Remove NaNs
    # Clip to avoid infinity at r=1.0
    rs = np.clip(rs, -0.99999, 0.99999)
    if len(rs) == 0:
        return np.nan
    z_vals = np.arctanh(rs)
    mean_z = np.mean(z_vals)
    return np.tanh(mean_z)


###############################################################################
# Data Loading
###############################################################################

def build_subject_vectors(
        df_sub: pd.DataFrame,
        lss_root: str,
        task: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build data matrix X [n_trials, n_vertices]
    """
    # Filter by task
    d = df_sub[df_sub["task"] == task].copy()
    if d.empty:
        return np.empty((0, 0)), np.array([])

    # Resolve paths
    beta_paths = []
    # Pre-construct potential paths to speed up loop
    base_path_1 = Path(lss_root) / task.lower()  # e.g. lss_results/emo

    valid_rows = []

    for idx, row in d.iterrows():
        # Try constructing path: root/task/sub/run/file
        # Note: adjust this logic based on your actual LSS folder structure
        p = base_path_1 / row["subject"] / row["file"]  # Try direct
        if not p.exists():
            p = base_path_1 / row["subject"] / f"run-{row['run']}" / row["file"]

        if p.exists():
            beta_paths.append(str(p))
            valid_rows.append(idx)

    if not beta_paths:
        return np.empty((0, 0)), np.array([])

    d = d.loc[valid_rows]
    runs = d["run"].to_numpy()

    # Load vectors
    vecs = []
    for p in beta_paths:
        v = load_gifti_flatten(p)
        if v is not None:
            vecs.append(v)
        else:
            # Handle loading error by removing corresponding run entry
            runs = np.delete(runs, len(vecs))

    if not vecs:
        return np.empty((0, 0)), np.array([])

    X = np.stack(vecs, axis=0)
    return X, runs


###############################################################################
# Splitting Logic
###############################################################################

def split_half_random(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    mid = n // 2
    return idx[:mid], idx[mid:]


###############################################################################
# Main Runner
###############################################################################

def run_reliability_check(
        aligned_csv: str,
        lss_root: str,
        task: str = "EMO",
        n_repeats: int = 50,
        out_csv: str = "reliability_results.csv"
):
    print(f"--- Starting Reliability Check for Task: {task} ---")
    df = pd.read_csv(aligned_csv)
    subjects = sorted(df["subject"].unique().tolist())
    rng = np.random.default_rng(42)

    results = []

    for sub in tqdm(subjects, desc="Processing Subjects"):
        df_sub = df[df["subject"] == sub]

        # 1. Load Data
        X, runs = build_subject_vectors(df_sub, lss_root, task)

        n_trials = X.shape[0]
        if n_trials < 4:  # Need at least 4 trials to split 2 vs 2
            results.append({
                "subject": sub,
                "n_trials": n_trials,
                "reliability": np.nan,
                "note": "Too few trials"
            })
            continue

        # 2. Check if we can split by Run
        uniq_runs = np.unique(runs)
        rs = []

        # 3. Random Split Strategy (Recommended for single-run tasks)
        # We use Odd/Even logic implicitly by doing many random splits
        for _ in range(n_repeats):
            idx_a, idx_b = split_half_random(n_trials, rng)

            # Aggregate: Mean Map of Half A vs Mean Map of Half B
            # This checks spatial pattern stability
            map_a = X[idx_a].mean(axis=0)
            map_b = X[idx_b].mean(axis=0)

            r = safe_corr(map_a, map_b)
            rs.append(r)

        # 4. Fisher Average
        final_r = fisher_mean(rs)

        results.append({
            "subject": sub,
            "n_trials": n_trials,
            "reliability": final_r,
            "note": "Success"
        })

    # Output
    res_df = pd.DataFrame(results)

    print("\n=== Reliability Summary ===")
    print(res_df["reliability"].describe())

    # Save
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")

    # Simple check interpretation
    bad_subs = res_df[res_df["reliability"] < 0.1]
    print(f"\n⚠️  Potential low-quality subjects (r < 0.1): {len(bad_subs)}")
    if not bad_subs.empty:
        print(bad_subs["subject"].tolist())


if __name__ == "__main__":
    # 配置
    ALIGNED_CSV = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"
    LSS_ROOT = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"

    # 运行 EMO 任务的检查
    run_reliability_check(
        aligned_csv=ALIGNED_CSV,
        lss_root=LSS_ROOT,
        task="EMO",
        n_repeats=20,  # 随机分半 20 次取平均
        out_csv="reliability_check_EMO.csv"
    )