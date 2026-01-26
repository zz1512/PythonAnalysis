import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter

def load_gifti_flatten(path):
    g = nib.load(path)
    return np.asarray(g.darrays[0].data).reshape(-1).astype(np.float32, copy=False)

def safe_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa, bb = a[m], b[m]
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return np.nan
    return float(np.dot(aa, bb) / denom)

def resolve_beta_path(lss_root, task, subject, run, rel_file):
    p1 = os.path.join(lss_root, task, subject, rel_file)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(lss_root, task, subject, f"run-{run}", rel_file)
    if os.path.exists(p2):
        return p2
    return None

def subject_to_mean_check(aligned_csv, lss_root, task, min_frac=0.9, max_subjects=None):
    df = pd.read_csv(aligned_csv)
    df = df[df["task"] == task].copy()

    subs = sorted(df["subject"].unique().tolist())
    if max_subjects:
        subs = subs[:max_subjects]

    # pick a "common stimulus_content" present in >= min_frac subjects
    cnt = Counter(df["stimulus_content"].tolist())
    need = int(np.ceil(min_frac * len(subs)))
    common = [k for k, v in cnt.items() if v >= need]
    if not common:
        raise RuntimeError("No common stimulus_content found under your threshold.")
    stim = common[0]
    print(f"Using stimulus_content: {stim} (need>={need}, got={cnt[stim]})")

    betas = []
    used_subs = []
    for sub in subs:
        row = df[(df["subject"] == sub) & (df["stimulus_content"] == stim)]
        if row.empty:
            continue
        row = row.iloc[0]
        p = resolve_beta_path(lss_root, task, sub, row["run"], row["file"])
        if not p:
            continue
        betas.append(load_gifti_flatten(p))
        used_subs.append(sub)

    betas = np.stack(betas, axis=0)
    mean_beta = betas.mean(axis=0)

    rs = [safe_corr(betas[i], mean_beta) for i in range(betas.shape[0])]
    rs = np.array(rs, dtype=float)

    print(f"N subjects used: {len(used_subs)}")
    print(f"Subject-to-mean r: mean={np.nanmean(rs):.4f}, sd={np.nanstd(rs):.4f}, "
          f"min={np.nanmin(rs):.4f}, median={np.nanmedian(rs):.4f}, max={np.nanmax(rs):.4f}")
    # show a few worst
    worst_idx = np.argsort(rs)[:5]
    print("Worst 5:", [(used_subs[i], float(rs[i])) for i in worst_idx])

    return used_subs, rs

# Example:
# subject_to_mean_check("results/lss_index_surface_L_aligned.csv",
#                       "LSS/lss_betas_root",
#                       task="EMO",
#                       min_frac=0.9)
if __name__ == "__main__":
    ALIGNED_CSV = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results/lss_index_surface_L_aligned.csv"
    LSS_ROOT = "/public/home/dingrui/fmri_analysis/zz_analysis/lss_results"
    # 检查 EMO 任务，取前 3 个最常见的刺激做平均
    subject_to_mean_check(
        aligned_csv=ALIGNED_CSV,
        lss_root=LSS_ROOT,
        task="EMO",
        min_frac=0.9
    )