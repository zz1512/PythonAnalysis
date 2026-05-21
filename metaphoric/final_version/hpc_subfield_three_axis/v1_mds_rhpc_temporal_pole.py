#!/usr/bin/env python3
"""V1: R hippocampus 整体 + 3 个右段 + R temporal pole 的 pre/post MDS 可视化。

距离矩阵：1 - Pearson r，跨被试 fisher-z 平均，per condition × time。
MDS：sklearn.manifold.MDS(n_components=2, dissimilarity='precomputed',
                          n_init=20, random_state=42)。

输出：
    v1_mds_visualisation/fig_mds_<roi>_pre_post.pdf / .png   1200 dpi
    v1_mds_visualisation/mds_coords.tsv
    v1_mds_visualisation/v1_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from shared_subfield import (
    add_common_args,
    default_config,
    log_text,
    module_dir,
    read_table,
    safe_output_path,
    write_outputs,
)

MODULE = "v1_mds_visualisation"
DEFAULT_ROIS = (
    "meta_R_hippocampus",
    "hpc_R_head",
    "hpc_R_body",
    "hpc_R_tail",
    "meta_R_temporal_pole",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--beta-long", type=Path, default=None,
                        help="子区 beta_long.tsv（S2 输出）。仅子区图必需；整体 ROI 通过 --roi-beta-long 传入。")
    parser.add_argument("--roi-beta-long", type=Path, default=None,
                        help="meta ROI 整体的 item-level beta 长表（subject / run_phase / condition / item_id / roi / beta_vector_path）。")
    parser.add_argument("--pair-table", type=Path, default=None,
                        help="pair-level 结构表（condition, item_id_a, item_id_b, edge_type），用于在 MDS 图上画 trained 实线 / pseudo 虚线。")
    parser.add_argument("--rois", nargs="*", default=list(DEFAULT_ROIS))
    parser.add_argument("--phases", nargs="*", default=["pre", "post"])
    parser.add_argument("--dpi", type=int, default=1200)
    return parser.parse_args()


def load_vector(path: str) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    if "beta" not in data.files:
        return None
    return np.asarray(data["beta"], dtype=float)


def default_meta_roi_masks(cfg) -> dict[str, Path]:
    return {
        "meta_R_hippocampus": cfg.base_dir / "roi_library" / "masks" / "meta_spatial" / "meta_R_hippocampus.nii.gz",
        "meta_R_temporal_pole": cfg.base_dir / "roi_library" / "masks" / "meta_metaphor" / "meta_R_temporal_pole.nii.gz",
    }


def extract_mask_vector(beta_path: Path, mask_path: Path) -> np.ndarray | None:
    if not beta_path.exists() or not mask_path.exists():
        return None
    beta = np.asarray(nib.load(str(beta_path)).dataobj)
    mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    if beta.shape != mask.shape:
        return None
    return beta[mask].astype(float)


def build_default_roi_beta_long(cfg, args: argparse.Namespace) -> pd.DataFrame:
    items_path = cfg.output_root / "p0_inputs" / "hpc_items.tsv"
    if not items_path.exists():
        return pd.DataFrame()
    items = read_table(items_path)
    items = items[items["run_phase"].astype(str).isin(args.phases)].copy()
    if items.empty:
        return pd.DataFrame()
    masks = default_meta_roi_masks(cfg)
    requested = {roi for roi in args.rois if roi in masks}
    if not requested:
        return pd.DataFrame()
    vectors_dir = module_dir(cfg, MODULE) / "roi_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for roi in sorted(requested):
        mask_path = masks[roi]
        for _, row in items.iterrows():
            beta_path = Path(str(row.get("beta_path", "")))
            vec = extract_mask_vector(beta_path, mask_path)
            if vec is None:
                continue
            npz_path = vectors_dir / f"{row['subject']}__{roi}__{row['run_phase']}__{row['item_id']}.npz"
            if npz_path.exists():
                raise FileExistsError(f"Output exists; refusing to overwrite: {npz_path}")
            np.savez_compressed(npz_path, beta=vec)
            rows.append(
                {
                    "subject": row["subject"],
                    "run_phase": row["run_phase"],
                    "condition": row["condition"],
                    "item_id": row["item_id"],
                    "subROI": roi,
                    "roi": roi,
                    "n_voxels": int(vec.shape[0]),
                    "beta_vector_path": str(npz_path),
                }
            )
    return pd.DataFrame(rows)


def fisher_z(r: float) -> float:
    if r is None or not np.isfinite(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def fisher_z_to_r(z: float) -> float:
    if z is None or not np.isfinite(z):
        return np.nan
    return float(np.tanh(z))


def build_distance_matrix(beta_long: pd.DataFrame, roi: str, phase: str) -> tuple[np.ndarray, list[str], list[str]]:
    sub = beta_long[(beta_long["subROI"].astype(str) == roi)
                    | (beta_long.get("roi", pd.Series(dtype=str)).astype(str) == roi)]
    sub = sub[sub["run_phase"].astype(str) == phase]
    if sub.empty:
        return np.zeros((0, 0)), [], []
    # 每被试每 item 一条向量
    items = sorted({(str(c), str(i)) for c, i in zip(sub["condition"], sub["item_id"])})
    n = len(items)
    if n < 2:
        return np.zeros((0, 0)), [], []
    item_index = {it: idx for idx, it in enumerate(items)}
    z_sum = np.zeros((n, n))
    z_cnt = np.zeros((n, n))
    for sub_id, grp in sub.groupby("subject"):
        vecs = {}
        for _, row in grp.iterrows():
            key = (str(row["condition"]), str(row["item_id"]))
            v = load_vector(str(row.get("beta_vector_path", "")))
            if v is not None:
                vecs[key] = v
        keys = list(vecs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = vecs[keys[i]], vecs[keys[j]]
                if a.size != b.size or a.size < 2:
                    continue
                if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                    continue
                r = float(np.corrcoef(a, b)[0, 1])
                z = fisher_z(r)
                ii, jj = item_index[keys[i]], item_index[keys[j]]
                z_sum[ii, jj] += z
                z_sum[jj, ii] += z
                z_cnt[ii, jj] += 1
                z_cnt[jj, ii] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        z_mean = np.where(z_cnt > 0, z_sum / np.maximum(z_cnt, 1), np.nan)
    r_mean = np.tanh(z_mean)
    dist = 1.0 - r_mean
    np.fill_diagonal(dist, 0.0)
    # 兜底：将仍为 nan 的值置为列均值
    if np.isnan(dist).any():
        col_mean = np.nanmean(dist)
        dist = np.where(np.isnan(dist), col_mean if np.isfinite(col_mean) else 1.0, dist)
    conditions = [it[0] for it in items]
    item_ids = [it[1] for it in items]
    return dist, conditions, item_ids


def run_mds(dist: np.ndarray) -> np.ndarray:
    if dist.shape[0] < 2:
        return np.zeros((dist.shape[0], 2))
    try:
        from sklearn.manifold import MDS
    except ImportError as exc:
        raise RuntimeError("需要 scikit-learn 才能跑 MDS。") from exc
    mds = MDS(n_components=2, dissimilarity="precomputed", n_init=20,
              random_state=42, normalized_stress="auto")
    return mds.fit_transform(dist)


def plot_mds(coords_pre, conds_pre, items_pre, coords_post, conds_post, items_post,
             roi: str, out_pdf: Path, out_png: Path, dpi: int,
             pair_table: pd.DataFrame | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("需要 matplotlib 才能绘图。") from exc
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    color_map = {"YY": "#d1495b", "KJ": "#2e8b8b", "baseline": "#999999"}
    edge_style = {"trained": "-", "pseudo": "--"}
    for ax, coords, conds, items, title in [
        (axes[0], coords_pre, conds_pre, items_pre, f"{roi} — pre"),
        (axes[1], coords_post, conds_post, items_post, f"{roi} — post"),
    ]:
        if coords.shape[0] == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            ax.set_axis_off()
            continue
        # 先按 pair_table 在配对点之间画 line2D（trained 实线 / pseudo 虚线）
        if pair_table is not None and not pair_table.empty:
            index_lookup = {(str(c), str(it)): i for i, (c, it) in enumerate(zip(conds, items))}
            for _, prow in pair_table.iterrows():
                edge_type = str(prow.get("edge_type", ""))
                if edge_type not in edge_style:
                    continue
                cond = str(prow.get("condition", ""))
                a_key = (cond, str(prow.get("item_id_a", "")))
                b_key = (cond, str(prow.get("item_id_b", "")))
                if a_key not in index_lookup or b_key not in index_lookup:
                    continue
                ia, ib = index_lookup[a_key], index_lookup[b_key]
                ax.plot(
                    [coords[ia, 0], coords[ib, 0]],
                    [coords[ia, 1], coords[ib, 1]],
                    linestyle=edge_style[edge_type],
                    color=color_map.get(cond, "#888"),
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )
        for i, (c, item) in enumerate(zip(conds, items)):
            ax.scatter(coords[i, 0], coords[i, 1], s=40, alpha=0.85,
                       color=color_map.get(c, "#888"), edgecolor="white", linewidth=0.6,
                       zorder=2)
            ax.annotate(str(item), (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(title)
        ax.set_xlabel("MDS dim 1")
        ax.set_ylabel("MDS dim 2")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"MDS: {roi} pre vs post")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_pdf, dpi=dpi)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = default_config(args)

    long_frames = []
    beta_path = Path(args.beta_long) if args.beta_long else (cfg.output_root / "s2_beta_extract" / "beta_long.tsv")
    if beta_path.exists():
        long_frames.append(read_table(beta_path))
    if args.roi_beta_long and Path(args.roi_beta_long).exists():
        roi_long = read_table(Path(args.roi_beta_long))
        if "subROI" not in roi_long.columns and "roi" in roi_long.columns:
            roi_long = roi_long.rename(columns={"roi": "subROI"})
        long_frames.append(roi_long)
    elif any(str(roi).startswith("meta_") for roi in args.rois):
        roi_long = build_default_roi_beta_long(cfg, args)
        if not roi_long.empty:
            long_frames.append(roi_long)
    beta_long = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

    pair_table = None
    pair_path = Path(args.pair_table) if args.pair_table else (cfg.output_root / "p0_inputs" / "hpc_pair_table.tsv")
    if pair_path.exists():
        pair_table = read_table(pair_path)

    coords_records = []
    log_lines = []

    out_dir = module_dir(cfg, MODULE)

    for roi in args.rois:
        coords_pair = {}
        meta_pair = {}
        for phase in args.phases:
            dist, conds, items = build_distance_matrix(beta_long, roi, phase)
            coords = run_mds(dist) if dist.shape[0] >= 2 else np.zeros((0, 2))
            coords_pair[phase] = coords
            meta_pair[phase] = (conds, items)
            for i in range(coords.shape[0]):
                coords_records.append({
                    "roi": roi,
                    "phase": phase,
                    "condition": conds[i],
                    "item_id": items[i],
                    "dim1": float(coords[i, 0]),
                    "dim2": float(coords[i, 1]),
                })
            log_lines.append(f"{roi} {phase} n_items={dist.shape[0]}")
        out_pdf = safe_output_path(cfg, MODULE, f"fig_mds_{roi}_pre_post.pdf")
        out_png = safe_output_path(cfg, MODULE, f"fig_mds_{roi}_pre_post.png")
        plot_mds(
            coords_pair.get("pre", np.zeros((0, 2))),
            meta_pair.get("pre", ([], []))[0],
            meta_pair.get("pre", ([], []))[1],
            coords_pair.get("post", np.zeros((0, 2))),
            meta_pair.get("post", ([], []))[0],
            meta_pair.get("post", ([], []))[1],
            roi=roi,
            out_pdf=out_pdf,
            out_png=out_png,
            dpi=args.dpi,
            pair_table=pair_table,
        )

    coords_frame = pd.DataFrame(coords_records, columns=["roi", "phase", "condition", "item_id", "dim1", "dim2"])
    write_outputs(cfg, MODULE, {
        "mds_coords.tsv": coords_frame,
        "v1_log.txt": log_text("V1 MDS visualisation log", log_lines),
    })


if __name__ == "__main__":
    main()
