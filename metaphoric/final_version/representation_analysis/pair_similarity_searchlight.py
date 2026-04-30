"""
pair_similarity_searchlight.py

Purpose
- Minimal S3 searchlight that matches the current todo target:
  local pair similarity maps for `yy` / `kj` / `baseline`, followed by
  group-level `post - pre` sign-flip permutation tests.

Design choice
- This script is separate from `rd_searchlight.py` on purpose.
- The existing RD script may already be running; do not modify that path.
- Here we compute a different quantity: mean within-pair similarity in each
  local searchlight neighborhood.
- Default paper outputs use pair-specific filenames so they do not overwrite
  the RD searchlight main figure/table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np


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
from common.pattern_metrics import (  # noqa: E402
    build_fixed_count_neighborhood,
    load_4d_data,
    load_mask,
    save_scalar_map,
)
from representation_analysis.rd_searchlight import (  # noqa: E402
    _compute_group_stats,
    _extract_peak_table,
    _resolve_subject_mask,
)


def _default_base_dir() -> Path:
    from rsa_analysis.rsa_config import BASE_DIR  # noqa: E402

    return Path(BASE_DIR)


def _default_pattern_root() -> Path:
    return _default_base_dir() / "pattern_root"


def _default_subject_mask_root() -> Path:
    return _default_base_dir() / "lss_betas_final"


def _default_output_root() -> Path:
    return _default_base_dir() / "paper_outputs"


def _normalize_pair_column(metadata: pd.DataFrame, pair_col: str) -> pd.Series:
    if pair_col in metadata.columns:
        out = metadata[pair_col]
    elif pair_col == "pair_id" and "pic_num" in metadata.columns:
        out = metadata["pic_num"]
    else:
        raise ValueError(f"Metadata missing pair column: {pair_col}")
    return out.astype(str).str.strip()


def _pair_indices_from_metadata(metadata: pd.DataFrame, pair_col: str) -> list[tuple[int, int]]:
    work = metadata.reset_index(drop=True).copy()
    work["_pair_key"] = _normalize_pair_column(work, pair_col)
    invalid = {"", "nan", "<na>", "none"}
    work = work[~work["_pair_key"].str.lower().isin(invalid)].copy()
    pairs: list[tuple[int, int]] = []
    for _, cell in work.groupby("_pair_key", sort=False):
        idx = cell.index.to_list()
        if len(idx) == 2:
            pairs.append((int(idx[0]), int(idx[1])))
    return pairs


def _pair_similarity_value(local_samples: np.ndarray, pair_indices: list[tuple[int, int]]) -> float:
    centered = local_samples - local_samples.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    if valid.sum() < 2:
        return float("nan")
    normalized = np.zeros_like(centered, dtype=float)
    normalized[valid] = centered[valid] / norms[valid, None]
    values = []
    for idx_a, idx_b in pair_indices:
        if valid[idx_a] and valid[idx_b]:
            values.append(float(np.dot(normalized[idx_a], normalized[idx_b])))
    if not values:
        return float("nan")
    return float(np.mean(values))


def compute_pair_similarity_map(
    image_path: Path,
    mask_path: Path,
    metadata_path: Path,
    *,
    voxel_count: int,
    pair_col: str,
    min_pairs: int,
) -> tuple[object, np.ndarray, np.ndarray, int]:
    img, data = load_4d_data(image_path)
    _, mask = load_mask(mask_path)
    if data.shape[:3] != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_path} vs {mask_path}")

    metadata = read_table(metadata_path).copy()
    if len(metadata) != data.shape[3]:
        raise ValueError(
            f"Image/metadata mismatch: n_vols={data.shape[3]} vs rows={len(metadata)} at {metadata_path}"
        )
    pair_indices = _pair_indices_from_metadata(metadata, pair_col)
    if len(pair_indices) < int(min_pairs):
        raise ValueError(f"Not enough valid pairs: {len(pair_indices)} < {min_pairs}")

    coords, neighborhoods = build_fixed_count_neighborhood(mask, voxel_count)
    samples = data[mask, :].T.astype(np.float64, copy=False)
    values = np.full(coords.shape[0], np.nan, dtype=np.float32)
    for center_index, neighbor_indices in enumerate(neighborhoods):
        local_samples = samples[:, neighbor_indices]
        values[center_index] = _pair_similarity_value(local_samples, pair_indices)
    return img, mask, values, len(pair_indices)


def _plot_group_deltas(plot_payload: list[dict[str, object]], figure_path: Path) -> None:
    if not plot_payload:
        return
    import matplotlib.pyplot as plt

    n_panels = len(plot_payload)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5))
    axes = np.atleast_1d(axes)
    vmax = 0.0
    slices = []
    masks = []
    for row in plot_payload:
        diff_map = np.asarray(row["mean_diff_map"], dtype=float)
        p_fwe_map = np.asarray(row["p_fwe_map"], dtype=float)
        slice_img = diff_map[:, :, diff_map.shape[2] // 2]
        sig_mask = p_fwe_map[:, :, p_fwe_map.shape[2] // 2] < 0.05
        slices.append(slice_img)
        masks.append(sig_mask)
        vmax = max(vmax, float(np.nanmax(np.abs(slice_img))))
    vmax = vmax if vmax > 0 else 1.0
    for ax, row, slice_img, sig_mask in zip(axes, plot_payload, slices, masks):
        im = ax.imshow(np.rot90(slice_img), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        if np.any(sig_mask):
            ax.contour(np.rot90(sig_mask.astype(float)), levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(str(row["comparison"]), fontsize=11)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Post - Pre pair similarity")
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_cell_maps(
    subject_dirs: list[Path],
    subject_mask_root: Path,
    output_dir: Path,
    *,
    time: str,
    condition: str,
    filename_template: str,
    metadata_template: str,
    voxel_count: int,
    pair_col: str,
    min_pairs: int,
    mask_filename: str,
) -> tuple[list[Path], list[dict[str, object]]]:
    paths: list[Path] = []
    qc_rows: list[dict[str, object]] = []
    for subject_dir in subject_dirs:
        image_path = subject_dir / filename_template.format(time=time, condition=condition)
        metadata_path = subject_dir / metadata_template.format(time=time, condition=condition)
        subject_mask = _resolve_subject_mask(subject_mask_root, subject_dir.name, mask_filename)
        if not image_path.exists():
            qc_rows.append(
                {
                    "subject": subject_dir.name,
                    "time": time,
                    "condition": condition,
                    "status": "skipped",
                    "skip_reason": "missing_pattern",
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "mask_path": str(subject_mask) if subject_mask else "",
                    "n_pairs": 0,
                    "output_path": "",
                }
            )
            continue
        if not metadata_path.exists():
            qc_rows.append(
                {
                    "subject": subject_dir.name,
                    "time": time,
                    "condition": condition,
                    "status": "skipped",
                    "skip_reason": "missing_metadata",
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "mask_path": str(subject_mask) if subject_mask else "",
                    "n_pairs": 0,
                    "output_path": "",
                }
            )
            continue
        if subject_mask is None:
            qc_rows.append(
                {
                    "subject": subject_dir.name,
                    "time": time,
                    "condition": condition,
                    "status": "skipped",
                    "skip_reason": "missing_mask",
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "mask_path": "",
                    "n_pairs": 0,
                    "output_path": "",
                }
            )
            continue
        try:
            reference_img, mask, values, n_pairs = compute_pair_similarity_map(
                image_path,
                subject_mask,
                metadata_path,
                voxel_count=voxel_count,
                pair_col=pair_col,
                min_pairs=min_pairs,
            )
            subject_output = ensure_dir(output_dir / subject_dir.name)
            out_path = subject_output / f"pair_similarity_{time}_{condition}.nii.gz"
            save_scalar_map(reference_img, mask, values, out_path)
            paths.append(out_path)
            qc_rows.append(
                {
                    "subject": subject_dir.name,
                    "time": time,
                    "condition": condition,
                    "status": "ok",
                    "skip_reason": "",
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "mask_path": str(subject_mask),
                    "n_pairs": int(n_pairs),
                    "output_path": str(out_path),
                }
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject_dir.name,
                    "time": time,
                    "condition": condition,
                    "status": "skipped",
                    "skip_reason": repr(exc),
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                    "mask_path": str(subject_mask),
                    "n_pairs": 0,
                    "output_path": "",
                }
            )
    return paths, qc_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-brain pair-similarity searchlight analysis.")
    parser.add_argument("pattern_root", type=Path, nargs="?", default=None)
    parser.add_argument("subject_mask_root", type=Path, nargs="?", default=None)
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--metadata-template", default="{time}_{condition}_metadata.tsv")
    parser.add_argument("--voxel-count", type=int, default=100)
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj", "baseline"])
    parser.add_argument("--pair-col", default="pair_id")
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--permutation-backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--permutation-batch-size", type=int, default=256)
    parser.add_argument(
        "--mask-filename",
        default="mask.nii",
        help="Mask filename under each subject folder (default: mask.nii; will also try .nii.gz).",
    )
    args = parser.parse_args()

    pattern_root = args.pattern_root or _default_pattern_root()
    subject_mask_root = args.subject_mask_root or _default_subject_mask_root()
    paper_output_root = ensure_dir(args.output_dir or _default_output_root())
    qc_root = ensure_dir(paper_output_root / "qc" / "pair_similarity_searchlight")
    tables_main = ensure_dir(paper_output_root / "tables_main")
    figures_main = ensure_dir(paper_output_root / "figures_main")

    subject_dirs = sorted([path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])
    cell_paths: dict[tuple[str, str], list[Path]] = {}
    map_qc_rows: list[dict[str, object]] = []
    for time in ["pre", "post"]:
        for condition in args.conditions:
            paths, qc_rows = compute_cell_maps(
                subject_dirs,
                subject_mask_root,
                qc_root,
                time=time,
                condition=condition,
                filename_template=args.filename_template,
                metadata_template=args.metadata_template,
                voxel_count=args.voxel_count,
                pair_col=args.pair_col,
                min_pairs=args.min_pairs,
                mask_filename=args.mask_filename,
            )
            cell_paths[(time, condition)] = paths
            map_qc_rows.extend(qc_rows)
    map_qc = pd.DataFrame(map_qc_rows)
    if not map_qc.empty:
        write_table(map_qc, qc_root / "pair_similarity_searchlight_map_qc.tsv")

    summaries: dict[str, dict[str, object]] = {}
    peak_frames: list[pd.DataFrame] = []
    plot_payload: list[dict[str, object]] = []
    for idx, condition in enumerate(args.conditions):
        pre_paths = cell_paths.get(("pre", condition), [])
        post_paths = cell_paths.get(("post", condition), [])
        if not pre_paths or not post_paths:
            continue
        label = f"{condition}_post_vs_pre"
        group_dir = ensure_dir(qc_root / f"group_{label}")
        summary, maps = _compute_group_stats(
            post_paths,
            pre_paths,
            output_dir=group_dir,
            prefix=label,
            n_permutations=args.n_permutations,
            rng_seed=args.rng_seed + idx,
            permutation_backend=args.permutation_backend,
            permutation_batch_size=args.permutation_batch_size,
        )
        summaries[label] = summary
        peaks = _extract_peak_table(
            maps["t_map"],
            maps["p_fwe_map"],
            maps["affine"],
            comparison_label=label,
        )
        if not peaks.empty:
            peak_frames.append(peaks)
        plot_payload.append({"comparison": label, **maps})

    summary_df = pd.DataFrame([{"comparison": key, **value} for key, value in summaries.items()])
    if plot_payload:
        _plot_group_deltas(plot_payload, figures_main / "fig_searchlight_pair_similarity_delta.png")
    peaks_df = pd.concat(peak_frames, ignore_index=True) if peak_frames else pd.DataFrame(
        columns=["comparison", "i", "j", "k", "x_mm", "y_mm", "z_mm", "t_value", "abs_t", "p_fwe", "significant_fwe"]
    )
    write_table(peaks_df, tables_main / "table_searchlight_pair_similarity_peaks.tsv")
    write_table(summary_df, qc_root / "pair_similarity_searchlight_summary.tsv")
    save_json(
        {
            "pattern_root": str(pattern_root),
            "subject_mask_root": str(subject_mask_root),
            "conditions": list(args.conditions),
            "pair_col": args.pair_col,
            "min_pairs": int(args.min_pairs),
            "voxel_count": int(args.voxel_count),
            "n_permutations": int(args.n_permutations),
            "cell_counts": {f"{time}_{condition}": len(paths) for (time, condition), paths in cell_paths.items()},
            "n_skipped_cells": int((map_qc["status"] == "skipped").sum()) if not map_qc.empty else 0,
            "map_qc": str(qc_root / "pair_similarity_searchlight_map_qc.tsv"),
            "qc_root": str(qc_root),
            "tables_main": str(tables_main),
            "figures_main": str(figures_main),
            "comparisons": summaries,
        },
        qc_root / "pair_similarity_searchlight_summary.json",
    )


if __name__ == "__main__":
    main()
