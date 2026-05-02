"""
rd_searchlight.py

用途
- 在全脑范围内做 Representational Dimensionality (RD) 的 searchlight 分析。
- 输入为 Step 4 生成的 4D patterns（例如 `pre_yy.nii.gz`），输出为每个被试一张 3D RD map，
  并在组水平上做配对统计（post vs pre）。

输入
- pattern_root: `${PATTERN_ROOT}`（每个 sub-xx 一个目录，包含 `{time}_{condition}.nii.gz`）
- subject_mask_root: `${SUBJECT_MASK_ROOT}`（每个 sub-xx 一个目录，包含 mask 文件）
- output_dir: 输出根目录（默认 `${BASE_DIR}/paper_outputs`）
- --filename-template: 默认 `{time}_{condition}.nii.gz`
- --mask-filename: 默认 `mask.nii`（也会自动尝试 `.nii.gz`）

输出（output_dir）
- `qc/rd_searchlight/sub-xx/rd_{time}_{condition}.nii.gz`：个体 RD map
- `qc/rd_searchlight/group_*`：组水平统计 map 与摘要
- `tables_main/table_searchlight_peaks.tsv`
- `figures_main/fig_searchlight_delta.png`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _final_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "final_version":
            return parent
    return current.parent


FINAL_ROOT = _final_root()
if str(FINAL_ROOT) not in sys.path:
    sys.path.append(str(FINAL_ROOT))

from common.final_utils import ensure_dir, save_json, write_table
from common.pattern_metrics import compute_searchlight_dimension_map, save_scalar_map


def _default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def _default_pattern_root() -> Path:
    return _default_base_dir() / "pattern_root"


def _default_subject_mask_root() -> Path:
    return _default_base_dir() / "lss_betas_final"


def _default_output_root() -> Path:
    return _default_base_dir() / "paper_outputs"


def _paired_t_map(diff_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = stats.ttest_1samp(diff_stack, popmean=0.0, axis=0, nan_policy="omit")
    t_values = np.asarray(test.statistic, dtype=float)
    p_values = np.asarray(test.pvalue, dtype=float)
    mean_diff = np.nanmean(diff_stack, axis=0)
    return t_values, p_values, mean_diff


def _prepare_permutation_arrays(diff_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(diff_stack)
    data = np.where(valid, diff_stack, 0.0).astype(np.float32, copy=False)
    counts = valid.sum(axis=0).astype(np.float32, copy=False)
    sum_x2 = np.sum(data ** 2, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    return data, counts, sum_x2


def _compute_t_from_signed_sums(sum_x: np.ndarray, counts: np.ndarray, sum_x2: np.ndarray) -> np.ndarray:
    mean_vals = np.divide(sum_x, counts, out=np.zeros_like(sum_x, dtype=np.float32), where=counts > 0)
    denom = np.maximum(counts - 1.0, 1.0)
    var_vals = np.divide(sum_x2 - counts * (mean_vals ** 2), denom, out=np.zeros_like(mean_vals), where=counts > 1)
    var_vals = np.clip(var_vals, 1e-10, None)
    se = np.sqrt(var_vals / np.maximum(counts, 1.0))
    t_vals = np.divide(mean_vals, se, out=np.zeros_like(mean_vals), where=(counts > 1) & np.isfinite(se) & (se > 0))
    t_vals[~np.isfinite(t_vals)] = 0.0
    return t_vals


def _sign_flip_permutation_max_t_cpu(
    data: np.ndarray,
    counts: np.ndarray,
    sum_x2: np.ndarray,
    *,
    n_permutations: int,
    rng_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    max_t = np.zeros(n_permutations, dtype=float)
    n_subjects = data.shape[0]
    for perm_idx in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_subjects, 1))
        signed_sum = (data * signs).sum(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        t_vals = _compute_t_from_signed_sums(signed_sum, counts, sum_x2)
        max_t[perm_idx] = float(np.nanmax(np.abs(t_vals)))
    return max_t


def _sign_flip_permutation_max_t_gpu(
    data: np.ndarray,
    counts: np.ndarray,
    sum_x2: np.ndarray,
    *,
    n_permutations: int,
    rng_seed: int,
    batch_size: int,
) -> np.ndarray:
    if not HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("CUDA backend is not available")
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng_seed))

    data_t = torch.tensor(data, dtype=torch.float32, device=device)
    counts_t = torch.tensor(counts, dtype=torch.float32, device=device)
    sum_x2_t = torch.tensor(sum_x2, dtype=torch.float32, device=device)
    max_t_batches: list[np.ndarray] = []

    for start in range(0, n_permutations, batch_size):
        current = min(batch_size, n_permutations - start)
        signs = torch.randint(0, 2, (current, data.shape[0]), device=device, generator=gen, dtype=torch.int8)
        signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
        signed_sum = torch.matmul(signs, data_t)
        mean_vals = torch.divide(signed_sum, counts_t.unsqueeze(0))
        denom = torch.clamp(counts_t - 1.0, min=1.0).unsqueeze(0)
        var_vals = (sum_x2_t.unsqueeze(0) - counts_t.unsqueeze(0) * (mean_vals ** 2)) / denom
        var_vals = torch.clamp(var_vals, min=1e-10)
        se = torch.sqrt(var_vals / torch.clamp(counts_t.unsqueeze(0), min=1.0))
        valid_mask = (counts_t > 1).unsqueeze(0) & torch.isfinite(se) & (se > 0)
        t_vals = torch.zeros_like(mean_vals)
        t_vals[valid_mask] = mean_vals[valid_mask] / se[valid_mask]
        batch_max = torch.max(torch.abs(t_vals), dim=1).values.detach().cpu().numpy()
        max_t_batches.append(batch_max.astype(float, copy=False))

    del data_t, counts_t, sum_x2_t
    torch.cuda.empty_cache()
    return np.concatenate(max_t_batches, axis=0)


def _sign_flip_permutation_max_t(
    diff_stack: np.ndarray,
    *,
    n_permutations: int,
    rng_seed: int,
    backend: str,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    data, counts, sum_x2 = _prepare_permutation_arrays(diff_stack)
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "gpu" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    if resolved_backend == "gpu":
        try:
            return (
                _sign_flip_permutation_max_t_gpu(
                    data,
                    counts,
                    sum_x2,
                    n_permutations=n_permutations,
                    rng_seed=rng_seed,
                    batch_size=batch_size,
                ),
                "gpu",
            )
        except Exception:
            resolved_backend = "cpu"
    return (
        _sign_flip_permutation_max_t_cpu(
            data,
            counts,
            sum_x2,
            n_permutations=n_permutations,
            rng_seed=rng_seed,
        ),
        "cpu",
    )


def _compute_group_stats(
    map_a_paths: list[Path],
    map_b_paths: list[Path],
    *,
    output_dir: Path,
    prefix: str,
    n_permutations: int,
    rng_seed: int,
    permutation_backend: str,
    permutation_batch_size: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    output_dir = ensure_dir(output_dir)
    if len(map_a_paths) != len(map_b_paths):
        raise ValueError(f"{prefix}: map counts do not match")
    if not map_a_paths:
        raise ValueError(f"{prefix}: no subject maps were found")

    first_img = nib.load(str(map_a_paths[0]))
    shape = first_img.shape
    a_stack = np.vstack([np.asarray(nib.load(str(path)).get_fdata()).reshape(-1) for path in map_a_paths])
    b_stack = np.vstack([np.asarray(nib.load(str(path)).get_fdata()).reshape(-1) for path in map_b_paths])
    # Subject-level searchlight maps are written as zeros outside the subject mask.
    # Restrict group inference to voxels with at least one non-zero finite input
    # across pre/post maps; otherwise outside-mask zeros and NaNs can dominate the
    # max-T null distribution and make FWE maps/peaks meaningless.
    analysis_mask = (
        np.isfinite(a_stack).all(axis=0)
        & np.isfinite(b_stack).all(axis=0)
        & (np.any(np.abs(a_stack) > 0, axis=0) | np.any(np.abs(b_stack) > 0, axis=0))
    )
    if not np.any(analysis_mask):
        raise ValueError(f"{prefix}: no valid searchlight voxels survived group masking")

    diff_stack_valid = a_stack[:, analysis_mask] - b_stack[:, analysis_mask]
    t_valid, p_valid, mean_diff_valid = _paired_t_map(diff_stack_valid)
    finite_stat_mask = np.isfinite(t_valid)
    if not np.any(finite_stat_mask):
        raise ValueError(f"{prefix}: no finite group statistics were produced")

    diff_for_permutation = diff_stack_valid[:, finite_stat_mask]
    max_t, resolved_backend = _sign_flip_permutation_max_t(
        diff_for_permutation,
        n_permutations=n_permutations,
        rng_seed=rng_seed,
        backend=permutation_backend,
        batch_size=permutation_batch_size,
    )
    p_fwe_valid_finite = (
        np.sum(max_t[:, None] >= np.abs(t_valid[finite_stat_mask])[None, :], axis=0) + 1.0
    ) / (len(max_t) + 1.0)

    t_values = np.full(a_stack.shape[1], np.nan, dtype=float)
    p_values = np.full(a_stack.shape[1], np.nan, dtype=float)
    mean_diff = np.full(a_stack.shape[1], np.nan, dtype=float)
    p_fwe = np.full(a_stack.shape[1], np.nan, dtype=float)

    valid_indices = np.flatnonzero(analysis_mask)
    t_values[valid_indices] = t_valid
    p_values[valid_indices] = p_valid
    mean_diff[valid_indices] = mean_diff_valid
    p_fwe[valid_indices[finite_stat_mask]] = p_fwe_valid_finite
    log_p_fwe = -np.log10(np.clip(p_fwe, 1e-12, 1.0))

    outputs = {
        f"{prefix}_t_map.nii.gz": t_values,
        f"{prefix}_p_map.nii.gz": p_values,
        f"{prefix}_mean_diff_map.nii.gz": mean_diff,
        f"{prefix}_p_fwe_map.nii.gz": p_fwe,
        f"{prefix}_logp_fwe_map.nii.gz": log_p_fwe,
    }
    for name, values in outputs.items():
        nib.save(
            nib.Nifti1Image(values.reshape(shape).astype(np.float32), first_img.affine, first_img.header),
            str(output_dir / name),
        )

    summary = {
        "n_subjects": int(len(map_a_paths)),
        "n_voxels_tested": int(np.isfinite(p_fwe).sum()),
        "n_permutations": int(n_permutations),
        "permutation_backend": resolved_backend,
        "t_map": str(output_dir / f"{prefix}_t_map.nii.gz"),
        "p_map": str(output_dir / f"{prefix}_p_map.nii.gz"),
        "mean_diff_map": str(output_dir / f"{prefix}_mean_diff_map.nii.gz"),
        "p_fwe_map": str(output_dir / f"{prefix}_p_fwe_map.nii.gz"),
        "logp_fwe_map": str(output_dir / f"{prefix}_logp_fwe_map.nii.gz"),
    }
    maps = {
        "t_map": t_values.reshape(shape),
        "mean_diff_map": mean_diff.reshape(shape),
        "p_fwe_map": p_fwe.reshape(shape),
        "affine": first_img.affine.copy(),
    }
    return summary, maps


def _voxel_to_world(affine: np.ndarray, ijk: tuple[int, int, int]) -> tuple[float, float, float]:
    xyz = nib.affines.apply_affine(affine, np.asarray(ijk, dtype=float))
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _extract_peak_table(
    t_map: np.ndarray,
    p_fwe: np.ndarray,
    affine: np.ndarray,
    comparison_label: str,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    abs_t = np.abs(t_map)
    if np.all(~np.isfinite(abs_t)):
        return pd.DataFrame()

    candidate_mask = np.isfinite(abs_t)
    local_max = ndimage.maximum_filter(abs_t, size=3, mode="constant", cval=np.nanmin(abs_t))
    is_peak = candidate_mask & np.isclose(abs_t, local_max, equal_nan=False)
    peak_indices = np.argwhere(is_peak)
    if peak_indices.size == 0:
        peak_indices = np.argwhere(candidate_mask)

    rows = []
    for idx in peak_indices:
        ijk = tuple(int(v) for v in idx.tolist())
        x_mm, y_mm, z_mm = _voxel_to_world(affine, ijk)
        rows.append(
            {
                "comparison": comparison_label,
                "i": ijk[0],
                "j": ijk[1],
                "k": ijk[2],
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
                "t_value": float(t_map[ijk]),
                "abs_t": float(abs_t[ijk]),
                "p_fwe": float(p_fwe[ijk]),
                "significant_fwe": bool(np.isfinite(p_fwe[ijk]) and p_fwe[ijk] < 0.05),
            }
        )
    peaks = pd.DataFrame(rows).sort_values(["significant_fwe", "abs_t"], ascending=[False, False]).head(top_n)
    return peaks.reset_index(drop=True)


def _signed_mid_slice(volume: np.ndarray) -> np.ndarray:
    return volume[:, :, volume.shape[2] // 2]


def _plot_group_deltas(plot_payload: list[dict[str, object]], figure_path: Path) -> None:
    if not plot_payload:
        return
    n_panels = len(plot_payload)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5))
    axes = np.atleast_1d(axes)
    vmax = 0.0
    slices = []
    masks = []
    for row in plot_payload:
        diff_map = np.asarray(row["mean_diff_map"], dtype=float)
        p_fwe_map = np.asarray(row["p_fwe_map"], dtype=float)
        slice_img = _signed_mid_slice(diff_map)
        sig_mask = _signed_mid_slice(p_fwe_map) < 0.05
        slices.append(slice_img)
        masks.append(sig_mask)
        vmax = max(vmax, float(np.nanmax(np.abs(slice_img))))
    vmax = vmax if vmax > 0 else 1.0
    for ax, row, slice_img, sig_mask in zip(axes, plot_payload, slices, masks):
        im = ax.imshow(np.rot90(slice_img), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        if np.any(sig_mask):
            ax.contour(np.rot90(sig_mask.astype(float)), levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(row["comparison"], fontsize=11)
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Post - Pre RD")
    fig.tight_layout()
    ensure_dir(figure_path.parent)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_subject_mask(subject_mask_root: Path, subject: str, mask_filename: str) -> Path | None:
    subject_root = subject_mask_root / subject
    requested = subject_root / mask_filename
    candidates = [requested]
    if requested.suffix == ".nii":
        candidates.append(requested.with_suffix(".nii.gz"))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    mask_name = Path(mask_filename).name
    nested_candidates = sorted(subject_root.glob(f"*/{mask_name}"))
    if Path(mask_name).suffix == ".nii":
        nested_candidates.extend(sorted(subject_root.glob(f"*/{Path(mask_name).with_suffix('.nii.gz').name}")))
    for candidate in nested_candidates:
        if candidate.exists():
            return candidate
    return None


def compute_cell_maps(
    subject_dirs,
    subject_mask_root: Path,
    output_dir: Path,
    time: str,
    condition: str,
    filename_template: str,
    explained_threshold: float,
    voxel_count: int,
    mask_filename: str = "mask.nii",
):
    paths = []
    qc_rows = []
    for subject_dir in subject_dirs:
        image_path = subject_dir / filename_template.format(time=time, condition=condition)
        subject_mask = _resolve_subject_mask(subject_mask_root, subject_dir.name, mask_filename)
        if not image_path.exists():
            qc_rows.append({
                "subject": subject_dir.name,
                "time": time,
                "condition": condition,
                "image_path": str(image_path),
                "mask_path": str(subject_mask) if subject_mask else "",
                "status": "skipped",
                "skip_reason": "missing_pattern",
                "output_path": "",
            })
            continue
        if subject_mask is None:
            qc_rows.append({
                "subject": subject_dir.name,
                "time": time,
                "condition": condition,
                "image_path": str(image_path),
                "mask_path": "",
                "status": "skipped",
                "skip_reason": "missing_mask",
                "output_path": "",
            })
            continue
        reference_img, mask, values = compute_searchlight_dimension_map(image_path, subject_mask, explained_threshold, voxel_count)
        subject_output = ensure_dir(output_dir / subject_dir.name)
        out_path = subject_output / f"rd_{time}_{condition}.nii.gz"
        save_scalar_map(reference_img, mask, values, out_path)
        paths.append(out_path)
        qc_rows.append({
            "subject": subject_dir.name,
            "time": time,
            "condition": condition,
            "image_path": str(image_path),
            "mask_path": str(subject_mask),
            "status": "ok",
            "skip_reason": "",
            "output_path": str(out_path),
        })
    return paths, qc_rows


def _reuse_cell_maps_from_qc(qc_path: Path, conditions: list[str]) -> tuple[dict[tuple[str, str], list[Path]], pd.DataFrame]:
    if not qc_path.exists():
        raise FileNotFoundError(
            f"Cannot reuse subject maps because QC table was not found: {qc_path}. "
            "Run once without --reuse-subject-maps to generate subject-level maps."
        )
    map_qc = pd.read_csv(qc_path, sep="\t")
    required_columns = {"time", "condition", "status", "output_path"}
    missing = required_columns.difference(map_qc.columns)
    if missing:
        raise ValueError(f"QC table {qc_path} is missing required columns: {sorted(missing)}")

    cell_paths: dict[tuple[str, str], list[Path]] = {}
    for time in ["pre", "post"]:
        for condition in conditions:
            rows = map_qc[
                (map_qc["time"] == time)
                & (map_qc["condition"] == condition)
                & (map_qc["status"] == "ok")
            ]
            paths = [Path(path) for path in rows["output_path"].dropna().astype(str) if Path(path).exists()]
            missing_count = int(len(rows) - len(paths))
            if missing_count:
                print(
                    f"[reuse-subject-maps] {time}/{condition}: {missing_count} QC-listed map(s) are missing on disk; "
                    "they will be excluded."
                )
            cell_paths[(time, condition)] = paths
    return cell_paths, map_qc


def _subject_id_from_map_path(path: Path) -> str:
    for part in reversed(path.parts):
        if str(part).startswith("sub-"):
            return str(part)
    stem = path.name
    for token in stem.replace("\\", "/").split("/"):
        if token.startswith("sub-"):
            return token
    raise ValueError(f"Could not infer subject id from map path: {path}")


def _align_pre_post_subject_maps(
    pre_paths: list[Path],
    post_paths: list[Path],
    *,
    condition: str,
    analysis_name: str,
) -> tuple[list[Path], list[Path], pd.DataFrame]:
    pre_by_subject = {_subject_id_from_map_path(path): path for path in pre_paths}
    post_by_subject = {_subject_id_from_map_path(path): path for path in post_paths}
    if len(pre_by_subject) != len(pre_paths):
        raise ValueError(f"{analysis_name}/{condition}: duplicate subject ids in pre maps.")
    if len(post_by_subject) != len(post_paths):
        raise ValueError(f"{analysis_name}/{condition}: duplicate subject ids in post maps.")
    pre_subjects = set(pre_by_subject)
    post_subjects = set(post_by_subject)
    paired_subjects = sorted(pre_subjects & post_subjects)
    missing_pre = sorted(post_subjects - pre_subjects)
    missing_post = sorted(pre_subjects - post_subjects)
    rows = []
    for subject in sorted(pre_subjects | post_subjects):
        rows.append(
            {
                "analysis": analysis_name,
                "condition": condition,
                "subject": subject,
                "has_pre": bool(subject in pre_by_subject),
                "has_post": bool(subject in post_by_subject),
                "paired": bool(subject in paired_subjects),
                "pre_path": str(pre_by_subject.get(subject, "")),
                "post_path": str(post_by_subject.get(subject, "")),
            }
        )
    manifest = pd.DataFrame(rows)
    if missing_pre or missing_post:
        raise ValueError(
            f"{analysis_name}/{condition}: pre/post subject sets differ. "
            f"missing_pre={missing_pre}; missing_post={missing_post}"
        )
    pre_aligned = [pre_by_subject[subject] for subject in paired_subjects]
    post_aligned = [post_by_subject[subject] for subject in paired_subjects]
    return pre_aligned, post_aligned, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-brain RD searchlight analysis.")
    parser.add_argument("pattern_root", type=Path, nargs="?", default=None)
    parser.add_argument("subject_mask_root", type=Path, nargs="?", default=None)
    parser.add_argument("output_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--filename-template", default="{time}_{condition}.nii.gz")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--voxel-count", type=int, default=100)
    parser.add_argument("--conditions", nargs="+", default=["yy", "kj", "baseline"])
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--permutation-backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--permutation-batch-size", type=int, default=256)
    parser.add_argument(
        "--mask-filename",
        default="mask.nii",
        help="Mask filename under each subject folder (default: mask.nii; will also try .nii.gz).",
    )
    parser.add_argument(
        "--reuse-subject-maps",
        action="store_true",
        help="Reuse existing subject-level RD maps from qc/rd_searchlight/rd_searchlight_map_qc.tsv and rerun only group-level statistics.",
    )
    args = parser.parse_args()

    pattern_root = args.pattern_root or _default_pattern_root()
    subject_mask_root = args.subject_mask_root or _default_subject_mask_root()
    paper_output_root = ensure_dir(args.output_dir or _default_output_root())
    qc_root = ensure_dir(paper_output_root / "qc" / "rd_searchlight")
    tables_main = ensure_dir(paper_output_root / "tables_main")
    figures_main = ensure_dir(paper_output_root / "figures_main")

    if args.reuse_subject_maps:
        cell_paths, map_qc = _reuse_cell_maps_from_qc(
            qc_root / "rd_searchlight_map_qc.tsv",
            list(args.conditions),
        )
    else:
        subject_dirs = sorted([path for path in pattern_root.iterdir() if path.is_dir() and path.name.startswith("sub-")])
        cell_paths = {}
        map_qc_rows = []
        for time in ["pre", "post"]:
            for condition in args.conditions:
                paths, qc_rows = compute_cell_maps(
                    subject_dirs,
                    subject_mask_root,
                    qc_root,
                    time,
                    condition,
                    args.filename_template,
                    args.threshold,
                    args.voxel_count,
                    mask_filename=args.mask_filename,
                )
                cell_paths[(time, condition)] = paths
                map_qc_rows.extend(qc_rows)
        map_qc = pd.DataFrame(map_qc_rows)
        if not map_qc.empty:
            write_table(map_qc, qc_root / "rd_searchlight_map_qc.tsv")

    summaries: dict[str, dict[str, object]] = {}
    peak_frames: list[pd.DataFrame] = []
    plot_payload: list[dict[str, object]] = []
    pairing_manifests: list[pd.DataFrame] = []
    for idx, condition in enumerate(args.conditions):
        pre_paths = cell_paths.get(("pre", condition), [])
        post_paths = cell_paths.get(("post", condition), [])
        if not pre_paths or not post_paths:
            continue
        pre_paths, post_paths, pairing_manifest = _align_pre_post_subject_maps(
            pre_paths,
            post_paths,
            condition=condition,
            analysis_name="rd_searchlight",
        )
        pairing_manifests.append(pairing_manifest)
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

    summary_df = pd.DataFrame(
        [
            {"comparison": key, **value}
            for key, value in summaries.items()
        ]
    )
    if pairing_manifests:
        write_table(pd.concat(pairing_manifests, ignore_index=True), qc_root / "searchlight_subject_pairing_manifest.tsv")
    if plot_payload:
        _plot_group_deltas(plot_payload, figures_main / "fig_searchlight_delta.png")
    peaks_df = pd.concat(peak_frames, ignore_index=True) if peak_frames else pd.DataFrame(
        columns=["comparison", "i", "j", "k", "x_mm", "y_mm", "z_mm", "t_value", "abs_t", "p_fwe", "significant_fwe"]
    )
    write_table(peaks_df, tables_main / "table_searchlight_peaks.tsv")
    write_table(summary_df, qc_root / "rd_searchlight_summary.tsv")
    save_json(
        {
            "pattern_root": str(pattern_root),
            "subject_mask_root": str(subject_mask_root),
            "conditions": list(args.conditions),
            "n_permutations": int(args.n_permutations),
            "reuse_subject_maps": bool(args.reuse_subject_maps),
            "cell_counts": {f"{time}_{condition}": len(paths) for (time, condition), paths in cell_paths.items()},
            "n_skipped_cells": int((map_qc["status"] == "skipped").sum()) if not map_qc.empty else 0,
            "map_qc": str(qc_root / "rd_searchlight_map_qc.tsv"),
            "qc_root": str(qc_root),
            "tables_main": str(tables_main),
            "figures_main": str(figures_main),
            "comparisons": summaries,
        },
        qc_root / "rd_searchlight_summary.json",
    )


if __name__ == "__main__":
    main()
