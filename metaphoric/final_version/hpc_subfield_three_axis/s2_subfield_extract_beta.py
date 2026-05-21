#!/usr/bin/env python3
"""S2: 在 6 个 subROI 上提取 LSS beta（multi-voxel pattern）。

输入：
    S1 输出的 subfield_manifest.tsv 与 mask_path；
    既有 derivatives/<sub>/lss/<phase>/beta_*.nii.gz（不重跑 LSS GLM）。

输出：
    s2_beta_extract/beta_long.tsv   每行一个 (subject, run_phase, condition, item_id, subROI)
                                    带 beta_vector_path（NPZ 路径）
    s2_beta_extract/beta_qc.tsv     每 subROI 的体素均值/方差/零方差体素数
    s2_beta_extract/s2_log.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

MODULE = "s2_beta_extract"
PHASES = ("pre", "learning", "post", "retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="S1 输出的 subfield_manifest.tsv；缺省时自动定位。")
    parser.add_argument("--lss-root", type=Path, default=None,
                        help="LSS beta 根目录，默认 base-dir/derivatives。")
    parser.add_argument("--phases", nargs="*", default=list(PHASES))
    parser.add_argument("--beta-pattern", default="{subject}/lss/{phase}/beta_{item_id}.nii.gz",
                        help="LSS beta 路径模板（相对 lss-root），占位符 {subject}/{phase}/{item_id}。")
    parser.add_argument("--items-table", type=Path, default=None,
                        help="包含 subject/run_phase/condition/item_id 的长表（缺省时报错）。")
    return parser.parse_args()


def load_manifest(cfg, args: argparse.Namespace) -> pd.DataFrame:
    path = args.manifest or (cfg.output_root / "s1_segmentation" / "subfield_manifest.tsv")
    if not Path(path).exists():
        if args.allow_empty:
            return pd.DataFrame()
        raise FileNotFoundError(f"S1 manifest 不存在：{path}")
    return read_table(Path(path))


def load_items(cfg, args: argparse.Namespace) -> pd.DataFrame:
    if args.items_table is not None and Path(args.items_table).exists():
        return read_table(Path(args.items_table))
    candidate = cfg.output_root / "p0_inputs" / "hpc_items.tsv"
    if candidate.exists():
        return read_table(candidate)
    if args.allow_empty:
        return pd.DataFrame()
    raise FileNotFoundError(
        "items-table 必须显式提供（含 subject / run_phase / condition / item_id）。"
        " 在数据机上指向 nc_converge/a2_schema_lock 已审计的 long-form 表。"
    )


def extract_subject_subfield(beta_path: Path, mask_path: Path) -> np.ndarray | None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("需要 nibabel 才能提取 beta。") from exc
    if not beta_path.exists() or not mask_path.exists():
        return None
    beta_img = nib.load(str(beta_path))
    mask_img = nib.load(str(mask_path))
    beta = np.asarray(beta_img.dataobj)
    mask = np.asarray(mask_img.dataobj) > 0
    if beta.shape != mask.shape:
        raise ValueError(f"shape mismatch: beta={beta.shape} mask={mask.shape} ({beta_path}; {mask_path})")
    return beta[mask].astype(float)


def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    manifest = load_manifest(cfg, args)
    items = load_items(cfg, args)
    lss_root = Path(args.lss_root or cfg.base_dir / "derivatives")

    long_rows: list[dict] = []
    qc_rows: list[dict] = []
    log_lines: list[str] = []

    vectors_dir = module_dir(cfg, MODULE) / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    if manifest.empty or items.empty:
        log_lines.append("manifest_or_items_empty=skip_extraction")
    else:
        for sub in sorted(manifest["subject"].unique()):
            sub_manifest = manifest[manifest["subject"] == sub]
            sub_items = items[items["subject"] == sub] if "subject" in items.columns else items
            for _, m_row in sub_manifest.iterrows():
                mask_path = Path(m_row["mask_path"])
                vox_means: list[float] = []
                vox_vars: list[float] = []
                extracted_vectors: list[np.ndarray] = []
                for _, it_row in sub_items.iterrows():
                    phase = str(it_row.get("run_phase") or it_row.get("phase") or "")
                    if phase not in args.phases:
                        continue
                    item_id = str(it_row.get("item_id") or it_row.get("condition_item_id") or "")
                    direct_beta = it_row.get("beta_path")
                    beta_path = Path(str(direct_beta)) if direct_beta and str(direct_beta).lower() != "nan" else None
                    if beta_path is None or not beta_path.exists():
                        beta_rel = args.beta_pattern.format(subject=sub, phase=phase, item_id=item_id)
                        beta_path = lss_root / beta_rel
                    vec = extract_subject_subfield(beta_path, mask_path)
                    if vec is None:
                        log_lines.append(
                            f"missing beta sub={sub} phase={phase} item={item_id} -> {beta_path}"
                        )
                        continue
                    npz_path = vectors_dir / f"{sub}__{m_row['subROI']}__{phase}__{item_id}.npz"
                    if npz_path.exists():
                        raise FileExistsError(f"Output exists; refusing to overwrite: {npz_path}")
                    np.savez_compressed(npz_path, beta=vec)
                    long_rows.append({
                        "subject": sub,
                        "run_phase": phase,
                        "condition": str(it_row.get("condition") or ""),
                        "item_id": item_id,
                        "subROI": m_row["subROI"],
                        "n_voxels": int(vec.shape[0]),
                        "beta_vector_path": str(npz_path),
                    })
                    vox_means.append(float(np.nanmean(vec)))
                    vox_vars.append(float(np.nanvar(vec, ddof=1)) if vec.size > 1 else 0.0)
                    extracted_vectors.append(vec)
                zero_var_count = float("nan")
                if extracted_vectors:
                    lengths = {int(vec.shape[0]) for vec in extracted_vectors}
                    if len(lengths) == 1:
                        stacked = np.vstack(extracted_vectors)
                        voxel_var = np.nanvar(stacked, axis=0, ddof=0)
                        zero_var_count = int(np.sum(np.isclose(voxel_var, 0.0, equal_nan=False)))
                    else:
                        log_lines.append(
                            f"vector_length_mismatch sub={sub} subROI={m_row['subROI']} lengths={sorted(lengths)}"
                        )
                qc_rows.append({
                    "subject": sub,
                    "subROI": m_row["subROI"],
                    "n_items_extracted": len([r for r in long_rows if r["subject"] == sub and r["subROI"] == m_row["subROI"]]),
                    "mean_of_voxel_means": float(np.nanmean(vox_means)) if vox_means else float("nan"),
                    "mean_of_voxel_vars": float(np.nanmean(vox_vars)) if vox_vars else float("nan"),
                    "zero_variance_voxels": zero_var_count,
                })

    long_frame = pd.DataFrame(long_rows, columns=[
        "subject", "run_phase", "condition", "item_id", "subROI", "n_voxels", "beta_vector_path",
    ])
    qc_frame = pd.DataFrame(qc_rows, columns=[
        "subject", "subROI", "n_items_extracted", "mean_of_voxel_means",
        "mean_of_voxel_vars", "zero_variance_voxels",
    ])
    write_outputs(cfg, MODULE, {
        "beta_long.tsv": long_frame,
        "beta_qc.tsv": qc_frame,
        "s2_log.txt": log_text("S2 beta extract log", log_lines),
    })


if __name__ == "__main__":
    main()
