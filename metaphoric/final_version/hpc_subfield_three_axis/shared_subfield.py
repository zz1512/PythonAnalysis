#!/usr/bin/env python3
"""海马亚区切分三轴分析的共享工具。

Safety policy:
- 只读上游 tsv / nii；
- 只在 paper_outputs/qc/hpc_subfield_three_axis 下写入；
- 禁止覆盖既有文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FINAL_ROOT = Path(__file__).resolve().parents[1]
NC_CONVERGE = FINAL_ROOT / "nc_converge"
if str(NC_CONVERGE) not in sys.path:
    sys.path.insert(0, str(NC_CONVERGE))

try:
    import shared_nc  # type: ignore
except Exception:  # pragma: no cover
    shared_nc = None

SAFE_RELATIVE_ROOT = Path("paper_outputs") / "qc" / "hpc_subfield_three_axis"
FDR_FAMILY_LABEL = "hpc_subfield_three_axis"
SEGMENT_ORDER = ("head", "body", "tail")
AXIS_POSITION = {"head": 1, "body": 0, "tail": -1}
HEMISPHERES = ("L", "R")


@dataclass(frozen=True)
class SubfieldConfig:
    base_dir: Path
    paper_output_root: Path
    output_root: Path
    final_root: Path
    fs60_root: Path | None
    mask_root: Path | None


def default_base_dir() -> Path:
    return Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    base = default_base_dir()
    parser.add_argument("--base-dir", type=Path, default=base)
    parser.add_argument("--paper-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--fs60-root", type=Path, default=None,
                        help="FreeSurfer FS60 hipposubfields 根目录（每被试一个子目录）。")
    parser.add_argument("--mask-root", type=Path, default=None,
                        help="既有 meta_L/R_hippocampus mask 根目录；FS60 缺失时用于 MNI y 轴三等分 fallback。")
    parser.add_argument("--allow-empty", action="store_true",
                        help="允许在上游输入缺失时写出空审计表，而非直接报错。")


def default_config(args: argparse.Namespace) -> SubfieldConfig:
    base_dir = Path(args.base_dir)
    paper_output_root = Path(args.paper_output_root or base_dir / "paper_outputs")
    output_root = Path(args.output_root or paper_output_root / "qc" / "hpc_subfield_three_axis")
    fs60_root = Path(args.fs60_root) if args.fs60_root is not None else None
    mask_root = Path(args.mask_root) if args.mask_root is not None else (base_dir / "roi_library" / "meta_sources")
    cfg = SubfieldConfig(
        base_dir=base_dir,
        paper_output_root=paper_output_root,
        output_root=output_root,
        final_root=Path(args.final_root),
        fs60_root=fs60_root,
        mask_root=mask_root,
    )
    ensure_root(cfg)
    return cfg


def resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def ensure_root(cfg: SubfieldConfig) -> Path:
    root = resolved(cfg.output_root)
    expected = resolved(cfg.paper_output_root / "qc" / "hpc_subfield_three_axis")
    if root != expected and expected not in root.parents:
        raise ValueError(
            f"Refusing output outside hpc_subfield_three_axis sandbox: {root}; expected under {expected}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def module_dir(cfg: SubfieldConfig, module_name: str) -> Path:
    target = resolved(cfg.output_root / module_name)
    sandbox = resolved(cfg.output_root)
    if target != sandbox and sandbox not in target.parents:
        raise ValueError(f"Refusing module output outside sandbox: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_output_path(cfg: SubfieldConfig, module_name: str, filename: str) -> Path:
    directory = module_dir(cfg, module_name)
    path = resolved(directory / filename)
    if directory not in path.parents:
        raise ValueError(f"Refusing path outside module directory: {path}")
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    return path


# ---------- 表格 IO（沿用 shared_nc，避免重复实现） ----------

def read_table(path: Path) -> pd.DataFrame:
    if shared_nc is not None:
        return shared_nc.read_table(Path(path))
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame.to_csv(path, sep=sep, index=False)


def write_text(text: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_outputs(cfg: SubfieldConfig, module_name: str, outputs: dict) -> list[str]:
    written: list[str] = []
    for name, payload in outputs.items():
        path = safe_output_path(cfg, module_name, name)
        if isinstance(payload, pd.DataFrame):
            write_table(payload, path)
        elif isinstance(payload, (dict,)):
            if shared_nc is not None:
                shared_nc.save_json(payload, path)
            else:
                import json
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            write_text(str(payload), path)
        written.append(str(path))
    return written


# ---------- 统计工具（FDR / mixed model）沿用 shared_nc ----------

def bh_fdr(pvalues: Sequence[float]) -> np.ndarray:
    if shared_nc is not None:
        return shared_nc.bh_fdr(pvalues)
    raise RuntimeError("shared_nc 不可用，无法执行 BH-FDR。请确认 nc_converge 模块存在。")


def fit_formula(frame: pd.DataFrame, formula: str, *, family: str = "gaussian",
                group_col: str | None = "subject", item_col: str | None = "condition_item_id") -> pd.DataFrame:
    if shared_nc is None:
        raise RuntimeError("shared_nc 不可用，无法执行 mixed model。")
    return shared_nc.fit_formula(frame, formula, family=family, group_col=group_col, item_col=item_col)


def build_condition_item_id(frame: pd.DataFrame) -> pd.DataFrame:
    if shared_nc is None:
        raise RuntimeError("shared_nc 不可用，无法生成 condition_item_id。")
    return shared_nc.build_condition_item_id(frame)


def markdown_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    if shared_nc is None:
        return "_shared_nc 不可用，无法渲染 markdown 表_"
    return shared_nc.markdown_table(frame, index=index)


def sha256(path: Path) -> str:
    if shared_nc is None:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return shared_nc.sha256(path)


def git_status(path: Path) -> str:
    if shared_nc is None:
        return "git_status_unavailable"
    return shared_nc.git_status(path)


# ---------- 亚区命名与结构工具 ----------

def assemble_subfield_label(hemisphere: str, segment: str) -> str:
    hemi = str(hemisphere).upper()
    if hemi not in HEMISPHERES:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    seg = str(segment).lower()
    if seg not in SEGMENT_ORDER:
        raise ValueError(f"Unknown segment: {segment}")
    return f"hpc_{hemi}_{seg}"


def iter_subfield_labels() -> list[str]:
    return [assemble_subfield_label(h, s) for h in HEMISPHERES for s in SEGMENT_ORDER]


def parse_subfield_label(label: str) -> tuple[str, str]:
    parts = label.split("_")
    if len(parts) != 3 or parts[0] != "hpc":
        raise ValueError(f"Bad subfield label: {label}")
    return parts[1], parts[2]


# ---------- mask split（MNI y 轴三等分 fallback） ----------

def split_mask_along_y(mask_nii_path: Path, n_segments: int = 3) -> list[dict]:
    """按 MNI y 坐标将 mask 体素分为 n_segments 段。

    返回列表 [{segment, voxel_indices (tuple of arrays), n_voxels, y_range}, ...]。
    不写盘；仅返回体素索引信息，供上游模块决定写 nii。
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("需要 nibabel 才能做 MNI y 轴三等分切分。") from exc
    img = nib.load(str(mask_nii_path))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Mask 不是 3D：{mask_nii_path} shape={data.shape}")
    voxel_idx = np.argwhere(data > 0)
    if voxel_idx.size == 0:
        return []
    # 将体素坐标转到 MNI 空间（mm），按 y 轴分位
    affine = img.affine
    mni = nib.affines.apply_affine(affine, voxel_idx)
    y_vals = mni[:, 1]
    # 按从后（y 小）到前（y 大）的分位切分：tail → body → head
    thresholds = np.quantile(y_vals, [1.0 / n_segments, 2.0 / n_segments])
    segments = []
    for seg_name, mask_bool in [
        ("tail", y_vals <= thresholds[0]),
        ("body", (y_vals > thresholds[0]) & (y_vals <= thresholds[1])),
        ("head", y_vals > thresholds[1]),
    ]:
        idx = voxel_idx[mask_bool]
        y_sub = y_vals[mask_bool]
        segments.append({
            "segment": seg_name,
            "voxel_ijk": idx,
            "n_voxels": int(idx.shape[0]),
            "y_min": float(y_sub.min()) if y_sub.size else float("nan"),
            "y_max": float(y_sub.max()) if y_sub.size else float("nan"),
            "affine": affine,
            "shape": data.shape,
        })
    return segments


def save_segment_mask(segment_info: dict, out_path: Path) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("需要 nibabel 才能写 mask nii。") from exc
    if out_path.exists():
        raise FileExistsError(f"Output exists; refusing to overwrite: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shape = segment_info["shape"]
    affine = segment_info["affine"]
    data = np.zeros(shape, dtype=np.int16)
    for i, j, k in segment_info["voxel_ijk"]:
        data[int(i), int(j), int(k)] = 1
    nib.Nifti1Image(data, affine).to_filename(str(out_path))


# ---------- 日志 ----------

def log_text(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines)
    return f"# {title}\n\n{body}\n"


def discover_subjects(root: Path, pattern: str = "sub-*") -> list[str]:
    if not root or not Path(root).exists():
        return []
    subs = sorted(p.name for p in Path(root).glob(pattern) if p.is_dir())
    return subs
