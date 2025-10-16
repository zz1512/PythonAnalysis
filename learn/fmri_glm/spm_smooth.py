"""
SPM-style smoothing (FWHM=6mm, prefix='smooth') for batch subjects & runs.

- Root layout:
  ../../Pro_proc_data/sub-XX/runY/sub-*.nii[.gz]

- Output:
  Same folder, filename prefixed with 'smooth'
"""

from pathlib import Path
from typing import Iterable, Optional, List
import numpy as np
import nibabel as nib
from nilearn import image

# ================ 配置区 =================
ROOT = Path(r"../../Pro_proc_data")

# 处理哪些 subject：
#   1) SUBJECTS=None  → 自动发现 ROOT 下所有 sub-*
#   2) SUBJECTS=["sub-01","sub-28"] → 仅这些
SUBJECTS: Optional[List[str]] = ["sub-01"]

# 要处理哪些 run（目录名），可随意改动/删减
RUNS: List[str] = ["run1"]

# 输入文件命名模式（按你数据命名来）
GLOB_PATTERNS = ["sub-*.nii", "sub-*.nii.gz"]

# 平滑参数
FWHM_MM = 6.0           # 等价 SPM [6 6 6]
OVERWRITE = True       # 已存在 smooth* 是否覆盖
NUM_WORKERS = 1         # >1 开并行（Windows 建议小心使用）

# =======================================

def add_prefix_to_filename(path: Path, prefix: str) -> Path:
    """给文件名加前缀，保持扩展名正确（支持 .nii.gz）"""
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
        return path.with_name(prefix + stem + ".nii.gz")
    else:
        stem = path.stem
        return path.with_name(prefix + stem + path.suffix)

def ensure_float32(img: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """平滑后以 float32 保存，避免整数类型造成的信息丢失。"""
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    out = nib.Nifti1Image(data, img.affine, img.header)
    # 确保 header 的数据类型也标记为 float32
    out.set_data_dtype(np.float32)
    return out

def list_subject_dirs(root: Path, subjects: Optional[Iterable[str]]) -> List[Path]:
    if subjects:
        return [root / s for s in subjects if (root / s).is_dir()]
    return [p for p in sorted(root.glob("sub-*")) if p.is_dir()]

def list_run_files(sub_dir: Path, run: str) -> List[Path]:
    run_dir = sub_dir / run
    if not run_dir.exists():
        return []
    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        files.extend(sorted(run_dir.glob(pat)))
    return files

def smooth_one_file(in_path: Path, fwhm: float, overwrite: bool=False) -> Path:
    out_path = add_prefix_to_filename(in_path, "smooth")
    if out_path.exists() and not overwrite:
        print(f"[Skip] Exists: {out_path}")
        return out_path
    print(f"[Smoothing] {in_path}  ->  {out_path}  (FWHM={fwhm}mm)")
    smoothed = image.smooth_img(str(in_path), fwhm)
    smoothed = ensure_float32(smoothed)
    nib.save(smoothed, str(out_path))
    return out_path

def main():
    subs = list_subject_dirs(ROOT, SUBJECTS)
    if not subs:
        print(f"[Info] No subjects found under {ROOT}")
        return

    jobs = []
    for sub_dir in subs:
        for run in RUNS:
            files = list_run_files(sub_dir, run)
            if not files:
                print(f"[Info] No files in {sub_dir / run} matching {GLOB_PATTERNS}")
                continue
            for f in files:
                jobs.append((f, FWHM_MM, OVERWRITE))

    if not jobs:
        print("[Info] Nothing to do.")
        return

    if NUM_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = [ex.submit(smooth_one_file, f, fwhm, ow) for (f, fwhm, ow) in jobs]
            for _ in as_completed(futs):
                pass
    else:
        for f, fwhm, ow in jobs:
            smooth_one_file(f, fwhm, ow)

    print("\n[Done] All smoothing finished.")

if __name__ == "__main__":
    main()
