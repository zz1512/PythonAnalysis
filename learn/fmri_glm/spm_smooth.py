"""
SPM-style smoothing (FWHM=6mm, prefix='smooth') for batch subjects & runs.

- Root layout:
  ../../Pro_proc_data/sub-XX/runY/sub-*.nii[.gz]

- Output:
  Same folder, filename prefixed with 'smooth'
  Original files remain unchanged.
"""

from pathlib import Path
from typing import Iterable, Optional, List, Tuple
import numpy as np
import nibabel as nib
from nilearn import image
import platform

# ================ 配置区 =================
ROOT = Path(r"../../Pro_proc_data")

# 处理哪些 subject：
#   1) SUBJECTS=None  → 自动发现 ROOT 下所有 sub-*
#   2) SUBJECTS=["sub-01","sub-28"] → 仅这些
SUBJECTS: Optional[List[str]] = None  # 处理所有subject

# 要处理哪些 run（目录名），可随意改动/删减
RUNS: List[str] = ["run1", "run2", "run3", "run4", "run5", "run6"]

# 输入文件命名模式（按你数据命名来）
GLOB_PATTERNS = ["sub-*.nii", "sub-*.nii.gz"]

# 平滑参数
FWHM_MM = 2.0           # 等价 SPM [2 2 2]
SMOOTH_PREFIX = "smooth"  # 输出文件前缀
OVERWRITE = True        # 已存在 smooth* 是否覆盖

# 并行处理 (Windows下自动设为1)
if platform.system() == "Windows":
    NUM_WORKERS = 1
    print("[Info] Windows系统 detected, 强制 NUM_WORKERS=1")
else:
    NUM_WORKERS = 4     # Linux/Mac可并行

# =======================================

def validate_parameters():
    """验证输入参数"""
    if not ROOT.exists():
        raise FileNotFoundError(f"根目录不存在: {ROOT}")
    if FWHM_MM <= 0:
        raise ValueError("平滑核FWHM必须大于0")
    if NUM_WORKERS < 1:
        raise ValueError("工作进程数必须至少为1")
    print(f"[Config] 平滑核FWHM: {FWHM_MM}mm")
    print(f"[Config] 输出前缀: '{SMOOTH_PREFIX}'")
    print(f"[Config] 工作进程: {NUM_WORKERS}")
    print(f"[Config] 覆盖模式: {OVERWRITE}")

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
    """列出所有要处理的subject目录"""
    if subjects:
        subject_dirs = [root / s for s in subjects if (root / s).is_dir()]
        print(f"[Info] 指定处理 {len(subject_dirs)} 个subjects")
    else:
        subject_dirs = [p for p in sorted(root.glob("sub-*")) if p.is_dir()]
        print(f"[Info] 自动发现 {len(subject_dirs)} 个subjects")
    return subject_dirs

def list_run_files(sub_dir: Path, run: str) -> List[Path]:
    """列出某个run目录下所有需要处理的文件"""
    run_dir = sub_dir / run
    if not run_dir.exists():
        return []

    files: List[Path] = []
    for pat in GLOB_PATTERNS:
        for file_path in sorted(run_dir.glob(pat)):
            # 排除已平滑的文件和其他处理文件
            if not file_path.name.startswith((SMOOTH_PREFIX, 'sw', 'w', 'r')):
                files.append(file_path)
    return files

def smooth_one_file(in_path: Path, fwhm: float, overwrite: bool = False) -> Path:
    """对单个文件进行平滑处理"""
    out_path = add_prefix_to_filename(in_path, SMOOTH_PREFIX)

    # 检查是否跳过
    if out_path.exists() and not overwrite:
        print(f"[Skip] 已存在: {out_path.name}")
        return out_path

    print(f"[Smoothing] {in_path.name}  ->  {out_path.name}  (FWHM={fwhm}mm)")

    try:
        # 使用nilearn进行平滑
        smoothed = image.smooth_img(str(in_path), fwhm)
        # 转换为float32并保存
        smoothed = ensure_float32(smoothed)
        nib.save(smoothed, str(out_path))

        # 验证输出文件
        if out_path.exists():
            print(f"[Success] 完成: {out_path.name}")
        else:
            print(f"[Warning] 输出文件可能未正确创建: {out_path.name}")

    except Exception as e:
        print(f"[Error] 处理失败 {in_path.name}: {e}")
        # 清理可能已损坏的输出文件
        if out_path.exists():
            out_path.unlink()
        raise

    return out_path

def collect_processing_jobs(subs: List[Path]) -> List[Tuple[Path, float, bool]]:
    """收集所有需要处理的任务"""
    jobs = []
    total_files_found = 0

    for sub_dir in subs:
        print(f"[Scanning] 扫描 {sub_dir.name}...")
        for run in RUNS:
            files = list_run_files(sub_dir, run)
            if not files:
                print(f"  [Info] {run} 中未找到匹配的文件")
                continue

            jobs.extend([(f, FWHM_MM, OVERWRITE) for f in files])
            total_files_found += len(files)
            print(f"  [Found] {run}: {len(files)} 个文件")

    return jobs, total_files_found

def process_serial(jobs: List[Tuple[Path, float, bool]]) -> Tuple[int, int]:
    """串行处理所有任务"""
    processed_count = 0
    skipped_count = 0

    total_jobs = len(jobs)
    for i, (in_path, fwhm, overwrite) in enumerate(jobs, 1):
        print(f"[Progress] {i}/{total_jobs} ({i/total_jobs*100:.1f}%)")

        try:
            out_path = smooth_one_file(in_path, fwhm, overwrite)
            if out_path.exists() and out_path != in_path:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception:
            skipped_count += 1

    return processed_count, skipped_count

def process_parallel(jobs: List[Tuple[Path, float, bool]]) -> Tuple[int, int]:
    """并行处理所有任务"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    processed_count = 0
    skipped_count = 0
    total_jobs = len(jobs)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        future_to_job = {
            executor.submit(smooth_one_file, in_path, fwhm, overwrite): (in_path, overwrite)
            for in_path, fwhm, overwrite in jobs
        }

        # 收集结果
        completed_count = 0
        for future in as_completed(future_to_job):
            completed_count += 1
            in_path, overwrite = future_to_job[future]

            print(f"[Progress] {completed_count}/{total_jobs} ({completed_count/total_jobs*100:.1f}%)")

            try:
                out_path = future.result()
                if out_path.exists() and out_path != in_path:
                    processed_count += 1
                    print(f"[Completed] 成功: {out_path.name}")
                else:
                    skipped_count += 1
            except Exception as e:
                skipped_count += 1
                print(f"[Failed] 失败: {in_path.name} -> {e}")

    return processed_count, skipped_count

def main():
    """主函数"""
    print("=" * 60)
    print("fMRI数据平滑处理脚本")
    print("=" * 60)

    # 参数验证
    try:
        validate_parameters()
    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] 参数错误: {e}")
        return

    # 查找subject目录
    subs = list_subject_dirs(ROOT, SUBJECTS)
    if not subs:
        print(f"[Error] 在 {ROOT} 下未找到任何subject目录")
        return

    # 收集处理任务
    jobs, total_files = collect_processing_jobs(subs)
    if not jobs:
        print("[Info] 没有需要处理的文件")
        return

    print(f"\n[Info] 开始处理 {total_files} 个文件")
    print(f"[Info] 输出文件将添加前缀: '{SMOOTH_PREFIX}'")

    # 选择处理模式
    if NUM_WORKERS > 1:
        print(f"[Info] 使用并行模式 ({NUM_WORKERS} 进程)")
        processed_count, skipped_count = process_parallel(jobs)
    else:
        print("[Info] 使用串行模式")
        processed_count, skipped_count = process_serial(jobs)

    # 输出总结报告
    print("\n" + "=" * 60)
    print("处理完成总结")
    print("=" * 60)
    print(f"📁 扫描的subject数量: {len(subs)}")
    print(f"📊 发现的总文件数: {total_files}")
    print(f"✅ 成功处理文件数: {processed_count}")
    print(f"⏭️  跳过/失败文件数: {skipped_count}")
    print(f"🎯 平滑核大小: {FWHM_MM}mm FWHM")
    print(f"📝 输出文件前缀: '{SMOOTH_PREFIX}'")

    if skipped_count > 0 and not OVERWRITE:
        print(f"\n💡 提示: 有 {skipped_count} 个文件被跳过（已存在且OVERWRITE=False）")

    print(f"\n🎉 所有原文件保持未修改状态！")
    print("=" * 60)

if __name__ == "__main__":
    main()