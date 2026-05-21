# fMRI LSS 分析缓存优化说明

## 优化概述

本次优化主要解决了 `fmri_lss_enhance.py` 中的硬盘缓存累积和内存泄漏问题，显著减少了磁盘空间占用并提高了分析效率。

## 主要优化内容

### 1. 临时缓存管理
- **问题**: 原代码使用固定缓存目录，导致缓存文件累积
- **解决**: 为每个 sub-run 创建独立的临时缓存目录
- **效果**: 每个 run 完成后自动清理，避免缓存累积

### 2. 内存优化
- **批处理大小**: 从 20 减少到 10，降低内存压力
- **内存清理**: 在每个 trial 处理完成后立即清理相关对象
- **GLM 参数**: 启用 `minimize_memory=True`，降低缓存级别

### 3. 缓存监控与清理
- **大小监控**: 实时监控缓存目录大小
- **自动清理**: 缓存超过 2GB 时自动清理
- **定期清理**: 每 5 个批次清理一次缓存
- **异常处理**: 程序异常时也会清理临时缓存

### 4. 新增配置参数
```python
CLEAN_CACHE_AFTER_RUN = True  # 每个run完成后清理缓存
MAX_CACHE_SIZE_GB = 2.0       # 最大缓存大小限制(GB)
```

## 新增功能函数

### 缓存管理函数
- `get_dir_size_gb()`: 获取目录大小
- `clean_cache_directory()`: 清理缓存目录
- `create_temp_cache_dir()`: 创建临时缓存目录
- `cleanup_temp_cache()`: 清理临时缓存
- `cleanup_all_temp_caches()`: 清理所有临时缓存

## 使用方法

### 运行优化后的分析
```bash
# 并行模式（推荐）
python fmri_lss_enhance.py

# 串行模式
# 在代码中设置 USE_PARALLEL = False
```

### 测试缓存优化效果
```bash
python test_cache_optimization.py
```

### 手动清理临时缓存
```python
from fmri_lss_enhance import cleanup_all_temp_caches
cleanup_all_temp_caches()
```

## 性能改进

### 磁盘使用
- **优化前**: 缓存文件持续累积，可能达到数十GB
- **优化后**: 每个 run 最大缓存 2GB，完成后自动清理

### 内存使用
- **优化前**: 内存泄漏，长时间运行内存持续增长
- **优化后**: 及时清理，内存使用稳定

### 运行稳定性
- **优化前**: 长时间运行可能因缓存过大而崩溃
- **优化后**: 自动监控和清理，运行更稳定

## 监控信息

运行时会显示详细的缓存监控信息：
```
[12:34:56] [sub-01|run-3] 使用临时缓存: lss_cache_abc123
[12:35:10] [sub-01|run-3] 批次完成: 10/50 (缓存: 0.85GB)
[12:35:25] [sub-01|run-3] 最终缓存大小: 1.23GB
[12:35:26] [sub-01|run-3] LSS分析完成: 50/50 成功，缓存已清理
```

## 注意事项

1. **临时目录**: 使用系统临时目录 `/tmp`，确保有足够空间
2. **并行处理**: 多个进程同时运行时，每个都有独立的临时缓存
3. **异常恢复**: 程序异常退出时，临时缓存会在下次运行时清理
4. **配置调整**: 可根据系统配置调整 `MAX_CACHE_SIZE_GB` 参数

## 故障排除

### 如果临时缓存未清理
```python
# 手动清理所有 LSS 临时缓存
import shutil
from pathlib import Path

temp_dirs = Path("/tmp").glob("lss_cache_*")
for temp_dir in temp_dirs:
    if temp_dir.is_dir():
        shutil.rmtree(temp_dir)
        print(f"已清理: {temp_dir}")
```

### 如果内存使用仍然过高
- 减少 `batch_size` (当前为 10)
- 降低 `N_JOBS_PARALLEL` (当前为 4)
- 减少 `MAX_CACHE_SIZE_GB` (当前为 2.0)

## 更新日志

- **v1.1**: 添加临时缓存管理和内存优化
- **v1.0**: 原始版本