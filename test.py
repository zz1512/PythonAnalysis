import time
import shutil
from pathlib import Path

# 配置参数
SUBJECTS = [f"sub-{i:02d}" for i in range(6, 29)]
RUNS = [3, 4]
OUTPUT_ROOT = Path(r"learn_LSS")
CHECK_INTERVAL = 7200  # 2小时（秒）
MAX_AGE = 7200  # 2小时（秒）


def get_folder_age_seconds(folder_path):
    """获取文件夹年龄（秒）"""
    if not folder_path.exists():
        return float('inf')
    try:
        mtime = folder_path.stat().st_mtime
        return time.time() - mtime
    except:
        return float('inf')


def cleanup_old_caches():
    """清理2小时没有更新的cache文件夹"""
    cache_dirs = []

    # 查找所有缓存目录
    for sub in SUBJECTS:
        for run in RUNS:
            lss_dir = OUTPUT_ROOT / sub / f"run-{run}_LSS"
            if lss_dir.exists():
                cache_dir = lss_dir / "cache"
                if cache_dir.exists():
                    cache_dirs.append(cache_dir)

    cleaned_count = 0
    for cache_dir in cache_dirs:
        try:
            age_seconds = get_folder_age_seconds(cache_dir)

            if age_seconds > MAX_AGE:
                # 计算缓存大小
                size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                shutil.rmtree(cache_dir)

                size_mb = size_before // (1024 * 1024)
                print(f"[{time.strftime('%H:%M:%S')}] 清理缓存: {cache_dir}")
                print(f"          释放空间: {size_mb} MB, 年龄: {age_seconds / 3600:.1f} 小时")
                cleaned_count += 1
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 跳过新缓存: {cache_dir} (年龄: {age_seconds / 3600:.1f} 小时)")

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 清理失败 {cache_dir}: {e}")

    # 检查磁盘空间
    if OUTPUT_ROOT.exists():
        total, used, free = shutil.disk_usage(OUTPUT_ROOT)
        free_gb = free // (2 ** 30)
        print(f"[{time.strftime('%H:%M:%S')}] 磁盘剩余空间: {free_gb} GB")

    return cleaned_count


def main():
    """主函数：每2小时执行一次清理"""
    print(f"[{time.strftime('%H:%M:%S')}] 启动缓存清理服务")
    print(f"清理策略: 每2小时清理年龄超过2小时的cache文件夹")
    print(f"监控目录: {OUTPUT_ROOT}")

    while True:
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] 开始检查缓存...")
            cleaned_count = cleanup_old_caches()

            if cleaned_count > 0:
                print(f"[{time.strftime('%H:%M:%S')}] 本次清理完成: {cleaned_count} 个缓存目录")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 无需清理")

            print(
                f"[{time.strftime('%H:%M:%S')}] 下次检查时间: {time.strftime('%H:%M:%S', time.localtime(time.time() + CHECK_INTERVAL))}")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[{time.strftime('%H:%M:%S')}] 用户中断，退出清理服务")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 执行出错: {e}")
            time.sleep(300)  # 出错后等待5分钟再继续


if __name__ == "__main__":
    main()