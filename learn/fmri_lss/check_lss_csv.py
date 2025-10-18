import os
import csv


def find_small_trial_info_files_fast(root_dir, min_rows=70):
    """
    快速版本：只计算行数而不加载整个CSV文件
    """
    found_files = []
    file_count = 0
    path_list = []
    print(f"快速搜索 trial_info.csv 文件中...")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "trial_info.csv":
                file_path = os.path.join(dirpath, filename)
                file_count += 1

                try:
                    # 快速计算CSV文件行数（减去标题行）
                    with open(file_path, 'r', encoding='utf-8') as f:
                        row_count = sum(1 for row in f) - 1  # 减去标题行

                    if row_count < min_rows:
                        print(f"Found: {file_path} (只有 {row_count} 行)")
                        found_files.append((file_path, row_count))
                    if row_count >= min_rows:
                        path_list.append(file_path)
                except Exception as e:
                    print(f"读取文件出错 {file_path}: {e}")

    print(f"\n搜索完成！总共处理 {file_count} 个文件，找到 {len(found_files)} 个行数少于 {min_rows} 的文件")
    for path in path_list:
        print(f"\t{path}")
    return found_files

if __name__ == "__main__":
    # 可以灵活指定搜索目录
    search_directory = r"../../learn_LSS"
    find_small_trial_info_files_fast(search_directory)