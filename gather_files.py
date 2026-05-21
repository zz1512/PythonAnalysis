import os
import csv
import shutil
from datetime import datetime
from collections import deque
from typing import Iterable, Set, Tuple


def fast_find_files(root_dir: str,
                    exts: Set[str] = {'.m', '.py', '.ipynb'},
                    exclude_dirs: Set[str] = {'.git', '__pycache__', '.ipynb_checkpoints', '.venv', 'build', 'dist'},
                    follow_symlinks: bool = False) -> Iterable[str]:
    exts = {e.lower() for e in exts}
    dq = deque([root_dir])
    while dq:
        cur = dq.popleft()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=follow_symlinks):
                        name = entry.name
                        if name in exclude_dirs or name.startswith('.'):
                            continue
                        dq.append(entry.path)
                    elif entry.is_file(follow_symlinks=follow_symlinks):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in exts:
                            yield entry.path
        except (FileNotFoundError, PermissionError):
            continue


def sanitize_filename(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, '_')
    return name


def make_unique_name(root_dir: str, file_path: str) -> str:
    rel_path = os.path.relpath(file_path, root_dir)
    parts = os.path.splitext(rel_path)
    name = sanitize_filename(parts[0].replace(os.sep, '_')) + parts[1]
    return name


def export_csv(rows, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "abs_path", "modified_time"])
        writer.writerows(rows)


def scan_and_collect(root_dir: str) -> list:
    rows = []
    for p in fast_find_files(root_dir):
        try:
            st = os.stat(p)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            mtime = ""
        rows.append([os.path.basename(p), os.path.abspath(p), mtime])
    return rows


def copy_all(files_csv_rows: list, root_dir: str, dst_dir: str, log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)
    ok = 0

    with open(log_path, "w", encoding="utf-8") as logf:
        for file_name, abs_path, _ in files_csv_rows:
            if not os.path.exists(abs_path):
                logf.write(f"[SKIP] not found: {abs_path}\n")
                continue
            # 基于相对路径重命名
            new_name = make_unique_name(root_dir, abs_path)
            dst_path = os.path.join(dst_dir, new_name)

            try:
                shutil.copy2(abs_path, dst_path)
                ok += 1
            except Exception as e:
                logf.write(f"[FAIL] {abs_path} -> {dst_path} | {e}\n")

    return ok


if __name__ == "__main__":
    # === 配置区 ===
    ROOT_DIR = r"H:\metaphor"                 # 需要扫描的根目录
    OUT_CSV  = r"H:\PythonAnalysis\gather_to_learn\files.csv" # CSV 输出路径
    COPY_DIR = r"H:\PythonAnalysis\gather_to_learn"     # 复制到的单一目标文件夹
    LOG_FILE = r"H:\PythonAnalysis\gather_to_learn\copy.log"  # 日志路径

    # 1) 扫描并导出 CSV
    rows = scan_and_collect(ROOT_DIR)
    export_csv(rows, OUT_CSV)
    print(f"已导出 CSV：{OUT_CSV}（共 {len(rows)} 条）")

    # 2) 复制到一个文件夹（重名文件使用路径命名）
    copied = copy_all(rows, ROOT_DIR, COPY_DIR, LOG_FILE)
    print(f"已复制 {copied}/{len(rows)} 个文件到：{COPY_DIR}")
    print(f"复制日志：{LOG_FILE}")
