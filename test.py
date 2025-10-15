import os
import glob

# 指定目录路径
directory = r"H:\PythonAnalysis\learn_mvpa\full_roi_mask"

# 要保留的文件名关键词
keep_keywords = [
    "2_Lingual_Gyrus_sphere.nii",
    "3_Angular_Gyrus_sphere.nii",
    "19_Background_sphere.nii",
    "20_Insular_Cortex_sphere.nii",
    "31_Lateral_Occipital_Cortex_superior_division_sphere.nii",
    "29_Background_sphere.nii",
    "14_Background_sphere.nii",
    "10_Precuneous_Cortex_sphere.nii",
    "4_Precuneous_Cortex_sphere.nii"
]


def cleanup_files():
    # 切换到指定目录
    original_dir = os.getcwd()
    try:
        os.chdir(directory)

        # 查找所有以 cluster_ 开头的文件
        files_to_process = glob.glob("*.nii.gz")

        if not files_to_process:
            print(f"在目录 {directory} 中没有找到以 'cluster_' 开头的文件")
            return

        deleted_count = 0
        kept_count = 0

        for file_path in files_to_process:
            if os.path.isfile(file_path):
                # 检查文件名是否包含任何要保留的关键词
                should_keep = any(keyword in file_path for keyword in keep_keywords)

                if not should_keep:
                    print(f"删除文件: {file_path}")
                    os.remove(file_path)
                    deleted_count += 1
                else:
                    print(f"保留文件: {file_path}")
                    kept_count += 1

        print(f"\n操作完成！")
        print(f"删除文件数: {deleted_count}")
        print(f"保留文件数: {kept_count}")
        print(f"处理文件总数: {len(files_to_process)}")

    except FileNotFoundError:
        print(f"错误：目录 {directory} 不存在")
    except PermissionError:
        print(f"错误：没有权限访问目录 {directory}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 切换回原始目录
        os.chdir(original_dir)


# 安全预览模式 - 先显示将要删除的文件，不实际删除
def preview_cleanup():
    original_dir = os.getcwd()
    try:
        os.chdir(directory)

        files_to_process = glob.glob("*.nii.gz")

        if not files_to_process:
            print(f"在目录 {directory} 中没有找到以 'cluster_' 开头的文件")
            return

        print("预览模式 - 以下文件将被删除:")
        print("=" * 50)

        to_delete = []
        to_keep = []

        for file_path in files_to_process:
            if os.path.isfile(file_path):
                should_keep = any(keyword in file_path for keyword in keep_keywords)

                if not should_keep:
                    to_delete.append(file_path)
                    print(f"[删除] {file_path}")
                else:
                    to_keep.append(file_path)
                    print(f"[保留] {file_path}")

        print("=" * 50)
        print(f"总计: {len(to_delete)} 个文件将被删除, {len(to_keep)} 个文件将被保留")

        if to_delete:
            response = input("\n确认执行删除操作？(y/N): ")
            if response.lower() == 'y':
                cleanup_files()
            else:
                print("操作已取消")

    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    print("文件清理脚本")
    print(f"目标目录: {directory}")
    print("保留包含以下关键词的文件:")
    for keyword in keep_keywords:
        print(f"  - {keyword}")

    print("\n选择模式:")
    print("1. 预览模式 (推荐)")
    print("2. 直接执行删除")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "1":
        preview_cleanup()
    elif choice == "2":
        cleanup_files()
    else:
        print("无效选择，退出程序")