import os
import shutil

# 定义需要保留的文件夹列表
keep_folders = [
    "sub-004", "sub-005", "sub-006", "sub-007", "sub-010", "sub-013", "sub-016",
    "sub-020", "sub-021", "sub-022", "sub-023", "sub-024", "sub-027", "sub-029",
    "sub-030", "sub-031", "sub-033", "sub-035", "sub-038", "sub-039", "sub-040",
    "sub-042", "sub-045", "sub-046", "sub-049", "sub-050", "sub-051", "sub-053",
    "sub-056", "sub-058", "sub-061", "sub-064", "sub-065", "sub-066", "sub-071",
    "sub-075", "sub-076", "sub-080", "sub-081", "sub-084", "sub-087", "sub-088",
    "sub-090", "sub-093", "sub-094", "sub-097", "sub-098", "sub-099", "sub-101",
    "sub-102", "sub-103", "sub-108", "sub-110", "sub-111", "sub-114", "sub-115",
    "sub-116", "sub-118", "sub-120", "sub-121", "sub-124", "sub-125", "sub-126",
    "sub-127", "sub-129"
]

# 目标路径
target_path = r"C:\python_fertility\first_level_exp2_conditions"


def delete_unwanted_folders():
    # 检查目标路径是否存在
    if not os.path.exists(target_path):
        print(f"错误：路径 {target_path} 不存在！")
        return

    # 获取路径下所有文件夹
    all_items = os.listdir(target_path)
    folders_to_delete = []

    # 筛选出需要删除的文件夹
    for item in all_items:
        item_path = os.path.join(target_path, item)
        # 只处理文件夹，排除文件
        if os.path.isdir(item_path) and item not in keep_folders:
            folders_to_delete.append(item_path)

    # 打印待删除的文件夹列表，供确认
    if folders_to_delete:
        print("以下文件夹将被删除：")
        for folder in folders_to_delete:
            print(f"- {folder}")

        # 确认是否执行删除
        confirm = input("\n确认删除以上文件夹？(输入 'yes' 确认，其他任意键取消)：")
        if confirm.lower() == 'yes':
            # 执行删除操作
            deleted_count = 0
            for folder in folders_to_delete:
                try:
                    # 删除文件夹及其所有内容
                    shutil.rmtree(folder)
                    print(f"已删除：{folder}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败 {folder}：{str(e)}")

            print(f"\n删除完成！共删除 {deleted_count} 个文件夹")
        else:
            print("删除操作已取消")
    else:
        print("未找到需要删除的文件夹")


if __name__ == "__main__":
    delete_unwanted_folders()