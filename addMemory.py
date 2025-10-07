import os
import glob
import pandas as pd
# ========= run7的行为分析：数据预处理，增加一列memory，作为是否记忆标识 =========
base_dir = r"/data_events"  # 根目录
fileList = []
# 遍历 sub-01 到 sub-28
for i in range(1, 29):
    sub_folder = f"sub-{i:02d}"  # 生成 sub-01, sub-02 ... sub-28
    folder_path = os.path.join(base_dir, sub_folder)

    if not os.path.exists(folder_path):
        continue  # 跳过不存在的文件夹

    # 查找 run-7 的 tsv 和 csv 文件
    tsv_files = glob.glob(os.path.join(folder_path, f"{sub_folder}_run-7_events.tsv"))
    if len(tsv_files) != 0:
        fileList.append(tsv_files[0])
for file in fileList:
    # 读取原始 tsv 文件
    df = pd.read_csv(file, sep="\t")
    df["memory"] = (df["action"].astype(str) == "3.0").astype(int)
    # 保存新的文件
    df.to_csv(file, sep="\t", index=False)