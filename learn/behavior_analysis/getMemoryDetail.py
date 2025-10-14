import os
import glob
import pandas as pd
# ========= run7的行为分析：数据预处理，增加2列：图片名，动作：3回忆，4回忆失败 =========
base_dir = r"../../data_events"  # 根目录
mapping = {}

# 遍历 sub-01 到 sub-28
for i in range(1, 29):
    sub_folder = f"sub-{i:02d}"  # 生成 sub-01, sub-02 ... sub-28
    folder_path = os.path.join(base_dir, sub_folder)

    if not os.path.exists(folder_path):
        continue  # 跳过不存在的文件夹

    # 查找 run-7 的 tsv 和 csv 文件
    tsv_files = glob.glob(os.path.join(folder_path, f"{sub_folder}_run-7_events.tsv"))
    csv_files = glob.glob(os.path.join(folder_path, f"{sub_folder}_run-7_events.csv"))

    if tsv_files and csv_files:  # 确保两个文件都存在
        mapping[tsv_files[0]] = csv_files[0]

# 打印结果
for file_path, csv_ref in mapping.items():
    # 读取原始 tsv 文件
    df = pd.read_csv(file_path, sep="\t")
    # 拼接 trial_type 和 pic_num
    df["pic_out"] = df["trial_type"].astype(str) + df["pic_num"].astype(str) + ".jpg"
    # 保存新的文件
    df.to_csv(file_path, sep="\t", index=False)
    ref = pd.read_csv(csv_ref)
    map_ci = {}
    map_w5 = {}
    map_eword = {}
    map_word = {}
    if {"ciyu", "eword7_RESP"}.issubset(ref.columns):
        map_ci = dict(zip(ref["ciyu"].astype(str), ref["eword7_RESP"]))
        map_eword = dict(zip(ref["ciyu"].astype(str), ref["eword7_RT"]))
    if {"word5_word", "word7_RESP"}.issubset(ref.columns):
        map_w5 = dict(zip(ref["word5_word"].astype(str), ref["word7_RESP"]))
        map_word = dict(zip(ref["word5_word"].astype(str), ref["word7_RT"]))
    action1 = df["pic_out"].astype(str).map(map_ci)        # ciyu 命中 → eword7_RESP
    time1 = df["pic_out"].astype(str).map(map_eword)
    action2 = df["pic_out"].astype(str).map(map_w5)        # 否则用 word5_word 命中 → word7_RESP
    time2 = df["pic_out"].astype(str).map(map_word)
    df["action"] = action1.where(action1.notna(), action2) # 优先 action1，缺失用 action2 填
    df["action_time"] = time1.where(time1.notna(), time2) # 优先 action1，缺失用 action2 填
    df.to_csv(file_path, sep="\t", index=False)
    print(f"已保存：{file_path}")