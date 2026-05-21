#!/bin/bash
set -e

# 配置路径
BIDS_DIR="/home/zhangze/data/fertility_fMRI/Nifti"
SUBJECT="sub-094"
TASK_LABEL="task-viewing"  # 和 sub-093 保持一致的任务名

# 进入被试的 func 目录
cd "${BIDS_DIR}/${SUBJECT}/func" || exit 1

echo "=== 开始批量重命名 ${SUBJECT} 的功能像文件 ==="
echo "目标格式: ${SUBJECT}_${TASK_LABEL}_run-*_bold.nii.gz"
echo "----------------------------------------"

# 遍历所有需要重命名的 bold.nii.gz 文件
find . -maxdepth 1 -name "${SUBJECT}_run-*_bold.nii.gz" -type f | while read -r old_file; do
    old_file=$(basename "${old_file}")
    
    # 提取 run 编号（如 run-8）
    run_part=$(echo "${old_file}" | grep -o "run-[0-9]*")
    
    # 构建新文件名
    new_file="${SUBJECT}_${TASK_LABEL}_${run_part}_bold.nii.gz"
    
    # 重命名 nii.gz
    echo "重命名: ${old_file} → ${new_file}"
    mv -v "${old_file}" "${new_file}"
    
    # 同步重命名对应的 json 文件
    old_json="${old_file%.nii.gz}.json"
    new_json="${new_file%.nii.gz}.json"
    if [ -f "${old_json}" ]; then
        echo "重命名: ${old_json} → ${new_json}"
        mv -v "${old_json}" "${new_json}"
    fi
    
    # 同步重命名对应的 bak 文件（如果存在）
    old_bak="${old_file%.nii.gz}.json.bak"
    new_bak="${new_file%.nii.gz}.json.bak"
    if [ -f "${old_bak}" ]; then
        echo "重命名: ${old_bak} → ${new_bak}"
        mv -v "${old_bak}" "${new_bak}"
    fi
    
    echo "---"
done

echo "🎉 批量重命名完成！"
echo "检查结果: ls -l ${BIDS_DIR}/${SUBJECT}/func/"