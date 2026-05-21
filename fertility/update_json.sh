#!/bin/bash
set -e

# ===================== 配置项（可根据实际情况调整） =====================
# 数据根目录（存放所有sub-xxx文件夹的目录）
DATA_ROOT="/home/zhangze/data/fertility_fMRI/Nifti"
# 要设置的RepetitionTime值（最终会转为浮点数）
REPETITION_TIME=2.0
# 要设置的PhaseEncodingDirection值
PHASE_ENCODING_DIR="j"
# 要插入的SliceTiming数组（固定值）
SLICE_TIMING='[0,1,0.0666667,1.06667,0.133333,1.13333,0.2,1.2,0.266667,1.26667,0.333333,1.33333,0.4,1.4,0.466667,1.46667,0.533333,1.53333,0.6,1.6,0.666667,1.66667,0.733333,1.73333,0.8,1.8,0.866667,1.86667,0.933333,1.93333]'

# ===================== 第一步：备份原始文件 =====================
echo -e "\033[32m[1/4] 正在备份所有bold.json文件（后缀.bak）...\033[0m"
find "${DATA_ROOT}" -path "*/func/*bold.json" -type f | while read json_file; do
    if [ -f "${json_file}" ]; then
        # 避免重复备份
        if [ ! -f "${json_file}.bak" ]; then
            cp "${json_file}" "${json_file}.bak"
            echo "已备份：${json_file}"
        else
            echo "跳过重复备份：${json_file}"
        fi
    fi
done

# ===================== 第二步：批量修改JSON字段 =====================
echo -e "\033[32m\n[2/4] 正在批量修改bold.json字段...\033[0m"
find "${DATA_ROOT}" -path "*/func/*bold.json" -type f | while read json_file; do
    # 跳过空文件
    if [ ! -s "${json_file}" ]; then
        echo -e "\033[33m跳过空文件：${json_file}\033[0m"
        continue
    fi

    echo "处理文件：${json_file}"

    # 核心操作：同时设置三个字段（保留其他所有原有字段）
    jq --argjson st "$SLICE_TIMING" \
       --arg ped "${PHASE_ENCODING_DIR}" \
       --arg rt "${REPETITION_TIME}" '
        .SliceTiming = $st |
        .PhaseEncodingDirection = $ped |
        .RepetitionTime = ($rt | tonumber * 1.0)  # 强制转为浮点数
       ' "${json_file}" > "${json_file}.tmp"

    # 替换原文件
    mv "${json_file}.tmp" "${json_file}"
    echo "已完成字段修改：${json_file}"
done

# ===================== 第三步：验证修改结果 =====================
echo -e "\033[32m\n[4/4] 验证修改结果（抽查第一个文件）...\033[0m"
sample_file=$(find "${DATA_ROOT}" -path "*/func/*bold.json" -type f | head -n1)
if [ -f "${sample_file}" ]; then
    echo "抽查文件路径：${sample_file}"
    echo -e "\n【关键字段验证】"
    echo "SliceTiming数组长度（预期30）：$(jq '.SliceTiming | length' "${sample_file}")"
    echo "PhaseEncodingDirection值（预期j）：$(jq '.PhaseEncodingDirection' "${sample_file}")"
    echo "RepetitionTime值（预期2.0）：$(jq '.RepetitionTime' "${sample_file}")"
    echo "RepetitionTime类型（预期number）：$(jq '.RepetitionTime | type' "${sample_file}")"

    # 可选：打印SliceTiming前5个值确认
    echo -e "\n【SliceTiming前5个值】"
    jq '.SliceTiming | .[0:5]' "${sample_file}"
else
    echo -e "\033[31m❌ 未找到任何bold.json文件，请检查数据根目录是否正确！\033[0m"
    exit 1
fi