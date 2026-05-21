#!/bin/bash
set -euo pipefail

# ===================== 【配置项：已为你适配新版BIDS】 =====================
# BIDS 根目录（你刚才整理输出的目录）
BIDS_ROOT="/home/zhangze/data/gifted_BIDS"

# BIDS 必须的文件：必须放在 BIDS 根目录下
OUTPUT_TSV="${BIDS_ROOT}/participants.tsv"

# 给 fmriprep 调用的纯被试ID列表（无格式）
PARTICIPANTS_TXT="${BIDS_ROOT}/code/participants.txt"

# ===================== 核心逻辑 =====================
echo -e "\n========================================"
echo "  生成 BIDS 被试文件 (适配 fMRIprep)"
echo "========================================"
echo "BIDS 根目录: ${BIDS_ROOT}"

# 1. 检查 BIDS 目录
if [ ! -d "${BIDS_ROOT}" ]; then
    echo "❌ 错误：BIDS 目录不存在！请检查路径"
    exit 1
fi

# 2. 查找所有 sub-* 被试（安全遍历，支持带空格路径）
SUBJECTS=()
while IFS= read -r sub_dir; do
    SUBJECTS+=("$sub_dir")
done < <(find "${BIDS_ROOT}" -maxdepth 1 -type d -name "sub-*" | sort)

# 3. 无被试则退出
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "⚠️ 未找到任何被试目录"
    exit 0
fi

# 4. 生成标准 BIDS participants.tsv（fMRIprep 强制要求）
echo "participant_id" > "${OUTPUT_TSV}"

for sub_dir in "${SUBJECTS[@]}"; do
    sub_id=$(basename "${sub_dir}")
    echo "${sub_id}" >> "${OUTPUT_TSV}"
done

# 5. 生成 fmriprep 可用的纯被试列表
mkdir -p "$(dirname ${PARTICIPANTS_TXT})"
tail -n +2 "${OUTPUT_TSV}" > "${PARTICIPANTS_TXT}"

# ===================== 结果输出 =====================
echo -e "\n✅ 生成成功！"
echo "📄 BIDS 被试表: ${OUTPUT_TSV}"
echo "📄 fMRIprep 调用列表: ${PARTICIPANTS_TXT}"
echo "🔢 总计被试数量: $(($(wc -l < "${OUTPUT_TSV}") - 1))"
echo -e "\n📋 被试列表："
tail -n +2 "${OUTPUT_TSV}"
echo -e "\n========================================"