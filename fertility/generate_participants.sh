#!/bin/bash
set -e

# ===================== 配置项（根据你的实际路径修改） =====================
# Nifti 根目录（存放所有 sub-xxx 文件夹的目录）
NIFTI_DIR="/home/zhangze/data/fertility_fMRI/Nifti"
# 生成的 participants.tsv 文件路径
OUTPUT_TSV="${NIFTI_DIR}/participants.tsv"

# ===================== 核心逻辑 =====================
echo "开始扫描目录: ${NIFTI_DIR}"

# 1. 检查 Nifti 目录是否存在
if [ ! -d "${NIFTI_DIR}" ]; then
    echo "错误：目录 ${NIFTI_DIR} 不存在！"
    exit 1
fi

# 2. 查找所有以 sub- 开头的目录（即被试文件夹）
#    过滤掉隐藏目录，只保留一级子目录
SUBJECTS=$(find "${NIFTI_DIR}" -maxdepth 1 -type d -name "sub-*" | sort)

# 3. 检查是否找到被试目录
if [ -z "${SUBJECTS}" ]; then
    echo "警告：在 ${NIFTI_DIR} 下未找到任何以 sub- 开头的被试目录！"
    exit 0
fi

# 4. 生成 participants.tsv 文件
echo "participant_id" > "${OUTPUT_TSV}"  # 写入表头
for sub_dir in ${SUBJECTS}; do
    # 提取被试ID（如从 /xxx/sub-004 中提取 sub-004）
    sub_id=$(basename "${sub_dir}")
    echo "${sub_id}" >> "${OUTPUT_TSV}"
done

# ===================== 结果输出 =====================
echo "✅ participants.tsv 文件已生成：${OUTPUT_TSV}"
echo "📋 共找到 $(($(wc -l < "${OUTPUT_TSV}") - 1)) 个被试："
tail -n +2 "${OUTPUT_TSV}"  # 打印所有被试ID（跳过表头）

# 可选：同时生成供 fmriprep 脚本使用的 participants.txt（纯被试ID列表，无表头）
PARTICIPANTS_TXT="${NIFTI_DIR}/../code/participants.txt"
mkdir -p "$(dirname ${PARTICIPANTS_TXT})"
tail -n +2 "${OUTPUT_TSV}" > "${PARTICIPANTS_TXT}"
echo "✅ 同时生成纯被试列表文件：${PARTICIPANTS_TXT}"
