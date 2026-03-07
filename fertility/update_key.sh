#!/bin/bash
set -e  # 遇到错误立即退出

# ===================== 配置项 =====================
# 替换为你的 BIDS 数据集根目录（即你的 inputdir 路径）
BIDS_DIR="/home/zhangze/data/fertility_fMRI/Nifti/sub-094"

# ===================== 核心逻辑 =====================
echo "=== 开始批量修改 bold.json 文件中的关键字 ==="
echo "目标目录: ${BIDS_DIR}"
echo "修改规则："
echo "  - EstimatedEffectiveEchoSpacing → EffectiveEchoSpacing"
echo "  - EstimatedTotalReadoutTime → TotalReadoutTime"
echo "----------------------------------------"

# 查找所有 bold.json 文件并逐个处理
find "${BIDS_DIR}" -name "*_bold.json" -type f | while read -r json_file; do
    echo "处理文件: ${json_file}"
    
    # 1. 先创建备份文件（后缀 .bak）
    cp "${json_file}" "${json_file}.bak"
    
    # 2. 使用 jq 工具修改关键字（确保系统已安装 jq）
    # jq 是处理 JSON 的专业工具，比 sed 更安全，避免破坏 JSON 格式
    jq '
        if has("EstimatedEffectiveEchoSpacing") then
            .EffectiveEchoSpacing = .EstimatedEffectiveEchoSpacing | del(.EstimatedEffectiveEchoSpacing)
        else . end
        |
        if has("EstimatedTotalReadoutTime") then
            .TotalReadoutTime = .EstimatedTotalReadoutTime | del(.EstimatedTotalReadoutTime)
        else . end
    ' "${json_file}.bak" > "${json_file}"
    
    echo "✅ 已完成修改（备份文件：${json_file}.bak）"
done

echo "----------------------------------------"
echo "🎉 所有 bold.json 文件处理完成！"
echo "验证方法：查看任意一个文件，执行命令：cat 你的文件路径 | grep -E 'EffectiveEchoSpacing|TotalReadoutTime'"