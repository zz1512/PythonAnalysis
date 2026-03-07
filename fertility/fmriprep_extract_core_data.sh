#!/bin/bash
set -euo pipefail

# 配置项
FMRIPREP_OUTPUT="/home/zhangze/data/fertility_fMRI/fmriprep_output"
EXTRACT_DIR="/home/zhangze/data/fertility_fMRI/fmriprep_core_data"
PARTICIPANT_FILE="/home/zhangze/data/fertility_fMRI/code/participants.txt"
TASK_LABEL="task-viewing"  # 你的文件固定包含这个task标签

# 初始化目录
mkdir -p "${EXTRACT_DIR}/logs"

# 遍历受试者
while IFS= read -r participant; do
    [[ -z "${participant}" || "${participant}" =~ ^# ]] && continue
    echo "=== 处理 ${participant} ==="

    # 定义核心路径
    func_dir="${FMRIPREP_OUTPUT}/${participant}/func"
    [ ! -d "${func_dir}" ] && { echo "⚠️ ${participant} 无func目录，跳过"; continue; }

    # 提取所有run编号（从文件名中截取）
    runs=$(ls "${func_dir}/${participant}_${TASK_LABEL}_run-"*_bold.* 2>/dev/null | grep -o "run-[0-9]*" | sort -u)
    
    for run in ${runs}; do
        echo "  处理 ${run}"
        dst_dir="${EXTRACT_DIR}/${participant}/${run}"
        mkdir -p "${dst_dir}"

        # 1. 提取Volume数据（直接拼接文件名）
        vol_src="${func_dir}/${participant}_${TASK_LABEL}_${run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        if [ -f "${vol_src}" ]; then
            cp -v "${vol_src}" "${dst_dir}/${participant}_${run}_MNI_preproc_bold.nii.gz"
            echo "    ✅ Volume提取成功"
        else
            echo "    ❌ Volume文件不存在: ${vol_src}"
        fi

        # 2. 提取Surface数据（直接拼接文件名）
        surf_src="${func_dir}/${participant}_${TASK_LABEL}_${run}_space-fsLR_den-91k_bold.dtseries.nii"
        if [ -f "${surf_src}" ]; then
            cp -v "${surf_src}" "${dst_dir}/${participant}_${run}_fsLR_91k_preproc_bold.dtseries.nii"
            echo "    ✅ Surface提取成功"
        else
            echo "    ❌ Surface文件不存在: ${surf_src}"
        fi

        # 3. 提取confound文件
        conf_src="${func_dir}/${participant}_${TASK_LABEL}_${run}_desc-confounds_timeseries.tsv"
        if [ -f "${conf_src}" ]; then
            cp -v "${conf_src}" "${dst_dir}/${participant}_${run}_confounds.tsv"
            echo "    ✅ Confound提取成功"
        else
            echo "    ❌ Confound文件不存在: ${conf_src}"
        fi

        # 4. 复制脑掩码（如果有）
        mask_src="${FMRIPREP_OUTPUT}/${participant}/anat/${participant}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        if [ -f "${mask_src}" ]; then
            cp -v "${mask_src}" "${dst_dir}/"
            echo "    ✅ 脑掩码复制成功"
        fi
    done
done < "${PARTICIPANT_FILE}"

echo "=== 提取完成！结果在 ${EXTRACT_DIR} ==="