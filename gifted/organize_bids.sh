#!/bin/bash
set -euo pipefail

# ====================== 【可修改配置】 ======================
RAW_ROOT="/home/zhangze/data/gifted_fMRI"
BIDS_ROOT="/home/zhangze/data/gifted_fMRI_BIDS"
SUBJECTS=("A001" "A002" "A003")
SESSIONS=("pre" "post")
# ===========================================================

mkdir_safe() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✅ 创建目录: $dir"
    fi
}

# 用cp替代ln，物理复制文件，彻底解决挂载问题
copy_safe() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
        # 过滤隐藏文件（.开头）
        if [[ "$(basename "$src")" == .* ]]; then
            echo "⚠️  跳过隐藏文件: $src"
            return
        fi
        if [ ! -f "$dest" ]; then
            cp -f "$src" "$dest"
            echo "📋 复制: $src -> $dest"
        else
            echo "⏭️  已存在，跳过: $dest"
        fi
    else
        echo "⚠️  源文件不存在，跳过: $src"
    fi
}

# 智能查找文件，过滤隐藏文件
find_files_smart_batch() {
    local raw_dir="$1"
    local pattern="$2"
    local files=""

    # 优先func子目录，过滤隐藏文件
    files=$(find "${raw_dir}/func" -maxdepth 1 -type f -name "$pattern" ! -name ".*" | sort)
    if [ -n "$files" ]; then
        echo "$files"
        return
    fi

    # 根目录查找，过滤隐藏文件
    files=$(find "$raw_dir" -maxdepth 1 -type f -name "$pattern" ! -name ".*" | sort)
    echo "$files"
}

# ====================== 主程序 ======================
echo -e "\n==================== BIDS整理脚本（物理复制最终版） ===================="
echo "原始数据目录：$RAW_ROOT"
echo "BIDS输出目录：$BIDS_ROOT"
echo "=======================================================================\n"

mkdir_safe "$BIDS_ROOT"
cat > "$BIDS_ROOT/dataset_description.json" << EOF
{
  "Name": "Gifted_fMRI",
  "BIDSVersion": "1.7.0",
  "License": "CC0"
}
EOF
echo "✅ 生成 BIDS 必需文件：dataset_description.json"

for sub in "${SUBJECTS[@]}"; do
    echo -e "\n========================================"
    echo "处理被试：$sub"
    echo "========================================"

    for ses in "${SESSIONS[@]}"; do
        echo -e "\n-------------------"
        echo "时间点：$ses"
        echo "-------------------"

        raw_dir="${RAW_ROOT}/${sub}_${ses}_NIFTI"
        [ ! -d "$raw_dir" ] && { echo "⚠️  目录不存在，跳过：$raw_dir"; continue; }

        sub_dir="${BIDS_ROOT}/sub-${sub}"
        ses_dir="${sub_dir}/ses-${ses}"
        bids_anat="${ses_dir}/anat"
        bids_func="${ses_dir}/func"
        mkdir_safe "$bids_anat"
        mkdir_safe "$bids_func"

        # 1. T1w结构像
        echo -e "\n📦 处理结构像 T1w..."
        t1_nii=$(find_files_smart_batch "$raw_dir" "*T1_MPRAGE*.nii.gz")
        if [ -n "$t1_nii" ]; then
            t1_json="${t1_nii%.nii.gz}.json"
            target_nii="${bids_anat}/sub-${sub}_ses-${ses}_T1w.nii.gz"
            target_json="${bids_anat}/sub-${sub}_ses-${ses}_T1w.json"
            copy_safe "$t1_nii" "$target_nii"
            copy_safe "$t1_json" "$target_json"
        fi

        # 2. REST静息态
        echo -e "\n📦 处理静息态 REST..."
        rest_nii=$(find_files_smart_batch "$raw_dir" "*${ses}_REST*.nii.gz")
        if [ -n "$rest_nii" ]; then
            rest_json="${rest_nii%.nii.gz}.json"
            target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-REST_bold.nii.gz"
            target_json="${bids_func}/sub-${sub}_ses-${ses}_task-REST_bold.json"
            copy_safe "$rest_nii" "$target_nii"
            copy_safe "$rest_json" "$target_json"
        fi

        # 3. AUT任务
        echo -e "\n📦 处理AUT任务..."
        aut_list=$(find_files_smart_batch "$raw_dir" "*${ses}_AUT[0-9]*.nii.gz")
        if [ -n "$aut_list" ]; then
            run=1
            unique_aut_list=$(echo "$aut_list" | awk '!seen[$0]++')
            for nii in $unique_aut_list; do
                json="${nii%.nii.gz}.json"
                target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-AUT_run-${run}_bold.nii.gz"
                target_json="${bids_func}/sub-${sub}_ses-${ses}_task-AUT_run-${run}_bold.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$json" "$target_json"
                run=$((run+1))
            done
        fi

        # 4. P任务
        echo -e "\n📦 处理P任务..."
        p_list=$(find_files_smart_batch "$raw_dir" "*${ses}_P[0-9]*.nii.gz")
        if [ -n "$p_list" ]; then
            run=1
            unique_p_list=$(echo "$p_list" | awk '!seen[$0]++')
            for nii in $unique_p_list; do
                json="${nii%.nii.gz}.json"
                target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-P_run-${run}_bold.nii.gz"
                target_json="${bids_func}/sub-${sub}_ses-${ses}_task-P_run-${run}_bold.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$json" "$target_json"
                run=$((run+1))
            done
        fi

        # 5. C任务（重点修复，之前匹配可能有问题）
        echo -e "\n📦 处理C任务..."
        # 精准匹配C1-C7，避免漏匹配
        c_list=$(find_files_smart_batch "$raw_dir" "*${ses}_C[0-9]*.nii.gz")
        if [ -n "$c_list" ]; then
            run=1
            unique_c_list=$(echo "$c_list" | awk '!seen[$0]++')
            for nii in $unique_c_list; do
                json="${nii%.nii.gz}.json"
                target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-C_run-${run}_bold.nii.gz"
                target_json="${bids_func}/sub-${sub}_ses-${ses}_task-C_run-${run}_bold.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$json" "$target_json"
                run=$((run+1))
            done
        else
            echo "ℹ️  未找到C任务数据，检查原始文件: ${raw_dir}/*${ses}_C*.nii.gz"
        fi

    done
done

echo -e "\n========================================================================"
echo "🎉 全部处理完成！"
echo "✅ 验证命令: bids-validator $BIDS_ROOT"
echo "========================================================================"