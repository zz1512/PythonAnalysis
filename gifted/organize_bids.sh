#!/bin/bash
set -euo pipefail

# 用途：将已转换好的 NIfTI（按 RAW_ROOT 的命名/目录规则）物理复制整理为 BIDS。
# 产物：BIDS_ROOT/sub-<ID>/ses-<pre|post>/{anat,func}/...（供 fMRIPrep 使用）。

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

copy_safe() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
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

# 【适配双T1】智能匹配T1w，自动选第一个，过滤REST
find_t1_smart() {
    local raw_dir="$1"
    local t1_file=""

    # 1. 优先匹配anat子目录里的T1_MPRAGE（你的真实路径）
    t1_file=$(find "${raw_dir}/anat" -maxdepth 1 -type f -name "*T1_MPRAGE*.nii.gz" ! -name ".*" | head -n 1)
    if [ -n "$t1_file" ]; then
        echo "$t1_file"
        return
    fi

    # 2. 根目录兜底匹配
    t1_file=$(find "$raw_dir" -maxdepth 1 -type f -name "*T1_MPRAGE*.nii.gz" ! -name ".*" | grep -v "REST" | head -n 1)
    echo "$t1_file"
}

# 功能像批量匹配
find_func_smart_batch() {
    local raw_dir="$1"
    local pattern="$2"
    local files=""

    # 优先func子目录
    files=$(find "${raw_dir}/func" -maxdepth 1 -type f -name "$pattern" ! -name ".*" | sort)
    if [ -n "$files" ]; then
        echo "$files"
        return
    fi

    # 根目录兜底
    files=$(find "$raw_dir" -maxdepth 1 -type f -name "$pattern" ! -name ".*" | sort)
    echo "$files"
}

# ====================== 主程序 ======================
echo -e "\n==================== BIDS整理脚本（双T1适配最终版） ===================="
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

        # ============== 【双T1适配】1. 处理 Anat (T1w) ==============
        echo -e "\n📦 处理结构像 T1w..."
        t1_nii=$(find_t1_smart "$raw_dir")
        if [ -n "$t1_nii" ]; then
            t1_json="${t1_nii%.nii.gz}.json"
            target_nii="${bids_anat}/sub-${sub}_ses-${ses}_T1w.nii.gz"
            target_json="${bids_anat}/sub-${sub}_ses-${ses}_T1w.json"
            copy_safe "$t1_nii" "$target_nii"
            copy_safe "$t1_json" "$target_json"
            echo "✅ 成功匹配T1w: $(basename "$t1_nii")"
        else
            echo "❌ 错误：未找到${sub}_${ses}的T1w结构像，请检查原始数据！"
        fi

        # ============== 2. 处理 Func - REST ==============
        echo -e "\n📦 处理静息态 REST..."
        rest_nii=$(find_func_smart_batch "$raw_dir" "*${ses}_REST*.nii.gz")
        if [ -n "$rest_nii" ]; then
            rest_json="${rest_nii%.nii.gz}.json"
            target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-REST_bold.nii.gz"
            target_json="${bids_func}/sub-${sub}_ses-${ses}_task-REST_bold.json"
            copy_safe "$rest_nii" "$target_nii"
            copy_safe "$rest_json" "$target_json"
        fi

        # ============== 3. 处理 Func - AUT ==============
        echo -e "\n📦 处理AUT任务..."
        aut_list=$(find_func_smart_batch "$raw_dir" "*${ses}_AUT[0-9]*.nii.gz")
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

        # ============== 4. 处理 Func - P ==============
        echo -e "\n📦 处理P任务..."
        p_list=$(find_func_smart_batch "$raw_dir" "*${ses}_P[0-9]*.nii.gz")
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

        # ============== 5. 处理 Func - C ==============
        echo -e "\n📦 处理C任务..."
        c_list=$(find_func_smart_batch "$raw_dir" "*${ses}_C[0-9]*.nii.gz")
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
        fi

    done
done

echo -e "\n========================================================================"
echo "🎉 全部处理完成！"
echo "✅ 验证命令: bids-validator $BIDS_ROOT"
echo "========================================================================"
