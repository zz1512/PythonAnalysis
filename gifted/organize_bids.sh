#!/bin/bash
set -euo pipefail

# 用途：精准匹配真实数据格式，将NIfTI复制整理为BIDS规范结构
# 适配路径：/home/zhangze/data/gifted/sub-<ID>_<ses>/
# 适配文件名：sub-A021_pre_P4_bold.nii.gz、sub-A021_pre_C7_bold.nii.gz、sub-A021_pre_rest.nii.gz等
# 产物：BIDS_ROOT/sub-<ID>/ses-<pre|post>/{anat,func,dwi}/...（供 fMRIPrep 使用）

# ====================== 【可修改配置】 ======================
RAW_ROOT="/home/zhangze/data/gifted"
BIDS_ROOT="/home/zhangze/data/gifted_BIDS"
SUBJECTS=($(printf "A%03d " {1..21}))  # A001-A021 全部被试
SESSIONS=("pre" "post")                # pre/post 两个时间点
# ===========================================================

# 安全创建目录（避免重复创建/权限问题）
mkdir_safe() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✅ 创建目录: $dir"
    fi
}

# 安全复制文件（跳过隐藏文件/已存在文件，保留日志）
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

# 【精准匹配】T1w识别（适配anat目录下的sub-AXXX_ses_T1w.nii.gz/T1w_ND.nii.gz）
find_t1_smart() {
    local raw_dir="$1"
    local sub="$2"
    local ses="$3"
    local t1_file=""
    local anat_dir="${raw_dir}/anat"

    # 优先匹配主T1w文件：sub-A021_pre_T1w.nii.gz
    if [ -d "$anat_dir" ]; then
        t1_file=$(find "$anat_dir" -maxdepth 1 -type f -name "sub-${sub}_${ses}_T1w.nii.gz" ! -name ".*" -print -quit)
        # 兜底匹配T1w_ND文件：sub-A021_pre_T1w_ND.nii.gz
        if [ -z "$t1_file" ]; then
            t1_file=$(find "$anat_dir" -maxdepth 1 -type f -name "sub-${sub}_${ses}_T1w_ND.nii.gz" ! -name ".*" -print -quit)
        fi
    fi
    echo "$t1_file"
}

# 【精准匹配】REST静息态识别（anat目录下的sub-AXXX_ses_rest.nii.gz）
find_rest_smart() {
    local raw_dir="$1"
    local sub="$2"
    local ses="$3"
    local anat_dir="${raw_dir}/anat"
    local rest_file=""

    if [ -d "$anat_dir" ]; then
        rest_file=$(find "$anat_dir" -maxdepth 1 -type f -name "sub-${sub}_${ses}_rest.nii.gz" ! -name ".*" -print -quit)
    fi
    echo "$rest_file"
}

# 【精准匹配】功能任务文件识别（func目录下的sub-AXXX_ses_<任务>_bold.nii.gz）
find_task_smart() {
    local raw_dir="$1"
    local sub="$2"
    local ses="$3"
    local task_pattern="$4"  # 如P[0-9]*、C[0-9]*、AUT
    local func_dir="${raw_dir}/func"
    local files=""

    if [ -d "$func_dir" ]; then
        # 核心匹配规则：sub-A021_pre_P4_bold.nii.gz 这类格式
        files=$(find "$func_dir" -maxdepth 1 -type f -name "sub-${sub}_${ses}_${task_pattern}_bold.nii.gz" ! -name ".*" | sort)
    fi
    echo "$files"
}

# 【精准匹配】DWI弥散像识别（anat目录下的sub-AXXX_ses_dwi_*.nii.gz）
find_dwi_smart() {
    local raw_dir="$1"
    local sub="$2"
    local ses="$3"
    local anat_dir="${raw_dir}/anat"
    local dwi_files=""

    if [ -d "$anat_dir" ]; then
        dwi_files=$(find "$anat_dir" -maxdepth 1 -type f -name "sub-${sub}_${ses}_dwi_*.nii.gz" ! -name ".*" | sort)
    fi
    echo "$dwi_files"
}

# ====================== 主程序 ======================
echo -e "\n==================== BIDS整理脚本（最终精准版） ===================="
echo "原始数据目录：$RAW_ROOT"
echo "BIDS输出目录：$BIDS_ROOT"
echo "被试范围：${SUBJECTS[@]}"
echo "=======================================================================\n"

# 创建BIDS根目录并生成必需的描述文件
mkdir_safe "$BIDS_ROOT"
cat > "$BIDS_ROOT/dataset_description.json" << EOF
{
  "Name": "Gifted_fMRI",
  "BIDSVersion": "1.7.0",
  "License": "CC0"
}
EOF
echo "✅ 生成 BIDS 必需文件：dataset_description.json"

# 遍历所有被试
for sub in "${SUBJECTS[@]}"; do
    echo -e "\n========================================"
    echo "处理被试：$sub"
    echo "========================================"

    # 遍历pre/post时间点
    for ses in "${SESSIONS[@]}"; do
        echo -e "\n-------------------"
        echo "时间点：$ses"
        echo "-------------------"

        # 【关键修正】匹配真实路径：sub-A021_pre （无_NIFTI后缀）
        raw_dir="${RAW_ROOT}/sub-${sub}_${ses}"
        if [ ! -d "$raw_dir" ]; then
            echo "⚠️  目录不存在，跳过：$raw_dir"
            continue
        fi

        # 创建BIDS规范子目录（严格分离anat/func/dwi）
        sub_dir="${BIDS_ROOT}/sub-${sub}"
        ses_dir="${sub_dir}/ses-${ses}"
        bids_anat="${ses_dir}/anat"   # T1w专属
        bids_func="${ses_dir}/func"   # REST/AUT/P/C专属
        bids_dwi="${ses_dir}/dwi"     # DWI专属
        mkdir_safe "$bids_anat"
        mkdir_safe "$bids_func"
        mkdir_safe "$bids_dwi"

        # ============== 1. 处理结构像 T1w（anat目录） ==============
        echo -e "\n📦 处理结构像 T1w..."
        t1_nii=$(find_t1_smart "$raw_dir" "$sub" "$ses")
        if [ -n "$t1_nii" ]; then
            t1_json="${t1_nii%.nii.gz}.json"
            target_nii="${bids_anat}/sub-${sub}_ses-${ses}_T1w.nii.gz"
            target_json="${bids_anat}/sub-${sub}_ses-${ses}_T1w.json"
            copy_safe "$t1_nii" "$target_nii"
            copy_safe "$t1_json" "$target_json"
            echo "✅ 成功匹配T1w: $(basename "$t1_nii")"
        else
            echo "❌ 未找到${sub}_${ses}的T1w文件！"
        fi

        # ============== 2. 处理静息态 REST（anat→func目录） ==============
        echo -e "\n📦 处理静息态 REST..."
        rest_nii=$(find_rest_smart "$raw_dir" "$sub" "$ses")
        if [ -n "$rest_nii" ]; then
            rest_json="${rest_nii%.nii.gz}.json"
            target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-REST_bold.nii.gz"
            target_json="${bids_func}/sub-${sub}_ses-${ses}_task-REST_bold.json"
            copy_safe "$rest_nii" "$target_nii"
            copy_safe "$rest_json" "$target_json"
            echo "✅ 成功匹配REST: $(basename "$rest_nii")"
        else
            echo "⚠️  未找到${sub}_${ses}的rest文件"
        fi

        # ============== 3. 处理P任务（P1-Pn→func目录） ==============
        echo -e "\n📦 处理P任务..."
        p_files=$(find_task_smart "$raw_dir" "$sub" "$ses" "P[0-9]*")
        if [ -n "$p_files" ]; then
            run=1
            while IFS= read -r nii; do
                [ -z "$nii" ] && continue
                json="${nii%.nii.gz}.json"
                run_label=$(printf "%02d" "$run")
                target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-P_run-${run_label}_bold.nii.gz"
                target_json="${bids_func}/sub-${sub}_ses-${ses}_task-P_run-${run_label}_bold.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$json" "$target_json"
                run=$((run+1))
            done <<< "$p_files"
            echo "✅ 成功匹配P任务文件（共$((run-1))个）"
        else
            echo "⚠️  未找到${sub}_${ses}的P任务文件"
        fi

        # ============== 4. 处理C任务（C1-Cn→func目录） ==============
        echo -e "\n📦 处理C任务..."
        c_files=$(find_task_smart "$raw_dir" "$sub" "$ses" "C[0-9]*")
        if [ -n "$c_files" ]; then
            run=1
            while IFS= read -r nii; do
                [ -z "$nii" ] && continue
                json="${nii%.nii.gz}.json"
                run_label=$(printf "%02d" "$run")
                target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-C_run-${run_label}_bold.nii.gz"
                target_json="${bids_func}/sub-${sub}_ses-${ses}_task-C_run-${run_label}_bold.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$json" "$target_json"
                run=$((run+1))
            done <<< "$c_files"
            echo "✅ 成功匹配C任务文件（共$((run-1))个）"
        else
            echo "⚠️  未找到${sub}_${ses}的C任务文件"
        fi

        # ============== 5. 处理AUT任务（AUT→func目录） ==============
        echo -e "\n📦 处理AUT任务..."
        aut_files=$(find_task_smart "$raw_dir" "$sub" "$ses" "AUT")
        if [ -n "$aut_files" ]; then
            while IFS= read -r nii; do
                [ -z "$nii" ] && continue
                json="${nii%.nii.gz}.json"
                target_nii="${bids_func}/sub-${sub}_ses-${ses}_task-AUT_bold.nii.gz"
                target_json="${bids_func}/sub-${sub}_ses-${ses}_task-AUT_bold.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$json" "$target_json"
            done <<< "$aut_files"
            echo "✅ 成功匹配AUT任务文件"
        else
            echo "⚠️  未找到${sub}_${ses}的AUT任务文件"
        fi

        # ============== 6. 处理DWI弥散像（anat→dwi目录） ==============
        echo -e "\n📦 处理DWI弥散像..."
        dwi_files=$(find_dwi_smart "$raw_dir" "$sub" "$ses")
        if [ -n "$dwi_files" ]; then
            dwi_count=0
            while IFS= read -r nii; do
                [ -z "$nii" ] && continue
                dwi_json="${nii%.nii.gz}.json"
                # 提取DWI子类型（如trace/pa_b3000/fa等）
                dwi_suffix=$(basename "$nii" | sed -e "s/sub-${sub}_${ses}_dwi_//" -e "s/\.nii\.gz//")
                target_nii="${bids_dwi}/sub-${sub}_ses-${ses}_dwi_${dwi_suffix}.nii.gz"
                target_json="${bids_dwi}/sub-${sub}_ses-${ses}_dwi_${dwi_suffix}.json"
                copy_safe "$nii" "$target_nii"
                copy_safe "$dwi_json" "$target_json"
                dwi_count=$((dwi_count+1))
            done <<< "$dwi_files"
            echo "✅ 成功匹配DWI文件（共$dwi_count个）"
        else
            echo "⚠️  未找到${sub}_${ses}的DWI文件"
        fi

    done  # 结束时间点遍历
done  # 结束被试遍历

# 完成提示
echo -e "\n========================================================================"
echo "🎉 全部处理完成！"
echo "✅ 验证BIDS合规性命令: bids-validator $BIDS_ROOT"
echo "✅ 查看生成结构命令: tree -L 5 $BIDS_ROOT"
echo "========================================================================"