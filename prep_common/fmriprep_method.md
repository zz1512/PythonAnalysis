教程：fMRIPrep Demonstration 
一、构建fmriprep文件
1. 基础信息文件
基础信息：anat结构相、func功能相
每个部分都要包含nii.gz文件和同名的json文件（注意是_bold.json）：被试有几个run，func中就有几对文件
[图片]
[图片]
如果func的文件名不包含task信息，fmriprep可能无法准确识别，所以需要以下脚本调整文件名（自行调整路径）
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
2. 配置文件
配置license文件 license.txt
配置数据描述文件 dataset_description.json
配置bids_filters.json用来查询文件 bids_filters.json
配置被试列表 participants.txt
二、完善配置参数（清华仪器需完善每个被试json文件的信息）
如果json文件不包含下面3个参数，都需要重新确认，再添加到每个被试的每个json文件中
"SliceTiming"：[0,1,0.0666667,1.06667,0.133333,1.13333,0.2,1.2,0.266667,1.26667,0.333333,1.33333,0.4,1.4,0.466667,1.46667,0.533333,1.53333,0.6,1.6,0.666667,1.66667,0.733333,1.73333,0.8,1.8,0.866667,1.86667,0.933333,1.93333]
"PhaseEncodingDirection": "j" （扫描方向，需要每次实验重新确认，之前的配置为j）
"RepetitionTime": 2 （扫描的TR，需要每次重新确认，上面的SliceTiming也是根据TR来算出来的）
可使用下面的脚本实现上述功能
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
如果json文件中不包含EffectiveEchoSpacing、TotalReadoutTime，只包含EstimatedEffectiveEchoSpacing、EstimatedTotalReadoutTime，需要保持valum不变，把这两个key，改成EffectiveEchoSpacing、TotalReadoutTime，可用下面脚本实现
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
三、执行fmriprep脚本：Surface & volume
运行前必须验证 BIDS 格式，确保没有文件缺失：
# 安装bids-validator（如果没装）
npm install -g bids-validator  
# 验证BIDS结构 
bids-validator /home/zhangze/data/gifted_fMRI_BIDS
因为fmriprep执行很慢，所以最好用会话执行
screen -S fmriprep_process 
./fmriprep_test.sh

暂时脱离回话：Ctrl + A → 松开 → 按 D

进入会话：查找已有会话的id，根据id重启会话
screen -ls
screen -r 1234
具体文件路径和资源配置，都需要灵活调整
#!/bin/bash
set -e

# ===================== 核心路径配置 =====================
PROJECT_DIR="/home/zhangze/data/fertility_fMRI"
inputdir="${PROJECT_DIR}/Nifti"
outputdir="${PROJECT_DIR}/fmriprep_output"
fslicense="${PROJECT_DIR}/license.txt"
workingdir="${PROJECT_DIR}/tmp/fmriprep_wd"
TEMPLATEFLOW_DIR="${PROJECT_DIR}/templateflow"
participant_files="${PROJECT_DIR}/code/participants.txt"
bids_filter_file="${PROJECT_DIR}/code/bids_filters.json"

# ===================== 资源配置（适配普通服务器） =====================
num_cores=4          
nthreads=2           
memMB=16000          
parallel_jobs=2      

# ===================== 目录创建与权限 =====================
mkdir -p "${outputdir}/logs" "${workingdir}" "${TEMPLATEFLOW_DIR}"
mkdir -p "$(dirname "${bids_filter_file}")"
chown -R "$(id -u):$(id -g)" "${outputdir}" "${workingdir}"

# ===================== 预定义BIDS过滤参数 =====================
if [ -f "${bids_filter_file}" ]; then
    BIDS_FILTER_ARG="--bids-filter-file /code/$(basename "${bids_filter_file}")"
else
    BIDS_FILTER_ARG=""
fi

# ===================== 并行处理核心逻辑 =====================
# 导出所有需要的变量
export inputdir outputdir fslicense workingdir num_cores nthreads memMB TEMPLATEFLOW_DIR bids_filter_file BIDS_FILTER_ARG

parallel -j ${parallel_jobs} --linebuffer '
    participant_id="{}"
    echo "=== [$(date)] Starting processing for: ${participant_id} ==="

    subj_workdir="${workingdir}/${participant_id}"
    mkdir -p "${subj_workdir}"
    sub_label=$(echo "${participant_id}" | sed "s/sub-//")

    # 用数组构建docker命令，彻底避免引号问题
    docker_args=(
        --rm
        -u "$(id -u):$(id -g)"
        -v "${fslicense}:/opt/freesurfer/license.txt:ro"
        -v "${inputdir}:/data:ro"
        -v "${outputdir}:/out"
        -v "${subj_workdir}:/scratch"
        -v "${TEMPLATEFLOW_DIR}:/opt/templateflow"
        -v "$(dirname "${bids_filter_file}"):/code:ro"
        -e "TEMPLATEFLOW_HOME=/opt/templateflow"
        nipreps/fmriprep:25.1.4
        /data /out participant
        --participant-label "${sub_label}"
        --work-dir /scratch
        --skip-bids-validation
        ${BIDS_FILTER_ARG}
        --nprocs "${num_cores}"
        --omp-nthreads "${nthreads}"
        --mem-mb "${memMB}"
        --fs-license-file /opt/freesurfer/license.txt
        --output-spaces MNI152NLin2009cAsym:res-2 fsLR:den-91k
        --skull-strip-template MNI152NLin2009cAsym
--cifti-output 91k
        --use-syn-sdc warn
        --fd-spike-threshold 0.5
        --dvars-spike-threshold 1.5
    )

    echo "Running command: docker run ${docker_args[*]}"
    docker run "${docker_args[@]}"

    echo "=== [$(date)] Finished processing for: ${participant_id} ==="
' :::: "${participant_files}"

echo "All participants processing tasks completed!"

四、结果检验
