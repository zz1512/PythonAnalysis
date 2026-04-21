#!/bin/bash
set -euo pipefail

# ===================== 【国内环境优化：TemplateFlow 镜像配置】 =====================
# 强制使用清华大学镜像，彻底解决 AWS 超时问题
export TEMPLATEFLOW_AWS_S3_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/templateflow/
export TEMPLATEFLOW_AWS_S3_NO_SIGN_REQUEST=YES
# 增加网络超时时间
export TEMPLATEFLOW_TIMEOUT=300

# ===================== 【核心路径配置：已适配新版BIDS】 =====================
BIDS_ROOT="/home/zhangze/data/gifted_BIDS"
inputdir="${BIDS_ROOT}"
outputdir="${BIDS_ROOT}/fmriprep"
fslicense="${BIDS_ROOT}/license.txt"
workingdir="${BIDS_ROOT}/tmp/fmriprep_wd"
TEMPLATEFLOW_DIR="${BIDS_ROOT}/templateflow"
export TEMPLATEFLOW_HOME="${TEMPLATEFLOW_DIR}"  # 绑定本地缓存目录
participant_files="${BIDS_ROOT}/code/participants.txt"
bids_filter_file="${BIDS_ROOT}/code/bids_filters.json"

# ===================== 【资源配置：可根据服务器调整】 =====================
num_cores=6
nthreads=4
memMB=24000
parallel_jobs=2

# ===================== 【目录创建与权限】 =====================
mkdir -p "${outputdir}/logs" "${workingdir}" "${TEMPLATEFLOW_DIR}"
mkdir -p "$(dirname "${bids_filter_file}")"
chown -R "$(id -u):$(id -g)" "${outputdir}" "${workingdir}" "${TEMPLATEFLOW_DIR}"

# ===================== 【BIDS过滤参数配置】 =====================
# 说明：
# - bids_filters.json 仅选择 anat(T1w) 与 func(bold)
# - 即使 BIDS 根目录中存在 dwi/，也不会被筛选进 fMRIPrep 任务
if [ -f "${bids_filter_file}" ]; then
    BIDS_FILTER_ARG="--bids-filter-file /code/$(basename "${bids_filter_file}")"
else
    BIDS_FILTER_ARG=""
    echo "⚠️  未找到bids_filters.json，跳过过滤"
fi

# ===================== 【并行处理核心逻辑】 =====================
export inputdir outputdir fslicense workingdir num_cores nthreads memMB TEMPLATEFLOW_DIR bids_filter_file BIDS_FILTER_ARG

echo -e "\n========================================"
echo "  fMRIprep 批量处理脚本 (国内环境优化版)"
echo "========================================"
echo "BIDS根目录: ${inputdir}"
echo "TemplateFlow 镜像: ${TEMPLATEFLOW_AWS_S3_ENDPOINT}"
echo "并行任务数: ${parallel_jobs}"
echo -e "========================================\n"

parallel -j ${parallel_jobs} --linebuffer '
    participant_id="{}"
    echo "=== [$(date)] 开始处理被试: ${participant_id} ==="

    subj_workdir="${workingdir}/${participant_id}"
    mkdir -p "${subj_workdir}"
    sub_label=$(echo "${participant_id}" | sed "s/sub-//")

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
        -e "TEMPLATEFLOW_AWS_S3_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/templateflow/"
        -e "TEMPLATEFLOW_AWS_S3_NO_SIGN_REQUEST=YES"
        -e "TEMPLATEFLOW_TIMEOUT=300"
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
        --use-syn-sdc warn
        --fd-spike-threshold 0.5
        --dvars-spike-threshold 1.5
    )

    echo "执行命令: docker run ${docker_args[*]}"
    docker run "${docker_args[@]}"

    echo "=== [$(date)] 完成处理被试: ${participant_id} ==="
' :::: "${participant_files}"

echo -e "\n========================================"
echo "✅ 所有被试处理完成！"
echo "📊 结果目录: ${outputdir}"
echo "========================================"
