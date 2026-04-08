#!/bin/bash
set -euo pipefail

# 用途：
# - 对同一个 BIDS 根目录中的 dwi 数据运行 QSIPrep（DTI/dMRI 预处理）
# - 输出写入 derivatives/qsiprep，不会影响 fMRIPrep 的 derivatives/fmriprep
# - 建议配合 qsiprep_bids_filters.json，仅选择 T1w 与 dwi，避免误选其他模态
#
# 运行前需要：
# - Docker
# - GNU parallel
# - ${BIDS_ROOT}/code/participants.txt（每行一个 sub-XXX）

BIDS_ROOT="/home/zhangze/data/gifted_fMRI_BIDS"
inputdir="${BIDS_ROOT}"
outputdir="${BIDS_ROOT}/derivatives/qsiprep"
fslicense="${BIDS_ROOT}/license.txt"
workingdir="${BIDS_ROOT}/tmp/qsiprep_wd"
TEMPLATEFLOW_DIR="${BIDS_ROOT}/templateflow"

participant_files="${BIDS_ROOT}/code/participants.txt"
qsiprep_filter_file="${BIDS_ROOT}/code/qsiprep_bids_filters.json"

# 可通过环境变量覆盖容器镜像，例如：
# QSIPREP_IMAGE=pennlinc/qsiprep:latest bash run_qsiprep.sh
QSIPREP_IMAGE="${QSIPREP_IMAGE:-pennlinc/qsiprep:latest}"

num_cores=4
nthreads=3
memMB=16000
parallel_jobs=2

export TEMPLATEFLOW_AWS_S3_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/templateflow/
export TEMPLATEFLOW_AWS_S3_NO_SIGN_REQUEST=YES
export TEMPLATEFLOW_TIMEOUT=300
export TEMPLATEFLOW_HOME="${TEMPLATEFLOW_DIR}"

mkdir -p "${outputdir}/logs" "${workingdir}" "${TEMPLATEFLOW_DIR}"
mkdir -p "$(dirname "${qsiprep_filter_file}")"
chown -R "$(id -u):$(id -g)" "${outputdir}" "${workingdir}" "${TEMPLATEFLOW_DIR}"

if [ -f "${qsiprep_filter_file}" ]; then
    BIDS_FILTER_ARG="--bids-filter-file /code/$(basename "${qsiprep_filter_file}")"
else
    BIDS_FILTER_ARG=""
    echo "⚠️  未找到qsiprep_bids_filters.json，跳过过滤"
fi

export inputdir outputdir fslicense workingdir num_cores nthreads memMB TEMPLATEFLOW_DIR qsiprep_filter_file BIDS_FILTER_ARG QSIPREP_IMAGE

echo -e "\n========================================"
echo "  QSIPrep 批量处理脚本 (DTI/dMRI)"
echo "========================================"
echo "BIDS根目录: ${inputdir}"
echo "输出目录:   ${outputdir}"
echo "镜像:       ${QSIPREP_IMAGE}"
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
        -v "$(dirname "${qsiprep_filter_file}"):/code:ro"
        -e "TEMPLATEFLOW_HOME=/opt/templateflow"
        -e "TEMPLATEFLOW_AWS_S3_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/templateflow/"
        -e "TEMPLATEFLOW_AWS_S3_NO_SIGN_REQUEST=YES"
        -e "TEMPLATEFLOW_TIMEOUT=300"
        ${QSIPREP_IMAGE}
        /data /out participant
        --participant-label "${sub_label}"
        --work-dir /scratch
        --skip-bids-validation
        ${BIDS_FILTER_ARG}
        --nprocs "${num_cores}"
        --omp-nthreads "${nthreads}"
        --mem-mb "${memMB}"
        --fs-license-file /opt/freesurfer/license.txt
    )

    echo "执行命令: docker run ${docker_args[*]}"
    docker run "${docker_args[@]}"

    echo "=== [$(date)] 完成处理被试: ${participant_id} ==="
' :::: "${participant_files}"

echo -e "\n========================================"
echo "✅ 所有被试处理完成！"
echo "📊 结果目录: ${outputdir}"
echo "========================================"
