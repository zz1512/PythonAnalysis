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
