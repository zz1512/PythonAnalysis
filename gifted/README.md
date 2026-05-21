# gifted

本目录包含 Gifted 数据集相关的 BIDS 整理与 BIDS App（fMRIPrep/QSIPrep）批处理脚本。

## 总览

- `organize_bids.sh`：把“已有 NIfTI”按规则复制/整理成 BIDS 目录结构
- `run_fmriprep.sh`：对 BIDS 数据集跑 fMRIPrep（功能/结构）
- `run_qsiprep.sh`：对 BIDS 数据集跑 QSIPrep（DTI/dMRI）
- `bids_filters.json`：fMRIPrep 用的过滤文件（只挑 T1w 与 bold）
- `qsiprep_bids_filters.json`：QSIPrep 用的过滤文件（只挑 T1w 与 dwi）

## 1) organize_bids.sh（已有 NIfTI → BIDS）

用途：
- 当你已经有 `.nii.gz + .json`（例如从 DICOM 转换后得到的 NIfTI）时，用该脚本物理复制并整理为 BIDS。

产物：
- `BIDS_ROOT/sub-<id>/ses-<pre|post>/{anat,func}/...`
- 会生成 BIDS 必需的 `dataset_description.json`

注意：
- 此脚本生成的 task 命名（REST/AUT/P/C）来自你原始 NIfTI 文件名模式。
- 如果你用 `prep_common/dcm2bids_wjq.py` 直接从 DICOM 生成 BIDS，一般不需要再跑 `organize_bids.sh`。

## 2) run_fmriprep.sh（fMRI 预处理）

用途：
- 针对 BIDS 数据集中 `anat/` 和 `func/` 运行 fMRIPrep。

关键点：
- 脚本使用 `bids_filters.json` 过滤输入，只选择：
  - `anat` + `suffix=T1w`
  - `func` + `suffix=bold`
  见 [bids_filters.json](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/gifted/bids_filters.json)
- 即使 BIDS 根目录下同时存在 `dwi/`，也不会被过滤选中；fMRIPrep 也不会把 `dwi` 当 fMRI 处理。

依赖：
- Docker
- GNU parallel
- `participants.txt`：每行一个被试 ID（如 `sub-A001`）

典型运行方式：
- 修改脚本开头的 `BIDS_ROOT`、资源参数（CPU/内存/并行数）
- 确保 `${BIDS_ROOT}/code/participants.txt` 存在
- 执行：
  ```bash
  bash run_fmriprep.sh
  ```

## 3) run_qsiprep.sh（DTI/dMRI 预处理）

用途：
- 针对 BIDS 数据集中 `dwi/`（以及必要时的 `anat/`）运行 QSIPrep。

关键点：
- 脚本使用 `qsiprep_bids_filters.json` 过滤输入，只选择 `T1w` 与 `dwi`，从而避免把 BOLD 等其他模态拉入 dMRI 流程。
- 与 `run_fmriprep.sh` 可以指向同一个 BIDS 根目录；输出会写到不同的 derivatives 子目录：
  - fMRIPrep：`derivatives/fmriprep`
  - QSIPrep：`derivatives/qsiprep`

运行方式：
```bash
bash run_qsiprep.sh
```

可选：
- 通过环境变量指定 QSIPrep 容器镜像（默认 `pennlinc/qsiprep:latest`）：
  ```bash
  QSIPREP_IMAGE=pennlinc/qsiprep:latest bash run_qsiprep.sh
  ```

## 4) participants.txt 怎么生成？

这两个运行脚本都依赖 `${BIDS_ROOT}/code/participants.txt`（每行一个 `sub-XXX`）。

生成方式之一（示例命令）：

```bash
cd "$BIDS_ROOT"
find . -maxdepth 1 -type d -name "sub-*" -printf "%f\n" | sort > code/participants.txt
```

如果你是用 `prep_common/dcm2bids_wjq.py` 输出 BIDS，可以先生成 BIDS，再用上面命令生成 participants.txt，然后分别跑 fMRIPrep 与 QSIPrep。

