# prep_common

本目录用于存放可复用的数据预处理脚本与工具函数。

## dcm2bids_wjq.py

将特定目录结构的 DICOM 数据批量转换为 BIDS 目录结构与命名（依赖 `dcm2niix`），并对输出文件做重命名与 JSON 修正，方便后续用 fMRIPrep / QSIPrep 等 BIDS App 处理。

### 产物（输出目录 out_dir）

- 根目录（可选生成）
  - `README.txt`
  - `dataset_description.json`
  - `participants.json`
- 每个被试（示例：`sub-A001`）
  - 结构像：`sub-A001/ses-pre/anat/sub-A001_ses-pre_T1w.nii.gz`（以及 `.json`）
  - 静息态：`sub-A001/ses-pre/func/sub-A001_ses-pre_task-rest_..._bold.nii.gz`（以及 `.json`）
  - 任务态：`sub-A001/ses-pre/func/sub-A001_ses-pre_task-..._bold.nii.gz`（以及 `.json`）
  - 扩散：`sub-A001/ses-pre/dwi/sub-A001_ses-pre_dwi.nii.gz`（以及 `.bval/.bvec/.json`）

说明：
- session 目录默认是 `ses-pre` / `ses-post`（可用参数自定义）。
- 如果同一 session 下存在多次重复扫描，建议开启 `--auto-run`，自动追加 `run-01/run-02...`，避免命名冲突导致跳过。

### 运行前准备

- 安装 `dcm2niix` 并确保可执行文件可用：
  - macOS（Homebrew）：`brew install dcm2niix`
  - 或者下载二进制并用 `--dcm2niix /path/to/dcm2niix` 指定路径

### 用法

仅打印将执行的动作（不真正转换/改名）：

```bash
python -m prep_common.dcm2bids_wjq \
  --dicom-dir /path/to/Dicom \
  --dry-run
```

实际执行（推荐开启自动 run 防冲突）：

```bash
python -m prep_common.dcm2bids_wjq \
  --dicom-dir /path/to/Dicom \
  --auto-run
```

自定义输出目录与 session 标签（对齐 gifted 的 pre/post 习惯）：

```bash
python -m prep_common.dcm2bids_wjq \
  --dicom-dir /path/to/Dicom \
  --out-dir /path/to/gifted_fMRI_BIDS \
  --ses-labels pre post \
  --auto-run
```

仅处理部分被试（被试目录名等于 DICOM 根目录下的子目录名）：

```bash
python -m prep_common.dcm2bids_wjq \
  --dicom-dir /path/to/Dicom \
  --subjects A001 A002 \
  --auto-run
```

### 输入目录结构假设（可兼容探测）

脚本按每个被试目录 `dicom_dir/<subject>/` 扫描，核心假设是：

- DWI：`session1/DTI_mx_137` 与 `session2/DTI_mx_137`
- Rest：`session1/RfMRI*` 与 `session2/RfMRI*`
- Task：优先 `session1/TfMRI*`、`session2/TfMRI*`；否则兜底 `TfMRI*` 归入 `ses-pre`
- Anat：优先 `session1/anat`、`session2/anat`；否则兜底 `anat` 归入 `ses-pre`

如果你的目录结构不同，建议先用 `--dry-run` 看日志，再按实际情况做小改动。

