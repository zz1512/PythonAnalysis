# 从核磁实验结束到跑完 fMRIPrep：给新手的 fMRI 预处理全流程指南

> 适用对象：第一次自己整理 fMRI 数据、第一次接触 BIDS、第一次跑 `fMRIPrep` 的同学。  
> 本文结合了你项目里的本地参考脚本,并参考了 BIDS / fMRIPrep / dcm2niix 官方文档。  
> 截至 **2026-04-16**，fMRIPrep stable 文档显示的最新稳定版为 **25.2.5（2026-03-10）**。你项目脚本里实际使用的是 `nipreps/fmriprep:25.1.4`，如果一个课题已经开跑，建议整批数据保持同一版本，不要中途混用版本。

## 1. 先用一句人话说清楚：我们到底在做什么

fMRI 预处理，不是“把数据变漂亮”，而是把扫描仪吐出来的原始影像，整理成：

1. 结构规范统一
2. 元数据完整可追溯
3. 能被标准工具正确识别
4. 空间和时间上的常见误差尽量被校正
5. 为后续一阶 GLM、功能连接、MVPA、RSA 等分析做好输入准备

最核心的一条链路就是：

```text
扫描结束 -> 数据预处理
-> 第一部分：
-> 导出原始 DICOM / zip
-> 整理被试和序列
-> dcm2niix 转成 NIfTI + JSON
-> 第二部分：
-> 按 BIDS 规范改名和归档
-> 补齐关键 JSON 元数据
-> bids-validator 检查
-> 准备 license / workdir / outputdir / TemplateFlow
-> 运行 fMRIPrep
-> 检查 HTML 报告和 confounds
```

如果你记住一句话，就是：

> `fMRIPrep` 用的是 **合格的 BIDS 数据集**”。

---

## 2. 扫描一结束，第一时间该做什么

很多人以为预处理从 `dcm2niix` 开始，其实不是。真正最容易出问题的是扫描刚结束的这一步。

### 2.1 你至少要拿到这些东西

1. 原始 DICOM，或者扫描中心导出的原始 `zip`
2. 被试清单
3. 扫描序列记录表
4. 行为任务日志
5. 重要扫描参数记录

### 2.2 强烈建议同时记录这些参数

因为后面补 BIDS JSON、补场图信息、排查 `fMRIPrep` 报错时，经常会用到：

| 参数 | 为什么重要 |
| --- | --- |
| TR (`RepetitionTime`) | 功能像时间采样间隔，几乎所有后续分析都会用到 |
| TE (`EchoTime`) | 决定 BOLD 对比特性，也影响部分场图处理 |
| FlipAngle | 基础采集参数 |
| 多带因子 (`MultibandAccelerationFactor`) | 判断切片时序和采集策略 |
| 切片数 / 切片顺序 | 关系到 `SliceTiming` 是否正确 |
| 相位编码方向 | 关系到几何畸变校正 |
| 是否采了 fieldmap / 反向相位编码 EPI | 关系到 SDC 是否可靠 |
| 每个 run 对应什么任务 | 后面命名 `task-xxx` 时必须知道 |
| 是否删掉了 dummy scans | 事件文件 `onset=0` 的定义会受影响 |

### 2.3 重要坑点

1. 不要只拿 `.nii`，一定尽量保留最原始的 DICOM 或压缩包。
2. 不要只记“这是任务 run1 / run2”，要记清楚它们在扫描机上的真实序列名。
3. 行为日志和影像文件名一定要能对上 run，否则后面 `events.tsv` 很容易错位。

---

## 3. 第一步：把原始压缩包和 DICOM 整理干净

`T1_dcm2bids_shengyu.m` 做的第一件事，不是预处理，而是**整理原始数据**。它的逻辑很典型：

1. 建 `RAW`、`Dicom`、`Nifti` 这几个目录
2. 把 `RAW` 里的 `zip` 解压
3. 找到里面真正的 `DICOM` 文件夹
4. 按顺序重命名成 `001`、`002`、`003` 这种被试目录
5. 再对每个被试做 DICOM 转换和后续 BIDS 归档

这一步的目的不是“好看”，而是为了防止：

1. 同一个被试的数据散落在多个导出包里
2. 研究助理手工复制时漏文件
3. 扫描中心导出的目录层级过深，后续脚本无法稳定定位

### 3.1 推荐的原始目录结构

```text
project/
├─ RAW/
│  ├─ 20260401_sub001.zip
│  ├─ 20260401_sub002.zip
│  └─ ...
├─ Dicom/
│  ├─ 001/
│  ├─ 002/
│  └─ ...
└─ Nifti/
```

### 3.2 这一步要检查什么

1. 每个被试是否都有结构像 `T1w`
2. 每个被试是否都有预期数量的功能 run
3. 有没有多余的 localizer、survey、ADC、FA 之类无关序列
4. 有没有明显缺失 run、重复 run、扫描中断

### 3.3 重要坑点

1. 不同扫描中心导出的 zip 结构不一样，别假设解压后一定是 `某目录/DICOM/`。
2. 如果你按时间排序自动编号，必须确认 zip 顺序确实和被试顺序一致。
3. 建议保留原始 zip，不要在唯一副本上反复解压和重命名。

---

## 4. 第二步：用 dcm2niix 把 DICOM 转成 NIfTI + JSON

这一步是把扫描仪格式转成分析软件更常用的格式。

### 4.1 dcm2niix 是做什么的

`dcm2niix` 的职责是：

1. 把 DICOM 转成 `NIfTI` (`.nii` / `.nii.gz`)
2. 尽可能从 DICOM 里提取元数据，写进同名 `.json` sidecar

这也是为什么大家通常不会手工从 DICOM 直接跑分析，而是先走 `dcm2niix`。

### 4.2 你项目里的典型命令

`T1_dcm2bids_shengyu.m` 里实际调用的是：

```bash
dcm2niix -z y -f %p_%t_%s -o <输出目录> <DICOM目录>
```

### 4.3 这些参数是什么意思

| 参数 | 含义 | 建议 |
| --- | --- | --- |
| `-z y` | 输出压缩后的 `.nii.gz` | 推荐开启，省空间 |
| `-f %p_%t_%s` | 输出文件名规则：协议名 + 时间 + 序列号 | 适合转换阶段做“保守命名” |
| `-o` | 输出目录 | 每个被试单独目录更安全 |

### 4.4 为什么转换阶段不直接命成 BIDS 名

新手最稳妥的做法，是先让 `dcm2niix` 保留更多原始序列信息，再在下一步单独做 BIDS 命名。  
原因很简单：如果一开始就命成 `sub-001_task-xxx_run-1_bold.nii.gz`，但你后面发现 run 对应关系搞错了，排查会更麻烦。

### 4.5 转换后的产物

一个典型序列会得到：

```text
some_series_name.nii.gz
some_series_name.json
```

其中：

1. `nii.gz` 是图像本体
2. `json` 是采集元数据

后面的 BIDS 和 `fMRIPrep` 其实两样都要用。

### 4.6 重要坑点

1. **别只拷贝 `.nii.gz`，漏掉 `.json`。**
2. `dcm2niix` 自动提取的元数据不一定完整，后面仍然要人工核对。
3. 不同厂家、不同序列下，`SliceTiming`、`PhaseEncodingDirection`、`EffectiveEchoSpacing` 的提取完整度不一样。
4. 本地脚本里会删除 `Survey`、`localizer` 之类序列，这是合理的，但一定要先确认不是你真正要用的数据。

---

## 5. 第三步：把 NIfTI 整成 BIDS 结构

这一步是整个流程的分水岭。  
**不是有 `nii.gz` 就能跑 `fMRIPrep`，而是要有“合格的 BIDS 数据集”。**

### 5.1 最小可用 BIDS 结构长什么样

```text
BIDS_ROOT/
├─ dataset_description.json
├─ participants.tsv
├─ participants.json           # 可选但推荐
├─ sub-001/
│  ├─ anat/
│  │  ├─ sub-001_T1w.nii.gz
│  │  └─ sub-001_T1w.json
│  └─ func/
│     ├─ sub-001_task-viewing_run-1_bold.nii.gz
│     ├─ sub-001_task-viewing_run-1_bold.json
│     ├─ sub-001_task-viewing_run-1_events.tsv
│     ├─ sub-001_task-viewing_run-2_bold.nii.gz
│     ├─ sub-001_task-viewing_run-2_bold.json
│     └─ sub-001_task-viewing_run-2_events.tsv
└─ code/
   ├─ participants.txt
   └─ bids_filters.json
```

### 5.2 你本地 MATLAB 脚本已经在做什么

`T1_dcm2bids_shengyu.m` 已经体现了一个常见做法：

1. 先按序列名识别哪个是 T1w
2. 识别哪些功能像对应 run-1、run-2
3. 把 T1w 放到 `anat/`
4. 把功能像放到 `func/`

思路是对的，但新手要特别明白：

> “能匹配到文件名”不等于“语义一定正确”。  
> 最后仍要人工确认每个 run 到底是什么任务、是否顺序正确。

### 5.3 功能像文件名里为什么一定要有 `task-xxx`

`fmriprep_method.pdf` 已经专门强调了这一点：  
如果功能文件名里没有任务信息，后续工具识别会出问题。

推荐格式：

```text
sub-094_task-viewing_run-1_bold.nii.gz
sub-094_task-viewing_run-1_bold.json
```

如果是静息态：

```text
sub-094_task-rest_run-1_bold.nii.gz
```

### 5.4 `dataset_description.json` 至少要有

```json
{
  "Name": "Your Study Name",
  "BIDSVersion": "1.10.1"
}
```

你本地脚本里已经自动生成过这个文件，这一步是必须的。

### 5.5 `participants.tsv` 和 `participants.json`

`participants.tsv` 不是跑 `fMRIPrep` 的绝对硬门槛，但强烈建议有。  
如果你后面做组分析、质量控制、排除被试管理，它会非常方便。

例如：

```tsv
participant_id	age	sex	group
sub-001	22	F	control
sub-002	24	M	experiment
```

### 5.6 `events.tsv` 是做什么的

如果是任务态 fMRI，后面做 GLM 时通常一定要有 `events.tsv`。  
最基本的列是：

1. `onset`
2. `duration`

常见还会加：

1. `trial_type`
2. `response_time`
3. 其他行为变量

最小示例：

```tsv
onset	duration	trial_type
0.5	2.0	word
4.5	2.0	picture
8.5	2.0	rest
```

### 5.7 重要坑点

1. **任务态一定要区分 `task` 和 `run`，不要只写 `run-1`。**
2. 一个 `bold.nii.gz` 必须有对应的 `bold.json`。
3. `events.tsv` 的 `onset=0` 指的是“第一个保留下来的 volume”，不是“扫描机刚开始响”的时间。
4. 如果前面删掉了 dummy scans，`events.tsv` 的起点也要一起对应调整。

---

## 6. 第四步：补齐最关键的 JSON 元数据

很多 `fMRIPrep` 报错，本质上不是软件坏了，而是元数据不完整。

你本地 `fmriprep_method.pdf` 里最重要的经验，其实就是在补这几类字段。

### 6.1 新手必须最关注的字段

| 字段 | 作用 | 什么时候必须认真核对 |
| --- | --- | --- |
| `TaskName` | 任务名称 | 所有任务态 |
| `RepetitionTime` | TR | 所有 BOLD |
| `SliceTiming` | 切片采样时间 | 需要做 slice timing correction 时 |
| `PhaseEncodingDirection` | 相位编码方向 | 做畸变校正时尤其重要 |
| `EffectiveEchoSpacing` | 有效回波间隔 | 场图 / 畸变校正相关 |
| `TotalReadoutTime` | 总读出时间 | 场图 / 畸变校正相关 |
| `B0FieldIdentifier` / `B0FieldSource` | 场图与功能像的关联 | 有 fieldmap 时强烈推荐 |
| `IntendedFor` | 旧式场图关联方式 | 兼容旧工具时可一起保留 |

### 6.2 `RepetitionTime`

这是每个 volume 之间的时间间隔，单位是秒。  
比如 TR = 2 秒，就写：

```json
"RepetitionTime": 2.0
```

### 6.3 `SliceTiming`

这是**每一层切片在一个 TR 内实际采样的时间点**，不是简单的“切片顺序编号”。  
如果这个字段正确，`fMRIPrep` 才能做 slice-timing correction。

你本地文档里用了这样的例子：

```json
"SliceTiming": [0,1,0.0666667,1.06667,0.133333,1.13333,...]
```

这个思路没有问题，但要注意：

1. 这组值是**该序列专属**的，不是所有任务都通用
2. 多带采集时，切片时间往往不是简单升序
3. 如果有 `SliceEncodingDirection` 且带负号，例如 `k-`，解释顺序还会反过来

### 6.4 `PhaseEncodingDirection`

这是 BIDS 里的 `i / i- / j / j- / k / k-`，它描述的是 **NIfTI 轴方向**，不是简单写成 `AP`、`PA`、`LR` 就行。

这是一个高频大坑：

> DICOM 里的相位编码方向，和 BIDS JSON 里的 `i/j/k`，不是一回事。  
> 不能凭经验把 `AP` 直接写成 `j`。

你必须结合：

1. 转换后的 NIfTI 方向
2. 厂家序列说明
3. dcm2niix 自动输出结果

一起核对。

### 6.5 `EffectiveEchoSpacing` 和 `TotalReadoutTime`

这两个字段是做 EPI 畸变校正时的关键字段。  
你本地文档特别提醒了一种常见情况：

```json
"EstimatedEffectiveEchoSpacing"
"EstimatedTotalReadoutTime"
```

有些转换结果里是 `Estimated...`，但后续工具更常期待的是：

```json
"EffectiveEchoSpacing"
"TotalReadoutTime"
```

本地脚本里已经给了用 `jq` 批量重命名字段的办法，这个经验非常实用。

但要注意：

1. **先确认数值本身没问题，再改 key 名。**
2. 不要“为了不报错”瞎填一个数字。
3. 这两个字段常受 partial Fourier、相位过采样、重建矩阵等影响，不是所有扫描都能拿一个固定值通吃。

### 6.6 `TaskName`

文件名里的 `task-viewing` 只是标签；JSON 里的 `TaskName` 是任务的正式名称。  
例如：

```json
"TaskName": "viewing"
```

### 6.7 重要坑点

1. **不要把一名被试的 JSON 改对后，机械复制到所有被试。**  
   同一个课题通常扫描参数一致，但仍要核对是否有 run、session、设备版本差异。
2. `SliceTiming` 错了，比没有还危险，因为它会让软件“自信地做错事”。
3. `PhaseEncodingDirection` 错了，会直接导致畸变校正方向反了。
4. `Estimated...` 改名不是万能补丁，如果原始值就不对，改完照样错。

---

## 7. 第六步：先跑 bids-validator，再跑 fMRIPrep

你本地文档里已经写了这一步，这个顺序是对的。

### 7.1 典型命令

```bash
bids-validator /path/to/BIDS_ROOT
```

### 7.2 新手怎么理解 validator 结果

1. `error`：先修，再跑 `fMRIPrep`
2. `warning`：不一定致命，但要逐条看
3. “能跑起来”不等于“数据合规”

### 7.3 高发报错

1. 缺 `dataset_description.json`
2. 功能文件名没有 `task-xxx`
3. 有 `bold.nii.gz` 但没有对应 `bold.json`
4. `events.tsv` 文件名与 run 对不上
5. 场图文件结构不合规

### 7.4 关于 `--skip-bids-validation`

`run_fmriprep.sh` 里用了：

```bash
--skip-bids-validation
```

这在**已经单独跑过 validator**、并且为了节省每次重复检查时间时是合理的。  
但如果你是第一次整理数据：

> 不建议一上来就跳过 BIDS 检查。

---

## 8. 第七步：准备运行环境

跑 `fMRIPrep` 前，不只是“有 Docker 就行”，还要把几个目录和资源想清楚。

### 8.1 最常见的几个路径

 `run_fmriprep.sh` 的组织方式很清晰：

```bash
BIDS_ROOT="/home/zhangze/data/gifted_fMRI_BIDS"
inputdir="${BIDS_ROOT}"
outputdir="${BIDS_ROOT}/derivatives/fmriprep"
fslicense="${BIDS_ROOT}/license.txt"
workingdir="${BIDS_ROOT}/tmp/fmriprep_wd"
TEMPLATEFLOW_DIR="${BIDS_ROOT}/templateflow"
participant_files="${BIDS_ROOT}/code/participants.txt"
bids_filter_file="${BIDS_ROOT}/code/bids_filters.json"
```

这几个路径分别是：

| 路径 | 作用 |
| --- | --- |
| `inputdir` | 输入 BIDS 根目录 |
| `outputdir` | 预处理结果输出目录 |
| `workingdir` | 中间缓存和临时文件目录 |
| `fslicense` | FreeSurfer license |
| `TEMPLATEFLOW_DIR` | 模板缓存目录 |
| `participants.txt` | 要处理哪些被试 |
| `bids_filters.json` | 只让 fMRIPrep 读哪些数据 |

### 8.2 FreeSurfer license

`fMRIPrep` 用到了 FreeSurfer 工具，所以需要 license 文件。  
没有的话经常会在运行开始不久就报错。

### 8.3 `bids_filters.json` 有什么用

你项目里的文件内容很简洁：

```json
{
  "t1w": {
    "datatype": "anat",
    "suffix": "T1w",
    "extension": [".nii.gz"]
  },
  "bold": {
    "datatype": "func",
    "suffix": "bold",
    "extension": [".nii.gz"]
  }
}
```

作用是：只选 `anat/T1w` 和 `func/bold`。  
这样做的好处是：

1. 避免不相关文件干扰
2. 提高稳定性
3. 大项目里更容易控制输入

### 8.4 `participants.txt`

每行一个被试，比如：

```text
sub-093
sub-094
sub-097
```

好处是可以分批跑、断点续跑、跳过坏数据。

### 8.5 TemplateFlow 缓存

`fMRIPrep` 需要模板资源。  
如果第一次运行时没有网络，或者算节点不能联网，就可能卡住。

你本地脚本用的是国内镜像环境变量，这是一种很实用的工程化方案：

```bash
export TEMPLATEFLOW_AWS_S3_ENDPOINT=https://mirrors.tuna.tsinghua.edu.cn/templateflow/
export TEMPLATEFLOW_AWS_S3_NO_SIGN_REQUEST=YES
export TEMPLATEFLOW_TIMEOUT=300
```

同时，官方 FAQ 也明确说明：

1. TemplateFlow 首次拉取模板时需要网络
2. 可以预先缓存模板
3. 即使你自定义 `--output-spaces`，`fMRIPrep` 仍会内部依赖某些模板

### 8.6 重要坑点

1. `workingdir` 常常会非常大，磁盘不够比内存不够更常见。
2. 如果 `outputdir` 和 `workingdir` 在慢盘上，速度会明显下降。
3. 跑 `--cifti-output 91k` 时，还会额外依赖相关模板和表面资源。

---

## 9. 第八步：正式运行 fMRIPrep

### 9.1 你项目里最实战的命令框架

下面是根据 `gifted/run_fmriprep.sh` 整理后的核心 Docker 命令：

```bash
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "${fslicense}:/opt/freesurfer/license.txt:ro" \
  -v "${inputdir}:/data:ro" \
  -v "${outputdir}:/out" \
  -v "${workingdir}:/scratch" \
  -v "${TEMPLATEFLOW_DIR}:/opt/templateflow" \
  -v "$(dirname "${bids_filter_file}"):/code:ro" \
  -e "TEMPLATEFLOW_HOME=/opt/templateflow" \
  nipreps/fmriprep:25.1.4 \
  /data /out participant \
  --participant-label 094 \
  --work-dir /scratch \
  --bids-filter-file /code/bids_filters.json \
  --nprocs 4 \
  --omp-nthreads 3 \
  --mem-mb 16000 \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin2009cAsym:res-2 fsLR:den-91k \
  --skull-strip-template MNI152NLin2009cAsym \
  --cifti-output 91k \
  --use-syn-sdc warn \
  --fd-spike-threshold 0.5 \
  --dvars-spike-threshold 1.5
```

### 9.2 这些参数怎么理解

| 参数 | 作用 | 新手建议 |
| --- | --- | --- |
| `--participant-label` | 只跑指定被试 | 调试时非常好用 |
| `--work-dir` | 中间文件目录 | 一定放到大磁盘 |
| `--bids-filter-file` | 限制输入文件类型 | 大项目推荐 |
| `--nprocs` | 总线程上限 | 不要超过机器承受能力 |
| `--omp-nthreads` | 单进程线程数 | 常设为 `2-8` |
| `--mem-mb` | 内存上限 | 宁可保守，不要瞎拉满 |
| `--output-spaces` | 输出空间 | 体素分析常要 MNI；表面分析可加 fsLR |
| `--cifti-output 91k` | 额外输出 CIFTI grayordinates | 只有你确实会用 CIFTI 时再开 |
| `--use-syn-sdc warn` | 无场图时尝试 fieldmap-less SDC | 无场图项目常用 |
| `--fd-spike-threshold` | 标记高运动时间点 | 影响 confounds 里的 spike regressor |
| `--dvars-spike-threshold` | 标记强信号跳变时间点 | 同上 |

### 9.3 `--output-spaces` 怎么选

你本地脚本里是：

```bash
--output-spaces MNI152NLin2009cAsym:res-2 fsLR:den-91k
```

意思是同时输出：

1. 2mm 分辨率的 MNI 标准空间体数据
2. `fsLR 91k` 的表面/CIFTI 数据

如果你后面主要做体素水平 GLM，这个设置很好用。  
如果你完全不做表面分析，也可以不加 `fsLR:den-91k` 和 `--cifti-output 91k`，减少时间和资源消耗。

### 9.4 `--skull-strip-template`

官方默认的 skull-strip template 是 `OASIS30ANTs`。  
你本地脚本改成了：

```bash
--skull-strip-template MNI152NLin2009cAsym
```

这不是绝对错误，但要注意：

1. 这是一个**研究组配置选择**
2. 一旦确定，最好整批数据保持一致
3. 不要同一课题一半用默认、一半手动改

### 9.5 并行怎么配

你本地脚本用了：

```bash
num_cores=4
nthreads=3
memMB=16000
parallel_jobs=3
```

这意味着如果三名被试并行跑，粗略资源预算接近：

1. CPU：`4 x 3 = 12` 核级别
2. 内存：`16GB x 3 = 48GB` 以上
3. 外加 Docker / 系统 / 文件缓存开销

所以一个很常见的坑是：

> 单个被试看上去只要 16GB，但你一并行 3 个，机器就爆了。

### 9.6 重要坑点

1. **不要把 `parallel_jobs`、`--nprocs`、`--omp-nthreads` 分开看。**
2. `--sloppy` 只是测试提速选项，不适合正式结果。
3. `--skip-bids-validation` 只适合你已提前验证过数据。
4. 同一个课题中途升级 `fMRIPrep` 版本，会影响可重复性。
5. 如果运行到一半卡住或出现 `BrokenProcessPool` 之类问题，常见原因是内存不足；这时除了增加内存，也可以考虑 `--low-mem`，但代价是 `workingdir` 会更占磁盘。

---

## 10. 第九步：fMRIPrep 在内部到底做了什么

理解这一步很重要，不然你会把它当成“黑盒子”。

### 10.1 结构像（T1w）部分

大致会做：

1. 脑提取（skull stripping）
2. 组织分割（GM / WM / CSF）
3. 配准到标准空间（如 MNI）
4. 如果开启 FreeSurfer，则做皮层表面重建

### 10.2 功能像（BOLD）部分

大致会做：

1. 生成 BOLD 参考图
2. 估计头动参数
3. 若有 `SliceTiming`，做 slice-timing correction
4. 若有 fieldmap 或可做 fieldmap-less SDC，做畸变校正
5. BOLD 配准到 T1w
6. 再重采样到标准空间
7. 生成 confounds

### 10.3 一个关键思想：尽量减少重复插值

官方文档特别强调，`fMRIPrep` 会把多个变换拼接起来，尽量一次性重采样，减少信息损失。  
这也是它一直被广泛采用的重要原因之一。

### 10.4 新手要知道的边界

`fMRIPrep` 不是：

1. 一阶 GLM 软件
2. 自动去噪一键神器
3. 自动帮你完成被试排除的工具

它做的是**高质量标准化预处理**，以及提供下游分析要用的 confounds。

---

## 11. 第十步：跑完后怎么看结果

### 11.1 先看 HTML 报告

官方输出里最值得先看的，是每个被试的 HTML report。  
它能快速看出：

1. 脑提取有没有明显失败
2. T1 到 MNI 是否对齐
3. BOLD 到 T1 是否对齐
4. 畸变校正是否合理
5. 头动和 carpet plot 是否异常

### 11.2 再看关键输出文件

常见输出包括：

1. `desc-preproc_bold.nii.gz`
2. `desc-brain_mask.nii.gz`
3. `desc-confounds_timeseries.tsv`
4. `desc-confounds_timeseries.json`
5. 表面/CIFTI 输出（如果开启）

### 11.3 `desc-confounds_timeseries.tsv` 怎么理解

这是下游分析时非常重要的噪声回归候选表。  
常见列包括：

1. `trans_x/y/z`
2. `rot_x/y/z`
3. `framewise_displacement`
4. `dvars`
5. `white_matter`
6. `csf`
7. `global_signal`
8. `a_comp_cor_*`
9. `t_comp_cor_*`
10. `non_steady_state_outlier_*`

### 11.4 一个特别重要的误区

> `fMRIPrep` 会生成很多 confounds，  
> 但这不等于你应该把所有列一股脑全部塞进 GLM。

官方文档明确提醒：

1. confounds 列可能超过 100 列
2. 不要全塞
3. 应该根据分析目标选择合适的去噪策略

### 11.5 重要坑点

1. 报告页“能打开”不等于“结果没问题”。
2. `desc-preproc_bold` 不是“已经完成统计所需一切预处理”的终点。
3. confounds 的选择本身也是分析决策，不是机械步骤。

---

## 12. 官方资料与参考链接

### fMRIPrep

1. fMRIPrep stable 首页：<https://fmriprep.org/en/stable/index.html>
2. Usage Notes：<https://fmriprep.org/en/stable/usage.html>
3. Processing pipeline details：<https://fmriprep.org/en/stable/workflows.html>
4. Outputs：<https://fmriprep.org/en/stable/outputs.html>
5. FAQ：<https://fmriprep.org/en/stable/faq.html>

### BIDS

1. MRI 数据规范：<https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html>
2. Events 文件规范：<https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html>

### dcm2niix

1. dcm2niix 官方仓库：<https://github.com/rordenlab/dcm2niix>
