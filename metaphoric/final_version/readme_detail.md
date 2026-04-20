# metaphoric/final_version 教学版详解

`README.md` 负责回答“怎么跑”；本文件负责回答“为什么这么跑、每一步在论文里说明什么、结果该怎么读”。

适合读者：
- 心理学/认知神经科学初学者
- 想先理解研究故事，再去跑脚本的人
- 能看懂表格和目录，但暂时不想先钻进源码的人

不适合把它当成什么：
- 不是统计教材完整版
- 不是逐行源码注释
- 不是最终论文定稿，实际文件名仍以脚本输出为准

## 1页总览

### 研究问题

这套分析主线想回答一个核心问题：

**当被试学会新的“隐喻联结”时，大脑表征会不会比学会“空间联结”时发生更强的重组？**

这里的“重组”不是只看“哪个脑区更亮”，而是看：
- 行为上，隐喻条件是不是学得更好、记得更牢
- 激活上，学习阶段哪些脑区更参与
- 表征上，同一类刺激在学习前后是不是变得更像、更稳定、或几何结构发生压缩
- 脑和行为之间，这些变化能不能互相解释

一句话故事线：

**行为告诉你“有没有学会”，GLM 告诉你“哪里在参与”，RSA 告诉你“表征关系怎么变”，RD/GPS 告诉你“几何结构怎么变”，脑-行为关联告诉你“这些神经变化和表现有没有同向”。**

### 实验设计

可以把实验拆成三段：

1. 前测 `run-1, run-2`
- 还没系统学习前，先测刺激在脑中的初始表征。
- 这些 run 会进入前测 LSS，并被堆叠成 `pre_yy.nii.gz`、`pre_kj.nii.gz` 等 patterns。

2. 学习阶段 `run-3, run-4`
- 被试真正建立联结的阶段。
- 这里既可以做学习阶段 LSS，也可以做常规 GLM。
- GLM 更像“定位器”，回答哪些脑区更响应 `yy > kj`。

3. 后测 `run-5, run-6`
- 学完以后再次测量。
- 和前测一样做 LSS，最后形成 `post_yy.nii.gz`、`post_kj.nii.gz`。

4. 行为测验 `run-7`
- 看记忆、反应时等行为表现。
- 它不是脑数据本身，但能回答“学到的东西有没有转化成可测行为收益”。

### 条件如何理解

当前主线重点是两类条件：
- `yy`：隐喻相关联结
- `kj`：空间相关联结

脚本里原始 trial type 可能长这样：
- `yyw`, `yyew`
- `kjw`, `kjew`

在下游很多地方会被统一归并成：
- `yy`
- `kj`

这样做是为了让下游文件名固定、分析口径统一。

### 数据结构

最重要的几类输入：
- `data_events/`
  - 每个被试、每个 run 的事件表 `*_events.tsv`
  - 关键列常见有 `onset`, `duration`, `trial_type`, `pic_num`
- `Pro_proc_data/`
  - 预处理后的 BOLD 数据
  - 也包含 confounds，例如头动、噪声回归项
- `stimuli_template.csv`
  - RSA 用的刺激顺序模板
  - 它不是可有可无的“参考表”，而是防止 trial 错位的关键锚点

最重要的中间产物：
- `lss_betas_final/`
  - 每个 trial 的 beta 图
  - 这是后续表征分析最重要的上游
- `pattern_root/`
  - 把很多单-trial beta 按 `时间 x 条件` 堆成 4D 文件
  - 例如 `pre_yy.nii.gz`
- `roi_masks_final/`
  - 你想在哪些脑区里提取 pattern，就需要这些 mask

最重要的输出：
- `glm_analysis_fwe_final/`
- `rsa_results_optimized/`
- `rd_results/`
- `gps_results/`
- `brain_behavior_results/`

### 证据链

如果把这套分析写成论文结果段，可以按下面理解：

1. 行为：隐喻条件是否比空间条件表现更好
2. GLM：学习时哪些脑区对隐喻联结更敏感
3. RSA：这些脑区里的表征关系，是否从前测到后测发生了更强变化
4. RD/GPS：这种变化是不是体现为几何压缩、稳定性提升或维度改变
5. 脑-行为关联：神经变化大的被试，行为上是否也学得更好
6. 机制扩展：如果再加 gPPI、有效连接、中介，可以开始讨论“可能怎么实现”

## 主线、控制、扩展

### 核心主线

最推荐先跑通的主线：

1. 行为
2. 前后测 LSS
3. 学习阶段 GLM
4. stack patterns
5. RSA + LMM
6. RD/GPS
7. 脑-行为关联

如果你只想先拿到一条足够完整、能讲故事的结果线，这几步足够。

### 控制验证

推荐但不是第一次复现必须跑：

- baseline 控制

它回答的是：

“前后变化是不是只是重复暴露带来的普通熟悉化，而不是隐喻学习本身带来的特异性变化？”

### 机制扩展

可选加分项：
- learning dynamics
- gPPI
- effective connectivity
- mediation
- lateralization
- BF10 / bootstrap CI

这些更适合在主线结果已经稳定后再做。

## 方法入门

下面每个方法都用“1 句话定义 + 1 个最常见误用”来讲。

### GLM

一句话定义：
- 把每个实验条件建成回归量，看某个条件是否比另一个条件引起更强 BOLD 响应。

它最擅长回答：
- “哪里更激活”

它不直接回答：
- “这个脑区里表征关系怎么变了”

最常见误用：
- 把 GLM 显著当成“已经证明表征变化”。GLM 更像定位，不等于表征分析。

### LSS

一句话定义：
- 对每个 trial 单独建模，得到每个 trial 自己的 beta 图。

它最擅长回答：
- “给后续 RSA/MVPA/RD/GPS 准备干净的单 trial pattern”

为什么不用普通 GLM 的条件平均 beta：
- 因为 RSA/MVPA 需要 trial 级别信息，条件平均会把很多细节抹平。

最常见误用：
- 对单 trial beta 做平滑再拿去 RSA/MVPA。平滑会把相邻体素模式抹得更像，可能污染多体素模式信息。

### MVPA

一句话定义：
- 用分类或解码方法看大脑模式能不能区分条件。

它最擅长回答：
- “脑模式里有没有可解码的信息”

最常见误用：
- 训练集和测试集没有严格隔离，导致数据泄漏，准确率虚高。

### RSA

一句话定义：
- 比较刺激之间的相似度结构，看神经表征关系是否符合理论或是否随学习发生改变。

它最擅长回答：
- “不是单个刺激强不强，而是刺激彼此之间关系有没有变化”

最常见误用：
- trial 顺序没对齐。RSA 最怕“拿错对象做比较”，错一个顺序，整个相似度矩阵都可能失真。

### RD

一句话定义：
- 估计要用多少个维度才能解释某个条件下的 pattern 变化。

直觉上可以理解成：
- 维度高：表征更分散、更复杂
- 维度低：表征更压缩、更收敛

最常见误用：
- 见到 RD 变低就直接说“变好了”。其实要结合理论：有时压缩代表抽象化，有时也可能只是信息损失。

### GPS

一句话定义：
- 看同一条件下 trial 之间平均有多像。

直觉上：
- GPS 高，说明同类 trial 模式更紧凑、更一致

最常见误用：
- 把 GPS 高简单等同于“更强激活”。GPS 看的是模式相似性，不是平均激活强度。

### gPPI

一句话定义：
- 看某个 seed 脑区和其他脑区的功能连接，是否会被心理条件调节。

它最擅长回答：
- “在 yy 相比 kj 时，两个脑区之间的耦合是否更强或更弱”

最常见误用：
- 把 gPPI 结果解读成因果方向。gPPI 是条件调制的功能连接，不是严格因果。

### 有效连接

一句话定义：
- 尝试估计方向性信息流，问“from ROI 的变化是否能独立预测 to ROI”。

当前仓库里的实现是：
- 基于 beta-series 的简化多元回归

最常见误用：
- 把它当成 DCM 或严格因果模型。这里更像探索性方向线索，不是金标准因果推断。

### 中介分析

一句话定义：
- 检查 X 对 Y 的关系，是否部分通过 M 这条路径传递。

在这套分析里的典型问题是：
- 行为优势是否通过某个神经指标变化来体现

最常见误用：
- 用横断面、小样本中介直接下机制定论。中介更适合作为“机制假设支持”，而不是最终证明。

## Step-by-Step 教学版

下面严格按 `README.md` 的 Step 顺序讲。

阅读姿势建议：
- 先看“这步回答什么问题”
- 再看“输入/输出”
- 最后看“怎么读结果”和“QC”

### Step 0：设置根目录

这步回答什么问题：
- 不是科学问题，而是复现问题。它保证脚本都指向同一套数据。

为什么要做：
- 当前很多脚本都默认根目录是 `E:/python_metaphor`。
- 如果你不先设 `PYTHON_METAPHOR_ROOT`，在 macOS/Linux 或其他磁盘路径下很容易直接找不到文件。

输入是什么：
- 你的数据根目录

输出是什么：
- 一个环境变量，不会生成分析结果文件

结果怎么读：
- 如果后续脚本不再报 “file not found”，说明这一步大概率是对的。

常见坑：
- 只在一个终端里 `export` 了变量，换了窗口后失效
- 路径里有空格但没有正确引用

最小 QC：
- `echo $PYTHON_METAPHOR_ROOT` 能输出正确路径
- 该目录下至少能看到 `data_events/` 和 `Pro_proc_data/`

### Step 1：行为（Run7）

涉及脚本：
- `behavior_analysis/addMemory.py`
- `behavior_analysis/getMemoryDetail.py`
- `behavior_analysis/getAccuracyScore.py`
- `behavior_analysis/getTimeActionScore.py`
- 可选：`behavior_analysis/behavior_lmm.py`

这步回答什么问题：
- 学习结果有没有体现在行为上
- 隐喻条件 `yy` 是否比空间条件 `kj` 记得更好，或者反应更快

为什么先做行为：
- 因为行为是最直观的“学习成功”指标
- 后面神经结果如果和行为方向一致，故事会更完整

#### Step 1A：`addMemory.py`

为什么做：
- 把原始 `action` 编码翻译成更清楚的 `memory` 二值列

输入：
- `data_events/sub-xx/sub-xx_run-7_events.tsv`
- 关键列：`action`

输出：
- 原地改写同一个 TSV
- 新增 `memory` 列

结果怎么读：
- `memory = 1` 表示该 trial 被记住
- `memory = 0` 表示未记住

常见坑：
- 不同实验版本里 `action == 3.0` 不一定都表示“记住”
- 原地覆盖写回，若之前文件已手工修改，最好先备份

最小 QC：
- 随机打开 1 个被试 `run-7` 事件表，确认 `memory` 列已出现
- `memory` 不应该全是 0，也不应该几乎全是 1

#### Step 1B：`getMemoryDetail.py`

为什么做：
- 把外部 CSV 中的记忆反应和反应时回填到 `run-7` 事件表

输入：
- `sub-xx_run-7_events.tsv`
- `sub-xx_run-7_events.csv`
- 常见关键列：`pic_num`，以及 CSV 里的响应/RT 列

输出：
- 原地改写 `run-7` TSV
- 新增或更新 `action`、`action_time`

结果怎么读：
- 后续你就能直接从一个统一 TSV 里拿记忆成功与 RT

常见坑：
- `pic_num` 和 CSV 里的材料编码不一致
- 列名改过但脚本里的映射逻辑没同步
- 映射成功率很低时，后面的 RT 统计基本不可信

最小 QC：
- 查看脚本打印的映射成功率
- `action`、`action_time` 非空比例应该足够高

#### Step 1C：`getAccuracyScore.py`

为什么做：
- 对 `yy` 和 `kj` 的记忆正确率做被试内比较

输入：
- 已经补好 `memory` 的 `run-7` TSV
- 关键列：`trial_type`, `memory`

输出：
- 控制台统计摘要
- `data_events/run7行为结果数据.csv`

结果怎么读：
- 最关键的是被试级差值 `YY_mean - KJ_mean`
- 如果均值差为正、p 显著，说明隐喻条件行为表现更好
- Cohen's dz 越大，被试内效应越强

论文式读法示例：
- “被试在隐喻条件下的记忆正确率高于空间条件，`YY-KJ > 0`，配对 t 检验显著，提示隐喻联结带来更强行为获益。”

常见坑：
- 某些被试只有 YY 或只有 KJ 试次，会被排除出配对 t 检验
- 第 12 号被试在脚本注释里提示可能缺数据，样本量不要想当然写成 28

最小 QC：
- 看 `YY_n` 和 `KJ_n` 是否大致平衡
- 检查输出 CSV 是否每个被试都有一行

#### Step 1D：`getTimeActionScore.py`

为什么做：
- 比较 `yy` 和 `kj` 的反应时差异

输入：
- 已经补好 `action_time` 的 `run-7` TSV

输出：
- `behavior_results/run7_action_time.csv`
- 控制台统计摘要

结果怎么读：
- 如果 `YY_mean - KJ_mean < 0`，说明隐喻条件更快
- 但 RT 容易受异常值影响，所以解释通常比准确率更保守

常见坑：
- 反应时缺失太多
- RT 右偏严重，却直接机械套 t 检验

最小 QC：
- 先看是否大量丢失 trial
- 看被试均值是否有离谱异常值

#### Step 1E：`behavior_lmm.py`（可选）

为什么做：
- 如果你不想只看被试均值，而想同时考虑 subject 和 item 差异，可以上 LMM

输入：
- 一张 trial-level 或 item-level 行为表
- 至少要有 `subject`, `condition`, `response`

输出：
- `behavior_lmm_summary.txt`
- `behavior_lmm_params.tsv`

结果怎么读：
- 重点看 `condition` 相关固定效应
- 如果条件效应显著且方向符合预期，说明结果不只是少数 item 驱动

---

### Step 2：前后测 LSS（Run1-2, 5-6）+ 质量检查

涉及脚本：
- `glm_analysis/run_lss.py`
- `glm_analysis/check_lss_comprehensive.py`

这步回答什么问题：
- 先不直接回答论文结论，而是为后面的表征分析准备“每个 trial 的神经图像单位”

为什么一定要做：
- RSA、RD、GPS、很多 MVPA 都需要 trial-level beta
- 如果没有这一步，后面就没有可靠的 pattern 输入

为什么用 LSS：
- 它把“当前目标 trial”和“其他 trial”分开建模
- 这样能减少 trial 之间互相污染，得到更干净的单 trial beta

关键参数为什么这样设：
- `TR = 2.0`：时间采样间隔，必须和数据采集一致
- `smoothing_fwhm = None`：主线脚本明确强调 RSA 专用时不能平滑
- `hrf_model = "spm + derivative"`：允许 HRF 时间形状有一点弹性
- `high_pass = 1/128`：滤掉低频漂移，是 fMRI GLM 里常见默认

输入是什么：
- `Pro_proc_data/{sub}/run{run}/*.nii(.gz)`
- `data_events/{sub}/{sub}_run-{run}_events.tsv`
- `confounds_timeseries.tsv`

事件表关键列：
- `onset`
- `duration`
- `trial_type`
- 最好有 `pic_num`

为什么 `pic_num` 很重要：
- 当前脚本会用 `trial_type + pic_num` 拼出 `unique_label`
- 这个 `unique_label` 会直接影响 RSA 的 trial 对齐

输出是什么：
- `lss_betas_final/sub-xx/run-1/` 到 `run-6/`
- 每个 trial 一个 beta 文件，如 `beta_trial-XXX_unique_label.nii.gz`
- 每个 run 下还有 `trial_info.csv`
- 汇总索引：`lss_betas_final/lss_metadata_index_final.csv`

结果怎么读：
- 这里的 beta 图通常不是最终要汇报的“结果图”，而是后续分析原料
- 你真正要读的是“有没有完整生成、标签有没有对齐、质量是否正常”

常见坑：
- `pic_num` 缺失，脚本会退回 `trial_index` 生成 label，这对 RSA 来说风险很高
- events 表里混入你其实不想分析的 trial_type，LSS 会照单全收
- 没有 confounds 或 sample mask 对不上，会导致 GLM 失败或输出异常

最小 QC：
- 每个有效 run 都应生成大量 beta 图，而不是只有少数几个
- `trial_info.csv` 行数应与生成 beta 数相符
- `lss_metadata_index_final.csv` 里应至少看到 `subject`, `run`, `unique_label`, `condition`, `beta_file`

#### `check_lss_comprehensive.py` 在做什么

它主要不是“再算一遍结果”，而是在问：
- beta 有没有生成完整
- 命名是否合理
- 上游 events 和下游 beta 是否能对得上

你可以把它理解成：
- “LSS 层的体检”

---

### Step 3：学习阶段 LSS（Run3-4）+ 学习阶段 GLM

涉及脚本：
- `fmri_lss/fmri_lss_enhance.py`
- `glm_analysis/main_analysis.py`

这步回答什么问题：
- 学习时哪些脑区更参与 `yy` 相比 `kj`
- 同时为可选的学习阶段动态分析准备更细粒度 beta

#### Step 3A：`fmri_lss_enhance.py`

为什么做：
- 学习阶段如果以后要做 dynamics、learning MVPA、连接扩展，这些单 trial beta 很有用

输入：
- `run-3`, `run-4` 的 BOLD、events、confounds

输出：
- `lss_betas_final/sub-xx/run-3/`
- `lss_betas_final/sub-xx/run-4/`
- 每个 run 目录下还会保存 `trial_info.csv` 和 `mask.nii.gz`
- 根目录会保存 `lss_betas_final/processing_results.csv`

结果怎么读：
- 和 Step 2 一样，它主要是中间层产物，不是最后论文证据
- 这个版本现在偏向“低 I/O + 可续跑”：同一个 run 的 fMRI 只加载一次，重跑优先复用已有 mask

常见坑：
- 把学习阶段和前后测混为一谈
- 用学习阶段 LSS 的结果去替代前后测 RSA 主线输入
- 看到 `Computing mask...` 以为卡死；很多时候其实是不同 `(subject, run)` 的日志交错，或者第一次为该 run 生成 mask
- 直接信任已有 beta；现在脚本会先验证 beta 是否可读，坏文件会删掉后重算

最小 QC：
- `run-3/4` 目录存在
- trial 数量和 events 数量大体相符
- `trial_info.csv` 的行数应和该 run 实际可用 beta 数量一致
- 重跑后若日志提示某个 beta 损坏被重算，最终 `trial_info.csv` 仍应完整覆盖该 run

#### Step 3B：`main_analysis.py`

为什么做：
- 这一步回答“哪里在参与”，是空间定位层的证据

为什么不是用它当主结论：
- 因为你的主问题是“表征是否重组”，不是“哪个脑区更亮”
- GLM 只告诉你学习阶段条件差异在哪些脑区更明显

输入：
- `run-3`, `run-4` 的 BOLD、events、confounds
- 对比定义来自 `glm_config.py`

输出：
- `glm_analysis_fwe_final/`
- 一阶被试对比图
- 二阶组水平阈值图
- cluster 报告
- 日志 `analysis.log`

结果怎么读：
- 重点看 `yy > kj` 或对应对比是否在理论相关 ROI 出现显著簇
- 这些显著区域之后可作为 ROI 候选来源

论文式读法示例：
- “学习阶段，隐喻相较空间条件在某些 ROI 内诱发更强响应，提示这些区域参与新语义联结建立。”

常见坑：
- 把一阶单被试图当成组水平结论
- 忽略多重比较校正方式，只看彩色图就下结论
- ROI 定义和后续分析使用同一批数据时，要意识到 circularity 风险

最小 QC：
- 二阶分析时有效被试数不能太低
- 报告里要能看出用了哪种校正方法，如 FDR 或 cluster-FWE
- 输出目录结构应完整，不只是空文件夹

#### 可选：`create_optimized_roi_masks.py`

为什么做：
- 把你后面要用的 ROI 统一放进 `roi_masks_final/`

什么时候需要：
- 你准备用固定 ROI 跑 RSA、RD、GPS、MVPA 时

最小 QC：
- mask 文件能在脑空间里正常打开
- ROI 不是空的，也不是整个脑都被选中

---

### Step 4：模式堆叠 `stack_patterns.py`

这步回答什么问题：
- 它不直接回答科学问题，它负责把 trial beta 变成统一格式的“分析底座”

为什么它很关键：
- 这是整个主线的“粘合点”
- 如果这一步命名错了、排序错了，后面的 RD/GPS/部分 RSA/MVPA 都会连锁出错

输入是什么：
- `lss_metadata_index_final.csv`

最关键字段：
- `subject`
- `run`
- `condition`
- `phase`
- `beta_path` 或 `beta_file`
- 推荐有 `trial_id`

脚本会做什么标准化：
- `yyw/yyew -> yy`
- `kjw/kjew -> kj`
- 默认排除 `fake`

输出是什么：
- `pattern_root/sub-xx/pre_yy.nii.gz`
- `pattern_root/sub-xx/pre_kj.nii.gz`
- `pattern_root/sub-xx/post_yy.nii.gz`
- `pattern_root/sub-xx/post_kj.nii.gz`
- 对应的 `*_metadata.tsv`

为什么输出是 4D NIfTI：
- 前 3 维是脑空间
- 最后一维是 trials
- 这样一个文件就能容纳“一个时间点、一个条件下的全部单 trial patterns”

结果怎么读：
- 这里只看“有没有堆对”
- 后续脚本会把这些 4D 文件再按 ROI 提取为 `trials x voxels` 矩阵

常见坑：
- `phase` 缺失或 run 映射错，导致 pre/post 混了
- 条件归并失败，输出成 `pre_yyw.nii.gz` 这种下游不认的文件
- metadata 里路径是相对路径，但相对基准目录理解错了

最小 QC：
- 每个被试至少应看到 `pre_yy`, `pre_kj`, `post_yy`, `post_kj`
- `*_metadata.tsv` 行数要和 4D 文件里的 trial 数一致

---

### Step 5：前后测 RSA（核心证据）+ LMM

涉及脚本：
- `rsa_analysis/check_data_integrity.py`
- `rsa_analysis/run_rsa_optimized.py`
- `rsa_analysis/rsa_lmm.py`

这步回答什么问题：
- 学习前后，刺激之间的神经相似性关系有没有发生系统改变
- 而且这种改变是不是在 `yy` 比 `kj` 更明显

为什么说 RSA 是主线核心：
- 因为你的研究问题是“表征重组”
- RSA 最直接地把“刺激之间关系的变化”量化出来

#### Step 5A：`check_data_integrity.py`

为什么做：
- 先确认 trial 能不能和 `stimuli_template.csv` 一一对上

为什么它重要到建议必跑：
- RSA 对对齐极度敏感
- 如果 `yyw_1` 被错匹配成 `yyw_10`，你表面上可能还能跑完，但结果已经不可信

输入：
- `lss_metadata_index_final.csv`
- `stimuli_template.csv`

输出：
- 无问题时控制台提示通过
- 有问题时会写出 `rsa_data_validation_report_fixed.csv`

结果怎么读：
- `ALL GREEN` 才建议继续跑 RSA

常见坑：
- `unique_label` 里多空格
- 同一个 label 在一个阶段重复出现
- beta 文件名存在，但物理文件已丢失

最小 QC：
- 每个被试 pre/post 所需 trial 都能匹配到
- 没有 `MISSING`、`DUPLICATE`、`FILE_NOT_FOUND`

#### Step 5B：`run_rsa_optimized.py`

为什么做：
- 在每个 ROI 内提取每个 trial 的 pattern，计算 trial 间相似度矩阵
- 再从矩阵里抽取你真正关心的 item-wise 指标

这个脚本在做什么：

1. 按模板顺序取 pre 或 post 的 beta 文件
2. 在 ROI 内提取每个 trial 的多体素 pattern
3. 对每个 trial 做标准化
4. 计算 trial-by-trial 相似度矩阵
5. 提取 Metaphor、Spatial、Baseline 等条件下的 item-wise 相似度
6. 生成汇总表和明细表

为什么要“按模板顺序取文件”：
- 因为相似度矩阵里第 1 行第 1 列必须始终对应同一个刺激位置
- 一旦顺序错了，不同被试之间、前后测之间就没法比较

输入是什么：
- `lss_metadata_index_final.csv`
- `stimuli_template.csv`
- ROI masks

输出是什么：
- `rsa_itemwise_details.csv`
- `rsa_summary_stats.csv`

如何理解这两个输出：

`rsa_itemwise_details.csv`
- 最重要
- 每行可以理解成“某个被试、某个 ROI、某个时间点、某个条件、某个 item pair 的相似度”
- 这是 LMM 的输入

`rsa_summary_stats.csv`
- 更适合快速看均值与方向
- 适合做初步 sanity check，但统计推断更建议看 LMM

结果怎么读：
- 如果 post 相比 pre，`yy` 的相似度变化比 `kj` 更明显，说明隐喻学习带来了更强表征重组
- 方向要结合理论看：
  - 相似度升高，可能表示同类表征更整合
  - 相似度降低，也可能表示分化增强
- 所以关键不是“越高越好”，而是“是否符合你对表征组织变化的理论预期”

最常见坑：
- trial 对齐错误
- ROI 太小，体素太少，相关矩阵不稳定
- Top-K 体素选择和 GLM 来源重叠时要考虑 circularity

最小 QC：
- 输出的 item-wise 行数应接近 `被试 x ROI x 时间 x 条件 x item数`
- 不应出现大量 NaN
- pre/post 两个阶段都应有数据

#### Step 5C：`rsa_lmm.py`

为什么做：
- RSA 明细是 item-wise 数据，同一个被试和同一个 item 都会重复出现
- 用 LMM 能更好处理这种层级结构

模型在回答什么：
- `similarity ~ condition * time + (1|subject) + (1|item)`

最关键的项：
- `condition * time` 交互

交互显著意味着什么：
- 从 pre 到 post 的变化幅度，在 `yy` 和 `kj` 之间不一样
- 这通常就是“隐喻联结引起更强表征变化”的统计核心

输入是什么：
- 最推荐直接用 `rsa_itemwise_details.csv`

输出是什么：
- `rsa_lmm_summary.txt`
- `rsa_lmm_params.tsv`
- `rsa_lmm_model.json`

结果怎么读：
- 重点不是只盯着主效应，而是先看交互项方向和显著性
- 若交互方向符合“yy 增加更多/降低更多”，再回到 summary 表解释具体模式

论文式读法示例：
- “混合模型显示 `condition × time` 交互显著，说明隐喻条件在学习前后诱发的神经相似性变化幅度大于空间条件。”

常见坑：
- 输入表列名不对，尤其 `time`/`stage`
- 把均值汇总后再做 LMM，失去 item 层信息

最小 QC：
- `rsa_lmm_summary.txt` 中模型成功收敛
- 参数表里能找到条件、时间、交互相关项

---

### Step 6：RD / GPS（互补证据）

涉及脚本：
- `representation_analysis/rd_analysis.py`
- `representation_analysis/gps_analysis.py`

这步回答什么问题：
- 如果 RSA 说明“关系变了”，RD/GPS 则补充说明“几何结构怎么变”

为什么要有互补指标：
- 单一指标很难说明全部
- 同样是“重组”，可能表现为维度减少、模式更紧凑、或结构更抽象

#### Step 6A：`rd_analysis.py`

为什么做：
- 估计一个 ROI 里 pattern 的有效维度

当前脚本的直觉：
- 从 `trials x voxels` 样本出发
- 看解释一定方差阈值所需的 PCA 成分数

关键参数：
- `--threshold 80.0`

这个阈值怎么理解：
- 不是显著性阈值
- 它表示“要解释 80% 方差需要多少个维度”

输入是什么：
- `pattern_root/`
- `roi_dir/`

输出是什么：
- `rd_subject_metrics.tsv`
- `rd_group_summary.tsv`
- 每个 ROI 自己的 `*_rd_summary.json`

结果怎么读：
- `rd_subject_metrics.tsv`：每个被试、ROI、时间、条件一条值
- `rd_group_summary.tsv`：看组水平 `yy_delta` 和 `kj_delta` 的差异

如果你看到：
- post 比 pre 的 RD 降低，且这种降低在 `yy` 更强

一种常见解释是：
- 学习后隐喻表征更压缩、更收敛

但一定要保留谨慎：
- RD 变低不自动等于“更高级”
- 需要和 RSA/GPS 一起看

最小 QC：
- 各 ROI 都有 pre/post、yy/kj 四格数据
- 没有因为 trial 数太少而大量缺值

#### Step 6B：`gps_analysis.py`

为什么做：
- 量化同条件 trial 之间整体有多像

输入：
- 同 `rd_analysis.py`

输出：
- `gps_subject_metrics.tsv`
- `gps_group_summary.tsv`
- 每个 ROI 的 `*_gps_summary.json`

结果怎么读：
- GPS 高，说明同类 trial 的模式更一致、更紧凑
- 如果 post 的 `yy` 比 pre 提升更多，可能表示隐喻学习后内部表征更稳定

和 RSA 的区别：
- RSA 更强调“关系矩阵”与 item-level 结构
- GPS 更像一个整体紧凑度指标

常见坑：
- 误把 GPS 当作平均激活
- 不看样本数，直接比较不同条件的 GPS

最小 QC：
- 先确认每个条件 trial 数大致可比
- 检查是否某个 ROI 体素数过少导致不稳定

---

### Step 7：脑-行为关联

涉及脚本：
- `brain_behavior/brain_behavior_correlation.py`

这步回答什么问题：
- 神经变化大的被试，行为上是不是也学得更好

为什么要做：
- 如果神经指标和行为没有任何关系，论文故事会停留在“平行现象”
- 有关联时，你可以说不同层次证据彼此支持

输入是什么：
- 若干 `subject-level` 指标文件
- 每个输入文件必须有 `subject` 列

比较理想的输入例子：
- 行为差值表
- RSA 汇总后的被试指标
- RD 或 GPS 的被试指标
- MVPA 准确率表

脚本会做什么：
- 先把各表按 `subject` 合并
- 对所有数值列做两两 Pearson 和 Spearman 相关

输出是什么：
- `brain_behavior_merged.tsv`
- `brain_behavior_correlations.tsv`
- `brain_behavior_summary.json`

结果怎么读：
- 先看哪组神经指标与行为指标相关
- 再看方向是否符合理论，比如：
  - 更强的 `yy` 表征重组
  - 对应更高的记忆正确率提升

常见坑：
- 把 trial-level 表直接塞进来，脚本会按 subject 求均值，可能把你想保留的信息抹掉
- 列名前缀很多，解读时容易混淆来源
- 多重比较很多，不能只挑显著的讲

最小 QC：
- 合并后被试数是否和预期接近
- 相关结果里 `n` 是否足够
- 重要结论最好同时看 Pearson 和 Spearman 是否一致

---

### Step 8：机制与稳健性（可选）

这一步不建议第一次复现就全跑。最推荐顺序是：

1. learning dynamics
2. gPPI
3. effective connectivity
4. mediation
5. 其他稳健性分析

#### 8A：learning dynamics

想回答：
- 学习不是一瞬间完成的，某些指标会不会沿时间窗逐渐变化

适合什么时候做：
- 主线已经证明前后测有变化，你现在想知道这个变化在学习期是如何累积的

#### 8B：gPPI

脚本：
- `connectivity_analysis/gPPI_analysis.py`

想回答：
- 学习阶段 `yy` 相比 `kj`，某个 seed ROI 与其他 ROI 的连接是否更强

脚本里的心理变量怎么编码：
- `yy = +1`
- `kj = -1`

你应该怎么读：
- 重点看 `gppi_beta`
- 正值不代表“绝对因果”，只表示在该条件编码下，交互项方向为正

常见坑：
- seed mask 选得太大或不稳定
- 把功能连接调制解释成信息流方向

#### 8C：effective connectivity

脚本：
- `connectivity_analysis/effective_connectivity.py`

想回答：
- 在控制其他 ROI 后，某个 ROI 的 beta-series 能否独立预测另一个 ROI

最该记住的一句话：
- 这是探索性方向线索，不是 DCM

适合怎么写在论文里：
- 作为“候选机制支持”或“补充分析”

#### 8D：mediation

脚本：
- `brain_behavior/mediation_analysis.py`

想回答：
- 行为优势是否部分通过某个神经指标变化而体现

最适合的使用前提：
- 你已经有一个比较清楚的 X、M、Y

例子：
- X：条件差异或某个学习指标
- M：RSA / RD / GPS 变化量
- Y：行为提升

最常见坑：
- 随便挑三列就跑中介
- 没有理论先验，只凭显著性讲机制

## 如何把结果翻译成论文句子

可以统一用这个模板：

1. 先说比较对象
- “与空间联结相比，隐喻联结在学习后表现出……”

2. 再说统计层
- 行为：`t(df)`, `p`, 效应量
- RSA：LMM 的 `condition × time`
- RD/GPS：组水平 summary 中的差异中的差异

3. 最后说理论含义
- “提示隐喻学习引发更强的表征整合/压缩/重组”

一个简单例子：
- 行为：`YY-KJ > 0`
- RSA：交互显著
- RD：`yy` 学后维度下降更明显

可以写成：
- “行为与神经指标共同表明，隐喻联结相较空间联结引发了更强的学习相关表征重组；该重组表现为 item-wise 相似度结构变化，并伴随表征维度压缩。”

## 最小 QC 总表

如果你只想快速判断“这一层能不能往下走”，记住下面几条。

### 行为层

- `run-7` TSV 已补上 `memory`、`action_time`
- `YY` 与 `KJ` 都有足够 trial
- 没有大量缺失或离谱 RT

### LSS 层

- 每个 trial 都生成 beta
- `lss_metadata_index_final.csv` 存在
- `unique_label` 看起来合理，不是全靠 trial index 临时拼的

### Pattern 层

- 每个被试都有 `pre/post x yy/kj`
- 4D 文件和 metadata 行数对得上

### RSA 层

- 数据完整性检查通过
- `rsa_itemwise_details.csv` 没有大面积缺失
- LMM 成功收敛

### RD/GPS 层

- 每个 ROI 都有四格数据
- 没有因为 trial 数或 mask 问题导致整块空值

### 脑-行为层

- 合并后的被试数合理
- 你知道每一列来自哪张输入表

## 常见问题与排错

### 1. 对齐错误

典型表现：
- RSA 跑得动，但结果很怪
- 某些被试只有 pre 没有 post

优先检查：
- `unique_label`
- `stimuli_template.csv`
- `run` 到 `pre/post` 的映射

### 2. trial 命名不统一

典型表现：
- 生成了 `pre_yyw.nii.gz`
- 下游脚本找不到 `pre_yy.nii.gz`

优先检查：
- `stack_patterns.py` 入口 metadata 的 `condition` 列
- 是否正确归并到了 `yy/kj`

### 3. 数据泄漏

最容易发生在：
- MVPA
- 使用 GLM 结果挑体素、又用同批数据汇报分类效果

处理原则：
- 训练和测试严格分开
- 对 ROI 选择和统计推断要意识到 circularity

### 4. mask 路径错误

典型表现：
- 输出空值
- 某 ROI 结果全部 NaN

优先检查：
- ROI 文件是否真的存在
- mask 是否与 BOLD 空间一致
- ROI 是否为空或极小

### 5. 缺失被试

典型表现：
- 组水平样本数忽然变小

优先检查：
- 某个 run 缺文件
- 某被试没有 YY 或 KJ 数据
- 物理 beta 文件缺失

### 6. 阈值与多重比较

GLM 层最重要：
- 先说你用的是 FDR 还是 FWE
- 不要只根据“图好看”解释结果

相关/中介层也要注意：
- 比较多的时候要警惕偶然显著

### 7. 把均值当成全部

典型表现：
- 只看 summary 表，不看 item-wise 或 subject-level 波动

建议：
- 行为看被试级分布
- RSA 看 item-wise 明细和 LMM
- 脑-行为看散点趋势，不只看 p 值

## 从 0 到 1 的学习路线

不放外链，只给关键词和顺序。

### 第 1 层：先懂研究故事

- 实验设计
- 前测 / 学习 / 后测
- 条件对比
- 被试内设计

### 第 2 层：先懂最小 fMRI 词汇

- TR
- HRF
- confounds
- first-level
- second-level
- ROI
- searchlight

### 第 3 层：先懂 GLM 和 LSS

- GLM design matrix
- contrast
- single-trial beta
- smoothing
- high-pass filtering

### 第 4 层：再懂表征分析

- multivoxel pattern
- representational similarity
- RDM
- item-wise analysis
- mixed-effects model

### 第 5 层：再懂几何和连接

- PCA dimensionality
- pattern compactness
- gPPI
- beta-series
- effective connectivity

### 第 6 层：最后懂统计解释

- paired t-test
- effect size
- confidence interval
- interaction
- multiple comparisons
- bootstrap confidence interval
- mediation

## 最后给小白的执行建议

第一次跑，建议只做这些：

1. Step 1 行为
2. Step 2 前后测 LSS
3. Step 3 学习 GLM
4. Step 4 stack patterns
5. Step 5 RSA + LMM
6. Step 6 RD/GPS
7. Step 7 脑-行为关联

第一次不要急着做：
- searchlight
- 所有 MVPA
- 所有连接分析
- 中介和 BF10

最稳妥的工作流是：

1. 先确认目录和文件名全部对
2. 再确认每一层 QC 通过
3. 最后才解释统计结果

如果你能用一句话复述整条主线，说明你已经真正理解了这套分析：

**先用行为确认学到了东西，用学习期 GLM 定位相关脑区，用前后测 RSA 证明表征关系发生重组，再用 RD/GPS 补充描述几何变化，最后用脑-行为关联把神经变化和行为收益连起来。**