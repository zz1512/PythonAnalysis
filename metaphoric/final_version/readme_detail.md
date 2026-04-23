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
- 表征上，同一类刺激在学习前后是不是变得更像、更稳定、或更分化（区分度增强）、或几何结构发生压缩
- 脑和行为之间，这些变化能不能互相解释

一句话故事线：

**行为告诉你“有没有学会”，GLM 告诉你“哪里在参与”，前后测 RSA 告诉你“表征关系怎么变”，Model-RSA 告诉你“为什么会这样变”，脑-行为关联告诉你“这些神经变化和表现有没有同向”；RD/GPS 则作为几何层面的互补证据。**

重要提示（根据当前阶段性结果）：
- 在本数据集中，隐喻（yy）的 RSA 主要表现为 **pair similarity 从 pre 到 post 更明显下降**，更接近“表征分化增强”而非“配对聚合/对齐”。后续 Model-RSA 与写作叙事需与这一方向一致。

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

当前主线重点是三类条件：
- `yy`：隐喻相关联结
- `kj`：空间相关联结
- `baseline`：无学习控制条件（原始标签通常是 `jx`）

脚本里原始 trial type 可能长这样：
- `yyw`, `yyew`
- `kjw`, `kjew`

在下游很多地方会被统一归并成：
- `yy`
- `kj`
- `baseline`

这样做是为了让下游文件名固定、分析口径统一。需要特别记住的是：
- `yyw/yyew -> yy`
- `kjw/kjew -> kj`
- `jx -> baseline`
- baseline 不是学习配对材料，而是前后测中的无学习控制条件

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
4. Model-RSA：这种变化更像预存语义、学习后配对对齐，还是学习后分化增强
5. 脑-行为关联：神经变化大的被试，行为上是否也学得更好
6. RD/GPS：如果需要补几何层证据，再看这种变化是否体现为维度改变或紧凑度改变
7. 机制扩展：如果主结果已经很稳，再考虑动态、偏侧化、连接性或中介

## 主线、控制、扩展

### 核心主线

最推荐先跑通的主线：

1. 行为
2. 前后测 LSS
3. 学习阶段 GLM
4. stack patterns
5. RSA + LMM
6. Model-RSA + Δρ LMM
7. 脑-行为关联
8. RD/GPS（互补证据）

如果你只想先拿到一条足够完整、能讲故事、也最符合当前顶刊叙事的结果线，这几步足够。

### 控制验证

在当前版本里，最重要的控制验证已经不再是“可做可不做”：

- baseline 控制

它回答的是：

“前后变化是不是只是重复暴露带来的普通熟悉化，而不是隐喻学习本身带来的特异性变化？”

因此现在的建议是：
- baseline 必须进入 `run_rsa_optimized.py` 和 `rsa_lmm.py` 的默认主链
- 主故事仍聚焦 `yy vs kj`
- 但结果汇报必须同时告诉读者：`baseline(jx)` 是否也发生了同等幅度变化

### 进阶扩展

当前最推荐的进阶顺序：
- 熟悉度/时间梯度参数调制
- 偏侧化指数（LI）
- learning dynamics
- mediation
- effective connectivity
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

#### 推荐：`roi_management/build_roi_library.py`

为什么做：
- 把 ROI 管理成统一资产，而不是每个脚本各自写路径
- 一次性生成三层 ROI：`main_functional`、`literature`、`atlas_robustness`

最关键产出：
- `roi_library/masks/.../*.nii.gz`
- `roi_library/manifest.tsv`

后续怎么用：
- RSA 默认会优先读取 `roi_library/manifest.tsv`
- 可通过环境变量 `METAPHOR_ROI_SET` 切换 ROI 层级，例如 `main_functional`、`literature`、`atlas_robustness`

#### 三层 ROI 在论文里的角色（非常重要）

不要把三层 ROI 混成一个 list 跑一次。它们在论文里承担**不同的论证角色**，必须**分别跑、分别报告**，才能把"三层证据链"讲清楚：

| 层 | 来源 | 在论文里做什么 | 放在哪里 |
|---|---|---|---|
| **A. `main_functional`** | 学习期 GLM 显著簇 | **主结果**：在自己数据的显著簇里看"学习是否让表征重组" | Figure 2/3 主面板 |
| **B. `literature`** | Harvard-Oxford 文献驱动 ROI（左右配对 6 对） | **独立假设检验**：用文献独立定义的区域复现主结果 | Figure 4 / Table 1 |
| **C. `atlas_robustness`** | AAL 图谱 | **稳健性**：换图谱再跑一遍，封堵 "atlas 依赖" 这种 Reviewer 攻击 | Supplementary Figure |

**跑法（三层依次跑，输出目录自动分开）**：

```bash
# macOS/Linux
export PYTHON_METAPHOR_ROOT=/path/to/python_metaphor
cd metaphoric/final_version/rsa_analysis

METAPHOR_ROI_SET=main_functional  python run_rsa_optimized.py
METAPHOR_ROI_SET=literature       python run_rsa_optimized.py
METAPHOR_ROI_SET=atlas_robustness python run_rsa_optimized.py
```

输出目录会自动带 `ROI_SET` 后缀（`rsa_results_optimized_main_functional/` 等），三次跑**不会互相覆盖**。若需显式指定目录，可用环境变量 `METAPHOR_RSA_OUT_DIR`。

> **同样的三层节奏适用于 RD / GPS / MVPA-ROI / gPPI / effective connectivity**
> 这些脚本的输出目录都支持同一套机制：
> - 默认值 = `${PYTHON_METAPHOR_ROOT}/<analysis>_<ROI_SET>`
> - 环境变量强制覆盖开关：`METAPHOR_RD_OUT_DIR` / `METAPHOR_GPS_OUT_DIR` / `METAPHOR_MVPA_ROI_OUT_DIR` / `METAPHOR_MVPA_PRE_POST_OUT_DIR` / `METAPHOR_MVPA_CROSS_PHASE_OUT_DIR` / `METAPHOR_GPPI_OUT_DIR` / `METAPHOR_EFFECTIVE_CONN_OUT_DIR`
>
> 完整一层一层跑的示例：
>
> ```bash
> export PYTHON_METAPHOR_ROOT=/path/to/python_metaphor
> cd metaphoric/final_version
>
> for ROI_SET in main_functional literature atlas_robustness; do
>   export METAPHOR_ROI_SET=$ROI_SET
>   # ROI-level 模式分析
>   python representation_analysis/rd_analysis.py  $PATTERN_ROOT $ROI_DIR
>   python representation_analysis/gps_analysis.py $PATTERN_ROOT $ROI_DIR
>   # MVPA
>   python fmri_mvpa/fmri_mvpa_roi/fmri_mvpa_roi_pipeline_merged.py
>   python fmri_mvpa/fmri_mvpa_roi/mvpa_pre_post_comparison.py $PATTERN_ROOT $ROI_DIR
>   python fmri_mvpa/fmri_mvpa_roi/cross_phase_decoding.py      $PATTERN_ROOT $ROI_DIR
>   # 连接（gPPI 按种子点依次跑；输出目录里会同时含 seed 名与 ROI_SET）
>   python connectivity_analysis/effective_connectivity.py $PATTERN_ROOT $ROI_DIR
> done
> ```
>
> 每个脚本都会把 `output_dir` 自动解析成 `<analysis>_<ROI_SET>/`，三层跑完得到三套独立结果目录，彼此不覆盖。

**一致性判据（论文里"converging evidence"的门槛）**

| 检查项 | 通过标准 |
|---|---|
| 方向一致 | 三层都看到同向效应（post > pre 的 yy-only RSA 增强，或 yy-kj × post-pre 交互显著） |
| 强度衰减 | 一般 A 层效应量 ≥ B 层 ≥ C 层（A 是最优 ROI 定义）；若 C 反而更强，说明你的功能 ROI 覆盖不全 |
| 显著性模式 | 不要求三层全显著，但至少 **A + B 同向**；C 层可仅作 trend |
| 其他条件固定 | 三层用同一套 pattern、同一份 trial 对齐、同一套 model RDM；**只换 ROI mask** |

**容易踩的坑**

- 把三层 ROI 并成一个 list 一次跑完 → 丢失论证价值，manifest 里 `roi_set` 的区分会被稀释。
- 只跑 A 层 → Reviewer 最常攻击的"结果是不是 ROI 选择驱动"无从回答。
- 把 `main_functional_extra_*`（`include_in_main=False`）当主结果报 → 它们是 `cluster_report` 之外的 remaining_components，严格说应放 Supplementary；主结果只报 `include_in_main=True` 的 ROI。
- 下游的 RD / GPS / MVPA-ROI / gPPI 也按同一套三层节奏跑（它们都走同一个 manifest）；Searchlight 是全脑分析，不需要切层。

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
- `phase`（若缺失，脚本可按 `run` 自动推断：`1-2=pre, 3-4=learn, 5-6=post`）
- `beta_path` 或 `beta_file`
- 推荐有 `trial_id`

脚本会做什么标准化：
- `yyw/yyew -> yy`
- `kjw/kjew -> kj`
- `jx -> baseline`
- 默认排除 `fake`

输出是什么：
- `pattern_root/sub-xx/pre_yy.nii.gz`
- `pattern_root/sub-xx/pre_kj.nii.gz`
- `pattern_root/sub-xx/pre_baseline.nii.gz`
- `pattern_root/sub-xx/post_yy.nii.gz`
- `pattern_root/sub-xx/post_kj.nii.gz`
- `pattern_root/sub-xx/post_baseline.nii.gz`
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
- 若 metadata 缺 `word_label` / `pair_id`，后面的 Model-RSA 会静默跳过或退化

最小 QC：
- 每个被试至少应看到 `pre_yy`, `pre_kj`, `pre_baseline`, `post_yy`, `post_kj`, `post_baseline`
- `*_metadata.tsv` 行数要和 4D 文件里的 trial 数一致

---

### Step 5：前后测 RSA（核心证据）+ LMM + Model-RSA（主流程机制解释）

涉及脚本：
- `rsa_analysis/check_data_integrity.py`
- `rsa_analysis/run_rsa_optimized.py`
- `rsa_analysis/rsa_lmm.py`

这步回答什么问题：
- 学习前后，刺激之间的神经相似性关系有没有发生系统改变
- 而且这种改变是不是在 `yy` 比 `kj` 更明显
- 如果改变存在，它更像“配对对齐/收敛”，还是更像“表征分化增强”

为什么说 RSA 是主线核心：
- 因为你的研究问题是“表征重组”
- RSA 最直接地把“刺激之间关系的变化”量化出来
- 在当前稿件结构里，Step 5 不只是“先看 RSA 再决定要不要做机制分析”，而是默认包含：
  - Step 5B/5C：先证明变化发生了（What）
  - Step 5D：再解释这种变化更匹配哪种机制模型（Why）

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

这里需要特别注意 baseline 的定义：
- `Metaphor` 对应模板里的 `yyw/yyew`
- `Spatial` 对应模板里的 `kjw/kjew`
- `Baseline` 对应模板里的 `jx`
- **baseline 不是成对学习材料**，因此脚本不会把 baseline 当成 `pair_id`=2 个词一组来处理；它会改用 baseline 条件内部所有非对角 trial-pair 的相似度作为控制分布

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
- 对 `Metaphor/Spatial`，这里的 item pair 是学习配对；对 `Baseline(jx)`，这里的 item pair 是 baseline 内部任意两两 trial 组合
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

结合当前阶段性结果的读法（推荐写进主文/讨论）：
- 目前结果中 yy 的 pair similarity 从 pre 到 post **下降更明显**。这类模式通常更适合解释为：学习后表征更 item-specific（分化增强），而不是简单“两个词更像/更对齐”。因此后续机制解释应优先围绕 differentiation（分化）路径组织。
- 在宣称“条件特异分化”之前，强烈建议先做 baseline 控制：确保 baseline 的 pre→post 变化不呈现同等幅度的下降，否则可能是重测噪声或阶段性 SNR 变化导致的假象。

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

当前默认条件：
- 默认会纳入 `Metaphor / Spatial / Baseline`
- baseline 不再只是 summary 里的 `Sim_Baseline`，而是会真正进入 `rsa_itemwise_details.csv` 并作为 LMM 的控制条件

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

### Step 5D：Model-RSA（核心五模型：M1/M2/M3/M7 + M8）+ Δρ LMM

涉及脚本：
- `rsa_analysis/build_stimulus_embeddings.py`（一次性预计算 M3 词向量）
- `rsa_analysis/build_memory_strength_table.py`（一次性预计算 M7 被试回忆表）
- `rsa_analysis/model_rdm_comparison.py`（M1/M2/M3/M7 + 偏相关 + 共线性）
- `rsa_analysis/delta_rho_lmm.py`（前后测 Δρ LMM 统计）

这步回答什么问题：
- 光比较 pre vs post 相似性变化还不够，还要回答"这个变化到底在匹配哪种理论模型"。
- Model-RSA 把 neural RDM 与"理论驱动的模型 RDM"相关，检验哪些认知模型解释力最强。

五个核心模型（故事骨架）：

| 模型 | 构造 | 在学习前应该 | 在学习后应该 | 解释哪部分故事 |
|---|---|---|---|---|
| M1_condition | 三水平条件（yy/kj/baseline），同条件=0、跨条件=1 | 弱到中 | 相对稳定或略增 | 是否存在条件分类表征（基线轴） |
| M2_pair | 同 `pair_id`=0，其余=1 | ≈0（无学习就不应该有配对结构） | 不预设方向：可能增强（对齐路径），也可能不增强（分化路径） | Convergence（对齐）假设的直接量化模型 |
| M8_reverse_pair | 同 `pair_id`=1，其余=0 | ≈0（无学习就不应该有配对结构） | 不预设方向：若分化增强，可能更敏感 | Differentiation（分化）假设的直接量化模型（与 M2 互补） |
| M3_embedding | 预训练 BERT cosine distance | 应该最主导（ROI 主要反映预存语义） | 被 M2/M7 分走部分方差 | "预存语义" vs "学习塑造"的分离 |
| M7_memory | 被试自己的回忆评分差 `|r_i−r_j|` | ≈0 | 不预设方向：既可能“记得越牢越对齐”，也可能“记得越牢越分化” | 行为记忆成绩 ↔ 表征重塑的桥梁 |

先用 5 句话记住这几个模型：
- **M1_condition**：问的是“这个 ROI 里有没有粗粒度的条件分类结构”，它更多是一个**基线轴**，不是学习机制本身。
- **M2_pair**：问的是“学习后同一配对是不是更趋向对齐/收敛”，它对应 **convergence** 路径。
- **M8_reverse_pair**：问的是“学习后同一配对是不是反而更分化”，它对应 **differentiation** 路径。
- **M3_embedding**：问的是“在没学之前，神经表征是不是主要跟随词汇的预存语义距离”，它是**语义锚点模型**。
- **M7_memory**：问的是“记忆得更牢的项目，是否在神经表征上表现出系统性差异”，它是**行为-神经桥梁模型**。

如何快速理解 “为什么必须同时看 M2 和 M8”：
- 如果 Step 5C 发现 `yy` 的 pair similarity **上升**，更像“配对对齐/收敛”，此时优先关注 M2。
- 如果 Step 5C 发现 `yy` 的 pair similarity **下降**，更像“表征分化增强”，此时优先关注 M8。
- 所以 M2 和 M8 不是“一个对一个错”，而是两条竞争机制路径的两种编码方式。
- 也正因为两者是互补编码，它们在偏相关里**不能互相作为对方控制变量**；否则 partial rho 会失去可解释性。

五个核心模型详解（构造 → 数学 → 假设 → 文献 → 常见坑）：

补充：M8 不是替代 M2，而是用于区分两条机制路径（convergence vs differentiation）。如果 Step 5C 显示 yy 的 pair similarity 明显下降（分化增强），优先关注 M8 的 Δρ 与偏相关结果是否在 yy 中更强。

#### M1_condition（条件类别 RDM，基线解释轴）

- 构造思路：把每个 trial 打上 `condition ∈ {yy, kj, baseline}` 标签，同条件 = 0，跨条件 = 1。是最粗的"语义分类"模型。
- 数学形式：`RDM_M1[i,j] = 0 if cond_i == cond_j else 1`。上三角向量化后长度 = C(n, 2)。
- 在脚本中：`model_from_condition_multi(metadata)`，位于 [model_rdm_comparison.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/rsa_analysis/model_rdm_comparison.py)。默认条件循环是 yy/kj/baseline，可用 `--conditions` 覆盖。
- 理论假设：
  - 前测：若 ROI 存在条件特异的加工，M1 应有弱到中等正相关（材料差异本身可驱动）。
  - 后测：相对稳定或略增——条件分类不因学习大幅改变。
- 文献依据：Mashal et al. 2007、Cardillo et al. 2012 的条件对比；Kriegeskorte 2008 把"categorical model"列为 RSA 第一块积木。
- 常见坑：
  - 如果只跑 yy/kj 两条件，M1 退化为二分，区分力下降；强烈建议保留 baseline 作为第三水平。
  - 条件内 trial 数不均衡（40 : 40 : 20）会让 M1 的 0 值占比偏离 1/3，对 Spearman 影响可控但报告时需说明。
  - baseline 在 M1 中的意义是“第三类控制条件”，不是学习配对本身；不要把 M1 里 baseline 的存在误解成“baseline 也被当成学习结构”。

#### M2_pair（配对身份 RDM，学习塑造的直接证据）⭐

- 构造思路：每对学习词（如"玩笑"-"面具"）的两个 concept word 在 M2 上距离 = 0，其余 = 1。只有学习过的配对才应该让这两个 word 在神经空间对齐。
- 数学形式：`RDM_M2[i,j] = 0 if pair_id_i == pair_id_j (且 i != j) else 1`。
- 在脚本中：`model_from_pair_identity(metadata, pair_col="pair_id")`；若 `pair_id` 缺失，会回退到 `pic_num`；两者都缺则抛 `ValueError`。
- 直觉图像：
  - 同一对词越像，M2 越容易与“低神经距离/高神经相似性”一致。
  - 所以 M2 更适合检验“学习是否把配对词拉近了”。
- 理论假设：
  - 前测：M2 对 neural RDM 的 ρ ≈ 0（无学习就不应出现配对结构）。**若前测 M2 显著 > 0，说明材料本身有混淆（比如配对词已有语义关联），需要在讨论中解释。**
  - 后测（两支并行，不预设方向）：
    - Convergence 路径：M2 显著 > 0，且 **yy > kj > baseline**（隐喻学习最能驱动配对对齐）。
    - Differentiation 路径：若 Step 5C 显示 yy 的 pair similarity 明显下降，M2 可能不升甚至接近 0；此时应使用“反向配对模型”（M8_reverse_pair）来检验“配对内分化增强”是否为主要机制。
  - Δρ(M2) = ρ_post − ρ_pre 是学习相关变化的指标之一，但不应强行解读为“必须为正”。在分化路径下，更关键的是 M8 或其它分化型模型的 Δρ。
- 文献依据：
  - Cardillo et al. 2012（JoCN）："novel → familiar metaphor tuning" 的表征层解释
  - Xue et al. 2010（Science）：记忆成功 ↔ 神经模式"再激活稳定性"
  - Ezzyat & Davachi 2011（Neuron）：新学习的 item 在海马/颞叶的表征压缩
- 常见坑：
  - `pair_id` 必须在 `stack_patterns.py` 生成 metadata 时就写入；如果只保留了 `pic_num`，确保同一 pair 的两个 concept word 共享同一 pic_num。
  - 若一个 concept word 出现在多对中（材料设计有复用），M2 的 0/1 定义会出现歧义，需要确认 1 个 word ↔ 1 个 pair 的一对一映射。

#### M3_embedding（预训练语义 RDM，预存语义基线）

- 构造思路：用预训练语言模型（默认 `bert-base-chinese`）把每个 concept word 编码成 768 维向量，计算两两 cosine 距离。表达"在见到实验之前，这两个词在通用汉语语料里离得多远"。
- 数学形式：
  - `v_i = mean-pool(BERT.encode(word_i))`（使用 attention_mask 过滤 padding）
  - `RDM_M3[i,j] = 1 − cos(v_i, v_j)`，范围近似 [0, 2]
- 在脚本中：
  - 预计算：[build_stimulus_embeddings.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/rsa_analysis/build_stimulus_embeddings.py) 一次性跑完所有刺激词，写入 `stimulus_embeddings_bert.tsv`（列：`word_label` + `dim_0..dim_767`）
  - 查询：`model_from_embedding(metadata, embeddings, word_col="word_label")`，缺词对应行列填 NaN，相关时自动掩码
- 理论假设：
  - 前测：M3 应该是**最主导的解释变量**——被试还没学习，ROI 编码的主要是词汇的预存分布式语义。
  - 后测：M3 的 ρ 可能**略降或持平**，部分方差被 M2/M7 分走；但不会消失，因为预存语义结构依然存在。
  - 偏相关：控制 M2/M7 后，M3 的 partial_rho 在 pre/post 都应 > 0，稳定性强的 ROI（如 L_ATL、L_AG）尤其明显。
- 实际读结果时怎么用：
  - 如果 M3 在 pre 很强、post 仍然存在，说明学习不是“抹掉原有语义”，而是在原有语义结构上叠加新的学习结构。
  - 如果 M3 在 pre/post 都很弱，先不要急着讲“学习塑造”，要先排查 ROI 是否根本不编码语义结构。
- 文献依据：
  - Bhattasali et al. 2018 / 2019：BERT-like 模型与语义 ROI 的表征相似
  - Caucheteux & King 2022 (Nature Communications)：大语言模型 embedding 预测大脑语言区激活
  - Pereira et al. 2018 (Nature Communications)：分布式语义解码 concept 表征
- 备选模型源（在 `--embedding-source` 中切换或并列对照）：
  - Tencent word2vec 200d（更轻量，适合句内词）
  - 腾讯 ChineseGPT / Qwen embedding API（若可联网）
  - `random` 兜底：仅为测试流程用，不应写入论文
- 常见坑：
  - 多字词被 BERT 切成多个 sub-token，默认用 mean-pool；若换成 `[CLS]` 表征，结果会略有差异——在方法学部分明确说明。
  - 预计算完成后 embedding 文件需固定（加入版本号/hash），确保论文可复现。
  - 若某些概念词在 BERT 词表中罕见（例如生僻字），vector 可能全接近零；`build_stimulus_embeddings.py` 会保留 NaN，RDM 构造时自动剔除该行列。

#### M7_memory（被试回忆强度 RDM，行为-神经桥梁）⭐

- 构造思路：每个被试自己在 Run-7 的 1-7 级回忆评分（或 0/1 成功失败）构成一个 `word → recall_score` 映射。把"同一对 word 学得都牢"编码成"它们在 M7 上距离小"。
- 数学形式：`RDM_M7[i,j] = |recall_i − recall_j|`，**per-subject** 计算。
- 在脚本中：
  - 预计算：[build_memory_strength_table.py](file:///Users/bytedance/Documents/trae_projects/PythonAnalysis/metaphoric/final_version/rsa_analysis/build_memory_strength_table.py) 从 events.tsv 的 `memory`/`action` 列派生，每被试输出 `memory_strength_sub-XX.tsv`
  - 查询：`model_from_subject_numeric(metadata, mem_table, value_col="recall_score")`，缺表的被试会被记录 `skip_reason=missing_memory_table`
- 理论假设：
  - 前测：M7 的 ρ ≈ 0（被试还没学，记忆强度未定义；若使用 0/1 编码，前测将全部为 0 → M7 退化，此时仅在 post 报告）。
  - 后测（不预设方向）：M7 反映“记忆强度差异是否能解释表征结构差异”。如果主效应是对齐（convergence），可能出现“记得越牢越对齐”；如果主效应是分化（differentiation），也可能出现“记得越牢越分化”。因此更推荐用偏相关与 Δρ LMM 来判断其独立贡献与方向。
  - 条件交互：yy 条件下 M7 的效应更可能更强（新颖性更高 → 学习强度个体差异更大），但方向需要以实际估计为准。
- 文献依据：
  - Xue et al. 2010 (Science)：记忆成功 ↔ 神经模式"稳定性"
  - LaRocque et al. 2013：与记忆的 representational similarity
  - Kuhl & Chun 2014：记忆成绩个体差异 ↔ reinstatement strength
- 为什么必须 per-subject：
  - 每个被试记忆曲线不同（顶尖被试可能 40 对全记住，差的只记住 10 对）
  - 跨被试平均后 `|recall_i − recall_j|` 会被压缩，信号减弱
  - 因此 **M7 的 neural RDM 也必须 per-subject 计算并相关**，然后在组水平做 one-sample t
- 直觉图像：
  - M7 不是在问“你记没记住这对词”，而是在问“不同项目之间的记忆强弱差异，能不能解释表征结构差异”。
  - 所以 M7 更像“项目学习强度地图”，不是简单的条件标签。
- 常见坑：
  - 若行为数据只有 0/1（`action==3` → 1 否则 0），M7 退化为"是否记住"的二值 RDM，类似 M2 但更个体化；此时与 M2 的共线性会被 `model_collinearity.tsv` 捕获
  - 若某被试几乎全部记住或全部失败（方差 ≈ 0），M7 的 RDM 几乎全 0，Spearman 计算会退化为 NaN，代码会跳过并写 `skip_reason=memory_variance_too_low`（未来优化）
  - 被试级 N 决定 M7 的统计功效；28 人已足够，但审稿人可能问"若拿掉 top/bottom 2 个 outlier 结论是否稳定"——建议在 Supplementary 做 leave-one-out

#### 五模型联合检验的核心故事线

- 单模型 ρ 回答：每个模型**单独**能解释多少 neural RDM 方差？
- 偏相关（`--partial-correlation`）回答：在排除高共线/互补模型干扰后，目标模型**独一无二**解释多少方差？这是区分"预存语义 (M3)"与"学习塑造 (M2/M7/M8)"的关键。
- Δρ LMM 回答：post − pre 这个增量是否显著，且是否受 condition × model 调节？
- 三者叠加后的"一张图讲完的故事"：
  - 前测：M3 单模型和偏相关都大，M2/M7 ≈ 0
  - 后测：用偏相关区分“预存语义 (M3)”与“学习塑造 (M2/M7 或分化型模型)”的独立贡献；M3 通常仍为正，但学习相关模型的贡献与方向以数据为准
  - Δρ：不假设必为正。结合 Step 5C 的方向来解释：
    - 若 Step 5C 显示配对更对齐（similarity 上升），重点报告 M2/M7 的正向 Δρ
    - 若 Step 5C 显示配对更分化（similarity 下降），重点报告分化型模型（如 M8_reverse_pair）的 Δρ，并用 M3 的稳定性作为语义锚点证据
  - 结论模板：**“学习前神经表征主要由预存语义（M3）解释；学习后出现额外的学习相关结构（可能表现为对齐或分化），且该变化在隐喻条件（yy）中最强。”**

如果你只看 3 个结果表，优先级建议是：
1. `model_rdm_subject_metrics.tsv`
   - 看每个模型在 pre/post 的单模型 ρ，大致判断谁在前测主导、谁在后测增强。
2. `model_rdm_partial_metrics.tsv`
   - 看控制其他模型后，哪个模型还有“独立贡献”。
3. `delta_rho_lmm_params.tsv`
   - 看真正的组水平增量检验，决定“哪个模型的变化”最能支撑论文主结论。

最容易误读的地方：
- **M2 显著** 不等于 “学习一定导致了配对收敛”，因为还要结合 Step 5C 的方向、M8 的结果、以及 partial_rho 一起看。
- **M3 仍显著** 不等于 “学习没发生”，它往往只是说明预存语义仍然是表征的底盘。
- **M7 显著** 不等于 “记忆导致了一切”，它只能说明行为学习强度与表征结构变化存在系统联系。

为什么要做多模型偏相关（`--partial-correlation`）：
- M3 和 M2 可能共线（同配对的两个词语义也更相近）
- 偏相关把每个模型"独一无二"的解释方差剥离出来
- 但 **M2_pair** 与 **M8_reverse_pair** 是互补编码（近似完美负相关），两者不应互相作为对方的控制变量；当前脚本会在偏相关时自动把这对互补模型从彼此控制集中剔除，同时把完整关系保留在 `model_collinearity.tsv`
- 输出 `model_rdm_partial_metrics.tsv`（列：subject/roi/time/model/partial_rho）和 `model_collinearity.tsv`（|ρ|>0.7 标 high）

输入是什么：
- pattern_root：前后测的 4D patterns（`stack_patterns.py` 的输出）
- roi_dir：本次三层 ROI 中的某一层
- `--embedding-file stimulus_embeddings/stimulus_embeddings_bert.tsv`（M3）
- `--memory-strength-dir memory_strength/`（M7）
- `--conditions yy kj baseline`（默认三条件）
- `--pair-id-col pair_id`（默认）
- `--partial-correlation`（开启偏相关 + 共线性诊断）

输出是什么（自动加 ROI_SET 后缀，可 `METAPHOR_MODEL_RDM_OUT_DIR` 覆盖）：
- `model_rdm_subject_metrics.tsv`：每 subject/roi/time/model 的 Spearman ρ
- `model_rdm_partial_metrics.tsv`：多模型偏相关
- `model_collinearity.tsv`：模型间两两 Spearman ρ
- `model_rdm_group_summary.tsv`：ROI×模型 pre-vs-post 配对 t
- `<roi>_<model>_rdm_summary.json`：每个 ROI×模型的配对 t 摘要

然后 Δρ LMM（`delta_rho_lmm.py`）回答什么：
- 把上一步按 subject/roi/model 合并 `delta_rho = ρ_post − ρ_pre`
- 拟合 `delta_rho ~ C(model) + (1|subject) + (1|roi)`，核心检验是“哪些模型的 Δρ 显著偏离 0、且在 ROI 间具有一致方向”；不应预设 M2/M7 必为正
- 实现注意：脚本在 statsmodels 中用 `groups=subject` + `vc_formula={"roi": "0 + C(roi)"}` 来近似 crossed random effects（而不是把 roi 真的当作 nested）；在方法与讨论里需要如实报告这一选择
- 若想加条件轴，先在 RSA 阶段按 condition 分别算 ρ（作为 `condition_col`），再用 `--condition-col condition`

典型跑法（三层 ROI 各跑一次互不覆盖，并结合偏相关）：

```bash
# 1) 预计算一次（全体 ROI_SET 共用）
python -m rsa_analysis.build_stimulus_embeddings \
  --embedding-source bert --model-name bert-base-chinese

python -m rsa_analysis.build_memory_strength_table \
  --subject-range 1 28

# 2) 主功能 ROI 层
export METAPHOR_ROI_SET=main_functional
python -m rsa_analysis.model_rdm_comparison \
  "$PYTHON_METAPHOR_ROOT/patterns_stacked" \
  "$PYTHON_METAPHOR_ROOT/roi_library/main_functional" \
  --embedding-file "$PYTHON_METAPHOR_ROOT/stimulus_embeddings/stimulus_embeddings_bert.tsv" \
  --memory-strength-dir "$PYTHON_METAPHOR_ROOT/memory_strength" \
  --partial-correlation

# 3) 同样跑文献层 literature、鲁棒层 atlas_robustness
# 4) 对每层输出跑 Δρ LMM
python -m rsa_analysis.delta_rho_lmm \
  --metrics-file "$PYTHON_METAPHOR_ROOT/model_rdm_results_main_functional/model_rdm_subject_metrics.tsv" \
  --output-dir "$PYTHON_METAPHOR_ROOT/model_rdm_results_main_functional/lmm"
```

端到端链路（写给论文 Figure 5 的故事线）：

```
[词表] build_stimulus_embeddings → stimulus_embeddings_bert.tsv ─┐
[行为] build_memory_strength_table → memory_strength_{sub}.tsv ──┼─┐
                                                                │ │
[fMRI] LSS beta → stack_patterns → pattern_root ────────────────┘ │
                                                                  │
[ROI]  build_roi_library → {main_functional, literature, atlas}   │
                                                                  ↓
                                   model_rdm_comparison.py
                                   ├─ M1_condition (yy/kj/baseline)
                                   ├─ M2_pair      (pair_id)
                                   ├─ M8_reverse_pair (reverse pair)
                                   ├─ M3_embedding (BERT cosine)
                                   └─ M7_memory    (recall diff)
                                         │
                                         ├─ Spearman ρ per subject/roi/time
                                         ├─ partial_rho (控制其他模型)
                                         └─ collinearity flag (ρ>0.7 预警)
                                         ↓
                                   delta_rho_lmm.py
                                   delta_rho = post − pre
                                   LMM: ~ C(model) + (1|subject) + (1|roi)
                                         ↓
                                   Figure 5 (前测 M3 主导 → 后测学习相关模型变化，方向由 M2/M8/M7 共同决定)
```

常见坑：
- metadata 里 `word_label` 列缺失会让 M3/M7 静默跳过（结果表里 `skip_reason=missing_*`）——一定要用 `stack_patterns.py` 生成含 `word_label` 的 metadata
- 若仅跑 `--conditions yy kj`（不含 baseline），M1 其实只区分两类，解释力会被削弱
- `--partial-correlation` 开启前先看 `model_collinearity.tsv`，|ρ|>0.9 的两个模型不应同时进入偏相关回归；尤其 `M2_pair` 与 `M8_reverse_pair` 会呈近似完美负相关
- Δρ LMM 要求 pre/post 都存在，某 subject/roi/model 缺一边会被剔除

最小 QC：
- `model_rdm_subject_metrics.tsv` 中 `skip_reason=""` 的行占主体
- `model_collinearity.tsv` 没有大面积 `flag=high`
- `delta_rho_lmm_model.json` 的 `n_obs` 接近 `n_subjects × n_rois × 5 models × 2 times`（默认主线包含 `M1/M2/M3/M7/M8`）

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
- `rsa_analysis/brain_behavior_correlation.py`
- `brain_behavior/brain_behavior_correlation.py`

这步回答什么问题：
- 神经变化大的被试，行为上是不是也学得更好

为什么要做：
- 如果神经指标和行为没有任何关系，论文故事会停留在“平行现象”
- 有关联时，你可以说不同层次证据彼此支持

主线推荐脚本（ROI 级相关）：
- `rsa_analysis/brain_behavior_correlation.py`
- 直接读取 `rsa_itemwise_details.csv`，先算每个 ROI、每个被试的 `Δ similarity = post - pre`
- 再与 Run-7 行为（例如 `accuracy_yy_run7`）做相关/偏相关
- 更适合当前主文主线，因为它直接对应 “表征变化是否预测行为收益”

通用汇总脚本（多指标拼表）：
- `brain_behavior/brain_behavior_correlation.py`
- 适合把多个 `subject-level` 指标文件先合并，再做 Pearson / Spearman 相关矩阵
- 更适合补充材料、探索分析或中介分析前的数据整理

输入是什么：
- 若干 `subject-level` 指标文件，或 `rsa_itemwise_details.csv + behavior.tsv`
- 每个输入文件必须有 `subject` 列

比较理想的输入例子：
- 行为差值表
- RSA 汇总后的被试指标
- RD 或 GPS 的被试指标
- MVPA 准确率表

脚本会做什么：
- 主线版：先从 RSA item-wise 表构造每个 ROI 的 subject-level `Δ similarity`，再与行为做 ROI 级相关/偏相关
- 汇总版：先把各表按 `subject` 合并，再对所有数值列做两两 Pearson 和 Spearman 相关

输出是什么：
- 主线版：
  - `brain_behavior_correlation.tsv`
  - `brain_behavior_scatter.png`
- 汇总版：
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
- 若主线版用偏相关，检查协变量列是否缺失过多
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

**先用行为确认学到了东西，用学习期 GLM 定位相关脑区，用前后测 RSA 证明表征关系发生重组，再用 Model-RSA 解释这种变化更接近哪条机制路径，随后用脑-行为关联把神经变化和行为收益连起来；RD/GPS 作为几何层面的互补证据。**
