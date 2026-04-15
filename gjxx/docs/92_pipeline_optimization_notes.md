> 归档说明：此文档由 `gjxx/整体分析链路优化建议.md` 移入 `gjxx/docs/`，内容基本不改，仅用于历史参考。

# 整体分析链路优化建议

> NOTE: 本文档记录的是重构前的历史现状与建议，文件名示例（如 `run_story_*`、`run_*.py`、`memory_dimension_analysis.py`）可能已在后续迭代中按 `run_stageNN_*.py` 规范重命名。

## 1. 背景与结论

这个目录里的内容，已经不只是“若干 Python 脚本”，而是把两类来源揉在了一起：

1. 一类来自 `GJXX_1_reanalysis` 那套更完整的上游分析链。
2. 一类来自 `FLXX1/Dimension` 那套更偏结果表达和论文故事的下游维度分析链。

因此，当前真正的问题不是单点命名差，而是：

- 主线故事不止一条。
- 入口层、方法层、历史映射层混放在同一层目录。
- 文件名同时混用了“MATLAB 来源名”“分析方法名”“故事化命名”“历史实验名”四种语言。

如果要把这套工程整理成一个更清晰、可维护、可继续扩展的分析工程，建议先统一叙事模型，再统一命名和目录。

---

## 2. 当前工程里实际存在的两条故事线

### 2.1 故事线 A：完整分析生产链

这条线主要由 `cli.py`、`events.py`、`first_level.py`、`patterns.py`、`roi.py`、`rd.py`、`group.py` 体现。

它描述的是一个相对完整的上游到下游流程：

`events 清洗/重标记 -> item-level GLM -> trial map 拼接成 4D pattern -> ROI/GPS/RD/searchlight/group regression`

对应的分析意图是：

1. 从事件文件出发重写标签。
2. 跑 item-level 一阶模型。
3. 把单 trial 图堆叠成 4D pattern。
4. 在 ROI、RDM、searchlight、group regression 上做统计。

这一套更像“可复用的分析流水线”。

### 2.2 故事线 B：论文/汇报导向的维度分析故事

这条线主要由 `dimension_analysis_core.py`、`dimension_story_utils.py`、`run_*.py`、`README_memory_dimension.md` 体现。

它描述的是一个更偏结果汇总的研究故事：

1. remembered / forgotten ROI 分析
2. memory-good / memory-poor ROI 分析
3. 主 ROI 维度与稳健性分析
4. 行为分数 pooling 分析
5. RD searchlight
6. seed connectivity searchlight

这一套更像“把已经存在的 MATLAB 分析故事转成少量 Python 入口”。

### 2.3 当前最核心的冲突

这两条线都合理，但它们回答的是不同问题：

- 故事线 A 回答“整个分析是怎么生产出来的”。
- 故事线 B 回答“最后要讲的科学结果有哪些部分”。

现在它们被放在同一层目录，导致使用者很难一眼判断：

- 哪个是主入口。
- 哪个是共享底层。
- 哪个是论文故事包装层。
- 哪个只是 MATLAB 来源说明。

---

## 3. 我建议采用的统一故事脉络

建议把整个工程统一讲成下面这条主线：

### 3.1 主故事

`原始事件 -> 事件标准化 -> item-level GLM -> pattern 组装 -> 条件/维度分析 -> 组水平统计 -> 汇总输出`

### 3.2 在主故事下再分分析分支

#### A. 数据准备层

- 事件重标记
- 一阶 GLM
- trial/stat map 堆叠

#### B. ROI 条件差异层

- GPS
- ROI-ROI DSM correlation
- remembered vs forgotten
- memory-good vs memory-poor

#### C. ROI 维度与稳健性层

- pattern PCA
- RDM PCA
- same-trial equalization
- demean
- voxel mask
- PC window robustness

#### D. 全脑映射层

- RD searchlight
- group paired map statistics
- behavior regression
- seed connectivity searchlight

#### E. 探索/历史映射层

- MATLAB 来源脚本梳理
- 历史命名保留
- 旧参数分支说明

这样整理以后，使用者会自然知道：

- 先看主故事。
- 再看自己关心的分析分支。
- MATLAB 映射作为旁路参考，而不是和主入口抢位置。

---

## 4. 当前命名和串联的主要问题

### 4.1 入口命名风格混杂

当前入口同时存在：

- `cli.py`
- `memory_dimension_analysis.py`
- `run_remembered_forgotten_dimension.py`
- `run_memory_group_dimension.py`
- `run_story_roi_suite.py`
- `run_story_behavior_suite.py`
- `run_story_searchlight_suite.py`
- `run_story_connectivity_suite.py`

问题在于：

- 有的名字强调执行动作，例如 `run_*`
- 有的名字强调分析对象，例如 `memory_group_dimension`
- 有的名字强调“故事”，例如 `run_story_*`
- 有的名字其实只是帮助说明，却看起来像主分析脚本，例如 `memory_dimension_analysis.py`

结果是目录首页没有一个统一入口语法。

### 4.2 底层模块命名风格也不统一

底层模块里同时存在两套语言：

- 通用流水线语言：`events.py`、`first_level.py`、`patterns.py`、`roi.py`、`rd.py`、`group.py`
- 研究故事语言：`dimension_analysis_core.py`、`dimension_story_utils.py`

前者像“可复用方法模块”，后者像“为这篇故事专门拼装的共享层”。

这会让维护者不知道：

- 新功能应该继续往通用模块里加。
- 还是继续往 `dimension_story_utils.py` 这种大杂糅文件里加。

### 4.3 历史实验名直接进入正式输出语义

例如：

- `untitled2_top10_bottom50`
- `untitled3_top10_bottom50`
- `hl_patternbutnotrdm`
- `pc_window_3_5_fixed20`

这些名字对“当时写 MATLAB 的作者”有记忆线索，但对后来维护者不够自解释。

建议保留历史映射，但不要让历史名直接承担正式对外命名。

### 4.4 包结构不完整

当前代码同时出现：

- 相对导入：`from .config import ...`
- 直接脚本式导入：`from dimension_analysis_core import ...`

但目录下没有 `__init__.py`，说明当前既不像完整包，也不像完全独立脚本集合。

这会带来至少两个问题：

1. 运行方式不稳定。
2. 后续继续拆模块时，导入规范会越来越乱。

### 4.5 文档入口不够明确

当前文档包括：

- `README_memory_dimension.md`
- `MATLAB_流程梳理.md`
- `MATLAB_脚本梳理.md`

它们都很有价值，但职责没有分开：

- 哪个是“当前工程总览”
- 哪个是“MATLAB 来源映射”
- 哪个是“历史资产审计”

用户第一次进入目录时，不知道应该先看哪一个。

### 4.6 配置与路径耦合过重

多个模块里直接内置了 `I:\`、`L:\` 等绝对路径默认值。

这虽然方便快速对接原工程数据，但也意味着：

- 代码结构层和运行环境层耦合太紧。
- 未来想迁移到别的磁盘、服务器、容器或共享工作站时成本很高。

---

## 5. 推荐的整合原则

### 5.1 一个主入口语言

建议统一采用“阶段 + 分支”的入口命名方式，而不是继续混用 `story`、`memory`、`dimension`、`run`。

推荐风格：

- `run_stage01_events.py`
- `run_stage02_item_glm.py`
- `run_stage03_patterns.py`
- `run_stage10_roi_core.py`
- `run_stage11_memory_split.py`
- `run_stage12_behavior_pooling.py`
- `run_stage20_rd_searchlight.py`
- `run_stage21_seed_connectivity.py`

这样做的好处是：

- 从文件名就能看到它在故事中的位置。
- 新分析以后也容易继续插入。
- 入口层天然形成阅读顺序。

### 5.2 一个统一模块分层

建议按下面四层理解和整理工程：

#### 第 1 层：入口层

只负责参数、日志、编排，不承载核心算法。

#### 第 2 层：分析管线层

负责把多个方法步骤串成一个可运行任务。

#### 第 3 层：方法/统计层

负责 PCA、RDM、searchlight、group stats、trial equalization 等核心逻辑。

#### 第 4 层：文档/历史映射层

负责解释 MATLAB 来源、历史版本、命名沿革，不与执行模块混放。

### 5.3 历史名保留为映射，不保留为主名

比如：

- `untitled2_top10_bottom50`
- `hl_patternbutnotrdm`

可以在文档或配置中保留：

- `legacy_name`
- `matlab_source`

但对新的 Python 入口和输出目录，建议改成更语义化名字。

### 5.4 输出命名也要统一

所有分析输出建议统一至少三类文件：

- `summary.json`
- `subject_metrics.tsv`
- `group_stats.json`

对于 searchlight / map 分析再补：

- `subject_maps/`
- `group_maps/`

这样后续无论是做汇总脚本、批处理还是自动报告，成本都会低很多。

---

## 6. 推荐的目标目录结构

下面是一个更适合长期维护的目标结构示意：

```text
gjxx/
  README.md
  docs/
    01_story_overview.md
    02_matlab_source_mapping.md
    03_optimization_plan.md
  pipeline/
    __init__.py
    config.py
    events.py
    first_level.py
    patterns.py
    roi_metrics.py
    rd_metrics.py
    searchlight.py
    connectivity.py
    group_stats.py
    io_utils.py
  apps/
    run_stage01_events.py
    run_stage02_item_glm.py
    run_stage03_patterns.py
    run_stage10_roi_core.py
    run_stage11_memory_split.py
    run_stage12_behavior_pooling.py
    run_stage20_rd_searchlight.py
    run_stage21_seed_connectivity.py
    run_all_story.py
  legacy/
    matlab_flow_review.md
    matlab_script_inventory.md
```

这不是要求一次性重构到位，而是给后续整合提供一个目标形态。

---

## 7. 当前文件到建议命名的映射

### 7.1 文档层

- `README_memory_dimension.md`
  - 建议改为：`README.md`
  - 或拆成：`docs/01_story_overview.md`

- `MATLAB_流程梳理.md`
  - 建议改为：`docs/02_dimension_matlab_flow.md`

- `MATLAB_脚本梳理.md`
  - 建议改为：`docs/03_gjxx1_matlab_inventory.md`

### 7.2 入口层

- `memory_dimension_analysis.py`
  - 建议改为：`apps/show_entrypoints.py`
  - 如果只是帮助展示入口，也可以直接删掉，用 README 替代

- `run_remembered_forgotten_dimension.py`
  - 建议改为：`apps/run_stage11_remember_forget_roi.py`

- `run_memory_group_dimension.py`
  - 建议改为：`apps/run_stage11_memory_group_roi.py`

- `run_story_roi_suite.py`
  - 建议改为：`apps/run_stage10_roi_robustness.py`

- `run_story_behavior_suite.py`
  - 建议改为：`apps/run_stage12_behavior_pooling.py`

- `run_story_searchlight_suite.py`
  - 建议改为：`apps/run_stage20_rd_searchlight.py`

- `run_story_connectivity_suite.py`
  - 建议改为：`apps/run_stage21_seed_connectivity.py`

### 7.3 底层模块

- `dimension_analysis_core.py`
  - 建议拆分为：
    - `pipeline/roi_metrics.py`
    - `pipeline/io_utils.py`
    - `pipeline/stats_utils.py`

- `dimension_story_utils.py`
  - 建议拆分为：
    - `pipeline/roi_robustness.py`
    - `pipeline/searchlight.py`
    - `pipeline/connectivity.py`
    - `pipeline/behavior_pooling.py`

- `roi.py`
  - 建议改为：`pipeline/roi_similarity.py`
  - 避免与更宽泛的 ROI 分析概念混淆

- `rd.py`
  - 建议改为：`pipeline/rd_metrics.py`

- `group.py`
  - 建议改为：`pipeline/group_stats.py`

---

## 8. 推荐的统一分析编号

为了让故事可讲、目录可排、输出可比，建议每条分析给一个稳定编号。

### 8.1 基础构建

- `01` 事件标准化
- `02` item-level GLM
- `03` pattern 组装

### 8.2 ROI 与条件差异

- `10` 核心 ROI 维度分析
- `11` memory split 分析
- `12` behavior pooling 分析
- `13` ROI similarity / DSM correlation

### 8.3 全脑映射

- `20` RD searchlight
- `21` seed connectivity searchlight
- `22` group regression

### 8.4 历史和映射

- `90` MATLAB 来源映射
- `91` 历史参数分支
- `92` 决策日志

这样以后无论是文件、输出目录还是汇报 PPT，都能用同一套编号体系。

---

## 9. 我建议优先落地的最小整合方案

### 第一阶段：只整理信息架构，不动分析逻辑

目标：先让人能看懂。

建议动作：

1. 增加一个真正的根 `README.md`
2. 把两个 MATLAB 梳理文档移到 `docs/`
3. 把当前入口按阶段重命名或至少建立别名说明
4. 明确标注“主入口”“共享模块”“历史文档”

这一阶段的收益最大，风险最低。

### 第二阶段：统一包结构

目标：先让运行方式稳定。

建议动作：

1. 增加 `__init__.py`
2. 统一导入方式
3. 明确是按包运行还是按脚本运行
4. 避免同时使用相对导入和裸模块导入

### 第三阶段：拆分过大的共享模块

目标：先让职责清晰。

建议动作：

1. 拆 `dimension_analysis_core.py`
2. 拆 `dimension_story_utils.py`
3. 把“数学方法”“IO”“分析编排”拆开
4. 保持旧接口一段时间，避免重构后全体入口一起失效

### 第四阶段：统一配置与输出

目标：先让工程更容易迁移和复跑。

建议动作：

1. 把绝对路径收进单独配置文件
2. 让输出目录结构稳定
3. 让每条分析都写统一格式的 summary 和 subject metrics

---

## 10. 一份更适合现在工程的总览说法

如果现在就要给别人介绍这个目录，我建议统一成下面这段话：

> 这是一个把两套相关 MATLAB 分析工程整合成 Python 工作区的中间版本。  
> 其中一部分负责复现从事件处理到一阶模型、pattern 组装、ROI/GPS/RD/searchlight/group regression 的完整分析链；另一部分负责把最终论文/汇报相关的维度分析故事整理成若干高层入口。  
> 当前最需要优化的不是单个算法，而是信息架构：需要把主线入口、共享方法、历史 MATLAB 映射和结果故事分层整理，让使用者能一眼看清“从哪里开始、每步产出什么、不同分析分支如何关联”。  

这段说法能比较准确地描述当前工程的真实状态。

---

## 11. 本次 review 的核心判断

### 应当保留的优点

- 你已经把 MATLAB 的散脚本压缩成了更少的 Python 入口。
- 研究故事的几条核心分支已经基本成型。
- 两份 MATLAB 梳理文档为后续整合提供了非常好的来源依据。

### 当前最影响可读性的点

- 不是“脚本太少”，而是“入口太平铺”。
- 不是“文档不够”，而是“文档职责没有分层”。
- 不是“名字单个不好”，而是“命名体系没有统一语法”。

### 我建议的主方向

先统一故事，再统一命名；先统一入口，再拆共享模块；先把历史映射降到文档层，再把执行层变成稳定的阶段化流水线。

---

## 12. 建议后续继续产出的文档

如果你后面准备继续整理这个工程，我建议下一步补三份文档：

1. `README.md`
   - 面向第一次进入工程的人
   - 只讲总览、入口、运行顺序

2. `docs/analysis_story_map.md`
   - 面向需要理解整体科学问题的人
   - 讲主故事和各分析分支的关系

3. `docs/matlab_to_python_mapping.md`
   - 面向复现/核对的人
   - 讲 MATLAB 脚本到 Python 模块和入口的对应关系

这三份文档一旦补齐，整个工程的可读性会明显提升。
