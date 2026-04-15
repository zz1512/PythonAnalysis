# MATLAB 脚本梳理

## 1. 我先给出的结论

这个目录不是一个“从入口脚本一路自动跑到底”的 MATLAB 工程，而是作者在同一批 `pattern` 数据上，分阶段手动运行很多独立脚本做分析、复现和参数试验。

当前最像“真实跑过并留下结果”的主线有两条：

1. `pattern/sub-*/glm_T_stats_HSC.nii` 与 `glm_T_stats_LSC.nii` 为核心输入。
2. 一条是基于 searchlight 的局部维度分析：
   - `searchlight_RD.m`
   - `my_var_measure.m`
   - `pairedttest.m`
3. 另一条是基于 ROI PCA 的连接性 searchlight：
   - `stage1_pca_conn_searchlight.m`
   - `my_measure.m`
   - `searchlight_pca_left_80/second/pairedttest.m`

除此之外，大量脚本是在同一输入上改 ROI、改阈值、改 trial 数、改 PCA 指标，属于派生分析或试验版本。

## 2. 目录里的数据分层

### 核心输入

- `pattern/sub-*/glm_T_stats_HSC.nii`
- `pattern/sub-*/glm_T_stats_LSC.nii`
- 部分旧分析还会用：
  - `pattern/sub-*/HSC_r.nii`
  - `pattern/sub-*/LSC_r.nii`

说明：

- 这些 `pattern` 结果是整个目录里几乎所有分析脚本的共同输入。
- 当前工作区里没有能直接生成这批 `glm_T_stats_*.nii` 的完整脚本。
- `Copy_2_of_get_rsa_patterns_gps.m` 能拼接别的路径下的一级图像，但它指向 `H:\GJXX_2_reanalysis\...`，不是当前项目主线。

### 中间结果

- `pattern/sub-*/rdvar_hsc_70.nii`
- `pattern/sub-*/rdvar_lsc_70.nii`
- `searchlight_pca_left_80/sub-*/searchlight_lHIP_hsc.nii`
- `searchlight_pca_left_80/sub-*/searchlight_lHIP_lsc.nii`
- `voxelmask.mat`

### 二级结果

- `second70/`：`rdvar_*_70.nii` 的二级配对 t 检验结果
- `searchlight_pca_left_80/second/`：连接性 searchlight 的二级配对 t 检验结果
- 一批 `sametrialn_*.mat`、`mack2020.mat`、`drop_voxel_*.mat`：ROI 级统计结果

## 3. 我判断出的实际分析流程

### 主线 A：searchlight 局部维度分析

#### 第一步：单被试 searchlight

- `searchlight_RD.m`
- 调用 `my_var_measure.m`

逻辑：

1. 对每个被试分别读取 `glm_T_stats_HSC.nii` 和 `glm_T_stats_LSC.nii`
2. 在全脑 `mask.nii` 内做球形邻域 searchlight
3. 每个球里计算 RDM，再算达到 70% 解释率所需维度数
4. 输出：
   - `pattern/sub-*/rdvar_hsc_70.nii`
   - `pattern/sub-*/rdvar_lsc_70.nii`

为什么判断它真实跑过：

- 脚本输出名与磁盘现有文件完全一致
- `searchlight_RD.m` 时间是 `2023-12-16 15:57`
- `pattern/sub-*` 下的 `rdvar_*_70.nii` 时间也是 `2023-12-16`

#### 第二步：组水平统计

- `pairedttest.m`

逻辑：

1. 收集每个被试的：
   - `rdvar_hsc_70.nii`
   - `rdvar_lsc_70.nii`
2. 做配对 t 检验
3. 输出到 `second70/`

为什么判断它真实跑过：

- `pairedttest.m` 明确写死输出目录 `second70`
- `second70/` 里有完整的 `SPM.mat`、`con_*.nii`、`spmT_*.nii`
- 时间戳与脚本修改时间高度一致

#### 相关但偏历史的分支

- `just_smooth.m`

它会平滑 `rdvar_hsc*.nii` 和 `rdvar_lsc*.nii`，生成前缀 `sss` 的图。
能对应上目录里 `sssrdvar_*.nii`，但它操作的是 `rdvar_hsc.nii / rdvar_lsc.nii` 这种旧命名，不是当前 `rdvar_*_70.nii` 主线。

因此它更像早期版本的后处理，而不是当前主线。

### 主线 B：ROI-PCA 连接性 searchlight

#### 第一步：单被试连接性 searchlight

- `stage1_pca_conn_searchlight.m`
- 调用 `my_measure.m`

逻辑：

1. 先在左海马 ROI 内，对 HSC 或 LSC 条件计算 PCA 基准结构
2. 再在全脑 searchlight 中，比较每个球的 PCA 结构与 ROI PCA 结构的相似度
3. 输出：
   - `searchlight_pca_left_80/sub-*/searchlight_lHIP_hsc.nii`
   - `searchlight_pca_left_80/sub-*/searchlight_lHIP_lsc.nii`

为什么判断它真实跑过：

- 输出文件与脚本完全匹配
- `searchlight_pca_left_80/sub-*` 有 23 个被试的成套结果
- 时间集中在 `2023-12-02`

#### 第二步：组水平统计

- `searchlight_pca_left_80/second/pairedttest.m`

逻辑：

1. 收集每个被试的 `searchlight_lHIP_hsc.nii` 和 `searchlight_lHIP_lsc.nii`
2. 做配对 t 检验
3. 输出到 `searchlight_pca_left_80/second/`

为什么判断它真实跑过：

- `second/` 下有完整 SPM 二级结果
- 时间与单被试结果一致

## 4. ROI 级分析脚本家族

这一组脚本基本都共享同一个骨架：

1. 读取 `pattern/sub-*/glm_T_stats_HSC.nii` 与 `glm_T_stats_LSC.nii`
2. 在某个 ROI 掩膜内取体素
3. 有时先平衡 trial 数
4. 用 PCA 或 RDM-PCA 提取“维度”指标
5. 对 HSC 和 LSC 做配对 t 检验
6. 部分脚本把结果 `save` 成 `.mat`

### 4.1 真正能对上现有输出的 ROI 脚本

- `hl_patternbutnotrdm.m`
  - 左海马
  - 直接对 pattern 做 PCA，不经过 RDM
  - 输出 `mack2020.mat`
  - 时间戳对得上，确定跑过

- `get_voxel_mask.m`
  - 对每个 voxel 做 HSC vs LSC 的跨被试 t 检验
  - 不显著的 voxel 标 1，显著的标 0
  - 输出 `voxelmask.mat`
  - 时间戳对得上，确定跑过

- `sametrialnumber_maskvoxel.m`
  - 依赖 `voxelmask.mat`
  - 只保留 `mymask==1` 的 voxel
  - 再做 trial 数平衡 + PCA 维度统计
  - 输出 `drop_voxel_sametrialn_70_90_lHPC.mat`
  - 时间戳对得上，确定跑过

- `demeantest.m`
  - 先把 HSC 和 LSC 拼起来做均值中心化，再分回两组
  - 输出 `sametrialn_70_90_HPC_demean.mat`
  - 时间戳对得上，确定跑过

- `demeantest2.m`
  - 与 `demeantest.m` 相同，但固定 `true_length=25`
  - 输出 `sametrialn_70_90_HPC_demean_v2.mat`
  - 时间戳对得上，确定跑过

- `Untitl1ed.m`
  - ROI 改成 `aHPC_L.nii`
  - 输出 `sametrialn_80_90_ahpc.mat`
  - 时间戳对得上，确定跑过

- `Untit3led.m`
  - 左海马，阈值范围改成 `1:49`
  - 输出 `sametrialn_1_49_lhpc.mat`
  - 时间戳对得上，确定跑过

### 4.2 很像主线，但更偏试验/探索

- `hl.m`
  - 右海马 `Hippocampus_R.nii`
  - 先算 RDM，再对 RDM 做 PCA
  - 只跑 `kkk = 80`
  - 不保存结果
  - 2025 年改动，像作者后期手动试验脚本

- `hl_headbodytail.m`
  - ROI 是每个被试自己的 `reslice-left-tail.nii`
  - 不保存结果
  - 很早期的局部 ROI 试验

- `only_consider_remembered_items.m`
  - 先检查 remembered item 数量够不够
  - 用 `HSC_r.nii` / `LSC_r.nii`
  - 不保存结果
  - 更像早期筛 remembered trials 的分析

- `stage_specific_pca_connectivity1.m`
  - 不是 searchlight
  - 直接比较两个 ROI 的 PCA 结构相似度
  - 当前脚本不保存结果，只做 t 检验
  - 很可能是 `searchlight_pca_left_80` 主线之前/之后的 ROI 版验证分析

### 4.3 PC 成分占比的试验脚本

- `sametrialnumber_first_2_pc.m`
  - 比较前 2 个主成分解释率之和
  - 不保存结果

- `sametrialnumber_3_pc.m`
  - 比较第 2 到第 4 个主成分解释率之和
  - 当前文件里 `save` 被注释掉

- `sametrialnumber_first_3_pc.m`
  - 当前参数是 `jj=3`, `jjj=5`
  - 实际上比较的是第 3 到第 5 个主成分，不是“前 3 个”
  - 不保存结果

说明：

- 这三份脚本彼此关系非常近，属于一组参数试验。
- 目录里有 `sametrialn_70_80_lhip_first_3_pc.mat`，但当前工作区没有脚本能精确复现它。
- 最可能是 `sametrialnumber_3_pc.m` 的更早版本曾经保存过这个文件，后面又把 `save` 注释掉了。

## 5. 行为分数相关的探索脚本

这一组不是当前目录主线结果的直接生产者，且依赖工作区之外的数据路径。

- `Untitled2.m`
- `Untitled3.m`
- `across_sub_dim.m`

共同特点：

1. 读取行为事件表：
   - `L:\FLXX_1\First_level\get_onset\newdata_to_use`
   - 或 `I:\FLXX1\events`
2. 读取外部 pattern：
   - `L:\FLXX_1\mvpa\GLM_item_allexample\rsa`
3. 按行为分数高低分组
4. 比较高分和低分样本的维度差异

它们没有在当前目录生成稳定输出文件，更像一次性探索分析。

## 6. 重复/历史版本归并

### 一组：基于行为分数分组的脚本

- `Untitled2.m`
- `Untitled3.m`
- `across_sub_dim.m`

区别主要是：

- 行为字段不同：`jiaocha_score` / `jiaocha_score_2`
- 高低分筛选比例不同
- 随机抽样比例和重复次数不同
- 输入路径不同

判断：这三份是同一路思路的不同版本，优先保留 `across_sub_dim.m` 和最新需求所对应的数据路径说明，另外两份视作历史试验。

### 一组：左海马 ROI 维度统计

- `hl_patternbutnotrdm.m`
- `demeantest.m`
- `demeantest2.m`
- `sametrialnumber_maskvoxel.m`
- `Untit3led.m`

共同目标都是比较 HSC/LSC 在左海马的维度差异，区别在于：

- 是否先构 RDM
- 是否做 demean
- 是否剔除差异 voxel
- 阈值区间不同

判断：不是简单重复，而是“同一问题的不同分析策略”。

### 一组：主成分片段分析

- `sametrialnumber_first_2_pc.m`
- `sametrialnumber_3_pc.m`
- `sametrialnumber_first_3_pc.m`

判断：高度相似，属于调参数试验。

### 一组：SPM 二级配对 t 检验模板

- `pairedttest.m`
- `searchlight_pca_left_80/second/pairedttest.m`

判断：模板相同，只是输入图和输出目录不同。

## 7. 我认为你接手时最应该先看的脚本

按优先级建议：

1. `searchlight_RD.m`
2. `my_var_measure.m`
3. `pairedttest.m`
4. `stage1_pca_conn_searchlight.m`
5. `my_measure.m`
6. `searchlight_pca_left_80/second/pairedttest.m`
7. `hl_patternbutnotrdm.m`
8. `demeantest.m`
9. `demeantest2.m`
10. `get_voxel_mask.m`
11. `sametrialnumber_maskvoxel.m`

如果你只想先搞懂“目录里最可能真正发表/汇报过的结果是怎么来的”，先看前 6 个就够了。

## 8. 目前无法完全定位来源的结果

以下结果文件在当前工作区里没有找到一一对应的现成脚本，或当前脚本版本已经改掉了：

- `sheng2022.mat`
- `sametrialn.mat`
- `sametrialn_70_80.mat`
- `sametrialn_80_90.mat`
- `sametrialn_70_80_phpc.mat`
- `sametrialn_80_90_phpc.mat`
- `sametrialn_70_80_vision.mat`
- `sametrialn_70_90_IFG.mat`
- `sametrialn_81_90_lhpc.mat`
- `sametrialn_50_69_lhpc.mat`
- `sametrialn_70_80_lhip_first_3_pc.mat`

这些大概率来自：

1. 当前脚本的旧版本
2. 临时改 ROI/阈值后直接在 MATLAB 命令行里运行再 `save`
3. 已经不在这个目录里的脚本

## 9. 接手建议

### 建议先把“真正的入口”当成下面两套

#### 套路 A

1. `searchlight_RD.m`
2. `pairedttest.m`

#### 套路 B

1. `stage1_pca_conn_searchlight.m`
2. `searchlight_pca_left_80/second/pairedttest.m`

### 再把 ROI 分析当成补充验证

优先看：

- `hl_patternbutnotrdm.m`
- `demeantest.m`
- `demeantest2.m`
- `sametrialnumber_maskvoxel.m`

### 最后再处理历史脚本

- `Untitled2.m`
- `Untitled3.m`
- `across_sub_dim.m`
- `hl_headbodytail.m`
- `only_consider_remembered_items.m`
- `hl.m`

这些脚本先不要急着当主流程，先当“作者曾经试过什么”。

