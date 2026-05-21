# 子刊冲刺收敛版 TODO

最后更新：2026-04-29

这份文档是当前阶段唯一的任务清单。目标不是“把所有能做的分析都立刻做完”，而是：

1. 先修掉会影响解释的硬问题
2. 再补最能增强主故事的证据
3. 最后再做高风险、高成本的扩展分析

相关文档：

- 核心结果口径：[result.md](./result.md)
- 方法与设计说明：[readme\_detail.md](./readme_detail.md)
- 结果逐步解读：[result\_walkthrough.md](./result_walkthrough.md)

***

## 一、当前固定主线

当前不再重构的核心故事：

- 行为层：`yy` 比 `kj` 记得更好，而且正确提取更快
- 表征层：`yy` 在多层 ROI 中都表现出更强的 pre→post similarity 下降
- 机制层：当前最支持 `pair-based differentiation + semantic reweighting`
- 边界层：
  - 现有 between-subject 脑-行为相关为阴性，不能当主支柱
  - `M7_memory` 当前存在度量粒度问题，不能继续并列为主机制模型
  - `Δρ LMM` 只能作为跨 ROI 一致性检查，不能替代单 ROI 机制定位

后续所有新增分析都必须直接服务这条主线，而不是平行堆砌结果。

***

## 二、优先级原则

### S层：冲刺 NN / NC / NHB 必做

这些任务是“从 PNAS 档次升到顶刊子刊档次”的关键闭环，缺一个就很难冲顶刊。
S 层的设计原则：**不是再堆新分析，而是把现有主线补成** **`行为 → 表征 → 机制 → 预测 → 网络 → 全脑`** **的完整闭环**。

### A层：必须先完成

这些任务不做，后面的机制扩展会建立在不稳的地基上。

### B层：强烈建议完成

这些任务最有可能把稿件从“能投”提升到“更有竞争力”。

### C层：保留但后置

这些任务有价值，但高成本、高风险，或者当前不是解释闭环的最短路径。

***

## 二·S、S层：冲刺顶刊必做四件套

### S0. 刺激控制（从 A4 抬到 S 层）

目标：提前封住顶刊审稿最常见的“刺激混淆”质疑，成本低但性价比极高。

- [ ] 整理：
  - 词频
  - 笔画
  - 具象性
  - 熟悉度
  - imageability
- [ ] 输出：
  - `paper_outputs/tables_si/table_stimulus_control.tsv`
- [ ] 如存在不平衡：
  - 将相关变量作为 covariate 重跑核心 LMM sensitivity
- [ ] 与 A4 共用同一组产物，完成后 A4 同步勾选

### S1. 机制 → 行为 / 记忆预测链（新增，核心闭环）

目标：把 Step 5D 的被试层机制分数与 run-7 行为/记忆直接连起来，建立
“被试 differentiation 越强 → 行为/记忆越好”的**个体差异预测闭环**。

- [ ] 新建 [`brain_behavior/mechanism_behavior_prediction.py`](./brain_behavior/mechanism_behavior_prediction.py)
- [ ] 输入（全部复用已有 paper\_outputs 产物）：
  - Step 5D 的 subject-level Δρ：`M3_embedding`、`M8_reverse_pair`（主）
  - `M7_continuous_confidence` 仅作为可选补充项，默认不进入主分析
  - Step 5C 的 subject-level 表征变化分数（可选作为 sanity check）
  - `paper_outputs/qc/behavior_results/refined/*.tsv` 的 run-7 memory / 正确提取效率
- [ ] 分析设计：
  - 样本单位：**被试**（不是 trial；明确区别于 MVPA）
  - 模型：被试间回归或 Spearman（样本量 n≈28 下优先用稳健方法 + permutation）
  - 至少报告：
    - `β / r / 95% CI / permutation p`
    - 对 `yy` 机制分数预测 `yy` 行为 / 记忆
    - 对 `kj` 机制分数预测 `kj` 行为 / 记忆（对照）
    - interaction：机制 × condition 是否存在
- [ ] 与 C8 MVPA 的区别（必须写清楚）：
  - **MVPA**：样本=trial，预测条件标签，回答“ROI 中是否含有可读出的条件信息”
  - **S1**：样本=被试，预测行为/记忆，回答“机制分数是否解释学习效果的个体差异”
  - 二者论点层不同，不冗余
- [ ] 输出：
  - `paper_outputs/tables_main/table_mechanism_behavior_prediction.tsv`
  - `paper_outputs/tables_si/table_mechanism_behavior_prediction_perm.tsv`
  - `paper_outputs/figures_main/fig_mechanism_behavior_prediction.png`
- [ ] 文稿口径：
  - 作为主文机制闭环的一部分，而非 SI
  - 不允许写成“脑结构因果证据”，只能写成“机制分数预测学习效果的个体差异”
  - `M8_reverse_pair` 与 `M2_pair` 属于同一 pair 规则的镜像编码，S1 里只保留 `M8_reverse_pair`，避免重复加权 pair-based 机制
  - `run3/4` 只代表 learning phase 在线表现，不作为 S1 主终点；S1 只预测 run-7 最终学习收益

### S2. 网络层重配置（C3 gPPI 正式产出，从 C 层抬到 S 层）

目标：把网络层从“阴性/未做”抬到“至少有一条正式证据”，这是顶刊对 RSA 类论文几乎必问的一条。

- [x] 代码已完成：[`connectivity_analysis/gPPI_analysis.py`](./connectivity_analysis/gPPI_analysis.py)
- [x] pre/post 主分析口径已固定
- [x] manifest 自动取 seed / target 已实现
- [x] generalized PPI 多条件项 + 可选 deconvolution 已实现
- [ ] 正式运行：
  - seed：`Metaphor > Spatial` 左颞极两个簇（分开跑）
  - targets：`literature + literature_spatial`
  - 汇总：`within_literature` / `within_literature_spatial` / `cross_sets`
- [ ] 输出（建议同时进主文和 SI）：
  - `paper_outputs/tables_main/table_gppi_summary_main.tsv`（至少 yy\_minus\_kj 的 post vs pre）
  - `paper_outputs/tables_si/table_gppi_summary_full.tsv`
  - `paper_outputs/qc/gppi_*`
- [ ] 允许的表述：
  - 作为“伴随 Step 5C 表征重组的网络再配置”的补充证据
- [ ] 禁止的表述：
  - 不能写成“连接驱动表征变化”的因果主张

### S3. Searchlight（B3 最小版，从 B 层抬到 S 层）

目标：补一张“where”大图，让 ROI 级主结论不再只停留在 ROI 层。

- [ ] 主版运行 [`representation_analysis/pair_similarity_searchlight.py`](./representation_analysis/pair_similarity_searchlight.py)
- [ ] 最低要求（最小版）：
  - `yy` / `kj` / `baseline` 的 `post - pre` delta pair similarity searchlight
  - group-level sign-flip permutation（`--permutation-backend auto`）
  - 只做 pair similarity searchlight，**不做 model-based searchlight**
- [ ] 输出：
  - `paper_outputs/tables_main/table_searchlight_peaks.tsv`
  - `paper_outputs/figures_main/fig_searchlight_delta.png`
- [ ] 补充探索版：
  - [`representation_analysis/rd_searchlight.py`](./representation_analysis/rd_searchlight.py) 保留为 supplementary / exploratory 几何描述
  - 只有在主版 pair similarity searchlight 已完成后，才决定是否把 RD searchlight 写入 SI
- [ ] model-based searchlight 仍保留“暂不实现”决策，只有在 S3 最小版完成后才考虑是否升级
- [ ] 时间紧张时只保留 `yy` 的 delta searchlight 亦可作为最小可接受版本

***

## 三、A层：必须先完成

### A1. 修正 M7 的解释风险

目标：不再把当前近似二值的 `M7_memory` 当作主机制模型。

- [x] 在 [`rsa_analysis/build_memory_strength_table.py`](./rsa_analysis/build_memory_strength_table.py) 中新增 `--score-mode`
  - `binary`
  - `confidence`
  - `learning_weighted`
  - `irt`（可先占位，后实现）
- [x] 最低要求：先跑出 `M7_continuous_confidence`
- [x] 在 [`rsa_analysis/model_rdm_comparison.py`](./rsa_analysis/model_rdm_comparison.py) 中固定主解释口径为 `M1/M2/M3/M8`
- [x] `M7` 改为 secondary / sensitivity analysis，但继续保留在模型执行链中
- [ ] 输出：
  - `paper_outputs/tables_si/table_m7_variants_comparison.tsv`
- [x] 文稿要求：
  - 在方法和讨论中明确写出当前 `M7_binary` 的局限
  - 明确“不显著不等于 memory 与 neural geometry 无关”

### A2. 修正 Δρ LMM 的异质 ROI 平均问题

目标：避免把异质 ROI 混成一个 pooled 结果来做过度解释。

- [x] 在 [`rsa_analysis/delta_rho_lmm.py`](./rsa_analysis/delta_rho_lmm.py) 中新增 family split / ROI split 支持
- [ ] 最低要求：
  - [x] 按 `base_contrast` 或等价家族拆开 `main_functional`
  - [x] 至少提供：
    - pooled
    - family-split
  - [ ] leave-one-roi-out sensitivity
- [ ] 输出：
  - [ ] `paper_outputs/tables_si/table_delta_rho_family_split.tsv`
  - [ ] `paper_outputs/tables_si/table_delta_rho_leave_one_roi_out.tsv`
  - [ ] `paper_outputs/tables_si/table_delta_rho_random_slope.tsv`（如实现）
- [x] 文稿要求：
  - `Δρ LMM` 只写“跨 ROI 一致性”，不写成机制定位本体

### A3. 将脑-行为分析升级到 within-subject item level

目标：替代当前阴性的 between-subject 相关，给出更敏感、更直接的“so what”证据。

- [x] 新建 [`brain_behavior/itemlevel_brain_behavior.py`](./brain_behavior/itemlevel_brain_behavior.py)
- [x] 分析结构：
  - 因变量：run-7 `memory`（0/1）
  - 自变量：`item_similarity_post - item_similarity_pre`
  - 模型：trial-level GEE（within-subject item-level decomposition）
- [x] 最低要求：
  - `yy`
  - `kj`
  - 至少主功能 ROI
- [x] 推荐扩展：
  - literature ROI
  - 正确 trial RT 版本
- [x] 输出：
  - `paper_outputs/tables_main/table_itemlevel_coupling.tsv`
  - `paper_outputs/figures_main/fig_itemlevel_coupling.png`

### A4. 刺激控制表（已抬升到 S0，保留此条以便定位）

目标：提前封住“yy / kj 刺激不均衡导致效应”的常见质疑。

> 本条目已整合进 **S0（S 层）**，请直接在 S0 勾选推进；此处仅保留任务上下文，不再重复追踪状态。

- 详见 S0 的执行项与输出要求

### A5. 统一输出目录到 `paper_outputs/`

目标：停止把结果散落在多个老目录中，保证后续图表、SI、主文可追踪。

- [x] 为关键脚本提供统一输出入口或包装层：
  - behavior
  - RSA
  - Model-RSA
  - brain-behavior
  - QC
- [x] 至少保证新增结果统一落到：
  - `paper_outputs/figures_main`
  - `paper_outputs/figures_si`
  - `paper_outputs/tables_main`
  - `paper_outputs/tables_si`
  - `paper_outputs/qc`

***

## 四、B层：强烈建议完成

### B1. Learning dynamics

目标：补上 “when”，但这里的 `when` 不再定义为“学习期所有句子的原始平均 similarity 随窗口变化”，而是定义为：**同一 pair / 同一句子在第一次与第二次暴露之间，表征是否发生了条件特异的重组**。

- [ ] 运行 [`temporal_dynamics/learning_dynamics.py`](./temporal_dynamics/learning_dynamics.py)
- [x] 主分析：**A2. pair-based repeated-exposure discrimination**
  - 设计前提：
    - `Run3` 与 `Run4` 使用**同一批句子**（每句各出现一次），但顺序不同
    - 每个句子稳定对应一个词对，因此可把 `Run3 -> Run4` 视为**第一次暴露 -> 第二次暴露**
  - 主指标：
    - `same_pair_cross_run_similarity`
      - 定义：`sim(run3_sentence_i, run4_sentence_i)`
    - `different_pair_cross_run_similarity`
      - 定义：`mean(sim(run3_sentence_i, run4_sentence_j))`, `j != i`
      - 限制：在**同一 condition 内**比较（`yy` 只和 `yy` 比；`kj` 只和 `kj` 比）
    - `pair_discrimination`
      - 定义：`same_pair_cross_run_similarity - different_pair_cross_run_similarity`
  - 核心检验：
    - 比较 `pair_discrimination_yy` 与 `pair_discrimination_kj`
    - 口径：检验 `yy` 是否在 repeated exposure 后形成了**更强的 pair-specific representational structure**
  - 解释边界：
    - 不把主指标写成 “yy 一定更稳定”
    - 因为当前总故事更接近 `yy` 的**重组 / 分化更强**，所以 `yy` 的 raw same-pair similarity 可能**不高于** `kj`
    - 因此主分析必须看 `same - different` 的 discrimination，而不是只看 raw similarity
- [x] 补充分析：**A1. same-sentence cross-run stability**
  - 补充指标：
    - `same_sentence_cross_run_similarity = sim(run3_sentence_i, run4_sentence_i)`
  - 用途：
    - 作为 repeated exposure 的稳定性描述指标
    - 可报告 `yy` 与 `kj` 的均值差异，但**不作为 B1 主指标**
  - 解释边界：
    - 若出现 `kj > yy`，并不与主线矛盾；这更可能说明 `kj` 更偏向重复稳定，而 `yy` 更偏向重组
- [x] 不采用的旧方案（明确废弃）
  - 不再用：`similarity ~ window_index`
  - 不再把 learning dynamics 写成“学习期所有不同句子的平均原始 similarity 轨迹”
  - 原因：学习期 trial 是**不同句子**，raw similarity 的窗口变化会被刺激内容差异严重混淆
- [ ] 结果输出（主指标 + 补充指标都要保留）：
  - `paper_outputs/tables_main/table_learning_repeated_exposure.tsv`
  - `paper_outputs/tables_si/table_learning_repeated_exposure_subject.tsv`
  - `paper_outputs/tables_si/table_learning_repeated_exposure_family.tsv`
  - `paper_outputs/figures_main/fig_learning_dynamics.png`
  - `paper_outputs/qc/learning_repeated_exposure_itemwise.tsv`
  - `paper_outputs/qc/learning_repeated_exposure_group.tsv`
  - `paper_outputs/qc/learning_dynamics_meta.json`
- [x] ROI 范围固定：
  - **主分析 ROI**：`literature`
  - **补充 ROI**：`main_functional`，但必须按 family split 分开报告：
    - `Metaphor > Spatial`
    - `Spatial > Metaphor`
  - **不允许**：把 `main_functional` 直接 pooled union 后当作 B1 主结果
  - `literature_spatial / atlas_robustness` 暂不作为 B1 第一轮主分析
- [ ] 输出：
  - 主文优先写 A2 主指标
  - A1 作为补充指标一起保留，必要时进入 SI 或结果补充段

### B2. Representational connectivity

目标：优先用更贴近主故事的方法补网络层证据，而不是一上来重押 gPPI / effective connectivity。

- [ ] 运行 [`representation_analysis/representational_connectivity.py`](./representation_analysis/representational_connectivity.py)
- [x] 方法定义固定：
  - 不是传统功能连接，而是 **all-by-all ROI pair representational connectivity**
  - 具体实现：比较两个 ROI 在同一批 item 上的 `post - pre` neural delta 轮廓是否一致
  - 主指标：`repr_connectivity_rho / repr_connectivity_z`
  - **主分析 ROI**：`main_functional`
    - 但必须做 **family-aware** 汇总：
      - `within_Metaphor_gt_Spatial`
      - `within_Spatial_gt_Metaphor`
      - `cross_main_functional_family`
    - 不允许再把 `main_functional` 直接 pooled union 后只给一个平均 connectivity 结论
  - **确认性分析 ROI**：`literature + literature_spatial`
    - 汇总：
      - `within_literature`
      - `within_literature_spatial`
      - `cross_independent_sets`
  - 不再使用“手工只挑少数 ROI 对”的旧版方案
- [ ] 输出：
  - `paper_outputs/tables_main/table_repr_connectivity.tsv`
  - `paper_outputs/tables_si/table_repr_connectivity_subject.tsv`
  - `paper_outputs/tables_si/table_repr_connectivity_edges.tsv`
  - `paper_outputs/figures_main/fig_repr_connectivity.png`

### B3. Searchlight RSA（已抬升到 S3，保留此条以便定位）

目标：补 “where”。

> 本条目已整合进 **S3（S 层）最小版**，请直接在 S3 勾选推进；此处仅保留原有优先级判断作为历史上下文。

- [x] 优先级判断（保留为决策记录）：
  - 当前正式 S3 主版改为 **pair similarity searchlight**，用于补“where”
  - **RD searchlight** 降为 supplementary / exploratory
  - **model-based Searchlight RSA 暂不实现**
  - 原因：
    - 现阶段主线更缺的是一个与 Step 5C 主结果直接同构的全脑空间定位补充
    - pair similarity searchlight 更直接回答“哪里出现了 pair-level representational reorganization”
    - RD searchlight 回答的是局部表征维度变化，适合作为几何补充，不适合和主线并列主打
    - model-based searchlight 工程量更大，且会显著增加计算成本
    - 应放到 S3 最小版结果出来之后再决定是否升级

### B4. 稳健性三件套

目标：用最少但高价值的稳健性分析支撑主文。

- [x] 新建 [`rsa_analysis/robustness_suite.py`](./rsa_analysis/robustness_suite.py)
- [ ] Bayes Factor for primary endpoint
- [ ] LOSO
- [ ] bootstrap CI
- [ ] split-half replication
- [ ] 输出：
  - `paper_outputs/tables_si/table_bayes_factors.tsv`
  - `paper_outputs/tables_si/table_loso.tsv`
  - `paper_outputs/tables_si/table_bootstrap_ci.tsv`
  - `paper_outputs/tables_si/table_splithalf.tsv`

### B5. Noise ceiling

- [x] 新建 [`rsa_analysis/noise_ceiling.py`](./rsa_analysis/noise_ceiling.py)
- [x] 方法定义固定：
  - Step 5C 默认覆盖四层 ROI：
    - `main_functional`
    - `literature`
    - `literature_spatial`
    - `atlas_robustness`
  - Model-RSA 基于 audit 导出的共享 item-pair 特征对齐后再做跨被试 ceiling
- [ ] 对 Step 5C / 关键 Model-RSA 结果实际运行 noise ceiling
- [ ] 输出：
  - `paper_outputs/tables_si/table_noise_ceiling.tsv`
- [ ] 仅在图上作为支持信息，不替代主统计

***

## 五、C层：保留但后置

这些任务都保留，不删，但默认不进入最近两周的主执行线。

### C7. LI / RD / GPS（C层内优先）

- [ ] 目标：用“几何指标/侧化”补充 Step 5C 的解释边界（更像压缩？分化？重组？）
- [ ] 推荐优先级（同为 C7 内部）：
  - [ ] RD / GPS（更贴主线、解释更直接）
  - [ ] LI（仅当需要回应侧化相关质疑时）
- [ ] 输出（默认放 SI / 扩展模块）：
  - `paper_outputs/tables_si/table_rd.tsv`（或 `rd_group_summary.tsv` 的 paper 版）
  - `paper_outputs/tables_si/table_gps.tsv`（或 `gps_group_summary.tsv` 的 paper 版）
  - `paper_outputs/tables_si/table_lateralization.tsv`（如做）

### C3. gPPI（已抬升到 S2，保留此条以便定位）

> 本条目已整合进 **S2（S 层）**，请直接在 S2 勾选推进；此处仅保留历史实现决策作为上下文。

- [x] 新建/重构 [`connectivity_analysis/gPPI_analysis.py`](./connectivity_analysis/gPPI_analysis.py)
- [x] 目标固定：补网络层机制证据（**pre/post：run1/2 vs run5/6** 的连接调制变化），作为 B2（representational connectivity）的互补而非替代
- [x] 已实现策略（避免返工）：
  - [x] 主分析：pre/post 口径
    - seed 定位可来自 learning `run3/4` 的 GLM（where）
    - gPPI 本体在 `run1/2/5/6` 上检验 `condition × time` 风格的连接调制变化（how/network）
  - [x] 支持 pilot：少 seed + 受控 target ROI 集（SI / confirmatory）
  - [x] 支持 manifest 自动取 seed / target ROI
  - [x] 支持 generalized PPI 多条件项 + 可选 deconvolution
  - [ ] 再决定是否扩展到多 seed / 全 target（强多重比较负担）

### C8. MVPA / Decoding（可选，按需触发）

- [ ] 目标：回答“ROI 是否能读出 yy vs kj（可分性/readout）”，避免被质疑“similarity 下降不等于更可分”
- [x] 与 **S1 机制→行为预测** 的区别（必须在代码和方法中写清）：
  - **C8 MVPA**：样本=trial，预测条件标签，回答“ROI 中是否含有可读出的条件信息”
  - **S1**：样本=被试，预测行为/记忆，回答“机制分数是否解释学习效果的个体差异”
  - 二者论点层不同，不冗余；S1 优先级高于 C8
- [ ] 触发条件（满足其一再做）：
  - [ ] 审稿/组会明确要求“分类证据”
  - [ ] 你希望把机制表述更靠近“可读出性变化”（而非仅几何变化）
- [ ] 约束（必须写清楚，避免泄漏/双重使用数据）：
  - [ ] 优先用 `literature` ROI（独立性更强）；`main_functional` 仅 exploratory 或明确声明来源
  - [ ] 严格 cross-validation（建议按 run/phase 分层），并做 label permutation / sign-flip
- [ ] 输出（默认 SI）：
  - `paper_outputs/tables_si/table_mvpa_decoding.tsv`
  - `paper_outputs/figures_si/fig_mvpa_decoding.png`

### C1. M9 relational + variance partitioning（A 层扩展项，顶刊冲刺备选）

> 本条被评估为“A 级加分、S 级次优”：建议在 S0–S3 四件套完成后、或其中任何一项卡住时，作为首选的机制深化任务立即推进。

- [ ] 关系类别 RDM
- [ ] source / target domain RDM
- [ ] variance partitioning
- [ ] 输出：
  - `paper_outputs/tables_main/table_variance_partitioning.tsv`
  - `paper_outputs/figures_main/fig_variance_partitioning.png`

说明：这是很有潜力的机制升级项，但定义和解释风险都高，不再强绑 A 层，但冲顶刊时价值远高于 C2/C4/C5/C6。

### C2. Hippocampal subfield / MTL 扩展（第四优先）

- [ ] subfield mask
- [ ] CA1 / CA3-DG / subiculum / anterior-posterior hippocampus
- [ ] 输出：
  - `paper_outputs/tables_main/table_hippocampal_subfield.tsv`
  - `paper_outputs/figures_main/fig_hippocampal_subfield.png`

### C5. CPM（第五优先）

- [ ] 保留
- [ ] 默认后置到较高投稿目标再做（n≈28 情况下严格 CV 很容易不稳定）

### C4. Effective connectivity（第六优先）

- [ ] 保留
- [ ] 默认 SI / exploratory

### C6. Career of Metaphor（第七优先）

- [ ] 保留
- [ ] 作为理论扩展，不阻塞当前主线

***

## 六、QC 与交付物

这些不是“可做可不做”，而是投稿前必须齐。

### QC

- [ ] `paper_outputs/qc/lss_qc_summary.tsv`
- [ ] `paper_outputs/qc/pattern_qc_summary.tsv`
- [ ] `paper_outputs/qc/motion_scrubbing_summary.tsv`
- [ ] `paper_outputs/qc/fake_handling_note.md`

### 交付管理

- [ ] `paper_outputs/analysis_plan.tsv`
- [ ] `paper_outputs/figure_stats_map.tsv`
- [ ] 主文图与 SI 图逐一绑定统计来源
- [ ] `result.md` 同步更新（review 提醒项）
  - [ ] B5 补写 `atlas_robustness` 的 Step 5C noise ceiling 汇报，保持与四层 ROI 默认覆盖一致
  - [ ] 第 12 节主文图路径统一到 `paper_outputs/figures_main` / `paper_outputs/figures_si`
  - [ ] Figure 1/2/4/5 的图片嵌入路径不再继续引用旧 `figures_main_story`

***

## 七、按投稿档次的停止规则

### 到达“可投 PNAS / Communications Biology / Cerebral Cortex”门槛

满足以下即可进入写稿：

- A 层全部完成（A4 以 S0 产出为准）
- B 层至少完成 2 项
  - 优先 `Learning dynamics`
  - 优先 `Representational connectivity` 或 `Searchlight`
- **S 层在该档次允许缓做**
  - S0 / S1 至少有一条进入 SI 级别的结果即可
  - S2 gPPI / S3 Searchlight 可暂缓到修稿阶段

### 到达“可冲 NN / NC / NHB”门槛

在上一层基础上，**S 层四件套必须全部有正式产出**：

- [ ] **S0 刺激控制**：`table_stimulus_control.tsv` 完成，若存在不平衡已完成 covariate sensitivity
- [ ] **S1 机制 → 行为 / 记忆预测链**：至少在主机制模型（`M3_embedding` / `M8_reverse_pair`）上报告 subject-level 预测 + permutation；`M7_continuous_confidence` 仅作可选补充
- [ ] **S2 gPPI**：pre/post 主分析至少跑出一条方向性证据（post vs pre 的 `yy_minus_kj`）
- [ ] **S3 Searchlight 最小版**：至少 `yy` 的 `post - pre` delta pair similarity group-level searchlight 完成

并满足：

- `Learning dynamics` 有主文级结论
- 一条稳定的网络层证据（S2 即可）
- 一条比现有 between-subject 更强的脑-行为证据（S1 即可）

### 冲更高档前再考虑

- M9 relational（C1）
- hippocampal subfield（C2）
- CPM（C5）
- Career of Metaphor（C6）
- C8 MVPA（仅当审稿明确要求，或 S1 不足以支撑“可读出性”表述时）

***

## 八、最近两周的现实推进顺序

以 **S 层四件套 + 关键 A/B 补洞** 为骨架，而非简单按 A→B→C 顺序。

### 第 1 阶段（本周上半）：S0 + S1 代码闭环

- [ ] **S0 刺激控制**：整理 5 个控制变量，跑出 `table_stimulus_control.tsv`；如不平衡，立刻加 covariate sensitivity
- [ ] **S1 代码落地**：`mechanism_behavior_prediction.py` 已完成；下一步跑默认主分析（`M3_embedding + M8_reverse_pair`），必要时再加 `--include-m7`
- [ ] A1 `M7` 变体对比表落盘
- [ ] A2 `Δρ LMM` family-split / leave-one-roi-out 产出 SI 表

### 第 2 阶段（本周下半）：S2 gPPI 正式产出

- [ ] **S2 gPPI**：pre/post 主分析在左颞极两 seed × `literature + literature_spatial` 上正式跑出 `post vs pre` 的 `yy_minus_kj`
- [ ] `table_gppi_summary_main.tsv` + `table_gppi_summary_full.tsv` 落盘（可由 `export_gppi_summary.py` 在现有 gPPI 运行结束后独立导出）
- [ ] 配套 `fig_gppi_network.png` 进 `figures_main`

### 第 3 阶段（下周上半）：S3 Searchlight 最小版 + B 层补洞

- [ ] **S3 Searchlight 最小版**：先跑 `pair_similarity_searchlight.py`，至少完成 `yy` / `kj` / `baseline` 的 `post - pre` delta pair similarity searchlight + sign-flip permutation
- [ ] **RD searchlight**：仅作为 supplementary / exploratory 几何描述，主版完成后再决定是否保留进 SI
- [ ] B1 learning dynamics 主文图定稿
- [ ] B2 representational connectivity 主文图定稿

### 第 4 阶段（下周下半）：稳健性 + ceiling + 文稿同步

- [ ] B4 稳健性三件套（Bayes / LOSO / bootstrap / split-half）
- [ ] B5 noise ceiling 全量跑完（含 `atlas_robustness`）并落 `table_noise_ceiling.tsv`
- [ ] `result.md` 两处同步项补齐（B5 atlas 摘要 + 第 12 节主文图路径）
- [ ] C 层任务（C1 M9 / C7 RD+GPS / 其他）按剩余时间和结果收益决定是否启动

***

## 九、每周检查模板

### 本周主目标

- [ ] 只写 1 项

### 本周支持任务

- [ ] 最多 2-3 项

### 本周输出

- [ ] 表
- [ ] 图
- [ ] 文稿口径更新

### 本周决策

- [ ] 继续当前路径
- [ ] 降级某个任务
- [ ] 推迟某个任务

***

## 十、已完成工作存档

以下内容已完成，不再重复立项，只在后续文稿中整合：

- [x] 主线口径固定
- [x] refined behavior
- [x] 主 ROI / literature / literature\_spatial / atlas\_robustness 的 Step 5C
- [x] 主 ROI / literature 的 Model-RSA 与 `Δρ LMM`
- [x] trial / voxel robustness
- [x] QC / reliability 第一轮
- [x] 主文关键图第一轮

这些是当前主线地基，不是当前待办。

***

## 十一、执行边界

继续允许的表述：

- representational differentiation / reorganization
- pair-based differentiation
- semantic reweighting

继续禁止的表述：

- “学习后同 pair 更像”
- “M7 不显著说明记忆强度与神经几何无关”
- “Δρ LMM 直接证明主导机制”
- “已经完全排除所有质量解释”

***

## 十二、如果进度受阻，优先牺牲什么

按顺序后撤：

1. 先牺牲 `effective connectivity`
2. 再牺牲 `CPM`
3. 再牺牲 `Career of Metaphor`
4. 再牺牲 `LI / RD / GPS`
5. 再牺牲 `C8 MVPA`（S1 已经覆盖个体差异闭环，MVPA 不是必需）
6. 最后才考虑把 **S3 Searchlight 从最小版再缩小**（例如只保留 `yy` 一个 contrast）

不应牺牲：

- `M7` 修复
- `Δρ LMM` 异质性处理
- item-level 脑-行为
- **S0 刺激控制**
- **S1 机制 → 行为/记忆预测链**
- **S2 gPPI 至少一条方向性 pre/post 结果**

因为这些直接关系到当前主文解释是否站得住，以及是否能冲 NN/NC/NHB。
