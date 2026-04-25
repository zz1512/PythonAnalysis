# Step 5C/5D 当前分析思路与下一步 TODO

## 1. 当前结果的核心解释框架

### 1.1 目前已经稳定得到的结果

- 行为层：`yy` 条件记忆表现优于 `kj`。
- RSA 层：在 `main_functional` 和 `literature` 两层 ROI 中，`yy` 的 pair similarity 都表现出更明显的 `pre -> post` 下降。
- baseline 控制：`baseline(jx)` 没有和 `yy` 一样出现系统性下降，多数 ROI 中稳定或轻度上升。
- `kj`：目前在已有主 ROI 和论文 ROI 中，变化较弱，且缺少跨 ROI 的稳定显著性。

### 1.2 当前最合理的结果故事

- 这组结果**不等于矛盾**。
- 行为上的“记得更好”不要求“pair members 在神经上更相似”。
- 更贴切的解释是：`yy` 学习成功后，并不是简单地把两个词的表征“拉近”，而是让它们进入一个**更精细、更可区分、干扰更小**的关系结构。
- 因此，当前主线更接近：
  - `representational differentiation`
  - `learning-specific reorganization`
  - `reduced interference / finer-grained relational coding`

### 1.3 为什么这在理论上说得通

- 若学习目标是建立更稳定、可检索、少混淆的关系，系统可能会把相关项目的表征拉开，而不是机械融合。
- `yy` 可能依赖更高阶的语义重组：学习后形成的是“更结构化的关系编码”，而不是“两词越来越像”。
- baseline 没有同步下降，使得“纯重测/熟悉化/噪声”解释变弱。

## 2. 可直接支持当前解释的文献线索

### 2.1 支持“更好记忆可以伴随神经分化”的论文

- Favila, Chanales, & Kuhl (2016, *Nature Communications*)
  - 学习可使相似事件的表征拉开，帮助降低干扰。
  - 链接: <https://www.nature.com/articles/ncomms11066>
- Hulbert & Norman (2015, *Cerebral Cortex*)
  - neural differentiation 与更好的 competing-memory recall 相关。
  - 链接: <https://academic.oup.com/cercor/article/25/10/3994/394658>
- Schlichting, Mumford, & Preston (2015, *Nature Communications*)
  - 学习后的表征变化既可能表现为 integration，也可能表现为 separation，方向取决于任务需求。
  - 链接: <https://www.nature.com/articles/ncomms8151>
- Sun et al. (2026, *Communications Biology*)
  - 海马环路中 differentiation 与 integration 可以共存，不同学习内容可能偏向不同路径。
  - 本地摘要位置: [隐喻与创造力相关论文整理.md](E:/PythonAnalysis/metaphoric/final_version/result_pdf/隐喻与创造力相关论文整理.md:178)

### 2.2 一个需要注意但不冲突的对照文献

- Xue et al. (2010, *Science*)
  - 更好的记忆与更高的 repetition-related pattern similarity 有关。
  - 但它主要讨论的是**同一项目 across repetitions 的稳定性**，不是这里的“同一 pair 内两个不同词是否更像”。
  - 因此它和当前结果可以并存：item-level stability 增强，同时 between-item pair similarity 下降。

## 3. 当前论文写作上的推荐口径

- 不要把当前结果写成“学习失败，所以没收敛”。
- 更推荐的表述：
  - `yy` 行为记忆更好，但这种学习收益并未表现为 pair members 的简单神经收敛；
  - 相反，它表现为更明显的 post-learning neural differentiation；
  - 该模式更符合 learning-specific representational reorganization，可能反映更精细的关系编码和更低的干扰。

### 3.1 暂时不要过度下结论

- 目前可以说“更支持 differentiation / reorganization”。
- 暂时不要直接下结论说“已经证明 hippocampal-style pattern separation”。
- 原因：
  - 目前主要结果 ROI 多在语义/视觉-顶叶/内侧颞叶网络，不完全等于经典海马分离框架；
  - 更稳妥的写法是“reduced similarity / differentiation consistent with reduced interference or finer-grained relational coding”。

## 4. 下一步重点问题：空间 ROI 会不会对 `kj` 显示更强前后分化？

### 4.1 为什么这一步值得做

- 现在的主 ROI / 论文 ROI 结果说明：`yy` 的分化是稳定的，但 `kj` 不稳定。
- 这有两种可能：
  1. `kj` 本来就不会产生和 `yy` 同等级别的表征重组。
  2. 现在使用的 ROI 更偏“隐喻/语义主线”，还没有充分击中真正的**空间表征网络**。
- 因此，单独建立一套“空间相关 literature ROI”，再测试 `kj` 在这些 ROI 中是否出现显著 `pre -> post` similarity 下降，是非常必要的控制分析。

### 4.2 这个分析想回答什么

- 不是重复问“`yy` 有没有效应”。
- 而是问：
  - 在经典空间/场景/导航网络中，`kj` 是否显示出更明显的学习后分化？
  - 如果显示，则说明当前 `kj` 的弱效应可能是 ROI 选取问题。
  - 如果仍不显示，则更支持“本实验中隐喻学习比空间学习更强烈地改变表征几何”这一主结论。

## 5. 空间相关 ROI 的候选清单

### 5.1 一级优先：场景/空间网络核心 ROI

- 双侧 posterior parahippocampal gyrus / PPA
  - 理由：场景、地点、环境布局表征核心区域。
- retrosplenial cortex / medial parietal cortex
  - 实操上可先用 `Precuneus/Retrosplenial` 邻近 atlas ROI 近似。
  - 理由：场景-导航转换、环境定位、空间参照。
- occipital place area / lateral occipital superior division
  - 理由：局部场景边界、可通行空间、视觉场景结构。

### 5.2 二级优先：与空间意象/场景编码相关的扩展 ROI

- bilateral precuneus
  - 本地文献摘要和当前结果都已提示它与空间意象/情境表征相关。
- posterior fusiform / scene-sensitive ventral visual cortex
  - 可用于检验视觉场景表征是否参与 `kj` 学习。
- hippocampus / posterior hippocampal formation
  - 如果 atlas 支持且 mask 稳定，可作为补充探索 ROI。

### 5.3 结合当前库，最容易先落地的一版

- 可以优先从你现有 ROI 库或 atlas 中抽这些 ROI 做一版 pilot：
  - `atlas_R_parahippocampal`
  - `atlas_L_precuneus`
  - `atlas_R_precuneus`
  - `atlas_L_fusiform`
  - 当前 `main_functional` 里的
    - posterior fusiform
    - bilateral precuneus
    - posterior parahippocampal
    - lateral occipital
- 这一版的价值是：
  - 不用先大改 pipeline；
  - 可以快速判断 `kj` 是否在“更空间导向”的 ROI 中开始出现稳定前后变化。

## 6. 空间 ROI 文献整理时优先关注的论文方向

### 6.1 已在本地摘要中出现、可直接利用的线索

- Beaty et al. (2017)
  - 文档里提到楔前叶、右海马旁回、右 IPS 等与隐喻生成相关，但这些区域本身也可作为空间/意象相关节点的文献入口。
  - 位置: [隐喻与创造力相关论文整理.md](E:/PythonAnalysis/metaphoric/final_version/result_pdf/隐喻与创造力相关论文整理.md:40)
- Sun et al. (2026)
  - 提供内侧颞叶与表征重组的概念支持，尤其适合解释后海马旁回相关结果。
  - 位置: [隐喻与创造力相关论文整理.md](E:/PythonAnalysis/metaphoric/final_version/result_pdf/隐喻与创造力相关论文整理.md:206)

### 6.2 还需要补充的经典空间 ROI 文献

- PPA / parahippocampal place area
- RSC / retrosplenial cortex
- OPA / occipital place area
- precuneus 与 spatial imagery / navigation
- hippocampus 与 relational spatial learning

## 7. 建议的分析设计

### 7.1 新建一个单独的 ROI 集

- 名称建议：`literature_spatial`
- 不要把它和当前 `literature` 混在一起。
- 原因：
  - 当前 `literature` 是“隐喻/语义主线 ROI”；
  - 新分析要回答的是“空间网络是否更支持 `kj` 的学习后分化”，逻辑上应独立成套。

### 7.2 最重要的统计问题

- 先跑三条件版本：
  - `similarity ~ condition * time + (1|subject) + (1|item)`
- 重点看两类效应：
  - `C(condition)[T.Spatial]:C(time)[T.Pre]`
  - `C(condition)[T.Metaphor]:C(time)[T.Pre]`
- 解释时始终以 `baseline` 为参照，不要只看 `yy vs kj`。

### 7.3 预期会出现的三种结果

- 结果 A：空间 ROI 中 `kj` 显著下降，且强于 baseline
  - 解释：`kj` 的学习效应存在，但需要在更合适的空间网络 ROI 中才能看见。
- 结果 B：空间 ROI 中 `kj` 仍不显著，而 `yy` 继续显著
  - 解释：隐喻学习确实比空间学习更强烈地改变表征几何，这是更强的主结论。
- 结果 C：`kj` 和 `yy` 在不同网络中都显著，但落点不同
  - 解释：两类学习都能重塑表征，但依赖不同神经系统，形成“语义网络 vs 空间网络”的双重 dissociation。

## 8. 操作层 TODO

本节改为“按优先级排序的可执行清单”，避免 TODO 混在长叙事里难以执行。

### P0（核心阻断项：不做会影响主结论可信度/机制分析无法落地）

- [ ] 记录并写清“反驳头动/SNR/测量变差”的最小证据包（用于主文或 SI）：
  - 提取每位被试 pre/post 的 motion/QC 指标（例如 mean FD、DVARS 或 motion_outlier 比例、scrub 后保留比例）。
  - 检验 `Δsimilarity(yy)` 是否被这些 QC 指标解释（相关/回归控制）；并同时报告 `kj` 与 `baseline` 的对应结果作参照。
  - 做至少一个 reliability 检验：pre 用 run1 vs run2、post 用 run5 vs run6 的 split-half / run-wise 稳定性，排除“post 更不可靠导致 yy similarity 下降”的替代解释。
- [ ] 解决“真实词条映射缺失导致 Model-RSA 的 word_label 当前不可用”的核心问题：
  - 找到真实词条对应的文件/表（word_id -> real_word 或 unique_label -> real_word）。
  - 统一落地到一个可追溯的输入（推荐：更新/新增 `stimuli_template.csv` 列，或新增一张 `stimuli_word_mapping.tsv` 并在文档里写清优先级）。
  - 更新文档：明确当前 `word_label` 字段到底是 unique_label 还是真实词条，避免 M3/M7 的概念错位。
  - 跑通 Model-RSA 的最小链路：`build_stimulus_embeddings.py`（M3）与 `build_memory_strength_table.py`（M7）都基于“真实词条”一致取值。
- [x] baseline 口径锁死为“模板固定伪配对（pair_id 两行成对）”，并保持文档与实现一致（`readme_detail.md` 已更新）。

### P1（主线增强项：提升顶刊抗审稿能力/封堵替代解释）

- [ ] 把当前主故事写作口径固定为：行为优势 + `yy` pre→post similarity 更明显下降并不矛盾，主解释为 differentiation / reorganization（避免写成“学习失败所以没收敛”）。
- [ ] 做 trial 数/体素数稳健性（建议放 SI，但最好能一键复现）：
  - 对 yy/kj/baseline 做等量重采样（或在分析中控制每条件 trial 数差异），验证主效应方向稳定。
  - 检查 `n_voxels` 在 pre/post 的变化是否系统偏向某条件/某 ROI；必要时作为敏感性分析报告。
- [ ] 单独整理“空间相关 ROI”文献清单（至少覆盖 PPA、RSC/precuneus、OPA、hippocampus），用于新 ROI 集的可辩护性。
- [ ] 新建 `literature_spatial` ROI 集（独立成套，不混入当前 `literature`），用于检验 `kj` 是否在经典空间网络中出现更明显的学习后变化。
- [ ] 先用现有 atlas/main 的 spatial candidate ROI 做 pilot（快速判断 `kj` 是否在更空间导向 ROI 中开始出现稳定 pre/post 变化），再决定是否扩展完整 literature_spatial。
- [ ] 对新 ROI 集运行 `run_rsa_optimized.py` + `rsa_lmm.py`，重点看 `Spatial × Pre` 是否显著；把结果按你在本文件 7.3 的 A/B/C 三种情形写成可直接放论文的结论句模板。

### P2（扩展/加分项：主线稳定后再做）

- [ ] 学习阶段 dynamics：窗口化 RD/GPS（必要时加 MVPA），回答“分化增强何时形成”（更适合冲 NN/Neuron）。
- [ ] gPPI / effective connectivity / mediation：只在主线稳定且理论路径清晰后做，作为机制补充（明确探索性声明与多重比较策略）。

## 9. 我下一步最推荐先做什么

- 第一优先：先确定一版最小可跑的“空间 ROI 候选集”。
- 推荐从下面 4 个开始做 pilot：
- 推荐从下面 4 个开始做 pilot：
  - posterior parahippocampal
  - bilateral precuneus
  - lateral occipital / OPA 对应区
  - posterior fusiform
- 如果 pilot 已经看到 `kj` 明显下降，再去扩展更完整的 literature spatial ROI 集。
- 如果 pilot 没看到，再决定是否需要加入 hippocampus / retrosplenial 等更专门的 ROI。

## 10. 执行顺序（建议）

1. 先做 P0 的 motion/QC + reliability：把“测量变差”的替代解释尽早封堵，否则后续所有机制解释都会被质疑。
2. 再补齐真实词条映射：让 Model-RSA 的 M3/M7 具备语义有效性与可复现性。
3. 然后推进 P1 的 literature_spatial：决定 `kj` 的弱效应是“ROI没击中”还是“机制上确实更弱”。
