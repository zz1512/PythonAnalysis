# 研究结果逐步汇报稿：每个分析步骤的方法与结论

> 本文件是面向导师汇报的 **Result 分步解读版**，与 [`result.md`](./result.md) 逐节对应。
>
> 每个步骤按 **"做了什么 → 怎么做（含技术细节）→ 关键数字 → 一句话结论"** 四段组织；技术细节（统计方法、多重比较、相似性算法、RDM 生成等）**加粗**显示，方便汇报中被问到时直接回答。
>
> 配套：
> - 完整数据与表格：[`result.md`](./result.md)
> - 下一步计划：[`todo.md`](./todo.md)
> - 方法细节：[`readme_detail.md`](./readme_detail.md)

---

## 汇报主线（30 秒开场）

> 老师，我们用前后测 × 三条件（隐喻 yy / 空间 kj / 基线 jx）× 多层 ROI 的 RSA + Model-RSA 框架，发现了一个反直觉但非常干净的结果：隐喻学习之后，配对词的神经相似性不是像我们最初预想的那样"更接近"，而是"更分开"——也就是 **representational differentiation**；这个分化在四套 ROI 定义下都稳定重复，baseline 封堵了重测解释，机制上对应 pair-based differentiation 加 semantic reweighting。目前唯一还没跑稳的是脑-行为 between-subject 的个体差异联系，这是下一步需要攻破的关键。

---

## 预热：共用的技术栈说明（被问到基础时先答这一页）

- **预处理流程**：**fMRIPrep 标准流程**（motion correction + slice timing + coregistration + MNI152NLin2009cAsym 空间标准化 + 6 mm FWHM 高斯平滑），输出保留 confound regressors
- **运动伪影控制**：**FD（Framewise Displacement）> 0.5 mm 或 DVARS > 1.5** 的 volume 做 scrubbing；nuisance 包含 6 个运动参数 + aCompCor top 6 + WM/CSF 信号 + cosine drift
- **单 trial beta 估计方法**：**LSS（Least Squares Separate, Mumford 2012 方法）**——每个 trial 单独建一个 GLM，target trial 为单独回归项，其余 trial 合并为一个总回归项，**HRF 卷积用 SPM canonical HRF**
- **trial 平衡性**：pre/post × yy/kj/baseline，每条件每被试都是 **yy = 70, kj = 70, baseline = 40** trial（逐被试 exact_match）
- **代码框架**：**nilearn + nibabel + statsmodels + pingouin + scipy + numpy + pandas**；LMM 用 `statsmodels.regression.mixed_linear_model.MixedLM`

---

## 一、行为结果（对应 result.md §1）

### 1.1 做了什么
回答："行为层面，隐喻学习是不是真的比空间学习学得更好？"

### 1.2 怎么做（技术细节）
- **数据**：27 名被试完成 run-7 最终记忆测验；每个被试每条件约 70 个 trial
- **主终点**：`memory`（**二值 0/1**，对应被试在 run-7 是否正确回忆出配对词）
- **补充终点**：`log_rt_correct`（**自然对数变换**后的反应时，**仅在答对 trial 上计算**，用于消除反应时右偏）
- **统计模型类型**：**广义线性混合效应模型（GLMM）**
  - **memory**：`memory ~ condition + (1|subject)`，**binomial family + logit link**
  - **log_rt_correct**：`log_rt_correct ~ condition + (1|subject)`，**Gaussian family**（因已 log 变换近似正态）
- **推断方法**：**Wald z 检验**（GLMM 默认），不是 t 检验
- **置信区间**：Wald CI（z × SE）
- **多重比较校正**：**本节无需校正**（只有一个 condition 主效应，单一假设）

### 1.3 关键数字
- **记忆正确率**：YY = 0.680，KJ = 0.588
  - `b = 0.092, SE = 0.014, z = 6.43, p < .001, 95% CI [0.064, 0.120]`
- **正确反应时**：YY = 1070 ms，KJ = 1100 ms
  - `b = -0.026, SE = 0.010, z = -2.61, p = .009, 95% CI [-0.046, -0.007]`

### 1.4 一句话结论
> 隐喻学习在组水平上同时带来了**更高的正确率**和**更高效的正确提取**——不是单维度优势，而是"记得更准 + 提取更快"的双重优势。

---

## 二、最小脑-行为相关（对应 result.md §1.1）

### 2.1 做了什么
回答："行为优势和神经分化，在个体差异层面是不是耦合的？"

### 2.2 怎么做（技术细节）
- **行为指标**：`behavior_primary_diff = accuracy_YY - accuracy_KJ`（被试层面差值）
- **神经指标**：`main_functional_primary_diff = (pre_yy - post_yy) - (pre_kj - post_kj)`（正值 = yy 比 kj 分化更强）
- **统计方法**：
  - **Pearson r**（线性相关）
  - **Spearman ρ**（秩相关，防极端值驱动）
  - **双侧检验 + Fisher z-transform 置信区间**
- **样本量**：**N = 27**（剔除 1 名记忆表缺失的被试）
- **多重比较**：**未做 FDR**——因为我们**严格克制**，主分析只预注册一个 ROI layer + 一个神经指标 + 一个行为指标的相关

### 2.3 关键数字（都不显著）
- 主功能 ROI：`Pearson r = -0.163, p = .417`；`Spearman ρ = -0.144, p = .474`
- yy 相对 baseline 差值版本：`r = -0.200, p = .316`
- 文献 ROI：`r = -0.252, p = .205`
- 对正确反应时差值：`r = 0.121, p = .547`

### 2.4 一句话结论
> 组水平 yy 学得更好、yy 分化更强**都成立**，但被试间"分化更强的人记得更好"这个个体差异命题**没有直接证据**。这是我最大的硬缺口，对应 todo.md 的 **P0.1（换成 within-subject item-level LMM 耦合）** 是救场关键。

---

## 三、学习阶段全脑 GLM（对应 result.md §2–§3）

### 3.1 做了什么
回答："学习过程本身，隐喻 vs 空间在哪里有激活差异？"（定位 Where）

### 3.2 怎么做（技术细节）
- **数据**：28 名被试，学习阶段 **run-3 和 run-4** 合并
- **一阶 GLM**（被试级）：
  - 设计矩阵包含 **yy / kj 两条条件回归项 + nuisance**（6 motion + aCompCor top 6 + WM/CSF + cosine drift + scrubbing spike regressors）
  - **HRF 卷积**：SPM canonical HRF
  - **高通滤波**：128 s
  - **对比**：`Metaphor > Spatial`（c01）和 `Spatial > Metaphor`（c02）
- **二阶 GLM**（组级）：
  - **单样本 t 检验**（one-sample t-test），检验 contrast map 是否偏离 0
  - **Nilearn `SecondLevelModel`**
- **多重比较校正**：**基于置换的 cluster-level FWE 校正**
  - **Cluster-forming threshold**：`voxel-level p < .001`（uncorrected）
  - **显著性阈值**：`cluster-level FWE-corrected p < .05`
  - **置换次数**：**5000 次**（符号翻转置换，sign-flip permutation）
  - **实现**：Nilearn `non_parametric_inference` + `threshold_stats_img`
- **坐标系**：MNI152NLin2009cAsym
- **cluster size 单位**：mm³（voxel size × voxel count）

### 3.3 关键数字
**Metaphor > Spatial（收敛到左前颞）**：
- 左颞极：MNI `-52, 6, -24`，**cluster = 1592 mm³**
- 左前颞邻近：MNI `-38, 12, -36`，**cluster = 392 mm³**

**Spatial > Metaphor（大范围后部视觉-内侧颞-顶叶网络）**：
- 左后梭状回：MNI `-28, -38, -16`，cluster = 4056 mm³
- 左楔前叶：MNI `-10, -54, 8`，cluster = 3648 mm³
- 右楔前叶：MNI `12, -52, 10`，cluster = 2544 mm³
- 右后海马旁回：MNI `28, -32, -18`，cluster = 2440 mm³
- 左外侧枕叶上部：MNI `-38, -78, 30`，cluster = 2200 mm³

### 3.4 一句话结论
> 隐喻**收敛**到左前颞语义枢纽；空间**铺开**到后部视觉-场景-顶叶网络——与 Mashal 2007 / Cardillo 2012 / Lai 2015 文献相容（反复学习后隐喻回到左半球语义主导是合理的）。这同时给后面 RSA / 连接分析提供了 ROI 来源。

---

## 四、前后测 item-wise RSA（对应 result.md §4，**主文核心**）

### 4.1 相似性是怎么算的？（**这里展开最细**）

**（1）单 trial beta 提取**：
- **方法**：**LSS（Least Squares Separate, Mumford 2012）**
- **原理**：对每个 trial $i$，单独建立一个 GLM，其中该 trial 为独立回归项，其他所有 trial 合并为单一"其他"回归项；**避免 LSA（Least Squares All）在快速事件相关设计中的多重共线性问题**
- **每个 trial 输出**：一张 whole-brain beta 图
- **QC**：NaN 剔除 + 极端值（|z| > 5）winsorize

**（2）ROI 内多体素模式提取**：
- 用每个 ROI mask 在 subject-native 空间（经过 MNI 逆变换）抽取每个 trial 的 **voxel-wise beta vector**
- **每个 trial = 一个向量**，长度 = ROI 体素数 `n_voxels`

**（3）相似性计算**：
- **距离度量**：**correlation distance = 1 - Pearson r**
  - $d_{ij} = 1 - \frac{\mathrm{cov}(x_i, x_j)}{\sigma_{x_i}\sigma_{x_j}}$
- **相似性转换**：$s_{ij} = 1 - d_{ij} = r_{ij}$（即 Pearson 相关系数本身）
- **为什么用 Pearson 而不是欧氏距离**：Pearson 对**全局 BOLD 幅度偏移不敏感**，只关心 pattern shape，这是 RSA 文献标准做法（Kriegeskorte 2008）
- **为什么不用 Mahalanobis / crossnobis**：当前未做**噪声协方差估计**，属于 Exploratory 层（todo.md P1.5 CV-RSA 会补）

**（4）item-wise 指标抽取**：
- **yy / kj**：取"**同一学习配对内部**"的相似度
  - 如 `sim(yyw_1, yyew_1)`、`sim(yyw_2, yyew_2)`……
  - 每个条件有 35 对 pair → **35 个 item-wise similarity 值**
- **baseline**：用模板预定义的 **`jx` 伪配对**（如 `sim(jx_1, jx_2)`），共 **20 对**
  - **关键**：不是所有 baseline trial 两两混合，而是固定预定义的伪配对结构，保证与 yy/kj 同质

### 4.2 Primary Endpoint 的 LMM 细节

#### 4.2.0 这个 LMM 到底在做什么？（**汇报时一定会被问**）

**一句话答案**：
> 它在回答一个核心因果问题——"**学习这个事件**本身（pre→post），是不是在**隐喻条件**下比在**空间条件**和**基线**下更强地改变了配对词的神经表征几何？"也就是检验 **`condition × time` 交互效应**。

**为什么叫 "Primary Endpoint"**：
- 在医学临床试验和顶刊方法学里，"primary endpoint" 指**全文只有一个被预先声明的主结论检验**，其他所有分析都是 secondary / exploratory
- 这样做的两个目的：**(1) 防止 p-hacking**（不是"跑了几十个分析挑显著的报告"）；**(2) 聚焦审稿人**（一张图、一个统计、一个结论，清晰可追溯）
- 我们的 primary endpoint 就是这一个 LMM 的 **`condition × time` 交互系数**，整个主文的"主效应"围绕它展开

#### 4.2.1 模型公式逐项解读

```
similarity ~ C(condition) * C(time) + (1|subject) + (1|item)
```

| 项 | 含义 | 这一项在问什么 |
|---|---|---|
| **`similarity`**（因变量） | item-wise pattern similarity（1 - Pearson correlation distance） | "两个配对词在 ROI 里的神经表征有多像" |
| **`C(condition)`** | 三水平分类变量：Metaphor / Spatial / Baseline | "条件本身对相似度的整体影响"（**次要**） |
| **`C(time)`** | 二水平分类变量：Pre / Post | "pre 和 post 整体相似度差异"（**次要**） |
| **`C(condition) × C(time)`**（**关键**） | 交互项 | **"条件对 pre→post 变化幅度的调节"** ← 这是 primary endpoint |
| **`(1|subject)`** | subject 随机截距 | 承认不同被试基线相似度有个体差异 |
| **`(1|item)`** | item 随机截距 | 承认不同配对词有自身难度差异 |

#### 4.2.2 为什么一定要"交互项"而不是简单做三个 paired t？（审稿人最常追问）

**(1) 交互项才是"学习效应"的数学定义**
- 如果只看 `yy 的 pre vs post`，分不清这是**学习造成的**还是**重测 / 熟悉 / 扫描设备漂移**造成的
- 只有"**yy 的变化幅度减去 baseline 的变化幅度**"才是干净的学习效应
- 交互系数 `condition[yy] × time[pre]` 在数学上就是这个差值

**(2) 同时利用了 baseline 和 kj 两个对照**
- **baseline（jx 伪配对）**：封堵"纯重测 / 熟悉化"
- **kj（空间学习）**：封堵"任何学习都会分化"——如果 kj 也分化，就不是隐喻特异
- 三水平 condition 放进同一个模型里一次性比较，比分别做三个 t 检验更有效（**一次估计残差，功效更高**）

**(3) LMM 允许 trial-level 分析，不把被试压成一个点**
- 配对 t 检验：每人每条件要压缩成一个均值 → 丢掉 item 层面信息，N=28
- LMM：用 **trial-level 数据**（每人每条件 35 个 item），**有效样本量 ≈ 28 × 35 × 2 ≈ 1960**，统计功效远高
- 同时用 `(1|subject) + (1|item)` **正确处理两层嵌套结构**，不会膨胀 I 型错误

#### 4.2.3 交互系数具体怎么解读（以 `main_functional` 左前颞家族为例）

```
b_interaction = 0.1109, SE = 0.0172, p < .001
```

- statsmodels 默认参考水平：**`Baseline` × `Post`**（按字母序）
- 所以这个系数在比较：`(Metaphor × Pre) - (Baseline × Post)` 相对 `(Metaphor × Post) - (Baseline × Post)` 的额外差
- **数学上等价于**：

$$
\underbrace{(sim_{yy,pre} - sim_{yy,post})}_{\text{yy 的下降量}} - \underbrace{(sim_{base,pre} - sim_{base,post})}_{\text{baseline 的下降量}} = 0.1109
$$

- **意思**：yy 在 pre→post 的相似度下降，比 baseline 多 0.1109 个单位——**这 0.1109 就是"学习特异的表征重组"**

#### 4.2.4 交互项一共有几个系数？（老师可能接着问）

**有两个**（condition 3 水平 × time 2 水平 → 2 个非参考组合）：

1. **`C(condition)[T.Metaphor]:C(time)[T.Pre]`** ← 主文关注的，**yy 的学习效应**
2. **`C(condition)[T.Spatial]:C(time)[T.Pre]`** ← **kj 的学习效应**（作为对照，预期 ≈ 0 或显著小于第 1 个）

两个系数**必须一起看**，才能讲完整故事："**yy 的交互显著 + kj 的交互不显著 = 学习-条件特异性**"。

#### 4.2.5 结论模板（可直接用）

> 在 ROI X 中，LMM 显示 `condition × time` 交互显著（b=0.11, SE=0.017, z=6.5, p<.001, 95% CI [0.078, 0.145]），说明 yy 的 pre→post 相似度变化幅度显著不同于 baseline；**方向上表现为 yy 的相似度下降**，即 **learning-specific representational differentiation**。

#### 4.2.6 其他工程细节

- **固定效应**：condition（yy/kj/baseline，分类变量）× time（pre/post，分类变量）
- **参考水平（reference）**：**`Baseline` × `Post`**（statsmodels 默认按字母序选择）
- **随机效应**：subject + item 随机截距（**不用随机斜率**，因 item 层 n 仅 35 会导致奇异拟合）
- **估计方法**：**REML（Restricted Maximum Likelihood）** via `statsmodels.MixedLM`
- **推断**：**Wald z 检验**（非 Satterthwaite / Kenward-Roger，因 statsmodels 不原生支持；若审稿人要求会用 R lme4 + lmerTest 复验）
- **ROI 家族分层拟合**（`main_functional` 层）：按 `base_contrast`（Metaphor>Spatial / Spatial>Metaphor）分开拟合，避免"两类 ROI 来源混合解释"
- **多重比较校正**：
  - **Primary endpoint（condition × time 交互）**：**不做额外校正**——因为严格预注册只有一个效应是主文主结论
  - **Secondary 单 ROI 逐个比较**：汇报原始 p 值 + 效应量 dz + 95% CI，**不转换为 ROI 级 FDR**（因为 ROI 间高度相关，Bonferroni/FDR 过保守；顶刊审稿人会要求一致性而不是统一 FDR）

### 4.3 `main_functional`（7 ROI）

**描述性均值**：

| 条件 | Pre | Post | 变化 |
|---|---|---|---|
| Metaphor | 0.1725 | 0.1013 | **下降 0.0712** |
| Spatial | 0.1538 | 0.1455 | 几乎不变 |
| Baseline | 0.0885 | 0.1039 | 轻微上升 |

**统计**（按 ROI 家族分层 LMM）：
- 左前颞家族：`b_interaction = 0.1109, SE = 0.0172, p < .001`
- 后部视觉家族：`b_interaction = 0.0768, p < .001`
- 单 ROI 层：**7/7 显著**，`b = 0.0604–0.1138`
- Spatial × Pre：**0/7 显著**

### 4.4 `literature`（12 ROI）

- **Metaphor × Pre：12/12 全部显著**，`b = 0.0635–0.1212`
- **Spatial × Pre：0/12 显著**（最接近左 IFG p=.050）

### 4.5 `literature_spatial`（12 ROI，**反证性分析**）

- **Metaphor × Pre：12/12 显著**（包括双侧 OPA、PPA、RSC、hippocampus）
- **Spatial × Pre：0/12 显著**，全部 p > .16

**一句话结论**：
> 即使限制在**最偏向空间认知的网络**，kj 也没有分化——**kj 的阴性不能归因为 ROI 选偏**。

### 4.6 `atlas_robustness`（9 ROI）

**描述性均值**（与前面三层 ROI 方向一致）：

| 条件 | Pre | Post | 变化 |
|---|---|---|---|
| Metaphor | 0.1725 | 0.1072 | **下降 0.0653** |
| Spatial | 0.1593 | 0.1619 | **轻微上升 +0.0026**（不是下降） |
| Baseline | 0.0777 | 0.1039 | 上升 +0.0262 |

**Metaphor × Pre**：**9/9 ROI 全部显著**，`b = 0.0586–0.1107`（全部 p ≤ .016）

**Spatial × Pre**：**3/9 ROI 显著**——**必须单独讲清楚**，因为老师会问

| ROI | 系数 b | p 值 | 方向解读 |
|---|---|---|---|
| **左 AG** | **0.0647** | **p = .010** | 相对 baseline 显著 |
| **左 IFGop** | **0.0399** | **p = .046** | 相对 baseline 显著（刚过阈） |
| **左 IFGtri** | **0.0506** | **p = .011** | 相对 baseline 显著 |
| 左 fusiform | — | p > .33 | 不显著 |
| 左 pMTG | — | p > .33 | 不显著 |
| 左 precuneus | — | p > .33 | 不显著 |
| 左 temporal pole | — | p > .33 | 不显著 |
| 右 parahippocampal | — | p > .33 | 不显著 |
| 右 precuneus | — | p > .95 | 不显著 |

#### 4.6.1 kj 这 3 个显著效应的正确解读（**老师会追问，必须讲清楚**）

**关键点 1：方向不是"下降"，而是"相对 baseline 显著"**
- 从描述性均值看，**Spatial 的原始相似度 Pre → Post 实际是**从 0.1593 **轻微上升**到 0.1619，**不是下降**
- 单 ROI 层面（左 AG / 左 IFGop / 左 IFGtri）的 `C(condition)[T.Spatial]:C(time)[T.Pre]` 显著为正，意味着"**kj 的 pre→post 变化幅度** vs **baseline 的 pre→post 变化幅度**"出现了显著差；这是**相对差**，**不等于 kj 自己在下降**
- 具体数学：baseline 在 post 阶段相似度**上升了** ~0.026（基线被熟悉化抬了一点），而 kj 几乎不变（+0.0026）；所以 kj 相对于 baseline 的"趋势"是**"baseline 抬了，kj 没抬"**——这个差被 LMM 捕获为 Spatial 系数显著，但**并不是 kj 自己出现了分化**

**关键点 2：和 yy 的对比显示量级和一致性都弱得多**
- yy：**9/9 ROI 显著**，系数 0.0586–0.1107
- kj：**3/9 ROI 显著**（仅左侧顶额/额下回），系数 0.0399–0.0647
- 即便在这 3 个 ROI 上，**yy 的系数（0.0972 / 0.1030 / 0.0977）都明显大于 kj 的系数（0.0647 / 0.0399 / 0.0506）**
- 所以定性结论依然是："**kj 有限、局部、弱于 yy**"，不是"kj 和 yy 一样分化"

**关键点 3：为什么这 3 个 ROI 出现 kj 信号？合理推测**
- 左 AG / 左 IFGop / 左 IFGtri 都属于**语义控制 + 高阶整合网络**——不只是 yy 用，kj 的"视空间联结"也会调用这些区域做跨词整合
- 它们并不是 kj 的"专属场地"（kj 专属场地是 OPA / PPA / RSC / hippocampus——见 §4.5 `literature_spatial`，那里 kj 是 **0/12 显著**）
- 所以这 3 个 ROI 的 kj 信号更像是**"任何配对学习都会引起的通用整合信号"**，不是"空间认知特异的表征分化"

#### 4.6.2 这一节老师可能追问 + 回答

**Q：你不是说 kj 在 atlas 层也没分化吗？这 3 个 ROI 算什么？**
> A：两点澄清：**(1) 方向上 kj 的原始相似度是轻微上升，不是下降**——这 3 个显著效应是"**kj 相对 baseline** 的变化幅度差异"，不等于 kj 自己发生了 differentiation；**(2) 量级上**，即便在这 3 个 ROI 上 yy 的系数也都明显大于 kj；**(3) 跨 ROI 一致性**，yy 9/9，kj 3/9 且局限于左侧语义控制区而非空间专属区。我们的主文写法是"kj 在 atlas 层出现有限、局部的信号，缺乏跨 ROI 一致的去相似化"，不是"kj 完全无变化"。

**Q：为什么主文 Fig 4 没突出这 3 个 ROI？**
> A：Fig 4 是组水平的 **delta similarity violin 图**（不是 ROI 级统计图），显示的是 `Post - Pre` 的被试分布。由于 kj 原始方向是**轻微上升**，它的 delta 在图上本来就趋近 0 / 轻微正向，不会被误读成"kj 也在下降"。ROI 级 3/9 这个结果在 result.md §4.6 已经完整汇报，主文正文的稳健性段落也会明确写"kj 仅在少数左侧顶额 parcel 出现有限信号，缺乏跨 ROI 一致性"。

**Q：这会不会动摇主结论？**
> A：不会。主结论的 three-way 结构是：**yy 跨 9/9 显著 + baseline 不下降 + kj 局部/弱/不一致**。kj 这 3 个局部信号恰恰反映了 AG/IFG 是**通用的跨词整合节点**（不分任务类型都要用），而**真正的空间专属网络（literature_spatial 12/12 kj 无效应）**才是检验 kj 是否分化的"家门口"——它在那里依然 0/12。这进一步支持了 yy 的分化是**隐喻学习特异**的。

### 4.7 Trial / voxel 稳健性检查（技术细节）

**为什么做**：堵"trial 数或体素数不平衡造假象"。

**方法**：
- **trial 数**：从 `pattern_root/*_metadata.tsv` 统计每被试每条件 pre/post 的原始 trial 数
- **downstream pair 数**：从 `rsa_itemwise_details.csv` 统计每个 `subject × ROI × stage × condition` 的 item-wise pair 数
- **voxel 数**：从 `rsa_summary_stats.csv` 抽取每个 ROI 的 `n_voxels`
- **统计**：
  - **`exact_match_all_subjects` 布尔值**：检验是否**逐被试逐 ROI 完全匹配**（不只是均值相等）
  - **paired t-test** 比较 pre 与 post 的 n_voxels（`ttest_p = 1.0` = 数值完全相等）

**发现**：
- yy 70 vs 70，kj 70 vs 70，baseline 40 vs 40，**全部 `exact_match = True`**
- ROI `n_voxels` 在 pre 与 post **数值完全一致**（`post_minus_pre_mean = 0, ttest_p = 1.0`）

### 4.8 四层 ROI 整合（**主文主结论**）

> 四套 ROI 定义（数据驱动 + 文献预设隐喻 + 文献预设空间 + atlas parcel）**全部收敛**到同一方向：**yy pre→post 显著分化，kj 不稳定，baseline 无下降**——这是 **representational differentiation / reorganization**，不是 pair convergence。

---

## 五、Model-RSA（对应 result.md §5，**机制层**）

### 5.1 neural RDM 是怎么生成的？

**（1）neural RDM 构造**：
- **分析模式**：**`analysis-mode=pooled`**（关键：**不是** yy/kj/baseline 分别做 RDM，而是把三类 trial 合并成**同一个 RDM**）
- **对每个 `subject × ROI × time`（即每个前/后测）单元**：
  1. 取出所有 trial（yy 70 + kj 70 + baseline 40 = 180 trial，但 item-wise 层面做聚合）
  2. 对每两个 item 计算 **pattern similarity（1 - Pearson correlation distance）**
  3. 构造成对称 RDM 矩阵 $R \in \mathbb{R}^{n \times n}$
  4. **抽取上三角 vectorized RDM**（去对角线）→ flatten 成 1D 向量
- **输出表**：`model_rdm_subject_metrics.tsv`，含 `subject × ROI × time × model` 的 Spearman ρ

### 5.2 模型 RDM 是怎么生成的？（**关键技术细节**）

当前正式报告 5 个模型 RDM，每个都是**与 neural RDM 同维度的向量**：

**M1_condition（条件类别）**：
- **定义**：`RDM[i,j] = 0 if condition(i) == condition(j) else 1`
- **粒度**：二值
- **含义**：检测 neural geometry 是否按 yy/kj/baseline 三类分块

**M2_pair（同 pair 接近 / convergence 探针）**：
- **定义**：`RDM[i,j] = 0 if pair(i) == pair(j) else 1`
  - 即，只有 `yyw_1` 与 `yyew_1` 这种"同一学习配对内部"距离为 0，其他为 1
- **含义**：如果学习让配对项聚合，M2 与 neural RDM 的相关应上升

**M3_embedding（预训练语义距离）**：
- **来源**：**中文 BERT-base-chinese 预训练模型（huggingface）**
- **抽取层**：`last_hidden_state` 的 `[CLS]` token（或 mean-pooled token embedding）
- **相似性度量**：**cosine distance = 1 − cos θ**
  - 公式：$d_{ij} = 1 - \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \cdot \|\vec{v}_j\|}$
  - 值域：$[0, 2]$（同向=0，正交=1，反向=2）
- **含义**：检测 neural geometry 是否按预存语义排列

**为什么用 cosine distance 而不是欧氏距离 / Pearson / Jaccard？（老师会追问）**

**理由 1：cosine 是词嵌入领域的事实标准**
- Word2Vec（Mikolov 2013）、GloVe（Pennington 2014）、BERT（Devlin 2019）**所有主流嵌入模型发布时都用 cosine 作为相似性度量**
- BERT 原始论文与下游语义相似性任务（STS-B、SICK、Word Similarity）**全部用 cosine**
- 这是最能与现有文献 / 审稿人直觉对齐的选择

**理由 2：BERT embedding 的几何特性适合 cosine**
- BERT embedding 是**高维稠密向量**（bert-base-chinese 为 768 维）
- 在高维空间中，**向量长度 `|v|` 主要反映词频/训练次数等非语义特征，而向量方向才反映语义内容**
- **cosine 只关心方向，自动消除 `|v|` 带来的偏差**；欧氏距离则会被长度差异污染（例如常用词的 embedding norm 通常更大）
- 这一点 Ethayarajh 2019（EMNLP）对 BERT 各层 embedding 的几何分析中明确指出：**contextualized embedding 在高层存在各向异性（anisotropy），cosine 是比欧氏更鲁棒的选择**

**理由 3：与 neural RSA 的"相关距离"形成互补，不冗余**
- neural RDM 用的是 `1 - Pearson r`（相关距离），而 M3 用 cosine distance
- **两者本质不同**：Pearson r = 先中心化再做 cosine（即 Pearson 等价于 "centered cosine"）；而 cosine 直接用原始向量
- 在 **BERT embedding 上通常不做中心化**（因为原始方向本身就携带语义），这是与 neural pattern（BOLD 信号有基线漂移，必须中心化）不同的常规做法
- 两边各自用最合适的度量，**不会因为度量类型不匹配造成系统性偏差**

**理由 4：Spearman 秩相关让度量选择的影响进一步降低**
- 我们最后比较 neural RDM 与 M3 RDM 时用的是 **Spearman 秩相关**（见 §5.3），只关心**两者排序是否一致**，不关心绝对距离数值
- 所以即便用 cosine 和 Pearson 得到的数值不同，**只要 RDM 上三角向量的秩相似，结果就稳健**
- 这也是 RSA 文献（Kriegeskorte 2008、Nili 2014）推荐 Spearman 的原因之一——**降低对距离度量选择的敏感性**

**是否合理？—— 总结**
> **合理**。cosine distance 是词嵌入领域的标准度量，符合 BERT 几何特性，与 neural 端的相关距离互补而不冗余，加上下游用 Spearman 秩相关比较，**M3 的结果对度量选择不敏感**。

**会被审稿人挑战的边界与应对**：
- **挑战 1：为什么不用静态 word2vec / GloVe 替代？**
  - 回应：BERT 是上下文词嵌入的 SOTA，更贴近人脑语义表征（Caucheteux & King 2022 Nature Human Behaviour 证明 Transformer embedding 与人脑语言区表征有显著对齐）；若审稿人要求会补 **word2vec 中文版** 与 **汉语 tencent AILab embedding** 做 sensitivity analysis
- **挑战 2：`[CLS]` token 还是 mean-pooling？**
  - 回应：两种均可；当前默认 `[CLS]`，但我们在 `build_stimulus_embeddings.py` 中保留了 `--pool-strategy cls / mean-pool` 两个选项，**如审稿人要求可以 sensitivity 两种都跑**
- **挑战 3：单词层 vs 词组层？**
  - 回应：当前对每个中文词/短语取单次 BERT forward，整体作为 single embedding；若词组由多 token 组成则 mean-pool token-level embedding

**M7_memory（回忆强度差异）**：
- **来源**：run-7 每个被试每个 item 的 **memory score**
- **数据管线**：
  1. [`build_memory_strength_table.py`](./rsa_analysis/build_memory_strength_table.py) 读 `run-7_events.tsv` 的 `memory`（0/1）列
  2. **按 `real_word` 做 groupby + mean 聚合**（见脚本 L91-94）
  3. 输出 `memory_strength_{subject}.tsv`，列为 `word_label, recall_score`
- **定义**：`RDM[i,j] = |recall_score(i) - recall_score(j)|`（被试特定）
- **特殊处理**：**`sub-12` 缺失记忆表，该被试 M7 计算被排除**（有效 N = 27 而不是 28）
- **含义**：检测 neural geometry 是否按回忆强度差异排列

**⚠️ 重要澄清：M7 是否退化成二值？（老师追问的核心问题）**

**问题本质**：如果 run-7 每个 `real_word` 只被测试**一次**（只出现一个 0 或 1 值），那么 `groupby.mean()` 之后 `recall_score` 仍然只有 0 / 1 两个值，进而 `|recall_i - recall_j|` 只会产生三种距离：**0（都记住或都没记住）/ 1（一个记住一个没记住）**，**M7 实际退化成二值矩阵**——这是 M7 的真实技术现状。

**当前数据结构**：
- 我们的 run-7 设计**每个 real_word 通常只测试 1 次**（run-7 是单次最终记忆测验，不是多次重复测验）
- 所以 `groupby('real_word').mean()` 聚合实际上**没有降低粒度**，输出的 `recall_score` 对绝大多数 item 就是原始 0 或 1
- **真正发生聚合的情况只有**：同一 `real_word` 在 run-7 里因为设计原因出现了多次 trial（比如含 fake 与 real 形式的冗余）才会出现连续值

**这会带来什么后果？**
1. **M7 粒度粗糙**：二值 RDM 只有 `{0, 1}` 两种距离，对 neural RDM 的 Spearman 相关功效天然低于 M3（连续 cosine）
2. **与 M1_condition 高度相似**：M1 也是二值，M7 退化后和 M1 的信息论容量接近（只是分组规则不同）
3. **"M7 不显著" 要谨慎解读**：结果中 **M7 在 7/7 ROI 不显著**，**这不一定意味着"回忆强度和神经几何无关"**，也可能是**度量太粗造成的统计功效不足**

**当前结果汇报的合适措辞**：
> ❌ 不写："M7 不显著，说明记忆强度不是主导变量" （过度解读）
> ✅ 写作："在当前二值化 M7 指标下，回忆强度与神经几何的耦合未达显著，不能排除粒度限制导致的功效不足；对回忆强度-几何关系的更精细检验在 todo.md P0.1（item-level within-subject LMM）中展开"

**改进方案（todo.md 已列入 P0.1 / P0.4）**：

**方案 A（最优）：用反应时 + 记忆联合构造连续 confidence**
- `confidence = memory × (1 - rank_percentile(log_rt))` 或 `memory + 0.5 × (1 - normalized_log_rt)`
- **直觉**：答对且反应快 → 高 confidence；答对但反应慢 → 中等；答错 → 低
- **优点**：变成 0–1 连续变量，M7 RDM 获得细粒度信息

**方案 B：用 yy 的三条件学习阶段表现加权**
- 把 run-3/4 学习阶段的 item-level accuracy（多次重复）纳入 recall_score 计算
- 学习阶段多次 trial 自然提供连续 0–1 分数

**方案 C：Bayesian item-response model**
- 用 IRT 模型（如 1PL/2PL）从 run-7 的 0/1 响应反推每个 item 的**潜在难度参数**
- 输出连续 latent recall parameter，作为 M7 的输入

**方案 D（最简实现）：直接用 0/1 但改成 **Jaccard 距离** / **Hamming 距离** 的概率化版本**
- 对每个被试内的 item 做 smoothing（加 pseudocount），避免纯 0/1
- 这是"最不动现有管线"的权宜方案

**推荐行动**：
- **主文**：把 M7 **降级为 Secondary / Exploratory 模型**（而不是和 M2/M3/M8 并列作为主机制模型）；在方法学章节明确声明 "**M7 因 run-7 单次测验结构，当前为二值近似；不显著结果不等同于'回忆强度与几何无关'**"
- **P0.1 within-subject item-level 耦合**：绕开 M7 的结构问题，**直接把 item-level memory (0/1) 作为 trial-level LMM 的因变量**，让随机效应结构吸收被试间方差，这才是 "So what" 层面真正有力的脑-行为桥梁
- **P0.4 机制深化**：用方案 A 重新构造 **M7_continuous_confidence**，若仍不显著再写入 SI

**M8_reverse_pair（同 pair 分化探针 / differentiation 镜像）**：
- **定义**：`RDM[i,j] = 1 - M2_pair[i,j]`（即 M2 的数学反转）
- **含义**：M2 和 M8 是**互补编码**，`model_collinearity.tsv` 中**唯一稳定 `high` 共线**的模型对
- **解读原则**：M2 和 M8 当作**同一条证据链的镜像**，不互相做控制变量

### 5.3 Model-neural 相关怎么计算？

- **方法**：**Spearman 秩相关**（非 Pearson）——RSA 文献标准做法，对 RDM 非线性单调关系更稳健
- **为什么不用 Pearson**：因 RDM 通常不满足线性假设，Spearman 更稳（Nili et al. 2014 RSA toolbox 推荐）
- **对每个 `subject × ROI × time × model`** 输出一个 `ρ` 值
- **偏相关 partial ρ**：进一步控制其他模型时用 **Spearman partial correlation**（控制 M1/M2/M3/M7 余项，估计每个模型的独立贡献）

### 5.4 ROI 级 Post vs Pre 比较（怎么统计）

- **数据**：对每个 `subject × ROI × model`，得到 `ρ_post` 和 `ρ_pre`
- **统计**：**配对 t 检验（paired t-test）**，**不是**独立样本 t 检验
- **效应量**：**Cohen's dz**（paired-sample 标准化效应量，dz = t / √n）
- **自由度**：df = n - 1 = 27（M7 为 26）
- **多重比较**：
  - **ROI × model 格子层面**：**不做统一 FDR**（报告原始 p 值 + dz + 一致性模式）
  - **理由**：子刊审稿人更关心"哪个 ROI 呈现稳定可解释模式"，而不是"多少格子过 p < .05 / FDR_BH"；严格 FDR 会过保守（ROI 间相关高 + 模型间 M2/M8 镜像）
  - **写作策略**：**报告 Cohen's dz + 95% CI**，让审稿人自己判断
  - **P0.5 会补 Bayes Factor**（`utils/bayesian_stats.py`）防 desk-reject

### 5.5 `main_functional` 关键发现

- **左 Precuneous**：M2_pair `t(27) = -3.06, p = .0049, dz = -0.58`；M8 镜像 `t = 3.06, dz = 0.58`
- **左 Temporal Pole**：M3_embedding `t(27) = -2.74, p = .0107, dz = -0.52`
- M7_memory：**7/7 ROI 全部不显著**
- 偏相关平均值接近 0：**真正稳定效应集中在少数关键 ROI，不是全局 shift**

### 5.6 `literature` 关键发现

- **左 IFG**：M2_pair `t(27) = -3.47, p = .0018, dz = -0.65`；M8 镜像 `t = 3.47, dz = 0.65`
- 其他 ROI 多为趋势水平（右 IFG p=.081，右 pSTS p=.096）

### 5.7 Δρ LMM（跨 ROI 的模型比较）

**做什么**：**原定位**是从"单 ROI 配对 t"升级到"跨 ROI 的模型整体轨迹"，检验"哪个模型在 ROI 平均上最能解释学习后的几何变化"。

**方法**：
- **因变量构造**：对每个 `subject × ROI × model`，算 `Δρ = ρ_post - ρ_pre`
- **模型公式**：`Δρ ~ C(model) + (1|subject) + (1|roi)`
- **参考模型**：**`M1_condition`**（截距 = M1 的平均 Δρ；其他系数 = 相对 M1 的额外变化）
- **估计**：**REML（statsmodels MixedLM）**
- **推断**：**Wald z 检验** + 95% Wald CI
- **多重比较**：**4 个非参考模型的 Bonferroni 为 α/4 = .0125**（本节没做 Bonferroni，因为结果只有 M3 一条过 .001 级别，远低于任何校正阈值）
- **收敛诊断**：`converged` 字段；**literature 层标记 `converged = false`**，所以只作趋势证据

**关键结果**：
- **main_functional（973 obs, 7 ROI，收敛）**：
  - M3_embedding：`b = -0.0119, SE = 0.0022, z = -5.43, p < .001`（**最稳定负向**）
  - M2_pair：`b = -0.0038, p = .087`（趋势）
- **literature（1668 obs, 12 ROI，未收敛）**：M2 `p = .065`

---

#### ⚠️ 5.7.1 Δρ LMM 的重大解读风险（老师已经问到 + 主文必须降级）

**老师追问（2026-04-27）**：
> "Δρ LMM 相对于单模型看单 ROI 有什么额外的意义？ROI 要么是论文参考（可多可少），要么是 metaphoric 和 spatial 比较的合集（没有单取 metaphoric 大的或者 spatial 大的），这种集合 ROI 的不确定性，对结果影响应该很大。"

**答案**：**老师的担心完全成立**。Δρ LMM 在当前 ROI 集合设计下有三个严重问题：

**问题 1：`main_functional` 是异质 ROI 合集**
- 这 7 个 ROI 是 **`Metaphor > Spatial` + `Spatial > Metaphor` 两个 GLM contrast 的 peak 合集**
- 这意味着里面**既有 yy 激活更强的 ROI，也有 kj 激活更强的 ROI**——**功能上相反**
- **当前设计没有"单独取 Metaphor>Spatial 派生 ROI"做 Δρ LMM**，这是主文机制定位的一处缺口

**问题 2：LMM 的"平均"会抵消 / 稀释异质方向**
- 如果 `Metaphor>Spatial` 家族 M2 下降 −0.010，`Spatial>Metaphor` 家族 M2 下降 −0.006
- Δρ LMM 直接平均成 −0.008，这个数值**既不属于前者也不属于后者**
- 极端情况（一个 −0.010，一个 +0.002）下，固定效应会显示"M2 不显著"，但**真相是"某亚群显著"被另一亚群抵消**

**问题 3：`(1|roi)` 不够用**
- `(1|roi)` 随机截距只吸收"每个 ROI 的基线 Δρ 起点差异"
- **它不吸收"ROI × model 交互"**——即不同 ROI 在同一模型上效应方向不同
- 要修正必须加 `(C(model)|roi)` 随机斜率，但 ROI 只有 7–12 个，随机斜率会严重奇异拟合
- 旁证：`literature` 层已经 `converged = false`，说明方差结构本身就不稳

**所以 Δρ LMM 的"额外意义"比论文描述的弱得多**：

| 项目 | 理论上 | 你的数据上 |
|---|---|---|
| 跨 ROI 一致性证据 | ✅ 更严谨 | ⚠️ 混合异质 ROI 后一致性被稀释 |
| 多模型同台对比 | ✅ 可行 | ✅ 仍可行（"M3 最负"是稳健的） |
| 两层方差估计 | ✅ 清晰 | ⚠️ literature 层未收敛说明方差结构不稳 |
| **机制定位** | ❌ **不承担** | ❌ **不承担** |

**正确的角色定位（主文必须改）**：
- ❌ **不要**把 `main_functional` M3 `p < .001` 当作"主机制证据"
- ✅ **把 Δρ LMM 定位为"跨 ROI 方向一致性的鲁棒性检查"**
- ✅ **机制定位由单 ROI 配对 t 承担**：左 Precuneous（M2↓M8↑）+ 左 IFG（M2↓）+ 左颞极（M3↓）
- ✅ **Fig 3 主画面换成单 ROI 配对 t 效应矩阵**；Δρ LMM 降级为副图或 SI forest plot

#### 5.7.2 修复任务（已加入 todo.md P0.0b）

**方案 A（最关键）：ROI 家族分层 Δρ LMM**
- 按 `base_contrast` 分两组分别拟合：`Metaphor>Spatial` 家族 / `Spatial>Metaphor` 家族
- 两家族方向一致 → 合集平均结论成立；方向相反 → 主文必须拆开写

**方案 B：只用 "Metaphor > Spatial" 派生 ROI 的 Δρ LMM**
- 这是**真正干净的"隐喻机制 ROI"子集合**
- 是**机制专属**证据（对应老师提的"单取 metaphoric 大的"思路）

**方案 C：随机斜率敏感性**
- 拟合 `Δρ ~ C(model) + (1 + C(model)|roi) + (1|subject)`
- 若随机斜率方差显著 → 证明模型在 ROI 间方向异质，固定效应不可用
- 若 ROI 过少奇异拟合 → 改用 **leave-one-ROI-out** 敏感性（每次去掉一个 ROI 看固定效应是否稳健）

**方案 D：literature 层 `converged = false` 的处理**
- 不再在主文中引用 literature 层 Δρ LMM 的具体数值
- 改为**单 ROI 配对 t 的 vote-counting 表** + BF10

#### 5.7.3 汇报时遇到这个问题的"安全回答"

> "老师您说得对——Δρ LMM 在 `main_functional` 合集上做平均，确实把两个功能相反的 ROI 家族混在一起估计，这个平均效应不对应单一机制。所以我们的**机制定位证据其实是单 ROI 配对 t**：左 Precuneous / 左 IFG 的 M2↓M8↑镜像 + 左颞极 M3↓。Δρ LMM 应该被降级为**跨 ROI 方向一致性的鲁棒性检查**，而不是机制定位工具。修复方案（todo.md P0.0b）包括：(1) 按 `base_contrast` 做家族分层 Δρ LMM；(2) 单独用 Metaphor>Spatial 派生 ROI 重跑；(3) 加随机斜率或 leave-one-ROI-out 敏感性。Fig 3 的主画面也会改成单 ROI 效应矩阵，Δρ LMM 放 SI。"

### 5.8 整合机制故事（修订后口径）

> **机制证据的主力是单 ROI 配对 t**：左 Precuneous + 左 IFG 的 M2↓M8↑ 镜像（pair-based differentiation）+ 左 Temporal Pole 的 M3↓（semantic reweighting）——两条并行机制路径；
>
> **Δρ LMM 作为跨 ROI 方向一致性的鲁棒性检查**，`main_functional` 层 M3 跨 ROI 平均最稳定负向，反映分化机制在合集内方向一致，但**不用于机制定位**；
>
> **M7_memory 受二值化限制**（见 §5.2 和 todo.md P0.0a），其不显著不等价于"回忆强度无关"，脑-行为耦合交由 P0.1 within-subject trial-level LMM 承担。

---

## 六、主文固定口径（对应 result.md §6–§7）

### 6.1 核心主张
1. 隐喻学习相较空间学习诱发了**更强的学习后表征重组**
2. 表现为 yy 的 pair similarity 从 pre 到 post **更明显下降**
3. 支持 **representational differentiation / reorganization**，不是 pair convergence
4. 机制层面：**pair-based differentiation + semantic reweighting**

### 6.2 必须保留的克制
- ❌ 不写"已证明隐喻学习失败所以没收敛"
- ❌ 不写"已完全排除所有质量解释"
- ❌ 不把 literature 层 Δρ LMM 当强裁决（`converged = false`）
- ❌ 不把 M7_memory 写成主导机制
- ❌ 不写"空间学习没有任何神经变化"——应写"kj 缺乏跨 ROI 一致可靠的去相似化"

### 6.3 QC / reliability 固定写法
> voxel split-half reliability 不支持"post 阶段整体测量崩坏"；motion/QC 指标在 post 确实略差且与 yy 变化相关——**质量因素可能影响效应幅度，但不足以单独解释主结果方向**。

---

## 七、主文 5 张图

| Fig | 标题 | 证据 | 技术制图 |
|---|---|---|---|
| **Fig 1** | Behavioral advantage for metaphor learning | memory + log_rt_correct 双终点 | **Violin-first + paired-line（同被试 pre/post 连线）+ 组均值 SD** |
| **Fig 2** | Step 5C item-wise RSA reveals stronger post-learning de-similarity | Pre/Post × 三条件 split-violin + ROI Δ similarity | **Split-violin + 半透明 dot + 黑色组均值** |
| **Fig 3** | Model-RSA constrains to pair-based differentiation + semantic reweighting | main_functional + literature 两层 Δρ 热图 | **heatmap + 星号标记（ROI 级配对 t 显著）** |
| **Fig 4** | Metaphor effect is robust across ROI-definition schemes | 四层 ROI × 三条件 delta similarity | **Violin-first + 被试级 dot** |
| **Fig 5** | Activation strength and representational reorganization are dissociable | 同一 ROI 内激活 z vs 重组 z | **Paired-point 连线，z-normalized axis** |

---

## 八、导师可能追问的问题 + 预备回答

### Q1：你用的是 t 检验还是其他？
**A**：**分三层**：
- 行为层：**GLMM + Wald z**（非 t）
- 学习 GLM 二阶：**one-sample t + 5000 次 sign-flip permutation**
- Item-wise RSA：**LMM + Wald z**
- Model-RSA（post vs pre）：**paired t-test + Cohen's dz**
- 跨 ROI Δρ：**LMM + Wald z**

### Q2：多重比较校正用了什么？
**A**：
- **学习 GLM**：**cluster-level FWE（5000 次置换），cluster-forming p<.001，FWE<.05**
- **RSA Primary endpoint**：**只有 1 个主效应，不校正**
- **ROI 级单 ROI 检验**：**不统一 FDR**，报告原始 p + dz + 95% CI，依靠**跨 ROI 一致性**（12/12、9/9）作为证据
- **Δρ LMM**：**默认无校正**，但结果 M3 `p < .001` 远超任何校正阈值
- **P0.5 会补 Bayes Factor** 增强稳健性

### Q3：相似性到底怎么算？
**A**：**1 - correlation distance（即 Pearson r）**，在 ROI 内对两个 trial 的 voxel-wise beta vector 求 Pearson 相关。**不用欧氏距离**（对全局幅度敏感），**不用 Mahalanobis/crossnobis**（未做噪声协方差估计，列 Exploratory 层）。

### Q4：RDM 模型是怎么构造的？
**A**：5 个模型 RDM 都是**与 neural RDM 同维度的向量**：
- **M1_condition**：二值（同类=0，否则=1）
- **M2_pair**：二值（同配对=0，否则=1）
- **M3_embedding**：**中文 BERT（bert-base-chinese）[CLS] token cosine distance**
- **M7_memory**：`|memory_score_i - memory_score_j|`，**被试特定**
- **M8_reverse_pair**：`1 - M2_pair`（M2 的数学反转）

### Q5：M2 和 M8 是镜像，为什么同时报告？
**A**：**这是有意的**。M2 和 M8 共线（`model_collinearity.tsv` 唯一 `high` 标记），偏相关时它们互相扣除会造成人为零效应；我们把它们当作**同一条"收敛 vs 分化"证据链的两面**，M2↓ 和 M8↑ 出现 `dz = 0.58` 的镜像恰恰是 differentiation 路径的**双向确认**，不是两个独立效应。

### Q6：neural RDM 用的是 pooled 还是 condition-specific？
**A**：**pooled**（`analysis-mode=pooled`）。对每个 `subject × ROI × time`，把 yy/kj/baseline 三类 trial 合并成**同一个 RDM**；这样才能让 M1_condition（跨条件分块）和 M2_pair（跨配对分块）这两类模型有意义地与同一个 neural RDM 比。

### Q7：LMM 用的是什么软件？推断怎么做？
**A**：**Python `statsmodels.regression.mixed_linear_model.MixedLM`，REML 估计**。推断用 **Wald z**（不是 Satterthwaite / Kenward-Roger，因 statsmodels 不原生支持）。**如果审稿人要求，会用 R `lme4::lmer` + `lmerTest::anova` 做 Satterthwaite 自由度修正复验**。

### Q8：为什么脑-行为相关是阴性的？会不会推翻主结论？
**A**：不会。**组水平主效应稳定，个体差异没跑出是不同层级的命题**。三个原因：
1. **N=27 对 between-subject 相关功效有限**
2. **神经指标用的是 pooled ROI 平均**，磨掉了 item-level 信息
3. **行为指标 `acc_YY - acc_KJ` 太粗**

下一步 **P0.1 换成 within-subject item-level LMM**（Kuhl/Rissman 方法），这是认知神经领域更敏感的指标。

### Q9：yy 变"更不像"，为什么行为反而更好？
**A**：不矛盾。**representational differentiation 在 Favila 2016（Nature Communications）、Chanales 2017、Schlichting 2015（Nature）中都是记忆增强的神经基础**。核心逻辑：分化让两个项目在几何上**占据更独立、区分度更高的位置**，这种 separable coding 更容易被准确检索。我们的方向与这条文献线一致。

### Q10：kj 为什么没分化？
**A**：不是 ROI 选偏——`literature_spatial` 12/12 ROI（双侧 OPA/PPA/RSC/hippocampus/PPC/precuneus）kj 都不显著。更可能是 **kj 的材料在视空间-场景网络已有现成表征**，重组空间小；yy 需要建立**全新的跨域映射**，重组幅度更大。这会在 discussion 写成"任务难度 × 表征新颖度"。

### Q11：baseline 用 `jx` 伪配对合理吗？
**A**：`jx` 是实验中**不进入学习的对照词对**，pre/post 都呈现但不做联结。它作为**"纯重测效应"控制**是合理的。关键证据：**baseline 三层 ROI pre→post 都没下降**（甚至轻微上升），恰好说明 yy 下降**不是重测造成的**。

### Q12：N=28 偏小怎么保证可靠？
**A**：(1) 主效应 LMM 是 **trial-level**（不是 subject-level），统计功效更高；(2) **四层 ROI 定义复现**；(3) **trial/voxel 稳健性逐被试精确匹配**（`exact_match = True`）；(4) **P0.5 会补 Bayes Factor + LOSO + Split-half replication** 进一步确认。

### Q13：preprocessing 用了什么？
**A**：**fMRIPrep 标准流程**：motion correction + slice timing + coregistration + MNI152NLin2009cAsym 空间标准化 + 6 mm FWHM 高斯平滑；nuisance 保留 6 motion + aCompCor top 6 + WM/CSF + cosine drift；scrubbing **FD > 0.5 mm 或 DVARS > 1.5** 的 volume。

### Q14：单 trial beta 用什么方法？为什么不用 LSA？
**A**：**LSS（Least Squares Separate, Mumford 2012）**。每个 trial 单独建 GLM，target 为独立回归项，其他合为一个总回归项。**为什么不用 LSA**：事件相关设计中 trial 间隔短，LSA 的设计矩阵会出现严重多重共线性，beta 估计不稳；LSS 通过"target + others"的分解显著降低共线性，是目前 RSA / MVPA 领域的标准选择。

### Q15：ROI 是怎么来的？
**A**：**四层定义防 cherry-pick**：
- **`main_functional`**：从学习期 GLM `Metaphor>Spatial` / `Spatial>Metaphor` 显著 cluster 反向定义（数据驱动）
- **`literature`**：基于 Huang 2023 隐喻 ALE 元分析 + Cardillo 系列定义 **AG/IFG/pMTG/pSTS/precuneus/temporal pole 双侧** 12 个
- **`literature_spatial`**：基于场景/导航文献定义 **OPA/PPA/PPC/precuneus/RSC/hippocampus 双侧** 12 个
- **`atlas_robustness`**：**Harvard-Oxford atlas** 9 个粗粒度 parcel

---

## 九、衔接下一阶段

> 现在主故事（yy 的 representational differentiation）是稳的、四层 ROI 稳健、baseline 稳健、trial/voxel 精确匹配。但要冲 NN / NC / NHB，还必须补三件事：
>
> 1. **个体差异**：把 between-subject 阴性相关换成 within-subject item-level LMM 耦合（**P0.1**）
> 2. **机制深化**：加 **M9 Relational Model + Variance Partitioning**（**P0.4**）；加 hippocampal subfield RSA（**P1.1**）
> 3. **方向性**：补 **Searchlight RSA 全脑 + cluster-FWE**（**P0.2**）+ gPPI / representational connectivity（**P1.2**）+ Learning dynamics sliding-window（**P0.3**）
>
> 具体排期见 [`todo.md`](./todo.md)。

---

*最后更新：2026-04-27；用于导师汇报；技术细节加粗，支持即问即答*
