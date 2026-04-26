## 1、行为结果总结：

当前行为部分已采用 refined 版本结果作为正式口径。主行为终点是 run-7 的 `memory`，补充终点是正确反应 trial 上的 `log_rt_correct`。在 27 名被试中，`YY` 条件下的平均正确率为 `0.680`，高于 `KJ` 条件的 `0.588`；trial-level 混合模型显示这一差异显著，`b = 0.092`, `SE = 0.014`, `z = 6.43`, `p < .001`, 95% CI `[0.064, 0.120]`。因此，行为主结果可以稳定写成：**被试在隐喻联结条件下的记忆表现显著优于空间联结条件。**

补充行为结果同样支持 `YY` 的行为优势。若只考察答对 trial 的反应时，则 `YY` 条件下的平均正确反应时为 `1070.43 ms`，略快于 `KJ` 条件的 `1100.29 ms`；对应的 `log_rt_correct` 混合模型显示 `YY` 的正确提取速度也显著更快，`b = -0.026`, `SE = 0.010`, `z = -2.61`, `p = .009`, 95% CI `[-0.046, -0.007]`。这说明 `yy` 的行为优势并不只体现为“更准”，而是同时表现为**更高的正确率与更高效的正确提取**。

从整条证据链的角度看，这一点非常重要。当前行为层最稳的结论不是单纯的“`yy` 记得更好”，而是：`yy` 在 run-7 的最终测验中表现出更优的学习结果；这种优势既体现在准确率上，也体现在正确反应的提取效率上。因此，后续如果要把神经结果与行为结果放在一起解释，更合适的写法是：**隐喻学习在组水平上同时带来了更好的行为表现和更强的学习后表征重组。**

### 1.1 最小脑-行为分析

在脑-行为关联部分，我们首先只做了一个最小、最克制的 subject-level 分析，避免一开始同时铺开太多 ROI、太多神经指标和太多行为指标。主行为指标定义为 `behavior_primary_diff_yy_minus_kj = accuracy_YY - accuracy_KJ`，表示某位被试在行为上对 `yy` 的优势有多强；主神经指标定义为 `main_functional_primary_diff_yy_minus_kj = (pre_yy - post_yy) - (pre_kj - post_kj)`，表示在主功能 ROI 中，某位被试的 `yy` 去相似化是否比 `kj` 更强。

结果显示，这一最小脑-行为相关并未形成稳定显著耦合。主分析中，`main_functional_primary_diff_yy_minus_kj` 与 `behavior_primary_diff_yy_minus_kj` 的相关较弱且不显著，Pearson `r = -0.163`, `p = .417`，Spearman `rho = -0.144`, `p = .474`。把神经指标改成 `yy` 相对 baseline 的差值后，结果仍不显著（Pearson `r = -0.200`, `p = .316`）。在文献 ROI 层，`literature_primary_diff_yy_minus_kj` 与行为差值也未见稳定相关（Pearson `r = -0.252`, `p = .205`）。同样，主功能 ROI 的神经差值与正确反应时差值之间也没有显著关系（Pearson `r = 0.121`, `p = .547`）。

因此，当前更稳妥的表述是：**行为优势本身成立，神经分化主结果本身也成立，但二者在被试差异层面尚未形成稳定的线性耦合。** 这并不推翻主线，而是说明当前最强的证据仍然来自条件层面的组水平差异，而不是个体差异层面的强相关。换句话说，`yy` 更好学、`yy` 更去相似化，这两件事在组水平上是同向成立的；但“谁分化得更强，谁就一定记得更好”这一更强的个体差异命题，目前并没有得到直接支持。

## 2、学习阶段全脑 GLM 结果总结（Step 3B: `main_analysis.py`）

在 28 名被试的学习阶段（run-3/4）全脑二阶分析中，我们采用了基于 5000 次置换检验的 cluster-level FWE 校正，cluster-forming threshold 设为 `p < .001`，显著性阈值为 cluster-level `FWE < .05`。结果显示，`Metaphor > Spatial` 与 `Spatial > Metaphor` 两个对比均获得了显著簇。

对于 `Metaphor > Spatial`，显著效应主要集中在左侧颞极区域。具体而言，检测到两个显著簇，峰值分别位于左侧颞极（MNI: `-52, 6, -24`，cluster size = `1592 mm³`）和左侧前颞叶/颞极邻近区域（MNI: `-38, 12, -36`，cluster size = `392 mm³`）。这一结果提示，在学习阶段，隐喻联结相较于空间联结更强地募集了左侧前颞语义相关区域。

对于 `Spatial > Metaphor`，显著效应范围更广，主要分布在后部视觉-内侧颞叶-顶叶网络。具体包括：左侧后部梭状回（MNI: `-28, -38, -16`，cluster size = `4056 mm³`）、左侧楔前叶（MNI: `-10, -54, 8`，cluster size = `3648 mm³`）、右侧楔前叶（MNI: `12, -52, 10`，cluster size = `2544 mm³`）、右侧后部海马旁回（MNI: `28, -32, -18`，cluster size = `2440 mm³`），以及左侧外侧枕叶上部（MNI: `-38, -78, 30`，cluster size = `2200 mm³`）。这一模式表明，空间联结学习相较于隐喻联结，更依赖后部视觉表征、场景/空间加工以及顶叶相关区域。

总体上，这一步的全脑 GLM 结果为后续 ROI 级表征分析提供了“空间定位”证据：隐喻条件的优势更偏向左侧前颞语义系统，而空间条件的优势则更多体现在后部视觉-内侧颞叶-顶叶网络。相应的 ROI mask 已在 `glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/` 下生成，可作为后续 RSA、RD/GPS、MVPA 和连接分析的 ROI 来源。

## 3、GLM结果与已有隐喻/空间加工研究的对照解释

从已有隐喻研究看，当前 `Metaphor > Spatial` 主要落在左侧颞极，并不构成明显异常，反而与“隐喻理解更依赖左侧语义整合/语义枢纽系统”的一类结果是相容的。已有文献和本地整理都提示，隐喻加工并不稳定地要求右半球特异参与，其激活格局会受到**新颖性、熟悉度、任务要求和刺激语境**的调节。对于新颖隐喻，Mashal 等（2007）更强调右侧PSTS/右IFG的参与；但 Cardillo 等（2012）和 Lai 等（2015）提示，当隐喻逐渐熟悉，或当任务并非强烈要求新颖意义重解释时，隐喻加工更常表现为以左侧语言-语义网络为主的模式。结合你当前的实验，学习阶段的 `yy` 条件已经经历了反复联结建立，因此没有出现很强的右半球优势，本身并不违背文献。

同时，你现在的 `Metaphor > Spatial` 结果比一些经典隐喻研究更“窄”，主要突出左侧颞极，而没有在 cluster-FWE 主结果中明显出现左IFG、左MTG/STS 或 AG。这一点**不一定是错误**，但提示当前效应更像是“前颞语义枢纽优势”，而不是完整铺开的经典隐喻网络。结合本地方案文档，原始理论预期里 IFG、AG、Precuneus 等也被视为重要候选区，因此后续最好把这一点写成**保守解释**：学习阶段全脑结果首先支持了左前颞语义系统参与隐喻联结学习，但更广泛的语言控制/语义整合网络是否同样参与，还需要后续 ROI 级 RSA、RD/GPS、MVPA 分析进一步补充。

对于 `Spatial > Metaphor`，你现在看到的后部梭状回、双侧楔前叶、后海马旁回和外侧枕叶网络，和空间/场景加工文献是高度一致的。无论是本地方案笔记，还是关于空间语言、场景表征和视空间联结学习的研究，都反复指向**楔前叶/顶叶的空间表征与心理意象功能**，以及**海马旁回/后部梭状回的场景、地点与空间关联编码功能**。因此，`kj > yy` 落在这一后部视觉-内侧颞叶-顶叶网络，从认知神经机制上是非常合理的，并不显示明显异常。

如果说当前结果有什么需要谨慎而不是“明显有问题”的地方，主要有三点。第一，你的主对比是 **隐喻联结 vs 空间联结**，而不是隐喻 vs 字面/无关条件，因此这里更适合被解释为“隐喻学习相对于空间学习更偏向语义系统，而空间学习更偏向视空间-场景系统”，而不是直接下结论说“这些区域就是隐喻加工专属脑区”。第二，`Spatial > Metaphor` 的效应范围明显大于 `Metaphor > Spatial`，这可能说明空间材料在视觉意象、场景表征或联结稳定性上具有更直接的后部皮层优势，也可能提示两类材料在意象性/可视化程度上并不完全等价。第三，当前全脑 GLM 结果本身更像是**定位证据（where）**，是否真的对应“表征重组（how）”，还要看你后面的 RSA、RD/GPS 和脑-行为关联是否能在这些 ROI 中给出同向支持。

可作为这一判断的代表性文献包括：Mashal et al. (2007) 强调**新颖隐喻**更容易招募右侧PSTS/右IFG，而 Cardillo et al. (2012) 与 Lai et al. (2015) 都提示，随着熟悉度上升或在控制新颖性/难度后，隐喻加工并不稳定地表现为右半球优势，反而更常回到以左侧语言-语义系统为主的模式；Rapp et al. (2004) 也报告了隐喻相对字面句更多激活左侧IFG和颞叶语义区。相对地，场景/空间加工研究一致指出海马旁回、枕叶场景区和楔前叶在空间布局、场景知觉和视空间意象中具有核心作用。因此，你当前“左前颞偏隐喻、后部视觉-内侧颞叶-顶叶偏空间”的总格局，与经典文献总体上是相容的，只是隐喻效应更偏向语义枢纽、范围也比部分新颖隐喻研究更收敛。

<br />

## 4、前后测 ROI-level RSA 结果（Step 5C: `run_rsa_optimized.py` + `rsa_lmm.py`）

### 4.1 分析方法简介

这一部分的核心目标，不是再去比较“哪个脑区激活更强”，而是检验**学习是否改变了 ROI 内部的表征结构**。具体做法是：首先基于前后测阶段的单 trial LSS beta，在每个 ROI 内提取每个词项目的多体素模式；随后计算项目间的 pattern similarity。这里采用的是基于 Pearson 相关的相似性指标，即先计算 correlation distance，再转换为 similarity，因此数值越高表示两项词在该 ROI 中的神经模式越相似。接着，我们从完整的相似度矩阵中抽取 item-wise 指标：对 `yy/kj` 来说，核心是“同一学习配对内部”的相似度，例如 `yyw_1` 与 `yyew_1`、`kjw_1` 与 `kjew_1`；对 baseline 来说，当前主流程使用模板中预定义的 `jx` 伪配对（如 `jx_1` 对 `jx_2`）作为控制分布，而不是把所有 baseline trial 两两混合。

在统计分析上，我们进一步采用线性混合模型检验 `similarity ~ condition * time + (1|subject) + (1|item)`。其中，`condition × time` 交互是最关键的效应，因为它直接回答：隐喻联结与空间联结在学习前后，其配对项目的神经相似性变化是否不同。与单纯的主效应相比，这一交互更能对应本研究的核心问题，即隐喻学习是否引发了不同于空间学习的表征重组。

需要特别说明的是：本节 `main_functional`、`literature` 与 `literature_spatial` 三层 ROI 结果都已经更新为**三条件正式版本**。因此，当前第 4 节的 RSA 叙事应以 `yy / kj / baseline(jx)` 同时进入默认主链的 Step 5C 输出为准。三套 ROI 的结果虽然来源不同，但方向高度一致：当前主故事更接近 **representational differentiation（表征分化增强）**，而不是简单的配对收敛。

### 4.2 主功能 ROI（`main_functional`）结果

`main_functional` 的 Step 5C 现在已经是**三条件版本**。当前主链输入来自 `rsa_itemwise_details.csv`，共 28 名被试、7 个主功能 ROI、35280 条 item-wise 记录；其中 baseline 不是额外旁支，而是与 `Metaphor / Spatial` 一起进入同一个 LMM。由于 7 个 ROI 来自两类不同的 GLM 对比家族，脚本默认按 `base_contrast` 分层拟合，因此本节的正式统计结论以 `Metaphor > Spatial` 家族（2 个 ROI）和 `Spatial > Metaphor` 家族（5 个 ROI）的模型为主，再辅以单 ROI 结果。

先看三条件的总体描述性模式，故事已经很清楚。跨 7 个主功能 ROI 的 item-wise 平均值显示：`Metaphor` 条件的 pair similarity 从 Pre 的 0.1725 下降到 Post 的 0.1013，下降幅度最大；`Spatial` 仅从 0.1538 轻度下降到 0.1455；`Baseline` 则从 0.0885 小幅上升到 0.1039。也就是说，当前数据最稳定的变化不是“学习后同 pair 更像”，而是 **yy 在学习后变得更不相似**；与此同时，baseline 没有出现同方向下降，因此这不像是一个普遍的重测/熟悉化效应。

在 ROI 家族层面，这一模式在两类主功能 ROI 中都成立。对于来源于 `Metaphor > Spatial` 的左前颞 ROI 家族，baseline 从 0.0635 上升到 0.1166，而 `Metaphor` 从 0.1699 降到 0.1121；对应的 `C(condition)[T.Metaphor]:C(time)[T.Pre]` 交互显著，b = 0.1109, SE = 0.0172, 95% CI [0.0772, 0.1446], p < .001。这里之所以系数为正，是因为模型把 `Baseline` 与 `Post` 设为参考水平；换成更直观的话说，就是 **Metaphor 的 pre→post 下降显著大于 baseline 的 pre→post 变化**。相比之下，同一家族中 `Spatial` 从 0.1363 轻度上升到 0.1571，`Spatial × Pre` 仅边缘显著，b = 0.0323, p = .061。

对于来源于 `Spatial > Metaphor` 的后部视觉-顶叶-内侧颞叶 ROI 家族，故事同样一致，但 baseline 更稳定：baseline 几乎不变（Pre = 0.0985, Post = 0.0987），`Metaphor` 则从 0.1735 降到 0.0970；对应的 `Metaphor × Pre` 交互高度显著，b = 0.0768, SE = 0.0105, 95% CI [0.0563, 0.0973], p < .001。`Spatial` 在该家族中只表现为较弱下降（0.1608 → 0.1409），其 `Spatial × Pre` 仍仅为边缘显著，b = 0.0202, p = .054。换言之，无论 ROI 是从“隐喻更强激活”的左前颞系统中定义，还是从“空间更强激活”的后部视觉-顶叶-内侧颞系统中定义，最稳定的结果都不是 `kj` 或 `baseline` 的改变，而是 `yy` 的明显去相似化。

单 ROI 结果进一步说明，这一效应并不是由某一个特定脑区偶然驱动。7 个主功能 ROI 中，`Metaphor × Pre` 项在全部 7 个 ROI 内都达到显著，效应范围约为 b = 0.0604–0.1138，p = .023 到 p < .001。两个左侧颞极 ROI 的效应最强，左后梭状回、双侧楔前叶、右后海马旁回和左外侧枕叶也都呈现同方向结果。相反，`Spatial × Pre` 在单 ROI 层面均未达到显著。这说明当前主功能 ROI 的 Step 5C 结果具有很高的一致性：**真正系统性发生前后重组的是隐喻配对，而不是空间配对或 baseline 配对。**

因此，`main_functional` 的当前结果故事可以概括为：学习后最显著的神经表征变化，并不是 pair similarity 的普遍升高，而是 `yy` 在多个主功能 ROI 中出现了稳定的 similarity 下降；`baseline` 基本稳定，部分 ROI 甚至轻度上升；`kj` 变化较弱且不稳定。这个模式对论文主线有两个直接含义。第一，它用 baseline 封堵了“纯重测/熟悉化”解释，使得当前的 `yy` 效应更像**学习特异性的 representational differentiation**。第二，它为后续 Model-RSA 提供了明确约束：Step 5D 不应再默认押注“学习后同 pair 更收敛”，而应重点检验当前这种 post 阶段更强去相似化，究竟更符合 differentiation 路径，还是某种在预存语义底盘之上发生的重加权重组。

### 4.3 论文 ROI（`literature`）结果

在文献 ROI 集（`literature`）上，Step 5C 同样已经是三条件版本。当前输入包含 28 名被试、12 个理论 ROI、60480 条 item-wise 记录。由于这套 ROI 并不来自同一组 GLM 对比家族，脚本没有再做 `base_contrast` 分层，而是直接按 ROI 分别拟合 LMM。因此，`literature` 层的重点不在“一个 pooled 总模型”，而在于：**理论 ROI 是否在方向上表现出高度一致的重复支持**。

先看总体描述性模式，文献 ROI 与主功能 ROI 的方向几乎完全一致，而且隐喻效应更突出。跨 12 个文献 ROI 的 item-wise 平均值显示：`Metaphor` 条件的 pair similarity 从 Pre 的 0.1731 下降到 Post 的 0.1074；`Spatial` 则从 0.1576 上升到 0.1694；`Baseline` 从 0.0726 上升到 0.1025。也就是说，在文献 ROI 中不仅 `yy` 出现了清晰的 post 下降，而且 `baseline` 与 `kj` 都没有呈现同方向下降，反而整体略有上升。这个模式比主功能 ROI 更强地表明：当前观察到的 `yy` 去相似化并不是一个普遍的前后测趋势，而是更接近隐喻学习特异性的表征变化。

在单 ROI 层面，这种一致性非常高。12 个文献 ROI 中，`C(condition)[T.Metaphor]:C(time)[T.Pre]` 在全部 12 个 ROI 内都达到显著，效应范围约为 b = 0.0635–0.1212，p = .0066 到 p < .001。具体而言，左侧 AG（b = 0.1042, p < .001）、IFG（b = 0.1027, p < .001）、pMTG（b = 0.0861, p = .001）、pSTS（b = 0.0947, p < .001）、precuneus（b = 0.0635, p = .007）和 temporal pole（b = 0.1212, p < .001）全部显著；右侧 AG（b = 0.1095, p < .001）、IFG（b = 0.1068, p < .001）、pMTG（b = 0.0984, p < .001）、pSTS（b = 0.0706, p = .002）、precuneus（b = 0.0728, p = .003）和 temporal pole（b = 0.1169, p < .001）也全部显著。按参考组解释，这些正向交互都意味着：**Metaphor 的 pre→post 下降显著大于 baseline 的 pre→post 变化**。

相比之下，`Spatial × Pre` 在文献 ROI 中并不稳定。12 个 ROI 里没有任何一个 ROI 在传统阈值下表现出稳健的 `Spatial × Pre` 效应；最接近显著的是左 IFG（b = 0.0396, p = .050）以及左/右 AG（p ≈ .084）等少数 ROI，但整体上都不足以支持一个系统性的 `kj` 前后重组模式。换句话说，文献 ROI 并没有把主结果改写成“隐喻和空间都在同等程度上变化”，而是进一步强化了这样一个结论：**真正稳定、广泛发生的，是隐喻配对的去相似化，而不是空间配对或 baseline 配对。**

### 4.4 空间文献 ROI（`literature_spatial`）结果

为了进一步检验“`kj` 的空间联结是否会在经典空间网络中表现出更明显的前后测去相似化”，我们又构建并分析了专门面向空间/场景/导航文献的 `literature_spatial` ROI 集。该分析包含 28 名被试、12 个空间理论 ROI、60480 条 item-wise 记录；ROI 包括双侧 OPA、PPA、PPC、precuneus、RSC 和 hippocampus。与 `literature` 一样，这套 ROI 并非来自同一组 GLM 对比家族，因此正式统计以单 ROI LMM 为主，而不是 pooled 总模型。

先看总体描述性模式，结果并没有支持“空间 ROI 中 `kj` 会显著下降”的预期。跨 12 个空间 ROI 的 item-wise 平均值显示：`Metaphor` 条件的 pair similarity 从 Pre 的 0.1623 下降到 Post 的 0.1018，下降幅度依然最大；`Spatial` 则从 0.1480 轻度上升到 0.1532；`Baseline` 从 0.0802 上升到 0.0963。也就是说，即使把 ROI 限定在最偏向空间认知的网络中，最稳定的前后测变化仍然不是 `kj` 的下降，而是 `yy` 的显著去相似化。

单 ROI 层面的结果进一步强化了这一点。12 个空间 ROI 中，`C(condition)[T.Metaphor]:C(time)[T.Pre]` 在全部 12 个 ROI 内都达到显著，效应范围约为 b = 0.0635–0.0956，p = .0066 到 p < .001。具体而言，左侧 OPA（b = 0.0751, p = .004）、PPA（b = 0.0734, p = .002）、PPC（b = 0.0715, p = .001）、RSC（b = 0.0662, p = .005）、hippocampus（b = 0.0693, p < .001）和 precuneus（b = 0.0635, p = .007）均显著；右侧 OPA（b = 0.0937, p < .001）、PPA（b = 0.0823, p = .001）、PPC（b = 0.0766, p < .001）、RSC（b = 0.0801, p < .001）、hippocampus（b = 0.0956, p < .001）和 precuneus（b = 0.0728, p = .003）也全部显著。按参考组解释，这些正向交互都意味着：**Metaphor 的 pre→post 下降显著大于 baseline 的 pre→post 变化**。

相比之下，`Spatial × Pre` 在 12 个空间 ROI 中没有一个达到显著。尽管个别 ROI 的原始均值出现了轻微下降，例如左 OPA（0.1758 → 0.1699）、左 precuneus（0.1813 → 0.1575）或右 precuneus（0.1610 → 0.1455），但这些变化在混合模型中都不稳定，`C(condition)[T.Spatial]:C(time)[T.Pre]` 的 p 值全部大于 .16。与此同时，也有若干经典空间 ROI 反而呈现 `kj` 的轻度上升，例如双侧 PPA、右 OPA、双侧 hippocampus 和右 RSC。因此，这套专门为空间认知构建的 ROI 集并没有把故事改写成“空间联结在空间网络里显著去相似化”，而是再次重复了与前两套 ROI 一致的结论：**最稳定的学习后重组依然发生在 `yy`，而不是 `kj`。**

从解释上看，这一结果的意义非常关键。它说明当前 `kj` 效应偏弱，并不能简单归因为“之前选的 ROI 不够空间化”。即便把分析范围明确收缩到 OPA/PPA/RSC/hippocampus/precuneus/PPC 这样的经典空间网络，`kj` 仍未表现出跨 ROI、一致、可靠的 pre→post similarity 下降。相反，`yy` 的去相似化却在这些空间 ROI 中依旧稳健存在。这意味着，Step 5C 当前支持的不是“不同类型材料分别在各自优势网络内对称重组”，而更接近一种**对隐喻学习更敏感、跨网络可见的表征几何重塑**。

### 4.5 Trial / voxel 稳健性检查

为了进一步排除“`yy` 的 pre→post 去相似化只是由 trial 数或 ROI 体素数不平衡造成”的解释，我们又对三层 ROI 结果做了专门的 trial/voxel 稳健性检查。该检查分别基于 `pattern_root/*_metadata.tsv`、`rsa_itemwise_details.csv` 和 `rsa_summary_stats.csv` 统计三类信息：第一，前后测阶段每位被试在 `yy / kj / baseline` 下的原始 trial 数是否完全匹配；第二，进入 item-wise RSA 之后，每个 `subject × ROI × stage × condition` 的 downstream pair 数是否仍然完全匹配；第三，每个 ROI 在 pre 与 post 阶段用于 RSA 的 `n_voxels` 是否存在系统漂移。

结果显示，这三类平衡性在三套 ROI 中都非常干净，而且不是“均值上差不多”，而是逐被试逐 ROI 的**完全匹配**。以原始 trial 数为例，三层 ROI 的 `trial_metadata_stage_tests.tsv` 都显示：所有被试在 `yy` 条件下均为 `Pre = 70, Post = 70`，在 `kj` 条件下均为 `Pre = 70, Post = 70`，在 `baseline` 条件下均为 `Pre = 40, Post = 40`，且 `exact_match_all_subjects = True`。在 downstream item-wise 层，所有 ROI 都表现为 `Metaphor = 35 vs 35`、`Spatial = 35 vs 35`、`Baseline = 20 vs 20`，同样全部 `exact_match_all_subjects = True`。这意味着当前 Step 5C 的 `yy` 去相似化并不是由 pre/post 条件下进入分析的项目数不平衡造成的。

体素数稳健性检查也给出了同样清晰的结论。`main_functional`、`literature` 与 `literature_spatial` 三层 ROI 的 `voxel_counts_stage_tests.tsv` 均显示：每个 ROI 的 `n_voxels` 在 pre 与 post 阶段完全一致，`post_minus_pre_mean = 0`，`ttest_p = 1.0`，且 `exact_match_all_subjects = True`。例如，主功能 ROI 中左 Temporal Pole 两个 ROI 分别稳定为 `199` 和 `49` 个体素，左后梭状回为 `507`，双侧 precuneus 为 `456` 与 `318`；文献 ROI 中左/右 IFG 分别稳定为 `1336` 和 `1151` 个体素；空间 ROI 中双侧 OPA 稳定为 `4955` 和 `4824` 个体素，双侧 hippocampus 稳定为 `932` 和 `946` 个体素。换句话说，当前主结果同样不能被解释为“post 阶段进入 RSA 的体素数更少或更多，从而机械地改写了 similarity”。

因此，这一步稳健性分析对当前主线提供了非常直接的支持：`yy` 的 pre→post 去相似化并不是由原始 trial 数失衡、downstream pair 数失衡，或 `n_voxels` 漂移造成的。结合前面的 baseline 控制、QC/reliability 结果和三层 ROI 的重复支持，当前更稳妥的判断是：**`yy` 的去相似化更可能反映真实的学习后表征重组，而不是由设计平衡性或 ROI 采样偏差驱动的假象。**

### 4.6 `atlas_robustness` 结果

为了进一步确认当前主结论并不依赖于现有 ROI 的定义方式，我们又在一套更“粗粒度”、更接近标准 atlas parcel 的 ROI 集上重复了 Step 5C。`atlas_robustness` 当前包含 9 个 atlas ROI：左 AG、左 IFGop、左 IFGtri、左 fusiform、左 pMTG、左 precuneus、左 temporal pole、右 parahippocampal 以及右 precuneus。与前三套 ROI 一样，这一层同样使用包含 `Metaphor / Spatial / Baseline` 的默认三条件 item-wise LMM 主链，因此这里要回答的问题非常直接：**如果把 ROI 改写成 atlas 级 parcel，`yy` 的学习后去相似化是否仍然存在。**

答案是肯定的。跨 9 个 atlas ROI 的描述性均值显示：`Metaphor` 的 pair similarity 从 Pre 的 `0.1725` 下降到 Post 的 `0.1072`；`Spatial` 则从 `0.1593` 轻微上升到 `0.1619`；`Baseline` 从 `0.0777` 上升到 `0.1039`。这一方向与前面的 `main_functional`、`literature` 和 `literature_spatial` 基本完全一致：最清晰的前后变化依旧是 `yy` 的下降，而不是 `kj` 或 baseline 的同步下降。换句话说，把 ROI 换成 atlas parcel 并没有把结果改写成“所有条件都一起下降”，而是继续保留了隐喻学习特异性的去相似化模式。

单 ROI 层面的证据同样稳定。9 个 atlas ROI 中，`C(condition)[T.Metaphor]:C(time)[T.Pre]` 在全部 9 个 ROI 内都达到显著，效应范围约为 `b = 0.0586–0.1107`，`p = .0158` 到 `p < .001`。具体而言，左 AG（`b = 0.0972, p < .001`）、左 IFGop（`b = 0.1030, p < .001`）、左 IFGtri（`b = 0.0977, p < .001`）、左 fusiform（`b = 0.1107, p < .001`）、左 pMTG（`b = 0.0951, p < .001`）、左 precuneus（`b = 0.0770, p = .002`）、左 temporal pole（`b = 0.1078, p < .001`）、右 parahippocampal（`b = 0.0762, p = .001`）和右 precuneus（`b = 0.0586, p = .016`）全部显著。这意味着，即便采用 atlas 级 ROI，`Metaphor` 的 pre→post 下降仍然系统性大于 baseline 的 pre→post 变化。

相比之下，`Spatial × Pre` 只有 3 个 atlas ROI 达到显著，分别是左 AG（`b = 0.0647, p = .010`）、左 IFGop（`b = 0.0399, p = .046`）和左 IFGtri（`b = 0.0506, p = .011`）；其余 6 个 atlas ROI 全部不显著（`p = .33–.95`）。因此，atlas 级重复分析并没有把故事改写成“`kj` 其实和 `yy` 一样稳健，只是之前 ROI 选得不对”。更准确的结论是：**`yy` 的去相似化在 atlas 粒度下依旧稳健，而 `kj` 只在少数左侧顶额/额下回 parcel 中出现有限信号，缺乏跨 ROI 的一致性。**

从稳健性角度看，这一层结果非常关键。它表明当前主现象并不是某一套功能性 ROI 或理论 ROI 的偶然产物；即便改用更标准、边界更宽的 atlas parcel，`yy` 的学习后去相似化仍然重复出现。因此，ROI 定义方式的改变不会推翻 Step 5C 的核心判断，反而进一步支持：**隐喻学习相关的表征重组是一个跨 ROI 定义方案都可观察到的稳定现象。**

### 4.7 四层 ROI 的整合结果故事

把 `main_functional`、`literature`、`literature_spatial` 和 `atlas_robustness` 四层 ROI 放在一起看，Step 5C 现在已经形成了一条相当完整的结果链。第一，在四套 ROI 中，`yy` 都表现出最稳定的 pre→post similarity 下降。主功能 ROI 上，这种模式出现在左前颞语义家族和后部视觉-顶叶-内侧颞家族两侧；隐喻文献 ROI 上，它进一步扩展到双侧 AG、IFG、pMTG、pSTS、precuneus 和 temporal pole；空间文献 ROI 上，它又重复出现在双侧 OPA、PPA、PPC、RSC、hippocampus 和 precuneus；atlas parcel 层上，它还在左 AG、左 IFG、左 fusiform、左 pMTG、左 temporal pole、双侧 precuneus 和右 parahippocampal 中稳定重现。这说明这一效应不仅不依赖某一种理论 ROI 选择，甚至在更粗粒度的 atlas 定义下也依然保留。第二，`baseline` 在已有稳健性链条中始终没有出现与 `yy` 相同方向的下降，反而多为稳定或轻度上升，因此当前结果很难被解释为简单的重测噪声、普遍熟悉化或整体信号衰减。第三，`kj` 的变化在四套 ROI 中都明显弱于 `yy`，且缺乏跨 ROI 的一致显著性；即便在专门为空间认知整理出来的 ROI 集中，`kj` 也没有呈现稳定的 pre→post 去相似化，而 atlas 层也只有少数左侧顶额 parcel 出现有限信号。这说明本研究中真正主导 Step 5C 结果的不是“所有学习材料都在各自优势网络内对称重组”，而是**隐喻学习诱发了更强、更跨网络、且跨 ROI 定义方案都可见的表征几何重塑**。

因此，第 4 节目前可以支持一个清晰而聚合的结论：无论是数据驱动得到的主功能 ROI、依据隐喻文献与空间认知文献分别预先定义的两套理论 ROI，还是更标准 atlas parcel 方案，结果都指向同一方向，即隐喻配对项目在学习后并没有普遍表现为“更相似”，反而更明显地表现为神经模式相似性的下降。这说明本研究中的隐喻学习更可能对应于**representational differentiation / reorganization**，而不是简单的配对聚合。换言之，学习并非只是让两个词在表征空间里机械地“靠近”，而更可能是在预存语义网络的基础上，把两者嵌入到一个更精细、区分度更高的新关系结构中；而 `kj` 在经典空间网络及 atlas 稳健性层中都缺乏稳定去相似化，也进一步表明这种重组并不是一个由 ROI 选择偏差造成的表面现象。

这也为后续 Model-RSA 提供了非常明确的方向性约束。到 Step 5D 时，核心问题已经不应再表述为“学习后同 pair 会不会更像”，而应转为：“这种在 `yy` 中最明显、且被 baseline 控制支持的 post 去相似化，到底更接近 differentiation 路径，还是某种建立在原有语义底盘之上的重加权重组？” 从这个意义上说，Step 5C 不只是一个前后测现象描述，而已经为整篇论文的机制解释部分规定了方向。

## 5、Model-RSA 主流程机制解释（Step 5D: `model_rdm_comparison.py`）

这一部分对应 Step 5D 的 pooled Model-RSA 结果。当前 `main_functional` 与 `literature` 两层 ROI 都使用 `analysis-mode=pooled`，因此这里的 neural RDM 是在每个 `subject × ROI × time` 单元内，把 `yy / kj / baseline` 三类 trial 合并后构造的一个 pooled neural RDM；结果表中的 `condition_group=all` 即表示这一 pooled 单元。统计上，`model_rdm_group_summary.tsv` 使用配对 t 检验比较 `post` 与 `pre` 的模型相关系数，其中 `mean_a = post`，`mean_b = pre`。因此，若某模型出现 `post < pre`，表示学习后该模型对神经几何结构的解释力减弱；若 `post > pre`，则表示该模型的解释力增强。

当前正式报告的模型包括：`M1_condition`（条件类别）、`M2_pair`（同 pair 接近）、`M3_embedding`（BERT 预训练语义距离）、`M7_memory`（被试回忆强度差异）和 `M8_reverse_pair`（同 pair 分化探针）。其中，`M2_pair` 与 `M8_reverse_pair` 是互补编码，因此在 `model_collinearity.tsv` 中它们是唯一稳定被标记为 `high` 的模型对；解释时应把二者视为同一条“pair 聚合 vs pair 分化”证据链的镜像，而不是两个相互独立的机制。

### 5.1 主功能 ROI（`main_functional`）结果

`main_functional` 层共 7 个 ROI。`M1_condition`、`M2_pair`、`M3_embedding` 和 `M8_reverse_pair` 均在 392 个 `subject × ROI × time` 单元中全部可估；`M7_memory` 因 `sub-12` 缺少记忆表，仅有 378 个有效单元。总体上，这一层的 Model-RSA 指向两条最清晰的机制信号。第一，在左侧楔前叶 ROI（`func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort`）中，`M2_pair` 从 pre 的 0.0060 下降到 post 的 0.0010，`t(27) = -3.06, p = .0049, dz = -0.58`；与之互补的 `M8_reverse_pair` 从 pre 的 -0.0060 上升到 post 的 -0.0010，`t(27) = 3.06, p = .0049, dz = 0.58`。这说明学习后该 ROI 的表征结构明显偏离了“同 pair 更接近”的组织方式，更符合 pair-based differentiation / reorganization，而不是简单的 pair convergence。第二，在左侧颞极 ROI（`func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort`）中，`M3_embedding` 从 pre 的 0.0132 下降到 post 的 -0.0061，`t(27) = -2.74, p = .0107, dz = -0.52`，提示学习后这里的神经几何不再主要按预训练语义距离组织，而是在原有语义底盘上发生了额外重组。

其余结果大多较弱。左侧另一枚颞极 ROI 的 `M1_condition` 仅边缘显著（post = -0.0011, pre = -0.0049, `p = .0517`），左后楔前叶 ROI 的 `M3_embedding` 也只有边缘效应（post = -0.0007, pre = 0.0151, `p = .0503`）。`M7_memory` 在 7 个主功能 ROI 中没有一个达到显著，说明当前主功能网络的前后重组并不能被简单归结为“记得更好所以几何结构更按回忆强度排列”。偏相关层面，五个模型的平均 partial rho 都接近 0（`M1 = -0.0035`, `M2 = 0.0069`, `M3 = 0.0035`, `M7 = -0.0029`, `M8 = -0.0069`），提示真正稳定的效应目前主要体现在少数关键 ROI，而不是一个全局性的 uniform shift。

主功能 ROI 的详细数值如下：

| ROI | Model | n | Post rho | Pre rho | t | p | dz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort | M1_condition | 28.0 | -0.0006 | -0.0021 | 0.71 | 0.485 | 0.13 |
| func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort | M2_pair | 28.0 | 0.0053 | 0.0081 | -1.45 | 0.157 | -0.27 |
| func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort | M3_embedding | 28.0 | -0.0061 | 0.0132 | -2.74 | 0.011 | -0.52 |
| func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort | M7_memory | 27.0 | -0.0028 | -0.0059 | 0.57 | 0.575 | 0.11 |
| func_Metaphor_gt_Spatial_c01_Temporal_Pole_HO_cort | M8_reverse_pair | 28.0 | -0.0053 | -0.0081 | 1.45 | 0.157 | 0.27 |
| func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort | M1_condition | 28.0 | -0.0011 | -0.0049 | 2.04 | 0.052 | 0.38 |
| func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort | M2_pair | 28.0 | 0.0056 | 0.0071 | -0.65 | 0.520 | -0.12 |
| func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort | M3_embedding | 28.0 | -0.0074 | 0.0000 | -1.20 | 0.240 | -0.23 |
| func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort | M7_memory | 27.0 | -0.0023 | -0.0013 | -0.19 | 0.847 | -0.04 |
| func_Metaphor_gt_Spatial_c02_Temporal_Pole_HO_cort | M8_reverse_pair | 28.0 | -0.0056 | -0.0071 | 0.65 | 0.520 | 0.12 |
| func_Spatial_gt_Metaphor_c01_Temporal_Fusiform_Cortex_posterior_division_HO_cort | M1_condition | 28.0 | -0.0039 | -0.0050 | 0.55 | 0.587 | 0.10 |
| func_Spatial_gt_Metaphor_c01_Temporal_Fusiform_Cortex_posterior_division_HO_cort | M2_pair | 28.0 | 0.0057 | 0.0078 | -1.04 | 0.306 | -0.20 |
| func_Spatial_gt_Metaphor_c01_Temporal_Fusiform_Cortex_posterior_division_HO_cort | M3_embedding | 28.0 | -0.0006 | 0.0107 | -1.54 | 0.135 | -0.29 |
| func_Spatial_gt_Metaphor_c01_Temporal_Fusiform_Cortex_posterior_division_HO_cort | M7_memory | 27.0 | 0.0005 | -0.0047 | 0.87 | 0.391 | 0.17 |
| func_Spatial_gt_Metaphor_c01_Temporal_Fusiform_Cortex_posterior_division_HO_cort | M8_reverse_pair | 28.0 | -0.0057 | -0.0078 | 1.04 | 0.306 | 0.20 |
| func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort | M1_condition | 28.0 | -0.0024 | -0.0037 | 0.51 | 0.618 | 0.10 |
| func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort | M2_pair | 28.0 | 0.0010 | 0.0060 | -3.06 | 0.005 | -0.58 |
| func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort | M3_embedding | 28.0 | -0.0008 | 0.0143 | -1.94 | 0.063 | -0.37 |
| func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort | M7_memory | 27.0 | -0.0023 | -0.0026 | 0.05 | 0.961 | 0.01 |
| func_Spatial_gt_Metaphor_c02_Precuneous_Cortex_HO_cort | M8_reverse_pair | 28.0 | -0.0010 | -0.0060 | 3.06 | 0.005 | 0.58 |
| func_Spatial_gt_Metaphor_c03_Precuneous_Cortex_HO_cort | M1_condition | 28.0 | -0.0030 | -0.0048 | 0.90 | 0.376 | 0.17 |
| func_Spatial_gt_Metaphor_c03_Precuneous_Cortex_HO_cort | M2_pair | 28.0 | 0.0014 | 0.0033 | -1.01 | 0.321 | -0.19 |
| func_Spatial_gt_Metaphor_c03_Precuneous_Cortex_HO_cort | M3_embedding | 28.0 | -0.0007 | 0.0151 | -2.05 | 0.050 | -0.39 |
| func_Spatial_gt_Metaphor_c03_Precuneous_Cortex_HO_cort | M7_memory | 27.0 | 0.0000 | -0.0044 | 0.74 | 0.467 | 0.14 |
| func_Spatial_gt_Metaphor_c03_Precuneous_Cortex_HO_cort | M8_reverse_pair | 28.0 | -0.0014 | -0.0033 | 1.01 | 0.321 | 0.19 |
| func_Spatial_gt_Metaphor_c04_Parahippocampal_Gyrus_posterior_division_HO_cort | M1_condition | 28.0 | -0.0043 | -0.0038 | -0.22 | 0.827 | -0.04 |
| func_Spatial_gt_Metaphor_c04_Parahippocampal_Gyrus_posterior_division_HO_cort | M2_pair | 28.0 | 0.0033 | 0.0056 | -0.89 | 0.383 | -0.17 |
| func_Spatial_gt_Metaphor_c04_Parahippocampal_Gyrus_posterior_division_HO_cort | M3_embedding | 28.0 | -0.0012 | 0.0063 | -1.12 | 0.274 | -0.21 |
| func_Spatial_gt_Metaphor_c04_Parahippocampal_Gyrus_posterior_division_HO_cort | M7_memory | 27.0 | -0.0039 | 0.0016 | -1.04 | 0.307 | -0.20 |
| func_Spatial_gt_Metaphor_c04_Parahippocampal_Gyrus_posterior_division_HO_cort | M8_reverse_pair | 28.0 | -0.0033 | -0.0056 | 0.89 | 0.383 | 0.17 |
| func_Spatial_gt_Metaphor_c05_Lateral_Occipital_Cortex_superior_division_HO_cort | M1_condition | 28.0 | -0.0020 | -0.0033 | 0.56 | 0.579 | 0.11 |
| func_Spatial_gt_Metaphor_c05_Lateral_Occipital_Cortex_superior_division_HO_cort | M2_pair | 28.0 | 0.0046 | 0.0051 | -0.21 | 0.839 | -0.04 |
| func_Spatial_gt_Metaphor_c05_Lateral_Occipital_Cortex_superior_division_HO_cort | M3_embedding | 28.0 | 0.0059 | 0.0027 | 0.48 | 0.632 | 0.09 |
| func_Spatial_gt_Metaphor_c05_Lateral_Occipital_Cortex_superior_division_HO_cort | M7_memory | 27.0 | -0.0076 | -0.0053 | -0.54 | 0.596 | -0.10 |
| func_Spatial_gt_Metaphor_c05_Lateral_Occipital_Cortex_superior_division_HO_cort | M8_reverse_pair | 28.0 | -0.0046 | -0.0051 | 0.21 | 0.839 | 0.04 |

### 5.2 论文 ROI（`literature`）结果

`literature` 层共 12 个理论 ROI。与 `main_functional` 类似，`M1`、`M2`、`M3` 和 `M8` 均在 672 个 pooled 单元中全部可估；`M7` 因 `sub-12` 缺失回忆表，在 648 个单元中有效。与主功能 ROI 相比，这一层的 Model-RSA 故事更收敛：绝大多数 ROI 和模型都没有表现出强而稳定的 pre-post 改变，唯一明确达到显著的效应出现在左 IFG。具体而言，`lit_L_IFG` 中 `M2_pair` 从 pre 的 0.0098 下降到 post 的 0.0033，`t(27) = -3.47, p = .0018, dz = -0.65`；其镜像模型 `M8_reverse_pair` 从 pre 的 -0.0098 上升到 post 的 -0.0033，`t(27) = 3.47, p = .0018, dz = 0.65`。这与主功能 ROI 左 Precuneous 的结果方向完全一致，说明至少在一个经典语义控制节点（L IFG）中，学习后神经几何同样脱离了“同 pair 更相似”的组织方式，更接近 pair-based differentiation。

除左 IFG 外，其余结果大多停留在趋势水平。右 IFG 的 `M2/M8`（`p = .081`）、右 pSTS 的 `M2/M8`（`p = .096`）以及 `R_pSTS` 的 `M1_condition`（`p = .058`）都提示学习后可能存在弱的 pair-structure 衰减或条件结构增强，但这些趋势尚不足以构成正式主结论。与 `main_functional` 不同，这一层并没有再出现一个明确的 `M3_embedding` 减弱 ROI；预训练语义模型在大多数文献 ROI 中前后都较稳定。`M7_memory` 仍然没有在任何文献 ROI 中达到显著，说明回忆强度差异暂时还不是解释 pooled neural geometry 重组的主导变量。偏相关层面，五个模型的平均 partial rho 也都接近 0（`M1 = -0.0030`, `M2 = 0.0093`, `M3 = -0.0005`, `M7 = -0.0025`, `M8 = -0.0093`），说明在控制其他模型后，最稳定保留下来的独立信号仍主要集中在 pair-model 这一条线上。

论文 ROI 的详细数值如下：

| ROI | Model | n | Post rho | Pre rho | t | p | dz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lit_L_AG | M1_condition | 28.0 | -0.0038 | -0.0025 | -0.62 | 0.541 | -0.12 |
| lit_L_AG | M2_pair | 28.0 | 0.0045 | 0.0067 | -1.25 | 0.221 | -0.24 |
| lit_L_AG | M3_embedding | 28.0 | 0.0048 | 0.0010 | 0.54 | 0.596 | 0.10 |
| lit_L_AG | M7_memory | 27.0 | 0.0010 | -0.0019 | 0.79 | 0.434 | 0.15 |
| lit_L_AG | M8_reverse_pair | 28.0 | -0.0045 | -0.0067 | 1.25 | 0.221 | 0.24 |
| lit_L_IFG | M1_condition | 28.0 | -0.0040 | -0.0039 | -0.06 | 0.953 | -0.01 |
| lit_L_IFG | M2_pair | 28.0 | 0.0033 | 0.0098 | -3.47 | 0.002 | -0.65 |
| lit_L_IFG | M3_embedding | 28.0 | 0.0041 | 0.0019 | 0.32 | 0.749 | 0.06 |
| lit_L_IFG | M7_memory | 27.0 | -0.0036 | -0.0009 | -0.56 | 0.580 | -0.11 |
| lit_L_IFG | M8_reverse_pair | 28.0 | -0.0033 | -0.0098 | 3.47 | 0.002 | 0.65 |
| lit_L_pMTG | M1_condition | 28.0 | -0.0028 | -0.0029 | 0.06 | 0.950 | 0.01 |
| lit_L_pMTG | M2_pair | 28.0 | 0.0087 | 0.0091 | -0.20 | 0.839 | -0.04 |
| lit_L_pMTG | M3_embedding | 28.0 | -0.0032 | 0.0022 | -1.00 | 0.327 | -0.19 |
| lit_L_pMTG | M7_memory | 27.0 | -0.0028 | -0.0022 | -0.10 | 0.925 | -0.02 |
| lit_L_pMTG | M8_reverse_pair | 28.0 | -0.0087 | -0.0091 | 0.20 | 0.839 | 0.04 |
| lit_L_precuneus | M1_condition | 28.0 | -0.0027 | -0.0017 | -0.46 | 0.648 | -0.09 |
| lit_L_precuneus | M2_pair | 28.0 | 0.0041 | 0.0058 | -0.94 | 0.355 | -0.18 |
| lit_L_precuneus | M3_embedding | 28.0 | 0.0004 | 0.0087 | -0.99 | 0.330 | -0.19 |
| lit_L_precuneus | M7_memory | 27.0 | -0.0043 | -0.0026 | -0.35 | 0.729 | -0.07 |
| lit_L_precuneus | M8_reverse_pair | 28.0 | -0.0041 | -0.0058 | 0.94 | 0.355 | 0.18 |
| lit_L_pSTS | M1_condition | 28.0 | -0.0004 | -0.0033 | 1.25 | 0.222 | 0.24 |
| lit_L_pSTS | M2_pair | 28.0 | 0.0040 | 0.0049 | -0.47 | 0.641 | -0.09 |
| lit_L_pSTS | M3_embedding | 28.0 | 0.0003 | 0.0003 | 0.00 | 0.998 | 0.00 |
| lit_L_pSTS | M7_memory | 27.0 | -0.0037 | -0.0002 | -0.82 | 0.419 | -0.16 |
| lit_L_pSTS | M8_reverse_pair | 28.0 | -0.0040 | -0.0049 | 0.47 | 0.641 | 0.09 |
| lit_L_temporal_pole | M1_condition | 28.0 | -0.0030 | -0.0038 | 0.33 | 0.745 | 0.06 |
| lit_L_temporal_pole | M2_pair | 28.0 | 0.0088 | 0.0095 | -0.32 | 0.750 | -0.06 |
| lit_L_temporal_pole | M3_embedding | 28.0 | 0.0021 | 0.0034 | -0.23 | 0.817 | -0.04 |
| lit_L_temporal_pole | M7_memory | 27.0 | -0.0042 | -0.0006 | -0.68 | 0.500 | -0.13 |
| lit_L_temporal_pole | M8_reverse_pair | 28.0 | -0.0088 | -0.0095 | 0.32 | 0.750 | 0.06 |
| lit_R_AG | M1_condition | 28.0 | -0.0031 | -0.0028 | -0.13 | 0.901 | -0.02 |
| lit_R_AG | M2_pair | 28.0 | 0.0043 | 0.0064 | -0.80 | 0.431 | -0.15 |
| lit_R_AG | M3_embedding | 28.0 | 0.0008 | -0.0034 | 0.78 | 0.443 | 0.15 |
| lit_R_AG | M7_memory | 27.0 | -0.0013 | -0.0054 | 0.86 | 0.399 | 0.16 |
| lit_R_AG | M8_reverse_pair | 28.0 | -0.0043 | -0.0064 | 0.80 | 0.431 | 0.15 |
| lit_R_IFG | M1_condition | 28.0 | -0.0009 | -0.0037 | 1.37 | 0.180 | 0.26 |
| lit_R_IFG | M2_pair | 28.0 | 0.0078 | 0.0108 | -1.81 | 0.081 | -0.34 |
| lit_R_IFG | M3_embedding | 28.0 | 0.0000 | -0.0008 | 0.13 | 0.901 | 0.02 |
| lit_R_IFG | M7_memory | 27.0 | -0.0042 | -0.0020 | -0.40 | 0.690 | -0.08 |
| lit_R_IFG | M8_reverse_pair | 28.0 | -0.0078 | -0.0108 | 1.81 | 0.081 | 0.34 |
| lit_R_pMTG | M1_condition | 28.0 | -0.0026 | -0.0031 | 0.23 | 0.818 | 0.04 |
| lit_R_pMTG | M2_pair | 28.0 | 0.0081 | 0.0111 | -1.66 | 0.108 | -0.31 |
| lit_R_pMTG | M3_embedding | 28.0 | -0.0064 | -0.0041 | -0.36 | 0.721 | -0.07 |
| lit_R_pMTG | M7_memory | 27.0 | -0.0081 | 0.0008 | -1.69 | 0.104 | -0.32 |
| lit_R_pMTG | M8_reverse_pair | 28.0 | -0.0081 | -0.0111 | 1.66 | 0.108 | 0.31 |
| lit_R_precuneus | M1_condition | 28.0 | -0.0040 | -0.0027 | -0.64 | 0.526 | -0.12 |
| lit_R_precuneus | M2_pair | 28.0 | 0.0030 | 0.0046 | -0.68 | 0.500 | -0.13 |
| lit_R_precuneus | M3_embedding | 28.0 | 0.0025 | 0.0092 | -0.82 | 0.420 | -0.15 |
| lit_R_precuneus | M7_memory | 27.0 | -0.0031 | -0.0065 | 0.65 | 0.519 | 0.13 |
| lit_R_precuneus | M8_reverse_pair | 28.0 | -0.0030 | -0.0046 | 0.68 | 0.500 | 0.13 |
| lit_R_pSTS | M1_condition | 28.0 | 0.0004 | -0.0037 | 1.98 | 0.058 | 0.37 |
| lit_R_pSTS | M2_pair | 28.0 | 0.0053 | 0.0083 | -1.72 | 0.096 | -0.33 |
| lit_R_pSTS | M3_embedding | 28.0 | -0.0019 | 0.0013 | -0.51 | 0.615 | -0.10 |
| lit_R_pSTS | M7_memory | 27.0 | 0.0004 | 0.0013 | -0.19 | 0.850 | -0.04 |
| lit_R_pSTS | M8_reverse_pair | 28.0 | -0.0053 | -0.0083 | 1.72 | 0.096 | 0.33 |
| lit_R_temporal_pole | M1_condition | 28.0 | 0.0003 | -0.0017 | 0.96 | 0.347 | 0.18 |
| lit_R_temporal_pole | M2_pair | 28.0 | 0.0090 | 0.0096 | -0.35 | 0.731 | -0.07 |
| lit_R_temporal_pole | M3_embedding | 28.0 | 0.0039 | 0.0003 | 0.66 | 0.514 | 0.12 |
| lit_R_temporal_pole | M7_memory | 27.0 | -0.0033 | -0.0006 | -0.51 | 0.615 | -0.10 |
| lit_R_temporal_pole | M8_reverse_pair | 28.0 | -0.0090 | -0.0096 | 0.35 | 0.731 | 0.07 |

### 5.3 跨 ROI 的 Δρ LMM（`delta_rho_lmm.py`）

为了进一步检验“哪一种模型的前后变化最稳定”，我们在 `model_rdm_subject_metrics.tsv` 基础上，对每个 `subject × ROI × model` 先计算 `Δρ = ρ_post − ρ_pre`，再在 ROI 层面拟合 `delta_rho ~ C(model) + (1|subject) + (1|roi)`。这里 `M1_condition` 为参考模型，因此截距表示 `M1` 的平均 `Δρ`，其他系数表示对应模型相对 `M1` 的额外变化量。若系数为负，表示该模型比 `M1` 更倾向于在 post 阶段下降；若系数为正，则表示它比 `M1` 更倾向于上升。

在 `main_functional` 层，Δρ LMM 使用 28 名被试、7 个 ROI、973 个观测，模型收敛良好。原始均值显示：`M1 = +0.00147`，`M2 = -0.00229`，`M3 = -0.01043`，`M7 = +0.00064`，`M8 = +0.00229`。其中，只有 `M3_embedding` 相对于 `M1_condition` 的额外下降达到显著，`b = -0.0119, SE = 0.0022, z = -5.43, p < .001, 95% CI [-0.016, -0.008]`；`M2_pair` 也呈边缘显著的更负变化，`b = -0.0038, SE = 0.0022, z = -1.71, p = .087`。这说明在主功能 ROI 集中，若从“跨 ROI 的平均模型轨迹”来比较，最稳定下行的其实不是条件模型，而是**预训练语义模型 `M3`**；pair-based 模型则表现出一个较弱但方向一致的“向分化侧偏移”的趋势。

在 `literature` 层，Δρ LMM 使用 28 名被试、12 个 ROI、1668 个观测。原始均值为：`M1 = +0.00076`，`M2 = -0.00214`，`M3 = -0.00105`，`M7 = -0.00139`，`M8 = +0.00214`。这里最接近显著的是 `M2_pair` 相对 `M1` 的更负变化，`b = -0.0029, SE = 0.0016, z = -1.85, p = .065, 95% CI [-0.006, 0.000]`；其余 `M3`、`M7` 与 `M8` 都未显著。不过，这一层的 MixedLM 标记为 `converged = false`，因此更稳妥的表述应是：**文献 ROI 层支持一个 pair-model 的趋势性证据，但不足以单独构成强主结论**。

Δρ LMM 的详细系数如下：

| ROI set | n_obs | n_roi | Model term | Estimate | SE | z | p | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_functional | 973 | 7 | Intercept (`M1_condition`) | 0.0015 | 0.0018 | 0.83 | 0.407 | [-0.002, 0.005] |
| main_functional | 973 | 7 | `C(model)[T.M2_pair]` | -0.0038 | 0.0022 | -1.71 | 0.087 | [-0.008, 0.001] |
| main_functional | 973 | 7 | `C(model)[T.M3_embedding]` | -0.0119 | 0.0022 | -5.43 | <0.001 | [-0.016, -0.008] |
| main_functional | 973 | 7 | `C(model)[T.M7_memory]` | -0.0007 | 0.0022 | -0.31 | 0.753 | [-0.005, 0.004] |
| main_functional | 973 | 7 | `C(model)[T.M8_reverse_pair]` | 0.0008 | 0.0022 | 0.37 | 0.709 | [-0.003, 0.005] |
| literature | 1668 | 12 | Intercept (`M1_condition`) | 0.0008 | 0.0013 | 0.58 | 0.563 | [-0.002, 0.003] |
| literature | 1668 | 12 | `C(model)[T.M2_pair]` | -0.0029 | 0.0016 | -1.85 | 0.065 | [-0.006, 0.000] |
| literature | 1668 | 12 | `C(model)[T.M3_embedding]` | -0.0018 | 0.0016 | -1.15 | 0.249 | [-0.005, 0.001] |
| literature | 1668 | 12 | `C(model)[T.M7_memory]` | -0.0021 | 0.0016 | -1.33 | 0.183 | [-0.005, 0.001] |
| literature | 1668 | 12 | `C(model)[T.M8_reverse_pair]` | 0.0014 | 0.0016 | 0.88 | 0.379 | [-0.002, 0.004] |

### 5.4 `main_functional` 与 `literature` 的整合结论

把 ROI 级结果和跨 ROI 的 Δρ LMM 合在一起看，Step 5D 当前已经给出一条比 Step 5C 更细的机制约束。第一，真正稳定重复的不是 `M1_condition` 或 `M7_memory`，而是 pair-based 模型与语义模型的组合效应：在主功能 ROI 中，左 Precuneous 出现了显著的 `M2_pair` 下降与 `M8_reverse_pair` 上升；在文献 ROI 中，左 IFG 出现了同方向、甚至更强的 `M2/M8` 效应。与此同时，跨 ROI 的 Δρ LMM 又表明，在 `main_functional` 层真正最稳定下行的是 `M3_embedding`，而在 `literature` 层 `M2_pair` 只呈趋势性更负变化。这说明学习后最稳的变化并不是“神经几何更按条件类别组织”，也不是“更按回忆强弱排序”，而是**一方面偏离同 pair 聚合，另一方面在部分语义枢纽中偏离原有预训练语义几何**。

第二，预训练语义模型 `M3_embedding` 只在主功能 ROI 的左侧颞极出现了明确的 ROI 级下降，而 Δρ LMM 又显示它在主功能 ROI 层的跨区平均下降最为稳定。这一点为 Step 5C 中观察到的 `yy` 去相似化提供了一个更细的机制入口，即隐喻学习可能并不是把两个词简单压到一起，而是在原有语义结构上重构关系几何。第三，记忆强度模型 `M7_memory` 在 ROI 级和 Δρ LMM 两层中都没有形成稳定显著模式，说明当前行为收益虽存在，但还不能简单解释为“记得越好，表征几何就越按回忆分数组织”。

因此，结合第 4 节的 Step 5C 结果，当前最稳妥的总故事已经比较清晰：`yy` 在多个 ROI 集中表现为学习后 pair similarity 下降；Step 5D 进一步说明，这种下降并不伴随稳定的 `M2_pair` 增强，反而在左 Precuneous 和左 IFG 中表现为 `M2` 下降、`M8` 上升，说明 neural geometry 正在偏离“pair convergence”而转向更强的 differentiation / reorganization。与此同时，左侧颞极的 `M3` 减弱，以及 `main_functional` 跨 ROI 层面 `M3` 的显著负向 Δρ，都提示学习后表征也不再完全受预存语义结构支配，而是在语义底盘之上引入了新的关系约束。换句话说，当前 `main_functional` 与 `literature` 两层 ROI 的 Model-RSA 汇总后，更支持这样一种机制解释：**隐喻学习引发的不是简单的配对聚合，而是沿着 pair-based differentiation 与 semantic reweighting 共同展开的表征重组。**

## 6、主文固定口径（当前版本）

这一节不是新增结果，而是把当前最稳妥、最适合放进主文的写法固定下来，避免后续表述来回摇摆。

### 6.1 核心主张

主文的 primary claim 建议固定为：

- 相比空间联结学习，隐喻联结学习诱发了更强的学习后表征重组。
- 这种重组在当前数据中主要表现为 `yy` 的 pair similarity 从 pre 到 post 更明显下降。
- 该模式更支持 **representational differentiation / reorganization**，而不是简单的 pair convergence。
- 在机制层面，当前结果更支持：
  - pair-based differentiation
  - semantic reweighting

### 6.2 结果段固定写法

主结果段建议优先这样写：

> 行为结果显示，被试对 `yy` 条件的学习表现优于 `kj`。在三条件 item-wise RSA 中，`yy` 在多个 ROI 集内均表现出比 `kj` 和 `baseline` 更明显的 pre-to-post similarity 下降，说明隐喻学习后的神经表征变化并未表现为简单的配对聚合，而更接近表征分化增强。进一步的 Model-RSA 显示，这种变化在左 Precuneous 与左 IFG 更符合 pair-based differentiation，在左 temporal pole 更符合 semantic reweighting。整体而言，隐喻学习引发的更可能是学习后表征几何的重组，而不是 pair members 的简单收敛。

### 6.3 讨论段固定写法

讨论段建议优先这样写：

> 当前结果表明，隐喻学习相较空间联结学习，更可能诱发学习后神经表征的重组。重要的是，这种变化并未表现为同 pair 项目在 post 阶段变得更相似，而是表现为 `yy` 的 pair similarity 更明显下降，并在部分 ROI 中伴随 pair-based 模型减弱与预训练语义模型减弱。这提示隐喻学习可能不是把两个词条简单压缩为一个更接近的联合表征，而是在原有语义结构之上形成了更精细、区分度更高的新关系几何。与此同时，即使把分析范围进一步限制到经典空间网络 ROI，`kj` 也没有表现出跨 ROI、一致、可靠的 pre/post 去相似化；因此，当前 `yy` 效应并不能简单归因为 ROI 选择偏向语义网络，而更像是一种对隐喻学习更敏感、跨网络可见的表征重组。因而，本研究更支持隐喻学习体现为 representational differentiation / reorganization，而不是简单的 pair convergence。

### 6.4 必须保留的克制表述

当前主文应明确保留以下边界：

- 不把结果写成“已经证明隐喻学习失败所以没有收敛”
- 不把结果写成“已经完全排除所有质量解释”
- 不把 `literature` 层 `delta_rho_lmm` 写成强裁决，因为该层 MixedLM 目前 `converged = false`
- 不把 `M7_memory` 写成当前主导机制，因为它尚未形成稳定显著模式
- 不把空间 ROI 的阴性结果写成“空间学习没有任何神经变化”；更准确的说法是：`kj` 缺乏跨 ROI、一致、可靠的去相似化模式

QC / reliability 建议固定写法为：

> voxel split-half reliability 没有显示 post 阶段整体 reliability 崩坏，因此当前结果不支持“`yy` 的下降只是整体测量不稳定”这一解释；不过，motion/QC 指标在 post 阶段确实略差，且与 `yy` 的变化存在相关，因此更稳妥的表述是：质量因素可能影响效应幅度，但不足以单独解释主结果方向。

### 6.5 不再建议使用的表述

后续主文里尽量避免以下说法：

- “学习后同 pair 会不会更像”
- “隐喻学习后发生了更强的 convergence”
- “结果说明隐喻没有真正学会”
- “记忆更好但神经相似性下降是矛盾的”

更推荐替换为：

- “学习后表征重组是否更强”
- “是否出现 pair-based differentiation”
- “是否存在 semantic reweighting”
- “行为收益与表征分化并不矛盾，二者可能共同反映更精细的关系编码”

## 7、主文图题与图注（当前版本）

这一节固定当前主文 5 张核心图的标题与图注写法，对应的图片文件位于 `E:\python_metaphor\figures_main_story\`。

### Figure 1

**图题**：Behavioral advantage for metaphor learning over spatial learning

**图注**：被试在 `YY` 条件下表现出更优的学习收益。左图显示 run-7 记忆正确率，`YY` 的平均正确率高于 `KJ`；右图显示仅在答对 trial 上计算的反应时，`YY` 在正确提取时略快于 `KJ`。两幅面板均采用 violin-first 展示：小提琴轮廓表示被试分布，浅灰连线表示同一被试在两条件之间的配对变化，彩色点表示单被试取值，白色箱体表示分位区间，黑点和误差线表示组均值及其标准差（SD）。该图对应 refined behavior analysis 的两个正式终点：`memory` 与 `log_rt_correct`。

![Figure 1](/E:/python_metaphor/figures_main_story/fig_behavior_accuracy_rt.png)

### Figure 2

**图题**：Step 5C item-wise RSA reveals stronger post-learning de-similarity for metaphor pairs

**图注**：左面板给出主功能 ROI 层面的 pooled Step 5C similarity 模式，采用 split-violin 展示 `Pre` 与 `Post` 在三类条件中的分布形态，显示 `Metaphor` 条件从 pre 到 post 的 pair similarity 下降最明显，而 `Spatial` 与 `Baseline` 变化较弱。右面板显示各主功能 ROI 中 `Post - Pre` 的 delta similarity；负值表示学习后 pair similarity 下降，且 `Metaphor` 在全部主功能 ROI 中都呈现更强的负向变化。点位表示组均值，误差线表示 ROI 内跨被试标准差（SD）。该图用于呈现当前最核心的现象结果：隐喻学习后的神经表征变化更接近去相似化，而不是 pair convergence。

![Figure 2](/E:/python_metaphor/figures_main_story/fig_step5c_main.png)

### Figure 3

**图题**：Model-RSA constrains the mechanism to pair-based differentiation and semantic reweighting

**图注**：图中颜色表示各 ROI 内模型相关系数的 `Post - Pre` 变化量（`Δρ`），蓝色表示 post 相对 pre 下降，红色表示 post 相对 pre 上升；星号标记 ROI 级配对比较达到显著的结果。上面板为 `main_functional`，下面板为 `literature`。当前最关键的模式是：左侧 precuneus / IFG 的 `M2_pair` 下降并伴随 `M8_reverse_pair` 上升，说明学习后表征偏离同 pair 聚合；左侧 temporal pole 的 `M3_embedding` 下降，则提示学习后表征不再主要按预训练语义距离组织。整体上，该图支持的不是 simple pair convergence，而是 `pair-based differentiation + semantic reweighting`。

![Figure 3](/E:/python_metaphor/figures_main_story/fig_model_rsa_mechanism.png)

### Figure 4

**图题**：The Step 5C metaphor effect is robust across ROI-definition schemes

**图注**：该图汇总四套 ROI 定义方案下的 `Post - Pre` delta similarity，包括 `main_functional`、`literature`、`literature_spatial` 与 `atlas_robustness`。三个面板分别对应 `Metaphor`、`Spatial` 与 `Baseline`；每个面板均采用 violin-first 展示，小提琴表示被试分布，半透明小点表示被试层面的 ROI-set 平均 delta similarity，白色箱体表示分位区间，黑点和误差线表示组均值及其标准差（SD）。在四套 ROI 中，`Metaphor` 都稳定表现为更负的 delta similarity，而 `Spatial` 与 `Baseline` 缺乏同等强度、跨 ROI 一致的下降。该图用于回答稳健性问题：当前主结论并不依赖某一种 ROI 选择方式，同时保留了被试间分布信息，而不仅仅是均值不确定性。

![Figure 4](/E:/python_metaphor/figures_main_story/fig_roi_set_robustness.png)

### Figure 5

**图题**：Activation strength and representational reorganization are dissociable

**图注**：该图不再使用散点关系，而是直接在同一 ROI 内比较两类指标：灰蓝色点表示学习期激活强度，红色点表示 `YY` 的表征重组强度（定义为 `Pre - Post`，数值越大表示学习后去相似化越强），两点之间的连线表示同一 ROI 在两种指标上的相对位置差异。为了便于跨指标比较，横轴使用各指标内部标准化后的 z 值。图中可见，某些 ROI 在激活上较高，但在表征重组上并不等幅增强；另一些 ROI 则表现出更强的表征重组而并无同等强的激活增益。该图用于更直观地支持“学习期激活增强”与“学习后表征重塑增强”并非同一件事。

![Figure 5](/E:/python_metaphor/figures_main_story/fig_activation_representation_dissociation.png)

