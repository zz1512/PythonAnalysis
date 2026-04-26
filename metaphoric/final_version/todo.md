# 当前阶段总体分析方案

## 一、当前总判断

当前主线已经基本成立，不需要重构。

现阶段最稳的证据链是：
- 行为上，`yy` 学得更好
- Step 5C：`yy` 的 pre->post similarity 下降更明显，且 `baseline / kj` 不支持“普遍熟悉化”解释
- Step 5D ROI 级：左 Precuneous、左 IFG 更支持 `pair-based differentiation`，左 temporal pole 更支持 `semantic reweighting`
- Step 5D 跨 ROI：`main_functional` 中 `M3_embedding` 的负向 `Δρ` 最稳定；`literature` 中 `M2_pair` 只有趋势
- QC / reliability：不支持“post 阶段整体 reliability 崩坏”，但 motion/QC 可能调制效应幅度

因此，后续工作不应再以“继续找新结果”为主，而应改成：
- 固定主文故事
- 补最小行为与稳健性证据
- 再决定是否做网络层扩展

---

## 二、当前主故事

当前论文主故事固定为：

1. 相比空间联结学习，隐喻联结学习引发更强的学习后表征重组。
2. 这种重组在当前数据中主要表现为 `yy` 的 pair similarity 从 pre 到 post 更明显下降。
3. 这种变化不支持简单的 pair convergence，更支持：
   - `pair-based differentiation`
   - `semantic reweighting`
4. 记忆强度模型 `M7_memory` 当前不是主导解释。
5. QC 结果提示 motion/QC 可能影响效应幅度，但不足以把主结果解释成整体测量崩坏。

写作上应避免：
- “学习后同 pair 更像”
- “结果证明隐喻学习失败所以没有收敛”
- “已经完全排除所有质量解释”
- “literature` 层 `delta_rho_lmm` 已经给出强裁决”

---

## 三、工作优先级

## P0：现在立刻做

- [x] 固定论文主文口径
  - 把主现象固定写成：`yy` 学习后出现更强的 pre->post similarity 下降
  - 把机制解释固定写成：`pair-based differentiation + semantic reweighting`
  - 明确 `literature` 层 `delta_rho_lmm` 仅为趋势支持，不写成强裁决
  - 明确 QC 结论写法：reliability 不支持整体测量崩坏，但 motion/QC 可能调制效应幅度

- [x] 细化行为分析
  - 目标不是增加很多行为表，而是把神经主结果和学习收益连接起来
  - 先固定 1 个主行为指标：`memory`
  - 再固定 1 个补充指标：`log_rt_correct`
  - 避免同时铺开太多行为指标
  - 统一使用 `behavior_analysis/behavior_refined.py`
  - 输出重点：
    - `subject_summary_wide.tsv`
    - `behavior_primary_diff_yy_minus_kj`
    - `rt_correct_diff_yy_minus_kj`

- [x] 做一个最小脑-行为主分析
  - 只固定 1 个行为指标和 1 个神经指标
  - 优先检验：更强的 `yy` 去相似化是否预测更好的学习/记忆表现
  - 优先从主 ROI 或文献 ROI 的核心指标出发，不一开始就全铺开

## P1：主文加固

- [x] 做 trial 数 / 体素数稳健性分析
  - 对 `yy / kj / baseline` 做等量重采样或控制 trial 数差异
  - 检查 `n_voxels` 在 pre/post 是否系统偏向某条件或某 ROI
  - 形成可放 SI 的一键复现结果

- [x] 视精力补一层 `atlas_robustness`
  - 不重写主故事
  - 只回答：主结论是否依赖当前 ROI 定义

- [x] 把空间 ROI 的负结果写进讨论定稿
  - 明确：即使在经典空间网络中，`kj` 也未表现出稳定的 pre/post 去相似化
  - 用来支持：`yy` 的效应不是 ROI 选择偏差造成

- [x] 固定“激活-表征解耦”图
  - 用来呈现：学习期激活更强，不等于学习后表征重塑更强
  - 这会是主文里很有价值的一张整合图

- [x] 补齐主文关键图示
  - 目标不是把所有结果都画出来，而是固定 4-5 张真正承载故事的主图
  - 当前建议的主文图顺序：
    - Figure 1：行为结果图
      - `YY` vs `KJ` 的 `memory`
      - `YY` vs `KJ` 的 `log_rt_correct`
      - 适合用 paired dot/line + mean/CI
    - Figure 2：Step 5C 主结果图
      - 主功能 ROI 的 `Pre/Post × condition`
      - 必要时配一个 pooled summary 或 family summary
      - 这是最核心的一张 RSA 现象图
    - Figure 3：Model-RSA 机制图
      - `main_functional + literature` 的关键 ROI × model 结果
      - 优先突出 `M2/M8` 与 `M3`
      - 更适合热图或点图，而不是铺满所有表格
    - Figure 4：四层 ROI 稳健性图
      - `main_functional / literature / literature_spatial / atlas_robustness`
      - 用来回答：主结论不依赖 ROI 定义
    - Figure 5：激活-表征解耦图
      - 呈现学习期激活与学习后表征重塑并非同一件事
  - 当前不建议进主文的图：
    - QC / reliability 全套图
    - trial/voxel 平衡性全套图
    - 脑-行为阴性相关大图
  - 这些更适合做补充材料（SI）

## P2：可以做，但不抢主线

- [ ] RD / GPS 作为补充几何证据
  - 只在主文主线稳定后再推进
  - 作为补充而不是替代当前 RSA/Model-RSA 主结果

- [ ] 偏侧化 / 熟悉度梯度
  - 用于和经典隐喻文献对话
  - 适合主故事已经写稳之后再上

- [ ] 补充材料图（SI）整理
  - QC / reliability
  - trial / voxel / atlas robustness
  - brain-behavior null result
  - 目标是支持主结论，而不是在主文里分散注意力

## P3：下一阶段扩展

- [ ] 脑网络分析
  - 可考虑：
    - gPPI
    - beta-series connectivity
    - effective connectivity
  - 这条线有理论价值，但当前不应抢主线
  - 更适合作为主结果稳定后的机制扩展

- [ ] dynamics / mediation
  - dynamics
  - mediation
  - 必须明确探索性声明与多重比较策略

---

## 四、老师新建议的处理原则

### 1. 细化行为分析

结论：有必要，而且适合现在做。

原因：
- 当前神经主结果已经清楚
- 行为层如果只停在“`yy` 记得更好”，证据链会偏短
- 行为细化最能帮助回答：神经上的分化增强，到底对应哪一种学习收益

执行原则：
- 少而精
- 先主指标，再补充指标
- 优先做脑-行为耦合，而不是只堆更多行为表

### 2. 扩展脑网络分析

结论：有价值，但不适合现在立刻升成主线。

原因：
- 网络分析和当前理论是匹配的
- 但它的方法自由度更高，也更容易把叙事带散
- 当前阶段应先把主终点、主机制和脑-行为证据链固定好

执行原则：
- 作为下一阶段机制扩展
- 不作为现在最优先任务
- 只有在主线写稳后再推进

---

## 五、建议执行顺序

1. 先固定论文故事、结果图和讨论口径
2. 细化行为分析，并做一个最小脑-行为主分析
3. 补 trial/voxel 与 atlas 稳健性
4. 固定“激活-表征解耦”图
5. 最后再决定是否上 RD/GPS、偏侧化、熟悉度梯度
6. 脑网络分析放到下一阶段

---

## 六、当前不建议优先推进

- [ ] 不优先做新的大规模全脑探索
- [ ] 不优先再扩很多新 ROI 集
- [ ] 不优先把 connectivity / mediation 抬成当前主结果
- [ ] 不优先同时铺开很多行为指标和很多神经指标
