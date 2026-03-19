# emo_ultra 显著脑区少于 emo 的潜在原因排查

> 现象：理论预期 `emo < emo_ultra < emo_new`（敏感性），但实际观察到 `emo_ultra` 显著脑区反而少于 `emo`。

## 一、先给结论（按影响优先级）

1. **统计功效下降（最可能）**：`emo_ultra` 是按 `stimulus_type` 分桶分别跑检验，每个桶内被试数会下降，且每个桶都做 FWER，显著数很容易减少。  
2. **信号被“去均值 + 表征化”后变弱**：`emo_ultra` Step1 默认启用 cocktail-blank（跨刺激去均值）并先构建 RSM，再做 ISC；这会压掉均值幅度类信号（而这类信号在 `emo` 中可能贡献了显著）。  
3. **多重比较负担更重**：`emo_ultra` 在 232 ROI 上做 model-wise FWER；相对 `emo_new` 的 200 ROI，阈值更严格。  
4. **样本筛选更严格**：`emo_ultra` 要求被试同时具备 EMO+SOC，外加每个 stimulus_type 内 80% 覆盖+全刺激完整，被试损失更大。  
5. **条件划分差异导致“分箱稀释”**：`emo_ultra` 默认按 `raw_emotion`（或 label 前缀）拆分条件；如果分类更细，每个条件有效样本更少。

---

## 二、代码层面的关键差异

### 1) `emo_ultra` 是“按刺激类型分开检验”

- `emo_ultra` Step1/2/3 全部按 `by_stimulus/<stimulus_type>/` 分目录处理，每个 stimulus_type 分开建模与置换。  
- 对比 `emo` 的旧链路是单一矩阵（全刺激合并）后再统一检验。

影响：
- 每个 stimulus_type 的 `n_subjects` 可能显著低于全量。
- 同样 5000 次置换下，`n_sub` 变小会直接降低统计功效。
- 你看到“显著 ROI 少”很可能只是“每个桶都太小”。

### 2) `emo_ultra` 默认启用 cocktail blank + RSM 路线

- `emo_ultra` 的 `spearman_rsm()` 在 ROI 内特征上默认执行 `X = X - X.mean(axis=0)`（cocktail blank）。
- 然后不是直接拿 beta 向量做被试相关，而是先做每被试 `stim×stim` RSM，再展开上三角做被试间 ISC。

影响：
- 去掉了很多“整体激活强度”信息；如果你的年龄效应主要体现在幅度层面，会明显掉检出率。
- RSM 二次相关路线通常更稳健但也更“保守”。

### 3) `emo_ultra` 的 FWER 负担更重（232 ROI）

- `emo_ultra`：surface 200 + volume 32 = 232 ROI。  
- `emo_new`：surface 200 ROI。  
- model-wise max-T FWER 下，ROI 数增加通常意味着更严格的阈值。

### 4) 被试筛选规则更苛刻

`emo_ultra` Step1 包含多重筛选：
- 同时具备 EMO + SOC 的被试才保留；
- 每个 stimulus_type 内先做 80% 覆盖筛刺激，再保留“拥有全部选中刺激”的被试；
- 每个刺激类型最少 3 人才输出。

影响：
- 任何一步都可能削减样本，尤其在分条件后更明显。

### 5) Step3 默认 Fisher-Z（可选关闭）

`emo_ultra` Step3 默认开启 Fisher-Z 再 zscore。理论上更接近正态，但在小样本 + 噪声大时，可能改变极值结构，和 `emo` 结果不完全可比。

---

## 三、最容易被忽略的“伪异常”

1. **比较口径不一致**：
   - 你可能把 `emo_ultra` 的“单个 stimulus_type 显著数”拿去比 `emo` 的“全刺激合并显著数”。这两者统计对象不同，不宜直接比较。
2. **只看 FWER 不看 raw/FDR**：
   - `emo_ultra` 若在 `p_perm_one_tailed` 或 FDR 下趋势增强，但 FWER 下不显著，这更像是功效问题，不一定是 pipeline 错误。
3. **每个 stimulus_type 样本量差异大**：
   - 某些类型 n_sub 太低，导致几乎不可能过 FWER。

---

## 四、建议你立即做的 6 个核查（优先级从高到低）

1. **核对每个 stimulus_type 的 n_subjects**（最关键）
   - 看 `roi_repr_matrix_232_summary.csv` 与 `roi_isc_dev_models_perm_fwer_summary.csv`。
   - 若很多条件 n_sub 明显偏小，先别纠结算法，先认定是功效瓶颈。

2. **在 ultra 上做一个“可比基线”**
   - 关闭 cocktail blank：`build_roi_repr_matrix_232.py --no-cocktail-blank`
   - 关闭 Fisher-Z：`joint_analysis_roi_isc_dev_models_perm_fwer.py --no-fisher-z`
   - 看显著数是否回升。

3. **对 ultra 做“跨 stimulus_type 汇总检验”**
   - 不要只盯单条件；先做跨条件的 meta 汇总（例如 Stouffer 或按条件平均效应再置换）。

4. **统一 ROI 集合后再比较**
   - 只比较 surface 200 ROI（先排除 232 vs 200 的多重比较差异）。

5. **检查条件列是否过细**
   - `stim_type_col=raw_emotion` 是否把条件切得太碎？可尝试更粗粒度标签。

6. **检查筛选阈值敏感性**
   - 把 `--threshold-ratio` 从 0.8 试到 0.7，观察 n_sub 与显著数变化曲线。

---

## 五、如果要快速验证“是不是功效问题”

推荐做一个最小实验（不改主逻辑，只改参数）：

- A 组（当前默认）：cocktail blank=ON，Fisher-Z=ON，threshold=0.8。  
- B 组：cocktail blank=OFF。  
- C 组：Fisher-Z=OFF。  
- D 组：threshold=0.7。  

只比较三件事：
1) 各条件 `n_subjects`；
2) `p_perm_one_tailed < 0.05` 的 ROI 数；
3) `p_fwer_model_wise < 0.05` 的 ROI 数。

如果 B/C/D 都比 A 明显增加，基本就能确认：**不是“代码错”，而是 ultra 设计更保守 + 功效不足**。

---

## 六、附：和你当前现象最一致的解释

结合脚本设计，最符合你描述（`emo_ultra` 比 `emo` 还少）的解释是：

- `emo_ultra` 把数据拆成多个 stimulus_type 分开检验，样本量变小；
- 同时默认做 cocktail blank + RSM，削弱了幅度型可检信号；
- 再叠加 232 ROI 的 FWER 校正，最终显著数下滑。

这在统计上是完全可能且常见的，并不一定意味着流程实现出错。
