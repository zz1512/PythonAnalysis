> 归档说明：此文档由 `gjxx/MATLAB_脚本梳理.md` 移入 `gjxx/docs/`，内容基本不改，仅用于历史参考。

# GJXX_1_reanalysis MATLAB 脚本梳理

## 1. 先给结论

- 当前仓库里一共有 `118` 个 `.m` 文件。
- 真正形成完整输入-输出链、而且目录里有现成结果可对上的，主要是 4 条线：
  - `no_too_easyorhard_item`：我判断这是你现在最应该先接手的一条主线。
  - `gps`：较早的 item-level / GPS 主线。
  - `rd_regression`：较早的全脑 RD / searchlight / 回归主线。
  - `activation`：条件激活分析主线。
- 其余目录大多是在这几条主线基础上做条件筛选、删 trial、改阈值、跨实验关联或只保留 remembered 项目的派生分析。
- `Copy_of_*`、`Copy_2_of_*`、`Untitled*`、`test*`、`ttttt.m` 基本都不是优先接手对象，除非你在复现某个分支结果。

## 2. 我判断“哪些脚本正在用”的依据

### 2.1 能对上完整产出链的目录

- `no_too_easyorhard_item\events`：有 `23` 个被试目录。
- `no_too_easyorhard_item\glm_item_hl`：有 `23` 个被试目录。
- `no_too_easyorhard_item\pattern`：有 `23` 个被试目录。
- `no_too_easyorhard_item\rca\patterns`：有分被试的 `glm_T_HSC.nii` / `glm_T_LSC.nii`。
- `gps\patterns`：有 `27` 个条目，其中主体是各被试目录，`gps\patterns\rsa` 里有 `26` 个条目。
- `rd_regression\rd`：有 `27` 个条目，`rd_regression\rd\rsa` 里有 `23` 个条目。
- `activation\glm_hlrf` 与 `activation\GLM_rf`：都已有批量一阶结果。

### 2.2 代码本身体现出的主流程入口名

- 一阶模型入口通常是：
  - `first_level_analysis_*`
  - `firstlevel_*`
  - `glm_*`
- pattern 拼接通常是：
  - `get_rsa_patterns*`
- ROI / GPS / RDM / RD 统计通常是：
  - `nGPS_NA.m`
  - `corr_between_roi_dsm.m`
  - `rd.m`
  - `searchlight*.m`
  - `doregression*.m`

### 2.3 我认为“当前主用”而不是“历史尝试”的特征

- 文件名明确，不是 `Copy_of_*` / `Untitled*`。
- 路径指向 `H:\GJXX_1_reanalysis\...` 自身目录，而不是明显跨项目的 `H:\GJXX_2_reanalysis\...`。
- 对应输出目录里真的有批量被试结果。

## 3. 建议你先接手的脚本顺序

### 3.1 第一优先级：`no_too_easyorhard_item`

这是当前仓库里最完整、最像“最后定稿版”的分支。

1. `no_too_easyorhard_item\get_events.m`
   - 作用：先统计所有图片在全体被试中的出现频率，把太容易/太难的项目改成 `tl` / `th`，并重写到 `no_too_easyorhard_item\events\sub-XX\*_events.txt`。
2. `no_too_easyorhard_item\firstlevel_hscandlsc.m`
   - 作用：用上一步重写后的事件文件做 item-level 一阶模型，输出到 `no_too_easyorhard_item\glm_item_hl\sub-XX`。
   - 这里把 trial 分成 `HSC`、`LSC`、`median0`、`tianchong`，并把每个 trial 单独建模再改名成 `HSCspmT_*` / `LSCspmT_*` 等。
3. `no_too_easyorhard_item\get_rsa_patterns_gps.m`
   - 作用：把 `glm_item_hl` 中的 `HSC*`、`LSC*`、`median*` 拼成 4D 文件 `glm_T_gps.nii`，输出到 `no_too_easyorhard_item\pattern\sub-XX`。
4. `no_too_easyorhard_item\nGPS_NA.m`
   - 作用：在 ROI 内读 `glm_T_gps.nii`，计算 HSC 与 LSC 的 GPS / similarity 指标并做配对 t 检验。
5. `no_too_easyorhard_item\rca\get_rsa_patterns.m`
   - 作用：分别把 `HSC*` 和 `LSC*` 拼成 `glm_T_HSC.nii` / `glm_T_LSC.nii`，输出到 `no_too_easyorhard_item\rca\patterns\sub-XX`。
6. `no_too_easyorhard_item\rca\corr_between_roi_dsm.m`
   - 作用：比较两个 ROI 的 DSM 相关在 HSC 和 LSC 条件下是否不同。
7. `no_too_easyorhard_item\rd.m`
   - 作用：比较 HSC / LSC 的 representational dimension。
8. `no_too_easyorhard_item\sametrialnumber1.m`
   - 作用：为了解决 HSC / LSC trial 数不一致，用随机抽样到相同 trial 数的方式反复估计 RD。

### 3.2 第二优先级：`gps`

这是较早的一条 item-level 主线，适合理解项目最原始的 GPS 思路。

1. `gps\scripts\first_level_analysis_HL_item.m`
   - 作用：按 run 建 item-level 模型，trial 被分成 `HSC` / `LSC` / `tianchong` / `notuse`，输出到 `gps\patterns\subXX`。
2. `gps\scripts\firstlevelforsomesubjects_HL.m`
   - 作用：只补跑个别被试的版本。
3. `gps\scripts\get_rsa_patterns_gps.m`
   - 作用：把 `HSC*`、`LSC*`、`tianchong*` 拼成 `gps\patterns\rsa\subXX\glm_T_gps.nii`。
4. `gps\scripts\nGPS_NA.m`
   - 作用：ROI 内计算 HSC 与 LSC 的 GPS 指标。

### 3.3 第三优先级：`rd_regression`

这条线处理的是“全部 example 的 item-level 图像 -> searchlight RD -> 全脑行为回归”。

1. `rd_regression\scripts\glm_allna.m`
   - 作用：给每个 example 单独建模，输出到 `rd_regression\rd\subXX`。
2. `rd_regression\scripts\get_rsa_patterns_na.m`
   - 作用：把 `example*nii` 拼成 `glm_T_allexamples.nii`，输出到 `rd_regression\rd\rsa\subXX`。
3. `rd_regression\scripts\searchlight_RD_allitem.m`
   - 作用：对 `glm_T_allexamples.nii` 做 searchlight，输出 `rdvar.nii`。
4. `rd_regression\scripts\doregression_flxx1_rd_C.m`
   - 作用：把 `rdvar.nii` 做 robust regression 到行为分数。

### 3.4 第四优先级：`activation`

这是更传统的条件激活分析。

1. `activation\scripts\first_level_analysis_4_types.m`
   - 作用：按 `HSC-r` / `HSC-f` / `LSC-r` / `LSC-f` / `tianchong` 建模。
2. `activation\scripts\firstlevel_glm_rf.m`
   - 作用：只按 remember / forget 条件建模。
3. `activation\scripts\NA_use_aal_C.m`
4. `activation\scripts\NA_use_aal_memory.m`
   - 作用：ROI 提取和配对统计。

## 4. 我认为“具体使用的脚本”

如果你问的是“先别管草稿，只看真正值得接手的脚本”，我建议默认先看下面这 16 个：

- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\get_events.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\firstlevel_hscandlsc.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\get_rsa_patterns_gps.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\nGPS_NA.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\rca\get_rsa_patterns.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\rca\corr_between_roi_dsm.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\rd.m`
- `H:\GJXX_1_reanalysis\no_too_easyorhard_item\sametrialnumber1.m`
- `H:\GJXX_1_reanalysis\gps\scripts\first_level_analysis_HL_item.m`
- `H:\GJXX_1_reanalysis\gps\scripts\get_rsa_patterns_gps.m`
- `H:\GJXX_1_reanalysis\gps\scripts\nGPS_NA.m`
- `H:\GJXX_1_reanalysis\rd_regression\scripts\glm_allna.m`
- `H:\GJXX_1_reanalysis\rd_regression\scripts\get_rsa_patterns_na.m`
- `H:\GJXX_1_reanalysis\rd_regression\scripts\searchlight_RD_allitem.m`
- `H:\GJXX_1_reanalysis\rd_regression\scripts\doregression_flxx1_rd_C.m`
- `H:\GJXX_1_reanalysis\activation\scripts\first_level_analysis_4_types.m`

## 5. 每个目录的脚本清单与状态

状态说明：

- `主线`：建议优先阅读。
- `补跑/派生`：用于补被试、换阈值、换条件、做扩展分析。
- `历史/低优先`：有参考价值，但不是当前首选。
- `草稿/副本`：优先级最低。

### 5.1 `activation\scripts`

- `first_level_analysis_4_types.m`：`主线`
- `firstlevel_glm_rf.m`：`主线`
- `NA_use_aal_C.m`：`主线`
- `NA_use_aal_memory.m`：`主线`
- `NA_use_aal_allexamples.m`：`历史/低优先`
- `Copy_of_first_level_analysis_4_types.m`：`草稿/副本`
- `Untitled.m`：`历史/低优先`
- `Untitled2.m`：`历史/低优先`
- `Untitled3.m`：`历史/低优先`
- `Untitled4.m`：`历史/低优先`
- `Untitled5.m`：`历史/低优先`

### 5.2 `activationprediction`

- `try_pre.m`：`历史/低优先`
- `gettheresults.m`：`工具脚本`

### 5.3 `activationprediction\scripts_for_memory`

- `doregression_flxx1_rd_memory.m`：`派生`
- `gettheresults.m`：`工具脚本`

### 5.4 `events`

- `Untitled6.m`：`历史/低优先`
- `Copy_of_Untitled6.m`：`草稿/副本`

### 5.5 `gps\scripts`

- `first_level_analysis_HL_item.m`：`主线`
- `firstlevelforsomesubjects_HL.m`：`补跑/派生`
- `get_rsa_patterns_gps.m`：`主线`
- `nGPS_NA.m`：`主线`
- `headbodytail.m`：`派生`
- `Copy_of_get_rsa_patterns_gps.m`：`历史/低优先`
- `Copy_2_of_get_rsa_patterns_gps.m`：`历史/低优先`
- `Copy_of_nGPS_NA.m`：`历史/低优先`
- `Copy_2_of_nGPS_NA.m`：`历史/低优先`
- `Untitled.m`：`历史/低优先`
- `Untitled2.m`：`历史/低优先`

### 5.6 `gps\scripts_tichujiaocha`

- `first_level_analysis_HL_item.m`：`派生`
- `firstlevelforsomesubjects_HL.m`：`补跑/派生`
- `get_rsa_patterns_gps.m`：`派生`
- `nGPS_NA.m`：`派生`

### 5.7 `gps` 根目录

- `only_rem_corr_between_roi_dsm.m`：`派生`

### 5.8 `gps_of_all_examples_prediction\scripts`

- `get_rsa_patterns_gps.m`：`派生`
- `searchlight.m`：`派生`
- `mymeasure.m`：`派生函数`
- `doregression_flxx1_gps_c.m`：`派生`
- `doregression_flxx1_gps_m.m`：`派生`
- `gettheresults.m`：`工具脚本`

### 5.9 `rd_regression\scripts`

- `glm_allna.m`：`主线`
- `get_rsa_patterns_na.m`：`主线`
- `searchlight_RD_allitem.m`：`主线`
- `doregression_flxx1_rd_C.m`：`主线`
- `my_var_measure.m`：`主线函数，但存在命名风险`
- `get_subjects_C_score.m`：`工具脚本`
- `get_subjects_memory_score.m`：`工具脚本`
- `gettheresults.m`：`工具脚本`

### 5.10 `rd_regression\scripts_for_memory`

- `doregression_flxx1_rd_memory.m`：`派生`
- `corr_allexamples.m`：`派生`
- `gettheresults.m`：`工具脚本`

### 5.11 `no_too_easyorhard_item`

- `get_events.m`：`主线`
- `firstlevel_hscandlsc.m`：`主线`
- `get_rsa_patterns_gps.m`：`主线`
- `nGPS_NA.m`：`主线`
- `rd.m`：`主线`
- `sametrialnumber1.m`：`主线`
- `conclude.m`：`结果汇总小工具`
- `Copy_of_rd.m`：`草稿/副本，且与 rd.m 完全一致`

### 5.12 `no_too_easyorhard_item\rca`

- `get_rsa_patterns.m`：`主线`
- `corr_between_roi_dsm.m`：`主线`

### 5.13 `tichu_jiaocha`

- `firstlevel_hscandlsc.m`：`派生`

### 5.14 `tichu_jiaocha\rca`

- `get_rsa_patterns.m`：`派生`
- `corr_between_roi_dsm.m`：`派生`

### 5.15 `jiaocha_vs_nojiaocha`

- `Untitled.m`：`派生`
- `Untitled5.m`：`派生`
- `rd.m`：`派生`

### 5.16 `jiaocha_vs_nojiaocha\gps`

- `get_rsa_patterns_gps.m`：`派生`
- `nGPS_NA.m`：`派生`

### 5.17 `jiaocha_vs_nojiaocha\rca`

- `get_rsa_patterns.m`：`派生`
- `corr_between_roi_dsm.m`：`派生`

### 5.18 `jiaocha_vs_nojiaocha\rd`

- `rd.m`：`派生`

### 5.19 `rd_tichu`

- `get_rsa_patterns.m`：`派生`
- `Untitled.m`：`派生`
- `Copy_of_Untitled.m`：`草稿/副本`

### 5.20 `biggerthan3`

- `firstlevel_hscandlsc.m`：`派生`
- `version1canuse.m`：`派生里最像定稿版`
- `v2.m`：`派生`
- `sheng.m`：`派生`
- `p1.m`：`派生`
- `p2.m`：`派生`
- `hl.m`：`派生`
- `Copy_of_firstlevel_hscandlsc.m`：`草稿/副本`
- `Copy_2_of_firstlevel_hscandlsc.m`：`草稿/副本`
- `Copy_2_of_get_rsa_patterns_gps.m`：`草稿/副本`
- `Copy_3_of_get_rsa_patterns_gps.m`：`草稿/副本`
- `Copy_4_of_get_rsa_patterns_gps.m`：`草稿/副本`
- `Copy_of_p1.m`：`草稿/副本`
- `Copy_2_of_p1.m`：`草稿/副本`
- `Copy_of_Untitled2.m`：`草稿/副本`
- `Untitled2.m`：`草稿/副本`
- `Untitled3.m`：`草稿/副本`
- `testtesttest.m`：`草稿/副本`
- `ttttt.m`：`草稿/副本`

### 5.21 `biggerthan3\pattern1`

- `hl.m`：`派生`
- `Untitled2.m`：`草稿/副本`
- `Copy_of_Untitled2.m`：`草稿/副本`

### 5.22 `biggerthan3\pattern1\rsa`

- `ggggppppsss.m`：`草稿/副本`

### 5.23 `only_rem`

- `only_remembered_dimension_hl.m`：`派生`
- `only_rem_corr_between_roi_dsm.m`：`派生`
- `sametrialnumber1.m`：`派生`
- `t111.m`：`派生`
- `test.m`：`派生`
- `untitled.m`：`草稿/副本`
- `Copy_of_nGPS_NA.m`：`草稿/副本`
- `Copy_of_sametrialnumber1.m`：`草稿/副本`
- `Copy_2_of_sametrialnumber1.m`：`草稿/副本`

### 5.24 `inter_sub`

- `usethis.m`：`派生里最像定稿版`
- `dimension_NA_all.m`：`派生`
- `compare_2_r.m`：`结果比较工具`
- `fisher_r_to_z_test.m`：`统计工具`
- `finaL_saetrialn.m`：`派生`
- `Untitled.m`：`历史/低优先`
- `Untitled2.m`：`历史/低优先`
- `Untitled3.m`：`历史/低优先`
- `Copy_of_Untitled3.m`：`草稿/副本`

### 5.25 `inter_sub_gps`

- `Untitled8.m`：`派生`
- `Untitled10.m`：`结果查看/辅助`

### 5.26 `mask`

- `Untitled2.m`：`辅助脚本`

## 6. 明确的重复文件

### 6.1 内容完全一样

- `no_too_easyorhard\rd.m`
- `no_too_easyorhard\Copy_of_rd.m`

### 6.2 完全一样的工具脚本

下面 5 个 `gettheresults.m` 内容一致，只是被复制到不同目录：

- `activationprediction\gettheresults.m`
- `activationprediction\scripts_for_memory\gettheresults.m`
- `gps_of_all_examples_prediction\scripts\gettheresults.m`
- `rd_regression\scripts\gettheresults.m`
- `rd_regression\scripts_for_memory\gettheresults.m`

## 7. 接手时最需要注意的风险

### 7.1 路径写死严重

很多脚本直接写死了：

- `E:\FLDATA_analysis\...`
- `E:\FLDATA_analysisearly\...`
- `I:\FLXX1\...`
- `L:\FLXX_1\...`
- `H:\GJXX_2_reanalysis\...`

也就是说，仓库里的 `.m` 文件不是“开箱即跑”，而是强依赖作者当时的盘符结构。

### 7.2 被试命名有两套风格

- 一套是 `sub02`
- 一套是 `sub-02`

这会直接影响 `dir([data_dir filesep 'sub*'])`、输出目录拼接和行为数据对齐。

### 7.3 `rd_regression\scripts\my_var_measure.m` 有明显命名风险

- 文件名是 `my_var_measure.m`
- 但文件第一行函数定义是 `function my_weidu=my_weidu_measure(my_ds,args)`
- 同时 `searchlight_RD_allitem.m` 用的是 `measure=@my_var_measure`

MATLAB 通常要求主函数名与文件名一致，所以这段代码按当前状态有较大概率直接报错。

### 7.4 一些脚本实际上是跨项目脚本

比如：

- `inter_sub\dimension_NA_all.m`
- `only_rem\only_remembered_dimension_hl.m`
- `gps_of_all_examples_prediction\scripts\doregression_flxx1_gps_c.m`

它们混用了 `GJXX_1` 和 `GJXX_2` 的路径，更像“跨实验比较脚本”，不是当前项目的纯主流程。

## 8. 我建议你的阅读顺序

如果你要最快理解整个项目，按下面顺序读最省时间：

1. `no_too_easyorhard_item\get_events.m`
2. `no_too_easyorhard_item\firstlevel_hscandlsc.m`
3. `no_too_easyorhard_item\get_rsa_patterns_gps.m`
4. `no_too_easyorhard_item\nGPS_NA.m`
5. `no_too_easyorhard_item\rca\get_rsa_patterns.m`
6. `no_too_easyorhard_item\rca\corr_between_roi_dsm.m`
7. `no_too_easyorhard_item\rd.m`
8. `no_too_easyorhard_item\sametrialnumber1.m`
9. `gps\scripts\first_level_analysis_HL_item.m`
10. `rd_regression\scripts\glm_allna.m`
11. `activation\scripts\first_level_analysis_4_types.m`

## 9. 一句话版项目流程图

### 9.1 当前最值得接手的分支

`原始 events/onset -> get_events.m 重写 trial 标签 -> firstlevel_hscandlsc.m 做 item-level GLM -> get_rsa_patterns_gps.m / rca/get_rsa_patterns.m 拼 pattern -> nGPS_NA.m / corr_between_roi_dsm.m / rd.m / sametrialnumber1.m 做统计`

### 9.2 较早的基础分支

`原始 onset -> gps/scripts/first_level_analysis_HL_item.m -> gps/scripts/get_rsa_patterns_gps.m -> gps/scripts/nGPS_NA.m`

### 9.3 全脑 RD 分支

`原始 onset -> rd_regression/scripts/glm_allna.m -> get_rsa_patterns_na.m -> searchlight_RD_allitem.m -> doregression_flxx1_rd_C.m`
