# metaphoric/final_version

这版 `final_version` 只保留分析计划的主干流程：行为 -> GLM/LSS -> MVPA -> RSA -> RD/GPS -> 脑-行为关联。

## 根目录

所有脚本统一围绕 `E:/python_metaphor` 工作。

现有输入：
- `E:/python_metaphor/data_events`
- `E:/python_metaphor/Pro_proc_data`
- `E:/python_metaphor/lss_betas_final`
- `E:/python_metaphor/glm_analysis_fwe_final`
- `E:/python_metaphor/stimuli_template.csv`
- `E:/python_metaphor/rsa_data_validation_report_fixed.csv`

主流程会生成的结果目录：
- `E:/python_metaphor/roi_masks_final`
- `E:/python_metaphor/mvpa_roi_results`
- `E:/python_metaphor/mvpa_searchlight_results`
- `E:/python_metaphor/rsa_learning_results`
- 你自己指定的 `pattern_root`、`rd_results`、`gps_results`、`brain_behavior_results` 等输出目录

## 主干故事

1. 行为层：先确认隐喻学习在记忆准确率、反应时上的优势。
2. 单变量层：在学习阶段做 GLM，定位 `yy > kj` 或 `kj > yy` 的显著脑区。
3. 单试次层：对学习阶段和前后测分别做 LSS，得到 trial-level beta。
4. 模式层：在 ROI 和全脑 searchlight 上检验条件可解码性，并比较 pre/post 与跨阶段泛化。
5. 表征层：用 RSA、RD、GPS 检验学习前后表征结构是否重组。
6. 关联层：把行为、MVPA、RSA、RD、GPS 汇总，做脑-行为相关。

## 脚本与功能

### 1. 行为分析
- `behavior_analysis/addMemory.py`：给行为事件表补 memory 列。
- `behavior_analysis/getMemoryDetail.py`：补 action / action_time 等行为细节。
- `behavior_analysis/getAccuracyScore.py`：计算准确率并做条件比较。
- `behavior_analysis/getTimeActionScore.py`：计算反应时并做条件比较。
- `behavior_analysis/behavior_lmm.py`：对行为表做 LMM。

### 2. GLM 与 LSS
- `glm_analysis/main_analysis.py`：学习阶段 run3-4 的一阶/二阶 GLM。
- `glm_analysis/run_lss.py`：前后测 run1-2、5-6 的 LSS。
- `glm_analysis/check_lss_comprehensive.py`：检查 `lss_betas_final` 的完整性和数值质量。
- `fmri_lss/fmri_lss_enhance.py`：学习阶段 run3-4 的 LSS，输出到 `E:/python_metaphor/lss_betas_final/sub-xx/run-3`、`run-4`。
- `glm_analysis/gpu_permutation.py`：GLM 组水平置换加速。
- `get_nii.py`：GLM 结果快速查看。

### 3. ROI 与 MVPA
- `fmri_mvpa/fmri_mvpa_roi/create_optimized_roi_masks.py`：根据 `glm_analysis_fwe_final` 的 `cluster_report.csv` 和图谱生成 ROI mask。
- `fmri_mvpa/fmri_mvpa_roi/fmri_mvpa_roi_pipeline_merged.py`：学习阶段 ROI-MVPA。
- `fmri_mvpa/fmri_mvpa_searchlight/fmri_mvpa_searchlight_pipeline_multi.py`：学习阶段 searchlight-MVPA。
- `fmri_mvpa/fmri_mvpa_roi/mvpa_pre_post_comparison.py`：比较 pre/post 的 ROI-MVPA。
- `fmri_mvpa/fmri_mvpa_roi/cross_phase_decoding.py`：learning 训练，pre/post 测试。

### 4. RSA
- `fmri_rsa/fmri_rsa_pipline.py`：学习阶段 RSA。
- `rsa_analysis/check_data_integrity.py`：检查 `lss_metadata_index_final.csv` 和 `stimuli_template.csv` 的项目匹配。
- `rsa_analysis/run_rsa_optimized.py`：前后测 item-wise RSA 主脚本。
- `rsa_analysis/rsa_lmm.py`：对 item-wise RSA 做 LMM。
- `rsa_analysis/cross_roi_rsa.py`：跨 ROI 的 RDM 一致性。
- `rsa_analysis/model_rdm_comparison.py`：模型 RDM 比较。
- `rsa_analysis/plot_rsa_results.py`：画 RSA 结果图。

### 5. 表征几何
- `representation_analysis/stack_patterns.py`：把 single-trial beta 按 `phase x condition` 堆成 4D NIfTI。
- `representation_analysis/rd_analysis.py`：ROI-level RD。
- `representation_analysis/gps_analysis.py`：ROI-level GPS。
- `representation_analysis/rd_searchlight.py`：全脑 RD searchlight。
- `representation_analysis/seed_connectivity.py`：基于局部几何的 seed connectivity searchlight。
- `representation_analysis/rd_robustness.py`：RD 稳健性检验。

### 6. 脑-行为关联
- `brain_behavior/brain_behavior_correlation.py`：汇总多层指标做相关分析。

## 推荐执行顺序

### Step 1：行为
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/behavior_analysis
python addMemory.py
python getMemoryDetail.py
python getAccuracyScore.py
python getTimeActionScore.py
python behavior_lmm.py INPUT.csv OUTPUT_DIR --response accuracy
```

### Step 2：前后测 LSS 与学习阶段 LSS
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/glm_analysis
python run_lss.py
python check_lss_comprehensive.py

cd E:/PythonAnalysis/metaphoric/final_version/fmri_lss
python fmri_lss_enhance.py
```

### Step 3：GLM 与 ROI
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/glm_analysis
python main_analysis.py

cd E:/PythonAnalysis/metaphoric/final_version/fmri_mvpa/fmri_mvpa_roi
python create_optimized_roi_masks.py
```

### Step 4：模式堆叠
如果只堆 pre/post，可直接用现成索引：
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/representation_analysis
python stack_patterns.py E:/python_metaphor/lss_betas_final/lss_metadata_index_final.csv E:/python_metaphor/pattern_root
```
如果要把 learning 也一起堆叠，需要先把 run3-4 的 `trial_info.csv` 合并成一个 trial-level metadata 文件，再作为 `metadata_path` 传入。

### Step 5：MVPA
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/fmri_mvpa/fmri_mvpa_roi
python fmri_mvpa_roi_pipeline_merged.py
python mvpa_pre_post_comparison.py E:/python_metaphor/pattern_root E:/python_metaphor/roi_masks_final E:/python_metaphor/mvpa_roi_prepost_results
python cross_phase_decoding.py E:/python_metaphor/pattern_root E:/python_metaphor/roi_masks_final E:/python_metaphor/cross_phase_results

cd E:/PythonAnalysis/metaphoric/final_version/fmri_mvpa/fmri_mvpa_searchlight
python fmri_mvpa_searchlight_pipeline_multi.py
```

### Step 6：RSA
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/fmri_rsa
python fmri_rsa_pipline.py

cd E:/PythonAnalysis/metaphoric/final_version/rsa_analysis
python check_data_integrity.py
python run_rsa_optimized.py
python rsa_lmm.py INPUT.tsv OUTPUT_DIR
python cross_roi_rsa.py INPUT_DIR ROI_DIR OUTPUT_DIR
python model_rdm_comparison.py NEURAL_RDM.tsv OUTPUT_DIR
python plot_rsa_results.py
```

### Step 7：RD / GPS / Seed
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/representation_analysis
python rd_analysis.py E:/python_metaphor/pattern_root E:/python_metaphor/roi_masks_final E:/python_metaphor/rd_results
python gps_analysis.py E:/python_metaphor/pattern_root E:/python_metaphor/roi_masks_final E:/python_metaphor/gps_results
python rd_searchlight.py E:/python_metaphor/pattern_root SUBJECT_MASK_ROOT E:/python_metaphor/rd_searchlight_results
python seed_connectivity.py E:/python_metaphor/pattern_root SUBJECT_MASK_ROOT SEED_MASK.nii.gz E:/python_metaphor/seed_results
```

### Step 8：脑-行为关联
```powershell
cd E:/PythonAnalysis/metaphoric/final_version/brain_behavior
python brain_behavior_correlation.py E:/python_metaphor/brain_behavior_results METRIC1.tsv METRIC2.tsv METRIC3.tsv
```

## 当前这版的使用约定

1. 学习阶段脚本默认分析 run3-4；前后测脚本默认分析 run1-2、5-6。
2. ROI-MVPA、Searchlight-MVPA、学习 RSA 已改成优先读取 `trial_info.csv` 里的 `beta_file`，并自动兼容 `run-1` / `run-1_LSS` 这两种目录命名。
3. Searchlight 默认优先使用 `E:/python_metaphor/glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Metaphor_gt_Spatial/group_mask.nii.gz`。
4. 如果你新生成了 `roi_masks_final`，后续表征脚本优先传这个目录；如果不传，当前保留脚本也能直接从 `glm_analysis_fwe_final/2nd_level_CLUSTER_FWE` 里读取现成 `*_roi_mask.nii.gz`。
