# Reviewer Supplementary Analyses 脚本说明

本目录是 reviewer/supplementary sandbox。所有脚本只读取既有 pattern 和 QC 输出，默认只写入：

```text
E:/python_metaphor/paper_outputs/qc/reviewer_supp/<module>/
```

这些分析只能作为 sanity check / supplementary / boundary / reviewer response，不能替代当前主线：

```text
learning condition geometry
-> post trained-edge differentiation
-> retrieval rebound / decoding
-> memory relevance mainly through stronger prior post separation
```

## 安全规则

- 不修改既有分析脚本。
- 不覆盖既有输出；如果输出文件已存在，脚本会报错退出。
- item-level mixed models 必须使用 `condition_item_id`，不能用裸 `pair_id`。
- `append_section27.py` 默认只生成 preview；只有显式传入 `--apply` 才会追加到 `result_new_meta_roi.md`。
- `run_all_reviewer_supp.py` 默认只跑 M2/M3/M4/M5/M6/M9/M11，不自动 append，不自动 isolation check。

## 模块

| 文件 | 模块 | 定位 |
| --- | --- | --- |
| `shared.py` | shared utilities | 统一 IO、ROI/mask 读取、FDR、mixed/GLM 拟合、`condition_item_id` 生成。 |
| `m2_univariate_sanity.py` | M2 | ROI mean activation sanity check + Step5C univariate-control model。Step5C merge 按 `subject + condition + roi_set + roi` 对齐。 |
| `m3_learning_dm.py` | M3 | learning-stage Dm。memory 使用 `memory >= 0.5` 的 lenient binary rule；item random effect 使用 `condition_item_id`。 |
| `m4_correct_ers.py` | M4 | learning-to-retrieval same-vs-other ERS with memory moderation。不是 correct-only filtering。 |
| `m5_theoretical_reframing.py` | M5 | 理论叙事重述，不做新统计。 |
| `m6_novelty_repetition.py` | M6 | pre/post ROI mean activation novelty/repetition sanity check。 |
| `m9_logistic_dm.py` | M9 | strict/lenient binary-memory sensitivity；支持 Stage4 long format 和 wide input。 |
| `m11_kj_fair_characterization.py` | M11 | 公平呈现 KJ signature；输出 `evidence_tier = FDR-corrected / uncorrected-only`，并标记 raw Stage1 retrieval shared-post downgrade。 |
| `append_section27.py` | Section 27 preview/apply | 默认生成 `section27_preview.md`；传 `--apply` 才 append。 |
| `isolation_check.py` | isolation report | 输出当前 git status / hash 报告。建议运行前后各保存一次 hash。 |
| `run_all_reviewer_supp.py` | orchestrator | 默认跑 M2/M3/M4/M5/M6/M9/M11；可选 `--append-preview` 和 `--isolation-check`。 |

## 建议执行顺序

先单模块运行，检查 QC 和模型状态：

```bash
cd E:/PythonAnalysis/metaphoric/final_version/reviewer_supp

python m5_theoretical_reframing.py
python m6_novelty_repetition.py
python m2_univariate_sanity.py
python m3_learning_dm.py
python m4_correct_ers.py
python m9_logistic_dm.py
python m11_kj_fair_characterization.py
```

每个模块跑完检查：

```text
QC rows 是否正常
status == ok 的模型比例
是否出现 mixed_failed_no_fallback
是否输出空表
是否存在 ID collision
是否存在 ROI merge 异常
```

批量运行可用：

```bash
python run_all_reviewer_supp.py
```

如需批量后生成 Section 27 preview：

```bash
python run_all_reviewer_supp.py --append-preview
```

如需追加到 result 文档，先审阅 preview，再显式执行：

```bash
python append_section27.py --apply
```

## 解释边界

M2/M6 是 univariate sanity checks。阴性结果只能说明没有明显 ROI mean activation 解释，不能证明 RSA 完全不受 SNR、pattern norm 或 variance 影响。

M3/M4 是 exploratory learning/memory checks。即使阳性，也不能直接升级为主机制，除非与 Stage5 主线和 post-edge differentiation 有清楚、预注册、稳健的连接。

M9 是 memory binary-coding sensitivity。strict/lenient 结果用于检查连续 memory-score 口径是否脆弱，不替代 Stage4/4.5 的连续模型和 component decomposition。

M11 是 KJ fairness summary。`FDR-corrected` 可作为较强补充证据；`uncorrected-only` 只能作为描述性线索；raw Stage1 retrieval rows 带 shared-post downgrade，不能当强 KJ signature。
