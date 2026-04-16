"""
rsa_config.py (优化版)

用途
- 统一定义 ROI-level RSA 所需的输入索引、ROI 来源、Top-K 体素筛选策略与输出目录。

论文意义（这一层回答什么问题）
- 这一层从“哪里激活更强”转向“脑表征是否更相似/更可分”，用于检验学习前后、
  不同条件在同一 ROI 内的表征结构是否发生系统性重组。

方法
- 读取 LSS trial beta 与刺激模板，按固定顺序对齐后在 ROI 内计算相似度矩阵。
- 可选 `Top-K` 体素筛选：仅保留一阶 GLM 中最活跃的体素，强调信号较强的子区域。

输入
- `LSS_META_FILE`：LSS trial 元数据总索引，决定每个 beta 文件如何映射到刺激。
- `STIMULI_TEMPLATE`：刺激模板，提供跨被试一致的 trial 排序标准。
- `ROI_MASKS`：通常来自组水平 GLM 的 ROI mask。
- `GLM_1ST_DIR`：Top-K 模式下用于体素筛选的一阶统计图目录。

输出
- `OUTPUT_DIR`：RSA 汇总统计、item-wise 明细及下游 LMM 可直接读取的结果表。

关键参数为何这样设
- `USE_TOP_K=False`：默认关闭，优先保证 ROI 定义透明与可复现；开启后需格外注意选择偏差。
- `TOP_K_VOXELS=500`：经验上在信号稳定性与过拟合风险间折中，过小会让结果高度依赖噪声体素。
- `METRIC='correlation'`：更关注模式形状而非整体幅度，适合比较表征几何。

结果解读
- 若同类项目在 post 的相似度更高，通常意味着学习后该 ROI 对任务相关维度的表征更稳定或更聚合。
- 若条件间差异主要体现在 item-wise 交互项，而不是总均值，建议继续用 LMM 而非只看 summary。

常见坑
- `STIMULI_TEMPLATE` 与 LSS 元数据排序不一致，会让 RSA 矩阵“看起来正常但语义错位”。
- 使用由同一对比定义的 ROI 再做同一数据的 Top-K 体素筛选，可能放大双重使用数据的问题。
- 被试缺 trial、文件名不是唯一映射、或 `beta_file` 相对/绝对路径混杂，都会导致对齐失败或假结果。
"""
import os
from pathlib import Path

# ... (保持原有的 BASE_DIR, LSS_META_FILE, STIMULI_TEMPLATE, ROI_MASKS 配置不变) ...
BASE_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor"))
LSS_META_FILE = BASE_DIR / "lss_betas_final/lss_metadata_index_final.csv"
STIMULI_TEMPLATE = BASE_DIR / "stimuli_template.csv"

# ROI 定义 (Group Level Mask)
ROI_MASKS = {
    "Metaphor_gt_Spatial": BASE_DIR / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Metaphor_gt_Spatial/Metaphor_gt_Spatial_roi_mask.nii.gz",
    "Spatial_gt_Metaphor": BASE_DIR / "glm_analysis_fwe_final/2nd_level_CLUSTER_FWE/Spatial_gt_Metaphor/Spatial_gt_Metaphor_roi_mask.nii.gz",
}

# ================= 新增配置 =================

# 1. Top-K 参数
USE_TOP_K = False      # 开关
TOP_K_VOXELS = 500    # 只取最活跃的 200 个体素 (经验值: 100-500)

# 2. 一阶 GLM 统计图路径模板 (用于筛选体素)
# {sub} 会被自动替换。请根据你 1st_level 的实际文件名修改
# 这里的 contrast 应该对应 ROI 的定义 (如 Metaphor > Spatial)
# 建议：如果做 IFG 分析，用 Metaphor_gt_Spatial 的 T图；做 Precuneus 用 Spatial_gt_Metaphor
GLM_1ST_DIR = BASE_DIR / "glm_analysis_fwe_final/1st_level"

# 定义 ROI 对应的 Contrast 名称 (用于寻找对应的 T-map)
ROI_CONTRAST_MAP = {
    "Metaphor_gt_Spatial": "Metaphor_gt_Spatial",
    "Spatial_gt_Metaphor": "Spatial_gt_Metaphor"
}

# 输出目录
OUTPUT_DIR = BASE_DIR / "rsa_results_optimized"

# 其他
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 29)]
METRIC = 'correlation'
N_JOBS = 4
