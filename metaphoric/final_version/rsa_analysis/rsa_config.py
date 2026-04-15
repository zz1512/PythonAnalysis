"""
rsa_config.py (优化版)
"""
from pathlib import Path

# ... (保持原有的 BASE_DIR, LSS_META_FILE, STIMULI_TEMPLATE, ROI_MASKS 配置不变) ...
BASE_DIR = Path("E:/python_metaphor")
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