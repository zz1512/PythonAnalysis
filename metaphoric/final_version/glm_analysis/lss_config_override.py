"""
lss_config_override.py
LSS 分析专用配置参数
"""
from pathlib import Path

# ==========================
# 核心设置
# ==========================
# 你的前测是 Run 1&2，后测是 Run 5&6
LSS_RUNS = [1, 2, 5, 6]

# LSS 输出目录 (建议放在专门的文件夹)
LSS_OUTPUT_DIR = Path("E:/python_metaphor/lss_betas_final")

# 关键参数：RSA 要求不做平滑！保留体素模式细节
LSS_SMOOTHING = None

# TR设置
LSS_TR = 2.0

# 并行进程数
# LSS 计算密集，建议设为 CPU 核心数的一半 (如 4 或 6)
# 如果内存报错，请调小这个数字
N_JOBS = 4