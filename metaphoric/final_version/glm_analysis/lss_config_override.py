"""
lss_config_override.py

用途
- 集中管理 LSS（least-squares separate）阶段的关键覆盖参数，避免在主脚本里散落硬编码。

论文意义（这一层回答什么问题）
- LSS 不是最终统计结论本身，而是为单 trial 表征分析提供“干净、可逐试次对齐”的 beta 图。
- 它决定后续 RSA/MVPA 看到的是不是 trial-level 神经模式，而不只是条件平均激活。

方法
- 为每个 trial 单独建模一个目标回归量，其他 trial 作为“其余事件”吸收，从而得到 trial-wise beta。

输入
- 依赖 `PYTHON_METAPHOR_ROOT` 与 `run_lss.py` / `glm_config.py` 中的路径模板。

输出
- 由 `LSS_OUTPUT_DIR` 指向的 trial-wise beta 目录及其元数据索引。

关键参数为何这样设
- `LSS_RUNS=[1,2,5,6]`：默认服务于前后测表征比较；如果改成学习阶段，需要同步修改结果解释。
- `LSS_SMOOTHING=None`：保留细粒度体素模式，避免空间平滑把相邻 trial 的表征差异“抹平”。
- `N_JOBS=4`：LSS 内存开销大，并行过高更容易触发缓存爆炸或系统 swap。

结果解读
- 这里本身不直接看显著性；主要检查 beta 是否完整生成、命名是否可与刺激模板一一对齐、每个 trial 是否都有记录。

常见坑
- 误把 learning runs 与 pre/post runs 混用，导致后续“学习前后变化”解释错位。
- 对 LSS 输出再次做平滑或重采样，破坏 MVPA/RSA 依赖的局部模式结构。
- trial 命名或 `unique_label` 不唯一，后续按刺激模板对齐时会静默错配。
"""
import os
from pathlib import Path

# ==========================
# 核心设置
# ==========================
# 你的前测是 Run 1&2，后测是 Run 5&6
LSS_RUNS = [1, 2, 5, 6]

# LSS 输出目录 (建议放在专门的文件夹)
LSS_OUTPUT_DIR = Path(os.environ.get("PYTHON_METAPHOR_ROOT", "E:/python_metaphor")) / "lss_betas_final"

# 关键参数：RSA 要求不做平滑！保留体素模式细节
LSS_SMOOTHING = None

# TR设置
LSS_TR = 2.0

# 并行进程数
# LSS 计算密集，建议设为 CPU 核心数的一半 (如 4 或 6)
# 如果内存报错，请调小这个数字
N_JOBS = 4
