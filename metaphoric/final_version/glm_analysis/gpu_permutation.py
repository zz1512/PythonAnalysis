"""
gpu_permutation.py (Memory Optimized Edition)
修复 CUDA OOM 问题：引入 Batching 分批计算机制。
将 5000 次置换拆分为小批次（如 1000 次/批），降低峰值显存占用。

用途
- 为二阶 GLM 提供基于置换的 cluster-FWE 推断：输出 cluster-level FWE 的 `-log10(p)` 图，
  以及对应的原始 T 统计图（用于可视化与阈值化）。

论文意义（这一层回答什么问题）
- 在“定位哪里”的层面上，通过非参数置换来控制全脑多重比较，回答：
  在 yy vs kj 对比下，哪些空间连续的簇在组水平上可靠显著（而非偶然噪声团块）。

方法
- 一样本 sign-flip 置换（每次对被试对比图随机乘以 ±1），在零假设（均值为 0）下生成最大簇大小分布。
- 以 cluster-forming 阈值（`cluster_p_thresh` -> t 阈值）先二值化，再用“最大簇大小”做 FWE 校正（max-statistic）。
- GPU 负责批量计算每次置换的体素 t 值；CPU 并行负责每个置换的 3D 聚类与最大簇提取。

输入
- `input_imgs`：每被试一张对比图（通常为一阶 GLM contrast 的 effect/t/z 图之一，具体由上游决定）。
- `mask_img`：组水平分析 mask（确保体素对齐且避免无效区域）。

输出
- `log_p_img`：cluster-level FWE 的 `-log10(p)` 图（同空间、同 mask）。
- `raw_t_img`：未阈值化的原始 T 图（用于绘图/QC）。

关键参数为何这样设
- `n_perms=5000`：经验上在计算成本与 p 值分辨率间折中（最小可达约 1/(n_perms+1)）。
- `cluster_p_thresh=0.001`：更严格的 cluster-forming 阈值，降低“噪声连成片”造成的假簇。
- `BATCH_SIZE=1000`：显存峰值控制参数；值越大越快但更易 OOM。

结果解读
- `log_p_img` 中值越大表示簇级 FWE p 越小；例如 `> -log10(0.05)≈1.3` 可作为显著性参考线。
- `raw_t_img` 可用于查看效应方向与峰值位置，但显著性应以 cluster-FWE 校正结果为准。

常见坑
- mask 与输入图的 affine/shape 不一致会导致体素错位（NiftiMasker 会报错或产生不可比结果）。
- `n_subs` 很小时（例如 < 10）置换分布不稳定，簇推断结果高度依赖阈值与样本。
- 将“定义 ROI 的同一对比图”再用于 ROI 内解释同一效应，会引入选择偏差；若要严格推断，需独立数据或 CV。
"""

import time
import logging
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.stats import t as t_dist
from nilearn.image import new_img_like
from nilearn.input_data import NiftiMasker
from joblib import Parallel, delayed

# PyTorch 检测
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        # 尝试优化显存分配策略 (减少碎片)
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        try:
            torch.set_float32_matmul_precision('medium')
        except AttributeError:
            pass
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

def compute_t_stats_kernel(data, signs, n_subs, sqrt_n, sum_x2):
    """
    T 值计算核心
    data: [n_subs, n_voxels]
    signs: [batch_size, n_subs]
    """
    # Mean
    # [batch, n_subs] @ [n_subs, n_voxels] -> [batch, n_voxels]
    sum_x = torch.mm(signs, data)
    mean_x = sum_x / n_subs

    # Var = (Sum(X^2) - N*Mean^2) / (N-1)
    # 注意：sum_x2 是预计算好的常量 [n_voxels]
    var_x = (sum_x2 - n_subs * (mean_x ** 2)) / (n_subs - 1)
    var_x = torch.clamp(var_x, min=1e-10)

    # T-value
    t_values = mean_x * sqrt_n / torch.sqrt(var_x)
    return t_values

def _find_max_cluster(binary_map, structure):
    """CPU 聚类函数"""
    labeled_array, num_features = ndimage.label(binary_map, structure=structure)
    if num_features == 0:
        return 0
    sizes = np.bincount(labeled_array.ravel())
    return sizes[1:].max() if len(sizes) > 1 else 0

def run_permutation(input_imgs, mask_img, n_perms=5000, cluster_p_thresh=0.001, n_jobs=8):
    """
    执行 GPU 加速置换检验 (分批版)
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    device = torch.device("cuda")
    # 获取显卡显存信息，动态调整 batch_size
    total_mem = torch.cuda.get_device_properties(0).total_memory
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} (Mem: {total_mem/1024**3:.1f} GB)")

    t0 = time.time()

    # 1. 加载数据 (CPU)
    masker = NiftiMasker(mask_img=mask_img).fit()
    data_np = masker.transform(input_imgs)
    n_subs, n_voxels = data_np.shape

    logger.info(f"Data shape: {n_subs} subjects x {n_voxels} voxels")

    # 2. 数据转 GPU
    # 注意：data_gpu 只有几百 MB，可以常驻显存
    data_gpu = torch.tensor(data_np, dtype=torch.float32, device=device)

    # 3. 预计算常量
    sum_x2 = torch.sum(data_gpu ** 2, dim=0)
    sqrt_n = float(np.sqrt(n_subs))

    # 计算 T 阈值
    df = n_subs - 1
    t_thresh = t_dist.isf(cluster_p_thresh, df)
    logger.info(f"Permutation Threshold (p<{cluster_p_thresh}): T > {t_thresh:.3f}")

    # =========================================================
    # 4. 分批计算 T 值 (Batch Processing) - 解决 OOM 的关键
    # =========================================================
    # 设定 Batch Size。保守点，设为 1000。
    # 1000 * 200k voxels * 4 bytes ≈ 800 MB，非常安全。
    BATCH_SIZE = 1000

    thresholded_maps_cpu_list = []
    raw_t_map_cpu = None

    logger.info(f"Starting Permutation Loop (Total: {n_perms}, Batch: {BATCH_SIZE})...")

    for i in range(0, n_perms, BATCH_SIZE):
        # 当前批次大小
        current_batch_size = min(BATCH_SIZE, n_perms - i)

        # 生成随机符号矩阵 [batch, n_subs]
        signs_batch = torch.randint(0, 2, (current_batch_size, n_subs),
                                  device=device, dtype=torch.float32) * 2 - 1

        # 强制第一个置换为原始数据 (真实标签)
        # 这样可以在同一次循环里同时得到“真实 T 图”和置换分布，避免再跑一遍。
        if i == 0:
            signs_batch[0, :] = 1.0

        # 计算 T 值 (GPU)
        t_vals_batch = compute_t_stats_kernel(data_gpu, signs_batch, n_subs, sqrt_n, sum_x2)

        # 记录原始 T 值图 (用于后续输出)
        if i == 0:
            raw_t_map_cpu = t_vals_batch[0].cpu().numpy()

        # 阈值处理 (GPU) -> bool
        thresh_batch = t_vals_batch > t_thresh

        # 移回 CPU 并清空 GPU 显存
        thresholded_maps_cpu_list.append(thresh_batch.cpu().numpy())

        # 手动清理显存，防止碎片堆积
        del signs_batch, t_vals_batch, thresh_batch
        torch.cuda.empty_cache()

        # 打印进度
        if (i + BATCH_SIZE) % 2000 == 0:
            logger.info(f"  Processed {i + current_batch_size}/{n_perms} permutations...")

    # 合并所有批次结果
    thresholded_maps_cpu = np.concatenate(thresholded_maps_cpu_list, axis=0)

    # 清理 GPU 常驻数据
    del data_gpu, sum_x2
    torch.cuda.empty_cache()

    # =========================================================
    # 5. CPU 并行聚类 (保持不变)
    # =========================================================
    t1 = time.time()
    logger.info(f"GPU Loop finished in {t1-t0:.2f}s. Starting CPU Clustering (n_jobs={n_jobs})...")

    mask_bool = masker.mask_img_.get_fdata() > 0
    mask_shape = mask_bool.shape
    structure = ndimage.generate_binary_structure(3, 3)

    def _process_perm(flat_binary_map):
        vol = np.zeros(mask_shape, dtype=bool)
        vol[mask_bool] = flat_binary_map
        return _find_max_cluster(vol, structure)

    # 这里的 max_sizes 可能很大，注意 CPU 内存
    # max-statistic：每次置换只保留“最大簇大小”，用于 cluster-level FWE 控制（全脑多重比较）。
    max_sizes = Parallel(n_jobs=n_jobs, batch_size=100)(
        delayed(_process_perm)(row) for row in thresholded_maps_cpu
    )
    max_sizes = np.array(max_sizes)

    # 6. 计算 P 值
    logger.info("Computing P-values...")

    # 计算真实数据的 Cluster
    raw_vol = np.zeros(mask_shape, dtype=bool)
    raw_vol[mask_bool] = (raw_t_map_cpu > t_thresh)
    labeled_array, num_features = ndimage.label(raw_vol, structure=structure)

    log_p_data = np.zeros(mask_shape)
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())
        # sizes[0] 是背景，从 1 开始
        for cluster_idx in range(1, len(sizes)):
            size = sizes[cluster_idx]
            # FWE P-value = (N_perms_with_max_cluster >= size + 1) / (N_perms + 1)
            p_fwe = (np.sum(max_sizes >= size) + 1) / (n_perms + 1)
            # 存储 -log10(p)
            log_p_data[labeled_array == cluster_idx] = -np.log10(p_fwe)

    # 生成 Nifti 对象
    log_p_img = new_img_like(mask_img, log_p_data)

    # 还原原始 T 图
    raw_t_data = np.zeros(mask_shape)
    raw_t_data[mask_bool] = raw_t_map_cpu
    raw_t_img = new_img_like(mask_img, raw_t_data)

    logger.info(f"Total Time: {time.time()-t0:.1f}s")
    return log_p_img, raw_t_img
