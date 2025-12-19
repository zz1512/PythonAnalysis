"""
gpu_permutation.py
使用 PyTorch 加速 fMRI 二阶分析的置换检验 (Cluster-level FWE)
针对 NVIDIA RTX 5060 Ti 优化
"""

import numpy as np
import sys
import pandas as pd
import nibabel as nib
from nilearn.image import new_img_like
from scipy.ndimage import label
import logging

# 尝试导入 PyTorch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)
# 导入配置
try:
    from glm_config_new import config
except ImportError:
    print("Error: config_enhance.py not found.")
    sys.exit(1)


def compute_t_stats_batch_gpu(data, signs):
    """
    在 GPU 上批量计算 T 统计量 (单样本 T 检验)
    data: [n_subjects, n_voxels]
    signs: [n_perms, n_subjects]  (-1 或 1)
    """
    n_perms, n_subs = signs.shape
    n_voxels = data.shape[1]

    # 将数据广播到每个置换
    # X: [n_perms, n_subjects, n_voxels]
    # 这里为了省显存，不直接扩展 data，而是利用矩阵乘法
    # Mean = (Signs * Data) / N

    # 转换为 PyTorch 张量
    # data: [n_subs, n_voxels]
    # signs: [n_perms, n_subs]

    # 1. 计算均值 (Mean)
    # [n_perms, n_subs] @ [n_subs, n_voxels] -> [n_perms, n_voxels]
    # sum(x) / n
    means = torch.mm(signs, data) / n_subs

    # 2. 计算标准差 (Std)
    # 这步比较 trick，标准差计算需要 (x - mean)^2
    # 由于我们是对每个 voxel 每个 perm 计算，直接展开显存可能不够 (5000 * 29 * 200000 * 4 bytes ≈ 116GB)
    # 所以我们必须分块计算或者使用流式计算
    # 但对于单样本 T 检验（零假设均值为0），随机翻转符号后的标准差其实等于原始数据的标准差！
    # 只要数据是以0为中心对称翻转的。
    # 等等，标准 T 统计量公式：T = Mean / (Std / sqrt(N))
    # 在符号翻转置换中，分母（方差）对于所有置换其实是不变的吗？
    # 不，符号翻转会改变均值，也会改变 (x - mean)^2。

    # 优化方案：分批次处理 Permutations，防止显存爆炸
    # 5060 Ti 显存约 8GB-16GB
    batch_size = 100  # 每次处理 100 个置换
    t_values_all = []

    sqrt_n = np.sqrt(n_subs)

    for i in range(0, n_perms, batch_size):
        end = min(i + batch_size, n_perms)
        batch_signs = signs[i:end, :]  # [batch, n_subs]

        # 扩展 data 到 batch 维度: [batch, n_subs, n_voxels]
        # 注意：这里会占用显存。
        # data [29, 200k] * 4 bytes ~ 23MB.
        # batch [100, 29, 200k] ~ 2.3GB. 5060 Ti (8G/16G) 吃得消。

        # data_expanded = data.unsqueeze(0) * batch_signs.unsqueeze(2)
        # 上面写法太费显存，利用广播

        # 实际计算: x_perm = data * sign
        # mean = x_perm.mean(dim=1)
        # std = x_perm.std(dim=1, unbiased=True)

        # 为了速度，我们手动实现 std
        # batch_data: [batch, n_subs, n_voxels]
        batch_data = data.unsqueeze(0) * batch_signs.unsqueeze(2)

        # Mean & Std
        b_mean = batch_data.mean(dim=1)
        b_std = batch_data.std(dim=1, unbiased=True)

        # T-value
        # 避免除以0
        b_t = b_mean / (b_std / sqrt_n + 1e-16)

        t_values_all.append(b_t)  # Keep on GPU for now

    return torch.cat(t_values_all, dim=0)


def run_gpu_permutation_test(second_level_input, mask_img, n_permutations=5000,
                             cluster_forming_p=0.001, output_dir=None):
    """
    主函数：执行 GPU 加速的置换检验
    """
    # [优化] 设置随机种子，确保结果可复现
    if config.RANDOM_SEED:
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)

    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed. Cannot use GPU acceleration.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logger.warning("CUDA not available. Falling back to CPU (will be slow).")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. 准备数据
    from nilearn.input_data import NiftiMasker
    masker = NiftiMasker(mask_img=mask_img)
    data_np = masker.fit_transform(second_level_input)  # [n_subjects, n_voxels]

    n_subs, n_voxels = data_np.shape
    logger.info(f"Data shape: {n_subs} subjects, {n_voxels} voxels")

    # 转为 Tensor
    data_gpu = torch.tensor(data_np, dtype=torch.float32, device=device)

    # 2. 生成随机符号翻转矩阵 (Sign Flipping)
    # 原始数据（不翻转）作为第一个
    # [n_perms, n_subs]
    logger.info(f"Generating {n_permutations} permutations...")
    # 随机生成 0 和 1，然后映射到 -1 和 1
    random_signs = torch.randint(0, 2, (n_permutations, n_subs), device=device).float() * 2 - 1
    # 确保第一次是不翻转的 (全1)
    random_signs[0, :] = 1.0

    # 3. 计算 T 值 (GPU Heavy Lifting)
    logger.info("Computing T-statistics on GPU...")
    with torch.no_grad():
        t_values_map = compute_t_stats_batch_gpu(data_gpu, random_signs)

    # 4. 阈值处理与簇大小计算 (CPU Hybrid)
    # 将 T 值转回 CPU 计算 clustering，因为 scipy.ndimage.label 只能在 CPU 跑
    # 这里我们按批次传回 CPU，防止内存爆炸

    from scipy.stats import t
    # 计算 Cluster-forming threshold (Z-score or T-score?)
    # Nilearn usually expects T-score threshold corresponding to p < 0.001
    df = n_subs - 1
    t_threshold = t.isf(cluster_forming_p, df)
    logger.info(f"Cluster forming T-threshold: {t_threshold:.3f} (p<{cluster_forming_p})")

    # 预分配最大簇大小数组
    max_cluster_sizes = np.zeros(n_permutations)

    # 获取 Mask 的 3D 形状，用于还原图像进行 label
    mask_data = mask_img.get_fdata() > 0
    mask_shape = mask_data.shape

    # 这是一个比较耗时的部分，但比算 T 值快
    logger.info("Calculating cluster sizes (CPU)...")

    # 将 T 值转为 Numpy (如果显存够大，可以一次性转；如果不够，分块)
    # 5000 * 200k * 4 bytes = 4GB. 内存肯定够。
    t_values_np = t_values_map.cpu().numpy()

    # 原始数据的 T map (第0个)
    original_t_map = t_values_np[0, :]

    # 循环计算每个置换的最大簇
    # 我们可以利用 multiprocessing 加速这部分 CPU 操作
    from joblib import Parallel, delayed

    def _get_max_cluster(t_map_1d, mask_shape, mask_bool, thresh):
        # 1. 阈值处理
        bin_map = np.zeros(mask_shape, dtype=bool)
        # 仅填充 mask 区域
        bin_map[mask_bool] = t_map_1d > thresh

        # 2. 连通分量标记
        labeled, n_labels = label(bin_map)

        if n_labels == 0:
            return 0

        # 3. 计算簇大小
        # bincount 比 unique 快得多
        sizes = np.bincount(labeled.ravel())
        # 第0个是背景，去掉
        if len(sizes) > 1:
            return sizes[1:].max()
        else:
            return 0

    # 并行计算 Cluster Sizes
    results = Parallel(n_jobs=-1, batch_size=100)(
        delayed(_get_max_cluster)(t_values_np[i], mask_shape, mask_data, t_threshold)
        for i in range(n_permutations)
    )

    max_cluster_sizes = np.array(results)

    # [优化] 保存原始的 T 统计图 (Ground Truth T-map)
    # 这张图是不经校正的 T 值，展示了原始效应强度
    from nilearn.image import new_img_like

    # 我们需要先算一次不带翻转的 T 值 (即 random_signs[0])
    # 由于 compute_t_stats_batch_gpu 是批量算的，t_values_map[0] 就是原始数据的 T 值

    # 这里需要把 t_values_map[0] 映射回 3D 空间
    t_map_3d = np.zeros(mask_shape)
    t_map_3d[mask_data] = t_values_np[0, :]  # t_values_np 是从 GPU 下回来的

    t_map_img = new_img_like(mask_img, t_map_3d)
    t_map_img.to_filename(output_dir / "group_t_stat_unthresholded.nii.gz")

    logger.info("Saved unthresholded T-map.")

    # 5. 计算 P 值
    # 对于原始图中的每个簇，看它的大小在 max_cluster_sizes 分布中的位置

    # 先计算原始图的 clusters
    bin_map_orig = np.zeros(mask_shape, dtype=bool)
    bin_map_orig[mask_data] = original_t_map > t_threshold
    labeled_orig, n_labels_orig = label(bin_map_orig)

    # 结果 Map: -log10(p_corrected)
    # 初始化为 0
    p_map_3d = np.zeros(mask_shape)

    if n_labels_orig > 0:
        sizes_orig = np.bincount(labeled_orig.ravel())

        for i in range(1, len(sizes_orig)):
            cluster_size = sizes_orig[i]
            # 计算 FWE p-value: (大于等于该大小的随机簇的比例)
            # 加上 1 是为了避免 p=0 (permutation test standard)
            n_greater = np.sum(max_cluster_sizes >= cluster_size)
            p_fwe = (n_greater + 1) / (n_permutations + 1)

            # 转换为 -log10
            neg_log_p = -np.log10(p_fwe)

            # 填回 3D 图
            p_map_3d[labeled_orig == i] = neg_log_p

    # 生成 Nifti 对象
    p_map_img = new_img_like(mask_img, p_map_3d)

    return p_map_img