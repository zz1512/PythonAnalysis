from nilearn import image, plotting, datasets
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table
import pandas as pd

stat_path = r"K:\metaphor\Activation\2nd_level\run3\yy_vs_kj\spmT_0001.nii.gz" 
fmri_img = image.load_img(stat_path)
fmri_data = fmri_img.get_fdata()
'''
mni_t1 = datasets.load_mni152_template()

# FDR 控制显著性，过滤掉 <20 体素的小簇；双侧检验
thresh_img, threshold = threshold_stats_img(
    stat_path, alpha=0.05, height_control='fdr',
    cluster_threshold=20, two_sided=True
)
print("FDR 阈值 =", threshold)

display = plotting.plot_stat_map(
    thresh_img, bg_img=mni_t1, display_mode='ortho',
    cut_coords=None, black_bg=True, dim=0.3
)
plotting.show()

# 可选：导出显著簇表（峰值坐标、簇大小、峰值统计量等）
table = get_clusters_table(thresh_img, stat_threshold=threshold, cluster_threshold=20)
print(table)            # 终端查看
# pd.DataFrame(table).to_csv("clusters.csv", index=False)  # 需要时保存'''