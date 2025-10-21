# config.py
# MVPA分析配置模块

import json
from pathlib import Path
from multiprocessing import cpu_count

class MVPAConfig:
    """MVPA分析配置参数"""
    def __init__(self):
        # 基本参数
        self.subjects = [f"sub-{i:02d}" for i in range(1, 29)]
        self.runs = [3]  # 支持多个run
        self.lss_root = Path(r"../../../learn_LSS")
        self.roi_dir = Path(r"../../../learn_mvpa/full_roi_mask")
        self.results_dir = Path(r"../../../learn_mvpa/metaphor_ROI_MVPA_corrected")
        
        # 分类器参数
        self.svm_params = {
            'random_state': 42
        }
        
        # 参数网格搜索配置
        self.svm_param_grid = {
            'C': [0.1, 1.0, 10.0],  # 可选的C值
            'kernel': ['linear']     # 保持线性核
        }
        
        # 交叉验证参数
        self.cv_folds = 5
        self.cv_random_state = 42
        
        # 置换检验参数（仅用于组水平分析）
        self.n_permutations = 1000
        self.permutation_random_state = 42
        
        # 显著性水平
        self.alpha_level = 0.05
        
        # 并行处理参数
        self.n_jobs = min(4, cpu_count() - 1)
        self.use_parallel = True
        
        # 内存优化参数
        self.memory_cache = 'nilearn_cache'
        self.memory_level = 1
        
        # 对比条件
        self.contrasts = [
            ('metaphor_vs_space', 'yy', 'kj'),  # 隐喻 vs 空间
        ]
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, filepath):
        """保存配置到JSON文件"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)