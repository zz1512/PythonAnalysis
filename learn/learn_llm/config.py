#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型蒸馏配置文件

这个文件包含了不同蒸馏实验的配置参数。
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class BaseConfig:
    """基础配置类"""
    # 设备配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    
    # 训练配置
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # 蒸馏配置
    temperature: float = 3.0
    alpha: float = 0.7  # 软损失权重
    
    # 日志配置
    log_interval: int = 10
    save_model: bool = True
    plot_results: bool = True

@dataclass
class SyntheticDataConfig(BaseConfig):
    """合成数据实验配置"""
    # 数据配置
    n_samples: int = 5000
    n_features: int = 20
    n_classes: int = 5
    test_size: float = 0.2
    
    # 教师模型配置
    teacher_hidden_dims: List[int] = None
    
    # 学生模型配置
    student_hidden_dims: List[int] = None
    
    # 训练配置
    teacher_epochs: int = 50
    student_epochs: int = 100
    
    def __post_init__(self):
        if self.teacher_hidden_dims is None:
            self.teacher_hidden_dims = [256, 128, 64]
        if self.student_hidden_dims is None:
            self.student_hidden_dims = [64, 32]

@dataclass
class ImageConfig(BaseConfig):
    """图像分类实验配置"""
    # 数据配置
    dataset: str = 'CIFAR10'
    data_dir: str = './data'
    num_workers: int = 2
    
    # 模型配置
    num_classes: int = 10
    
    # 训练配置
    teacher_epochs: int = 50
    student_epochs: int = 80
    batch_size: int = 128
    
    # 蒸馏配置
    temperature: float = 4.0
    alpha: float = 0.8
    
    # 优化器配置
    learning_rate: float = 0.001
    step_size: int = 30
    gamma: float = 0.1

@dataclass
class TextConfig(BaseConfig):
    """文本分类实验配置"""
    # 数据配置
    dataset: str = '20newsgroups'
    subset_size: int = 1000
    max_length: int = 128
    test_size: float = 0.2
    val_size: float = 0.2
    
    # 教师模型配置（BERT）
    teacher_model_name: str = 'bert-base-uncased'
    teacher_learning_rate: float = 2e-5
    teacher_epochs: int = 3
    teacher_batch_size: int = 16
    
    # 学生模型配置（LSTM）
    vocab_size: int = 5000
    embedding_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 2
    student_epochs: int = 20
    student_batch_size: int = 16
    
    # 蒸馏配置
    temperature: float = 5.0
    alpha: float = 0.7
    
    # 优化器配置
    step_size: int = 10
    gamma: float = 0.5

# 预定义的实验配置
EXPERIMENT_CONFIGS = {
    'synthetic_small': SyntheticDataConfig(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        teacher_hidden_dims=[128, 64],
        student_hidden_dims=[32],
        teacher_epochs=30,
        student_epochs=50,
        temperature=2.0,
        alpha=0.6
    ),
    
    'synthetic_medium': SyntheticDataConfig(
        n_samples=5000,
        n_features=20,
        n_classes=5,
        teacher_hidden_dims=[256, 128, 64],
        student_hidden_dims=[64, 32],
        teacher_epochs=50,
        student_epochs=100,
        temperature=3.0,
        alpha=0.7
    ),
    
    'synthetic_large': SyntheticDataConfig(
        n_samples=10000,
        n_features=50,
        n_classes=10,
        teacher_hidden_dims=[512, 256, 128, 64],
        student_hidden_dims=[128, 64],
        teacher_epochs=80,
        student_epochs=150,
        temperature=4.0,
        alpha=0.8
    ),
    
    'image_quick': ImageConfig(
        teacher_epochs=20,
        student_epochs=30,
        batch_size=64,
        temperature=3.0,
        alpha=0.7
    ),
    
    'image_standard': ImageConfig(
        teacher_epochs=50,
        student_epochs=80,
        batch_size=128,
        temperature=4.0,
        alpha=0.8
    ),
    
    'text_quick': TextConfig(
        subset_size=500,
        teacher_epochs=2,
        student_epochs=10,
        temperature=4.0,
        alpha=0.6
    ),
    
    'text_standard': TextConfig(
        subset_size=1000,
        teacher_epochs=3,
        student_epochs=20,
        temperature=5.0,
        alpha=0.7
    )
}

def get_config(experiment_name: str) -> BaseConfig:
    """获取指定实验的配置"""
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"未知的实验配置: {experiment_name}. "
                        f"可用配置: {list(EXPERIMENT_CONFIGS.keys())}")
    
    return EXPERIMENT_CONFIGS[experiment_name]

def list_available_configs() -> List[str]:
    """列出所有可用的配置"""
    return list(EXPERIMENT_CONFIGS.keys())

def print_config(config: BaseConfig) -> None:
    """打印配置信息"""
    print(f"配置类型: {type(config).__name__}")
    print("=" * 50)
    
    for field_name, field_value in config.__dict__.items():
        print(f"{field_name}: {field_value}")
    
    print("=" * 50)

# 温度参数实验配置
TEMPERATURE_EXPERIMENTS = {
    'temp_1': {'temperature': 1.0, 'alpha': 0.7},
    'temp_2': {'temperature': 2.0, 'alpha': 0.7},
    'temp_3': {'temperature': 3.0, 'alpha': 0.7},
    'temp_4': {'temperature': 4.0, 'alpha': 0.7},
    'temp_5': {'temperature': 5.0, 'alpha': 0.7},
    'temp_10': {'temperature': 10.0, 'alpha': 0.7},
}

# Alpha参数实验配置
ALPHA_EXPERIMENTS = {
    'alpha_0.1': {'temperature': 3.0, 'alpha': 0.1},
    'alpha_0.3': {'temperature': 3.0, 'alpha': 0.3},
    'alpha_0.5': {'temperature': 3.0, 'alpha': 0.5},
    'alpha_0.7': {'temperature': 3.0, 'alpha': 0.7},
    'alpha_0.9': {'temperature': 3.0, 'alpha': 0.9},
}

def create_parameter_sweep_configs(base_config: BaseConfig, 
                                 parameter_experiments: Dict[str, Dict[str, Any]]) -> Dict[str, BaseConfig]:
    """创建参数扫描实验配置"""
    configs = {}
    
    for exp_name, params in parameter_experiments.items():
        # 复制基础配置
        config = type(base_config)(**base_config.__dict__)
        
        # 更新参数
        for param_name, param_value in params.items():
            setattr(config, param_name, param_value)
        
        configs[exp_name] = config
    
    return configs

if __name__ == "__main__":
    # 演示配置使用
    print("可用的实验配置:")
    for config_name in list_available_configs():
        print(f"  - {config_name}")
    
    print("\n" + "="*60)
    print("示例配置 (synthetic_medium):")
    config = get_config('synthetic_medium')
    print_config(config)
    
    print("\n" + "="*60)
    print("温度参数扫描实验:")
    temp_configs = create_parameter_sweep_configs(
        get_config('synthetic_medium'), 
        TEMPERATURE_EXPERIMENTS
    )
    
    for name, config in temp_configs.items():
        print(f"{name}: temperature={config.temperature}, alpha={config.alpha}")