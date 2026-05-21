#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型蒸馏主运行脚本

这个脚本提供了一个统一的接口来运行不同类型的模型蒸馏实验。
支持合成数据、图像分类和文本分类的蒸馏实验。

使用方法:
    python run_distillation.py --experiment synthetic_medium
    python run_distillation.py --experiment image_quick
    python run_distillation.py --experiment text_quick
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, list_available_configs, print_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distillation_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_synthetic_experiment(config) -> Dict[str, Any]:
    """运行合成数据蒸馏实验"""
    logger.info("开始合成数据蒸馏实验")
    
    try:
        from model_distillation import (
            TeacherModel, StudentModel, ModelDistillation,
            create_synthetic_dataset, compare_models
        )
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # 创建数据集
        X_train, X_test, y_train, y_test = create_synthetic_dataset(
            n_samples=config.n_samples,
            n_features=config.n_features,
            n_classes=config.n_classes
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # 创建模型
        teacher_model = TeacherModel(
            input_dim=config.n_features,
            hidden_dims=config.teacher_hidden_dims,
            num_classes=config.n_classes
        )
        
        student_model = StudentModel(
            input_dim=config.n_features,
            hidden_dims=config.student_hidden_dims,
            num_classes=config.n_classes
        )
        
        # 创建蒸馏器
        distiller = ModelDistillation(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=config.temperature,
            alpha=config.alpha,
            device=config.device
        )
        
        # 训练教师模型
        distiller.train_teacher(train_loader, epochs=config.teacher_epochs)
        
        # 蒸馏训练学生模型
        distiller.distill_student(train_loader, test_loader, epochs=config.student_epochs)
        
        # 比较模型性能
        results = compare_models(teacher_model, student_model, test_loader, config.device)
        
        # 绘制训练历史
        if config.plot_results:
            distiller.plot_training_history()
        
        logger.info("合成数据蒸馏实验完成")
        return results
        
    except Exception as e:
        logger.error(f"合成数据实验失败: {e}")
        raise

def run_image_experiment(config) -> Dict[str, Any]:
    """运行图像分类蒸馏实验"""
    logger.info("开始图像分类蒸馏实验")
    
    try:
        from image_distillation import (
            SimpleTeacherCNN, SimpleStudentCNN, ImageDistillationTrainer,
            get_cifar10_data, compare_model_performance, visualize_samples
        )
        
        # 获取数据
        train_loader, val_loader, test_loader = get_cifar10_data(batch_size=config.batch_size)
        
        # 可视化样本
        if config.plot_results:
            visualize_samples(train_loader)
        
        # 创建模型
        teacher_model = SimpleTeacherCNN(num_classes=config.num_classes)
        student_model = SimpleStudentCNN(num_classes=config.num_classes)
        
        # 创建训练器
        trainer = ImageDistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=config.temperature,
            alpha=config.alpha,
            device=config.device
        )
        
        # 训练教师模型
        trainer.train_teacher(train_loader, val_loader, epochs=config.teacher_epochs)
        
        # 蒸馏训练学生模型
        trainer.distill_student(train_loader, val_loader, epochs=config.student_epochs)
        
        # 比较模型性能
        results = compare_model_performance(teacher_model, student_model, test_loader, config.device)
        
        # 绘制训练曲线
        if config.plot_results:
            trainer.plot_training_curves()
        
        logger.info("图像分类蒸馏实验完成")
        return results
        
    except Exception as e:
        logger.error(f"图像分类实验失败: {e}")
        raise

def run_text_experiment(config) -> Dict[str, Any]:
    """运行文本分类蒸馏实验"""
    logger.info("开始文本分类蒸馏实验")
    
    try:
        from text_distillation import (
            TeacherBERT, StudentLSTM, TextDistillationTrainer,
            prepare_text_data, TextDataset, SimpleLSTMDataset, SimpleVocabulary
        )
        from transformers import BertTokenizer
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
        
        # 准备数据
        texts, labels, categories = prepare_text_data(subset_size=config.subset_size)
        
        # 划分数据集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=config.test_size, random_state=42, stratify=labels
        )
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=config.val_size, random_state=42, stratify=train_labels
        )
        
        # 创建BERT数据集
        tokenizer = BertTokenizer.from_pretrained(config.teacher_model_name)
        
        teacher_train_dataset = TextDataset(train_texts, train_labels, tokenizer, config.max_length)
        teacher_val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_length)
        teacher_test_dataset = TextDataset(test_texts, test_labels, tokenizer, config.max_length)
        
        teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=config.teacher_batch_size, shuffle=True)
        teacher_val_loader = DataLoader(teacher_val_dataset, batch_size=config.teacher_batch_size, shuffle=False)
        teacher_test_loader = DataLoader(teacher_test_dataset, batch_size=config.teacher_batch_size, shuffle=False)
        
        # 创建LSTM数据集
        vocab = SimpleVocabulary(train_texts, max_vocab_size=config.vocab_size)
        
        student_train_dataset = SimpleLSTMDataset(train_texts, train_labels, vocab, config.max_length)
        student_val_dataset = SimpleLSTMDataset(val_texts, val_labels, vocab, config.max_length)
        student_test_dataset = SimpleLSTMDataset(test_texts, test_labels, vocab, config.max_length)
        
        student_train_loader = DataLoader(student_train_dataset, batch_size=config.student_batch_size, shuffle=True)
        student_val_loader = DataLoader(student_val_dataset, batch_size=config.student_batch_size, shuffle=False)
        student_test_loader = DataLoader(student_test_dataset, batch_size=config.student_batch_size, shuffle=False)
        
        # 创建模型
        num_classes = len(categories)
        
        teacher_model = TeacherBERT(num_classes=num_classes, model_name=config.teacher_model_name)
        student_model = StudentLSTM(
            vocab_size=len(vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_classes=num_classes,
            num_layers=config.num_layers
        )
        
        # 创建训练器
        trainer = TextDistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=config.temperature,
            alpha=config.alpha,
            device=config.device
        )
        
        # 训练教师模型
        trainer.train_teacher(teacher_train_loader, teacher_val_loader, epochs=config.teacher_epochs)
        
        # 蒸馏训练学生模型
        trainer.distill_student(
            teacher_train_loader,
            student_train_loader,
            student_val_loader,
            epochs=config.student_epochs
        )
        
        # 评估最终性能
        teacher_acc = trainer._evaluate_teacher(teacher_test_loader)
        student_acc = trainer._evaluate_student(student_test_loader)
        
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        results = {
            'teacher_accuracy': teacher_acc,
            'student_accuracy': student_acc,
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': teacher_params / student_params,
            'accuracy_retention': student_acc / teacher_acc if teacher_acc > 0 else 0
        }
        
        # 绘制训练曲线
        if config.plot_results:
            trainer.plot_training_curves()
        
        logger.info("文本分类蒸馏实验完成")
        return results
        
    except Exception as e:
        logger.error(f"文本分类实验失败: {e}")
        if "transformers" in str(e):
            logger.info("请安装transformers库: pip install transformers")
        raise

def run_experiment(experiment_name: str) -> Dict[str, Any]:
    """运行指定的蒸馏实验"""
    logger.info(f"开始运行实验: {experiment_name}")
    
    # 获取配置
    config = get_config(experiment_name)
    
    # 打印配置信息
    print_config(config)
    
    # 记录开始时间
    start_time = time.time()
    
    # 根据实验类型运行相应的实验
    if 'synthetic' in experiment_name:
        results = run_synthetic_experiment(config)
    elif 'image' in experiment_name:
        results = run_image_experiment(config)
    elif 'text' in experiment_name:
        results = run_text_experiment(config)
    else:
        raise ValueError(f"未知的实验类型: {experiment_name}")
    
    # 记录结束时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 添加实验信息到结果中
    results['experiment_name'] = experiment_name
    results['duration_seconds'] = duration
    results['config'] = config.__dict__
    
    # 打印结果摘要
    print_results_summary(results)
    
    logger.info(f"实验 {experiment_name} 完成，耗时: {duration:.2f} 秒")
    
    return results

def print_results_summary(results: Dict[str, Any]) -> None:
    """打印实验结果摘要"""
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    
    print(f"实验名称: {results.get('experiment_name', 'Unknown')}")
    print(f"实验耗时: {results.get('duration_seconds', 0):.2f} 秒")
    print()
    
    if 'teacher_accuracy' in results:
        print(f"教师模型准确率: {results['teacher_accuracy']:.2f}%")
    
    if 'student_accuracy' in results:
        print(f"学生模型准确率: {results['student_accuracy']:.2f}%")
    
    if 'teacher_parameters' in results:
        print(f"教师模型参数量: {results['teacher_parameters']:,}")
    
    if 'student_parameters' in results:
        print(f"学生模型参数量: {results['student_parameters']:,}")
    
    if 'compression_ratio' in results:
        print(f"模型压缩比: {results['compression_ratio']:.2f}x")
    
    if 'accuracy_retention' in results:
        print(f"准确率保持: {results['accuracy_retention']:.3f}")
    
    print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='模型蒸馏实验运行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_distillation.py --experiment synthetic_medium
  python run_distillation.py --experiment image_quick
  python run_distillation.py --experiment text_quick
  python run_distillation.py --list-configs
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='要运行的实验名称'
    )
    
    parser.add_argument(
        '--list-configs', '-l',
        action='store_true',
        help='列出所有可用的实验配置'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='不绘制结果图表'
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        print("可用的实验配置:")
        for config_name in list_available_configs():
            print(f"  - {config_name}")
        return
    
    if not args.experiment:
        parser.print_help()
        print("\n错误: 请指定要运行的实验名称")
        print("使用 --list-configs 查看可用的配置")
        sys.exit(1)
    
    try:
        # 修改配置以禁用绘图（如果指定）
        if args.no_plot:
            config = get_config(args.experiment)
            config.plot_results = False
        
        # 运行实验
        results = run_experiment(args.experiment)
        
        # 保存结果（可选）
        import json
        results_file = f"results_{args.experiment}_{int(time.time())}.json"
        
        # 将不可序列化的对象转换为字符串
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存到: {results_file}")
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"实验失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()