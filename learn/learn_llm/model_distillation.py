#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型蒸馏学习示例

这个脚本演示了如何使用知识蒸馏技术训练一个小的学生模型来模仿大的教师模型。
知识蒸馏是一种模型压缩技术，通过让小模型学习大模型的"软标签"来提高性能。

主要概念：
1. 教师模型（Teacher Model）：预训练的大模型，性能较好
2. 学生模型（Student Model）：要训练的小模型，参数更少
3. 蒸馏损失（Distillation Loss）：结合硬标签和软标签的损失函数
4. 温度参数（Temperature）：控制软标签的平滑程度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeacherModel(nn.Module):
    """教师模型：一个较大的神经网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super(TeacherModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建多层网络
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class StudentModel(nn.Module):
    """学生模型：一个较小的神经网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super(StudentModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建较小的网络
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DistillationLoss(nn.Module):
    """蒸馏损失函数"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        """
        Args:
            temperature: 温度参数，控制软标签的平滑程度
            alpha: 蒸馏损失的权重，(1-alpha)为硬标签损失的权重
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                true_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型的输出
            teacher_logits: 教师模型的输出
            true_labels: 真实标签
            
        Returns:
            总损失和各部分损失的字典
        """
        # 软标签损失（KL散度）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, true_labels)
        
        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item()
        }
        
        return total_loss, loss_dict

class ModelDistillation:
    """模型蒸馏训练器"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 temperature: float = 3.0, alpha: float = 0.7, device: str = 'cpu'):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        
        # 蒸馏损失函数
        self.distill_loss = DistillationLoss(temperature, alpha)
        
        # 优化器
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        
        # 训练历史
        self.train_history = {
            'total_loss': [],
            'soft_loss': [],
            'hard_loss': [],
            'accuracy': []
        }
        
    def train_teacher(self, train_loader: DataLoader, epochs: int = 50) -> None:
        """训练教师模型"""
        logger.info("开始训练教师模型...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.teacher_model.parameters(), lr=0.001)
        
        self.teacher_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.teacher_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            if (epoch + 1) % 10 == 0:
                accuracy = 100. * correct / total
                logger.info(f'教师模型 Epoch {epoch+1}/{epochs}, '
                          f'Loss: {total_loss/len(train_loader):.4f}, '
                          f'Accuracy: {accuracy:.2f}%')
        
        logger.info("教师模型训练完成")
    
    def distill_student(self, train_loader: DataLoader, val_loader: DataLoader, 
                       epochs: int = 100) -> None:
        """使用知识蒸馏训练学生模型"""
        logger.info("开始知识蒸馏训练学生模型...")
        
        # 教师模型设为评估模式
        self.teacher_model.eval()
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.student_model.train()
            total_loss_dict = {'total_loss': 0, 'soft_loss': 0, 'hard_loss': 0}
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 获取教师和学生的输出
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                
                student_logits = self.student_model(data)
                
                # 计算蒸馏损失
                loss, loss_dict = self.distill_loss(student_logits, teacher_logits, target)
                
                loss.backward()
                self.optimizer.step()
                
                # 统计
                for key in total_loss_dict:
                    total_loss_dict[key] += loss_dict[key]
                
                pred = student_logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            # 验证阶段
            val_acc = self.evaluate(val_loader)
            
            # 记录训练历史
            train_acc = 100. * correct / total
            for key in total_loss_dict:
                total_loss_dict[key] /= len(train_loader)
                self.train_history[key].append(total_loss_dict[key])
            self.train_history['accuracy'].append(train_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student_model.state_dict(), 'best_student_model.pth')
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'学生模型 Epoch {epoch+1}/{epochs}, '
                          f'Total Loss: {total_loss_dict["total_loss"]:.4f}, '
                          f'Soft Loss: {total_loss_dict["soft_loss"]:.4f}, '
                          f'Hard Loss: {total_loss_dict["hard_loss"]:.4f}, '
                          f'Train Acc: {train_acc:.2f}%, '
                          f'Val Acc: {val_acc:.2f}%')
        
        logger.info(f"学生模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """评估模型性能"""
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.student_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def plot_training_history(self) -> None:
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 总损失
        axes[0, 0].plot(self.train_history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 软损失
        axes[0, 1].plot(self.train_history['soft_loss'], label='Soft Loss')
        axes[0, 1].plot(self.train_history['hard_loss'], label='Hard Loss')
        axes[0, 1].set_title('Soft vs Hard Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 准确率
        axes[1, 0].plot(self.train_history['accuracy'])
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].grid(True)
        
        # 损失比例
        axes[1, 1].plot([s/(s+h) for s, h in zip(self.train_history['soft_loss'], 
                                                 self.train_history['hard_loss'])])
        axes[1, 1].set_title('Soft Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Soft Loss / Total Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/bytedance/PycharmProjects/PythonAnalysis/learn/learn_llm/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_synthetic_dataset(n_samples: int = 5000, n_features: int = 20, 
                           n_classes: int = 5) -> Tuple[torch.Tensor, torch.Tensor, 
                                                       torch.Tensor, torch.Tensor]:
    """创建合成数据集"""
    logger.info(f"创建合成数据集: {n_samples} 样本, {n_features} 特征, {n_classes} 类别")
    
    # 生成分类数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test

def compare_models(teacher_model: nn.Module, student_model: nn.Module, 
                  test_loader: DataLoader, device: str = 'cpu') -> Dict[str, float]:
    """比较教师模型和学生模型的性能"""
    logger.info("比较模型性能...")
    
    def evaluate_model(model):
        model.eval()
        correct = 0
        total = 0
        total_params = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy, total_params
    
    teacher_acc, teacher_params = evaluate_model(teacher_model)
    student_acc, student_params = evaluate_model(student_model)
    
    results = {
        'teacher_accuracy': teacher_acc,
        'student_accuracy': student_acc,
        'teacher_parameters': teacher_params,
        'student_parameters': student_params,
        'compression_ratio': teacher_params / student_params,
        'accuracy_retention': student_acc / teacher_acc
    }
    
    logger.info(f"教师模型: 准确率 {teacher_acc:.2f}%, 参数量 {teacher_params:,}")
    logger.info(f"学生模型: 准确率 {student_acc:.2f}%, 参数量 {student_params:,}")
    logger.info(f"压缩比: {results['compression_ratio']:.2f}x")
    logger.info(f"准确率保持: {results['accuracy_retention']:.2f}")
    
    return results

def main():
    """主函数：演示完整的模型蒸馏流程"""
    logger.info("开始模型蒸馏学习示例")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    X_train, X_test, y_train, y_test = create_synthetic_dataset()
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 定义模型架构
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    # 教师模型（大模型）
    teacher_model = TeacherModel(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],  # 较大的隐藏层
        num_classes=num_classes
    )
    
    # 学生模型（小模型）
    student_model = StudentModel(
        input_dim=input_dim,
        hidden_dims=[64, 32],  # 较小的隐藏层
        num_classes=num_classes
    )
    
    logger.info(f"教师模型参数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
    logger.info(f"学生模型参数量: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # 创建蒸馏器
    distiller = ModelDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=3.0,
        alpha=0.7,
        device=device
    )
    
    # 训练教师模型
    distiller.train_teacher(train_loader, epochs=50)
    
    # 蒸馏训练学生模型
    distiller.distill_student(train_loader, test_loader, epochs=100)
    
    # 比较模型性能
    results = compare_models(teacher_model, student_model, test_loader, device)
    
    # 绘制训练历史
    distiller.plot_training_history()
    
    logger.info("模型蒸馏学习示例完成！")
    
    return results

if __name__ == "__main__":
    results = main()