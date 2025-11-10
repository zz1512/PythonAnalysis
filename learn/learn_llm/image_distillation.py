#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类的模型蒸馏示例

这个脚本演示了如何在CIFAR-10数据集上使用知识蒸馏技术。
使用ResNet作为教师模型，简单的CNN作为学生模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTeacherCNN(nn.Module):
    """教师模型：较复杂的CNN"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleTeacherCNN, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleStudentCNN(nn.Module):
    """学生模型：较简单的CNN"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleStudentCNN, self).__init__()
        
        # 特征提取层（更简单）
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ImageDistillationLoss(nn.Module):
    """图像分类的蒸馏损失"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.8):
        super(ImageDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                true_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 软标签损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, true_labels)
        
        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item()
        }

class ImageDistillationTrainer:
    """图像分类的蒸馏训练器"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.8, device: str = 'cpu'):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        
        self.distill_loss = ImageDistillationLoss(temperature, alpha)
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        self.train_history = {
            'total_loss': [],
            'soft_loss': [],
            'hard_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
    def train_teacher(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """训练教师模型"""
        logger.info("开始训练教师模型...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.teacher_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.teacher_model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.teacher_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            # 验证阶段
            val_acc = self._evaluate_model(self.teacher_model, val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.teacher_model.state_dict(), 'best_teacher_model.pth')
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                train_acc = 100. * correct / total
                logger.info(f'教师模型 Epoch {epoch+1}/{epochs}, '
                          f'Loss: {train_loss/len(train_loader):.4f}, '
                          f'Train Acc: {train_acc:.2f}%, '
                          f'Val Acc: {val_acc:.2f}%')
        
        logger.info(f"教师模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    
    def distill_student(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 80):
        """蒸馏训练学生模型"""
        logger.info("开始蒸馏训练学生模型...")
        
        self.teacher_model.eval()  # 教师模型设为评估模式
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
            val_acc = self._evaluate_model(self.student_model, val_loader)
            
            # 记录训练历史
            train_acc = 100. * correct / total
            for key in total_loss_dict:
                total_loss_dict[key] /= len(train_loader)
                self.train_history[key].append(total_loss_dict[key])
            
            self.train_history['train_accuracy'].append(train_acc)
            self.train_history['val_accuracy'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student_model.state_dict(), 'best_student_model.pth')
            
            self.scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'学生模型 Epoch {epoch+1}/{epochs}, '
                          f'Total Loss: {total_loss_dict["total_loss"]:.4f}, '
                          f'Train Acc: {train_acc:.2f}%, '
                          f'Val Acc: {val_acc:.2f}%')
        
        logger.info(f"学生模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return 100. * correct / total
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['total_loss'], label='Total Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 软损失 vs 硬损失
        axes[0, 1].plot(self.train_history['soft_loss'], label='Soft Loss (KD)')
        axes[0, 1].plot(self.train_history['hard_loss'], label='Hard Loss (CE)')
        axes[0, 1].set_title('Soft vs Hard Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 准确率曲线
        axes[1, 0].plot(self.train_history['train_accuracy'], label='Train Accuracy')
        axes[1, 0].plot(self.train_history['val_accuracy'], label='Validation Accuracy')
        axes[1, 0].set_title('Accuracy Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率变化（如果有的话）
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [self.scheduler.get_last_lr()[0] for _ in range(len(self.train_history['total_loss']))]
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('/Users/bytedance/PycharmProjects/PythonAnalysis/learn/learn_llm/image_distillation_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def get_cifar10_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """获取CIFAR-10数据集"""
    logger.info("加载CIFAR-10数据集...")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 下载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 创建验证集（从训练集中分出一部分）
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(testset)}")
    
    return train_loader, val_loader, test_loader

def visualize_samples(data_loader: DataLoader, num_samples: int = 8):
    """可视化数据样本"""
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 获取一个批次的数据
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # 反标准化
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i]
        # 反标准化
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {classes[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/bytedance/PycharmProjects/PythonAnalysis/learn/learn_llm/cifar10_samples.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def compare_model_performance(teacher_model: nn.Module, student_model: nn.Module, 
                            test_loader: DataLoader, device: str = 'cpu') -> Dict[str, float]:
    """比较教师模型和学生模型的性能"""
    logger.info("比较模型性能...")
    
    def evaluate_and_count_params(model):
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
    
    teacher_acc, teacher_params = evaluate_and_count_params(teacher_model)
    student_acc, student_params = evaluate_and_count_params(student_model)
    
    results = {
        'teacher_accuracy': teacher_acc,
        'student_accuracy': student_acc,
        'teacher_parameters': teacher_params,
        'student_parameters': student_params,
        'compression_ratio': teacher_params / student_params,
        'accuracy_retention': student_acc / teacher_acc if teacher_acc > 0 else 0
    }
    
    logger.info(f"教师模型: 准确率 {teacher_acc:.2f}%, 参数量 {teacher_params:,}")
    logger.info(f"学生模型: 准确率 {student_acc:.2f}%, 参数量 {student_params:,}")
    logger.info(f"压缩比: {results['compression_ratio']:.2f}x")
    logger.info(f"准确率保持: {results['accuracy_retention']:.3f}")
    
    return results

def main():
    """主函数：演示图像分类的模型蒸馏"""
    logger.info("开始图像分类模型蒸馏示例")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 获取数据
    train_loader, val_loader, test_loader = get_cifar10_data(batch_size=128)
    
    # 可视化一些样本
    visualize_samples(train_loader)
    
    # 创建模型
    teacher_model = SimpleTeacherCNN(num_classes=10)
    student_model = SimpleStudentCNN(num_classes=10)
    
    logger.info(f"教师模型参数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
    logger.info(f"学生模型参数量: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # 创建蒸馏训练器
    trainer = ImageDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=4.0,
        alpha=0.8,
        device=device
    )
    
    # 训练教师模型
    trainer.train_teacher(train_loader, val_loader, epochs=50)
    
    # 蒸馏训练学生模型
    trainer.distill_student(train_loader, val_loader, epochs=80)
    
    # 比较模型性能
    results = compare_model_performance(teacher_model, student_model, test_loader, device)
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    logger.info("图像分类模型蒸馏示例完成！")
    
    return results

if __name__ == "__main__":
    results = main()