#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分类的模型蒸馏示例

这个脚本演示了如何在文本分类任务中使用知识蒸馏。
使用BERT作为教师模型，简单的LSTM作为学生模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging
import re
from collections import Counter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SimpleVocabulary:
    """简单的词汇表类"""
    
    def __init__(self, texts: List[str], max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.build_vocab(texts)
        
    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        word_counts = Counter()
        
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def tokenize(self, text: str) -> List[str]:
        """简单的分词"""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return text.split()
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """将文本编码为数字序列"""
        words = self.tokenize(text)
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 是 <UNK>
        
        # 截断或填充
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([0] * (max_length - len(indices)))  # 0 是 <PAD>
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)

class SimpleLSTMDataset(Dataset):
    """LSTM模型的数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: SimpleVocabulary, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 编码文本
        encoded = self.vocab.encode(text, self.max_length)
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TeacherBERT(nn.Module):
    """教师模型：BERT"""
    
    def __init__(self, num_classes: int, model_name: str = 'bert-base-uncased'):
        super(TeacherBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class StudentLSTM(nn.Module):
    """学生模型：简单的LSTM"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_classes: int, num_layers: int = 2):
        super(StudentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=0.3,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, input_ids):
        # 嵌入
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        # 对于双向LSTM，连接前向和后向的最后隐藏状态
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # 分类
        output = self.dropout(hidden)
        logits = self.classifier(output)
        
        return logits

class TextDistillationLoss(nn.Module):
    """文本分类的蒸馏损失"""
    
    def __init__(self, temperature: float = 5.0, alpha: float = 0.7):
        super(TextDistillationLoss, self).__init__()
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

class TextDistillationTrainer:
    """文本分类的蒸馏训练器"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 5.0, alpha: float = 0.7, device: str = 'cpu'):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        
        self.distill_loss = TextDistillationLoss(temperature, alpha)
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        self.train_history = {
            'total_loss': [],
            'soft_loss': [],
            'hard_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def train_teacher(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 3):
        """训练教师模型（BERT）"""
        logger.info("开始训练教师模型（BERT）...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.teacher_model.parameters(), lr=2e-5)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.teacher_model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.teacher_model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                
                if batch_idx % 100 == 0:
                    logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # 验证阶段
            val_acc = self._evaluate_teacher(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.teacher_model.state_dict(), 'best_teacher_bert.pth')
            
            train_acc = 100. * correct / total
            logger.info(f'教师模型 Epoch {epoch+1}/{epochs}, '
                      f'Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2f}%, '
                      f'Val Acc: {val_acc:.2f}%')
        
        logger.info(f"教师模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    
    def _evaluate_teacher(self, data_loader: DataLoader) -> float:
        """评估教师模型"""
        self.teacher_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.teacher_model(input_ids, attention_mask)
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        
        return 100. * correct / total
    
    def distill_student(self, teacher_loader: DataLoader, student_loader: DataLoader, 
                       student_val_loader: DataLoader, epochs: int = 20):
        """蒸馏训练学生模型"""
        logger.info("开始蒸馏训练学生模型（LSTM）...")
        
        self.teacher_model.eval()  # 教师模型设为评估模式
        best_val_acc = 0
        
        # 确保两个数据加载器有相同的数据顺序
        teacher_iter = iter(teacher_loader)
        
        for epoch in range(epochs):
            # 训练阶段
            self.student_model.train()
            total_loss_dict = {'total_loss': 0, 'soft_loss': 0, 'hard_loss': 0}
            correct = 0
            total = 0
            
            teacher_iter = iter(teacher_loader)  # 重新初始化迭代器
            
            for batch_idx, student_batch in enumerate(student_loader):
                try:
                    teacher_batch = next(teacher_iter)
                except StopIteration:
                    break
                
                # 教师模型输入
                teacher_input_ids = teacher_batch['input_ids'].to(self.device)
                teacher_attention_mask = teacher_batch['attention_mask'].to(self.device)
                
                # 学生模型输入
                student_input_ids = student_batch['input_ids'].to(self.device)
                labels = student_batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 获取教师和学生的输出
                with torch.no_grad():
                    teacher_logits = self.teacher_model(teacher_input_ids, teacher_attention_mask)
                
                student_logits = self.student_model(student_input_ids)
                
                # 计算蒸馏损失
                loss, loss_dict = self.distill_loss(student_logits, teacher_logits, labels)
                
                loss.backward()
                self.optimizer.step()
                
                # 统计
                for key in total_loss_dict:
                    total_loss_dict[key] += loss_dict[key]
                
                pred = student_logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                
                if batch_idx % 50 == 0:
                    logger.info(f'Batch {batch_idx}/{len(student_loader)}, Loss: {loss.item():.4f}')
            
            # 验证阶段
            val_acc = self._evaluate_student(student_val_loader)
            
            # 记录训练历史
            train_acc = 100. * correct / total
            for key in total_loss_dict:
                total_loss_dict[key] /= len(student_loader)
                self.train_history[key].append(total_loss_dict[key])
            
            self.train_history['train_accuracy'].append(train_acc)
            self.train_history['val_accuracy'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.student_model.state_dict(), 'best_student_lstm.pth')
            
            self.scheduler.step()
            
            logger.info(f'学生模型 Epoch {epoch+1}/{epochs}, '
                      f'Total Loss: {total_loss_dict["total_loss"]:.4f}, '
                      f'Train Acc: {train_acc:.2f}%, '
                      f'Val Acc: {val_acc:.2f}%')
        
        logger.info(f"学生模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    
    def _evaluate_student(self, data_loader: DataLoader) -> float:
        """评估学生模型"""
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.student_model(input_ids)
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        
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
        
        # 损失比例
        if self.train_history['soft_loss'] and self.train_history['hard_loss']:
            soft_ratio = [s/(s+h) if (s+h) > 0 else 0 for s, h in 
                         zip(self.train_history['soft_loss'], self.train_history['hard_loss'])]
            axes[1, 1].plot(soft_ratio)
            axes[1, 1].set_title('Soft Loss Ratio')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Soft Loss / (Soft + Hard)')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('/Users/bytedance/PycharmProjects/PythonAnalysis/learn/learn_llm/text_distillation_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def prepare_text_data(subset_size: int = 1000) -> Tuple[List[str], List[int], List[str]]:
    """准备文本数据（20newsgroups的子集）"""
    logger.info("准备文本数据...")
    
    # 选择几个类别
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    
    # 获取数据
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    # 限制数据大小以便快速演示
    if len(newsgroups_train.data) > subset_size:
        indices = np.random.choice(len(newsgroups_train.data), subset_size, replace=False)
        texts = [newsgroups_train.data[i] for i in indices]
        labels = [newsgroups_train.target[i] for i in indices]
    else:
        texts = newsgroups_train.data
        labels = newsgroups_train.target.tolist()
    
    logger.info(f"数据大小: {len(texts)}")
    logger.info(f"类别: {categories}")
    logger.info(f"标签分布: {Counter(labels)}")
    
    return texts, labels, categories

def main():
    """主函数：演示文本分类的模型蒸馏"""
    logger.info("开始文本分类模型蒸馏示例")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 准备数据
    texts, labels, categories = prepare_text_data(subset_size=800)  # 小数据集用于演示
    
    # 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # 创建BERT tokenizer和数据集
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    teacher_train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    teacher_val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    teacher_test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=16, shuffle=True)
    teacher_val_loader = DataLoader(teacher_val_dataset, batch_size=16, shuffle=False)
    teacher_test_loader = DataLoader(teacher_test_dataset, batch_size=16, shuffle=False)
    
    # 创建LSTM词汇表和数据集
    vocab = SimpleVocabulary(train_texts, max_vocab_size=5000)
    
    student_train_dataset = SimpleLSTMDataset(train_texts, train_labels, vocab)
    student_val_dataset = SimpleLSTMDataset(val_texts, val_labels, vocab)
    student_test_dataset = SimpleLSTMDataset(test_texts, test_labels, vocab)
    
    student_train_loader = DataLoader(student_train_dataset, batch_size=16, shuffle=True)
    student_val_loader = DataLoader(student_val_dataset, batch_size=16, shuffle=False)
    student_test_loader = DataLoader(student_test_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    num_classes = len(categories)
    
    teacher_model = TeacherBERT(num_classes=num_classes)
    student_model = StudentLSTM(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=2
    )
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    logger.info(f"教师模型（BERT）参数量: {teacher_params:,}")
    logger.info(f"学生模型（LSTM）参数量: {student_params:,}")
    logger.info(f"压缩比: {teacher_params/student_params:.2f}x")
    
    # 创建蒸馏训练器
    trainer = TextDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=5.0,
        alpha=0.7,
        device=device
    )
    
    # 训练教师模型
    trainer.train_teacher(teacher_train_loader, teacher_val_loader, epochs=2)  # 少量epoch用于演示
    
    # 蒸馏训练学生模型
    trainer.distill_student(
        teacher_train_loader, 
        student_train_loader, 
        student_val_loader, 
        epochs=10
    )
    
    # 评估最终性能
    teacher_acc = trainer._evaluate_teacher(teacher_test_loader)
    student_acc = trainer._evaluate_student(student_test_loader)
    
    logger.info(f"最终测试结果:")
    logger.info(f"教师模型（BERT）准确率: {teacher_acc:.2f}%")
    logger.info(f"学生模型（LSTM）准确率: {student_acc:.2f}%")
    logger.info(f"准确率保持: {student_acc/teacher_acc:.3f}")
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    logger.info("文本分类模型蒸馏示例完成！")
    
    return {
        'teacher_accuracy': teacher_acc,
        'student_accuracy': student_acc,
        'compression_ratio': teacher_params / student_params,
        'accuracy_retention': student_acc / teacher_acc
    }

if __name__ == "__main__":
    # 注意：这个示例需要transformers库
    # pip install transformers
    try:
        results = main()
    except ImportError as e:
        logger.error(f"缺少依赖库: {e}")
        logger.info("请安装transformers库: pip install transformers")