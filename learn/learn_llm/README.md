# 模型蒸馏学习项目

这个项目提供了完整的模型蒸馏（Knowledge Distillation）学习示例，包含理论解释、代码实现和实际应用案例。

## 📚 理论背景

### 什么是模型蒸馏？

模型蒸馏是一种模型压缩技术，由Hinton等人在2015年提出。其核心思想是让一个小的"学生模型"学习一个大的"教师模型"的知识，从而在保持较好性能的同时大幅减少模型参数。

### 核心概念

1. **教师模型（Teacher Model）**：预训练的大模型，通常参数多、性能好
2. **学生模型（Student Model）**：要训练的小模型，参数少、推理快
3. **软标签（Soft Labels）**：教师模型输出的概率分布，包含更丰富的信息
4. **硬标签（Hard Labels）**：原始的真实标签
5. **温度参数（Temperature）**：控制软标签平滑程度的超参数
6. **蒸馏损失（Distillation Loss）**：结合软标签和硬标签的损失函数

### 蒸馏损失函数

```
L_total = α * L_soft + (1-α) * L_hard

其中：
L_soft = KL_divergence(softmax(z_s/T), softmax(z_t/T)) * T²
L_hard = CrossEntropy(z_s, y_true)

z_s: 学生模型输出
z_t: 教师模型输出
T: 温度参数
α: 软损失权重
y_true: 真实标签
```

## 🚀 项目结构

```
learn_llm/
├── README.md                 # 项目说明文档
├── config.py                 # 配置文件
├── model_distillation.py     # 基础模型蒸馏实现
├── image_distillation.py     # 图像分类蒸馏示例
├── text_distillation.py      # 文本分类蒸馏示例
├── run_distillation.py       # 主运行脚本
└── requirements.txt          # 依赖包列表
```

## 📦 安装依赖

```bash
# 基础依赖
pip install torch torchvision matplotlib scikit-learn numpy

# 文本分类示例额外依赖
pip install transformers

# 或者直接安装所有依赖
pip install -r requirements.txt
```

## 🎯 使用方法

### 1. 快速开始

```bash
# 查看所有可用的实验配置
python run_distillation.py --list-configs

# 运行合成数据蒸馏实验
python run_distillation.py --experiment synthetic_medium

# 运行图像分类蒸馏实验
python run_distillation.py --experiment image_quick

# 运行文本分类蒸馏实验（需要transformers库）
python run_distillation.py --experiment text_quick
```

### 2. 单独运行示例

```bash
# 运行基础模型蒸馏示例
python model_distillation.py

# 运行图像分类蒸馏示例
python image_distillation.py

# 运行文本分类蒸馏示例
python text_distillation.py
```

### 3. 自定义配置

你可以在 `config.py` 中修改实验参数，或者创建新的配置：

```python
from config import SyntheticDataConfig

# 创建自定义配置
custom_config = SyntheticDataConfig(
    n_samples=2000,
    n_features=15,
    n_classes=4,
    temperature=4.0,
    alpha=0.8,
    teacher_epochs=40,
    student_epochs=80
)
```

## 📊 实验类型

### 1. 合成数据实验 (`model_distillation.py`)

- **目的**：理解蒸馏的基本原理
- **数据**：sklearn生成的合成分类数据
- **教师模型**：多层全连接网络
- **学生模型**：较小的全连接网络
- **特点**：快速运行，适合理解概念

### 2. 图像分类实验 (`image_distillation.py`)

- **目的**：在真实视觉任务上验证蒸馏效果
- **数据**：CIFAR-10数据集
- **教师模型**：复杂的CNN网络
- **学生模型**：简化的CNN网络
- **特点**：实际应用场景，效果明显

### 3. 文本分类实验 (`text_distillation.py`)

- **目的**：在NLP任务上应用蒸馏技术
- **数据**：20newsgroups文本数据
- **教师模型**：BERT预训练模型
- **学生模型**：简单的LSTM网络
- **特点**：展示大模型到小模型的知识转移

## 🔧 配置参数说明

### 核心蒸馏参数

- **temperature**：温度参数，通常在2-10之间
  - 较小值：软标签更接近硬标签
  - 较大值：软标签更平滑，包含更多信息

- **alpha**：软损失权重，通常在0.5-0.9之间
  - 接近1：更依赖教师模型的软标签
  - 接近0：更依赖真实的硬标签

### 训练参数

- **teacher_epochs**：教师模型训练轮数
- **student_epochs**：学生模型训练轮数
- **learning_rate**：学习率
- **batch_size**：批次大小

## 📈 实验结果分析

运行实验后，你将看到以下结果：

1. **模型性能对比**
   - 教师模型准确率
   - 学生模型准确率
   - 准确率保持比例

2. **模型复杂度对比**
   - 教师模型参数量
   - 学生模型参数量
   - 压缩比

3. **训练过程可视化**
   - 损失曲线（总损失、软损失、硬损失）
   - 准确率曲线
   - 学习率变化

## 🎨 可视化结果

项目会自动生成以下图表：

- `training_history.png`：基础蒸馏训练历史
- `image_distillation_curves.png`：图像分类训练曲线
- `text_distillation_curves.png`：文本分类训练曲线
- `cifar10_samples.png`：CIFAR-10数据样本

## 🔬 实验建议

### 参数调优

1. **温度参数实验**
   ```bash
   # 尝试不同的温度值
   python run_distillation.py --experiment synthetic_medium
   # 然后修改config.py中的temperature参数
   ```

2. **Alpha参数实验**
   - 尝试不同的α值，观察软损失和硬损失的平衡

3. **模型架构实验**
   - 尝试不同的教师-学生模型架构组合

### 性能评估

- **压缩效果**：参数量减少比例
- **性能保持**：准确率保持比例
- **推理速度**：实际部署时的速度提升
- **内存占用**：模型大小减少

## 🚨 常见问题

### 1. 学生模型性能不佳

**可能原因**：
- 温度参数过小或过大
- Alpha参数设置不当
- 学生模型容量不足
- 训练轮数不够

**解决方案**：
- 调整温度参数（建议3-5）
- 增加学生模型容量
- 延长训练时间

### 2. 蒸馏效果不明显

**可能原因**：
- 教师模型性能不够好
- 数据集过于简单
- 蒸馏参数设置问题

**解决方案**：
- 确保教师模型充分训练
- 尝试更复杂的数据集
- 调整蒸馏参数

### 3. 训练过程不稳定

**可能原因**：
- 学习率过大
- 批次大小不合适
- 梯度爆炸

**解决方案**：
- 降低学习率
- 调整批次大小
- 添加梯度裁剪

## 📚 扩展阅读

1. **原始论文**：
   - Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.

2. **相关技术**：
   - 模型剪枝（Pruning）
   - 模型量化（Quantization）
   - 神经架构搜索（NAS）

3. **应用场景**：
   - 移动端部署
   - 边缘计算
   - 实时推理

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。

---

**Happy Learning! 🎉**

如果你觉得这个项目有用，请给它一个⭐️！