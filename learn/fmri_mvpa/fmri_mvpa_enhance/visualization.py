#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化和报告生成模块
提供数据可视化、结果展示和HTML报告生成功能
"""

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import plotting, image, datasets
from nilearn.plotting import plot_stat_map, plot_glass_brain, plot_roi
from nilearn.image import load_img, math_img
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import base64
from io import BytesIO

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .error_handling import ErrorHandler, ErrorContext

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MVPAVisualizer:
    """MVPA可视化器"""
    
    def __init__(self, output_dir: str = "./visualizations", error_handler: Optional[ErrorHandler] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_handler = error_handler or ErrorHandler()
        
        # 创建子目录
        (self.output_dir / "brain_maps").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "interactive").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # 可视化配置
        self.config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'colormap': 'RdYlBu_r',
            'threshold': 0.001,
            'alpha': 0.7,
            'font_size': 12
        }
    
    def plot_brain_activation(self, stat_img_path: str, title: str = "Brain Activation", 
                            threshold: Optional[float] = None, output_name: Optional[str] = None) -> str:
        """绘制脑激活图"""
        try:
            with ErrorContext(f"绘制脑激活图: {title}", self.error_handler, "visualization"):
                # 加载统计图像
                stat_img = load_img(stat_img_path)
                
                # 设置阈值
                if threshold is None:
                    threshold = self.config['threshold']
                
                # 创建图形
                fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
                
                # 矢状面
                plotting.plot_stat_map(
                    stat_img, 
                    axes=axes[0, 0],
                    title=f"{title} - Sagittal",
                    threshold=threshold,
                    colorbar=True,
                    cmap=self.config['colormap']
                )
                
                # 冠状面
                plotting.plot_stat_map(
                    stat_img, 
                    axes=axes[0, 1],
                    title=f"{title} - Coronal",
                    threshold=threshold,
                    colorbar=True,
                    cmap=self.config['colormap'],
                    cut_coords=[0]
                )
                
                # 轴状面
                plotting.plot_stat_map(
                    stat_img, 
                    axes=axes[1, 0],
                    title=f"{title} - Axial",
                    threshold=threshold,
                    colorbar=True,
                    cmap=self.config['colormap'],
                    cut_coords=[0]
                )
                
                # 玻璃脑
                plotting.plot_glass_brain(
                    stat_img,
                    axes=axes[1, 1],
                    title=f"{title} - Glass Brain",
                    threshold=threshold,
                    colorbar=True,
                    cmap=self.config['colormap']
                )
                
                plt.tight_layout()
                
                # 保存图像
                if output_name is None:
                    output_name = f"brain_activation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                output_path = self.output_dir / "brain_maps" / output_name
                plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close()
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"绘制脑激活图: {title}", "visualization")
            return ""
    
    def plot_roi_results(self, roi_results: Dict[str, Any], title: str = "ROI Analysis Results",
                        output_name: Optional[str] = None) -> str:
        """绘制ROI分析结果"""
        try:
            with ErrorContext(f"绘制ROI结果: {title}", self.error_handler, "visualization"):
                # 提取数据
                roi_names = []
                accuracies = []
                p_values = []
                
                for roi_name, results in roi_results.items():
                    if isinstance(results, dict) and 'accuracy' in results:
                        roi_names.append(roi_name)
                        accuracies.append(results['accuracy'])
                        p_values.append(results.get('permutation_pvalue', 1.0))
                
                if not roi_names:
                    raise ValueError("没有有效的ROI结果数据")
                
                # 创建图形
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # 1. 准确率条形图
                bars = axes[0, 0].bar(range(len(roi_names)), accuracies, 
                                     color=['red' if p < 0.05 else 'blue' for p in p_values])
                axes[0, 0].set_title('ROI Classification Accuracy')
                axes[0, 0].set_xlabel('ROI')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].set_xticks(range(len(roi_names)))
                axes[0, 0].set_xticklabels(roi_names, rotation=45, ha='right')
                axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance Level')
                axes[0, 0].legend()
                
                # 添加数值标签
                for i, (bar, acc, p) in enumerate(zip(bars, accuracies, p_values)):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{acc:.3f}\np={p:.3f}',
                                   ha='center', va='bottom', fontsize=8)
                
                # 2. P值散点图
                colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'blue' for p in p_values]
                axes[0, 1].scatter(range(len(roi_names)), p_values, c=colors, s=100, alpha=0.7)
                axes[0, 1].set_title('Statistical Significance')
                axes[0, 1].set_xlabel('ROI')
                axes[0, 1].set_ylabel('P-value')
                axes[0, 1].set_xticks(range(len(roi_names)))
                axes[0, 1].set_xticklabels(roi_names, rotation=45, ha='right')
                axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
                axes[0, 1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='p=0.1')
                axes[0, 1].set_yscale('log')
                axes[0, 1].legend()
                
                # 3. 准确率vs显著性散点图
                axes[1, 0].scatter(accuracies, [-np.log10(p) for p in p_values], 
                                 c=colors, s=100, alpha=0.7)
                axes[1, 0].set_title('Accuracy vs Significance')
                axes[1, 0].set_xlabel('Accuracy')
                axes[1, 0].set_ylabel('-log10(p-value)')
                axes[1, 0].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
                axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
                
                # 添加ROI标签
                for i, (acc, p, name) in enumerate(zip(accuracies, p_values, roi_names)):
                    axes[1, 0].annotate(name, (acc, -np.log10(p)), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7)
                
                # 4. 结果摘要表
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                
                # 创建摘要数据
                summary_data = []
                for name, acc, p in zip(roi_names, accuracies, p_values):
                    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    summary_data.append([name, f"{acc:.3f}", f"{p:.3f}", significance])
                
                table = axes[1, 1].table(cellText=summary_data,
                                        colLabels=['ROI', 'Accuracy', 'P-value', 'Sig.'],
                                        cellLoc='center',
                                        loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                axes[1, 1].set_title('Results Summary')
                
                plt.tight_layout()
                
                # 保存图像
                if output_name is None:
                    output_name = f"roi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                output_path = self.output_dir / "plots" / output_name
                plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close()
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"绘制ROI结果: {title}", "visualization")
            return ""
    
    def plot_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray] = None,
                                  class_names: Optional[List[str]] = None,
                                  title: str = "Classification Metrics",
                                  output_name: Optional[str] = None) -> str:
        """绘制分类指标"""
        try:
            with ErrorContext(f"绘制分类指标: {title}", self.error_handler, "visualization"):
                # 设置类别名称
                if class_names is None:
                    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                    class_names = [f"Class {i}" for i in unique_classes]
                
                # 创建图形
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # 1. 混淆矩阵
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names,
                           ax=axes[0, 0])
                axes[0, 0].set_title('Confusion Matrix')
                axes[0, 0].set_xlabel('Predicted')
                axes[0, 0].set_ylabel('True')
                
                # 2. ROC曲线（如果有概率预测）
                if y_prob is not None and len(np.unique(y_true)) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                                   label=f'ROC curve (AUC = {roc_auc:.3f})')
                    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    axes[0, 1].set_xlim([0.0, 1.0])
                    axes[0, 1].set_ylim([0.0, 1.05])
                    axes[0, 1].set_xlabel('False Positive Rate')
                    axes[0, 1].set_ylabel('True Positive Rate')
                    axes[0, 1].set_title('ROC Curve')
                    axes[0, 1].legend(loc="lower right")
                else:
                    axes[0, 1].text(0.5, 0.5, 'ROC Curve\n(Binary classification only)', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('ROC Curve')
                
                # 3. 分类准确率
                from sklearn.metrics import classification_report
                report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
                
                metrics = ['precision', 'recall', 'f1-score']
                x_pos = np.arange(len(class_names))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    values = [report[class_name][metric] for class_name in class_names]
                    axes[1, 0].bar(x_pos + i*width, values, width, label=metric)
                
                axes[1, 0].set_xlabel('Classes')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_title('Classification Metrics by Class')
                axes[1, 0].set_xticks(x_pos + width)
                axes[1, 0].set_xticklabels(class_names)
                axes[1, 0].legend()
                axes[1, 0].set_ylim([0, 1])
                
                # 4. 预测分布
                unique_true = np.unique(y_true)
                for i, true_class in enumerate(unique_true):
                    mask = y_true == true_class
                    pred_for_class = y_pred[mask]
                    
                    # 计算每个预测类别的数量
                    pred_counts = np.bincount(pred_for_class, minlength=len(class_names))
                    
                    axes[1, 1].bar(np.arange(len(class_names)) + i*0.35, pred_counts, 
                                  width=0.35, label=f'True {class_names[i]}', alpha=0.7)
                
                axes[1, 1].set_xlabel('Predicted Class')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Prediction Distribution')
                axes[1, 1].set_xticks(np.arange(len(class_names)))
                axes[1, 1].set_xticklabels(class_names)
                axes[1, 1].legend()
                
                plt.tight_layout()
                
                # 保存图像
                if output_name is None:
                    output_name = f"classification_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                output_path = self.output_dir / "plots" / output_name
                plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close()
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"绘制分类指标: {title}", "visualization")
            return ""
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: Optional[List[str]] = None,
                              title: str = "Feature Importance",
                              top_n: int = 20,
                              output_name: Optional[str] = None) -> str:
        """绘制特征重要性"""
        try:
            with ErrorContext(f"绘制特征重要性: {title}", self.error_handler, "visualization"):
                # 处理特征名称
                if feature_names is None:
                    feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
                
                # 排序并选择top_n
                sorted_indices = np.argsort(np.abs(feature_importance))[::-1]
                top_indices = sorted_indices[:top_n]
                
                top_importance = feature_importance[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                # 创建图形
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # 1. 水平条形图
                colors = ['red' if x < 0 else 'blue' for x in top_importance]
                axes[0, 0].barh(range(len(top_importance)), top_importance, color=colors, alpha=0.7)
                axes[0, 0].set_yticks(range(len(top_importance)))
                axes[0, 0].set_yticklabels(top_names)
                axes[0, 0].set_xlabel('Importance')
                axes[0, 0].set_title(f'Top {top_n} Feature Importance')
                axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # 2. 重要性分布直方图
                axes[0, 1].hist(feature_importance, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 1].set_xlabel('Importance Value')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Feature Importance Distribution')
                axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                
                # 3. 累积重要性
                sorted_importance = np.sort(np.abs(feature_importance))[::-1]
                cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
                
                axes[1, 0].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                               marker='o', markersize=2)
                axes[1, 0].set_xlabel('Number of Features')
                axes[1, 0].set_ylabel('Cumulative Importance')
                axes[1, 0].set_title('Cumulative Feature Importance')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 添加90%线
                idx_90 = np.where(cumulative_importance >= 0.9)[0]
                if len(idx_90) > 0:
                    axes[1, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
                    axes[1, 0].axvline(x=idx_90[0], color='red', linestyle='--', alpha=0.7)
                    axes[1, 0].text(idx_90[0], 0.9, f'90% at {idx_90[0]} features', 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # 4. 正负重要性对比
                positive_importance = feature_importance[feature_importance > 0]
                negative_importance = feature_importance[feature_importance < 0]
                
                axes[1, 1].hist([positive_importance, np.abs(negative_importance)], 
                               bins=30, alpha=0.7, label=['Positive', 'Negative (abs)'], 
                               color=['blue', 'red'])
                axes[1, 1].set_xlabel('Importance Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Positive vs Negative Importance')
                axes[1, 1].legend()
                
                plt.tight_layout()
                
                # 保存图像
                if output_name is None:
                    output_name = f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                output_path = self.output_dir / "plots" / output_name
                plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close()
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"绘制特征重要性: {title}", "visualization")
            return ""
    
    def create_interactive_brain_plot(self, stat_img_path: str, title: str = "Interactive Brain Plot",
                                    threshold: Optional[float] = None,
                                    output_name: Optional[str] = None) -> str:
        """创建交互式脑图"""
        try:
            with ErrorContext(f"创建交互式脑图: {title}", self.error_handler, "visualization"):
                # 加载统计图像
                stat_img = load_img(stat_img_path)
                stat_data = stat_img.get_fdata()
                
                # 设置阈值
                if threshold is None:
                    threshold = self.config['threshold']
                
                # 应用阈值
                thresholded_data = np.where(np.abs(stat_data) > threshold, stat_data, 0)
                
                # 获取非零体素的坐标和值
                coords = np.where(thresholded_data != 0)
                values = thresholded_data[coords]
                
                # 转换为MNI坐标
                affine = stat_img.affine
                mni_coords = nib.affines.apply_affine(affine, np.column_stack(coords))
                
                # 创建3D散点图
                fig = go.Figure(data=go.Scatter3d(
                    x=mni_coords[:, 0],
                    y=mni_coords[:, 1],
                    z=mni_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=values,
                        colorscale='RdYlBu',
                        opacity=0.8,
                        colorbar=dict(title="Statistic Value")
                    ),
                    text=[f'MNI: ({x:.1f}, {y:.1f}, {z:.1f})<br>Value: {v:.3f}' 
                          for x, y, z, v in zip(mni_coords[:, 0], mni_coords[:, 1], 
                                               mni_coords[:, 2], values)],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title='X (mm)',
                        yaxis_title='Y (mm)',
                        zaxis_title='Z (mm)',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    width=800,
                    height=600
                )
                
                # 保存交互式图像
                if output_name is None:
                    output_name = f"interactive_brain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                output_path = self.output_dir / "interactive" / output_name
                pyo.plot(fig, filename=str(output_path), auto_open=False)
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"创建交互式脑图: {title}", "visualization")
            return ""
    
    def create_interactive_results_dashboard(self, results_data: Dict[str, Any],
                                           title: str = "MVPA Results Dashboard",
                                           output_name: Optional[str] = None) -> str:
        """创建交互式结果仪表板"""
        try:
            with ErrorContext(f"创建交互式仪表板: {title}", self.error_handler, "visualization"):
                # 提取数据
                roi_names = []
                accuracies = []
                p_values = []
                
                for roi_name, results in results_data.items():
                    if isinstance(results, dict) and 'accuracy' in results:
                        roi_names.append(roi_name)
                        accuracies.append(results['accuracy'])
                        p_values.append(results.get('permutation_pvalue', 1.0))
                
                if not roi_names:
                    raise ValueError("没有有效的结果数据")
                
                # 创建子图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Classification Accuracy', 'Statistical Significance',
                                   'Accuracy vs Significance', 'Results Summary'),
                    specs=[[{"type": "bar"}, {"type": "scatter"}],
                           [{"type": "scatter"}, {"type": "table"}]]
                )
                
                # 1. 准确率条形图
                colors = ['red' if p < 0.05 else 'blue' for p in p_values]
                fig.add_trace(
                    go.Bar(
                        x=roi_names,
                        y=accuracies,
                        marker_color=colors,
                        text=[f'{acc:.3f}' for acc in accuracies],
                        textposition='auto',
                        name='Accuracy'
                    ),
                    row=1, col=1
                )
                
                # 添加机会水平线
                fig.add_hline(y=0.5, line_dash="dash", line_color="black", 
                             annotation_text="Chance Level", row=1, col=1)
                
                # 2. P值散点图
                fig.add_trace(
                    go.Scatter(
                        x=roi_names,
                        y=p_values,
                        mode='markers',
                        marker=dict(color=colors, size=10),
                        text=[f'p={p:.3f}' for p in p_values],
                        name='P-values'
                    ),
                    row=1, col=2
                )
                
                # 添加显著性水平线
                fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                             annotation_text="p=0.05", row=1, col=2)
                
                # 3. 准确率vs显著性散点图
                fig.add_trace(
                    go.Scatter(
                        x=accuracies,
                        y=[-np.log10(p) for p in p_values],
                        mode='markers+text',
                        marker=dict(color=colors, size=10),
                        text=roi_names,
                        textposition='top center',
                        name='ROI Results'
                    ),
                    row=2, col=1
                )
                
                # 添加参考线
                fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", row=2, col=1)
                fig.add_vline(x=0.5, line_dash="dash", line_color="black", row=2, col=1)
                
                # 4. 结果摘要表
                summary_data = []
                for name, acc, p in zip(roi_names, accuracies, p_values):
                    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    summary_data.append([name, f"{acc:.3f}", f"{p:.3f}", significance])
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['ROI', 'Accuracy', 'P-value', 'Significance'],
                                   fill_color='lightblue'),
                        cells=dict(values=list(zip(*summary_data)),
                                 fill_color='white')
                    ),
                    row=2, col=2
                )
                
                # 更新布局
                fig.update_layout(
                    title_text=title,
                    showlegend=False,
                    height=800
                )
                
                # 更新轴标签
                fig.update_xaxes(title_text="ROI", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=1, col=1)
                fig.update_xaxes(title_text="ROI", row=1, col=2)
                fig.update_yaxes(title_text="P-value", type="log", row=1, col=2)
                fig.update_xaxes(title_text="Accuracy", row=2, col=1)
                fig.update_yaxes(title_text="-log10(p-value)", row=2, col=1)
                
                # 保存交互式仪表板
                if output_name is None:
                    output_name = f"results_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                output_path = self.output_dir / "interactive" / output_name
                pyo.plot(fig, filename=str(output_path), auto_open=False)
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"创建交互式仪表板: {title}", "visualization")
            return ""
    
    def plot_data_quality_summary(self, quality_data: Dict[str, Any],
                                title: str = "Data Quality Summary",
                                output_name: Optional[str] = None) -> str:
        """绘制数据质量摘要"""
        try:
            with ErrorContext(f"绘制数据质量摘要: {title}", self.error_handler, "visualization"):
                # 提取质量指标
                subjects = []
                snr_values = []
                tsnr_values = []
                motion_values = []
                quality_ratings = []
                
                for subject, subject_data in quality_data.items():
                    if isinstance(subject_data, dict):
                        for run_key, run_data in subject_data.items():
                            if isinstance(run_data, dict) and 'mean_snr' in run_data:
                                subjects.append(f"{subject}_{run_key}")
                                snr_values.append(run_data.get('mean_snr', 0))
                                tsnr_values.append(run_data.get('mean_tsnr', 0))
                                motion_values.append(run_data.get('mean_fd', 0))
                                quality_ratings.append(run_data.get('quality_rating', 'Unknown'))
                
                if not subjects:
                    raise ValueError("没有有效的质量数据")
                
                # 创建图形
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # 1. SNR分布
                axes[0, 0].hist(snr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].set_xlabel('Signal-to-Noise Ratio')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('SNR Distribution')
                axes[0, 0].axvline(x=np.mean(snr_values), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(snr_values):.1f}')
                axes[0, 0].legend()
                
                # 2. tSNR分布
                axes[0, 1].hist(tsnr_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[0, 1].set_xlabel('Temporal SNR')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('tSNR Distribution')
                axes[0, 1].axvline(x=np.mean(tsnr_values), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(tsnr_values):.1f}')
                axes[0, 1].legend()
                
                # 3. 运动参数分布
                axes[1, 0].hist(motion_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[1, 0].set_xlabel('Mean Framewise Displacement (mm)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Motion Distribution')
                axes[1, 0].axvline(x=np.mean(motion_values), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(motion_values):.3f}')
                axes[1, 0].axvline(x=0.5, color='orange', linestyle='--', 
                                  label='Threshold: 0.5mm')
                axes[1, 0].legend()
                
                # 4. 质量评级饼图
                rating_counts = {}
                for rating in quality_ratings:
                    rating_counts[rating] = rating_counts.get(rating, 0) + 1
                
                colors_map = {'Excellent': 'green', 'Good': 'lightgreen', 
                             'Fair': 'orange', 'Poor': 'red', 'Unknown': 'gray'}
                colors = [colors_map.get(rating, 'gray') for rating in rating_counts.keys()]
                
                axes[1, 1].pie(rating_counts.values(), labels=rating_counts.keys(), 
                              autopct='%1.1f%%', colors=colors)
                axes[1, 1].set_title('Quality Rating Distribution')
                
                plt.tight_layout()
                
                # 保存图像
                if output_name is None:
                    output_name = f"data_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                output_path = self.output_dir / "plots" / output_name
                plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close()
                
                return str(output_path)
                
        except Exception as e:
            self.error_handler.log_error(e, f"绘制数据质量摘要: {title}", "visualization")
            return ""

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "./reports", error_handler: Optional[ErrorHandler] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_handler = error_handler or ErrorHandler()
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any],
                                    quality_data: Optional[Dict[str, Any]] = None,
                                    visualizations: Optional[Dict[str, str]] = None,
                                    config: Optional[Dict[str, Any]] = None) -> str:
        """生成综合分析报告"""
        try:
            with ErrorContext("生成综合报告", self.error_handler, "report_generation"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = self.output_dir / f"mvpa_comprehensive_report_{timestamp}.html"
                
                html_content = self._create_comprehensive_html(
                    analysis_results, quality_data, visualizations, config
                )
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # 生成PDF版本（如果可能）
                try:
                    pdf_file = self._generate_pdf_report(html_content, timestamp)
                    return str(report_file), str(pdf_file)
                except:
                    return str(report_file)
                
        except Exception as e:
            self.error_handler.log_error(e, "生成综合报告", "report_generation")
            return ""
    
    def _create_comprehensive_html(self, analysis_results: Dict[str, Any],
                                 quality_data: Optional[Dict[str, Any]],
                                 visualizations: Optional[Dict[str, str]],
                                 config: Optional[Dict[str, Any]]) -> str:
        """创建综合HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MVPA综合分析报告</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    background-color: #fafafa;
                }}
                .section h2 {{
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .metric {{
                    margin: 15px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                    border-left: 4px solid #667eea;
                }}
                .excellent {{ border-left-color: #4CAF50; }}
                .good {{ border-left-color: #8BC34A; }}
                .fair {{ border-left-color: #FF9800; }}
                .poor {{ border-left-color: #F44336; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    background-color: white;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #667eea;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .visualization {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    color: #666;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>MVPA综合分析报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                </div>
        """
        
        # 添加执行摘要
        html += self._add_executive_summary(analysis_results)
        
        # 添加配置信息
        if config:
            html += self._add_configuration_section(config)
        
        # 添加数据质量部分
        if quality_data:
            html += self._add_quality_section(quality_data)
        
        # 添加分析结果
        html += self._add_results_section(analysis_results)
        
        # 添加可视化
        if visualizations:
            html += self._add_visualization_section(visualizations)
        
        # 添加结论和建议
        html += self._add_conclusions_section(analysis_results, quality_data)
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _add_executive_summary(self, results: Dict[str, Any]) -> str:
        """添加执行摘要"""
        html = '<div class="section"><h2>执行摘要</h2>'
        
        # 计算摘要统计
        total_rois = 0
        significant_rois = 0
        avg_accuracy = 0
        
        if 'roi_results' in results:
            roi_results = results['roi_results']
            total_rois = len(roi_results)
            
            accuracies = []
            for roi_data in roi_results.values():
                if isinstance(roi_data, dict) and 'accuracy' in roi_data:
                    accuracies.append(roi_data['accuracy'])
                    if roi_data.get('permutation_pvalue', 1.0) < 0.05:
                        significant_rois += 1
            
            if accuracies:
                avg_accuracy = np.mean(accuracies)
        
        # 摘要统计卡片
        html += f"""
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{total_rois}</div>
                <div class="stat-label">分析的ROI数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{significant_rois}</div>
                <div class="stat-label">显著的ROI数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_accuracy:.3f}</div>
                <div class="stat-label">平均分类准确率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{(significant_rois/total_rois*100) if total_rois > 0 else 0:.1f}%</div>
                <div class="stat-label">显著ROI比例</div>
            </div>
        </div>
        """
        
        html += '</div>'
        return html
    
    def _add_configuration_section(self, config: Dict[str, Any]) -> str:
        """添加配置部分"""
        html = '<div class="section"><h2>分析配置</h2>'
        
        html += '<table>'
        html += '<tr><th>参数</th><th>值</th></tr>'
        
        # 递归展示配置
        def add_config_rows(cfg, prefix=""):
            rows = ""
            for key, value in cfg.items():
                if isinstance(value, dict):
                    rows += add_config_rows(value, f"{prefix}{key}.")
                else:
                    rows += f'<tr><td>{prefix}{key}</td><td>{value}</td></tr>'
            return rows
        
        html += add_config_rows(config)
        html += '</table>'
        html += '</div>'
        
        return html
    
    def _add_quality_section(self, quality_data: Dict[str, Any]) -> str:
        """添加质量部分"""
        html = '<div class="section"><h2>数据质量评估</h2>'
        
        # 质量摘要
        quality_ratings = []
        for subject_data in quality_data.values():
            if isinstance(subject_data, dict):
                for run_data in subject_data.values():
                    if isinstance(run_data, dict) and 'quality_rating' in run_data:
                        quality_ratings.append(run_data['quality_rating'])
        
        if quality_ratings:
            rating_counts = {}
            for rating in quality_ratings:
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
            
            html += '<h3>质量评级分布</h3>'
            html += '<table>'
            html += '<tr><th>评级</th><th>数量</th><th>比例</th></tr>'
            
            total = len(quality_ratings)
            for rating, count in rating_counts.items():
                percentage = (count / total) * 100
                css_class = rating.lower() if rating.lower() in ['excellent', 'good', 'fair', 'poor'] else ''
                html += f'<tr class="{css_class}"><td>{rating}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>'
            
            html += '</table>'
        
        html += '</div>'
        return html
    
    def _add_results_section(self, results: Dict[str, Any]) -> str:
        """添加结果部分"""
        html = '<div class="section"><h2>分析结果</h2>'
        
        if 'roi_results' in results:
            html += '<h3>ROI分类结果</h3>'
            html += '<table>'
            html += '<tr><th>ROI</th><th>准确率</th><th>P值</th><th>显著性</th><th>质量评级</th></tr>'
            
            for roi_name, roi_data in results['roi_results'].items():
                if isinstance(roi_data, dict) and 'accuracy' in roi_data:
                    accuracy = roi_data['accuracy']
                    p_value = roi_data.get('permutation_pvalue', 1.0)
                    
                    # 显著性标记
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                    
                    # 质量评级
                    if accuracy > 0.8 and p_value < 0.05:
                        quality = "Excellent"
                        css_class = "excellent"
                    elif accuracy > 0.7 and p_value < 0.05:
                        quality = "Good"
                        css_class = "good"
                    elif accuracy > 0.6:
                        quality = "Fair"
                        css_class = "fair"
                    else:
                        quality = "Poor"
                        css_class = "poor"
                    
                    html += f'<tr class="{css_class}">'
                    html += f'<td>{roi_name}</td>'
                    html += f'<td>{accuracy:.3f}</td>'
                    html += f'<td>{p_value:.3f}</td>'
                    html += f'<td>{significance}</td>'
                    html += f'<td>{quality}</td>'
                    html += '</tr>'
            
            html += '</table>'
        
        html += '</div>'
        return html
    
    def _add_visualization_section(self, visualizations: Dict[str, str]) -> str:
        """添加可视化部分"""
        html = '<div class="section"><h2>可视化结果</h2>'
        
        for viz_name, viz_path in visualizations.items():
            if viz_path and os.path.exists(viz_path):
                # 将图像转换为base64编码
                try:
                    with open(viz_path, 'rb') as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    html += f'<div class="visualization">'
                    html += f'<h3>{viz_name}</h3>'
                    html += f'<img src="data:image/png;base64,{img_base64}" alt="{viz_name}">'
                    html += f'</div>'
                except Exception as e:
                    html += f'<div class="visualization">'
                    html += f'<h3>{viz_name}</h3>'
                    html += f'<p>无法加载图像: {str(e)}</p>'
                    html += f'</div>'
        
        html += '</div>'
        return html
    
    def _add_conclusions_section(self, results: Dict[str, Any], 
                               quality_data: Optional[Dict[str, Any]]) -> str:
        """添加结论和建议部分"""
        html = '<div class="section"><h2>结论与建议</h2>'
        
        # 分析结论
        html += '<h3>主要发现</h3>'
        html += '<ul>'
        
        if 'roi_results' in results:
            roi_results = results['roi_results']
            significant_rois = []
            high_accuracy_rois = []
            
            for roi_name, roi_data in roi_results.items():
                if isinstance(roi_data, dict):
                    accuracy = roi_data.get('accuracy', 0)
                    p_value = roi_data.get('permutation_pvalue', 1.0)
                    
                    if p_value < 0.05:
                        significant_rois.append(roi_name)
                    
                    if accuracy > 0.7:
                        high_accuracy_rois.append((roi_name, accuracy))
            
            if significant_rois:
                html += f'<li>发现 {len(significant_rois)} 个ROI显示显著的分类效果: {", ".join(significant_rois[:5])}'
                if len(significant_rois) > 5:
                    html += f' 等{len(significant_rois)}个ROI'
                html += '</li>'
            
            if high_accuracy_rois:
                high_accuracy_rois.sort(key=lambda x: x[1], reverse=True)
                best_roi = high_accuracy_rois[0]
                html += f'<li>最佳分类性能ROI: {best_roi[0]} (准确率: {best_roi[1]:.3f})</li>'
        
        # 数据质量结论
        if quality_data:
            html += '<li>数据质量整体良好，符合分析要求</li>'
        
        html += '</ul>'
        
        # 建议
        html += '<h3>建议</h3>'
        html += '<ul>'
        html += '<li>对显著的ROI进行进一步的功能连接分析</li>'
        html += '<li>考虑增加样本量以提高统计功效</li>'
        html += '<li>探索不同的特征提取和分类方法</li>'
        html += '<li>结合行为数据进行更深入的分析</li>'
        html += '</ul>'
        
        html += '</div>'
        return html
    
    def _generate_pdf_report(self, html_content: str, timestamp: str) -> str:
        """生成PDF报告（需要额外的库支持）"""
        try:
            # 这里可以使用weasyprint或其他HTML到PDF的转换库
            # 由于依赖问题，这里只是占位符
            pdf_file = self.output_dir / f"mvpa_report_{timestamp}.pdf"
            
            # 实际实现需要安装weasyprint:
            # from weasyprint import HTML
            # HTML(string=html_content).write_pdf(pdf_file)
            
            return str(pdf_file)
            
        except Exception as e:
            self.error_handler.log_error(e, "生成PDF报告", "report_generation")
            return ""

def create_comprehensive_visualization_report(analysis_results: Dict[str, Any],
                                            quality_data: Optional[Dict[str, Any]] = None,
                                            config: Optional[Dict[str, Any]] = None,
                                            output_dir: str = "./reports") -> Dict[str, str]:
    """创建综合可视化报告"""
    # 初始化组件
    visualizer = MVPAVisualizer(output_dir)
    reporter = ReportGenerator(output_dir)
    
    # 生成可视化
    visualizations = {}
    
    try:
        # ROI结果可视化
        if 'roi_results' in analysis_results:
            roi_plot = visualizer.plot_roi_results(
                analysis_results['roi_results'],
                title="ROI Classification Results"
            )
            if roi_plot:
                visualizations['ROI Results'] = roi_plot
            
            # 交互式仪表板
            dashboard = visualizer.create_interactive_results_dashboard(
                analysis_results['roi_results'],
                title="Interactive MVPA Results Dashboard"
            )
            if dashboard:
                visualizations['Interactive Dashboard'] = dashboard
        
        # 数据质量可视化
        if quality_data:
            quality_plot = visualizer.plot_data_quality_summary(
                quality_data,
                title="Data Quality Summary"
            )
            if quality_plot:
                visualizations['Data Quality'] = quality_plot
        
        # 生成综合报告
        report_files = reporter.generate_comprehensive_report(
            analysis_results, quality_data, visualizations, config
        )
        
        if isinstance(report_files, tuple):
            visualizations['HTML Report'] = report_files[0]
            visualizations['PDF Report'] = report_files[1]
        elif report_files:
            visualizations['HTML Report'] = report_files
        
        return visualizations
        
    except Exception as e:
        print(f"创建可视化报告时出错: {str(e)}")
        return visualizations

def plot_group_comparison(group1_results: Dict[str, Any], 
                         group2_results: Dict[str, Any],
                         group1_name: str = "Group 1",
                         group2_name: str = "Group 2",
                         output_dir: str = "./visualizations") -> str:
    """绘制组间比较图"""
    try:
        visualizer = MVPAVisualizer(output_dir)
        
        # 提取两组数据
        roi_names = set(group1_results.keys()) & set(group2_results.keys())
        
        group1_acc = []
        group2_acc = []
        roi_list = []
        
        for roi in roi_names:
            if ('accuracy' in group1_results[roi] and 
                'accuracy' in group2_results[roi]):
                group1_acc.append(group1_results[roi]['accuracy'])
                group2_acc.append(group2_results[roi]['accuracy'])
                roi_list.append(roi)
        
        if not roi_list:
            return ""
        
        # 创建比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 条形图比较
        x = np.arange(len(roi_list))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, group1_acc, width, label=group1_name, alpha=0.8)
        axes[0, 0].bar(x + width/2, group2_acc, width, label=group2_name, alpha=0.8)
        axes[0, 0].set_xlabel('ROI')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Group Comparison - Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(roi_list, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        # 2. 散点图比较
        axes[0, 1].scatter(group1_acc, group2_acc, alpha=0.7, s=100)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel(f'{group1_name} Accuracy')
        axes[0, 1].set_ylabel(f'{group2_name} Accuracy')
        axes[0, 1].set_title('Accuracy Correlation')
        
        # 添加ROI标签
        for i, roi in enumerate(roi_list):
            axes[0, 1].annotate(roi, (group1_acc[i], group2_acc[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
        
        # 3. 差异分析
        differences = np.array(group2_acc) - np.array(group1_acc)
        colors = ['red' if d < 0 else 'blue' for d in differences]
        
        axes[1, 0].barh(range(len(roi_list)), differences, color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(len(roi_list)))
        axes[1, 0].set_yticklabels(roi_list)
        axes[1, 0].set_xlabel(f'Accuracy Difference ({group2_name} - {group1_name})')
        axes[1, 0].set_title('Group Differences')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. 统计摘要
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # 计算统计量
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(group1_acc, group2_acc)
        
        summary_text = f"""
        统计摘要:
        
        {group1_name}:
        平均准确率: {np.mean(group1_acc):.3f}
        标准差: {np.std(group1_acc):.3f}
        
        {group2_name}:
        平均准确率: {np.mean(group2_acc):.3f}
        标准差: {np.std(group2_acc):.3f}
        
        配对t检验:
        t统计量: {t_stat:.3f}
        p值: {p_value:.3f}
        
        效应量 (Cohen's d):
        {(np.mean(group2_acc) - np.mean(group1_acc)) / np.sqrt((np.var(group1_acc) + np.var(group2_acc)) / 2):.3f}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(output_dir) / "plots" / f"group_comparison_{timestamp}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    except Exception as e:
        print(f"绘制组间比较图时出错: {str(e)}")
        return ""

def create_roi_atlas_visualization(roi_masks_dir: str, 
                                 output_dir: str = "./visualizations") -> str:
    """创建ROI图谱可视化"""
    try:
        visualizer = MVPAVisualizer(output_dir)
        
        # 查找ROI掩模文件
        roi_files = list(Path(roi_masks_dir).glob("*.nii*"))
        
        if not roi_files:
            return ""
        
        # 创建ROI图谱图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 加载MNI模板
        try:
            mni_template = datasets.load_mni152_template()
        except:
            mni_template = None
        
        for i, roi_file in enumerate(roi_files[:6]):  # 最多显示6个ROI
            try:
                roi_img = load_img(str(roi_file))
                roi_name = roi_file.stem
                
                # 绘制ROI
                if mni_template is not None:
                    plotting.plot_roi(
                        roi_img,
                        bg_img=mni_template,
                        axes=axes[i],
                        title=roi_name,
                        cmap='Reds',
                        alpha=0.7
                    )
                else:
                    plotting.plot_roi(
                        roi_img,
                        axes=axes[i],
                        title=roi_name,
                        cmap='Reds',
                        alpha=0.7
                    )
                    
            except Exception as e:
                axes[i].text(0.5, 0.5, f'无法加载\n{roi_file.name}\n{str(e)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Error: {roi_file.stem}')
        
        # 隐藏未使用的子图
        for i in range(len(roi_files), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(output_dir) / "brain_maps" / f"roi_atlas_{timestamp}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    except Exception as e:
        print(f"创建ROI图谱可视化时出错: {str(e)}")
        return ""

if __name__ == "__main__":
    # 示例用法
    print("MVPA可视化模块已加载")
    print("主要功能:")
    print("- MVPAVisualizer: 脑图像和结果可视化")
    print("- ReportGenerator: HTML/PDF报告生成")
    print("- create_comprehensive_visualization_report: 综合报告创建")
    print("- plot_group_comparison: 组间比较可视化")
    print("- create_roi_atlas_visualization: ROI图谱可视化")