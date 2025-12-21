"""
工具函数

包含:
- 数据加载
- 学习率调度
- 评估指标
- 可视化工具
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class CosineWarmupScheduler:
    """
    余弦退火学习率调度器 (带warmup)
    
    Args:
        optimizer: 优化器
        warmup_epochs: 预热轮数
        total_epochs: 总轮数
        min_lr_ratio: 最小学习率比例
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.current_epoch = 0
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """更新学习率"""
        self.current_epoch += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_epoch <= self.warmup_epochs:
                # Warmup阶段: 线性增加
                lr = base_lr * (self.current_epoch / self.warmup_epochs)
            else:
                # Cosine退火阶段
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * \
                     0.5 * (1 + np.cos(np.pi * progress))
            
            param_group['lr'] = lr
    
    def get_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


def get_class_weights(labels, num_classes):
    """
    计算类别权重 (用于处理类别不平衡)
    
    Args:
        labels: 标签数组
        num_classes: 类别数
    
    Returns:
        class_weights: 类别权重张量
    """
    # 统计每个类别的样本数
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # 计算权重 (inverse frequency)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    
    # 归一化
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return torch.FloatTensor(class_weights)


def evaluate_metrics(y_true, y_pred, y_probs=None, uncertainty=None):
    """
    计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率 (可选)
        uncertainty: 不确定性 (可选)
    
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # 不确定性相关指标
    if uncertainty is not None:
        metrics['mean_uncertainty'] = float(np.mean(uncertainty))
        metrics['std_uncertainty'] = float(np.std(uncertainty))
        
        # 计算不确定性与错误的相关性
        errors = (y_true != y_pred).astype(float)
        correlation = np.corrcoef(errors, uncertainty)[0, 1]
        metrics['uncertainty_error_correlation'] = float(correlation)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_uncertainty_distribution(uncertainty, errors, save_path=None):
    """
    绘制不确定性分布
    
    Args:
        uncertainty: 不确定性数组
        errors: 错误标记 (0=正确, 1=错误)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图: 不确定性分布
    axes[0].hist(uncertainty[errors == 0], bins=50, alpha=0.7, label='Correct', density=True)
    axes[0].hist(uncertainty[errors == 1], bins=50, alpha=0.7, label='Error', density=True)
    axes[0].set_xlabel('Epistemic Uncertainty')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Uncertainty Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 右图: 箱线图
    data = [uncertainty[errors == 0], uncertainty[errors == 1]]
    axes[1].boxplot(data, labels=['Correct', 'Error'])
    axes[1].set_ylabel('Epistemic Uncertainty')
    axes[1].set_title('Uncertainty by Prediction Result')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

