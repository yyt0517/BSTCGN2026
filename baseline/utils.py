"""
基线模型通用工具函数
"""

import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def setup_logging(log_dir: str, model_name: str) -> logging.Logger:
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'{model_name}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(model_name)
    logger.info(f"日志文件: {log_file}")
    
    return logger


def load_data(data_dir: str = 'data/processed/time_divided') -> Tuple:
    """
    加载数据集
    
    Returns:
        train_features, train_labels, val_features, val_labels, 
        test_features, test_labels, encoders
    """
    data_dir = Path(data_dir)
    
    # 加载特征和标签
    train_features = np.load(data_dir / 'train_features.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    
    val_features = np.load(data_dir / 'val_features.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    
    test_features = np.load(data_dir / 'test_features.npy')
    test_labels = np.load(data_dir / 'test_labels.npy')
    
    # 加载编码器
    with open(data_dir / 'encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    return (train_features, train_labels, val_features, val_labels,
            test_features, test_labels, encoders)


def get_class_names(encoders: Dict) -> list:
    """获取类别名称"""
    label_encoder = encoders.get('label_encoder')
    if label_encoder is not None:
        return label_encoder.classes_.tolist()
    else:
        # 如果没有label_encoder，直接从训练标签推断
        # 假设标签是0到n-1的整数
        return list(range(9))  # 默认9个类别


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    计算评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        metrics: 指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'per_class_precision': precision_score(y_true, y_pred, average=None, zero_division=0),
        'per_class_recall': recall_score(y_true, y_pred, average=None, zero_division=0),
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics: Dict, class_names: list, logger: logging.Logger):
    """打印评估指标"""
    logger.info("\n" + "=" * 80)
    logger.info("评估指标")
    logger.info("=" * 80)

    logger.info(f"\n总体指标:")
    logger.info(f"  Accuracy:        {metrics['accuracy']:.6f}")
    logger.info(f"  Macro Precision: {metrics['macro_precision']:.6f}")
    logger.info(f"  Macro Recall:    {metrics['macro_recall']:.6f}")
    logger.info(f"  Macro F1:        {metrics['macro_f1']:.6f}")
    logger.info(f"  Weighted Precision: {metrics['weighted_precision']:.6f}")
    logger.info(f"  Weighted Recall: {metrics['weighted_recall']:.6f}")
    logger.info(f"  Weighted F1:     {metrics['weighted_f1']:.6f}")
    logger.info(f"  Micro Precision: {metrics['micro_precision']:.6f}")
    logger.info(f"  Micro Recall:    {metrics['micro_recall']:.6f}")
    logger.info(f"  Micro F1:        {metrics['micro_f1']:.6f}")

    logger.info(f"\n各类别指标:")
    for i, name in enumerate(class_names):
        logger.info(f"  {name:<20} P: {metrics['per_class_precision'][i]:.4f}  "
                   f"R: {metrics['per_class_recall'][i]:.4f}  "
                   f"F1: {metrics['per_class_f1'][i]:.4f}")


def save_results(metrics: Dict, class_names: list, result_dir: str, 
                model_name: str, train_time: float, inference_time: float,
                logger: logging.Logger):
    """保存评估结果"""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存指标到文本文件
    report_file = result_dir / 'evaluation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name} 评估报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("总体指标:\n")
        f.write(f"  Accuracy:           {metrics['accuracy']:.6f}\n")
        f.write(f"  Macro Precision:    {metrics['macro_precision']:.6f}\n")
        f.write(f"  Macro Recall:       {metrics['macro_recall']:.6f}\n")
        f.write(f"  Macro F1:           {metrics['macro_f1']:.6f}\n")
        f.write(f"  Weighted Precision: {metrics['weighted_precision']:.6f}\n")
        f.write(f"  Weighted Recall:    {metrics['weighted_recall']:.6f}\n")
        f.write(f"  Weighted F1:        {metrics['weighted_f1']:.6f}\n")
        f.write(f"  Micro Precision:    {metrics['micro_precision']:.6f}\n")
        f.write(f"  Micro Recall:       {metrics['micro_recall']:.6f}\n")
        f.write(f"  Micro F1:           {metrics['micro_f1']:.6f}\n\n")

        f.write("各类别指标:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {name:<20} P: {metrics['per_class_precision'][i]:.4f}  "
                   f"R: {metrics['per_class_recall'][i]:.4f}  "
                   f"F1: {metrics['per_class_f1'][i]:.4f}\n")

        f.write(f"\n训练时间: {train_time:.2f}秒\n")
        f.write(f"推理时间: {inference_time:.2f}秒\n")
    
    logger.info(f"评估报告已保存: {report_file}")
    
    # 保存各类别指标到CSV
    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Precision': metrics['per_class_precision'],
        'Recall': metrics['per_class_recall'],
        'F1-Score': metrics['per_class_f1']
    })
    csv_file = result_dir / 'per_class_metrics.csv'
    per_class_df.to_csv(csv_file, index=False)
    logger.info(f"各类别指标已保存: {csv_file}")
    
    # 保存混淆矩阵
    cm_file = result_dir / 'confusion_matrix.npy'
    np.save(cm_file, metrics['confusion_matrix'])
    logger.info(f"混淆矩阵已保存: {cm_file}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, 
                         result_dir: str, model_name: str,
                         logger: logging.Logger):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    result_dir = Path(result_dir)
    cm_plot_file = result_dir / 'confusion_matrix.png'
    plt.savefig(cm_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"混淆矩阵图已保存: {cm_plot_file}")


def plot_per_class_metrics(metrics: Dict, class_names: list,
                           result_dir: str, model_name: str,
                           logger: logging.Logger):
    """绘制各类别性能"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    # 绘制F1分数
    bars = ax.bar(x, metrics['per_class_f1'], width, label='F1-Score', color='steelblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title(f'{model_name} - Per-Class F1-Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # 保存图片
    result_dir = Path(result_dir)
    plot_file = result_dir / 'per_class_metrics.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"各类别性能图已保存: {plot_file}")


def save_model(model: Any, model_dir: str, model_name: str, logger: logging.Logger):
    """保存模型"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / f'{model_name}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"模型已保存: {model_file}")


def load_model(model_dir: str, model_name: str) -> Any:
    """加载模型"""
    model_file = Path(model_dir) / f'{model_name}.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def get_class_weights(train_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """计算类别权重"""
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)
    class_weights = total_samples / (num_classes * class_counts)
    return class_weights


def compute_probabilistic_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """
    计算概率预测相关的评估指标

    Args:
        y_true: 真实标签 (n_samples,)
        y_probs: 预测概率 (n_samples, n_classes)

    Returns:
        metrics: 概率指标字典
    """
    from sklearn.metrics import log_loss, brier_score_loss

    n_samples, n_classes = y_probs.shape

    # 预测类别
    y_pred = np.argmax(y_probs, axis=1)

    # 预测置信度（最大概率）
    max_probs = np.max(y_probs, axis=1)

    # 计算预测熵（不确定性度量）
    epsilon = 1e-10
    entropy = -np.sum(y_probs * np.log(y_probs + epsilon), axis=1)

    # 计算对数损失（交叉熵）
    logloss = log_loss(y_true, y_probs)

    # 计算Brier分数（仅对二分类有效，多分类需要one-vs-rest）
    # 这里我们计算平均Brier分数
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    brier_scores = np.mean((y_probs - y_true_onehot) ** 2, axis=1)
    mean_brier_score = np.mean(brier_scores)

    # 计算期望校准误差 (Expected Calibration Error, ECE)
    ece = compute_ece(y_true, y_probs, n_bins=10)

    metrics = {
        'log_loss': logloss,
        'mean_brier_score': mean_brier_score,
        'ece': ece,
        'mean_confidence': np.mean(max_probs),
        'std_confidence': np.std(max_probs),
        'mean_entropy': np.mean(entropy),
        'std_entropy': np.std(entropy),
        'max_entropy': np.log(n_classes),  # 最大可能熵
        'predictions': y_pred,
        'confidences': max_probs,
        'entropies': entropy
    }

    return metrics


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    计算期望校准误差 (Expected Calibration Error)

    Args:
        y_true: 真实标签
        y_probs: 预测概率
        n_bins: 分箱数量

    Returns:
        ece: 期望校准误差
    """
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if i == n_bins - 1:  # 最后一个bin包含右边界
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])

        if np.sum(mask) > 0:
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def analyze_uncertainty_by_class(y_true: np.ndarray, y_probs: np.ndarray,
                                 class_names: list, logger: logging.Logger):
    """
    按类别分析不确定性

    Args:
        y_true: 真实标签
        y_probs: 预测概率
        class_names: 类别名称列表
        logger: 日志记录器
    """
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    epsilon = 1e-10
    entropies = -np.sum(y_probs * np.log(y_probs + epsilon), axis=1)

    logger.info("\n" + "=" * 80)
    logger.info("各类别不确定性分析")
    logger.info("=" * 80)

    results = []
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) == 0:
            continue

        class_confidences = confidences[class_mask]
        class_entropies = entropies[class_mask]
        class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])

        logger.info(f"\n{class_name}:")
        logger.info(f"  样本数: {np.sum(class_mask)}")
        logger.info(f"  准确率: {class_accuracy:.4f}")
        logger.info(f"  平均置信度: {np.mean(class_confidences):.4f} ± {np.std(class_confidences):.4f}")
        logger.info(f"  平均熵: {np.mean(class_entropies):.4f} ± {np.std(class_entropies):.4f}")

        results.append({
            'class': class_name,
            'n_samples': np.sum(class_mask),
            'accuracy': class_accuracy,
            'mean_confidence': np.mean(class_confidences),
            'std_confidence': np.std(class_confidences),
            'mean_entropy': np.mean(class_entropies),
            'std_entropy': np.std(class_entropies)
        })

    return pd.DataFrame(results)


def plot_uncertainty_analysis(y_true: np.ndarray, y_probs: np.ndarray,
                              class_names: list, result_dir: str,
                              model_name: str, logger: logging.Logger):
    """
    绘制不确定性分析图

    Args:
        y_true: 真实标签
        y_probs: 预测概率
        class_names: 类别名称列表
        result_dir: 结果保存目录
        model_name: 模型名称
        logger: 日志记录器
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    epsilon = 1e-10
    entropies = -np.sum(y_probs * np.log(y_probs + epsilon), axis=1)

    correct_mask = (y_pred == y_true)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 置信度分布（正确 vs 错误）
    axes[0, 0].hist(confidences[correct_mask], bins=50, alpha=0.6,
                    label='Correct', color='green', edgecolor='black')
    axes[0, 0].hist(confidences[~correct_mask], bins=50, alpha=0.6,
                    label='Incorrect', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Confidence', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Confidence Distribution (Correct vs Incorrect)', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 熵分布（正确 vs 错误）
    axes[0, 1].hist(entropies[correct_mask], bins=50, alpha=0.6,
                    label='Correct', color='green', edgecolor='black')
    axes[0, 1].hist(entropies[~correct_mask], bins=50, alpha=0.6,
                    label='Incorrect', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Entropy (Uncertainty)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Entropy Distribution (Correct vs Incorrect)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 校准曲线
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])

        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_pred[mask] == y_true[mask])
            bin_accuracies.append(bin_accuracy)
        else:
            bin_accuracies.append(0)

    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    axes[1, 0].plot(bin_centers, bin_accuracies, 'o-', label=model_name,
                    linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Confidence', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('Calibration Curve', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 各类别平均不确定性
    class_entropies = []
    for i in range(len(class_names)):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_entropies.append(np.mean(entropies[class_mask]))
        else:
            class_entropies.append(0)

    axes[1, 1].bar(range(len(class_names)), class_entropies, alpha=0.7, color='steelblue')
    axes[1, 1].set_xlabel('Class', fontsize=12)
    axes[1, 1].set_ylabel('Average Entropy', fontsize=12)
    axes[1, 1].set_title('Average Uncertainty by Class', fontsize=14)
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = result_dir / 'uncertainty_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"不确定性分析图已保存: {plot_file}")


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载...")
    data = load_data()
    train_features, train_labels, val_features, val_labels, test_features, test_labels, encoders = data

    print(f"训练集: {train_features.shape}, {train_labels.shape}")
    print(f"验证集: {val_features.shape}, {val_labels.shape}")
    print(f"测试集: {test_features.shape}, {test_labels.shape}")

    class_names = get_class_names(encoders)
    print(f"\n类别名称: {class_names}")

    # 测试类别权重计算
    class_weights = get_class_weights(train_labels, len(class_names))
    print(f"\n类别权重:")
    for name, weight in zip(class_names, class_weights):
        print(f"  {name:<20} {weight:.3f}")

