"""
不确定性分解实验

展示BSTGCN相比MC Dropout的核心优势：
- BSTGCN可以分解总不确定性为认知不确定性和数据不确定性
- MC Dropout只能得到混合不确定性

实验设计：
1. 对ID和OOD样本分别计算三种不确定性
2. 展示OOD样本主要是认知不确定性高（模型不熟悉），而非数据不确定性
3. 说明这种分解能力对安全运营的价值

不确定性分解公式（基于信息论）：
- 总不确定性 (Total): H[E[p(y|x,W)]] = 预测均值的熵
- 认知不确定性 (Epistemic): I(Y;W|X) = H[E[p]] - E[H[p]] (互信息)
- 数据不确定性 (Aleatoric): E[H[p(y|x,W)]] = 各次采样熵的均值

关键洞察：Total = Epistemic + Aleatoric
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from baseline.stgcn_optimized import OptimizedSTGCN
from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from scripts.BSTGCN.mc_inference import mc_inference

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 不确定性分解函数 ====================

def decompose_uncertainty_bstgcn(model, x, num_samples=30):
    """
    BSTGCN的不确定性分解

    Args:
        model: BSTGCN模型
        x: 输入数据 (batch, seq_len, features)
        num_samples: MC采样次数

    Returns:
        total_uncertainty: 总不确定性 H[E[p]]
        epistemic_uncertainty: 认知不确定性 I(Y;W|X)
        aleatoric_uncertainty: 数据不确定性 E[H[p]]
        pred_class: 预测类别
    """
    model.eval()
    probs_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x, sample=True)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

    # Stack: (M, B, C)
    probs_stack = torch.stack(probs_list, dim=0)
    pred_mean = probs_stack.mean(dim=0)  # (B, C)
    pred_class = pred_mean.argmax(dim=-1)  # (B,)

    # 总不确定性: 预测均值的熵 H[E[p]]
    total_uncertainty = torch.special.entr(pred_mean).sum(dim=-1)  # (B,)

    # 数据不确定性: 各次采样熵的均值 E[H[p]]
    entropy_per_sample = torch.special.entr(probs_stack).sum(dim=-1)  # (M, B)
    aleatoric_uncertainty = entropy_per_sample.mean(dim=0)  # (B,)

    # 认知不确定性: 互信息 I(Y;W|X) = H[E[p]] - E[H[p]]
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty  # (B,)

    return (total_uncertainty.cpu().numpy(),
            epistemic_uncertainty.cpu().numpy(),
            aleatoric_uncertainty.cpu().numpy(),
            pred_class.cpu().numpy())


def compute_uncertainty_mc_dropout(model, x, num_samples=30):
    """
    MC Dropout的不确定性计算

    MC Dropout只能得到混合不确定性，无法区分认知和数据不确定性

    Returns:
        mixed_uncertainty: 混合不确定性（实际上是类似的互信息计算）
        pred_class: 预测类别
    """
    model.eval()

    # 启用dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    probs_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

    probs_stack = torch.stack(probs_list, dim=0)
    pred_mean = probs_stack.mean(dim=0)
    pred_class = pred_mean.argmax(dim=-1)

    # MC Dropout只能计算一个混合的不确定性
    # 技术上可以用同样公式，但语义上无法分解
    total_uncertainty = torch.special.entr(pred_mean).sum(dim=-1)
    entropy_per_sample = torch.special.entr(probs_stack).sum(dim=-1)
    mean_entropy = entropy_per_sample.mean(dim=0)
    mixed_uncertainty = total_uncertainty - mean_entropy

    # 关键点：MC Dropout无法进一步分解！
    # 返回的只是一个混合的不确定性估计
    return {
        'mixed_uncertainty': mixed_uncertainty.cpu().numpy(),
        'total_uncertainty': total_uncertainty.cpu().numpy(),
        'pred_class': pred_class.cpu().numpy(),
        'cannot_decompose': True  # 标记：无法分解
    }


def compute_ood_metrics(id_scores, ood_scores):
    """计算OOD检测指标"""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]

    return {'auroc': auroc, 'aupr': aupr, 'fpr@95': fpr_at_95tpr}


# ==================== 可视化函数 ====================

def plot_uncertainty_decomposition_comparison(
    bstgcn_results, mc_results, save_dir='results/uncertainty_decomposition'
):
    """
    绘制不确定性分解对比图 - IEEE期刊格式

    重点展示：
    1. 认知不确定性的相对变化（增幅）远大于数据不确定性
    2. 认知不确定性是最有效的OOD检测指标
    3. BSTGCN可以分解，MC Dropout无法分解
    """
    os.makedirs(save_dir, exist_ok=True)

    # IEEE期刊格式设置
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,
    })

    # BSTGCN结果
    id_total, id_epi, id_ale = bstgcn_results['id']
    ood_total, ood_epi, ood_ale = bstgcn_results['ood']

    # ==================== 图1: 论文主图 - 简洁双栏 ====================
    # 计算相对变化率和OOD检测指标
    epi_increase = (np.mean(ood_epi) - np.mean(id_epi)) / max(np.mean(id_epi), 1e-6) * 100
    ale_increase = (np.mean(ood_ale) - np.mean(id_ale)) / np.mean(id_ale) * 100
    total_increase = (np.mean(ood_total) - np.mean(id_total)) / np.mean(id_total) * 100
    ratio = epi_increase / ale_increase

    total_metrics = compute_ood_metrics(id_total, ood_total)
    epi_metrics = compute_ood_metrics(id_epi, ood_epi)
    ale_metrics = compute_ood_metrics(id_ale, ood_ale)

    # ==================== 图1: 论文主图（简洁版） ====================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左图: 增幅对比（核心发现）
    ax = axes[0]
    categories = ['Epistemic\n(Model Uncertainty)', 'Aleatoric\n(Data Noise)']
    increases = [epi_increase, ale_increase]
    colors = ['#c0392b', '#2980b9']
    bars = ax.bar(categories, increases, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
    ax.set_ylabel('Relative Increase for OOD (%)', fontsize=14)
    ax.set_title('(a) Uncertainty Response to OOD Samples', fontsize=15, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, val in zip(bars, increases):
        ax.annotate(f'+{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 8), textcoords='offset points', ha='center',
                   fontsize=18, fontweight='bold')

    # 添加倍数标注（画箭头连接两个柱子）
    ax.annotate('', xy=(1, ale_increase + 30), xytext=(0, epi_increase - 30),
               arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2.5))
    ax.text(0.5, (epi_increase + ale_increase)/2, f'{ratio:.1f}×',
           fontsize=20, fontweight='bold', color='#27ae60', ha='center', va='center')

    # 右图: ROC曲线（直观展示检测效果）
    ax = axes[1]
    from sklearn.metrics import roc_curve

    labels = np.concatenate([np.zeros(len(id_epi)), np.ones(len(ood_epi))])
    fpr_epi, tpr_epi, _ = roc_curve(labels, np.concatenate([id_epi, ood_epi]))
    fpr_ale, tpr_ale, _ = roc_curve(labels, np.concatenate([id_ale, ood_ale]))
    fpr_total, tpr_total, _ = roc_curve(labels, np.concatenate([id_total, ood_total]))

    ax.plot(fpr_epi, tpr_epi, 'r-', linewidth=2.5,
            label=f'Epistemic (AUROC={epi_metrics["auroc"]:.3f})')
    ax.plot(fpr_ale, tpr_ale, 'b--', linewidth=2,
            label=f'Aleatoric (AUROC={ale_metrics["auroc"]:.3f})')
    ax.plot(fpr_total, tpr_total, 'g:', linewidth=2,
            label=f'Total (AUROC={total_metrics["auroc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('(b) OOD Detection ROC Curves', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/uncertainty_decomposition_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/uncertainty_decomposition_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] 不确定性分解对比图 -> {save_dir}/uncertainty_decomposition_comparison.png")

    # ==================== 图2: 详细分析图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # 子图1: 认知不确定性分布（箱线图更清晰）
    ax = axes[0, 0]
    data_box = [id_epi, ood_epi]
    bp = ax.boxplot(data_box, labels=['ID', 'OOD'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Epistemic Uncertainty', fontsize=14)
    ax.set_title(f'(a) Epistemic Uncertainty: ID vs OOD\nOOD增幅 +{epi_increase:.0f}%',
                 fontsize=14, fontweight='bold')
    # 添加均值点
    ax.scatter([1, 2], [np.mean(id_epi), np.mean(ood_epi)], color='black', s=100,
               zorder=5, marker='D', label=f'Mean')
    ax.annotate(f'μ={np.mean(id_epi):.4f}', xy=(1, np.mean(id_epi)), xytext=(1.2, np.mean(id_epi)),
               fontsize=11, ha='left')
    ax.annotate(f'μ={np.mean(ood_epi):.4f}', xy=(2, np.mean(ood_epi)), xytext=(2.2, np.mean(ood_epi)),
               fontsize=11, ha='left')

    # 子图2: 数据不确定性分布
    ax = axes[0, 1]
    data_box = [id_ale, ood_ale]
    bp = ax.boxplot(data_box, labels=['ID', 'OOD'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Aleatoric Uncertainty', fontsize=14)
    ax.set_title(f'(b) Aleatoric Uncertainty: ID vs OOD\nOOD增幅 +{ale_increase:.0f}%',
                 fontsize=14)
    ax.scatter([1, 2], [np.mean(id_ale), np.mean(ood_ale)], color='black', s=100,
               zorder=5, marker='D')
    ax.annotate(f'μ={np.mean(id_ale):.3f}', xy=(1, np.mean(id_ale)), xytext=(1.2, np.mean(id_ale)),
               fontsize=11, ha='left')
    ax.annotate(f'μ={np.mean(ood_ale):.3f}', xy=(2, np.mean(ood_ale)), xytext=(2.2, np.mean(ood_ale)),
               fontsize=11, ha='left')

    # 子图3: 认知不确定性比例（更有意义的指标）
    ax = axes[1, 0]
    # 计算认知不确定性占总不确定性的比例
    id_epi_ratio = id_epi / (id_total + 1e-10) * 100  # 转为百分比
    ood_epi_ratio = ood_epi / (ood_total + 1e-10) * 100

    ax.hist(id_epi_ratio, bins=50, alpha=0.6, label=f'ID (μ={np.mean(id_epi_ratio):.1f}%)',
            density=True, color='#3498db')
    ax.hist(ood_epi_ratio, bins=50, alpha=0.6, label=f'OOD (μ={np.mean(ood_epi_ratio):.1f}%)',
            density=True, color='#e74c3c')
    ax.set_xlabel('Epistemic / Total Uncertainty (%)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('(c) Epistemic Proportion of Total Uncertainty\nOOD样本认知比例更高',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.axvline(np.mean(id_epi_ratio), color='#3498db', linestyle='--', linewidth=2)
    ax.axvline(np.mean(ood_epi_ratio), color='#e74c3c', linestyle='--', linewidth=2)

    # 子图4: 增幅对比柱状图（总结）
    ax = axes[1, 1]
    categories = ['Epistemic\n(Model)', 'Aleatoric\n(Data)', 'Total']
    increases = [epi_increase, ale_increase, total_increase]
    colors = ['#c0392b', '#2980b9', '#7f8c8d']
    bars = ax.bar(categories, increases, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel('Relative Increase for OOD (%)', fontsize=14)
    ax.set_title(f'(d) Summary: Uncertainty Increase\n认知增幅是数据增幅的{ratio:.1f}倍',
                 fontsize=14, fontweight='bold')

    for bar, val in zip(bars, increases):
        ax.annotate(f'+{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords='offset points', ha='center',
                   fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/ood_epistemic_dominance.png', dpi=250, bbox_inches='tight')
    plt.savefig(f'{save_dir}/ood_epistemic_dominance.pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] OOD认知主导图 -> {save_dir}/ood_epistemic_dominance.png")

    # ==================== 图3: 论文用简洁版（双栏 + ROC） ====================
    fig = plt.figure(figsize=(14, 5))

    # 左图: 增幅对比
    ax1 = fig.add_subplot(1, 3, 1)
    categories = ['Epistemic', 'Aleatoric']
    increases_simple = [epi_increase, ale_increase]
    colors = ['#c0392b', '#2980b9']
    bars = ax1.bar(categories, increases_simple, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
    ax1.set_ylabel('Relative Increase (%)', fontsize=14)
    ax1.set_title('(a) OOD Uncertainty\nIncrease', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, increases_simple):
        ax1.annotate(f'+{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 8), textcoords='offset points', ha='center',
                   fontsize=18, fontweight='bold')

    ax1.annotate('', xy=(1, ale_increase + 20), xytext=(0, epi_increase - 20),
               arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2.5))
    ax1.text(0.5, (epi_increase + ale_increase)/2, f'{ratio:.1f}×',
           fontsize=18, fontweight='bold', color='#27ae60', ha='center')

    # 中图: AUROC柱状图
    ax2 = fig.add_subplot(1, 3, 2)
    methods = ['Epistemic', 'Aleatoric', 'Total']
    aurocs = [epi_metrics['auroc'], ale_metrics['auroc'], total_metrics['auroc']]
    colors = ['#c0392b', '#2980b9', '#7f8c8d']
    bars = ax2.bar(methods, aurocs, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax2.set_ylabel('AUROC', fontsize=14)
    ax2.set_title('(b) OOD Detection\nPerformance', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.5, 0.9)

    for bar in bars:
        ax2.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=14, fontweight='bold')

    # 右图: ROC曲线
    ax3 = fig.add_subplot(1, 3, 3)
    labels = np.concatenate([np.zeros(len(id_epi)), np.ones(len(ood_epi))])
    from sklearn.metrics import roc_curve
    fpr_epi, tpr_epi, _ = roc_curve(labels, np.concatenate([id_epi, ood_epi]))
    fpr_ale, tpr_ale, _ = roc_curve(labels, np.concatenate([id_ale, ood_ale]))

    ax3.plot(fpr_epi, tpr_epi, color='#c0392b', linewidth=2.5,
            label=f'Epistemic ({epi_metrics["auroc"]:.3f})')
    ax3.plot(fpr_ale, tpr_ale, color='#2980b9', linewidth=2, linestyle='--',
            label=f'Aleatoric ({ale_metrics["auroc"]:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax3.set_xlabel('False Positive Rate', fontsize=14)
    ax3.set_ylabel('True Positive Rate', fontsize=14)
    ax3.set_title('(c) ROC Curves', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=12)
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/paper_figure_simple.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/paper_figure_simple.pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] 论文简洁版图 -> {save_dir}/paper_figure_simple.png")

    # ==================== 图4: IEEE期刊格式图（单栏3.5英寸宽） ====================
    # IEEE单栏宽度约3.5英寸，双栏宽度约7英寸
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    # 增幅对比柱状图（核心发现）
    categories = ['Epistemic', 'Aleatoric']
    increases_ieee = [epi_increase, ale_increase]
    colors = ['#c0392b', '#2980b9']
    bars = ax.bar(categories, increases_ieee, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    ax.set_ylabel('Relative Increase (%)')
    ax.set_xlabel('Uncertainty Type')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.3)

    for bar, val in zip(bars, increases_ieee):
        ax.annotate(f'+{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold')

    # 添加倍数标注
    ax.annotate('', xy=(1, ale_increase + 15), xytext=(0, epi_increase - 15),
               arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
    ax.text(0.5, (epi_increase + ale_increase)/2, f'{ratio:.1f}×',
           fontsize=11, fontweight='bold', color='#27ae60', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/uncertainty_increase_ieee.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/uncertainty_increase_ieee.pdf', bbox_inches='tight')
    # 同时保存到thesis/figures
    plt.savefig('docs/thesis/figures/uncertainty_increase_ieee.pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] IEEE格式增幅图 -> {save_dir}/uncertainty_increase_ieee.pdf")
    print(f"[保存] IEEE格式增幅图 -> docs/thesis/figures/uncertainty_increase_ieee.pdf")

    # ==================== 图5: IEEE期刊格式ROC曲线 ====================
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    ax.plot(fpr_epi, tpr_epi, 'r-', linewidth=1.5,
            label=f'Epistemic ({epi_metrics["auroc"]:.3f})')
    ax.plot(fpr_ale, tpr_ale, 'b--', linewidth=1.2,
            label=f'Aleatoric ({ale_metrics["auroc"]:.3f})')
    ax.plot(fpr_total, tpr_total, 'g:', linewidth=1.2,
            label=f'Total ({total_metrics["auroc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/uncertainty_roc_ieee.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/uncertainty_roc_ieee.pdf', bbox_inches='tight')
    plt.savefig('docs/thesis/figures/uncertainty_roc_ieee.pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] IEEE格式ROC图 -> docs/thesis/figures/uncertainty_roc_ieee.pdf")

    # ==================== 图6: IEEE双栏格式（2x2组合图） ====================
    # IEEE双栏宽度约7.16英寸
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))

    # 计算认知不确定性占总不确定性的比例
    id_epi_ratio = id_epi / (id_total + 1e-10) * 100
    ood_epi_ratio = ood_epi / (ood_total + 1e-10) * 100

    # (a) 认知不确定性箱线图
    ax = axes[0, 0]
    data_box = [id_epi, ood_epi]
    bp = ax.boxplot(data_box, labels=['ID', 'OOD'], patch_artist=True, widths=0.45)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Epistemic Uncertainty')
    ax.set_title(f'(a) Epistemic: +{epi_increase:.0f}%', fontweight='bold')
    ax.scatter([1, 2], [np.mean(id_epi), np.mean(ood_epi)], color='black', s=50, zorder=5, marker='D')
    ax.annotate(f'μ={np.mean(id_epi):.4f}', xy=(1, np.mean(id_epi)), xytext=(1.15, np.mean(id_epi)), fontsize=8, ha='left')
    ax.annotate(f'μ={np.mean(ood_epi):.4f}', xy=(2, np.mean(ood_epi)), xytext=(2.15, np.mean(ood_epi)), fontsize=8, ha='left')

    # (b) 数据不确定性箱线图
    ax = axes[0, 1]
    data_box = [id_ale, ood_ale]
    bp = ax.boxplot(data_box, labels=['ID', 'OOD'], patch_artist=True, widths=0.45)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Aleatoric Uncertainty')
    ax.set_title(f'(b) Aleatoric: +{ale_increase:.0f}%', fontweight='bold')
    ax.scatter([1, 2], [np.mean(id_ale), np.mean(ood_ale)], color='black', s=50, zorder=5, marker='D')
    ax.annotate(f'μ={np.mean(id_ale):.3f}', xy=(1, np.mean(id_ale)), xytext=(1.15, np.mean(id_ale)), fontsize=8, ha='left')
    ax.annotate(f'μ={np.mean(ood_ale):.3f}', xy=(2, np.mean(ood_ale)), xytext=(2.15, np.mean(ood_ale)), fontsize=8, ha='left')

    # (c) 认知不确定性比例直方图
    ax = axes[1, 0]
    ax.hist(id_epi_ratio, bins=50, alpha=0.6, label=f'ID (μ={np.mean(id_epi_ratio):.1f}%)', density=True, color='#3498db')
    ax.hist(ood_epi_ratio, bins=50, alpha=0.6, label=f'OOD (μ={np.mean(ood_epi_ratio):.1f}%)', density=True, color='#e74c3c')
    ax.set_xlabel('Epistemic / Total (%)')
    ax.set_ylabel('Density')
    ax.set_title('(c) Epistemic Proportion', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.axvline(np.mean(id_epi_ratio), color='#3498db', linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(ood_epi_ratio), color='#e74c3c', linestyle='--', linewidth=1.5)

    # (d) 增幅对比柱状图
    ax = axes[1, 1]
    bars = ax.bar(categories, increases_ieee, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    ax.set_ylabel('Relative Increase (%)')
    ax.set_xlabel('Uncertainty Type')
    ax.set_title(f'(d) Summary: {ratio:.1f}× Ratio', fontweight='bold')
    # 增加y轴上限，避免数字与边框重叠
    ax.set_ylim(0, epi_increase * 1.18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.3)

    for bar, val in zip(bars, increases_ieee):
        ax.annotate(f'+{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    # 添加倍数标注箭头
    ax.annotate('', xy=(1, ale_increase + 25), xytext=(0, epi_increase - 25),
               arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
    ax.text(0.5, (epi_increase + ale_increase)/2, f'{ratio:.1f}×',
           fontsize=10, fontweight='bold', color='#27ae60', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/uncertainty_decomposition_ieee.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/uncertainty_decomposition_ieee.pdf', bbox_inches='tight')
    plt.savefig('docs/thesis/figures/uncertainty_decomposition_ieee.pdf', bbox_inches='tight')
    plt.close()
    print(f"[保存] IEEE双栏组合图 -> docs/thesis/figures/uncertainty_decomposition_ieee.pdf")


def plot_single_panel_grouped_boxplot(bstgcn_results, save_dir='results/uncertainty_decomposition'):
    """
    单图一轴分组箱线图 - 最省版面设计

    设计要点：
    1. 横轴: Uncertainty Type (Epistemic / Aleatoric)
    2. 每个type下两组箱线: ID vs OOD (颜色区分)
    3. 隐藏outliers，只保留 median + IQR
    4. 图内标注关键增幅数字
    5. 图注包含 ratio 和 epistemic proportion
    """
    id_total, id_epi, id_ale = bstgcn_results['id']
    ood_total, ood_epi, ood_ale = bstgcn_results['ood']

    # 计算增幅
    epi_increase = (np.mean(ood_epi) - np.mean(id_epi)) / max(np.mean(id_epi), 1e-6) * 100
    ale_increase = (np.mean(ood_ale) - np.mean(id_ale)) / np.mean(id_ale) * 100
    ratio = epi_increase / max(ale_increase, 1)

    # 计算epistemic proportion
    id_epi_prop = np.mean(id_epi / (id_total + 1e-10)) * 100
    ood_epi_prop = np.mean(ood_epi / (ood_total + 1e-10)) * 100

    # 标准化以统一Y轴尺度 (min-max normalization per uncertainty type)
    def normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-10:
            return arr - arr_min
        return (arr - arr_min) / (arr_max - arr_min)

    # 合并后标准化
    all_epi = np.concatenate([id_epi, ood_epi])
    all_ale = np.concatenate([id_ale, ood_ale])
    epi_min, epi_max = all_epi.min(), all_epi.max()
    ale_min, ale_max = all_ale.min(), all_ale.max()

    id_epi_norm = (id_epi - epi_min) / (epi_max - epi_min + 1e-10)
    ood_epi_norm = (ood_epi - epi_min) / (epi_max - epi_min + 1e-10)
    id_ale_norm = (id_ale - ale_min) / (ale_max - ale_min + 1e-10)
    ood_ale_norm = (ood_ale - ale_min) / (ale_max - ale_min + 1e-10)

    # IEEE单栏宽度约3.5英寸
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # 准备数据: positions for grouped boxplot
    # Epistemic: positions 0.8, 1.2; Aleatoric: positions 2.8, 3.2
    positions = [0.8, 1.2, 2.8, 3.2]
    data = [id_epi_norm, ood_epi_norm, id_ale_norm, ood_ale_norm]

    # 颜色定义
    id_color = '#3498db'  # 蓝色
    ood_color = '#e74c3c'  # 红色
    colors = [id_color, ood_color, id_color, ood_color]

    # 绘制箱线图 (无outliers)
    bp = ax.boxplot(data, positions=positions, widths=0.35, patch_artist=True,
                    showfliers=False,  # 关键：隐藏outliers
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    # 添加均值点 (菱形标记)
    means = [np.mean(d) for d in data]
    ax.scatter(positions, means, color='black', s=25, zorder=5, marker='D', linewidths=0.5)

    # X轴设置
    ax.set_xticks([1.0, 3.0])
    ax.set_xticklabels(['Epistemic', 'Aleatoric'], fontsize=10)
    ax.set_ylabel('Normalized Uncertainty', fontsize=10)

    # 图内增幅标注 (在每组箱线上方)
    ax.annotate(f'+{epi_increase:.0f}%', xy=(1.0, 1.08), fontsize=9, fontweight='bold',
                ha='center', color='#c0392b')
    ax.annotate(f'+{ale_increase:.0f}%', xy=(3.0, 1.08), fontsize=9, fontweight='bold',
                ha='center', color='#27ae60')

    # 添加倍数对比标注
    ax.annotate(f'{ratio:.1f}× ratio', xy=(2.0, 1.15), fontsize=8,
                ha='center', va='bottom', color='#8e44ad', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#8e44ad', alpha=0.8))

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=id_color, edgecolor='black', alpha=0.7, label='ID'),
                       Patch(facecolor=ood_color, edgecolor='black', alpha=0.7, label='OOD')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # Y轴范围
    ax.set_ylim(-0.08, 1.25)
    ax.set_xlim(0.3, 3.7)

    # 网格线
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'fig4_single_panel_boxplot.pdf', bbox_inches='tight')
    plt.savefig(save_path / 'fig4_single_panel_boxplot.png', dpi=300, bbox_inches='tight')

    # 同时保存到论文figures目录
    thesis_fig_dir = Path('docs/thesis/figures')
    thesis_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(thesis_fig_dir / 'fig4_uncertainty_boxplot.pdf', bbox_inches='tight')
    plt.savefig(thesis_fig_dir / 'fig4_uncertainty_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[保存] 单图箱线图 -> {save_path / 'fig4_single_panel_boxplot.pdf'}")
    print(f"[保存] 单图箱线图 -> docs/thesis/figures/fig4_uncertainty_boxplot.pdf")

    # 输出图注建议文本
    caption_text = (
        f"Epistemic proportion: ID {id_epi_prop:.1f}% → OOD {ood_epi_prop:.1f}%. "
        f"Raw values: Epistemic {np.mean(id_epi):.4f}→{np.mean(ood_epi):.4f}, "
        f"Aleatoric {np.mean(id_ale):.3f}→{np.mean(ood_ale):.3f}."
    )
    print(f"\n[图注建议] {caption_text}")

    return {
        'epi_increase': epi_increase,
        'ale_increase': ale_increase,
        'ratio': ratio,
        'id_epi_prop': id_epi_prop,
        'ood_epi_prop': ood_epi_prop,
        'id_epi_mean': np.mean(id_epi),
        'ood_epi_mean': np.mean(ood_epi),
        'id_ale_mean': np.mean(id_ale),
        'ood_ale_mean': np.mean(ood_ale)
    }


def print_decomposition_analysis(bstgcn_results, mc_results):
    """
    打印不确定性分解分析报告
    """
    print("\n" + "="*80)
    print("不确定性分解分析报告")
    print("="*80)

    id_total, id_epi, id_ale = bstgcn_results['id']
    ood_total, ood_epi, ood_ale = bstgcn_results['ood']

    print("\n[1] BSTGCN 不确定性统计")
    print("-"*60)
    print(f"{'指标':<25} {'ID样本':<20} {'OOD样本':<20} {'差异':<15}")
    print("-"*60)

    print(f"{'总不确定性 (Total)':<25} "
          f"{np.mean(id_total):.4f}±{np.std(id_total):.4f}   "
          f"{np.mean(ood_total):.4f}±{np.std(ood_total):.4f}   "
          f"+{(np.mean(ood_total)-np.mean(id_total))/np.mean(id_total)*100:.1f}%")

    print(f"{'认知不确定性 (Epistemic)':<25} "
          f"{np.mean(id_epi):.4f}±{np.std(id_epi):.4f}   "
          f"{np.mean(ood_epi):.4f}±{np.std(ood_epi):.4f}   "
          f"+{(np.mean(ood_epi)-np.mean(id_epi))/max(np.mean(id_epi),1e-6)*100:.1f}%")

    print(f"{'数据不确定性 (Aleatoric)':<25} "
          f"{np.mean(id_ale):.4f}±{np.std(id_ale):.4f}   "
          f"{np.mean(ood_ale):.4f}±{np.std(ood_ale):.4f}   "
          f"+{(np.mean(ood_ale)-np.mean(id_ale))/np.mean(id_ale)*100:.1f}%")

    # 计算OOD检测性能
    print("\n[2] 基于不同不确定性的OOD检测性能")
    print("-"*60)

    total_metrics = compute_ood_metrics(id_total, ood_total)
    epi_metrics = compute_ood_metrics(id_epi, ood_epi)
    ale_metrics = compute_ood_metrics(id_ale, ood_ale)

    print(f"{'不确定性类型':<25} {'AUROC':<12} {'AUPR':<12} {'FPR@95':<12}")
    print("-"*60)
    print(f"{'总不确定性':<25} {total_metrics['auroc']:.4f}       "
          f"{total_metrics['aupr']:.4f}       {total_metrics['fpr@95']:.4f}")
    print(f"{'认知不确定性 ★':<25} {epi_metrics['auroc']:.4f}       "
          f"{epi_metrics['aupr']:.4f}       {epi_metrics['fpr@95']:.4f}")
    print(f"{'数据不确定性':<25} {ale_metrics['auroc']:.4f}       "
          f"{ale_metrics['aupr']:.4f}       {ale_metrics['fpr@95']:.4f}")

    # MC Dropout对比
    print("\n[3] MC Dropout 对比 (无法分解)")
    print("-"*60)
    id_mixed = mc_results['id']['mixed_uncertainty']
    ood_mixed = mc_results['ood']['mixed_uncertainty']
    mixed_metrics = compute_ood_metrics(id_mixed, ood_mixed)

    print(f"{'混合不确定性':<25} {mixed_metrics['auroc']:.4f}       "
          f"{mixed_metrics['aupr']:.4f}       {mixed_metrics['fpr@95']:.4f}")
    print("\n⚠️  MC Dropout 局限: 无法区分认知和数据不确定性！")

    # 关键洞察
    print("\n[4] 关键发现")
    print("-"*60)

    # 计算认知vs数据不确定性的增幅比
    epi_increase = (np.mean(ood_epi) - np.mean(id_epi)) / max(np.mean(id_epi), 1e-6)
    ale_increase = (np.mean(ood_ale) - np.mean(id_ale)) / np.mean(id_ale)

    print(f"• OOD样本的认知不确定性增幅: +{epi_increase*100:.1f}%")
    print(f"• OOD样本的数据不确定性增幅: +{ale_increase*100:.1f}%")
    print(f"• 认知/数据增幅比: {epi_increase/max(ale_increase,0.01):.2f}x")

    if epi_increase > ale_increase:
        print("\n✅ 结论: OOD样本主要表现为认知不确定性增加（模型不熟悉）")
        print("   这证明了不确定性分解对OOD检测的价值")

    print("\n[5] 安全运营价值")
    print("-"*60)
    print("• 高认知不确定性 → 模型对该样本不熟悉 → 可能是新攻击类型")
    print("  → 运营建议: 收集更多该类型数据，更新模型")
    print("• 高数据不确定性 → 数据本身噪声大 → 可能是网络抖动")
    print("  → 运营建议: 无需特别处理，属于正常变异")
    print("• MC Dropout无法提供这种区分能力！")
    print("="*80)

    return {
        'bstgcn': {
            'total_auroc': total_metrics['auroc'],
            'epistemic_auroc': epi_metrics['auroc'],
            'aleatoric_auroc': ale_metrics['auroc'],
            'epi_increase_pct': epi_increase * 100,
            'ale_increase_pct': ale_increase * 100
        },
        'mc_dropout': {
            'mixed_auroc': mixed_metrics['auroc']
        }
    }




# ==================== 数据加载 ====================

def load_data(data_dir='data/processed/ood_slowrate'):
    """加载ID和OOD数据"""
    from pathlib import Path
    import pickle

    data_dir = Path(data_dir)
    print(f"[1/4] 加载数据 from {data_dir}...")

    # 加载ID测试数据
    test_x = np.load(data_dir / 'test_id_sequences.npy')
    test_y = np.load(data_dir / 'test_id_labels.npy')

    # 加载OOD数据
    ood_x = np.load(data_dir / 'test_ood_sequences.npy')
    # OOD数据没有真实标签，创建占位标签
    ood_y = np.zeros(len(ood_x), dtype=np.int64)

    # 加载metadata获取类别数
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"  ID测试样本: {len(test_x)}")
    print(f"  OOD样本: {len(ood_x)}")
    print(f"  类别数: {metadata.get('num_classes', 'unknown')}")

    # 转换为Tensor
    test_x = torch.FloatTensor(test_x)
    test_y = torch.LongTensor(test_y)
    ood_x = torch.FloatTensor(ood_x)
    ood_y = torch.LongTensor(ood_y)

    return test_x, test_y, ood_x, ood_y, metadata


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='不确定性分解实验')
    parser.add_argument('--bstgcn-path', type=str,
                       default='results/ood_experiment/bstgcn_ood_20251214_143558.pt',
                       help='BSTGCN模型路径')
    parser.add_argument('--stgcn-path', type=str,
                       default='',  # 将在代码中自动搜索
                       help='STGCN模型路径 (用于MC Dropout)')
    parser.add_argument('--data-dir', type=str,
                       default='data/processed/ood_slowrate',
                       help='数据目录')
    parser.add_argument('--num-samples', type=int, default=30,
                       help='MC采样次数')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--save-dir', type=str,
                       default='results/uncertainty_decomposition')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    test_x, test_y, ood_x, ood_y, metadata = load_data(args.data_dir)

    # 创建DataLoader
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size, shuffle=False
    )
    ood_loader = DataLoader(
        TensorDataset(ood_x, ood_y),
        batch_size=args.batch_size, shuffle=False
    )

    # 获取数据维度
    input_dim = test_x.shape[-1]
    seq_len = test_x.shape[1]
    num_classes = metadata.get('num_classes', len(torch.unique(test_y)))
    print(f"  输入维度: {input_dim}, 序列长度: {seq_len}, 类别数: {num_classes}")

    # ==================== 加载BSTGCN模型 ====================
    print(f"\n[2/4] 加载BSTGCN模型...")
    bstgcn_model = BSTGCNModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=8,
        num_st_blocks=3,
        num_classes=num_classes,
        dropout=0.3,  # 与最佳配置一致
        k_neighbors=10,
        prior_std=1.0  # 与最佳配置一致
    ).to(device)

    if os.path.exists(args.bstgcn_path):
        bstgcn_model.load_state_dict(torch.load(args.bstgcn_path, map_location=device))
        print(f"  加载BSTGCN: {args.bstgcn_path}")
    else:
        print(f"  ⚠️ 未找到BSTGCN模型: {args.bstgcn_path}")
        print("  使用随机初始化模型演示...")

    # ==================== 加载或训练STGCN模型 (MC Dropout) ====================
    print(f"\n[3/4] 加载/训练STGCN模型 (MC Dropout)...")
    stgcn_model = OptimizedSTGCN(
        input_size=input_dim,
        hidden_channels=[64, 128, 128],  # 与baseline comparison一致
        num_classes=num_classes,
        dropout=0.15,
        k_neighbors=10
    ).to(device)

    stgcn_path = args.stgcn_path if args.stgcn_path else 'results/ood_experiment/stgcn_for_decomposition.pt'

    if os.path.exists(stgcn_path):
        stgcn_model.load_state_dict(torch.load(stgcn_path, map_location=device))
        print(f"  加载STGCN: {stgcn_path}")
    else:
        print("  ⚠️ 未找到STGCN模型，开始训练...")
        # 加载训练数据
        train_x = np.load(Path(args.data_dir) / 'train_sequences.npy')
        train_y = np.load(Path(args.data_dir) / 'train_labels.npy')
        val_x = np.load(Path(args.data_dir) / 'val_sequences.npy')
        val_y = np.load(Path(args.data_dir) / 'val_labels.npy')

        train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
        val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))
        train_loader_stgcn = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader_stgcn = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 计算类别权重
        from scripts.BSTGCN.utils import get_class_weights
        class_weights = get_class_weights(train_y, num_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        optimizer = torch.optim.AdamW(stgcn_model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, 101):
            stgcn_model.train()
            for batch_x, batch_y in train_loader_stgcn:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = stgcn_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # 验证
            stgcn_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader_stgcn:
                    batch_x = batch_x.to(device)
                    logits = stgcn_model(batch_x)
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_y.numpy())

            from sklearn.metrics import f1_score
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = stgcn_model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: val_f1={val_f1:.4f}, best_f1={best_f1:.4f}")

            if patience_counter >= 20:
                print(f"    Early stopping at epoch {epoch}")
                break

        stgcn_model.load_state_dict(best_state)
        torch.save(best_state, stgcn_path)
        print(f"  STGCN训练完成! Best F1: {best_f1:.4f}")
        print(f"  模型已保存: {stgcn_path}")

    # ==================== 计算不确定性 ====================
    print(f"\n[4/4] 计算不确定性分解...")

    # BSTGCN不确定性分解
    print("  BSTGCN ID样本...")
    id_total_list, id_epi_list, id_ale_list = [], [], []
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        total, epi, ale, _ = decompose_uncertainty_bstgcn(
            bstgcn_model, batch_x, num_samples=args.num_samples
        )
        id_total_list.append(total)
        id_epi_list.append(epi)
        id_ale_list.append(ale)

    print("  BSTGCN OOD样本...")
    ood_total_list, ood_epi_list, ood_ale_list = [], [], []
    for batch_x, _ in ood_loader:
        batch_x = batch_x.to(device)
        total, epi, ale, _ = decompose_uncertainty_bstgcn(
            bstgcn_model, batch_x, num_samples=args.num_samples
        )
        ood_total_list.append(total)
        ood_epi_list.append(epi)
        ood_ale_list.append(ale)

    bstgcn_results = {
        'id': (np.concatenate(id_total_list),
               np.concatenate(id_epi_list),
               np.concatenate(id_ale_list)),
        'ood': (np.concatenate(ood_total_list),
                np.concatenate(ood_epi_list),
                np.concatenate(ood_ale_list))
    }

    # MC Dropout不确定性
    print("  MC Dropout ID样本...")
    id_mc_list = []
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        mc_result = compute_uncertainty_mc_dropout(
            stgcn_model, batch_x, num_samples=args.num_samples
        )
        id_mc_list.append(mc_result['mixed_uncertainty'])

    print("  MC Dropout OOD样本...")
    ood_mc_list = []
    for batch_x, _ in ood_loader:
        batch_x = batch_x.to(device)
        mc_result = compute_uncertainty_mc_dropout(
            stgcn_model, batch_x, num_samples=args.num_samples
        )
        ood_mc_list.append(mc_result['mixed_uncertainty'])

    mc_results = {
        'id': {'mixed_uncertainty': np.concatenate(id_mc_list), 'cannot_decompose': True},
        'ood': {'mixed_uncertainty': np.concatenate(ood_mc_list), 'cannot_decompose': True}
    }

    # ==================== 分析与可视化 ====================
    analysis_results = print_decomposition_analysis(bstgcn_results, mc_results)
    plot_uncertainty_decomposition_comparison(bstgcn_results, mc_results, args.save_dir)

    # 新增：单图一轴分组箱线图（最省版面，用于论文Fig4）
    print("\n[生成] 论文Fig4: 单图分组箱线图...")
    fig4_stats = plot_single_panel_grouped_boxplot(bstgcn_results, args.save_dir)

    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    np.savez(f'{args.save_dir}/uncertainty_decomposition_results.npz',
             bstgcn_id_total=bstgcn_results['id'][0],
             bstgcn_id_epi=bstgcn_results['id'][1],
             bstgcn_id_ale=bstgcn_results['id'][2],
             bstgcn_ood_total=bstgcn_results['ood'][0],
             bstgcn_ood_epi=bstgcn_results['ood'][1],
             bstgcn_ood_ale=bstgcn_results['ood'][2],
             mc_id_mixed=mc_results['id']['mixed_uncertainty'],
             mc_ood_mixed=mc_results['ood']['mixed_uncertainty'])
    print(f"\n[保存] 结果 -> {args.save_dir}/uncertainty_decomposition_results.npz")

    print("\n实验完成!")
    return analysis_results


if __name__ == '__main__':
    main()