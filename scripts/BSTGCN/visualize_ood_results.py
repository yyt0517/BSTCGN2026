"""
OOD检测结果可视化脚本

生成以下图表:
1. ID vs OOD不确定性分布直方图
2. ROC曲线
3. 不确定性箱线图对比
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(result_path: str):
    """加载实验结果"""
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_uncertainty_histogram(id_unc, ood_unc, save_path=None):
    """绘制ID vs OOD不确定性分布直方图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算bins
    all_unc = np.concatenate([id_unc, ood_unc])
    bins = np.linspace(0, np.percentile(all_unc, 99), 50)
    
    ax.hist(id_unc, bins=bins, alpha=0.6, label=f'ID (8 classes)', 
            color='steelblue', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(ood_unc, bins=bins, alpha=0.6, label='OOD (SlowrateDoS)', 
            color='coral', density=True, edgecolor='black', linewidth=0.5)
    
    # 添加均值线
    ax.axvline(id_unc.mean(), color='steelblue', linestyle='--', linewidth=2,
               label=f'ID mean: {id_unc.mean():.4f}')
    ax.axvline(ood_unc.mean(), color='coral', linestyle='--', linewidth=2,
               label=f'OOD mean: {ood_unc.mean():.4f}')
    
    ax.set_xlabel('Epistemic Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Epistemic Uncertainty Distribution: ID vs OOD', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def plot_roc_curve(id_unc, ood_unc, save_path=None):
    """绘制ROC曲线 - IEEE期刊格式"""
    # IEEE格式设置
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
    })

    # 构建标签和分数
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # IEEE单栏宽度约3.5英寸
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.plot(fpr, tpr, color='steelblue', lw=1.5,
            label=f'BSTGCN (AUROC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

    # 标记FPR@95TPR
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    ax.scatter([fpr[idx_95]], [tpr[idx_95]], color='red', s=40, zorder=5,
               label=f'FPR@95TPR = {fpr[idx_95]:.3f}')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # 同时保存到thesis/figures
        thesis_path = Path('docs/thesis/figures') / Path(save_path).name
        plt.savefig(thesis_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        print(f"已保存: {thesis_path}")
    plt.close()

    return roc_auc


def plot_boxplot_comparison(id_unc, ood_unc, save_path=None):
    """绘制不确定性箱线图对比"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [id_unc, ood_unc]
    positions = [1, 2]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ID (8 classes)', 'OOD (SlowrateDoS)'], fontsize=12)
    ax.set_ylabel('Epistemic Uncertainty', fontsize=12)
    ax.set_title('Uncertainty Comparison: ID vs OOD', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    stats_text = (f"ID: μ={id_unc.mean():.4f}, σ={id_unc.std():.4f}\n"
                  f"OOD: μ={ood_unc.mean():.4f}, σ={ood_unc.std():.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()


def generate_report(results, save_path=None):
    """生成文本报告"""
    ood_metrics = results['ood_detection']
    id_metrics = results['id_classification']
    
    report = f"""
================================================================================
Leave-One-Out OOD Detection Experiment Report
OOD Class: {results['metadata']['ood_class_name']}
================================================================================

1. ID Classification Performance (8 classes)
   - Accuracy: {id_metrics['accuracy']:.4f}
   - Macro F1: {id_metrics['macro_f1']:.4f}

2. OOD Detection Performance
   - AUROC: {ood_metrics['auroc']:.4f}
   - AUPR: {ood_metrics['aupr']:.4f}
   - FPR@95TPR: {ood_metrics['fpr_at_95tpr']:.4f}

3. Uncertainty Statistics
   - ID samples:  {ood_metrics['id_uncertainty_mean']:.4f} ± {ood_metrics['id_uncertainty_std']:.4f}
   - OOD samples: {ood_metrics['ood_uncertainty_mean']:.4f} ± {ood_metrics['ood_uncertainty_std']:.4f}
   - Separation ratio: {ood_metrics['ood_uncertainty_mean']/ood_metrics['id_uncertainty_mean']:.2f}x

4. Dataset Statistics
   - Training samples: {results['metadata']['train_size']:,}
   - ID test samples: {results['metadata']['test_id_size']:,}
   - OOD test samples: {results['metadata']['test_ood_size']:,}

================================================================================
"""
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"报告已保存: {save_path}")
    
    return report


def main(result_path: str, output_dir: str):
    """主函数"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载实验结果...")
    results = load_results(result_path)

    id_unc = results['id_uncertainties']
    ood_unc = results['ood_uncertainties']

    print(f"\n生成可视化图表...")

    # 1. 不确定性分布直方图
    plot_uncertainty_histogram(
        id_unc, ood_unc,
        save_path=output_dir / 'ood_uncertainty_histogram.pdf'
    )

    # 2. ROC曲线
    plot_roc_curve(
        id_unc, ood_unc,
        save_path=output_dir / 'ood_roc_curve.pdf'
    )

    # 3. 箱线图
    plot_boxplot_comparison(
        id_unc, ood_unc,
        save_path=output_dir / 'ood_uncertainty_boxplot.pdf'
    )

    # 4. 文本报告
    generate_report(results, save_path=output_dir / 'ood_experiment_report.txt')

    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize OOD Experiment Results')
    parser.add_argument('--result-path', type=str, required=True,
                        help='Path to OOD experiment result pickle file')
    parser.add_argument('--output-dir', type=str, default='results/ood_experiment/figures',
                        help='Directory to save figures')

    args = parser.parse_args()
    main(args.result_path, args.output_dir)

