"""
生成论文所需的补充实验图表和指标

包括：
1. 校准指标表格 (ECE, Brier score, NLL)
2. Reliability Diagram (校准曲线)
3. Risk-Coverage 曲线
4. OOD ROC 曲线
5. 不确定性分位数错误率分析
6. 完整的9类F1热力图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
RESULTS_V2_DIR = Path("baseline/results_v2")
BSTGCN_DIR = Path("baseline/results/bstgcn_50packets/20251205_141432")  # 最新的BSTGCN结果
DATA_DIR = Path("data/processed/raw_packet_sequences")  # baseline使用的数据目录
BSTGCN_DATA_DIR = Path("data/processed/raw_packet_sequences_50packets")  # BSTGCN使用的数据目录
OUTPUT_DIR = Path("docs/thesis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan',
               'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

# 全局缓存测试标签
_y_test_cache = None
_y_test_bstgcn_cache = None

def get_test_labels():
    """获取baseline测试集标签（带缓存）"""
    global _y_test_cache
    if _y_test_cache is None:
        label_file = DATA_DIR / "test_labels.npy"
        if label_file.exists():
            _y_test_cache = np.load(label_file)
        else:
            print(f"Warning: Test labels not found at {label_file}")
    return _y_test_cache


def get_bstgcn_test_labels():
    """获取BSTGCN测试集标签（带缓存）"""
    global _y_test_bstgcn_cache
    if _y_test_bstgcn_cache is None:
        label_file = BSTGCN_DATA_DIR / "test_labels.npy"
        if label_file.exists():
            _y_test_bstgcn_cache = np.load(label_file)
        else:
            print(f"Warning: BSTGCN test labels not found at {label_file}")
    return _y_test_bstgcn_cache


def load_predictions(model_dir: Path):
    """加载模型预测结果"""
    pred_file = model_dir / "predictions.npy"
    y_test = get_test_labels()

    if pred_file.exists() and y_test is not None:
        y_pred = np.load(pred_file)
        # predictions.npy 只包含预测标签，没有概率
        # 返回 y_true, y_pred, y_probs (None)
        return y_test, y_pred, None
    return None, None, None


def load_bstgcn_results():
    """加载BSTGCN结果"""
    pred_file = BSTGCN_DIR / "test_predictions.npy"
    unc_file = BSTGCN_DIR / "test_uncertainties.npy"

    if pred_file.exists() and unc_file.exists():
        preds = np.load(pred_file)
        uncs = np.load(unc_file)
        return preds, uncs
    return None, None


def compute_ece(y_true, y_probs, n_bins=10):
    """计算Expected Calibration Error"""
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def compute_brier_score(y_true, y_probs):
    """计算多类Brier score"""
    n_samples, n_classes = y_probs.shape
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    return np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))


def compute_calibration_metrics(y_true, y_probs):
    """计算所有校准指标"""
    metrics = {
        'ece': compute_ece(y_true, y_probs),
        'brier': compute_brier_score(y_true, y_probs),
        'nll': log_loss(y_true, y_probs),
        'accuracy': accuracy_score(y_true, np.argmax(y_probs, axis=1)),
        'macro_f1': f1_score(y_true, np.argmax(y_probs, axis=1), average='macro')
    }
    return metrics


def plot_reliability_diagram(y_true, y_probs, model_name, ax, n_bins=10):
    """绘制单个模型的reliability diagram"""
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(accuracies[mask]))
            bin_confs.append(np.mean(confidences[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(np.nan)
            bin_confs.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_counts.append(0)
    
    # 绘制柱状图
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
    width = 0.08
    ax.bar(bin_centers, bin_accs, width=width, alpha=0.7, label=model_name)
    
    return bin_accs, bin_confs, bin_counts


def generate_confusion_matrix_comparison():
    """生成混淆矩阵对比图"""
    print("Generating confusion matrix comparison...")

    # BSTGCN混淆矩阵
    preds, uncs = load_bstgcn_results()
    y_test = get_bstgcn_test_labels()

    if preds is None or y_test is None:
        print("Warning: BSTGCN results not found")
        return

    cm = confusion_matrix(y_test, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=15, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=15, fontweight='bold')
    ax.set_title('BSTGCN Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)

    # 调整colorbar字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bstgcn_confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'bstgcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'bstgcn_confusion_matrix.png'}")
    plt.close()


def generate_uncertainty_error_analysis():
    """生成不确定性与错误率的分位数分析图"""
    print("Generating uncertainty-error analysis...")

    preds, uncs = load_bstgcn_results()
    if preds is None:
        print("Warning: BSTGCN results not found")
        return

    y_test = get_bstgcn_test_labels()
    if y_test is None:
        print("Warning: BSTGCN test labels not found")
        return

    errors = (preds != y_test).astype(float)

    # 按不确定性分成10个分位数
    n_bins = 10
    quantiles = np.percentile(uncs, np.linspace(0, 100, n_bins + 1))

    bin_error_rates = []
    bin_sizes = []
    bin_labels = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (uncs >= quantiles[i]) & (uncs <= quantiles[i+1])
        else:
            mask = (uncs >= quantiles[i]) & (uncs < quantiles[i+1])

        if np.sum(mask) > 0:
            bin_error_rates.append(np.mean(errors[mask]) * 100)
            bin_sizes.append(np.sum(mask))
            bin_labels.append(f'{i*10}-{(i+1)*10}%')

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(bin_labels))
    bars = ax.bar(x, bin_error_rates, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_bins)))

    ax.set_xlabel('Uncertainty Percentile', fontsize=15, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=15, fontweight='bold')
    ax.set_title('BSTGCN: Error Rate by Uncertainty Percentile', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 标注数值
    for bar, rate in zip(bars, bin_error_rates):
        ax.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'uncertainty_error_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'uncertainty_error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'uncertainty_error_analysis.png'}")
    plt.close()


def generate_risk_coverage_curve():
    """生成Risk-Coverage曲线"""
    print("Generating risk-coverage curve...")

    preds, uncs = load_bstgcn_results()
    if preds is None:
        print("Warning: BSTGCN results not found")
        return

    y_test = get_bstgcn_test_labels()
    if y_test is None:
        print("Warning: BSTGCN test labels not found")
        return

    errors = (preds != y_test).astype(float)

    # 按不确定性排序（从低到高）
    sorted_indices = np.argsort(uncs)
    sorted_errors = errors[sorted_indices]

    # 计算不同coverage下的risk (error rate)
    coverages = np.linspace(0.1, 1.0, 50)
    risks = []

    for cov in coverages:
        n_keep = int(len(sorted_errors) * cov)
        risk = np.mean(sorted_errors[:n_keep]) * 100
        risks.append(risk)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(coverages * 100, risks, 'b-', linewidth=2.5, label='BSTGCN (uncertainty-based)')

    # 添加随机基线
    random_risk = np.mean(errors) * 100
    ax.axhline(y=random_risk, color='r', linestyle='--', linewidth=2, label=f'Random baseline ({random_risk:.1f}%)')

    ax.set_xlabel('Coverage (%)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Risk (Error Rate %)', fontsize=15, fontweight='bold')
    ax.set_title('Risk-Coverage Curve', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_coverage_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'risk_coverage_curve.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'risk_coverage_curve.png'}")
    plt.close()


def generate_full_f1_heatmap():
    """生成完整的9类F1热力图（使用编号替代模型名字）"""
    print("Generating full F1 heatmap...")

    # 50-packet数据集路径
    RESULTS_50_DIR = Path("baseline/results_50packets")
    BSTGCN_50_DIR = Path("baseline/results/bstgcn_50packets/20251205_141432")
    STGCN_50_DIR = Path("baseline/results/hybrid_model_improved_50packets")
    DATA_50_DIR = Path("data/processed/raw_packet_sequences_50packets")

    # 模型列表（包含所有baseline + STGCN + BSTGCN）
    models_info = [
        ('cnn1d', 'CNN1D', RESULTS_50_DIR / 'cnn1d'),
        ('gru', 'GRU', RESULTS_50_DIR / 'gru'),
        ('tcn', 'TCN', RESULTS_50_DIR / 'tcn'),
        ('lightgbm', 'LightGBM', RESULTS_50_DIR / 'lightgbm'),
        ('xgboost', 'XGBoost', RESULTS_50_DIR / 'xgboost'),
        ('gcn', 'GCN', RESULTS_50_DIR / 'gcn'),
        ('gat', 'GAT', RESULTS_50_DIR / 'gat'),
        ('lstm_gnn', 'LSTM+GNN', RESULTS_50_DIR / 'lstm_gnn'),
        ('seq_stgcn', 'Seq-STGCN', RESULTS_50_DIR / 'stgcn'),
        ('stgcn', 'STGCN (Ours)', STGCN_50_DIR),
        ('bstgcn', 'BSTGCN (Ours)', BSTGCN_50_DIR),
    ]

    # 加载测试标签
    y_test_50 = np.load(DATA_50_DIR / "test_labels.npy")

    f1_matrix = []
    model_labels = []

    for model_id, model_label, model_dir in models_info:
        model_labels.append(model_label)

        # 根据不同模型类型加载预测结果
        if model_id == 'bstgcn':
            pred_file = model_dir / "test_predictions.npy"
        else:
            pred_file = model_dir / "predictions.npy"

        if pred_file.exists():
            y_pred = np.load(pred_file)
            per_class_f1 = f1_score(y_test_50, y_pred, average=None)
            f1_matrix.append(per_class_f1)
            print(f"  Loaded {model_label}: mean F1 = {np.mean(per_class_f1):.4f}")
        else:
            print(f"  Warning: predictions not found for {model_label} at {pred_file}")
            f1_matrix.append([np.nan] * 9)

    f1_matrix = np.array(f1_matrix)

    # 创建图形，增大尺寸
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=model_labels,
                vmin=0.6, vmax=1.0, ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_xlabel('Attack Type', fontsize=16, fontweight='bold')
    ax.set_ylabel('Model', fontsize=16, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison', fontsize=18, fontweight='bold')

    # 调整坐标轴标签字体大小
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=13, rotation=35)

    # 调整colorbar字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_f1_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_class_f1_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'per_class_f1_heatmap.png'}")
    plt.close()


def generate_classification_comparison_table():
    """生成分类性能对比表"""
    print("Generating classification comparison table...")

    models = ['cnn1d', 'gru', 'tcn', 'lightgbm', 'xgboost', 'gcn', 'gat', 'lstm_gnn', 'stgcn']
    model_labels = ['CNN1D', 'GRU', 'TCN', 'LightGBM', 'XGBoost', 'GCN', 'GAT', 'LSTM+GNN', 'STGCN']

    results = []

    for model, label in zip(models, model_labels):
        model_dir = RESULTS_V2_DIR / model
        y_true, y_pred, _ = load_predictions(model_dir)

        if y_true is not None:
            metrics = {
                'model': label,
                'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro'),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted')
            }
            results.append(metrics)

    # 添加BSTGCN结果
    preds, _ = load_bstgcn_results()
    y_test = get_bstgcn_test_labels()
    if preds is not None and y_test is not None:
        results.append({
            'model': 'BSTGCN',
            'accuracy': accuracy_score(y_test, preds),
            'macro_f1': f1_score(y_test, preds, average='macro'),
            'weighted_f1': f1_score(y_test, preds, average='weighted')
        })

    # 打印LaTeX表格
    print("\n" + "="*60)
    print("LaTeX Table: Classification Performance")
    print("="*60)
    print("\\begin{table}[!t]")
    print("\\caption{Classification Performance Comparison\\label{tab:classification}}")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro F1} & \\textbf{Weighted F1} \\\\")
    print("\\hline")

    for r in results:
        if r['model'] == 'BSTGCN':
            print(f"\\textbf{{{r['model']} (Ours)}} & \\textbf{{{r['accuracy']:.4f}}} & \\textbf{{{r['macro_f1']:.4f}}} & \\textbf{{{r['weighted_f1']:.4f}}} \\\\")
        else:
            print(f"{r['model']} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & {r['weighted_f1']:.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    return results


def main():
    """主函数"""
    print("="*60)
    print("Generating thesis figures and tables")
    print("="*60)

    # 1. 分类性能表格
    generate_classification_comparison_table()

    # 2. 不确定性-错误分析
    generate_uncertainty_error_analysis()

    # 3. Risk-Coverage曲线
    generate_risk_coverage_curve()

    # 4. 完整F1热力图
    generate_full_f1_heatmap()

    # 5. 混淆矩阵
    generate_confusion_matrix_comparison()

    print("\n" + "="*60)
    print("All figures generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

