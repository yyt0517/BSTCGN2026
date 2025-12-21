"""
可视化所有模型的对比结果
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_report(report_path):
    """解析评估报告"""
    # 尝试多种编码
    for encoding in ['utf-8', 'gbk', 'latin-1']:
        try:
            with open(report_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"⚠️  无法读取 {report_path}")
        return None
    
    # 提取指标
    acc_match = re.search(r'Accuracy:\s+([\d.]+)', content)
    macro_f1_match = re.search(r'Macro F1:\s+([\d.]+)', content)
    weighted_f1_match = re.search(r'Weighted F1:\s+([\d.]+)', content)
    
    if acc_match and macro_f1_match and weighted_f1_match:
        return {
            'accuracy': float(acc_match.group(1)),
            'macro_f1': float(macro_f1_match.group(1)),
            'weighted_f1': float(weighted_f1_match.group(1))
        }
    return None


def collect_results():
    """收集所有模型结果"""
    results_dir = Path('baseline/results')
    
    models = {
        'LSTM': 'lstm_raw_fixed',
        'GRU': 'gru_raw_fixed',
        'BiLSTM': 'bilstm_raw_fixed',
        'CNN1D': 'cnn1d_raw_fixed',
        'Random Forest': 'random_forest_raw_fixed',
        'XGBoost': 'xgboost_raw_fixed',
        'LightGBM': 'lightgbm_raw_fixed',
        'Transformer': 'transformer_raw_fixed',
        'Hybrid Model': 'hybrid_model_fixed'
    }
    
    results = {}
    for name, dir_name in models.items():
        report_path = results_dir / dir_name / 'evaluation_report.txt'
        if report_path.exists():
            metrics = parse_report(report_path)
            if metrics:
                results[name] = metrics
                print(f"✓ {name}: Macro F1 = {metrics['macro_f1']:.4f}")
        else:
            print(f"✗ {name}: 报告不存在")
    
    return results


def plot_comparison(results):
    """绘制对比图"""
    # 按Macro F1排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'])
    
    models = [item[0] for item in sorted_results]
    accuracy = [item[1]['accuracy'] for item in sorted_results]
    macro_f1 = [item[1]['macro_f1'] for item in sorted_results]
    weighted_f1 = [item[1]['weighted_f1'] for item in sorted_results]
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 颜色映射
    colors = ['#ff6b6b' if 'Hybrid' in m else '#4ecdc4' if m in ['CNN1D', 'LightGBM', 'GRU'] 
              else '#95e1d3' for m in models]
    
    # 1. Accuracy
    axes[0].barh(models, accuracy, color=colors, alpha=0.8)
    axes[0].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('模型准确率对比', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.75, 0.95)
    for i, v in enumerate(accuracy):
        axes[0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. Macro F1
    axes[1].barh(models, macro_f1, color=colors, alpha=0.8)
    axes[1].set_xlabel('Macro F1', fontsize=12, fontweight='bold')
    axes[1].set_title('模型Macro F1对比', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0.80, 0.93)
    for i, v in enumerate(macro_f1):
        axes[1].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
    axes[1].grid(axis='x', alpha=0.3)
    
    # 3. Weighted F1
    axes[2].barh(models, weighted_f1, color=colors, alpha=0.8)
    axes[2].set_xlabel('Weighted F1', fontsize=12, fontweight='bold')
    axes[2].set_title('模型Weighted F1对比', fontsize=14, fontweight='bold')
    axes[2].set_xlim(0.75, 0.95)
    for i, v in enumerate(weighted_f1):
        axes[2].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = Path('baseline/results/all_models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {save_path}")
    
    plt.show()


def plot_category_comparison(results):
    """绘制不同类型模型的对比"""
    # 分类
    deep_learning = ['LSTM', 'GRU', 'BiLSTM', 'CNN1D', 'Transformer']
    tree_models = ['Random Forest', 'XGBoost', 'LightGBM']
    hybrid = ['Hybrid Model']
    
    categories = {
        '深度学习': [results[m]['macro_f1'] for m in deep_learning if m in results],
        '树模型': [results[m]['macro_f1'] for m in tree_models if m in results],
        '混合模型': [results[m]['macro_f1'] for m in hybrid if m in results]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = []
    labels = []
    data = []
    colors_list = ['#4ecdc4', '#95e1d3', '#ff6b6b']
    
    pos = 0
    for i, (cat, values) in enumerate(categories.items()):
        for v in values:
            positions.append(pos)
            data.append(v)
            pos += 1
        labels.append(cat)
        pos += 0.5
    
    # 绘制箱线图
    bp = ax.boxplot([categories['深度学习'], categories['树模型'], categories['混合模型']],
                     positions=[1, 2, 3],
                     widths=0.6,
                     patch_artist=True,
                     labels=labels)
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax.set_title('不同类型模型的性能分布', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标注
    for i, (cat, values) in enumerate(categories.items(), 1):
        for v in values:
            ax.scatter(i, v, color='red', s=50, zorder=3, alpha=0.6)
            ax.text(i + 0.15, v, f'{v:.4f}', fontsize=9, va='center')
    
    plt.tight_layout()
    
    save_path = Path('baseline/results/model_category_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    
    plt.show()


def print_summary_table(results):
    """打印汇总表格"""
    print("\n" + "="*100)
    print("所有模型性能汇总".center(100))
    print("="*100)
    
    print("\n| 排名 | 模型 | Accuracy | Macro F1 | Weighted F1 |")
    print("|------|------|----------|----------|-------------|")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
    
    medals = ['🥇', '🥈', '🥉']
    for i, (name, metrics) in enumerate(sorted_results, 1):
        medal = medals[i-1] if i <= 3 else '  '
        print(f"| {medal} {i:<2} | {name:<15} | {metrics['accuracy']:.4f}   | {metrics['macro_f1']:.4f}   | {metrics['weighted_f1']:.4f}    |")
    
    print("\n" + "="*100)
    
    # 计算统计信息
    best_model = sorted_results[0]
    second_model = sorted_results[1]
    
    print(f"\n🏆 最佳模型: {best_model[0]}")
    print(f"   - Accuracy:    {best_model[1]['accuracy']:.4f}")
    print(f"   - Macro F1:    {best_model[1]['macro_f1']:.4f}")
    print(f"   - Weighted F1: {best_model[1]['weighted_f1']:.4f}")
    
    print(f"\n📊 vs 第二名 ({second_model[0]}):")
    print(f"   - Accuracy:    +{(best_model[1]['accuracy'] - second_model[1]['accuracy'])*100:.2f}%")
    print(f"   - Macro F1:    +{(best_model[1]['macro_f1'] - second_model[1]['macro_f1'])*100:.2f}%")
    print(f"   - Weighted F1: +{(best_model[1]['weighted_f1'] - second_model[1]['weighted_f1'])*100:.2f}%")
    
    # 找到Transformer
    transformer_metrics = results.get('Transformer')
    if transformer_metrics and best_model[0] == 'Hybrid Model':
        print(f"\n📈 Hybrid Model vs Transformer Baseline:")
        print(f"   - Accuracy:    +{(best_model[1]['accuracy'] - transformer_metrics['accuracy'])*100:.2f}%")
        print(f"   - Macro F1:    +{(best_model[1]['macro_f1'] - transformer_metrics['macro_f1'])*100:.2f}%")
        print(f"   - Weighted F1: +{(best_model[1]['weighted_f1'] - transformer_metrics['weighted_f1'])*100:.2f}%")


def main():
    """主函数"""
    print("="*100)
    print("收集所有模型结果...".center(100))
    print("="*100)
    
    results = collect_results()
    
    if not results:
        print("\n❌ 未找到任何结果！")
        return
    
    print(f"\n✓ 成功收集 {len(results)} 个模型的结果")
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 绘制对比图
    print("\n" + "="*100)
    print("生成可视化图表...".center(100))
    print("="*100)
    
    plot_comparison(results)
    plot_category_comparison(results)
    
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()

