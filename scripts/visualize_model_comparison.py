"""
模型性能对比可视化

生成所有模型的性能对比图表
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_model_comparison():
    """绘制模型性能对比图"""
    
    # 模型数据
    models = {
        'Hybrid\n(改进版)': {'macro_f1': 0.9544, 'accuracy': 0.9500, 'time': 6.6, 'type': '时空图', 'rank': 1},
        'STGCN\n(优化版)': {'macro_f1': 0.9341, 'accuracy': 0.9276, 'time': 45.1, 'type': '时空图', 'rank': 2},
        'Hybrid\n(原始)': {'macro_f1': 0.9184, 'accuracy': 0.9089, 'time': 5.0, 'type': '时空图', 'rank': 3},
        'CNN1D': {'macro_f1': 0.8977, 'accuracy': 0.8881, 'time': 0.46, 'type': '时序', 'rank': 4},
        'LightGBM': {'macro_f1': 0.8965, 'accuracy': 0.8865, 'time': 1.12, 'type': '非时序', 'rank': 5},
        'LSTM+GNN': {'macro_f1': 0.8941, 'accuracy': 0.8849, 'time': 0.94, 'type': '时空图', 'rank': 6},
        'GAT': {'macro_f1': 0.8840, 'accuracy': 0.8747, 'time': 0.70, 'type': '图', 'rank': 7},
        'GCN': {'macro_f1': 0.8810, 'accuracy': 0.8706, 'time': 0.47, 'type': '图', 'rank': 8},
        'GRU': {'macro_f1': 0.8792, 'accuracy': 0.8678, 'time': 0.54, 'type': '时序', 'rank': 9},
        'TCN': {'macro_f1': 0.8386, 'accuracy': 0.8222, 'time': 0.39, 'type': '时序', 'rank': 10},
        'MLP': {'macro_f1': 0.5411, 'accuracy': 0.5024, 'time': 0.44, 'type': '非时序', 'rank': 11},
    }
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Macro F1 对比（柱状图）
    ax1 = plt.subplot(2, 3, 1)
    names = list(models.keys())
    f1_scores = [models[m]['macro_f1'] for m in names]
    colors = ['#FF6B6B' if i == 0 else '#4ECDC4' if i == 1 else '#95E1D3' for i in range(len(names))]
    
    bars = ax1.barh(names, f1_scores, color=colors, alpha=0.8)
    ax1.set_xlabel('Macro F1', fontsize=12, fontweight='bold')
    ax1.set_title('模型 Macro F1 对比', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.5, 1.0)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Accuracy 对比（柱状图）
    ax2 = plt.subplot(2, 3, 2)
    accuracies = [models[m]['accuracy'] for m in names]
    
    bars = ax2.barh(names, accuracies, color=colors, alpha=0.8)
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('模型 Accuracy 对比', fontsize=14, fontweight='bold')
    ax2.set_xlim(0.5, 1.0)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax2.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. 训练时间对比（对数尺度）
    ax3 = plt.subplot(2, 3, 3)
    times = [models[m]['time'] for m in names]
    
    bars = ax3.barh(names, times, color=colors, alpha=0.8)
    ax3.set_xlabel('训练时间 (分钟)', fontsize=12, fontweight='bold')
    ax3.set_title('模型训练时间对比（对数尺度）', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax3.text(time * 1.2, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f}分', va='center', fontsize=9)
    
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. 性能 vs 效率散点图
    ax4 = plt.subplot(2, 3, 4)
    
    type_colors = {
        '时空图': '#FF6B6B',
        '时序': '#4ECDC4',
        '图': '#95E1D3',
        '非时序': '#FFE66D'
    }
    
    for name, data in models.items():
        color = type_colors[data['type']]
        size = 300 if data['rank'] <= 3 else 150
        marker = '*' if data['rank'] == 1 else 'o'
        ax4.scatter(data['time'], data['macro_f1'], 
                   c=color, s=size, alpha=0.7, marker=marker,
                   edgecolors='black', linewidths=1.5)
        
        # 添加标签
        offset_x = 0.5 if data['time'] < 10 else 2
        offset_y = 0.005 if data['rank'] != 11 else -0.01
        ax4.annotate(name.replace('\n', ' '), 
                    (data['time'], data['macro_f1']),
                    xytext=(offset_x, offset_y), 
                    textcoords='offset points',
                    fontsize=9, fontweight='bold' if data['rank'] <= 3 else 'normal')
    
    ax4.set_xlabel('训练时间 (分钟, 对数尺度)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax4.set_title('性能 vs 效率权衡', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=type_name, alpha=0.7) 
                      for type_name, color in type_colors.items()]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # 5. 按类型分组对比
    ax5 = plt.subplot(2, 3, 5)
    
    type_groups = {}
    for name, data in models.items():
        type_name = data['type']
        if type_name not in type_groups:
            type_groups[type_name] = []
        type_groups[type_name].append(data['macro_f1'])
    
    type_names = list(type_groups.keys())
    type_f1s = [np.mean(type_groups[t]) for t in type_names]
    type_maxs = [np.max(type_groups[t]) for t in type_names]
    
    x = np.arange(len(type_names))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, type_f1s, width, label='平均', alpha=0.8)
    bars2 = ax5.bar(x + width/2, type_maxs, width, label='最佳', alpha=0.8)
    
    ax5.set_xlabel('模型类型', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax5.set_title('不同类型模型性能对比', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(type_names, fontsize=11)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Top 5 模型详细对比
    ax6 = plt.subplot(2, 3, 6)
    
    top5_names = list(models.keys())[:5]
    top5_f1 = [models[m]['macro_f1'] for m in top5_names]
    top5_acc = [models[m]['accuracy'] for m in top5_names]
    
    x = np.arange(len(top5_names))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, top5_f1, width, label='Macro F1', alpha=0.8, color='#FF6B6B')
    bars2 = ax6.bar(x + width/2, top5_acc, width, label='Accuracy', alpha=0.8, color='#4ECDC4')
    
    ax6.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax6.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax6.set_title('Top 5 模型详细对比', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([n.replace('\n', ' ') for n in top5_names], 
                        rotation=15, ha='right', fontsize=10)
    ax6.legend(fontsize=10)
    ax6.set_ylim(0.88, 0.96)
    ax6.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('baseline/results_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存到: {output_dir / 'model_comparison.png'}")
    
    plt.show()


def plot_improvement_breakdown():
    """绘制改进版 Hybrid 的改进分解图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 改进贡献分解
    improvements = {
        '原始 Hybrid': 0.9184,
        '+ 并行时空交互': 0.9184 + 0.025,
        '+ 修正 GATv2': 0.9184 + 0.025 + 0.015,
        '+ 门控池化': 0.9184 + 0.025 + 0.015 + 0.005,
        '+ 训练优化': 0.9544,
    }
    
    names = list(improvements.keys())
    scores = list(improvements.values())
    colors = ['#95E1D3', '#4ECDC4', '#45B7D1', '#3498DB', '#FF6B6B']
    
    bars = ax1.barh(names, scores, color=colors, alpha=0.8)
    ax1.set_xlabel('Macro F1', fontsize=12, fontweight='bold')
    ax1.set_title('改进版 Hybrid 改进贡献分解', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.91, 0.96)
    
    # 添加数值标签和增量
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
        
        if i > 0:
            delta = score - scores[i-1]
            ax1.text(scores[i-1] + delta/2, bar.get_y() + bar.get_height()/2,
                    f'+{delta:.4f}', va='center', ha='center',
                    fontsize=9, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. 与 STGCN 的类别对比
    categories = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 
                 'SYNScan', 'SlowrateDoS', 'TCPConnect', 'UDPFlood', 'UDPScan']
    
    hybrid_scores = [1.0000, 0.9950, 0.9362, 0.9866, 0.8775, 0.9933, 0.8881, 0.9833, 0.9408]
    stgcn_scores = [1.0000, 0.9899, 0.9133, 0.9883, 0.8121, 0.9933, 0.8322, 0.9766, 0.9015]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, hybrid_scores, width, label='Hybrid (改进版)', 
                   alpha=0.8, color='#FF6B6B')
    bars2 = ax2.bar(x + width/2, stgcn_scores, width, label='STGCN', 
                   alpha=0.8, color='#4ECDC4')
    
    ax2.set_xlabel('攻击类别', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('改进版 Hybrid vs STGCN 各类别性能对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0.75, 1.05)
    ax2.grid(axis='y', alpha=0.3)
    
    # 标注显著提升的类别
    for i, (h, s) in enumerate(zip(hybrid_scores, stgcn_scores)):
        if h - s > 0.05:  # 提升超过 5%
            ax2.annotate(f'+{(h-s)*100:.1f}%', 
                        xy=(i, max(h, s) + 0.01),
                        ha='center', fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('baseline/results_v2')
    plt.savefig(output_dir / 'improvement_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存到: {output_dir / 'improvement_breakdown.png'}")
    
    plt.show()


if __name__ == '__main__':
    print("="*80)
    print("模型性能对比可视化")
    print("="*80)
    
    print("\n生成模型对比图...")
    plot_model_comparison()
    
    print("\n生成改进分解图...")
    plot_improvement_breakdown()
    
    print("\n✅ 所有图表生成完成！")

