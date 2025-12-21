"""
可视化综合基线实验结果
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验结果数据
models = ['MLP', 'TCN', 'GRU', 'GCN', 'GAT', 'LightGBM', 'CNN1D', 'LSTM+GNN', 'Hybrid', 'STGCN']
f1_scores = [0.5411, 0.8386, 0.8792, 0.8810, 0.8840, 0.8965, 0.8977, 0.8941, 0.9184, 0.9399]
accuracies = [0.5024, 0.8222, 0.8678, 0.8706, 0.8747, 0.8865, 0.8881, 0.8849, 0.9089, 0.9349]
train_times = [24.66, 21.14, 30.58, 26.73, 39.30, 67.28, 25.35, 54.81, 300, 2604.61]

# 模型类型
model_types = ['非时序', '时序', '时序', '图结构', '图结构', '非时序', '时序', '时空结合', '时空结合', '时空结合']
type_colors = {
    '非时序': '#e74c3c',
    '时序': '#3498db',
    '图结构': '#2ecc71',
    '时空结合': '#f39c12'
}
colors = [type_colors[t] for t in model_types]

# 创建图表
fig = plt.figure(figsize=(20, 12))

# 1. F1分数对比
ax1 = plt.subplot(2, 3, 1)
bars = ax1.barh(models, f1_scores, color=colors, alpha=0.8)
ax1.set_xlabel('Macro F1 Score', fontsize=12)
ax1.set_title('模型性能对比 (Macro F1)', fontsize=14, fontweight='bold')
ax1.set_xlim(0.4, 1.0)
ax1.axvline(x=0.9, color='red', linestyle='--', alpha=0.5, label='优秀线 (0.90)')
ax1.grid(axis='x', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center', fontsize=10)
ax1.legend()

# 2. 训练时间对比（对数尺度）
ax2 = plt.subplot(2, 3, 2)
bars = ax2.barh(models, train_times, color=colors, alpha=0.8)
ax2.set_xlabel('训练时间 (秒, 对数尺度)', fontsize=12)
ax2.set_title('训练效率对比', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, time) in enumerate(zip(bars, train_times)):
    if time < 60:
        text = f'{time:.1f}s'
    elif time < 3600:
        text = f'{time/60:.1f}min'
    else:
        text = f'{time/3600:.1f}h'
    ax2.text(time * 1.2, bar.get_y() + bar.get_height()/2, 
             text, va='center', fontsize=9)

# 3. F1 vs 训练时间散点图
ax3 = plt.subplot(2, 3, 3)
for model, f1, time, color, mtype in zip(models, f1_scores, train_times, colors, model_types):
    ax3.scatter(time, f1, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.annotate(model, (time, f1), xytext=(5, 5), textcoords='offset points', 
                fontsize=9, alpha=0.8)
ax3.set_xlabel('训练时间 (秒, 对数尺度)', fontsize=12)
ax3.set_ylabel('Macro F1 Score', fontsize=12)
ax3.set_title('性能-效率权衡', fontsize=14, fontweight='bold')
ax3.set_xscale('log')
ax3.grid(alpha=0.3)
ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)

# 4. 模型类型分组对比
ax4 = plt.subplot(2, 3, 4)
type_f1 = {}
for mtype, f1 in zip(model_types, f1_scores):
    if mtype not in type_f1:
        type_f1[mtype] = []
    type_f1[mtype].append(f1)

type_names = list(type_f1.keys())
type_means = [np.mean(type_f1[t]) for t in type_names]
type_maxs = [np.max(type_f1[t]) for t in type_names]
type_colors_list = [type_colors[t] for t in type_names]

x = np.arange(len(type_names))
width = 0.35
bars1 = ax4.bar(x - width/2, type_means, width, label='平均F1', 
               color=type_colors_list, alpha=0.6)
bars2 = ax4.bar(x + width/2, type_maxs, width, label='最高F1', 
               color=type_colors_list, alpha=0.9)

ax4.set_ylabel('Macro F1 Score', fontsize=12)
ax4.set_title('模型类型对比', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(type_names, fontsize=11)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 5. 性能提升分析
ax5 = plt.subplot(2, 3, 5)
baseline_f1 = f1_scores[0]  # MLP
improvements = [(f1 - baseline_f1) / baseline_f1 * 100 for f1 in f1_scores]
bars = ax5.barh(models, improvements, color=colors, alpha=0.8)
ax5.set_xlabel('相对MLP的提升 (%)', fontsize=12)
ax5.set_title('性能提升分析 (相对MLP基线)', fontsize=14, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    ax5.text(imp + 2, bar.get_y() + bar.get_height()/2, 
             f'{imp:.1f}%', va='center', fontsize=10)

# 6. 关键发现总结
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

findings = [
    "关键发现总结",
    "",
    "1. 时序建模的必要性 ✓",
    "   MLP → GRU: +62.5% (+0.338)",
    "",
    "2. 图结构的贡献 ✓",
    "   GRU → LSTM+GNN: +1.7% (+0.015)",
    "",
    "3. 时空建模策略 ⚠",
    "   • 并行 (STGCN): 0.9399 🥇",
    "   • 串行 (Hybrid): 0.9184 🥈",
    "   • 差距: 2.3% (+0.022)",
    "",
    "4. 意外发现",
    "   • 简单 > 复杂",
    "   • 卷积 > 注意力",
    "   • 并行 > 串行",
    "",
    "5. 最佳模型: STGCN",
    "   • F1: 0.9399",
    "   • 参数: ~200K",
    "   • 策略: 并行时空卷积"
]

y_pos = 0.95
for line in findings:
    if line.startswith("关键发现"):
        ax6.text(0.05, y_pos, line, fontsize=16, fontweight='bold', 
                transform=ax6.transAxes)
    elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
        ax6.text(0.05, y_pos, line, fontsize=12, fontweight='bold', 
                transform=ax6.transAxes)
    elif "🥇" in line or "🥈" in line:
        ax6.text(0.1, y_pos, line, fontsize=11, 
                transform=ax6.transAxes, color='#e74c3c')
    else:
        ax6.text(0.1, y_pos, line, fontsize=11, 
                transform=ax6.transAxes)
    y_pos -= 0.04

plt.tight_layout()
plt.savefig('baseline/results_v2/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 可视化结果已保存到: baseline/results_v2/comprehensive_comparison.png")

# 创建第二个图：详细的模型架构对比
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# STGCN vs Hybrid 详细对比
ax1 = axes[0, 0]
comparison_data = {
    'F1 Score': [0.9399, 0.9184],
    'Accuracy': [0.9349, 0.9089],
    'Parameters (K)': [200, 500],
    'Train Time (min)': [43.4, 5.0]
}

x = np.arange(len(comparison_data))
width = 0.35
stgcn_values = [comparison_data[k][0] for k in comparison_data.keys()]
hybrid_values = [comparison_data[k][1] for k in comparison_data.keys()]

# 归一化显示
stgcn_norm = [v / max(stgcn_values[i], hybrid_values[i]) for i, v in enumerate(stgcn_values)]
hybrid_norm = [v / max(stgcn_values[i], hybrid_values[i]) for i, v in enumerate(hybrid_values)]

bars1 = ax1.bar(x - width/2, stgcn_norm, width, label='STGCN', color='#f39c12', alpha=0.8)
bars2 = ax1.bar(x + width/2, hybrid_norm, width, label='Hybrid', color='#3498db', alpha=0.8)

ax1.set_ylabel('归一化值', fontsize=12)
ax1.set_title('STGCN vs Hybrid 详细对比', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_data.keys(), fontsize=10, rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加实际值标注
for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    ax1.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02,
            f'{stgcn_values[i]:.2f}', ha='center', va='bottom', fontsize=9)
    ax1.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
            f'{hybrid_values[i]:.2f}', ha='center', va='bottom', fontsize=9)

# 时空建模策略对比
ax2 = axes[0, 1]
strategies = ['LSTM+GNN\n(串行)', 'Hybrid\n(串行增强)', 'STGCN\n(并行)']
strategy_f1 = [0.8941, 0.9184, 0.9399]
strategy_colors = ['#3498db', '#9b59b6', '#f39c12']

bars = ax2.bar(strategies, strategy_f1, color=strategy_colors, alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Macro F1 Score', fontsize=12)
ax2.set_title('时空建模策略对比', fontsize=14, fontweight='bold')
ax2.set_ylim(0.85, 0.95)
ax2.grid(axis='y', alpha=0.3)

for bar, f1 in zip(bars, strategy_f1):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{f1:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加提升箭头
ax2.annotate('', xy=(1, 0.9184), xytext=(0, 0.8941),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax2.text(0.5, 0.905, '+2.7%', ha='center', fontsize=10, color='green', fontweight='bold')

ax2.annotate('', xy=(2, 0.9399), xytext=(1, 0.9184),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax2.text(1.5, 0.929, '+2.3%', ha='center', fontsize=10, color='green', fontweight='bold')

# 模型复杂度 vs 性能
ax3 = axes[1, 0]
model_params = [50, 100, 150, 100, 56, 200, 120, 250, 500, 200]  # 估计参数量(K)
ax3.scatter(model_params, f1_scores, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

for model, param, f1 in zip(models, model_params, f1_scores):
    ax3.annotate(model, (param, f1), xytext=(5, 5), textcoords='offset points', 
                fontsize=9, alpha=0.8)

ax3.set_xlabel('参数量 (K)', fontsize=12)
ax3.set_ylabel('Macro F1 Score', fontsize=12)
ax3.set_title('模型复杂度 vs 性能', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# 添加帕累托前沿
pareto_models = ['STGCN', 'CNN1D', 'GRU']
pareto_params = [200, 120, 150]
pareto_f1 = [0.9399, 0.8977, 0.8792]
ax3.plot(pareto_params, pareto_f1, 'r--', alpha=0.5, linewidth=2, label='帕累托前沿')
ax3.legend()

# 架构组件对比
ax4 = axes[1, 1]
ax4.axis('off')

arch_comparison = [
    "STGCN vs Hybrid 架构对比",
    "",
    "STGCN (并行时空建模):",
    "  ✓ 时间卷积 (Conv1D)",
    "  ✓ 空间图卷积 (GCN)",
    "  ✓ 并行交替处理",
    "  ✓ 简单直接",
    "  → F1: 0.9399, 参数: 200K",
    "",
    "Hybrid (串行时空建模):",
    "  • Transformer (全局注意力)",
    "  • 动态图构建",
    "  • GNN (图注意力)",
    "  • 攻击模式记忆库",
    "  • 知识融合层",
    "  → F1: 0.9184, 参数: 500K",
    "",
    "关键差异:",
    "  1. 并行 vs 串行",
    "  2. 卷积 vs 注意力",
    "  3. 简单 vs 复杂",
    "  4. 静态图 vs 动态图",
    "",
    "结论: 简单并行 > 复杂串行"
]

y_pos = 0.95
for line in arch_comparison:
    if "架构对比" in line:
        ax4.text(0.05, y_pos, line, fontsize=14, fontweight='bold', 
                transform=ax4.transAxes)
    elif line.startswith(("STGCN", "Hybrid", "关键差异", "结论")):
        color = '#f39c12' if 'STGCN' in line else '#3498db' if 'Hybrid' in line else 'black'
        ax4.text(0.05, y_pos, line, fontsize=12, fontweight='bold', 
                transform=ax4.transAxes, color=color)
    elif line.startswith("  ✓"):
        ax4.text(0.1, y_pos, line, fontsize=10, 
                transform=ax4.transAxes, color='green')
    elif line.startswith("  •"):
        ax4.text(0.1, y_pos, line, fontsize=10, 
                transform=ax4.transAxes, color='blue')
    elif line.startswith("  →"):
        ax4.text(0.1, y_pos, line, fontsize=10, fontweight='bold',
                transform=ax4.transAxes, color='red')
    else:
        ax4.text(0.1, y_pos, line, fontsize=10, 
                transform=ax4.transAxes)
    y_pos -= 0.038

plt.tight_layout()
plt.savefig('baseline/results_v2/stgcn_vs_hybrid_detailed.png', dpi=300, bbox_inches='tight')
print("✅ 详细对比图已保存到: baseline/results_v2/stgcn_vs_hybrid_detailed.png")

plt.show()

