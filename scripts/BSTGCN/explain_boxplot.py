"""
箱型图详细解释和可视化

运行方法：
    python scripts/BSTGCN/explain_boxplot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def create_boxplot_explanation():
    """创建箱型图解释图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== 左图：箱型图结构说明 ==========
    ax1 = axes[0]
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('箱型图结构说明', fontsize=14, fontweight='bold', pad=20)
    
    # 绘制箱型图示意
    x_center = 1.0
    
    # 离群点
    ax1.plot(x_center, 0.95, 'ko', markersize=8)
    ax1.text(x_center + 0.15, 0.95, '离群点\n(异常值)', fontsize=10, va='center')
    
    # 上须
    ax1.plot([x_center, x_center], [0.75, 0.90], 'k-', linewidth=2)
    ax1.plot([x_center-0.05, x_center+0.05], [0.90, 0.90], 'k-', linewidth=2)
    ax1.text(x_center + 0.15, 0.90, '上须 (Q3 + 1.5×IQR)', fontsize=10, va='center')
    
    # 箱体
    rect = Rectangle((x_center-0.1, 0.45), 0.2, 0.3, 
                     linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.5)
    ax1.add_patch(rect)
    ax1.text(x_center + 0.15, 0.75, 'Q3 (75%分位数)', fontsize=10, va='center')
    ax1.text(x_center + 0.15, 0.45, 'Q1 (25%分位数)', fontsize=10, va='center')
    ax1.text(x_center - 0.35, 0.60, '箱体\n(50%数据)', fontsize=10, va='center', ha='center')
    
    # 中位数
    ax1.plot([x_center-0.1, x_center+0.1], [0.60, 0.60], 'r-', linewidth=3)
    ax1.text(x_center + 0.15, 0.60, '中位数 (Q2)', fontsize=10, va='center', color='red')
    
    # 下须
    ax1.plot([x_center, x_center], [0.30, 0.45], 'k-', linewidth=2)
    ax1.plot([x_center-0.05, x_center+0.05], [0.30, 0.30], 'k-', linewidth=2)
    ax1.text(x_center + 0.15, 0.30, '下须 (Q1 - 1.5×IQR)', fontsize=10, va='center')
    
    # 添加说明
    ax1.text(1.0, 0.10, 'IQR = Q3 - Q1 (四分位距)', 
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== 右图：当前结果解释 ==========
    ax2 = axes[1]
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 0.5)
    ax2.set_title('当前结果解释', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Epistemic Uncertainty', fontsize=12)
    ax2.set_xticks([0.7, 1.8])
    ax2.set_xticklabels(['Correct\n(正确预测)', 'Error\n(错误预测)'], fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # 模拟数据（基于你的结果）
    correct_data = {
        'median': 0.01,
        'q1': 0.005,
        'q3': 0.02,
        'whisker_low': 0.001,
        'whisker_high': 0.05,
        'outliers': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.44]
    }
    
    error_data = {
        'median': 0.03,
        'q1': 0.02,
        'q3': 0.11,
        'whisker_low': 0.01,
        'whisker_high': 0.25,
        'outliers': [0.3, 0.35, 0.4, 0.43, 0.45, 0.48]
    }
    
    # 绘制Correct箱型图
    x1 = 0.7
    # 箱体
    rect1 = Rectangle((x1-0.15, correct_data['q1']), 0.3, correct_data['q3']-correct_data['q1'],
                     linewidth=2, edgecolor='C0', facecolor='C0', alpha=0.3)
    ax2.add_patch(rect1)
    # 中位数
    ax2.plot([x1-0.15, x1+0.15], [correct_data['median'], correct_data['median']], 
            'orange', linewidth=3, label='中位数')
    # 须
    ax2.plot([x1, x1], [correct_data['whisker_low'], correct_data['q1']], 'C0-', linewidth=1.5)
    ax2.plot([x1, x1], [correct_data['q3'], correct_data['whisker_high']], 'C0-', linewidth=1.5)
    ax2.plot([x1-0.1, x1+0.1], [correct_data['whisker_low'], correct_data['whisker_low']], 'C0-', linewidth=1.5)
    ax2.plot([x1-0.1, x1+0.1], [correct_data['whisker_high'], correct_data['whisker_high']], 'C0-', linewidth=1.5)
    # 离群点
    for outlier in correct_data['outliers']:
        ax2.plot(x1, outlier, 'ko', markersize=4, alpha=0.5)
    
    # 绘制Error箱型图
    x2 = 1.8
    # 箱体
    rect2 = Rectangle((x2-0.15, error_data['q1']), 0.3, error_data['q3']-error_data['q1'],
                     linewidth=2, edgecolor='C1', facecolor='C1', alpha=0.3)
    ax2.add_patch(rect2)
    # 中位数
    ax2.plot([x2-0.15, x2+0.15], [error_data['median'], error_data['median']], 
            'orange', linewidth=3)
    # 须
    ax2.plot([x2, x2], [error_data['whisker_low'], error_data['q1']], 'C1-', linewidth=1.5)
    ax2.plot([x2, x2], [error_data['q3'], error_data['whisker_high']], 'C1-', linewidth=1.5)
    ax2.plot([x2-0.1, x2+0.1], [error_data['whisker_low'], error_data['whisker_low']], 'C1-', linewidth=1.5)
    ax2.plot([x2-0.1, x2+0.1], [error_data['whisker_high'], error_data['whisker_high']], 'C1-', linewidth=1.5)
    # 离群点
    for outlier in error_data['outliers']:
        ax2.plot(x2, outlier, 'ko', markersize=4, alpha=0.5)
    
    # 添加标注
    ax2.annotate(f'中位数: {correct_data["median"]:.3f}', 
                xy=(x1, correct_data['median']), xytext=(x1-0.4, correct_data['median']),
                fontsize=9, color='C0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='C0', lw=1.5))
    
    ax2.annotate(f'中位数: {error_data["median"]:.3f}', 
                xy=(x2, error_data['median']), xytext=(x2+0.4, error_data['median']),
                fontsize=9, color='C1', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='C1', lw=1.5))
    
    # 添加重叠区域标注
    overlap_y = 0.025
    ax2.axhspan(0.02, 0.05, alpha=0.2, color='yellow')
    ax2.text(1.25, overlap_y, '重叠区域\n(区分困难)', 
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('docs/BSTGCN/boxplot_explanation.png', dpi=150, bbox_inches='tight')
    print("✅ 图片已保存到: docs/BSTGCN/boxplot_explanation.png")
    plt.show()


def print_interpretation():
    """打印文字解释"""
    
    print("\n" + "=" * 80)
    print("箱型图详细解释")
    print("=" * 80)
    
    print("\n📊 箱型图的5个关键数值：")
    print("  1. 最小值（下须）：排除离群点后的最小值")
    print("  2. Q1（25%分位数）：25%的数据小于此值")
    print("  3. Q2（中位数）：50%的数据小于此值")
    print("  4. Q3（75%分位数）：75%的数据小于此值")
    print("  5. 最大值（上须）：排除离群点后的最大值")
    
    print("\n📈 当前结果分析：")
    print("\n  Correct（正确预测）：")
    print("    - 中位数 ≈ 0.01  ← 50%的正确预测不确定性 < 0.01")
    print("    - Q1 ≈ 0.005     ← 25%的正确预测不确定性 < 0.005")
    print("    - Q3 ≈ 0.02      ← 75%的正确预测不确定性 < 0.02")
    print("    - 箱体很窄       ← 数据集中，不确定性一致性好")
    print("    - 有离群点       ← 少数正确预测有高不确定性")
    
    print("\n  Error（错误预测）：")
    print("    - 中位数 ≈ 0.03  ← 50%的错误预测不确定性 < 0.03")
    print("    - Q1 ≈ 0.02      ← 25%的错误预测不确定性 < 0.02")
    print("    - Q3 ≈ 0.11      ← 75%的错误预测不确定性 < 0.11")
    print("    - 箱体较宽       ← 数据分散，不确定性差异大")
    print("    - 有离群点       ← 部分错误预测有极高不确定性")
    
    print("\n✅ 好消息：")
    print("  - 错误预测的中位数（0.03）是正确预测（0.01）的 3倍")
    print("  - 错误预测的Q3（0.11）明显高于正确预测（0.02）")
    print("  - 不确定性与错误有正相关（0.2815）")
    
    print("\n⚠️  问题：")
    print("  - 两个分布有重叠（0.02-0.05区域）")
    print("  - 25%的错误预测不确定性很低（< 0.02）")
    print("  - 部分正确预测有高不确定性（离群点）")
    print("  - 区分度不够理想")
    
    print("\n💡 理想情况：")
    print("  - 错误预测中位数 > 0.05")
    print("  - 两个分布重叠更少")
    print("  - 相关性 > 0.35")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_interpretation()
    create_boxplot_explanation()

