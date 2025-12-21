"""
分析不确定性阈值的效果

基于箱型图结果，分析不同阈值下的性能

运行方法：
    python scripts/BSTGCN/analyze_uncertainty_threshold.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def simulate_threshold_analysis():
    """模拟不同阈值下的效果"""
    
    # 基于你的结果模拟数据
    np.random.seed(42)
    
    # Correct预测的不确定性分布（正态分布，中心0.5，标准差0.15）
    n_correct = 30000
    correct_unc = np.clip(np.random.normal(0.5, 0.15, n_correct), 0, 1.2)
    
    # Error预测的不确定性分布（正态分布，中心0.8，标准差0.12）
    n_error = 2000
    error_unc = np.clip(np.random.normal(0.8, 0.12, n_error), 0.3, 1.2)
    
    # 测试不同阈值
    thresholds = np.linspace(0, 1.2, 100)
    
    results = {
        'threshold': [],
        'reject_rate': [],
        'error_rate_before': [],
        'error_rate_after': [],
        'precision_gain': []
    }
    
    total_samples = n_correct + n_error
    initial_error_rate = n_error / total_samples
    
    for threshold in thresholds:
        # 拒绝高不确定性样本
        correct_kept = correct_unc <= threshold
        error_kept = error_unc <= threshold
        
        n_correct_kept = np.sum(correct_kept)
        n_error_kept = np.sum(error_kept)
        n_kept = n_correct_kept + n_error_kept
        
        if n_kept == 0:
            continue
        
        reject_rate = 1 - (n_kept / total_samples)
        error_rate_after = n_error_kept / n_kept if n_kept > 0 else 0
        precision_gain = (initial_error_rate - error_rate_after) / initial_error_rate * 100
        
        results['threshold'].append(threshold)
        results['reject_rate'].append(reject_rate * 100)
        results['error_rate_before'].append(initial_error_rate * 100)
        results['error_rate_after'].append(error_rate_after * 100)
        results['precision_gain'].append(precision_gain)
    
    return results


def plot_threshold_analysis(results):
    """绘制阈值分析图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========== 图1：拒绝率 vs 阈值 ==========
    ax1 = axes[0, 0]
    ax1.plot(results['threshold'], results['reject_rate'], 'b-', linewidth=2)
    ax1.set_xlabel('不确定性阈值', fontsize=11)
    ax1.set_ylabel('拒绝率 (%)', fontsize=11)
    ax1.set_title('拒绝率 vs 阈值', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 标注关键点
    key_thresholds = [0.35, 0.50, 0.65, 0.70, 0.80, 0.90]
    for t in key_thresholds:
        idx = np.argmin(np.abs(np.array(results['threshold']) - t))
        ax1.axvline(t, color='r', linestyle='--', alpha=0.3)
        ax1.text(t, results['reject_rate'][idx] + 2, f'{t:.2f}', 
                ha='center', fontsize=9, color='red')
    
    # ========== 图2：错误率 vs 阈值 ==========
    ax2 = axes[0, 1]
    ax2.plot(results['threshold'], results['error_rate_before'], 'r--', 
            linewidth=2, label='拒绝前', alpha=0.5)
    ax2.plot(results['threshold'], results['error_rate_after'], 'g-', 
            linewidth=2, label='拒绝后')
    ax2.set_xlabel('不确定性阈值', fontsize=11)
    ax2.set_ylabel('错误率 (%)', fontsize=11)
    ax2.set_title('错误率 vs 阈值', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # ========== 图3：精度提升 vs 阈值 ==========
    ax3 = axes[1, 0]
    ax3.plot(results['threshold'], results['precision_gain'], 'purple', linewidth=2)
    ax3.set_xlabel('不确定性阈值', fontsize=11)
    ax3.set_ylabel('精度提升 (%)', fontsize=11)
    ax3.set_title('精度提升 vs 阈值', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # ========== 图4：推荐阈值 ==========
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title('推荐阈值策略', fontsize=12, fontweight='bold')
    
    strategies = [
        ('保守策略', 0.70, '拒绝75%的错误预测'),
        ('平衡策略', 0.80, '拒绝50%的错误预测'),
        ('激进策略', 0.90, '拒绝25%的错误预测'),
    ]
    
    y_pos = 0.8
    for name, threshold, desc in strategies:
        idx = np.argmin(np.abs(np.array(results['threshold']) - threshold))
        reject_rate = results['reject_rate'][idx]
        error_reduction = results['precision_gain'][idx]
        
        text = f"{name}（阈值 = {threshold:.2f}）\n"
        text += f"  • {desc}\n"
        text += f"  • 拒绝率: {reject_rate:.1f}%\n"
        text += f"  • 精度提升: {error_reduction:.1f}%\n"
        
        color = 'orange' if '保守' in name else ('green' if '平衡' in name else 'blue')
        ax4.text(0.1, y_pos, text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                verticalalignment='top')
        y_pos -= 0.25
    
    plt.tight_layout()
    plt.savefig('docs/BSTGCN/threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 阈值分析图已保存到: docs/BSTGCN/threshold_analysis.png")
    plt.show()


def print_recommendations():
    """打印推荐策略"""
    
    print("\n" + "=" * 80)
    print("不确定性阈值推荐")
    print("=" * 80)
    
    print("\n基于你的箱型图结果：")
    print("  • Correct中位数: 0.50")
    print("  • Correct Q3: 0.65")
    print("  • Error Q1: 0.70")
    print("  • Error中位数: 0.80")
    print("  • Error Q3: 0.90")
    
    print("\n" + "-" * 80)
    print("策略1：保守策略（高精度，低召回）")
    print("-" * 80)
    print("  阈值: 0.70（Error的Q1）")
    print("  效果:")
    print("    • 拒绝75%的错误预测")
    print("    • 保留约90%的正确预测")
    print("    • 总拒绝率: ~15%")
    print("  适用场景:")
    print("    • 错误代价高（如医疗诊断、金融欺诈）")
    print("    • 可以接受较高拒绝率")
    print("    • 需要人工审核被拒绝的样本")
    
    print("\n" + "-" * 80)
    print("策略2：平衡策略（推荐）")
    print("-" * 80)
    print("  阈值: 0.80（Error的中位数）")
    print("  效果:")
    print("    • 拒绝50%的错误预测")
    print("    • 保留约95%的正确预测")
    print("    • 总拒绝率: ~8%")
    print("  适用场景:")
    print("    • 平衡精度和召回")
    print("    • 大多数实际应用")
    print("    • 网络流量分类")
    
    print("\n" + "-" * 80)
    print("策略3：激进策略（高召回，中等精度）")
    print("-" * 80)
    print("  阈值: 0.90（Error的Q3）")
    print("  效果:")
    print("    • 拒绝25%的错误预测")
    print("    • 保留约98%的正确预测")
    print("    • 总拒绝率: ~3%")
    print("  适用场景:")
    print("    • 需要高召回率")
    print("    • 拒绝代价高")
    print("    • 实时系统")
    
    print("\n" + "=" * 80)
    print("实现示例")
    print("=" * 80)
    print("""
# 在推理时应用阈值
def predict_with_uncertainty(model, data, threshold=0.80):
    predictions, uncertainties = model.predict(data)
    
    # 标记高不确定性样本
    reliable_mask = uncertainties <= threshold
    
    # 返回可靠预测和需要审核的样本
    reliable_preds = predictions[reliable_mask]
    review_samples = data[~reliable_mask]
    
    return reliable_preds, review_samples
    """)
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_recommendations()
    results = simulate_threshold_analysis()
    plot_threshold_analysis(results)

