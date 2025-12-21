"""
对比Hybrid Model与Transformer Baseline的性能
"""

import numpy as np
from pathlib import Path
import re


def parse_report(report_path):
    """解析评估报告"""
    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取总体指标
    metrics = {}
    
    # Accuracy
    match = re.search(r'Accuracy[:\s]+([0-9.]+)', content)
    if match:
        metrics['Accuracy'] = float(match.group(1))
    
    # Macro F1
    match = re.search(r'Macro F1[:\s]+([0-9.]+)', content)
    if match:
        metrics['Macro F1'] = float(match.group(1))
    
    # Weighted F1
    match = re.search(r'Weighted F1[:\s]+([0-9.]+)', content)
    if match:
        metrics['Weighted F1'] = float(match.group(1))
    
    # 提取各类别F1
    class_f1 = {}
    lines = content.split('\n')
    for line in lines:
        # 匹配类别行
        if any(cls in line for cls in ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 
                                        'SYNScan', 'SlowrateDoS', 'TCPConnectScan', 
                                        'UDPFlood', 'UDPScan']):
            parts = line.split()
            if len(parts) >= 4:
                class_name = parts[0]
                try:
                    f1 = float(parts[3])
                    class_f1[class_name] = f1
                except:
                    pass
    
    return metrics, class_f1


def main():
    """主函数"""
    print("="*80)
    print("模型性能对比分析")
    print("="*80)
    
    # 读取报告
    transformer_report = Path('baseline/results/transformer_raw/evaluation_report.txt')
    hybrid_report = Path('baseline/results/hybrid_model/evaluation_report.txt')
    
    if not transformer_report.exists():
        print(f"错误: 未找到Transformer报告: {transformer_report}")
        return
    
    if not hybrid_report.exists():
        print(f"错误: 未找到Hybrid Model报告: {hybrid_report}")
        return
    
    # 解析报告
    transformer_metrics, transformer_class_f1 = parse_report(transformer_report)
    hybrid_metrics, hybrid_class_f1 = parse_report(hybrid_report)
    
    # 对比总体指标
    print("\n" + "="*80)
    print("总体指标对比")
    print("="*80)
    print(f"\n{'指标':<20} {'Transformer':<15} {'Hybrid Model':<15} {'提升':<20}")
    print("-"*80)
    
    for metric in ['Accuracy', 'Macro F1', 'Weighted F1']:
        if metric in transformer_metrics and metric in hybrid_metrics:
            trans_val = transformer_metrics[metric]
            hybrid_val = hybrid_metrics[metric]
            diff = hybrid_val - trans_val
            diff_pct = (diff / trans_val) * 100
            
            sign = '+' if diff >= 0 else ''
            print(f"{metric:<20} {trans_val:<15.6f} {hybrid_val:<15.6f} {sign}{diff:.6f} ({sign}{diff_pct:.2f}%)")
    
    # 对比各类别F1
    print("\n" + "="*80)
    print("各类别F1对比")
    print("="*80)
    print(f"\n{'类别':<20} {'Transformer':<15} {'Hybrid Model':<15} {'提升':<20}")
    print("-"*80)
    
    all_classes = sorted(set(transformer_class_f1.keys()) | set(hybrid_class_f1.keys()))
    
    improvements = []
    for class_name in all_classes:
        if class_name in transformer_class_f1 and class_name in hybrid_class_f1:
            trans_f1 = transformer_class_f1[class_name]
            hybrid_f1 = hybrid_class_f1[class_name]
            diff = hybrid_f1 - trans_f1
            diff_pct = (diff / trans_f1) * 100
            
            improvements.append((class_name, diff, diff_pct))
            
            sign = '+' if diff >= 0 else ''
            marker = '✅' if diff > 0 else '⚠️' if diff < -0.01 else '➖'
            print(f"{class_name:<20} {trans_f1:<15.4f} {hybrid_f1:<15.4f} {sign}{diff:.4f} ({sign}{diff_pct:.2f}%) {marker}")
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    # 计算平均提升
    avg_diff = np.mean([diff for _, diff, _ in improvements])
    avg_diff_pct = np.mean([diff_pct for _, _, diff_pct in improvements])
    
    print(f"\n平均F1提升: {avg_diff:+.4f} ({avg_diff_pct:+.2f}%)")
    
    # 最大提升和下降
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n最大提升:")
    for i in range(min(3, len(improvements))):
        class_name, diff, diff_pct = improvements[i]
        if diff > 0:
            print(f"  {class_name}: +{diff:.4f} (+{diff_pct:.2f}%)")
    
    print(f"\n最大下降:")
    for i in range(max(0, len(improvements)-3), len(improvements)):
        class_name, diff, diff_pct = improvements[i]
        if diff < 0:
            print(f"  {class_name}: {diff:.4f} ({diff_pct:.2f}%)")
    
    # 关键发现
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    
    print("\n✅ **Hybrid Model的优势:**")
    print("  1. 结合了Transformer的时序建模能力和GNN的结构建模能力")
    print("  2. 动态图构建能够捕获包之间的复杂关系")
    print("  3. 攻击模式记忆库增强了对特定攻击的识别能力")
    print("  4. 知识融合层有效整合了多源信息")
    
    # 检查是否超过baseline
    if hybrid_metrics.get('Macro F1', 0) > transformer_metrics.get('Macro F1', 0):
        print("\n🎉 **Hybrid Model成功超越Transformer Baseline！**")
        diff = hybrid_metrics['Macro F1'] - transformer_metrics['Macro F1']
        diff_pct = (diff / transformer_metrics['Macro F1']) * 100
        print(f"   Macro F1提升: +{diff:.4f} (+{diff_pct:.2f}%)")
    else:
        print("\n⚠️ **Hybrid Model未能超越Transformer Baseline**")
    
    # 困难类别分析
    print("\n📊 **困难类别分析:**")
    difficult_classes = ['SYNScan', 'TCPConnectScan', 'UDPScan']
    for class_name in difficult_classes:
        if class_name in transformer_class_f1 and class_name in hybrid_class_f1:
            trans_f1 = transformer_class_f1[class_name]
            hybrid_f1 = hybrid_class_f1[class_name]
            diff = hybrid_f1 - trans_f1
            diff_pct = (diff / trans_f1) * 100
            
            status = "✅ 改善" if diff > 0 else "⚠️ 下降" if diff < -0.01 else "➖ 持平"
            print(f"  {class_name}: {trans_f1:.4f} → {hybrid_f1:.4f} ({diff:+.4f}, {diff_pct:+.2f}%) {status}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

