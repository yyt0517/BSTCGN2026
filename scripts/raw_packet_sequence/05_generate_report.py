"""
生成原始包序列模型与基线模型的对比报告
"""
import os
import numpy as np
import pandas as pd

print("="*100)
print("Generate Comparison Report".center(100))
print("="*100)

def parse_evaluation_report(report_path):
    """解析评估报告，提取关键指标"""
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(report_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            return None
    
    metrics = {}
    
    # Extract accuracy
    if 'Accuracy:' in content:
        for line in content.split('\n'):
            if 'Accuracy:' in line:
                metrics['accuracy'] = float(line.split(':')[1].strip())
                break
    
    # Extract Macro F1
    if 'Macro F1:' in content or 'macro avg' in content:
        for line in content.split('\n'):
            if 'Macro F1:' in line:
                metrics['macro_f1'] = float(line.split(':')[1].strip())
                break
            elif 'macro avg' in line:
                parts = line.split()
                metrics['macro_f1'] = float(parts[-2])
                break
    
    # Extract Weighted F1
    if 'Weighted F1:' in content or 'weighted avg' in content:
        for line in content.split('\n'):
            if 'Weighted F1:' in line:
                metrics['weighted_f1'] = float(line.split(':')[1].strip())
                break
            elif 'weighted avg' in line:
                parts = line.split()
                metrics['weighted_f1'] = float(parts[-2])
                break
    
    return metrics

def collect_baseline_results():
    """收集所有基线模型的结果"""
    results = {}
    
    baseline_models = {
        'LightGBM': 'baseline/results/lightgbm/evaluation_report.txt',
        'XGBoost': 'baseline/results/xgboost/evaluation_report.txt',
        'LSTM': 'baseline/results/lstm/evaluation_report.txt',
        'GRU': 'baseline/results/gru/evaluation_report.txt',
        'BiLSTM': 'baseline/results/bilstm/evaluation_report.txt',
        'CNN1D': 'baseline/results/cnn1d/evaluation_report.txt',
        'STGNN (基础)': 'baseline/results/stgnn/evaluation_report.txt',
        'STGNN (增强)': 'baseline/results/stgnn_enhanced/evaluation_report.txt',
        'Random Forest': 'baseline/results/random_forest_packet/evaluation_report.txt',
    }
    
    for model_name, report_path in baseline_models.items():
        metrics = parse_evaluation_report(report_path)
        if metrics:
            results[model_name] = metrics
    
    return results

def collect_transformer_results():
    """收集Transformer模型的结果"""
    results = {}
    
    transformer_models = {
        'Transformer (原始包序列)': 'baseline/results/transformer_raw/evaluation_report.txt',
    }
    
    for model_name, report_path in transformer_models.items():
        metrics = parse_evaluation_report(report_path)
        if metrics:
            results[model_name] = metrics
    
    return results

def generate_comparison_report(output_path='baseline/results/RAW_PACKET_COMPARISON.txt'):
    """生成对比报告"""
    
    print("\n收集基线模型结果...")
    baseline_results = collect_baseline_results()
    print(f"  找到 {len(baseline_results)} 个基线模型")
    
    print("\n收集Transformer模型结果...")
    transformer_results = collect_transformer_results()
    print(f"  找到 {len(transformer_results)} 个Transformer模型")
    
    # Combine results
    all_results = {**baseline_results, **transformer_results}
    
    if len(all_results) == 0:
        print("\n✗ 没有找到任何模型结果！")
        return
    
    # Sort by Macro F1
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1].get('macro_f1', 0), 
                           reverse=True)
    
    # Generate report
    print(f"\n生成对比报告: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("原始包序列模型 vs 聚合特征模型 性能对比\n")
        f.write("="*100 + "\n\n")
        
        f.write("数据类型对比:\n")
        f.write("  聚合特征模型: 使用1秒窗口聚合后的统计特征 (avg, std, count等)\n")
        f.write("  原始包序列模型: 使用原始包级别的特征 (每个包的packet_length, ttl, flags等)\n\n")
        
        f.write("="*100 + "\n")
        f.write("性能排名\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'排名':<6}{'模型':<30}{'Macro F1':<12}{'Weighted F1':<14}{'准确率':<12}{'数据类型':<20}\n")
        f.write("-"*100 + "\n")
        
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            macro_f1 = metrics.get('macro_f1', 0)
            weighted_f1 = metrics.get('weighted_f1', 0)
            accuracy = metrics.get('accuracy', 0)
            
            # Determine data type
            if '原始包序列' in model_name or 'Transformer' in model_name:
                data_type = '原始包序列'
            else:
                data_type = '聚合特征'
            
            medal = ''
            if rank == 1:
                medal = '🥇'
            elif rank == 2:
                medal = '🥈'
            elif rank == 3:
                medal = '🥉'
            
            f.write(f"{rank:<6}{model_name:<30}{macro_f1:<12.4f}{weighted_f1:<14.4f}"
                   f"{accuracy:<12.4f}{data_type:<20}{medal}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("关键发现\n")
        f.write("="*100 + "\n\n")
        
        # Find best models
        best_model = sorted_results[0]
        best_aggregated = None
        best_raw = None
        
        for model_name, metrics in sorted_results:
            if '原始包序列' in model_name or 'Transformer' in model_name:
                if best_raw is None:
                    best_raw = (model_name, metrics)
            else:
                if best_aggregated is None:
                    best_aggregated = (model_name, metrics)
        
        f.write(f"1. 最佳模型: {best_model[0]}\n")
        f.write(f"   - Macro F1: {best_model[1].get('macro_f1', 0):.4f}\n")
        f.write(f"   - 准确率: {best_model[1].get('accuracy', 0):.4f}\n\n")
        
        if best_aggregated:
            f.write(f"2. 最佳聚合特征模型: {best_aggregated[0]}\n")
            f.write(f"   - Macro F1: {best_aggregated[1].get('macro_f1', 0):.4f}\n")
            f.write(f"   - 准确率: {best_aggregated[1].get('accuracy', 0):.4f}\n\n")
        
        if best_raw:
            f.write(f"3. 最佳原始包序列模型: {best_raw[0]}\n")
            f.write(f"   - Macro F1: {best_raw[1].get('macro_f1', 0):.4f}\n")
            f.write(f"   - 准确率: {best_raw[1].get('accuracy', 0):.4f}\n\n")
            
            if best_aggregated:
                diff = best_raw[1].get('macro_f1', 0) - best_aggregated[1].get('macro_f1', 0)
                f.write(f"4. 原始包序列 vs 聚合特征:\n")
                f.write(f"   - Macro F1差距: {diff:+.4f} ({diff/best_aggregated[1].get('macro_f1', 1)*100:+.2f}%)\n")
                
                if diff > 0:
                    f.write(f"   - ✓ 原始包序列模型更好！\n")
                else:
                    f.write(f"   - ✗ 聚合特征模型更好\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("结论\n")
        f.write("="*100 + "\n\n")
        
        if best_raw and best_aggregated:
            diff = best_raw[1].get('macro_f1', 0) - best_aggregated[1].get('macro_f1', 0)
            
            if diff > 0.05:  # 5% improvement
                f.write("✓ 原始包序列模型显著优于聚合特征模型！\n\n")
                f.write("原因:\n")
                f.write("  - 原始包序列保留了细粒度的时序信息\n")
                f.write("  - Transformer能够捕捉包之间的依赖关系\n")
                f.write("  - 没有信息损失（聚合会丢失细节）\n\n")
                f.write("推荐: 使用原始包序列 + Transformer模型\n")
            elif diff > 0:
                f.write("✓ 原始包序列模型略优于聚合特征模型\n\n")
                f.write("原因:\n")
                f.write("  - 原始包序列保留了更多信息\n")
                f.write("  - 但提升不明显，可能是因为:\n")
                f.write("    * 模型容量不足\n")
                f.write("    * 训练数据不足\n")
                f.write("    * 超参数未优化\n\n")
                f.write("推荐: 继续优化原始包序列模型，或使用聚合特征模型（更简单）\n")
            else:
                f.write("✗ 聚合特征模型优于原始包序列模型\n\n")
                f.write("原因:\n")
                f.write("  - 聚合特征已经包含了足够的信息\n")
                f.write("  - 树模型（LightGBM/XGBoost）更擅长处理聚合特征\n")
                f.write("  - 原始包序列模型可能:\n")
                f.write("    * 过拟合\n")
                f.write("    * 训练不足\n")
                f.write("    * 模型架构不适合\n\n")
                f.write("推荐: 使用聚合特征 + LightGBM模型（性能最佳，训练最快）\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("报告生成时间: 2025-11-10\n")
        f.write("="*100 + "\n")
    
    print(f"✓ 报告已保存: {output_path}")
    
    # Print summary
    print("\n" + "="*100)
    print("对比结果总结".center(100))
    print("="*100)
    
    if best_raw and best_aggregated:
        print(f"\n最佳聚合特征模型: {best_aggregated[0]}")
        print(f"  Macro F1: {best_aggregated[1].get('macro_f1', 0):.4f}")
        
        print(f"\n最佳原始包序列模型: {best_raw[0]}")
        print(f"  Macro F1: {best_raw[1].get('macro_f1', 0):.4f}")
        
        diff = best_raw[1].get('macro_f1', 0) - best_aggregated[1].get('macro_f1', 0)
        print(f"\n性能差距: {diff:+.4f} ({diff/best_aggregated[1].get('macro_f1', 1)*100:+.2f}%)")
        
        if diff > 0:
            print("✓ 原始包序列模型更好！")
        else:
            print("✗ 聚合特征模型更好")

if __name__ == "__main__":
    generate_comparison_report()
    
    print("\n" + "="*100)
    print("Report Generation Complete!".center(100))
    print("="*100)

