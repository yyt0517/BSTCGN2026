"""
对比不同运行的BSTGCN结果

使用方法：
    python scripts/BSTGCN/compare_runs.py
"""

import json
from pathlib import Path
from datetime import datetime

def compare_runs(base_dir='baseline/results/bstgcn_50packets'):
    """对比所有运行的结果"""
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"目录不存在: {base_dir}")
        return
    
    # 收集所有运行的结果
    runs = []
    
    for run_dir in sorted(base_path.iterdir()):
        if not run_dir.is_dir():
            continue
            
        # 跳过非时间戳目录
        if not run_dir.name.replace('_', '').isdigit():
            continue
        
        results_file = run_dir / 'test_results.json'
        config_file = run_dir / 'config.json'
        
        if results_file.exists() and config_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            with open(config_file) as f:
                config = json.load(f)
            
            runs.append({
                'timestamp': run_dir.name,
                'accuracy': results.get('test_accuracy', 0),
                'f1': results.get('test_macro_f1', 0),
                'uncertainty': results.get('test_mean_uncertainty', 0),
                'unc_corr': results.get('uncertainty_error_correlation', 0),
                'best_epoch': results.get('best_epoch', 0),
                'best_val_f1': results.get('best_val_f1', 0),
                'patience': config.get('patience', 0),
                'learning_rate': config.get('learning_rate', 0),
            })
    
    if not runs:
        print("没有找到任何运行结果")
        return
    
    # 打印对比表格
    print("=" * 120)
    print("BSTGCN 运行结果对比")
    print("=" * 120)
    print(f"{'时间戳':<17} {'准确率':>8} {'F1':>8} {'不确定性':>10} {'不确定性相关':>12} "
          f"{'最佳轮次':>8} {'最佳Val F1':>10} {'Patience':>8} {'学习率':>10}")
    print("-" * 120)
    
    for run in runs:
        print(f"{run['timestamp']:<17} "
              f"{run['accuracy']:>8.4f} "
              f"{run['f1']:>8.4f} "
              f"{run['uncertainty']:>10.4f} "
              f"{run['unc_corr']:>12.4f} "
              f"{run['best_epoch']:>8d} "
              f"{run['best_val_f1']:>10.4f} "
              f"{run['patience']:>8d} "
              f"{run['learning_rate']:>10.6f}")
    
    print("=" * 120)
    
    # 找出最佳运行
    best_f1_run = max(runs, key=lambda x: x['f1'])
    best_acc_run = max(runs, key=lambda x: x['accuracy'])
    
    print("\n最佳结果:")
    print(f"  最高F1: {best_f1_run['f1']:.4f} (时间戳: {best_f1_run['timestamp']})")
    print(f"  最高准确率: {best_acc_run['accuracy']:.4f} (时间戳: {best_acc_run['timestamp']})")
    
    # 统计信息
    if len(runs) > 1:
        avg_f1 = sum(r['f1'] for r in runs) / len(runs)
        std_f1 = (sum((r['f1'] - avg_f1) ** 2 for r in runs) / len(runs)) ** 0.5
        
        print(f"\n统计信息 (共{len(runs)}次运行):")
        print(f"  平均F1: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"  F1范围: [{min(r['f1'] for r in runs):.4f}, {max(r['f1'] for r in runs):.4f}]")


if __name__ == '__main__':
    compare_runs()

