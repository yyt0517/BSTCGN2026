"""
计算 Selective Prediction 指标：
- Selective Accuracy @ 80% Coverage
- AURC (Area Under Risk-Coverage Curve)
"""

import numpy as np
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

DATA_DIR = Path("data/processed/raw_packet_sequences_50packets")
BSTGCN_DIR = Path("baseline/results/bstgcn_50packets/20251205_141432")
CALIB_DIR = Path("baseline/results/calibration_analysis_50packets")


def compute_selective_metrics(y_true, y_pred, uncertainties):
    """计算 Selective Prediction 指标"""
    # 按不确定性排序（低到高）
    sorted_idx = np.argsort(uncertainties)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]
    
    n = len(y_true)
    coverages = np.arange(1, n + 1) / n
    
    # 计算累积准确率
    correct = (y_true_sorted == y_pred_sorted).astype(float)
    cumsum_correct = np.cumsum(correct)
    cumsum_total = np.arange(1, n + 1)
    selective_acc = cumsum_correct / cumsum_total
    
    # 计算 risk（1 - accuracy）
    risks = 1 - selective_acc
    
    # AURC: Area Under Risk-Coverage Curve
    aurc = np.trapz(risks, coverages)
    
    # Selective Accuracy @ 80% coverage
    idx_80 = int(0.8 * n)
    sel_acc_80 = selective_acc[idx_80 - 1]
    
    # Selective Accuracy @ 90% coverage  
    idx_90 = int(0.9 * n)
    sel_acc_90 = selective_acc[idx_90 - 1]
    
    return {
        'aurc': aurc,
        'sel_acc_80': sel_acc_80,
        'sel_acc_90': sel_acc_90
    }


def main():
    test_labels = np.load(DATA_DIR / "test_labels.npy")
    print(f"Test samples: {len(test_labels)}")
    
    results = []
    
    # 1. BSTGCN (Bayesian - 认知不确定性)
    print("\n=== BSTGCN (Bayesian) ===")
    bstgcn_preds = np.load(BSTGCN_DIR / "test_predictions.npy")
    bstgcn_uncs = np.load(BSTGCN_DIR / "test_uncertainties.npy")
    
    metrics = compute_selective_metrics(test_labels, bstgcn_preds, bstgcn_uncs)
    print(f"  AURC: {metrics['aurc']:.4f}")
    print(f"  Sel Acc @80%: {metrics['sel_acc_80']:.4f}")
    print(f"  Sel Acc @90%: {metrics['sel_acc_90']:.4f}")
    results.append({"model": "BSTGCN", "method": "Bayesian", **metrics})
    
    # 2. BSTGCN 使用预测熵（总不确定性）
    print("\n=== BSTGCN (Predictive Entropy) ===")
    bstgcn_probs = np.load(CALIB_DIR / "bstgcn_probs.npy")
    epsilon = 1e-10
    bstgcn_entropy = -np.sum(bstgcn_probs * np.log(bstgcn_probs + epsilon), axis=1)
    
    metrics = compute_selective_metrics(test_labels, bstgcn_preds, bstgcn_entropy)
    print(f"  AURC: {metrics['aurc']:.4f}")
    print(f"  Sel Acc @80%: {metrics['sel_acc_80']:.4f}")
    print(f"  Sel Acc @90%: {metrics['sel_acc_90']:.4f}")
    results.append({"model": "BSTGCN", "method": "Entropy", **metrics})
    
    # 3. MC Dropout 基线
    mc_results_path = Path("baseline/results/mc_dropout_results.json")
    if mc_results_path.exists():
        with open(mc_results_path) as f:
            mc_data = json.load(f)
        
        for item in mc_data:
            model_name = item['model'].replace(" (MC Dropout)", "")
            print(f"\n=== {model_name} (MC Dropout) ===")
            
            # 加载预测和不确定性
            unc_path = Path(f"baseline/results/mc_dropout_{model_name.lower()}_uncertainties.npy")
            pred_path = Path(f"baseline/results/mc_dropout_{model_name.lower()}_predictions.npy")
            
            if unc_path.exists() and pred_path.exists():
                mc_uncs = np.load(unc_path)
                mc_preds = np.load(pred_path)
                
                metrics = compute_selective_metrics(test_labels, mc_preds, mc_uncs)
                print(f"  AURC: {metrics['aurc']:.4f}")
                print(f"  Sel Acc @80%: {metrics['sel_acc_80']:.4f}")
                print(f"  Sel Acc @90%: {metrics['sel_acc_90']:.4f}")
                results.append({"model": model_name, "method": "MC Dropout", **metrics})
            else:
                print(f"  Files not found, skipping...")
    
    # 4. 确定性模型（使用预测熵）
    for model_name in ["stgcn", "gat", "cnn1d"]:
        probs_path = CALIB_DIR / f"{model_name}_probs.npy"
        if probs_path.exists():
            print(f"\n=== {model_name.upper()} (Entropy) ===")
            probs = np.load(probs_path)
            preds = np.argmax(probs, axis=1)
            entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
            
            metrics = compute_selective_metrics(test_labels, preds, entropy)
            print(f"  AURC: {metrics['aurc']:.4f}")
            print(f"  Sel Acc @80%: {metrics['sel_acc_80']:.4f}")
            print(f"  Sel Acc @90%: {metrics['sel_acc_90']:.4f}")
            results.append({"model": model_name.upper(), "method": "Entropy", **metrics})
    
    # 保存结果
    with open("baseline/results/selective_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"{'Model':<20} {'Method':<12} {'AURC↓':<10} {'SelAcc@80%↑':<12} {'SelAcc@90%↑':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['model']:<20} {r['method']:<12} {r['aurc']:.4f}     {r['sel_acc_80']:.4f}        {r['sel_acc_90']:.4f}")


if __name__ == "__main__":
    main()

