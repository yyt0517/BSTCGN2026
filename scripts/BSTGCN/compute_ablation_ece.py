"""
计算消融实验的ECE指标，以及完整的不确定性指标表格
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

# 路径配置
DATA_DIR = Path("data/processed/raw_packet_sequences_50packets")
BSTGCN_DIR = Path("baseline/results/bstgcn_50packets/20251205_141432")
STGCN_DIR = Path("baseline/results/hybrid_model_improved_50packets")
SEQ_STGCN_DIR = Path("baseline/results_50packets/seq_stgcn")


def compute_ece(y_true, y_probs, n_bins=10):
    """计算ECE"""
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_weight = np.sum(mask) / len(y_true)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    
    return ece


def load_test_labels():
    """加载测试标签"""
    return np.load(DATA_DIR / "test_labels.npy")


def main():
    test_labels = load_test_labels()
    print(f"Test samples: {len(test_labels)}")
    
    results = {}
    
    # 1. BSTGCN
    print("\n=== BSTGCN ===")
    bstgcn_preds = np.load(BSTGCN_DIR / "test_predictions.npy")
    bstgcn_uncs = np.load(BSTGCN_DIR / "test_uncertainties.npy")
    bstgcn_probs = np.load(Path("baseline/results/calibration_analysis_50packets/bstgcn_probs.npy"))
    
    errors = (bstgcn_preds != test_labels).astype(float)
    unc_err_corr = np.corrcoef(errors, bstgcn_uncs)[0, 1]
    mean_unc = np.mean(bstgcn_uncs)
    ece = compute_ece(test_labels, bstgcn_probs)
    
    print(f"  Unc-Err Corr: {unc_err_corr:.4f}")
    print(f"  Mean Uncertainty: {mean_unc:.4f}")
    print(f"  ECE: {ece:.4f}")
    results["BSTGCN"] = {"unc_err_corr": float(unc_err_corr), "mean_unc": float(mean_unc), "ece": float(ece)}
    
    # 2. STGCN (Deterministic) - 使用预测熵作为不确定性代理
    print("\n=== STGCN (Det.) ===")
    stgcn_probs = np.load(Path("baseline/results/calibration_analysis_50packets/stgcn_probs.npy"))
    stgcn_preds = np.argmax(stgcn_probs, axis=1)
    epsilon = 1e-10
    # 预测熵（总不确定性）
    stgcn_entropy = -np.sum(stgcn_probs * np.log(stgcn_probs + epsilon), axis=1)

    errors = (stgcn_preds != test_labels).astype(float)
    unc_err_corr_entropy = np.corrcoef(errors, stgcn_entropy)[0, 1]
    mean_entropy = np.mean(stgcn_entropy)
    ece = compute_ece(test_labels, stgcn_probs)

    print(f"  Unc-Err Corr (Total, Entropy): {unc_err_corr_entropy:.4f}")
    print(f"  Mean Entropy: {mean_entropy:.4f}")
    print(f"  ECE: {ece:.4f}")
    results["STGCN"] = {
        "unc_err_corr_entropy": float(unc_err_corr_entropy),
        "mean_entropy": float(mean_entropy),
        "ece": float(ece)
    }
    
    # 3. Seq-STGCN (从classification_report提取)
    print("\n=== Seq-STGCN (Det.) ===")
    # 尝试加载Seq-STGCN的概率输出
    seq_stgcn_probs_path = SEQ_STGCN_DIR / "test_probs.npy"
    if seq_stgcn_probs_path.exists():
        seq_stgcn_probs = np.load(seq_stgcn_probs_path)
    else:
        # 从calibration analysis加载
        seq_stgcn_probs_path2 = Path("baseline/results_50packets/seq_stgcn/test_probs.npy")
        if seq_stgcn_probs_path2.exists():
            seq_stgcn_probs = np.load(seq_stgcn_probs_path2)
        else:
            print("  Seq-STGCN probs not found, using estimated values")
            seq_stgcn_probs = None
    
    if seq_stgcn_probs is not None:
        seq_stgcn_preds = np.argmax(seq_stgcn_probs, axis=1)
        seq_stgcn_entropy = -np.sum(seq_stgcn_probs * np.log(seq_stgcn_probs + epsilon), axis=1)
        
        errors = (seq_stgcn_preds != test_labels).astype(float)
        unc_err_corr = np.corrcoef(errors, seq_stgcn_entropy)[0, 1]
        mean_unc = np.mean(seq_stgcn_entropy)
        ece = compute_ece(test_labels, seq_stgcn_probs)
        
        print(f"  Unc-Err Corr (entropy): {unc_err_corr:.4f}")
        print(f"  Mean Entropy: {mean_unc:.4f}")
        print(f"  ECE: {ece:.4f}")
        results["Seq-STGCN"] = {"unc_err_corr": float(unc_err_corr), "mean_unc": float(mean_unc), "ece": float(ece)}
    
    # 4. MC Dropout baselines
    print("\n=== MC Dropout Results ===")
    mc_results = json.load(open("baseline/results/mc_dropout_results.json"))
    for r in mc_results:
        print(f"  {r['model']}: Unc-Err Corr={r['unc_err_corr']:.4f}, ECE={r['ece']:.4f}")
    
    # 保存完整结果
    output = {
        "uncertainty_table": results,
        "mc_dropout": mc_results
    }
    
    with open("baseline/results/uncertainty_analysis_full.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n=== Summary for Table V ===")
    print("Model                    | Unc-Err Corr | Mean Unc  | ECE")
    print("-" * 60)
    print(f"CNN1D (MC Dropout)       | {mc_results[0]['unc_err_corr']:.4f}       | {mc_results[0]['mean_uncertainty']:.4f}    | {mc_results[0]['ece']:.4f}")
    print(f"GRU (MC Dropout)         | {mc_results[1]['unc_err_corr']:.4f}       | {mc_results[1]['mean_uncertainty']:.4f}    | {mc_results[1]['ece']:.4f}")
    print(f"GAT (MC Dropout)         | {mc_results[2]['unc_err_corr']:.4f}       | {mc_results[2]['mean_uncertainty']:.4f}    | {mc_results[2]['ece']:.4f}")
    if "STGCN" in results:
        print(f"STGCN (Entropy)          | {results['STGCN']['unc_err_corr']:.4f}       | {results['STGCN']['mean_unc']:.4f}    | {results['STGCN']['ece']:.4f}")
    if "Seq-STGCN" in results:
        print(f"Seq-STGCN (Entropy)      | {results['Seq-STGCN']['unc_err_corr']:.4f}       | {results['Seq-STGCN']['mean_unc']:.4f}    | {results['Seq-STGCN']['ece']:.4f}")
    print(f"BSTGCN (Ours)            | {results['BSTGCN']['unc_err_corr']:.4f}       | {results['BSTGCN']['mean_unc']:.4f}    | {results['BSTGCN']['ece']:.4f}")


if __name__ == "__main__":
    main()

