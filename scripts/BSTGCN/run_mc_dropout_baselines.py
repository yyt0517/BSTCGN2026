"""
运行MC Dropout基线实验，计算不确定性指标
用于扩充 Table V (Uncertainty Quantification Quality)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import f1_score

# 数据路径
DATA_DIR = Path("data/processed/raw_packet_sequences_50packets")
RESULTS_DIR = Path("baseline/results_50packets")
CALIB_DIR = Path("baseline/results/calibration_analysis_50packets")
OUTPUT_FILE = Path("baseline/results/mc_dropout_results.json")
SOFTMAX_OUTPUT_FILE = Path("baseline/results/softmax_calibration_results.json")

# 模型定义（与baseline相同）
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from baseline.run_comprehensive_baselines import CNN1DClassifier, GRUClassifier, GATClassifier, STGCNClassifier


def enable_dropout(model):
    """在推理时启用dropout"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_inference(model, data, num_samples=30):
    """MC Dropout推理"""
    model.eval()
    enable_dropout(model)
    
    probs_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(data)
            probs = F.softmax(output, dim=-1)
            probs_list.append(probs)
    
    probs_stack = torch.stack(probs_list, dim=0)  # (M, B, C)
    pred_mean = probs_stack.mean(dim=0)  # (B, C)
    pred_class = pred_mean.argmax(dim=-1)  # (B,)
    
    # 计算认知不确定性 (互信息)
    epsilon = 1e-10
    entropy_mean = -torch.sum(pred_mean * torch.log(pred_mean + epsilon), dim=-1)
    entropy_samples = -torch.sum(probs_stack * torch.log(probs_stack + epsilon), dim=-1)
    mean_entropy = entropy_samples.mean(dim=0)
    epistemic_uncertainty = entropy_mean - mean_entropy
    
    return pred_mean, pred_class, epistemic_uncertainty


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


def compute_brier_score(y_true, y_probs, num_classes=9):
    """计算Brier Score"""
    y_onehot = np.eye(num_classes)[y_true]
    return np.mean(np.sum((y_probs - y_onehot) ** 2, axis=1))


def compute_nll(y_true, y_probs):
    """计算Negative Log-Likelihood"""
    epsilon = 1e-10
    probs_clipped = np.clip(y_probs, epsilon, 1 - epsilon)
    correct_probs = probs_clipped[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(correct_probs))


def run_mc_dropout_experiment(model_name, model_path, model_class, device):
    """运行单个模型的MC Dropout实验"""
    print(f"\n{'='*50}")
    print(f"Running MC Dropout for {model_name}")
    print(f"{'='*50}")

    # 加载数据
    test_data = torch.tensor(np.load(DATA_DIR / "test_sequences.npy"), dtype=torch.float32)
    test_labels = np.load(DATA_DIR / "test_labels.npy")

    # 加载模型
    if model_name == "CNN1D":
        model = model_class(input_size=25, num_classes=9, dropout=0.3)
    elif model_name == "GRU":
        model = model_class(input_size=25, hidden_size=128, num_classes=9, dropout=0.3)
    elif model_name == "GAT":
        model = model_class(input_size=25, hidden_dim=128, num_heads=4, num_classes=9, dropout=0.2)
    elif model_name == "STGCN":
        model = model_class(input_size=25, num_classes=9, dropout=0.2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # 批量MC Dropout推理
    batch_size = 512
    all_probs, all_preds, all_uncs = [], [], []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size].to(device)
        probs, preds, uncs = mc_dropout_inference(model, batch, num_samples=30)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_uncs.append(uncs.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_uncs = np.concatenate(all_uncs)
    
    # 计算指标
    errors = (all_preds != test_labels).astype(float)
    unc_err_corr = np.corrcoef(errors, all_uncs)[0, 1]
    mean_unc = np.mean(all_uncs)
    ece = compute_ece(test_labels, all_probs)
    brier = compute_brier_score(test_labels, all_probs)
    nll = compute_nll(test_labels, all_probs)
    macro_f1 = f1_score(test_labels, all_preds, average='macro')
    accuracy = np.mean(all_preds == test_labels)

    results = {
        "model": f"{model_name} (MC Dropout)",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "unc_err_corr": float(unc_err_corr),
        "mean_uncertainty": float(mean_unc),
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll)
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier: {brier:.4f}")
    print(f"  NLL: {nll:.4f}")
    
    return results


def softmax_inference(model, data, device):
    """标准Softmax推理（无MC采样）"""
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        probs = F.softmax(output, dim=-1)
        pred_class = probs.argmax(dim=-1)
    return probs.cpu().numpy(), pred_class.cpu().numpy()


def run_softmax_experiment(model_name, model_path, model_class, device):
    """运行Softmax校准实验"""
    print(f"\n{'='*50}")
    print(f"Running Softmax for {model_name}")
    print(f"{'='*50}")

    # 加载数据
    test_data = torch.tensor(np.load(DATA_DIR / "test_sequences.npy"), dtype=torch.float32)
    test_labels = np.load(DATA_DIR / "test_labels.npy")

    # 加载模型
    if model_name == "CNN1D":
        model = model_class(input_size=25, num_classes=9, dropout=0.3)
    elif model_name == "GRU":
        model = model_class(input_size=25, hidden_size=128, num_classes=9, dropout=0.3)
    elif model_name == "GAT":
        model = model_class(input_size=25, hidden_dim=128, num_heads=4, num_classes=9, dropout=0.2)
    elif model_name == "STGCN":
        model = model_class(input_size=25, num_classes=9, dropout=0.2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 批量推理
    batch_size = 512
    all_probs, all_preds = [], []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        probs, preds = softmax_inference(model, batch, device)
        all_probs.append(probs)
        all_preds.append(preds)

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    # 计算指标
    ece = compute_ece(test_labels, all_probs)
    brier = compute_brier_score(test_labels, all_probs)
    nll = compute_nll(test_labels, all_probs)
    macro_f1 = f1_score(test_labels, all_preds, average='macro')
    accuracy = np.mean(all_preds == test_labels)

    results = {
        "model": f"{model_name}",
        "method": "Softmax",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll)
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier: {brier:.4f}")
    print(f"  NLL: {nll:.4f}")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 统一模型配置：STGCN, GAT, CNN1D
    models_config = [
        ("STGCN", RESULTS_DIR / "stgcn" / "best_model.pth", STGCNClassifier),
        ("GAT", RESULTS_DIR / "gat" / "best_model.pth", GATClassifier),
        ("CNN1D", RESULTS_DIR / "cnn1d" / "best_model.pth", CNN1DClassifier),
    ]

    # 1. Softmax实验
    print("\n" + "="*60)
    print("Part 1: Softmax Calibration")
    print("="*60)
    softmax_results = []
    for model_name, model_path, model_class in models_config:
        if model_path.exists():
            results = run_softmax_experiment(model_name, model_path, model_class, device)
            softmax_results.append(results)
        else:
            print(f"Warning: {model_path} not found")

    with open(SOFTMAX_OUTPUT_FILE, 'w') as f:
        json.dump(softmax_results, f, indent=2)
    print(f"\nSoftmax results saved to {SOFTMAX_OUTPUT_FILE}")

    # 2. MC Dropout实验
    print("\n" + "="*60)
    print("Part 2: MC Dropout Calibration")
    print("="*60)
    mc_results = []
    for model_name, model_path, model_class in models_config:
        if model_path.exists():
            results = run_mc_dropout_experiment(model_name, model_path, model_class, device)
            mc_results.append(results)
        else:
            print(f"Warning: {model_path} not found")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(mc_results, f, indent=2)
    print(f"\nMC Dropout results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

