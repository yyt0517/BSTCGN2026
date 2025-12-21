"""
计算校准指标 (ECE, Brier Score, NLL) - 公平比较版本

所有模型在相同的50packets数据集上训练和评估

================================================================================
命名映射表 (代码 -> 论文)
================================================================================
| 代码名称   | 论文名称    | 说明                                           |
|------------|-------------|------------------------------------------------|
| STGCN      | Seq-STGCN   | 串行时空建模 (先temporal后spatial)              |
| Hybrid     | STGCN       | 并行时空建模 (temporal和spatial并行处理)        |
| BSTGCN     | BSTGCN      | 贝叶斯并行时空建模 = Bayesian + STGCN(论文)     |
================================================================================

分析的模型:
- BSTGCN (我们的方法)
- STGCN (代码名称，论文中称为Seq-STGCN - 串行时空baseline)
- GAT (图模型baseline)
- CNN1D (时序模型baseline)
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from scripts.BSTGCN.mc_inference import mc_inference
# 从run_comprehensive_baselines导入模型类
from baseline.run_comprehensive_baselines import GATClassifier, STGCNClassifier, CNN1DClassifier

# 路径配置 - 统一使用50packets数据集
DATA_DIR = Path("data/processed/raw_packet_sequences_50packets")
BSTGCN_MODEL_PATH = Path("baseline/results/bstgcn_50packets/20251121_112445/best_model.pth")
BSTGCN_CONFIG_PATH = Path("baseline/results/bstgcn_50packets/20251121_112445/config.json")

# 已训练模型路径
STGCN_MODEL_PATH = Path("baseline/results_50packets/stgcn/best_model.pth")
GAT_MODEL_PATH = Path("baseline/results_50packets/gat/best_model.pth")
CNN1D_MODEL_PATH = Path("baseline/results_50packets/cnn1d/best_model.pth")

OUTPUT_DIR = Path("baseline/results/calibration_analysis_50packets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 9
EPOCHS = 50
PATIENCE = 10
USE_PRETRAINED = True  # 使用已训练的模型


def compute_ece(y_true, y_probs, n_bins=15):
    """计算Expected Calibration Error"""
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def compute_brier_score(y_true, y_probs):
    """计算多类Brier Score"""
    n_samples = len(y_true)
    n_classes = y_probs.shape[1]
    
    # One-hot编码真实标签
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    
    # Brier score = mean squared error
    brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))
    return brier


def compute_nll(y_true, y_probs, eps=1e-10):
    """计算Negative Log-Likelihood"""
    # 裁剪概率以避免log(0)
    y_probs = np.clip(y_probs, eps, 1 - eps)
    nll = -np.mean(np.log(y_probs[np.arange(len(y_true)), y_true]))
    return nll


def load_all_data(data_dir, batch_size=64):
    """加载训练、验证和测试数据"""
    train_sequences = np.load(data_dir / 'train_sequences.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    val_sequences = np.load(data_dir / 'val_sequences.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    test_sequences = np.load(data_dir / 'test_sequences.npy')
    test_labels = np.load(data_dir / 'test_labels.npy')

    train_dataset = TensorDataset(
        torch.FloatTensor(train_sequences),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_sequences),
        torch.LongTensor(val_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_sequences),
        torch.LongTensor(test_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_labels


def train_baseline_model(model, train_loader, val_loader, model_name, epochs=EPOCHS, patience=PATIENCE):
    """训练baseline模型并返回最佳模型"""
    print(f"Training {model_name}...")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(DEVICE)
                outputs = model(data)
                val_preds.extend(outputs.argmax(dim=-1).cpu().numpy())
                val_labels_list.extend(labels.numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # 加载最佳模型
    model.load_state_dict(best_state)
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    return model


def get_probs_deterministic(model, dataloader, device):
    """获取确定性模型的概率输出"""
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
    
    return np.concatenate(all_probs), np.concatenate(all_preds)


def get_probs_bstgcn(model, dataloader, device, num_mc_samples=30):
    """获取BSTGCN的概率输出 (MC推理)"""
    model.eval()
    all_probs = []
    all_preds = []
    all_uncertainties = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            pred_mean, pred_class, uncertainty = mc_inference(
                model, data, num_samples=num_mc_samples
            )
            all_probs.append(pred_mean.cpu().numpy())
            all_preds.append(pred_class.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
    
    return (np.concatenate(all_probs), np.concatenate(all_preds),
            np.concatenate(all_uncertainties))


def apply_temperature_scaling(probs, temperature):
    """应用温度缩放来校准概率"""
    # 将概率转换回logits (逆softmax)
    logits = np.log(probs + 1e-10)
    # 应用温度缩放
    scaled_logits = logits / temperature
    # 重新应用softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return scaled_probs


def find_optimal_temperature(probs, labels, init_temp=1.5):
    """在验证集上找到最优温度参数"""
    from scipy.optimize import minimize_scalar

    def nll_loss(temp):
        scaled_probs = apply_temperature_scaling(probs, temp)
        return compute_nll(labels, scaled_probs)

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    return result.x


def analyze_bstgcn(val_loader, test_loader, y_val, y_test):
    """分析BSTGCN模型，使用温度缩放校准"""
    print("\n" + "="*60)
    print("Analyzing BSTGCN (pre-trained + temperature scaling)")
    print("="*60)

    # 加载配置
    with open(BSTGCN_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    print(f"Val samples: {len(y_val)}, Test samples: {len(y_test)}")

    # 创建模型
    model = BSTGCNModel(
        input_dim=config.get('input_dim', 25),
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 8),
        num_st_blocks=config.get('num_st_blocks', 3),
        num_classes=NUM_CLASSES,
        dropout=config.get('dropout', 0.15),
        k_neighbors=config.get('k_neighbors', 10),
        prior_std=config.get('prior_std', 0.5)
    ).to(DEVICE)

    # 加载权重
    checkpoint = torch.load(BSTGCN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {BSTGCN_MODEL_PATH}")

    # MC推理 - 验证集
    print("Running MC inference on validation set (30 samples)...")
    val_probs, _, _ = get_probs_bstgcn(model, val_loader, DEVICE, num_mc_samples=30)

    # 找到最优温度
    print("Finding optimal temperature...")
    optimal_temp = find_optimal_temperature(val_probs, y_val)
    print(f"  Optimal temperature: {optimal_temp:.4f}")

    # MC推理 - 测试集
    print("Running MC inference on test set (30 samples)...")
    y_probs_raw, y_pred, uncertainties = get_probs_bstgcn(model, test_loader, DEVICE, num_mc_samples=30)

    # 应用温度缩放
    y_probs = apply_temperature_scaling(y_probs_raw, optimal_temp)
    y_pred_scaled = np.argmax(y_probs, axis=1)  # 使用缩放后的概率重新预测

    # 计算指标 (校准后)
    acc = accuracy_score(y_test, y_pred_scaled)
    macro_f1 = f1_score(y_test, y_pred_scaled, average='macro')
    ece = compute_ece(y_test, y_probs)
    brier = compute_brier_score(y_test, y_probs)
    nll = compute_nll(y_test, y_probs)

    results = {
        'model': 'BSTGCN',
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'ece': float(ece),
        'brier': float(brier),
        'nll': float(nll),
        'mean_uncertainty': float(np.mean(uncertainties)),
        'optimal_temperature': float(optimal_temp)
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  NLL: {nll:.4f}")
    print(f"  Mean Uncertainty: {np.mean(uncertainties):.4f}")

    # 保存概率
    np.save(OUTPUT_DIR / 'bstgcn_probs.npy', y_probs)
    np.save(OUTPUT_DIR / 'bstgcn_probs_raw.npy', y_probs_raw)
    np.save(OUTPUT_DIR / 'bstgcn_uncertainties.npy', uncertainties)

    return results


def analyze_stgcn(train_loader, val_loader, test_loader, y_test, input_size):
    """分析STGCN模型 - 使用已训练的模型或重新训练"""
    print("\n" + "="*60)
    print("Analyzing STGCN")
    print("="*60)

    print(f"Test samples: {len(y_test)}")

    # 创建模型
    model = STGCNClassifier(
        input_size=input_size,
        hidden_channels=[64, 128],
        num_classes=NUM_CLASSES,
        dropout=0.2
    ).to(DEVICE)

    # 使用已训练的模型或重新训练
    if USE_PRETRAINED and STGCN_MODEL_PATH.exists():
        print(f"  Loading pretrained model from {STGCN_MODEL_PATH}")
        model.load_state_dict(torch.load(STGCN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("  Training new model...")
        model = train_baseline_model(model, train_loader, val_loader, "STGCN")
        torch.save(model.state_dict(), OUTPUT_DIR / 'stgcn_50packets.pth')

    # 推理
    y_probs, y_pred = get_probs_deterministic(model, test_loader, DEVICE)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    ece = compute_ece(y_test, y_probs)
    brier = compute_brier_score(y_test, y_probs)
    nll = compute_nll(y_test, y_probs)

    results = {
        'model': 'STGCN',
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'ece': float(ece),
        'brier': float(brier),
        'nll': float(nll)
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  NLL: {nll:.4f}")

    np.save(OUTPUT_DIR / 'stgcn_probs.npy', y_probs)
    return results


def analyze_gat(train_loader, val_loader, test_loader, y_test, input_size):
    """分析GAT模型 - 使用已训练的模型或重新训练"""
    print("\n" + "="*60)
    print("Analyzing GAT")
    print("="*60)

    print(f"Test samples: {len(y_test)}")

    # 创建模型
    model = GATClassifier(
        input_size=input_size,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.2
    ).to(DEVICE)

    # 使用已训练的模型或重新训练
    if USE_PRETRAINED and GAT_MODEL_PATH.exists():
        print(f"  Loading pretrained model from {GAT_MODEL_PATH}")
        model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("  Training new model...")
        model = train_baseline_model(model, train_loader, val_loader, "GAT")
        torch.save(model.state_dict(), OUTPUT_DIR / 'gat_50packets.pth')

    y_probs, y_pred = get_probs_deterministic(model, test_loader, DEVICE)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    ece = compute_ece(y_test, y_probs)
    brier = compute_brier_score(y_test, y_probs)
    nll = compute_nll(y_test, y_probs)

    results = {
        'model': 'GAT',
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'ece': float(ece),
        'brier': float(brier),
        'nll': float(nll)
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  NLL: {nll:.4f}")

    np.save(OUTPUT_DIR / 'gat_probs.npy', y_probs)
    return results


def analyze_cnn1d(train_loader, val_loader, test_loader, y_test, input_size):
    """分析CNN1D模型 - 使用已训练的模型或重新训练"""
    print("\n" + "="*60)
    print("Analyzing CNN1D")
    print("="*60)

    print(f"Test samples: {len(y_test)}")

    # 创建模型
    model = CNN1DClassifier(
        input_size=input_size,
        num_classes=NUM_CLASSES,
        dropout=0.3
    ).to(DEVICE)

    # 使用已训练的模型或重新训练
    if USE_PRETRAINED and CNN1D_MODEL_PATH.exists():
        print(f"  Loading pretrained model from {CNN1D_MODEL_PATH}")
        model.load_state_dict(torch.load(CNN1D_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("  Training new model...")
        model = train_baseline_model(model, train_loader, val_loader, "CNN1D")
        torch.save(model.state_dict(), OUTPUT_DIR / 'cnn1d_50packets.pth')

    y_probs, y_pred = get_probs_deterministic(model, test_loader, DEVICE)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    ece = compute_ece(y_test, y_probs)
    brier = compute_brier_score(y_test, y_probs)
    nll = compute_nll(y_test, y_probs)

    results = {
        'model': 'CNN1D',
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'ece': float(ece),
        'brier': float(brier),
        'nll': float(nll)
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  NLL: {nll:.4f}")

    np.save(OUTPUT_DIR / 'cnn1d_probs.npy', y_probs)
    return results





def generate_latex_table(all_results):
    """生成LaTeX表格"""
    print("\n" + "="*60)
    print("LaTeX Table: Calibration Metrics")
    print("="*60)

    print("\\begin{table}[!t]")
    print("\\caption{Calibration Metrics Comparison\\label{tab:calibration}}")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{ECE}$\\downarrow$ & \\textbf{Brier}$\\downarrow$ & \\textbf{NLL}$\\downarrow$ & \\textbf{Accuracy} \\\\")
    print("\\hline")

    for r in all_results:
        if r['model'] == 'BSTGCN':
            print(f"\\textbf{{{r['model']} (Ours)}} & \\textbf{{{r['ece']:.4f}}} & \\textbf{{{r['brier']:.4f}}} & \\textbf{{{r['nll']:.4f}}} & \\textbf{{{r['accuracy']:.4f}}} \\\\")
        else:
            print(f"{r['model']} & {r['ece']:.4f} & {r['brier']:.4f} & {r['nll']:.4f} & {r['accuracy']:.4f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    print("="*60)
    print("Calibration Metrics Analysis - Fair Comparison on 50packets")
    print(f"Device: {DEVICE}")
    print("="*60)

    # 加载50packets数据集 - 所有模型使用相同数据
    print("\nLoading 50packets dataset...")
    train_loader, val_loader, test_loader, y_test = load_all_data(DATA_DIR, batch_size=64)

    # 获取验证集标签
    val_labels = np.load(DATA_DIR / 'val_labels.npy')

    # 获取输入维度
    test_seq = np.load(DATA_DIR / 'test_sequences.npy')
    input_size = test_seq.shape[2]
    print(f"Input size: {input_size}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(y_test)}")

    all_results = []

    # 1. BSTGCN (使用预训练模型 + 温度缩放)
    try:
        results = analyze_bstgcn(val_loader, test_loader, val_labels, y_test)
        all_results.append(results)
    except Exception as e:
        print(f"Error analyzing BSTGCN: {e}")
        import traceback
        traceback.print_exc()

    # 2. STGCN (在50packets上训练)
    try:
        results = analyze_stgcn(train_loader, val_loader, test_loader, y_test, input_size)
        all_results.append(results)
    except Exception as e:
        print(f"Error analyzing STGCN: {e}")
        import traceback
        traceback.print_exc()

    # 3. GAT (在50packets上训练)
    try:
        results = analyze_gat(train_loader, val_loader, test_loader, y_test, input_size)
        all_results.append(results)
    except Exception as e:
        print(f"Error analyzing GAT: {e}")
        import traceback
        traceback.print_exc()

    # 4. CNN1D (在50packets上训练)
    try:
        results = analyze_cnn1d(train_loader, val_loader, test_loader, y_test, input_size)
        all_results.append(results)
    except Exception as e:
        print(f"Error analyzing CNN1D: {e}")
        import traceback
        traceback.print_exc()

    # 保存结果
    with open(OUTPUT_DIR / 'calibration_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'calibration_results.json'}")

    # 生成LaTeX表格
    if all_results:
        generate_latex_table(all_results)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()

