"""
Heteroscedastic BSTGCN的Leave-One-Out OOD检测实验

实验流程:
1. 在8类数据上训练Heteroscedastic BSTGCN模型 (排除SlowrateDoS)
2. 在ID测试集上评估分类性能
3. 计算ID和OOD样本的epistemic和aleatoric uncertainty
4. 评估OOD检测性能 (AUROC, AUPR, FPR@95TPR)

关键改进:
- 使用heteroscedastic建模学习数据不确定性
- 正确分解epistemic和aleatoric uncertainty
- 支持多种OOD score组合策略
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, average_precision_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from scripts.BSTGCN.elbo_loss import HeteroscedasticELBOLoss, KLWeightScheduler
from scripts.BSTGCN.mc_inference import mc_inference_heteroscedastic
from scripts.BSTGCN.utils import get_class_weights


def load_ood_data(data_dir: str, batch_size: int = 512):
    """加载OOD实验数据"""
    data_dir = Path(data_dir)

    train_seq = np.load(data_dir / 'train_sequences.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    val_seq = np.load(data_dir / 'val_sequences.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    test_id_seq = np.load(data_dir / 'test_id_sequences.npy')
    test_id_labels = np.load(data_dir / 'test_id_labels.npy')
    test_ood_seq = np.load(data_dir / 'test_ood_sequences.npy')

    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_seq), torch.LongTensor(train_labels)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_seq), torch.LongTensor(val_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_id_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_id_seq), torch.LongTensor(test_id_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_ood_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_ood_seq), torch.zeros(len(test_ood_seq), dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_id_loader, test_ood_loader, metadata


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray, score_name: str = "score"):
    """计算OOD检测指标"""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]

    return {
        f'{score_name}_auroc': auroc,
        f'{score_name}_aupr': aupr,
        f'{score_name}_fpr_at_95tpr': fpr_at_95tpr,
        f'{score_name}_id_mean': id_scores.mean(),
        f'{score_name}_id_std': id_scores.std(),
        f'{score_name}_ood_mean': ood_scores.mean(),
        f'{score_name}_ood_std': ood_scores.std(),
    }


def train_epoch_heteroscedastic(model, loader, loss_fn, kl_scheduler, optimizer, device):
    """训练一个epoch (Heteroscedastic模式)"""
    model.train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    all_preds, all_labels = [], []

    current_kl_weight = kl_scheduler.get_kl_weight()
    loss_fn.kl_weight = current_kl_weight

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        # Heteroscedastic前向传播: 返回logits和log_variance
        logits, log_variance = model(batch_x, sample=True, return_variance=True)
        loss, rec_loss, kl_loss = loss_fn(logits, log_variance, batch_y)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rec += rec_loss.item()
        total_kl += kl_loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, current_kl_weight


def validate_heteroscedastic(model, loader, device, num_samples=8, num_aleatoric_samples=5):
    """验证并计算不确定性分解"""
    model.eval()
    all_probs, all_labels = [], []
    all_epistemic, all_aleatoric, all_total = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            # Heteroscedastic MC推理
            probs, _, epistemic, aleatoric, total, _ = mc_inference_heteroscedastic(
                model, batch_x, num_samples=num_samples,
                num_aleatoric_samples=num_aleatoric_samples
            )
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_aleatoric.append(aleatoric.cpu().numpy())
            all_total.append(total.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(axis=1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return (acc, f1, np.array(all_labels), all_preds,
            np.concatenate(all_epistemic),
            np.concatenate(all_aleatoric),
            np.concatenate(all_total))


def get_uncertainties_heteroscedastic(model, loader, device, num_samples=30, num_aleatoric_samples=10):
    """获取样本的不确定性分解"""
    model.eval()
    all_epistemic, all_aleatoric, all_total = [], [], []
    all_variance = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            _, _, epistemic, aleatoric, total, variance = mc_inference_heteroscedastic(
                model, batch_x, num_samples=num_samples,
                num_aleatoric_samples=num_aleatoric_samples
            )
            all_epistemic.append(epistemic.cpu().numpy())
            all_aleatoric.append(aleatoric.cpu().numpy())
            all_total.append(total.cpu().numpy())
            all_variance.append(variance.cpu().numpy())

    return {
        'epistemic': np.concatenate(all_epistemic),
        'aleatoric': np.concatenate(all_aleatoric),
        'total': np.concatenate(all_total),
        'variance': np.concatenate(all_variance)
    }


def run_ood_experiment_heteroscedastic(config: dict):
    """运行Heteroscedastic OOD实验主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n[1/4] 加载OOD数据集...")
    train_loader, val_loader, test_id_loader, test_ood_loader, metadata = load_ood_data(
        config['data_dir'], config['batch_size']
    )

    num_classes = metadata['num_id_classes']
    num_features = metadata['num_features']

    print(f"  ID类别数: {num_classes}")
    print(f"  OOD类别: {metadata['ood_class_name']}")
    print(f"  训练样本: {metadata['train_size']:,}")
    print(f"  ID测试样本: {metadata['test_id_size']:,}")
    print(f"  OOD测试样本: {metadata['test_ood_size']:,}")

    # 创建Heteroscedastic模型
    print("\n[2/4] 创建Heteroscedastic BSTGCN模型...")
    model = BSTGCNModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_st_blocks=config['num_st_blocks'],
        k_neighbors=config['k_neighbors'],
        prior_std=config['prior_std'],
        dropout=config['dropout'],
        heteroscedastic=True  # 启用heteroscedastic
    ).to(device)

    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # Heteroscedastic损失函数
    train_labels = np.load(Path(config['data_dir']) / 'train_labels.npy')
    class_weights = get_class_weights(train_labels, num_classes).to(device)

    loss_fn = HeteroscedasticELBOLoss(
        model=model,
        kl_weight=config['kl_weight'],
        label_smoothing=config['label_smoothing'],
        class_weights=class_weights,
        num_batches=len(train_loader),
        num_mc_samples=config.get('loss_mc_samples', 10),
        variance_reg_weight=config.get('variance_reg_weight', 0.01),
        target_variance=config.get('target_variance', 0.5)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    kl_scheduler = KLWeightScheduler(
        kl_weight_max=config['kl_weight'],
        warmup_epochs=config['kl_warmup_epochs'],
        mode='linear'
    )

    # 训练
    print("\n[3/4] 训练Heteroscedastic模型...")
    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(config['epochs']):
        train_loss, train_acc, kl_weight = train_epoch_heteroscedastic(
            model, train_loader, loss_fn, kl_scheduler, optimizer, device
        )
        val_acc, val_f1, _, _, _, _, _ = validate_heteroscedastic(
            model, val_loader, device, num_samples=8, num_aleatoric_samples=5
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        kl_scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, kl_w={kl_weight:.6f}")

        if patience_counter >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # 加载最佳模型
    model.load_state_dict(best_state)

    # OOD检测评估
    print("\n[4/4] 评估OOD检测性能...")

    # ID测试集
    id_acc, id_f1, id_labels, id_preds, id_epistemic, id_aleatoric, id_total = validate_heteroscedastic(
        model, test_id_loader, device, num_samples=30, num_aleatoric_samples=10
    )
    print(f"\nID测试集分类性能:")
    print(f"  Accuracy: {id_acc:.4f}")
    print(f"  Macro F1: {id_f1:.4f}")

    # OOD测试集
    ood_uncertainties = get_uncertainties_heteroscedastic(
        model, test_ood_loader, device, num_samples=30, num_aleatoric_samples=10
    )

    # 计算多种OOD score的指标
    print(f"\n{'='*60}")
    print("OOD检测性能 (SlowrateDoS as Unknown)")
    print(f"{'='*60}")

    results = {
        'config': config,
        'id_classification': {'accuracy': id_acc, 'macro_f1': id_f1},
    }

    # 测试不同的OOD score
    score_configs = [
        ('epistemic', id_epistemic, ood_uncertainties['epistemic']),
        ('aleatoric', id_aleatoric, ood_uncertainties['aleatoric']),
        ('total', id_total, ood_uncertainties['total']),
    ]

    for name, id_scores, ood_scores in score_configs:
        metrics = compute_ood_metrics(id_scores, ood_scores, name)
        results[f'ood_{name}'] = metrics
        print(f"\n{name.upper()} Uncertainty:")
        print(f"  AUROC: {metrics[f'{name}_auroc']:.4f}")
        print(f"  AUPR: {metrics[f'{name}_aupr']:.4f}")
        print(f"  FPR@95TPR: {metrics[f'{name}_fpr_at_95tpr']:.4f}")
        print(f"  ID: {metrics[f'{name}_id_mean']:.4f} ± {metrics[f'{name}_id_std']:.4f}")
        print(f"  OOD: {metrics[f'{name}_ood_mean']:.4f} ± {metrics[f'{name}_ood_std']:.4f}")

    # 组合策略: epistemic + total (加权)
    print("\n--- 组合策略 ---")
    for alpha in [0.3, 0.5, 0.7]:
        combined_id = alpha * id_epistemic + (1 - alpha) * id_total
        combined_ood = alpha * ood_uncertainties['epistemic'] + (1 - alpha) * ood_uncertainties['total']
        combined_metrics = compute_ood_metrics(combined_id, combined_ood, f'epi_total_a{alpha}')
        results[f'ood_epi_total_a{alpha}'] = combined_metrics
        print(f"\nEPISTEMIC + TOTAL (α={alpha}):")
        print(f"  AUROC: {combined_metrics[f'epi_total_a{alpha}_auroc']:.4f}")
        print(f"  AUPR: {combined_metrics[f'epi_total_a{alpha}_aupr']:.4f}")

    # 保存结果
    results['id_uncertainties'] = {
        'epistemic': id_epistemic, 'aleatoric': id_aleatoric, 'total': id_total
    }
    results['ood_uncertainties'] = ood_uncertainties
    results['metadata'] = metadata

    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = save_dir / f'ood_heteroscedastic_{timestamp}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n结果已保存: {result_path}")

    model_path = save_dir / f'bstgcn_heteroscedastic_{timestamp}.pt'
    torch.save(best_state, model_path)
    print(f"模型已保存: {model_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Heteroscedastic OOD Experiment')
    parser.add_argument('--data-dir', type=str, default='data/processed/ood_slowrate')
    parser.add_argument('--save-dir', type=str, default='results/ood_experiment')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-st-blocks', type=int, default=3)
    parser.add_argument('--k-neighbors', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)

    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'hidden_dim': args.hidden_dim,
        'num_st_blocks': args.num_st_blocks,
        'k_neighbors': args.k_neighbors,
        'prior_std': 1.0,
        'kl_weight': 1.5e-4,
        'kl_warmup_epochs': 40,
        'dropout': 0.15,
        'label_smoothing': 0.1,
        'patience': args.patience,
        'loss_mc_samples': 10,  # 损失函数中的MC采样次数
        'variance_reg_weight': 0.005,  # 方差正则化权重 (较小，让模型自由学习)
        'target_variance': 0.5,  # 目标方差
    }

    print("=" * 80)
    print("Heteroscedastic BSTGCN OOD Experiment".center(80))
    print("OOD Class: SlowrateDoS".center(80))
    print("=" * 80)

    results = run_ood_experiment_heteroscedastic(config)

    print("\n" + "=" * 80)
    print("实验完成!".center(80))
    print("=" * 80)

