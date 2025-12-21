"""
Leave-One-Out OOD检测实验

实验流程:
1. 在8类数据上训练BSTGCN模型 (排除SlowrateDoS)
2. 在ID测试集上评估分类性能
3. 计算ID和OOD样本的epistemic uncertainty
4. 评估OOD检测性能 (AUROC, AUPR, FPR@95TPR)
"""

import os
import sys
import argparse
import numpy as np
import torch
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
from scripts.BSTGCN.elbo_loss import ELBOLoss, KLWeightScheduler
from scripts.BSTGCN.mc_inference import mc_inference
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


def compute_ood_metrics(id_uncertainties: np.ndarray, ood_uncertainties: np.ndarray):
    """计算OOD检测指标"""
    # 构建二分类标签: ID=0, OOD=1
    labels = np.concatenate([
        np.zeros(len(id_uncertainties)),
        np.ones(len(ood_uncertainties))
    ])
    scores = np.concatenate([id_uncertainties, ood_uncertainties])
    
    # AUROC: 用uncertainty作为OOD score
    auroc = roc_auc_score(labels, scores)
    
    # AUPR
    aupr = average_precision_score(labels, scores)
    
    # FPR@95TPR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]
    threshold_at_95tpr = thresholds[idx]
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95tpr': fpr_at_95tpr,
        'threshold_at_95tpr': threshold_at_95tpr,
        'id_uncertainty_mean': id_uncertainties.mean(),
        'id_uncertainty_std': id_uncertainties.std(),
        'ood_uncertainty_mean': ood_uncertainties.mean(),
        'ood_uncertainty_std': ood_uncertainties.std(),
    }


def train_epoch(model, loader, elbo_loss, kl_scheduler, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    all_preds, all_labels = [], []

    # 更新KL权重
    current_kl_weight = kl_scheduler.get_kl_weight()
    elbo_loss.kl_weight = current_kl_weight

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x, sample=True)
        loss, rec_loss, kl_loss = elbo_loss(logits, batch_y)

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


def validate(model, loader, device, num_samples=8, enable_dropout=False):
    """验证并计算不确定性"""
    model.eval()
    all_probs, all_labels = [], []
    all_epistemic, all_total = [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            # mc_inference 返回: pred_mean, pred_class, epistemic, total
            probs, _, epistemic, total = mc_inference(
                model, batch_x, num_samples=num_samples, return_total_uncertainty=True,
                enable_dropout=enable_dropout
            )

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_epistemic.append(epistemic.cpu().numpy())
            all_total.append(total.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_epistemic = np.concatenate(all_epistemic)
    all_total = np.concatenate(all_total)
    all_preds = all_probs.argmax(axis=1)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return acc, f1, np.array(all_labels), all_preds, all_probs, all_epistemic, all_total


def get_uncertainties(model, loader, device, num_samples=30, enable_dropout=False):
    """获取样本的不确定性 (epistemic和total)"""
    model.eval()
    all_epistemic, all_total = [], []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            # mc_inference 返回: pred_mean, pred_class, epistemic, total
            _, _, epistemic, total = mc_inference(
                model, batch_x, num_samples=num_samples, return_total_uncertainty=True,
                enable_dropout=enable_dropout
            )
            all_epistemic.append(epistemic.cpu().numpy())
            all_total.append(total.cpu().numpy())

    return np.concatenate(all_epistemic), np.concatenate(all_total)


def run_ood_experiment(config: dict):
    """运行OOD实验主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n[1/4] 加载OOD数据集...")
    train_loader, val_loader, test_id_loader, test_ood_loader, metadata = load_ood_data(
        config['data_dir'], config['batch_size']
    )

    num_classes = metadata['num_id_classes']
    num_features = metadata['num_features']
    seq_len = metadata['sequence_length']

    print(f"  ID类别数: {num_classes}")
    print(f"  OOD类别: {metadata['ood_class_name']}")
    print(f"  训练样本: {metadata['train_size']:,}")
    print(f"  ID测试样本: {metadata['test_id_size']:,}")
    print(f"  OOD测试样本: {metadata['test_ood_size']:,}")

    # 创建模型
    print("\n[2/4] 创建BSTGCN模型...")
    model = BSTGCNModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_st_blocks=config['num_st_blocks'],
        k_neighbors=config['k_neighbors'],
        prior_std=config['prior_std'],
        dropout=config['dropout']
    ).to(device)

    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    train_labels = np.load(Path(config['data_dir']) / 'train_labels.npy')
    class_weights = get_class_weights(train_labels, num_classes).to(device)

    elbo_loss = ELBOLoss(
        model=model,
        kl_weight=config['kl_weight'],
        label_smoothing=config['label_smoothing'],
        class_weights=class_weights,
        num_batches=len(train_loader)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    # KL权重调度器
    kl_scheduler = KLWeightScheduler(
        kl_weight_max=config['kl_weight'],
        warmup_epochs=config['kl_warmup_epochs'],
        mode='linear'
    )

    # 训练
    print("\n[3/4] 训练模型...")
    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(config['epochs']):
        train_loss, train_acc, kl_weight = train_epoch(
            model, train_loader, elbo_loss, kl_scheduler, optimizer, device
        )
        val_acc, val_f1, _, _, _, _, _ = validate(model, val_loader, device, num_samples=20)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # 更新KL调度器
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

    # 方法1: 仅贝叶斯权重采样
    print("\n" + "="*60)
    print("方法1: 贝叶斯权重采样 (100 MC samples)")
    print("="*60)

    id_acc, id_f1, id_labels, id_preds, id_probs, id_epistemic, id_total = validate(
        model, test_id_loader, device, num_samples=100, enable_dropout=False
    )
    print(f"\nID测试集分类性能:")
    print(f"  Accuracy: {id_acc:.4f}")
    print(f"  Macro F1: {id_f1:.4f}")

    ood_epistemic, ood_total = get_uncertainties(
        model, test_ood_loader, device, num_samples=100, enable_dropout=False
    )

    epistemic_metrics = compute_ood_metrics(id_epistemic, ood_epistemic)
    print(f"\n--- EPISTEMIC Uncertainty (互信息) ---")
    print(f"  AUROC: {epistemic_metrics['auroc']:.4f}")
    print(f"  AUPR: {epistemic_metrics['aupr']:.4f}")
    print(f"  FPR@95TPR: {epistemic_metrics['fpr_at_95tpr']:.4f}")

    total_metrics = compute_ood_metrics(id_total, ood_total)
    print(f"\n--- TOTAL Uncertainty (预测熵) ---")
    print(f"  AUROC: {total_metrics['auroc']:.4f}")
    print(f"  AUPR: {total_metrics['aupr']:.4f}")

    # 方法2: 贝叶斯权重采样 + Dropout
    print("\n" + "="*60)
    print("方法2: 贝叶斯 + Dropout (100 MC samples)")
    print("="*60)

    id_acc2, id_f12, _, _, _, id_epistemic2, id_total2 = validate(
        model, test_id_loader, device, num_samples=100, enable_dropout=True
    )
    print(f"\nID测试集分类性能:")
    print(f"  Accuracy: {id_acc2:.4f}")
    print(f"  Macro F1: {id_f12:.4f}")

    ood_epistemic2, ood_total2 = get_uncertainties(
        model, test_ood_loader, device, num_samples=100, enable_dropout=True
    )

    epistemic_metrics2 = compute_ood_metrics(id_epistemic2, ood_epistemic2)
    print(f"\n--- EPISTEMIC Uncertainty (互信息) ---")
    print(f"  AUROC: {epistemic_metrics2['auroc']:.4f}")
    print(f"  AUPR: {epistemic_metrics2['aupr']:.4f}")
    print(f"  FPR@95TPR: {epistemic_metrics2['fpr_at_95tpr']:.4f}")

    total_metrics2 = compute_ood_metrics(id_total2, ood_total2)
    print(f"\n--- TOTAL Uncertainty (预测熵) ---")
    print(f"  AUROC: {total_metrics2['auroc']:.4f}")
    print(f"  AUPR: {total_metrics2['aupr']:.4f}")

    # 保存结果
    results = {
        'config': config,
        'id_classification': {'accuracy': id_acc, 'macro_f1': id_f1},
        'bayesian_only': {
            'epistemic': epistemic_metrics,
            'total': total_metrics,
        },
        'bayesian_dropout': {
            'epistemic': epistemic_metrics2,
            'total': total_metrics2,
        },
        'metadata': metadata
    }

    # 保存结果
    results = {
        'config': config,
        'id_classification': {'accuracy': id_acc, 'macro_f1': id_f1},
        'ood_epistemic': epistemic_metrics,
        'ood_total': total_metrics,
        'id_epistemic': id_epistemic,
        'id_total': id_total,
        'ood_epistemic_values': ood_epistemic,
        'ood_total_values': ood_total,
        'metadata': metadata
    }

    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = save_dir / f'ood_results_{timestamp}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n结果已保存: {result_path}")

    # 保存模型
    model_path = save_dir / f'bstgcn_ood_{timestamp}.pt'
    torch.save(best_state, model_path)
    print(f"模型已保存: {model_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Leave-One-Out OOD Experiment')
    parser.add_argument('--data-dir', type=str, default='data/processed/ood_slowrate',
                        help='OOD dataset directory')
    parser.add_argument('--save-dir', type=str, default='results/ood_experiment',
                        help='Directory to save results')
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
        'prior_std': 1.0,  # 标准先验
        'kl_weight': 2e-4,  # 增加KL权重以防止过拟合
        'kl_warmup_epochs': 20,  # 更早引入KL正则
        'dropout': 0.3,  # 增加dropout以增强不确定性
        'label_smoothing': 0.1,
        'patience': args.patience
    }

    print("=" * 80)
    print("Leave-One-Out OOD Experiment".center(80))
    print("OOD Class: SlowrateDoS".center(80))
    print("=" * 80)

    results = run_ood_experiment(config)

    print("\n" + "=" * 80)
    print("实验完成!".center(80))
    print("=" * 80)

