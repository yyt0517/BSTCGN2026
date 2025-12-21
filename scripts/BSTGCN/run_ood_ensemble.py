"""
Deep Ensemble OOD检测实验

通过训练多个独立的BSTGCN模型并组合预测来估计不确定性
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

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


def train_single_model(train_loader, val_loader, num_features, num_classes, device, config, seed):
    """训练单个BSTGCN模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BSTGCNModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_heads=8,
        num_st_blocks=config['num_st_blocks'],
        num_classes=num_classes,
        dropout=config['dropout'],
        k_neighbors=config['k_neighbors'],
        prior_std=config['prior_std']
    ).to(device)

    class_weights = get_class_weights(train_loader.dataset.tensors[1].numpy(), num_classes)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    elbo_loss = ELBOLoss(
        model=model, kl_weight=0.0, label_smoothing=config['label_smoothing'],
        num_batches=len(train_loader), class_weights=class_weights_tensor
    )
    kl_scheduler = KLWeightScheduler(
        kl_weight_max=config['kl_weight'], warmup_epochs=config['kl_warmup_epochs']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        kl_scheduler.step()
        elbo_loss.kl_weight = kl_scheduler.get_kl_weight()

        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, sample=True)
            loss, _, _ = elbo_loss(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x, sample=False)
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            break

    model.load_state_dict(best_state)
    return model, best_val_f1


def ensemble_inference(models, x, num_mc_samples=30):
    """集成推理：组合多个模型的预测"""
    all_probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            # 每个模型进行MC采样
            probs_list = []
            for _ in range(num_mc_samples):
                logits = model(x, sample=True)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)
            # 模型内平均
            model_probs = torch.stack(probs_list, dim=0).mean(dim=0)
            all_probs.append(model_probs)

    # 模型间堆叠: (num_models, B, C)
    all_probs = torch.stack(all_probs, dim=0)

    # 集成预测均值
    pred_mean = all_probs.mean(dim=0)  # (B, C)

    # 计算不确定性
    epsilon = 1e-10
    # 总不确定性: 预测均值的熵
    total_uncertainty = -torch.sum(pred_mean * torch.log(pred_mean + epsilon), dim=-1)

    # 数据不确定性: 各模型熵的均值
    entropy_per_model = -torch.sum(all_probs * torch.log(all_probs + epsilon), dim=-1)
    aleatoric_uncertainty = entropy_per_model.mean(dim=0)

    # 认知不确定性: 互信息
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return pred_mean, epistemic_uncertainty, total_uncertainty


def compute_ood_metrics(id_scores, ood_scores):
    """计算OOD检测指标"""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]

    return {
        'auroc': auroc, 'aupr': aupr, 'fpr_at_95tpr': fpr_at_95tpr,
        'id_mean': id_scores.mean(), 'id_std': id_scores.std(),
        'ood_mean': ood_scores.mean(), 'ood_std': ood_scores.std()
    }


def main():
    parser = argparse.ArgumentParser(description='Deep Ensemble OOD Detection')
    parser.add_argument('--data-dir', type=str, default='data/processed/ood_slowrate')
    parser.add_argument('--save-dir', type=str, default='results/ood_experiment')
    parser.add_argument('--num-models', type=int, default=5, help='训练模型数量')
    parser.add_argument('--top-k', type=int, default=0, help='选择top-k个最佳模型(0=全部)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-mc-samples', type=int, default=30)
    parser.add_argument('--f1-threshold', type=float, default=0.90, help='模型选择的F1阈值')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'batch_size': 512,
        'epochs': args.epochs,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'hidden_dim': 128,
        'num_st_blocks': 3,
        'k_neighbors': 10,
        'prior_std': 1.0,
        'kl_weight': 2e-4,
        'kl_warmup_epochs': 20,
        'dropout': 0.3,
        'label_smoothing': 0.1,
        'patience': args.patience
    }

    # 加载数据
    print("\n[1/4] 加载数据...")
    train_loader, val_loader, test_id_loader, test_ood_loader, metadata = load_ood_data(
        config['data_dir'], config['batch_size']
    )
    num_features = train_loader.dataset.tensors[0].shape[-1]
    num_classes = metadata['num_id_classes']
    print(f"  ID类别数: {num_classes}")
    print(f"  训练样本: {len(train_loader.dataset):,}")

    # 训练多个模型
    print(f"\n[2/4] 训练 {args.num_models} 个独立模型...")
    all_models = []
    all_f1s = []
    for i in range(args.num_models):
        print(f"\n  训练模型 {i+1}/{args.num_models} (seed={42+i*10})...")
        model, val_f1 = train_single_model(
            train_loader, val_loader, num_features, num_classes, device, config, seed=42+i*10
        )
        all_models.append(model)
        all_f1s.append(val_f1)
        print(f"    最佳验证F1: {val_f1:.4f}")

    # 选择最佳模型
    if args.top_k > 0:
        # 选择top-k个最佳模型
        sorted_indices = np.argsort(all_f1s)[::-1][:args.top_k]
        models = [all_models[i] for i in sorted_indices]
        selected_f1s = [all_f1s[i] for i in sorted_indices]
        print(f"\n  选择top-{args.top_k}模型, F1: {[f'{f:.4f}' for f in selected_f1s]}")
    elif args.f1_threshold > 0:
        # 选择F1大于阈值的模型
        models = [m for m, f in zip(all_models, all_f1s) if f >= args.f1_threshold]
        selected_f1s = [f for f in all_f1s if f >= args.f1_threshold]
        if len(models) == 0:
            print(f"\n  警告: 无模型达到F1阈值 {args.f1_threshold}, 使用全部模型")
            models = all_models
        else:
            print(f"\n  选择 {len(models)}/{args.num_models} 个F1>={args.f1_threshold}的模型")
    else:
        models = all_models

    print(f"  集成模型数量: {len(models)}")

    # 评估集成模型
    print("\n[3/4] 评估集成模型...")

    # ID样本
    id_epistemic_list, id_total_list, id_preds_list, id_labels_list = [], [], [], []
    for batch_x, batch_y in test_id_loader:
        batch_x = batch_x.to(device)
        pred_mean, epistemic, total = ensemble_inference(models, batch_x, args.num_mc_samples)
        id_epistemic_list.append(epistemic.cpu().numpy())
        id_total_list.append(total.cpu().numpy())
        id_preds_list.extend(pred_mean.argmax(dim=-1).cpu().numpy())
        id_labels_list.extend(batch_y.numpy())

    id_epistemic = np.concatenate(id_epistemic_list)
    id_total = np.concatenate(id_total_list)
    id_preds = np.array(id_preds_list)
    id_labels = np.array(id_labels_list)

    # OOD样本
    ood_epistemic_list, ood_total_list = [], []
    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, epistemic, total = ensemble_inference(models, batch_x, args.num_mc_samples)
        ood_epistemic_list.append(epistemic.cpu().numpy())
        ood_total_list.append(total.cpu().numpy())

    ood_epistemic = np.concatenate(ood_epistemic_list)
    ood_total = np.concatenate(ood_total_list)

    # 计算指标
    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    print(f"\n[4/4] 结果汇总")
    print("="*60)
    print(f"ID测试集分类性能:")
    print(f"  Accuracy: {id_acc:.4f}")
    print(f"  Macro F1: {id_f1:.4f}")

    epistemic_metrics = compute_ood_metrics(id_epistemic, ood_epistemic)
    print(f"\n--- EPISTEMIC Uncertainty (互信息) ---")
    print(f"  AUROC: {epistemic_metrics['auroc']:.4f}")
    print(f"  AUPR: {epistemic_metrics['aupr']:.4f}")
    print(f"  FPR@95TPR: {epistemic_metrics['fpr_at_95tpr']:.4f}")
    print(f"  ID mean: {epistemic_metrics['id_mean']:.4f} ± {epistemic_metrics['id_std']:.4f}")
    print(f"  OOD mean: {epistemic_metrics['ood_mean']:.4f} ± {epistemic_metrics['ood_std']:.4f}")

    total_metrics = compute_ood_metrics(id_total, ood_total)
    print(f"\n--- TOTAL Uncertainty (预测熵) ---")
    print(f"  AUROC: {total_metrics['auroc']:.4f}")
    print(f"  AUPR: {total_metrics['aupr']:.4f}")
    print(f"  FPR@95TPR: {total_metrics['fpr_at_95tpr']:.4f}")

    # 保存结果
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = {
        'id_accuracy': id_acc, 'id_f1': id_f1,
        'epistemic_metrics': epistemic_metrics,
        'total_metrics': total_metrics,
        'num_models': args.num_models,
        'config': config
    }

    with open(save_dir / f'ensemble_results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n结果已保存: {save_dir / f'ensemble_results_{timestamp}.pkl'}")
    print("="*60)


if __name__ == '__main__':
    main()

