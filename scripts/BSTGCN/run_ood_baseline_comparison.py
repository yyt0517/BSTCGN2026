"""
OOD检测Baseline对比实验

对比方法:
1. MSP (Maximum Softmax Probability) - Hendrycks & Gimpel 2017
2. Energy Score - Liu et al. 2020
3. Entropy - 预测熵
4. MC Dropout - Gal & Ghahramani 2016
5. BSTGCN (Ours) - 变分推断

实验设置: Leave-One-Attack-Out (SlowrateDoS as OOD)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from baseline.stgcn_optimized import OptimizedSTGCN
from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from scripts.BSTGCN.elbo_loss import ELBOLoss, KLWeightScheduler
from scripts.BSTGCN.mc_inference import mc_inference
from scripts.BSTGCN.utils import get_class_weights


# ==================== OOD Score计算函数 ====================

def compute_msp_score(logits: torch.Tensor) -> np.ndarray:
    """Maximum Softmax Probability Score (越高越可能是OOD)"""
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1)[0]
    return (1 - max_probs).cpu().numpy()  # 1 - max_prob作为OOD score


def compute_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Energy Score (越高越可能是OOD)"""
    # Energy = -T * log(sum(exp(f_c / T)))
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return (-energy).cpu().numpy()  # 取负使得越高越OOD


def compute_entropy_score(logits: torch.Tensor) -> np.ndarray:
    """Entropy Score (越高越可能是OOD)"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.cpu().numpy()


def mc_dropout_inference(model, x, num_samples=30):
    """MC Dropout推理 - 开启dropout进行多次采样"""
    model.eval()

    # 开启所有dropout层
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    probs_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

    probs_stack = torch.stack(probs_list, dim=0)  # (M, B, C)
    pred_mean = probs_stack.mean(dim=0)  # (B, C)

    # 计算epistemic uncertainty (互信息)
    epsilon = 1e-10
    entropy_mean = -torch.sum(pred_mean * torch.log(pred_mean + epsilon), dim=-1)
    entropy_samples = -torch.sum(probs_stack * torch.log(probs_stack + epsilon), dim=-1)
    mean_entropy = entropy_samples.mean(dim=0)
    epistemic_uncertainty = entropy_mean - mean_entropy

    return pred_mean, epistemic_uncertainty.cpu().numpy()


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
    """计算OOD检测指标"""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]

    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95tpr': fpr_at_95tpr,
        'id_mean': id_scores.mean(),
        'id_std': id_scores.std(),
        'ood_mean': ood_scores.mean(),
        'ood_std': ood_scores.std()
    }


# ==================== 数据加载 ====================

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


# ==================== 训练函数 ====================

def train_deterministic_stgcn(train_loader, val_loader, num_features, num_classes,
                               device, config):
    """训练确定性STGCN模型 - 使用优化版以提高训练速度"""
    model = OptimizedSTGCN(
        input_size=num_features,
        hidden_channels=[64, 128, 128],  # 增加一层
        num_classes=num_classes,
        k_neighbors=config['k_neighbors'],
        dropout=config['dropout']
    ).to(device)

    class_weights = get_class_weights(train_loader.dataset.tensors[1].numpy(), num_classes)
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device),
        label_smoothing=config['label_smoothing']
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        # 训练
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: val_f1={val_f1:.4f}, best={best_val_f1:.4f}")

        if patience_counter >= config['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


def train_bstgcn(train_loader, val_loader, num_features, num_classes, device, config):
    """训练BSTGCN模型"""
    model = BSTGCNModel(
        input_dim=num_features,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_st_blocks=config['num_st_blocks'],
        num_classes=num_classes,
        dropout=config['dropout'],
        k_neighbors=config['k_neighbors'],
        prior_std=config['prior_std']
    ).to(device)

    class_weights = get_class_weights(train_loader.dataset.tensors[1].numpy(), num_classes)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    elbo_loss = ELBOLoss(
        model=model,
        kl_weight=0.0,
        label_smoothing=config['label_smoothing'],
        num_batches=len(train_loader),
        class_weights=class_weights_tensor
    )
    kl_scheduler = KLWeightScheduler(
        kl_weight_max=config['kl_weight_max'],
        warmup_epochs=config['kl_warmup_epochs']
    )

    # 使用较低的初始学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'] * 0.5,
                                   weight_decay=config['weight_decay'])
    # 添加学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    best_val_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        # 更新KL权重
        kl_scheduler.step()
        kl_weight = kl_scheduler.get_kl_weight()
        elbo_loss.kl_weight = kl_weight

        # 训练
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, sample=True)
            loss, _, _ = elbo_loss(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 放宽梯度裁剪
            optimizer.step()

        lr_scheduler.step()

        # 验证 (使用确定性推理)
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x, sample=False)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: val_f1={val_f1:.4f}, best={best_val_f1:.4f}, "
                  f"kl_w={kl_weight:.6f}, lr={current_lr:.6f}")

        if patience_counter >= config['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model



# ==================== 评估函数 ====================

def evaluate_deterministic_ood(model, test_id_loader, test_ood_loader, device):
    """评估确定性模型的OOD检测 (MSP, Energy, Entropy)"""
    model.eval()

    # 收集ID样本的logits和labels
    id_logits_list, id_labels_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_id_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            id_logits_list.append(logits)
            id_labels_list.extend(batch_y.numpy())

    id_logits = torch.cat(id_logits_list, dim=0)
    id_preds = id_logits.argmax(dim=-1).cpu().numpy()
    id_labels = np.array(id_labels_list)

    # 收集OOD样本的logits
    ood_logits_list = []
    with torch.no_grad():
        for batch_x, _ in test_ood_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            ood_logits_list.append(logits)

    ood_logits = torch.cat(ood_logits_list, dim=0)

    # 计算ID分类性能
    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    # 计算各种OOD scores
    results = {
        'id_accuracy': id_acc,
        'id_f1': id_f1
    }

    # MSP Score
    id_msp = compute_msp_score(id_logits)
    ood_msp = compute_msp_score(ood_logits)
    results['MSP'] = compute_ood_metrics(id_msp, ood_msp)

    # Energy Score
    id_energy = compute_energy_score(id_logits)
    ood_energy = compute_energy_score(ood_logits)
    results['Energy'] = compute_ood_metrics(id_energy, ood_energy)

    # Entropy Score
    id_entropy = compute_entropy_score(id_logits)
    ood_entropy = compute_entropy_score(ood_logits)
    results['Entropy'] = compute_ood_metrics(id_entropy, ood_entropy)

    return results


def evaluate_mc_dropout_ood(model, test_id_loader, test_ood_loader, device, num_samples=30):
    """评估MC Dropout的OOD检测"""
    # 收集ID样本
    id_uncertainties_list, id_preds_list, id_labels_list = [], [], []
    for batch_x, batch_y in test_id_loader:
        batch_x = batch_x.to(device)
        pred_mean, uncertainty = mc_dropout_inference(model, batch_x, num_samples)
        id_uncertainties_list.append(uncertainty)
        id_preds_list.extend(pred_mean.argmax(dim=-1).cpu().numpy())
        id_labels_list.extend(batch_y.numpy())

    id_uncertainties = np.concatenate(id_uncertainties_list)
    id_preds = np.array(id_preds_list)
    id_labels = np.array(id_labels_list)

    # 收集OOD样本
    ood_uncertainties_list = []
    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, uncertainty = mc_dropout_inference(model, batch_x, num_samples)
        ood_uncertainties_list.append(uncertainty)

    ood_uncertainties = np.concatenate(ood_uncertainties_list)

    # 计算指标
    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    return {
        'id_accuracy': id_acc,
        'id_f1': id_f1,
        'MC_Dropout': compute_ood_metrics(id_uncertainties, ood_uncertainties)
    }


def evaluate_bstgcn_ood(model, test_id_loader, test_ood_loader, device, num_samples=30):
    """评估BSTGCN的OOD检测"""
    model.eval()

    # 收集ID样本
    id_uncertainties_list, id_preds_list, id_labels_list = [], [], []
    for batch_x, batch_y in test_id_loader:
        batch_x = batch_x.to(device)
        _, pred_class, uncertainty = mc_inference(model, batch_x, num_samples=num_samples)
        id_uncertainties_list.append(uncertainty.cpu().numpy())
        id_preds_list.extend(pred_class.cpu().numpy())
        id_labels_list.extend(batch_y.numpy())

    id_uncertainties = np.concatenate(id_uncertainties_list)
    id_preds = np.array(id_preds_list)
    id_labels = np.array(id_labels_list)

    # 收集OOD样本
    ood_uncertainties_list = []
    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, _, uncertainty = mc_inference(model, batch_x, num_samples=num_samples)
        ood_uncertainties_list.append(uncertainty.cpu().numpy())

    ood_uncertainties = np.concatenate(ood_uncertainties_list)

    # 计算指标
    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    return {
        'id_accuracy': id_acc,
        'id_f1': id_f1,
        'BSTGCN': compute_ood_metrics(id_uncertainties, ood_uncertainties)
    }



# ==================== 主函数 ====================

def run_ood_baseline_comparison(config, bstgcn_only=False):
    """运行OOD baseline对比实验"""
    print("=" * 80)
    print("          OOD Detection Baseline Comparison Experiment")
    print("                  OOD Class: SlowrateDoS")
    if bstgcn_only:
        print("                  [BSTGCN Only Mode]")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 加载数据
    print("\n[1/5] 加载OOD数据集...")
    train_loader, val_loader, test_id_loader, test_ood_loader, metadata = \
        load_ood_data(config['data_dir'], config['batch_size'])

    num_classes = metadata['num_id_classes']
    num_features = metadata['num_features']
    print(f"  ID类别数: {num_classes}")
    print(f"  特征维度: {num_features}")
    print(f"  训练样本: {len(train_loader.dataset):,}")
    print(f"  ID测试样本: {len(test_id_loader.dataset):,}")
    print(f"  OOD测试样本: {len(test_ood_loader.dataset):,}")

    all_results = {}

    if not bstgcn_only:
        # ==================== 训练并评估确定性STGCN ====================
        print("\n[2/5] 训练确定性STGCN模型...")
        stgcn_model = train_deterministic_stgcn(
            train_loader, val_loader, num_features, num_classes, device,
            {
                'epochs': config['epochs'],
                'learning_rate': config['learning_rate'],
                'weight_decay': config['weight_decay'],
                'patience': config['patience'],
                'label_smoothing': config['label_smoothing'],
                'dropout': config['dropout'],
                'k_neighbors': config['k_neighbors']
            }
        )

        print("\n  评估确定性方法 (MSP, Energy, Entropy)...")
        det_results = evaluate_deterministic_ood(stgcn_model, test_id_loader, test_ood_loader, device)
        all_results['STGCN'] = det_results

        print(f"  ID Accuracy: {det_results['id_accuracy']:.4f}")
        print(f"  ID F1: {det_results['id_f1']:.4f}")
        for method in ['MSP', 'Energy', 'Entropy']:
            print(f"  {method}: AUROC={det_results[method]['auroc']:.4f}, "
                  f"AUPR={det_results[method]['aupr']:.4f}, "
                  f"FPR@95={det_results[method]['fpr_at_95tpr']:.4f}")
    else:
        print("\n[2/5] 跳过STGCN训练 (--bstgcn-only mode)")
        stgcn_model = None

    # ==================== 评估MC Dropout ====================
    if not bstgcn_only:
        print("\n[3/5] 评估MC Dropout...")
        mc_results = evaluate_mc_dropout_ood(
            stgcn_model, test_id_loader, test_ood_loader, device,
            num_samples=config['num_mc_samples']
        )
        all_results['MC_Dropout'] = mc_results

        print(f"  ID Accuracy: {mc_results['id_accuracy']:.4f}")
        print(f"  ID F1: {mc_results['id_f1']:.4f}")
        print(f"  MC Dropout: AUROC={mc_results['MC_Dropout']['auroc']:.4f}, "
              f"AUPR={mc_results['MC_Dropout']['aupr']:.4f}, "
              f"FPR@95={mc_results['MC_Dropout']['fpr_at_95tpr']:.4f}")
    else:
        print("\n[3/5] 跳过MC Dropout评估 (--bstgcn-only mode)")

    # ==================== 训练并评估BSTGCN ====================
    print("\n[4/5] 训练BSTGCN模型...")
    bstgcn_model = train_bstgcn(
        train_loader, val_loader, num_features, num_classes, device,
        {
            'epochs': config['epochs'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'patience': config['patience'],
            'label_smoothing': config['label_smoothing'],
            'dropout': config['dropout'],
            'k_neighbors': config['k_neighbors'],
            'hidden_dim': config['hidden_dim'],
            'num_heads': config['num_heads'],
            'num_st_blocks': config['num_st_blocks'],
            'prior_std': config['prior_std'],
            'kl_weight_max': config['kl_weight_max'],
            'kl_warmup_epochs': config['kl_warmup_epochs']
        }
    )

    print("\n  评估BSTGCN...")
    bstgcn_results = evaluate_bstgcn_ood(
        bstgcn_model, test_id_loader, test_ood_loader, device,
        num_samples=config['num_mc_samples']
    )
    all_results['BSTGCN'] = bstgcn_results

    print(f"  ID Accuracy: {bstgcn_results['id_accuracy']:.4f}")
    print(f"  ID F1: {bstgcn_results['id_f1']:.4f}")
    print(f"  BSTGCN: AUROC={bstgcn_results['BSTGCN']['auroc']:.4f}, "
          f"AUPR={bstgcn_results['BSTGCN']['aupr']:.4f}, "
          f"FPR@95={bstgcn_results['BSTGCN']['fpr_at_95tpr']:.4f}")

    # ==================== 汇总结果 ====================
    print("\n[5/5] 生成对比报告...")
    print("\n" + "=" * 80)
    print("                    OOD Detection Results Summary")
    print("=" * 80)
    print(f"{'Method':<15} {'AUROC':>10} {'AUPR':>10} {'FPR@95':>10} {'ID Acc':>10} {'ID F1':>10}")
    print("-" * 80)

    summary = []

    # 确定性方法（仅在非bstgcn_only模式下）
    if not bstgcn_only:
        for method in ['MSP', 'Energy', 'Entropy']:
            r = det_results[method]
            print(f"{method:<15} {r['auroc']:>10.4f} {r['aupr']:>10.4f} "
                  f"{r['fpr_at_95tpr']:>10.4f} {det_results['id_accuracy']:>10.4f} "
                  f"{det_results['id_f1']:>10.4f}")
            summary.append({
                'method': method,
                'auroc': r['auroc'],
                'aupr': r['aupr'],
                'fpr_at_95tpr': r['fpr_at_95tpr'],
                'id_accuracy': det_results['id_accuracy'],
                'id_f1': det_results['id_f1']
            })

        # MC Dropout
        r = mc_results['MC_Dropout']
        print(f"{'MC Dropout':<15} {r['auroc']:>10.4f} {r['aupr']:>10.4f} "
              f"{r['fpr_at_95tpr']:>10.4f} {mc_results['id_accuracy']:>10.4f} "
              f"{mc_results['id_f1']:>10.4f}")
        summary.append({
            'method': 'MC Dropout',
            'auroc': r['auroc'],
            'aupr': r['aupr'],
            'fpr_at_95tpr': r['fpr_at_95tpr'],
            'id_accuracy': mc_results['id_accuracy'],
            'id_f1': mc_results['id_f1']
        })

    # BSTGCN
    r = bstgcn_results['BSTGCN']
    print(f"{'BSTGCN (Ours)':<15} {r['auroc']:>10.4f} {r['aupr']:>10.4f} "
          f"{r['fpr_at_95tpr']:>10.4f} {bstgcn_results['id_accuracy']:>10.4f} "
          f"{bstgcn_results['id_f1']:>10.4f}")
    summary.append({
        'method': 'BSTGCN (Ours)',
        'auroc': r['auroc'],
        'aupr': r['aupr'],
        'fpr_at_95tpr': r['fpr_at_95tpr'],
        'id_accuracy': bstgcn_results['id_accuracy'],
        'id_f1': bstgcn_results['id_f1']
    })

    print("=" * 80)

    # 保存结果
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = save_dir / f'ood_baseline_comparison_{timestamp}.pkl'

    with open(results_path, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'summary': summary,
            'config': config
        }, f)

    print(f"\n结果已保存: {results_path}")

    return all_results, summary



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OOD Detection Baseline Comparison')
    parser.add_argument('--data-dir', type=str, default='data/processed/ood_slowrate',
                        help='OOD数据目录')
    parser.add_argument('--save-dir', type=str, default='results/ood_experiment',
                        help='结果保存目录')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--patience', type=int, default=40, help='早停patience')
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小')
    parser.add_argument('--num-mc-samples', type=int, default=30, help='MC采样次数')
    parser.add_argument('--bstgcn-only', action='store_true',
                        help='只训练BSTGCN（跳过STGCN基线）')
    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'num_mc_samples': args.num_mc_samples,

        # 通用参数 - 优化后的超参数
        'learning_rate': 0.001,  # 降低学习率以稳定训练
        'weight_decay': 1e-4,
        'label_smoothing': 0.05,
        'dropout': 0.2,          # 降低dropout
        'k_neighbors': 10,

        # BSTGCN特有参数 - 尝试更高KL权重
        'hidden_dim': 128,
        'num_heads': 8,
        'num_st_blocks': 3,
        'prior_std': 0.5,
        'kl_weight_max': 5e-4,   # 进一步提高KL权重
        'kl_warmup_epochs': 15   # 更早引入KL正则
    }

    run_ood_baseline_comparison(config, bstgcn_only=args.bstgcn_only)