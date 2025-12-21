"""
训练优化版 STGCN 模型

优化点：
1. 降低维度扩展：21→64 (约3x，替代原来的21→128即6x)
2. 增强类别加权：对召回率低的类别增加权重
3. 增加ST-Block数量：3→4 (利用节省的参数)
4. Focal Loss替代Label Smoothing (更好处理类别不平衡)
"""

import os
import sys
import argparse
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# 随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.hybrid_model.hybrid_model_improved import GraphSAGEHybridModel


class FocalLoss(nn.Module):
    """Focal Loss - 更好处理类别不平衡"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCE(nn.Module):
    """Label Smoothing Cross Entropy with class weights"""
    def __init__(self, smoothing=0.05, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, targets):
        n_classes = logits.size(1)

        # One-hot with smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)

        if self.weight is not None:
            # Apply class weights
            per_sample_weight = self.weight[targets]
            loss = -torch.sum(smooth_targets * log_probs, dim=-1) * per_sample_weight
        else:
            loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        return loss.mean()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine退火 + Warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (base_lr - self.min_lr) * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                    for base_lr in self.base_lrs]


class PacketSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_data(data_dir):
    """加载数据"""
    print(f"加载数据: {data_dir}")
    data_dir = Path(data_dir)
    
    train_seq = np.load(data_dir / 'train_sequences.npy')
    train_lbl = np.load(data_dir / 'train_labels.npy')
    val_seq = np.load(data_dir / 'val_sequences.npy')
    val_lbl = np.load(data_dir / 'val_labels.npy')
    test_seq = np.load(data_dir / 'test_sequences.npy')
    test_lbl = np.load(data_dir / 'test_labels.npy')
    
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"  Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
    print(f"  Shape: {train_seq.shape[1]} packets × {train_seq.shape[2]} features")
    print(f"  Classes: {metadata['num_classes']}")
    
    return train_seq, train_lbl, val_seq, val_lbl, test_seq, test_lbl, metadata


def get_class_weights(labels, num_classes, boost_classes=None, boost_factor=1.5, use_sqrt=True):
    """计算类别权重，可对特定类别增加权重"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    # 使用平方根平滑，避免极端权重
    if use_sqrt:
        weights = np.sqrt(total / (class_counts + 1e-6))
    else:
        weights = total / (num_classes * class_counts + 1e-6)

    # 对召回率低的类别增加权重
    if boost_classes and boost_factor > 1.0:
        for cls_idx in boost_classes:
            weights[cls_idx] *= boost_factor

    # 归一化
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(seqs)
            loss = criterion(logits, labels)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        logits = model(seqs)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), acc, f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='50packets_optimized',
                       choices=['50packets_optimized', '50packets_no_leak', '50packets_23feat'])
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏维度(默认64)')
    parser.add_argument('--num_st_blocks', type=int, default=4, help='ST-Block数量(默认4)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--boost_factor', type=float, default=2.0, help='弱类别权重提升因子')
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'label_smoothing'],
                       help='损失函数类型')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing参数')
    args = parser.parse_args()

    # 路径设置
    loss_tag = 'ls' if args.loss == 'label_smoothing' else 'fl'
    if args.data == '50packets_optimized':
        DATA_DIR = 'data/processed/raw_packet_sequences_50packets_optimized'
        SAVE_DIR = f'baseline/results/hybrid_opt_h{args.hidden_dim}_{loss_tag}'
    elif args.data == '50packets_23feat':
        DATA_DIR = 'data/processed/raw_packet_sequences_50packets_23feat'
        SAVE_DIR = f'baseline/results/hybrid_opt_23feat_h{args.hidden_dim}_{loss_tag}'
    else:
        DATA_DIR = 'data/processed/raw_packet_sequences_50packets_no_leak'
        SAVE_DIR = f'baseline/results/hybrid_opt_25feat_h{args.hidden_dim}_{loss_tag}'

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("="*80)
    print(f"训练优化版 STGCN - hidden_dim={args.hidden_dim}, st_blocks={args.num_st_blocks}")
    print("="*80)

    # 配置
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 20
    WARMUP_EPOCHS = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    train_seq, train_lbl, val_seq, val_lbl, test_seq, test_lbl, metadata = load_data(DATA_DIR)

    train_dataset = PacketSequenceDataset(train_seq, train_lbl)
    val_dataset = PacketSequenceDataset(val_seq, val_lbl)
    test_dataset = PacketSequenceDataset(test_seq, test_lbl)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                             num_workers=0, pin_memory=True)

    # 类别权重 - 对召回率低的类别增加权重
    # SYNScan(4), TCPConnectScan(6), UDPScan(8) 召回率较低
    boost_classes = [4, 6, 8]
    class_weights = get_class_weights(train_lbl, metadata['num_classes'],
                                       boost_classes, args.boost_factor)
    class_weights = class_weights.to(device)

    print(f"\n类别权重 (boost_classes={boost_classes}, factor={args.boost_factor}):")
    for i, name in enumerate(metadata['class_names']):
        print(f"  {name}: {class_weights[i].item():.3f}")

    # 模型
    input_dim = train_seq.shape[2]
    model = GraphSAGEHybridModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=4 if args.hidden_dim == 64 else 8,  # 适配hidden_dim
        num_st_blocks=args.num_st_blocks,
        num_classes=metadata['num_classes'],
        dropout=0.2,
        k_neighbors=10,
        sage_aggr='mean'
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {num_params:,}")
    print(f"维度扩展: {input_dim} → {args.hidden_dim} ({args.hidden_dim/input_dim:.1f}x)")

    # 损失函数
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"损失函数: Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = LabelSmoothingCE(smoothing=args.label_smoothing, weight=class_weights)
        print(f"损失函数: Label Smoothing CE (smoothing={args.label_smoothing})")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # 训练
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0

    print(f"\n开始训练 (epochs={NUM_EPOCHS}, patience={PATIENCE})...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
                  f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f} | "
                  f"Val: Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    print(f"\n训练完成! 用时: {train_time:.1f}s, 最佳Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    # 测试
    print("\n" + "="*80)
    print("测试集评估")
    print("="*80)

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    test_loss, test_acc, test_f1, test_preds, test_labels_arr = evaluate(
        model, test_loader, criterion, device
    )

    test_wf1 = f1_score(test_labels_arr, test_preds, average='weighted')

    print(f"\n  Accuracy:    {test_acc:.4f}")
    print(f"  Macro F1:    {test_f1:.4f}")
    print(f"  Weighted F1: {test_wf1:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(test_labels_arr, test_preds,
                                target_names=metadata['class_names'], digits=4))

    # 推理性能
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, train_seq.shape[1], input_dim).to(device)
        for _ in range(10):
            _ = model(dummy)

        torch.cuda.synchronize()
        start = time.time()
        n_iters = 100
        batch_size = 128
        dummy_batch = torch.randn(batch_size, train_seq.shape[1], input_dim).to(device)
        for _ in range(n_iters):
            _ = model(dummy_batch)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        throughput = (n_iters * batch_size) / elapsed
        latency = elapsed / (n_iters * batch_size) * 1000

    print(f"\n推理性能:")
    print(f"  吞吐量: {throughput:,.0f} samples/sec")
    print(f"  延迟: {latency:.4f} ms/sample")

    # 保存结果
    config = {
        'model': 'STGCN Optimized',
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'num_st_blocks': args.num_st_blocks,
        'num_params': num_params,
        'focal_gamma': args.focal_gamma,
        'boost_classes': boost_classes,
        'boost_factor': args.boost_factor,
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_f1),
        'test_weighted_f1': float(test_wf1),
        'best_val_f1': float(best_val_f1),
        'best_epoch': best_epoch,
        'train_time': train_time,
        'throughput': throughput,
        'latency_ms': latency
    }

    with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    np.save(os.path.join(SAVE_DIR, 'predictions.npy'), test_preds)

    # 保存报告
    with open(os.path.join(SAVE_DIR, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"STGCN 优化版评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"配置:\n")
        f.write(f"  hidden_dim: {args.hidden_dim}\n")
        f.write(f"  num_st_blocks: {args.num_st_blocks}\n")
        f.write(f"  focal_gamma: {args.focal_gamma}\n")
        f.write(f"  boost_classes: {boost_classes}\n")
        f.write(f"  boost_factor: {args.boost_factor}\n")
        f.write(f"  参数量: {num_params:,}\n\n")
        f.write(f"测试结果:\n")
        f.write(f"  Accuracy: {test_acc:.4f}\n")
        f.write(f"  Macro F1: {test_f1:.4f}\n")
        f.write(f"  Weighted F1: {test_wf1:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(classification_report(test_labels_arr, test_preds,
                                      target_names=metadata['class_names'], digits=4))

    print(f"\n结果已保存到: {SAVE_DIR}")


if __name__ == '__main__':
    main()

