"""
训练改进版 Hybrid 模型

关键改进：
1. 并行时空交互（ST-Block）
2. 修正的 GATv2 实现
3. 静态+动态混合图（带缓存）
4. 门控注意力池化
5. Label Smoothing
6. Cosine LR with Warmup
7. 可选的 SWA/EMA
"""

import os
import sys
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# CUDNN 设置
# 训练阶段：deterministic=True 保证复现，benchmark=False
# 最终跑分阶段：可改为 benchmark=True 加速（牺牲复现性）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Torch 2.0+ 可以设置矩阵乘法精度（加速）
# torch.set_float32_matmul_precision("high")  # 需要 Torch 2.0+

# 导入改进的模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.hybrid_model.hybrid_model_improved import ImprovedHybridModel


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing 交叉熵损失"""
    def __init__(self, epsilon=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, logits, target):
        """
        logits: (batch, num_classes)
        target: (batch,)

        修正：先计算 smoothed loss，再用目标类权重做样本加权
        避免对非目标类的平滑质量也加权，保持校准
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot encoding with smoothing (不加权)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)

        # 计算逐样本 loss
        loss_per_sample = -torch.sum(true_dist * log_probs, dim=-1)

        # 用目标类权重做样本加权
        if self.weight is not None:
            sample_weights = self.weight[target]  # (batch,)
            loss_per_sample = loss_per_sample * sample_weights

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine LR with Warmup (修正版 - 避免 off-by-one)"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 修正：使用 last_epoch + 1 避免第一个 epoch 后 LR 掉到 0
        current_epoch = self.last_epoch + 1

        if current_epoch <= self.warmup_epochs:
            # Warmup phase: 从 0 线性增长到 base_lr
            if self.warmup_epochs == 0:
                alpha = 1.0
            else:
                alpha = current_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)  # 防止超过 1.0
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]


class PacketSequenceDataset(Dataset):
    """包序列数据集"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_data(data_dir='data/processed/raw_packet_sequences'):
    """加载数据"""
    print(f"加载数据: {data_dir}")
    data_dir = Path(data_dir)

    train_sequences = np.load(data_dir / 'train_sequences.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    val_sequences = np.load(data_dir / 'val_sequences.npy')
    val_labels = np.load(data_dir / 'val_labels.npy')
    test_sequences = np.load(data_dir / 'test_sequences.npy')
    test_labels = np.load(data_dir / 'test_labels.npy')

    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"训练样本: {len(train_sequences)}")
    print(f"验证样本: {len(val_sequences)}")
    print(f"测试样本: {len(test_sequences)}")
    print(f"序列长度: {train_sequences.shape[1]}")
    print(f"特征数: {train_sequences.shape[2]}")
    print(f"类别数: {metadata['num_classes']}")

    return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels, metadata


def get_class_weights(labels, num_classes):
    """计算类别权重"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(weights)


def count_parameters(model):
    """统计参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=True):
    """训练一个 epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        scaler: GradScaler for AMP (可选)
        use_amp: 是否使用混合精度训练
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        # 混合精度训练
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(sequences)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1, all_preds, all_labels


def benchmark_inference(model, dataloader, device, num_runs=3):
    """测试推理速度"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            _ = model(sequences)
            break

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    total_samples = 0
    with torch.no_grad():
        for _ in range(num_runs):
            for sequences, _ in dataloader:
                sequences = sequences.to(device)
                _ = model(sequences)
                total_samples += sequences.size(0)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time

    throughput = total_samples / elapsed
    latency = elapsed / total_samples * 1000  # ms per sample

    return throughput, latency


def save_results(test_preds, test_labels, class_names, train_time,
                best_val_f1, best_epoch, test_acc, test_f1, output_dir,
                inference_throughput=None, inference_latency=None):
    """保存结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    weighted_f1 = f1_score(test_labels, test_preds, average='weighted')

    report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)

    with open(output_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("改进版 Hybrid 模型评估报告 (50包实验)\n")
        f.write("="*80 + "\n\n")

        f.write("关键改进:\n")
        f.write("  1. 并行时空交互（ST-Block）\n")
        f.write("  2. 修正的 GATv2（真正的多头注意力）\n")
        f.write("  3. 静态+动态混合图（带缓存）\n")
        f.write("  4. 门控注意力池化\n")
        f.write("  5. Label Smoothing + Cosine LR\n")
        f.write("  6. 混合精度训练 (AMP)\n\n")

        f.write(f"数据划分: 70:15:15 (train:val:test)\n")
        f.write(f"随机种子: {RANDOM_SEED}\n\n")

        f.write(f"最佳验证F1: {best_val_f1:.6f} (Epoch {best_epoch})\n\n")

        f.write("测试集结果:\n")
        f.write(f"  Accuracy:           {accuracy:.6f}\n")
        f.write(f"  Macro F1:           {macro_f1:.6f}\n")
        f.write(f"  Weighted F1:        {weighted_f1:.6f}\n\n")

        f.write(f"训练时间: {train_time:.2f}秒\n")

        if inference_throughput is not None:
            f.write(f"\n推理性能:\n")
            f.write(f"  吞吐量: {inference_throughput:.2f} samples/sec\n")
            f.write(f"  延迟: {inference_latency:.4f} ms/sample\n")

        f.write("\n分类报告:\n")
        f.write(report)

    np.save(output_dir / 'predictions.npy', test_preds)

    print(f"\n结果已保存到: {output_dir}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    if inference_throughput is not None:
        print(f"\n推理性能:")
        print(f"  吞吐量: {inference_throughput:.2f} samples/sec")
        print(f"  延迟: {inference_latency:.4f} ms/sample")


def save_config(config, save_dir):
    """保存配置"""
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ config.json")


def main():
    """主函数"""
    print("="*80)
    print("训练改进版 Hybrid 模型")
    print("="*80)

    # 配置
    DATA_DIR = 'data/processed/raw_packet_sequences_50packets'  # 使用50包数据
    SAVE_DIR = 'baseline/results/hybrid_model_improved_50packets'
    BATCH_SIZE = 128  # 增大batch size以提高GPU利用率
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 20
    WARMUP_EPOCHS = 5
    LABEL_SMOOTHING = 0.05
    USE_AMP = True  # 使用混合精度训练

    # 模型参数
    HIDDEN_DIM = 128
    NUM_HEADS = 8
    NUM_ST_BLOCKS = 3
    DROPOUT = 0.2
    K_NEIGHBORS = 10

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 保存配置
    config = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'patience': PATIENCE,
        'warmup_epochs': WARMUP_EPOCHS,
        'label_smoothing': LABEL_SMOOTHING,
        'hidden_dim': HIDDEN_DIM,
        'num_heads': NUM_HEADS,
        'num_st_blocks': NUM_ST_BLOCKS,
        'dropout': DROPOUT,
        'k_neighbors': K_NEIGHBORS,
        'random_seed': RANDOM_SEED,
        'data_split': '70:15:15 (train:val:test)'
    }
    save_config(config, SAVE_DIR)

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载数据
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels, metadata = \
        load_data(DATA_DIR)

    # 创建数据集
    train_dataset = PacketSequenceDataset(train_sequences, train_labels)
    val_dataset = PacketSequenceDataset(val_sequences, val_labels)
    test_dataset = PacketSequenceDataset(test_sequences, test_labels)

    # 优化 DataLoader（pin_memory 加速 CPU->GPU 传输）
    # 注意：Windows上num_workers>0可能导致问题，但可以尝试2-4
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # 尝试使用多进程加载数据
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # 丢弃最后不完整的 batch，保持时序稳定
        persistent_workers=True  # 保持worker进程，避免重复创建
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # 验证时可以用更大的batch
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,  # 测试时可以用更大的batch
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    # 创建模型
    model = ImprovedHybridModel(
        input_dim=metadata['num_features'],
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_st_blocks=NUM_ST_BLOCKS,
        num_classes=metadata['num_classes'],
        dropout=DROPOUT,
        k_neighbors=K_NEIGHBORS
    ).to(device)

    print(f"\n模型参数数量: {count_parameters(model):,}")

    # 损失函数和优化器
    class_weights = get_class_weights(train_labels, metadata['num_classes']).to(device)
    criterion = LabelSmoothingCrossEntropy(epsilon=LABEL_SMOOTHING, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, min_lr=1e-6)

    # 混合精度训练 (AMP)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and torch.cuda.is_available() else None
    if scaler:
        print("  ✓ 启用混合精度训练 (AMP)")

    # 训练
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0

    print(f"\n开始训练 (epochs={NUM_EPOCHS}, patience={PATIENCE})...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # 每个 epoch 开始时重置图缓存
        model.reset_graph_cache()

        # 训练
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=USE_AMP
        )

        # 验证
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f} | "
                  f"LR: {new_lr:.6f}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= PATIENCE:
            print(f"\n早停触发 (patience={PATIENCE})")
            break

    train_time = time.time() - start_time

    print(f"\n训练完成！")
    print(f"训练时间: {train_time:.2f}秒")
    print(f"最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    # 测试集评估
    print("\n" + "="*80)
    print("最终测试集评估".center(80))
    print("="*80)

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    test_loss, test_acc, test_f1, test_preds, test_labels_list = \
        evaluate(model, test_loader, criterion, device)

    # 推理性能测试
    print("\n" + "="*80)
    print("推理性能测试".center(80))
    print("="*80)

    throughput, latency = benchmark_inference(model, test_loader, device, num_runs=3)
    print(f"吞吐量: {throughput:.2f} samples/sec")
    print(f"延迟: {latency:.4f} ms/sample")

    save_results(test_preds, test_labels_list, metadata['class_names'],
                train_time, best_val_f1, best_epoch, test_acc, test_f1, SAVE_DIR,
                inference_throughput=throughput, inference_latency=latency)

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

