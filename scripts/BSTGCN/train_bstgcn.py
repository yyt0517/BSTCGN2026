"""
训练贝叶斯时空图神经网络 (BSTGCN)

关键特性:
1. 贝叶斯推断框架
2. ELBO损失 (重建损失 + KL散度)
3. 蒙特卡洛推理
4. 不确定性量化
5. KL权重退火
6. 混合精度训练

用法:
    python train_bstgcn.py --data 50packets_no_leak  # 默认，25维特征
    python train_bstgcn.py --data 50packets_23feat   # 23维优化特征
    python train_bstgcn.py --data 100packets         # 100 packets数据
"""

import os
import sys
import argparse
import numpy as np
import pickle
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
  
# CUDNN设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 导入BSTGCN组件
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.BSTGCN.bstgcn_model import BSTGCNModel
from scripts.BSTGCN.elbo_loss import ELBOLoss, KLWeightScheduler
from scripts.BSTGCN.mc_inference import mc_inference, batch_mc_inference, adaptive_mc_inference
from scripts.BSTGCN.utils import (
    CosineWarmupScheduler,
    get_class_weights,
    evaluate_metrics,
    plot_confusion_matrix,
    plot_uncertainty_distribution
)


class PacketSequenceDataset(Dataset):
    """包序列数据集"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_data(data_dir='data/processed/raw_packet_sequences_50packets'):
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


def train_epoch_bayesian(model, dataloader, elbo_loss, kl_scheduler, optimizer, device,
                         scaler=None, use_amp=True, train_mc_samples=1):
    """
    训练一个epoch (贝叶斯)

    Args:
        model: BSTGCN模型
        dataloader: 数据加载器
        elbo_loss: ELBO损失函数
        kl_scheduler: KL权重调度器
        optimizer: 优化器
        device: 设备
        scaler: 混合精度scaler
        use_amp: 是否使用混合精度
        train_mc_samples: 训练时前向采样次数 (多次采样可减少梯度方差)

    Returns:
        avg_loss: 平均总损失
        avg_rec: 平均重建损失
        avg_kl: 平均KL散度
        f1: Macro F1分数
    """
    model.train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    all_preds = []
    all_labels = []

    # 更新KL权重
    current_kl_weight = kl_scheduler.get_kl_weight()
    elbo_loss.kl_weight = current_kl_weight

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # 混合精度训练
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                if train_mc_samples > 1:
                    # 多次采样取平均logits (减少梯度方差)
                    logits_list = []
                    for _ in range(train_mc_samples):
                        logits_list.append(model(sequences, sample=True))
                    logits = torch.stack(logits_list, dim=0).mean(dim=0)
                else:
                    # 单次采样前向传播
                    logits = model(sequences, sample=True)

                # ELBO损失
                loss, rec_loss, kl_loss = elbo_loss(logits, labels)

            # 检查NaN (在backward之前)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到NaN/Inf损失! Rec: {rec_loss.item():.4f}, KL: {kl_loss.item():.4f}")
                continue

            scaler.scale(loss).backward()

            # 使用scaler内置的inf/nan检测，自动跳过不稳定的更新
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step会自动检测inf梯度并跳过更新
            scaler.step(optimizer)
            scaler.update()

            # 如果梯度为inf/nan，跳过统计
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                continue
        else:
            if train_mc_samples > 1:
                logits_list = []
                for _ in range(train_mc_samples):
                    logits_list.append(model(sequences, sample=True))
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
            else:
                logits = model(sequences, sample=True)

            loss, rec_loss, kl_loss = elbo_loss(logits, labels)

            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到NaN/Inf损失! Rec: {rec_loss.item():.4f}, KL: {kl_loss.item():.4f}")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"警告: 检测到NaN/Inf梯度! 跳过此batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

        total_loss += loss.item()
        total_rec += rec_loss.item()
        total_kl += kl_loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    avg_rec = total_rec / num_batches if num_batches > 0 else float('nan')
    avg_kl = total_kl / num_batches if num_batches > 0 else float('nan')

    if len(all_labels) > 0:
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        f1 = float('nan')

    return avg_loss, avg_rec, avg_kl, f1


def evaluate_deterministic(model, dataloader, device):
    """
    确定性评估 (使用均值权重，不采样)

    用于训练期间的稳定验证，避免MC采样带来的方差
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            # 使用sample=False，使用均值权重进行推理
            logits = model(sequences, sample=False)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, f1, all_preds, all_labels


def evaluate_bayesian(model, dataloader, device, num_mc_samples=30, use_adaptive=False):
    """
    评估贝叶斯模型 (使用MC推理)

    Args:
        model: BSTGCN模型
        dataloader: 数据加载器
        device: 设备
        num_mc_samples: MC采样次数
        use_adaptive: 是否使用自适应MC采样

    Returns:
        accuracy: 准确率
        f1: Macro F1分数
        mean_uncertainty: 平均不确定性
        all_preds: 所有预测
        all_uncertainties: 所有不确定性
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_uncertainties = []
    total_samples_used = 0
    num_batches = 0

    for sequences, labels in dataloader:
        sequences = sequences.to(device)

        # MC推理
        if use_adaptive:
            # 自适应采样：初始3次，最多8次
            pred_mean, pred_class, uncertainty, samples_used = adaptive_mc_inference(
                model, sequences,
                min_samples=3,
                max_samples=num_mc_samples,
                confidence_threshold=0.95,
                uncertainty_threshold=0.05,
                use_stable_entropy=True
            )
            total_samples_used += samples_used
        else:
            # 固定采样
            pred_mean, pred_class, uncertainty = mc_inference(
                model, sequences, num_mc_samples, use_stable_entropy=True
            )
            total_samples_used += num_mc_samples

        all_preds.extend(pred_class.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_uncertainties.extend(uncertainty.cpu().numpy())
        num_batches += 1

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    mean_uncertainty = np.mean(all_uncertainties)
    avg_samples_used = total_samples_used / num_batches if num_batches > 0 else num_mc_samples

    return accuracy, f1, mean_uncertainty, all_preds, all_uncertainties, all_labels, avg_samples_used


def evaluate_fast(model, dataloader, elbo_loss, device):
    """
    快速评估 (单次采样，不计算不确定性)

    用于训练过程中的快速验证
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # 单次采样
            logits = model(sequences, sample=True)
            loss, _, _ = elbo_loss(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    """主训练函数"""

    # ==================== 命令行参数 ====================
    parser = argparse.ArgumentParser(description='训练BSTGCN模型')
    parser.add_argument('--data', type=str, default='50packets_23feat',
                        choices=['50packets', '100packets', '50packets_no_leak', '50packets_23feat'],
                        help='数据集选择: 50packets, 100packets, 50packets_no_leak, 50packets_23feat')
    args = parser.parse_args()

    # 根据选择设置数据路径
    if args.data == '100packets':
        data_dir = 'data/processed/raw_packet_sequences'
        save_dir_base = 'baseline/results/bstgcn_100packets'
        print("=" * 60)
        print("使用 100-packet 数据集")
        print("=" * 60)
    elif args.data == '50packets_no_leak':
        data_dir = 'data/processed/raw_packet_sequences_50packets_no_leak'
        save_dir_base = 'baseline/results/bstgcn_50packets_no_leak'
        print("=" * 60)
        print("使用 50-packet 无泄露数据集 (25维)")
        print("=" * 60)
    elif args.data == '50packets_23feat':
        data_dir = 'data/processed/raw_packet_sequences_50packets_23feat'
        save_dir_base = 'baseline/results/bstgcn_50packets_23feat'
        print("=" * 60)
        print("使用 50-packet 优化特征数据集 (23维)")
        print("=" * 60)
    else:
        data_dir = 'data/processed/raw_packet_sequences_50packets'
        save_dir_base = 'baseline/results/bstgcn_50packets'
        print("=" * 60)
        print("使用 50-packet 数据集")
        print("=" * 60)

    # ==================== 配置 ====================
    config = {
        # 数据
        'data_dir': data_dir,
        'save_dir_base': save_dir_base,  # 基础目录

        # 模型
        'hidden_dim': 128,
        'num_heads': 8,
        'num_st_blocks': 3,
        'dropout': 0.15,
        'k_neighbors': 10,
        'sage_aggr': 'mean',

        # 贝叶斯参数 (优化版v2 - 更接近STGCN性能)
        'prior_std': 1.5,           # 更大的先验std，减少正则化
        'kl_weight_max': 2e-5,      # 进一步降低KL权重
        'kl_warmup_epochs': 80,     # 更长的warmup
        'num_mc_samples_train': 30,
        'num_mc_samples': 50,
        'use_deterministic_val': True,
        'use_stable_entropy': True,
        'train_mc_samples': 1,
        'auroc_eval_freq': 5,

        # 关闭Hybrid初始化 (架构不兼容)
        'init_from_hybrid': False,

        # 训练 (优化版v2)
        'batch_size': 128,          # 减小batch size，增加梯度噪声
        'num_epochs': 200,          # 增加训练轮数
        'learning_rate': 0.0008,    # 稍微提高学习率
        'weight_decay': 1e-5,       # 进一步减小weight decay
        'patience': 40,             # 增加耐心
        'warmup_epochs': 15,        # 延长warmup
        'label_smoothing': 0.05,    # 减小label smoothing
        'grad_clip': 1.5,           # 增大梯度裁剪阈值
        'scan_class_boost': 1.5,    # 更强的扫描类权重提升
        'lr_min_factor': 0.02,      # 允许更小的学习率

        # 梯度监控
        'grad_norm_threshold': 8.0, # 提高阈值
        'nan_patience': 5,          # NaN容忍次数

        # 模型选择策略
        'selection_metric': 'f1',
        'auroc_weight': 0.3,
        'kl_threshold': 0.0001,
        'stage_epoch': 30,

        # 其他
        'use_amp': True,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(config['save_dir_base']) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    config['save_dir'] = str(save_dir)  # 添加完整路径到配置中

    print("=" * 80)
    print("BSTGCN训练配置")
    print("=" * 80)
    for key, value in config.items():
        print(f"{key:25s}: {value}")
    print("=" * 80)
    print(f"\n结果将保存到: {save_dir}\n")

    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # ==================== 加载数据 ====================
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels, metadata = load_data(
        config['data_dir']
    )

    # 创建数据集和加载器
    train_dataset = PacketSequenceDataset(train_sequences, train_labels)
    val_dataset = PacketSequenceDataset(val_sequences, val_labels)
    test_dataset = PacketSequenceDataset(test_sequences, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # ==================== 创建模型 ====================
    device = torch.device(config['device'])

    model = BSTGCNModel(
        input_dim=metadata['num_features'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_st_blocks=config['num_st_blocks'],
        num_classes=metadata['num_classes'],
        dropout=config['dropout'],
        k_neighbors=config['k_neighbors'],
        sage_aggr=config['sage_aggr'],
        prior_std=config['prior_std']
    ).to(device)

    print(f"\n模型参数数量: {model.get_num_params():,}")

    # ==================== 从Hybrid模型初始化 (可选) ====================
    if config.get('init_from_hybrid', False):
        hybrid_path = config['hybrid_model_path']
        if Path(hybrid_path).exists():
            print(f"\n从Hybrid模型加载权重: {hybrid_path}")
            hybrid_state = torch.load(hybrid_path, map_location=device)
            loaded_count = model.load_from_hybrid(
                hybrid_state,
                init_rho=config.get('init_rho', -3.0)
            )
            print(f"成功加载 {loaded_count} 个参数")
        else:
            print(f"\n警告: Hybrid模型路径不存在: {hybrid_path}")
            print("将从随机初始化开始训练")

    # ==================== 损失函数 ====================
    # 计算基础类别权重（按频率的反比）
    class_weights = get_class_weights(train_labels, metadata['num_classes'])

    # 可选：对扫描类（SYNScan / TCPConnectScan / UDPScan）额外放大权重
    scan_boost = config.get('scan_class_boost', 1.0)
    if scan_boost != 1.0:
        # 从 metadata 中解析类别名称
        if 'class_names' in metadata:
            class_names = metadata['class_names']
        elif 'label_mapping' in metadata:
            class_names = [metadata['label_mapping'][str(i)] for i in range(metadata['num_classes'])]
        else:
            class_names = None

        if class_names is not None:
            scan_class_names = ['SYNScan', 'TCPConnectScan', 'UDPScan']
            for name in scan_class_names:
                if name in class_names:
                    idx = class_names.index(name)
                    class_weights[idx] *= scan_boost

            # 重新归一化权重，使整体尺度保持稳定
            class_weights = class_weights / class_weights.sum() * metadata['num_classes']

    class_weights = class_weights.to(device)

    elbo_loss = ELBOLoss(
        model=model,
        kl_weight=config['kl_weight_max'],  # 初始值，会被调度器更新
        label_smoothing=config['label_smoothing'],
        class_weights=class_weights,
        num_batches=len(train_loader)
    )

    # KL权重调度器
    kl_scheduler = KLWeightScheduler(
        kl_weight_max=config['kl_weight_max'],
        warmup_epochs=config['kl_warmup_epochs'],
        mode='linear'
    )

    # ==================== 优化器和学习率调度器 ====================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['num_epochs'],
        min_lr_ratio=config.get('lr_min_factor', 0.1)  # 最小学习率比例
    )

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] and torch.cuda.is_available() else None

    # ==================== 训练循环 ====================
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    best_val_f1 = 0
    best_val_auroc = 0
    best_val_score = 0  # 用于combined模式
    patience_counter = 0
    nan_counter = 0  # NaN计数器
    nan_patience = config.get('nan_patience', 3)

    train_history = {
        'train_loss': [], 'train_rec': [], 'train_kl': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [], 'val_uncertainty': [], 'val_auroc': [],
        'kl_weight': [], 'lr': []
    }

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start_time = time.time()

        # 训练 (支持多次采样减少梯度方差)
        train_loss, train_rec, train_kl, train_f1 = train_epoch_bayesian(
            model, train_loader, elbo_loss, kl_scheduler, optimizer, device,
            scaler, config['use_amp'], config.get('train_mc_samples', 1)
        )

        # 检测NaN/Inf - 如果训练出现严重问题，提前停止
        if np.isnan(train_loss) or np.isnan(train_f1):
            nan_counter += 1
            print(f"警告: Epoch {epoch} 训练出现NaN! (连续{nan_counter}次)")
            if nan_counter >= nan_patience:
                print(f"\n训练不稳定，提前停止 (连续{nan_patience}次NaN)")
                break
            continue  # 跳过这个epoch的验证和记录
        else:
            nan_counter = 0  # 重置计数器

        # 验证策略：根据配置选择确定性或MC验证
        use_det_val = config.get('use_deterministic_val', False)

        if use_det_val:
            # 确定性验证 (稳定的模型选择)
            _, val_f1_det, val_preds, val_labels = evaluate_deterministic(
                model, val_loader, device
            )
            # MC验证 (根据auroc_eval_freq评估不确定性质量)
            auroc_eval_freq = config.get('auroc_eval_freq', 10)
            if epoch % auroc_eval_freq == 0 or epoch == 1:
                _, _, val_unc, mc_preds, val_uncertainties, mc_labels, _ = evaluate_bayesian(
                    model, val_loader, device, config['num_mc_samples_train'],
                    use_adaptive=False
                )
            else:
                val_unc = 0.0
                val_uncertainties = []
                mc_preds = val_preds
                mc_labels = val_labels
            val_f1 = val_f1_det
        else:
            # MC验证 (每个epoch都用MC)
            _, val_f1, val_unc, mc_preds, val_uncertainties, mc_labels, _ = evaluate_bayesian(
                model, val_loader, device, config['num_mc_samples_train'],
                use_adaptive=False
            )
            val_preds = mc_preds
            val_labels = mc_labels

        # 计算AUROC (不确定性与错误的相关性)
        val_uncertainties = np.array(val_uncertainties) if len(val_uncertainties) > 0 else np.array([0.0])
        mc_errors = (np.array(mc_preds) != np.array(mc_labels)).astype(int)
        if np.any(np.isnan(val_uncertainties)):
            val_uncertainties = np.nan_to_num(val_uncertainties, nan=0.0)
            val_auroc = float('nan')
        elif mc_errors.sum() > 0 and mc_errors.sum() < len(mc_errors) and val_uncertainties.sum() > 0:
            val_auroc = roc_auc_score(mc_errors, val_uncertainties)
        else:
            val_auroc = float('nan')
        val_loss = 0.0
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # 更新学习率和KL权重
        lr_scheduler.step()
        kl_scheduler.step()

        epoch_time = time.time() - epoch_start_time

        # 记录历史（全部转为Python float，便于JSON序列化）
        train_history['train_loss'].append(float(train_loss))
        train_history['train_rec'].append(float(train_rec))
        train_history['train_kl'].append(float(train_kl))
        train_history['train_f1'].append(float(train_f1))
        train_history['val_loss'].append(float(val_loss))
        train_history['val_f1'].append(float(val_f1))
        train_history['val_uncertainty'].append(float(val_unc))
        train_history['val_auroc'].append(float(val_auroc) if not np.isnan(val_auroc) else None)
        train_history['kl_weight'].append(float(kl_scheduler.get_kl_weight()))
        train_history['lr'].append(float(lr_scheduler.get_lr()[0]))

        # 打印日志
        auroc_str = f"{val_auroc:.4f}" if not np.isnan(val_auroc) else "N/A"
        det_str = "[DET]" if config.get('use_deterministic_val', False) else "[MC]"
        print(f"Epoch [{epoch:3d}/{config['num_epochs']}] {det_str} | "
              f"Time: {epoch_time:.1f}s | "
              f"Loss: {train_loss:.4f} (Rec: {train_rec:.4f}, KL: {train_kl:.4f}) | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
              f"Val AUROC: {auroc_str} | Val Unc: {val_unc:.4f} | "
              f"KL_w: {kl_scheduler.get_kl_weight():.4f} | "
              f"LR: {lr_scheduler.get_lr()[0]:.6f}")

        # 根据配置选择模型选择策略
        selection_metric = config.get('selection_metric', 'combined')
        save_model = False

        if selection_metric == 'f1':
            # 只用F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_model = True
        elif selection_metric == 'auroc':
            # 只用AUROC
            if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                save_model = True
        elif selection_metric == 'balanced':
            # 平衡策略：F1优先，但需要KL权重达到一定阈值
            kl_threshold = config.get('kl_threshold', 0.0001)
            current_kl_weight = kl_scheduler.get_kl_weight()

            # 如果KL权重还很小，只看F1
            if current_kl_weight < kl_threshold:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    save_model = True
            else:
                # KL权重足够大后，使用F1+AUROC组合
                if not np.isnan(val_auroc):
                    auroc_weight = config.get('auroc_weight', 0.3)
                    f1_weight = 1.0 - auroc_weight
                    normalized_auroc = (val_auroc - 0.5) * 2.0

                    val_score = f1_weight * val_f1 + auroc_weight * normalized_auroc

                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_val_f1 = val_f1
                        best_val_auroc = val_auroc
                        save_model = True
        elif selection_metric == 'staged':
            # 分阶段策略：前期看F1，后期看组合指标
            stage_epoch = config.get('stage_epoch', 30)

            if epoch <= stage_epoch:
                # 前期：只看F1
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    save_model = True
            else:
                # 后期：看F1+AUROC组合
                if not np.isnan(val_auroc):
                    auroc_weight = config.get('auroc_weight', 0.3)
                    f1_weight = 1.0 - auroc_weight
                    normalized_auroc = (val_auroc - 0.5) * 2.0

                    val_score = f1_weight * val_f1 + auroc_weight * normalized_auroc

                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_val_f1 = val_f1
                        best_val_auroc = val_auroc
                        save_model = True
        else:  # 'combined'
            # F1 + AUROC 加权组合
            if not np.isnan(val_auroc):
                auroc_weight = config.get('auroc_weight', 0.3)
                f1_weight = 1.0 - auroc_weight
                # 将AUROC从[0.5, 1.0]映射到[0, 1.0]，使其与F1尺度一致
                normalized_auroc = (val_auroc - 0.5) * 2.0
                val_score = f1_weight * val_f1 + auroc_weight * normalized_auroc

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_val_f1 = val_f1
                    best_val_auroc = val_auroc
                    save_model = True

        # 保存最佳模型
        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auroc': val_auroc if not np.isnan(val_auroc) else None,
                'config': config
            }, save_dir / 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config['patience']:
            print(f"\n早停触发 (patience={config['patience']})")
            break

    # 保存训练历史
    with open(save_dir / 'train_history.json', 'w') as f:
        json.dump(train_history, f, indent=4)

    # ==================== 测试集评估 ====================
    print("\n" + "=" * 80)
    print("测试集评估")
    print("=" * 80)

    # 加载最佳模型
    # PyTorch 2.6 之后 torch.load 默认 weights_only=True，这里需要显式设为 False
    checkpoint = torch.load(save_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 打印最佳模型信息
    best_epoch_info = f"加载最佳模型 (Epoch {checkpoint['epoch']}, Val F1: {checkpoint['val_f1']:.4f}"
    if 'val_auroc' in checkpoint and checkpoint['val_auroc'] is not None:
        best_epoch_info += f", Val AUROC: {checkpoint['val_auroc']:.4f}"
    best_epoch_info += ")"
    print(best_epoch_info)

    # ===== 确定性评估 (MAP) =====
    print("\n--- 确定性评估 (MAP推理) ---")
    test_acc_det, test_f1_det, test_preds_det, test_labels_list = evaluate_deterministic(
        model, test_loader, device
    )
    print(f"  Accuracy: {test_acc_det:.4f}")
    print(f"  Macro F1: {test_f1_det:.4f}")

    # ===== MC推理评估 =====
    print("\n--- 贝叶斯评估 (MC推理) ---")
    test_acc, test_f1, test_unc, test_preds, test_uncertainties, test_labels_list, _ = evaluate_bayesian(
        model, test_loader, device, config['num_mc_samples'],
        use_adaptive=False  # 测试集使用完整采样以获得最佳性能
    )
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")
    print(f"  Mean Uncertainty: {test_unc:.4f}")

    # 使用确定性评估结果作为主要指标，MC结果用于不确定性分析
    print(f"\n--- 主要结果 (确定性推理) ---")
    print(f"  Accuracy: {test_acc_det:.4f}")
    print(f"  Macro F1: {test_f1_det:.4f}")

    # 详细分类报告
    # 自适应选择类别名称来源
    if 'label_mapping' in metadata:
        class_names = [metadata['label_mapping'][str(i)] for i in range(metadata['num_classes'])]
    elif 'class_names' in metadata:
        class_names = metadata['class_names']
    else:
        class_names = [str(i) for i in range(metadata['num_classes'])]

    print("\n分类报告 (确定性推理):")
    print(classification_report(
        test_labels_list,
        test_preds_det,
        target_names=class_names,
        digits=4
    ))

    # 计算不确定性与错误的相关性（使用MC预测的错误）
    errors = (np.array(test_labels_list) != np.array(test_preds)).astype(float)
    uncertainty_error_corr = np.corrcoef(errors, test_uncertainties)[0, 1]
    print(f"\n不确定性与错误相关性 (MC): {uncertainty_error_corr:.4f}")

    # 计算不确定性与确定性预测错误的相关性
    errors_det = (np.array(test_labels_list) != np.array(test_preds_det)).astype(float)
    uncertainty_det_error_corr = np.corrcoef(errors_det, test_uncertainties)[0, 1]
    print(f"不确定性与错误相关性 (DET): {uncertainty_det_error_corr:.4f}")

    # 保存结果 - 主要使用确定性评估结果
    results = {
        # 主要指标 (确定性推理)
        'test_accuracy': float(test_acc_det),
        'test_macro_f1': float(test_f1_det),
        # MC推理结果
        'test_accuracy_mc': float(test_acc),
        'test_macro_f1_mc': float(test_f1),
        'test_mean_uncertainty': float(test_unc),
        'uncertainty_error_correlation_mc': float(uncertainty_error_corr),
        'uncertainty_error_correlation_det': float(uncertainty_det_error_corr),
        'num_parameters': model.get_num_params(),
        'best_epoch': checkpoint['epoch'],
        'best_val_f1': float(checkpoint['val_f1']),
        'best_val_auroc': float(checkpoint['val_auroc']) if 'val_auroc' in checkpoint and checkpoint['val_auroc'] is not None else None,
        'selection_metric': config.get('selection_metric', 'combined')
    }

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # 保存预测和不确定性
    # 注意: 使用确定性预测作为主要结果 (与论文报告的accuracy/F1一致)
    # 不确定性来自MC推理 (用于不确定性分析)
    np.save(save_dir / 'test_predictions.npy', test_preds_det)
    np.save(save_dir / 'test_uncertainties.npy', test_uncertainties)
    # 同时保存MC预测用于对比分析
    np.save(save_dir / 'test_predictions_mc.npy', test_preds)

    # ==================== 可视化 ====================
    print("\n生成可视化...")

    # 混淆矩阵 (使用确定性预测)
    plot_confusion_matrix(
        test_labels_list,
        test_preds_det,
        class_names=class_names,
        save_path=save_dir / 'confusion_matrix.png'
    )

    # 不确定性分布
    plot_uncertainty_distribution(
        np.array(test_uncertainties),
        errors,
        save_path=save_dir / 'uncertainty_distribution.png'
    )

    print(f"\n所有结果已保存到: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()