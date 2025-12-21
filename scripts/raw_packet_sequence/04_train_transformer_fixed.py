"""
训练基于Transformer的原始包序列分类模型 (修复版)

修复问题:
1. 使用独立的验证集进行早停和学习率调度
2. 测试集只在最后评估一次
3. 添加随机种子确保可复现
4. 记录完整的训练配置
"""
import os
import sys
import numpy as np
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import model
try:
    from scripts.raw_packet_sequence.transformer_model_03 import (
        PacketTransformer, EnhancedPacketTransformer, LSTMAttentionModel, count_parameters
    )
except:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "transformer_model", 
        "scripts/raw_packet_sequence/03_transformer_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PacketTransformer = module.PacketTransformer
    EnhancedPacketTransformer = module.EnhancedPacketTransformer
    LSTMAttentionModel = module.LSTMAttentionModel
    count_parameters = module.count_parameters

print("="*100)
print("Train Transformer Model (Fixed Version - No Test Leakage)".center(100))
print("="*100)


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
    """加载数据（包含验证集）"""
    print(f"\n加载数据: {data_dir}")
    
    X_train = np.load(os.path.join(data_dir, 'train_sequences.npy'))
    X_val = np.load(os.path.join(data_dir, 'val_sequences.npy'))
    X_test = np.load(os.path.join(data_dir, 'test_sequences.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  验证集: {X_val.shape}, 标签: {y_val.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"  特征维度: {metadata['num_features']}")
    print(f"  类别数量: {metadata['num_classes']}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
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
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, macro_f1, weighted_f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_dir, patience=20):
    """训练模型（使用验证集进行早停）"""
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n开始训练 (epochs={num_epochs}, patience={patience})...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_f1, val_weighted_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # 学习率调度（基于验证集loss）
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 打印进度
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型（基于验证集F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发 (patience={patience})")
            break
    
    print(f"\n训练完成！")
    print(f"最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    
    return best_val_f1, best_epoch


def save_config(config, save_dir):
    """保存训练配置"""
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ config.json")


def main():
    """主函数"""
    # Configuration
    DATA_DIR = 'data/processed/raw_packet_sequences'
    SAVE_DIR = 'baseline/results/transformer_raw_fixed'
    MODEL_TYPE = 'enhanced'  # 'basic', 'enhanced', 'lstm'
    
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01
    PATIENCE = 20
    
    # Model parameters
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.2
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 保存配置
    config = {
        'model_type': MODEL_TYPE,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'patience': PATIENCE,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'num_layers': NUM_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'random_seed': RANDOM_SEED,
        'data_split': '70:15:15 (train:val:test)'
    }
    save_config(config, SAVE_DIR)
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Save directory: {SAVE_DIR}")
    print(f"  Model type: {MODEL_TYPE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Data split: 70:15:15 (train:val:test)")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_data(DATA_DIR)
    
    # Create datasets
    train_dataset = PacketSequenceDataset(X_train, y_train)
    val_dataset = PacketSequenceDataset(X_val, y_val)
    test_dataset = PacketSequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # Create model
    num_features = metadata['num_features']
    num_classes = metadata['num_classes']
    
    if MODEL_TYPE == 'basic':
        model = PacketTransformer(num_features, num_classes, d_model=D_MODEL, 
                                 nhead=NHEAD, num_layers=NUM_LAYERS,
                                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT)
    elif MODEL_TYPE == 'enhanced':
        model = EnhancedPacketTransformer(num_features, num_classes, d_model=D_MODEL,
                                         nhead=NHEAD, num_layers=NUM_LAYERS,
                                         dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT)
    else:  # lstm
        model = LSTMAttentionModel(num_features, num_classes, hidden_dim=D_MODEL,
                                  num_layers=NUM_LAYERS, dropout=DROPOUT)
    
    model = model.to(device)
    
    print(f"\n模型: {MODEL_TYPE}")
    print(f"  参数数量: {count_parameters(model):,}")
    
    # Loss function with class weights
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=10, min_lr=1e-6)
    
    # Train
    train_start = time.time()
    best_val_f1, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS, device, SAVE_DIR, PATIENCE
    )
    train_time = time.time() - train_start
    
    print(f"\n训练时间: {train_time:.2f}秒")
    
    # Load best model and evaluate on test set (ONLY ONCE!)
    print("\n" + "="*100)
    print("最终测试集评估 (只评估一次)".center(100))
    print("="*100)
    
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Save results
    report = classification_report(test_labels, test_preds, 
                                   target_names=metadata['class_names'], digits=4)
    
    with open(os.path.join(SAVE_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("Transformer Model Evaluation Report (Fixed - No Test Leakage)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data Split: 70:15:15 (train:val:test)\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        f.write(f"Best Validation F1: {best_val_f1:.6f} (Epoch {best_epoch})\n\n")
        f.write(f"Test Set Results:\n")
        f.write(f"  Accuracy:    {test_acc:.6f}\n")
        f.write(f"  Macro F1:    {test_macro_f1:.6f}\n")
        f.write(f"  Weighted F1: {test_weighted_f1:.6f}\n\n")
        f.write(f"Training Time: {train_time:.2f}秒\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    np.save(os.path.join(SAVE_DIR, 'predictions.npy'), test_preds)
    np.save(os.path.join(SAVE_DIR, 'confusion_matrix.npy'), 
            confusion_matrix(test_labels, test_preds))
    
    print("\n" + "="*100)
    print("Training Complete!".center(100))
    print("="*100)
    print(f"\n最终结果 (测试集):")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  Macro F1:    {test_macro_f1:.4f}")
    print(f"  Weighted F1: {test_weighted_f1:.4f}")
    print(f"  训练时间:    {train_time:.2f}秒")
    print(f"\n结果保存在: {SAVE_DIR}")


if __name__ == "__main__":
    main()

