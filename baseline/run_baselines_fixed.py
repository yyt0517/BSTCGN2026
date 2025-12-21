"""
在原始包序列数据上运行基线模型 - 修复版

修复内容:
1. 数据划分: 70:15:15 (train:val:test)
2. 早停和学习率调度: 使用验证集
3. 测试集: 只在最后评估一次
4. 随机种子: 固定为42
5. 配置记录: 保存完整配置
6. 支持单独运行指定模型

包含模型:
- LSTM
- GRU
- BiLSTM
- CNN1D
- Random Forest
- XGBoost
- LightGBM

使用方法:
    # 运行所有模型
    python baseline/run_baselines_fixed.py

    # 运行单个模型
    python baseline/run_baselines_fixed.py --models lstm

    # 运行多个模型
    python baseline/run_baselines_fixed.py --models lstm gru cnn1d

    # 列出所有可用模型
    python baseline/run_baselines_fixed.py --list
"""

import os
import sys
import time
import json
import numpy as np
import pickle
import argparse
from pathlib import Path

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost未安装")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM未安装")

# 可用模型列表
AVAILABLE_MODELS = {
    'lstm': 'LSTM',
    'gru': 'GRU',
    'bilstm': 'BiLSTM',
    'cnn1d': 'CNN1D',
    'rf': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

print("="*100)
print("基线模型实验 - 修复版 (无测试集泄露)".center(100))
print("="*100)


# ==================== 模型定义 ====================

class LSTMClassifier(nn.Module):
    """LSTM分类器"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


class GRUClassifier(nn.Module):
    """GRU分类器"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
        return self.fc(out)


class BiLSTMClassifier(nn.Module):
    """BiLSTM分类器"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        out = self.dropout(bilstm_out[:, -1, :])
        return self.fc(out)


class CNN1DClassifier(nn.Module):
    """1D CNN分类器"""
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ==================== 数据加载 ====================

def load_data(data_dir='data/processed/raw_packet_sequences'):
    """加载数据"""
    data_dir = Path(data_dir)
    
    train_X = np.load(data_dir / 'train_sequences.npy')
    train_y = np.load(data_dir / 'train_labels.npy')
    val_X = np.load(data_dir / 'val_sequences.npy')
    val_y = np.load(data_dir / 'val_labels.npy')
    test_X = np.load(data_dir / 'test_sequences.npy')
    test_y = np.load(data_dir / 'test_labels.npy')
    
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"\n数据加载完成:")
    print(f"  训练集: {train_X.shape}")
    print(f"  验证集: {val_X.shape}")
    print(f"  测试集: {test_X.shape}")
    
    return train_X, train_y, val_X, val_y, test_X, test_y, metadata


def get_class_weights(labels, num_classes):
    """计算类别权重"""
    counts = np.bincount(labels, minlength=num_classes)
    weights = len(labels) / (num_classes * counts)
    return weights


# ==================== 训练和评估 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, save_path):
    """训练模型"""
    print("\n开始训练...")
    best_val_acc = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += output.argmax(1).eq(target).sum().item()
            train_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_correct += output.argmax(1).eq(target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            print(f"\n早停触发 (patience={patience})")
            break
    
    train_time = time.time() - start_time
    print(f"\n训练完成！耗时: {train_time:.2f}秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    return train_time, best_epoch


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def save_results(preds, labels, class_names, save_dir, model_name):
    """保存结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    report = classification_report(labels, preds, target_names=class_names, digits=4)

    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{model_name} Evaluation Report (Fixed - No Test Leakage)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data Split: 70:15:15 (train:val:test)\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        f.write(f"Test Set Results:\n")
        f.write(f"  Accuracy:    {acc:.6f}\n")
        f.write(f"  Macro F1:    {macro_f1:.6f}\n")
        f.write(f"  Weighted F1: {weighted_f1:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    np.save(save_dir / 'predictions.npy', preds)
    np.save(save_dir / 'confusion_matrix.npy', confusion_matrix(labels, preds))

    print(f"\n测试集结果:")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


def train_tree_model(model, train_X, train_y, val_X, val_y, model_name):
    """训练树模型（使用验证集进行早停）"""
    print("\n开始训练...")
    start_time = time.time()

    # 展平特征: (N, 100, 25) -> (N, 2500)
    train_X_flat = train_X.reshape(train_X.shape[0], -1)
    val_X_flat = val_X.reshape(val_X.shape[0], -1)

    if model_name == 'Random Forest':
        # Random Forest不支持早停，直接训练
        model.fit(train_X_flat, train_y)
        val_preds = model.predict(val_X_flat)
        val_acc = accuracy_score(val_y, val_preds)
        best_iteration = None

    elif model_name == 'XGBoost':
        # XGBoost支持早停
        model.fit(
            train_X_flat, train_y,
            eval_set=[(val_X_flat, val_y)],
            verbose=False
        )
        val_preds = model.predict(val_X_flat)
        val_acc = accuracy_score(val_y, val_preds)
        best_iteration = model.best_iteration

    elif model_name == 'LightGBM':
        # LightGBM支持早停
        model.fit(
            train_X_flat, train_y,
            eval_set=[(val_X_flat, val_y)],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        val_preds = model.predict(val_X_flat)
        val_acc = accuracy_score(val_y, val_preds)
        best_iteration = model.best_iteration_

    train_time = time.time() - start_time
    print(f"\n训练完成！耗时: {train_time:.2f}秒")
    print(f"验证集准确率: {val_acc*100:.2f}%")
    if best_iteration:
        print(f"最佳迭代: {best_iteration}")

    return train_time


def evaluate_tree_model(model, test_X, test_y):
    """评估树模型"""
    test_X_flat = test_X.reshape(test_X.shape[0], -1)
    preds = model.predict(test_X_flat)
    return preds


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行基线模型实验')
    parser.add_argument('--models', nargs='+', choices=list(AVAILABLE_MODELS.keys()) + ['all'],
                        default=['all'], help='要运行的模型 (默认: all)')
    parser.add_argument('--list', action='store_true', help='列出所有可用模型')
    parser.add_argument('--data', type=str, default='50packets_no_leak',
                        choices=['50packets', '100packets', '50packets_no_leak'],
                        help='数据集选择 (默认: 50packets_no_leak)')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 列出可用模型
    if args.list:
        print("\n可用模型:")
        for key, name in AVAILABLE_MODELS.items():
            print(f"  {key:<12} - {name}")
        print("\n使用方法:")
        print("  python baseline/run_baselines_fixed.py --models lstm gru")
        print("  python baseline/run_baselines_fixed.py --models all")
        return

    # 确定要运行的模型
    if 'all' in args.models:
        models_to_run = list(AVAILABLE_MODELS.keys())
    else:
        models_to_run = args.models

    print(f"\n将运行以下模型: {', '.join([AVAILABLE_MODELS[m] for m in models_to_run])}")

    # 选择数据路径
    if args.data == '100packets':
        data_dir = 'data/processed/raw_packet_sequences'
    elif args.data == '50packets_no_leak':
        data_dir = 'data/processed/raw_packet_sequences_50packets_no_leak'
    else:
        data_dir = 'data/processed/raw_packet_sequences_50packets'

    print(f"\n使用数据集: {data_dir}")

    # 加载数据
    train_X, train_y, val_X, val_y, test_X, test_y, metadata = load_data(data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    input_size = train_X.shape[2]

    # 类别权重
    class_weights = torch.FloatTensor(get_class_weights(train_y, num_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    results = {}

    # 模型配置
    all_models_config = {
        'lstm': (LSTMClassifier, {'input_size': input_size, 'hidden_size': 128, 'num_layers': 2, 'num_classes': num_classes}, False, 'LSTM'),
        'gru': (GRUClassifier, {'input_size': input_size, 'hidden_size': 128, 'num_layers': 2, 'num_classes': num_classes}, False, 'GRU'),
        'bilstm': (BiLSTMClassifier, {'input_size': input_size, 'hidden_size': 128, 'num_layers': 2, 'num_classes': num_classes}, False, 'BiLSTM'),
        'cnn1d': (CNN1DClassifier, {'input_size': input_size, 'num_classes': num_classes}, True, 'CNN1D'),
    }

    # 只运行选定的模型
    models_config = {k: v for k, v in all_models_config.items() if k in models_to_run}

    for model_key, (ModelClass, model_params, is_cnn, model_display_name) in models_config.items():
        print("\n" + "="*100)
        print(f"{model_display_name} 模型".center(100))
        print("="*100)

        save_dir = Path(f'baseline/results/{model_key}_raw_fixed')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        if is_cnn:
            # CNN: (N, T, D) -> (N, D, T)
            train_loader = DataLoader(TensorDataset(
                torch.FloatTensor(train_X).transpose(1, 2),
                torch.LongTensor(train_y)
            ), batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(
                torch.FloatTensor(val_X).transpose(1, 2),
                torch.LongTensor(val_y)
            ), batch_size=64, shuffle=False)
            test_loader = DataLoader(TensorDataset(
                torch.FloatTensor(test_X).transpose(1, 2),
                torch.LongTensor(test_y)
            ), batch_size=64, shuffle=False)
        else:
            train_loader = DataLoader(TensorDataset(
                torch.FloatTensor(train_X),
                torch.LongTensor(train_y)
            ), batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(
                torch.FloatTensor(val_X),
                torch.LongTensor(val_y)
            ), batch_size=64, shuffle=False)
            test_loader = DataLoader(TensorDataset(
                torch.FloatTensor(test_X),
                torch.LongTensor(test_y)
            ), batch_size=64, shuffle=False)
        
        # 创建模型
        model = ModelClass(**model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 保存配置
        config = {
            'model': model_display_name,
            'params': {k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                      for k, v in model_params.items()},
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'random_seed': RANDOM_SEED,
            'data_split': '70:15:15 (train:val:test)'
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # 训练
        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path)
        
        # 测试（只评估一次）
        print("\n最终测试集评估 (只评估一次)...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        
        # 保存结果
        metrics = save_results(test_preds, test_labels, class_names, save_dir, model_display_name)
        results[model_display_name] = metrics

    # ==================== 树模型 ====================

    # Random Forest
    if 'rf' in models_to_run:
        print("\n" + "="*100)
        print("Random Forest 模型".center(100))
        print("="*100)

        save_dir = Path('baseline/results/rf_raw_fixed')
        save_dir.mkdir(parents=True, exist_ok=True)

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=1
        )

        config = {
            'model': 'Random Forest',
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_seed': RANDOM_SEED,
            'data_split': '70:15:15 (train:val:test)'
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        train_time = train_tree_model(rf_model, train_X, train_y, val_X, val_y, 'Random Forest')

        print("\n最终测试集评估 (只评估一次)...")
        test_preds = evaluate_tree_model(rf_model, test_X, test_y)
        metrics = save_results(test_preds, test_y, class_names, save_dir, 'Random Forest')
        results['Random Forest'] = metrics

    # XGBoost
    if 'xgboost' in models_to_run and XGBOOST_AVAILABLE:
        print("\n" + "="*100)
        print("XGBoost 模型".center(100))
        print("="*100)

        save_dir = Path('baseline/results/xgboost_raw_fixed')
        save_dir.mkdir(parents=True, exist_ok=True)

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=20,
            eval_metric='mlogloss'
        )

        config = {
            'model': 'XGBoost',
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 20,
            'random_seed': RANDOM_SEED,
            'data_split': '70:15:15 (train:val:test)'
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        train_time = train_tree_model(xgb_model, train_X, train_y, val_X, val_y, 'XGBoost')

        print("\n最终测试集评估 (只评估一次)...")
        test_preds = evaluate_tree_model(xgb_model, test_X, test_y)
        metrics = save_results(test_preds, test_y, class_names, save_dir, 'XGBoost')
        results['XGBoost'] = metrics

    # LightGBM
    if 'lightgbm' in models_to_run and LIGHTGBM_AVAILABLE:
        print("\n" + "="*100)
        print("LightGBM 模型".center(100))
        print("="*100)

        save_dir = Path('baseline/results/lightgbm_raw_fixed')
        save_dir.mkdir(parents=True, exist_ok=True)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        )

        config = {
            'model': 'LightGBM',
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'random_seed': RANDOM_SEED,
            'data_split': '70:15:15 (train:val:test)'
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        train_time = train_tree_model(lgb_model, train_X, train_y, val_X, val_y, 'LightGBM')

        print("\n最终测试集评估 (只评估一次)...")
        test_preds = evaluate_tree_model(lgb_model, test_X, test_y)
        metrics = save_results(test_preds, test_y, class_names, save_dir, 'LightGBM')
        results['LightGBM'] = metrics

    # 总结
    print("\n" + "="*100)
    print("实验总结".center(100))
    print("="*100)

    print("\n| 模型 | Accuracy | Macro F1 | Weighted F1 |")
    print("|------|----------|----------|-------------|")
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        print(f"| {name:<15} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |")

    print("\n完成！")


if __name__ == '__main__':
    main()

