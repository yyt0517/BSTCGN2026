"""
综合基线模型实验 - 重新设计版

================================================================================
命名映射表 (代码 -> 论文)
================================================================================
| 代码名称   | 论文名称    | 说明                                           |
|------------|-------------|------------------------------------------------|
| STGCN      | Seq-STGCN   | 串行时空建模 (先temporal后spatial)              |
| Hybrid     | STGCN       | 并行时空建模 (temporal和spatial并行处理)        |
| BSTGCN     | BSTGCN      | 贝叶斯并行时空建模 = Bayesian + STGCN(论文)     |
================================================================================
论文命名逻辑: BSTGCN = Bayesian + STGCN, 所以STGCN是BSTGCN的确定性版本

包含11个模型:
A. 时序模型 (3个):
  1. CNN1D - 局部时序特征
  2. GRU - 全局时序依赖
  3. TCN - 因果时序建模

B. 非时序模型 (2个):
  4. LightGBM - 表格数据基线
  5. XGBoost - 梯度提升基线

C. 图结构模型 (2个):
  6. GCN - 图卷积基线
  7. GAT - 图注意力

D. 时空结合模型 (4个):
  8. LSTM+GNN - 串行时空建模
  9. STGCN - 串行时空建模 (论文中称为Seq-STGCN)
  10. Hybrid - 并行时空建模 (论文中称为STGCN，是BSTGCN的确定性版本)
  11. BSTGCN - 贝叶斯并行时空图神经网络 (不确定性量化)

数据划分: 70:15:15 (train:val:test)
随机种子: 42

使用方法:
  python baseline/run_comprehensive_baselines.py --models cnn1d gru
  python baseline/run_comprehensive_baselines.py --models stgcn
  python baseline/run_comprehensive_baselines.py --models bstgcn
  python baseline/run_comprehensive_baselines.py --models all
  python baseline/run_comprehensive_baselines.py --list
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pickle
from pathlib import Path

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Scikit-learn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM未安装")

# 导入Hybrid模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from scripts.hybrid_model.hybrid_model_01 import HybridModel
except ImportError:
    try:
        from scripts.hybrid_model.hybrid_model_improved import HybridModel
    except ImportError:
        HybridModel = None
        print("⚠️  Hybrid模型未找到，将跳过Hybrid模型训练")

# 导入BSTGCN模型
try:
    from scripts.BSTGCN.bstgcn_model import BSTGCNModel
    from scripts.BSTGCN.elbo_loss import ELBOLoss, KLWeightScheduler
    from scripts.BSTGCN.mc_inference import batch_mc_inference
    from scripts.BSTGCN.utils import get_class_weights as bstgcn_get_class_weights
    BSTGCN_AVAILABLE = True
except ImportError as e:
    BSTGCN_AVAILABLE = False
    print(f"⚠️  BSTGCN模块未找到: {e}")

# 可用模型映射
AVAILABLE_MODELS = {
    'cnn1d': 'CNN1D',
    'gru': 'GRU',
    'tcn': 'TCN',
    'lightgbm': 'LightGBM',
    'xgboost': 'XGBoost',
    'gcn': 'GCN',
    'gat': 'GAT',
    'lstm_gnn': 'LSTM+GNN',
    'stgcn': 'STGCN',
    'hybrid': 'Hybrid',
    'bstgcn': 'BSTGCN'
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行综合基线模型实验')
    parser.add_argument('--models', nargs='+',
                        choices=list(AVAILABLE_MODELS.keys()) + ['all'],
                        default=['all'],
                        help='要运行的模型 (默认: all)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用模型')
    parser.add_argument('--data', type=str, default='50packets_no_leak',
                        choices=['original', '50packets', '50packets_no_leak'],
                        help='数据集选择: original, 50packets, 50packets_no_leak (默认: 50packets_no_leak)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='结果输出目录 (默认根据数据集自动选择)')
    return parser.parse_args()

print("="*100)
print("综合基线模型实验 - 重新设计版".center(100))
print("="*100)


# ==================== 模型定义 ====================

class GRUClassifier(nn.Module):
    """GRU分类器"""
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
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
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class TCNClassifier(nn.Module):
    """TCN分类器 - 简化版时序卷积网络"""
    def __init__(self, input_size, num_classes, hidden_dim=64, dropout=0.2):
        super().__init__()
        # 使用标准卷积代替复杂的TCN块
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


class MLPClassifier(nn.Module):
    """MLP分类器 - 验证时序必要性"""
    def __init__(self, input_size, seq_len, num_classes, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_size * seq_len
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, seq_len * features)
        x = x.reshape(x.size(0), -1)
        return self.network(x)


class GCNLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_dim)
        adj: (batch, num_nodes, num_nodes)
        """
        # 归一化邻接矩阵: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)
        
        # 图卷积: H' = σ(D^{-1/2} A D^{-1/2} H W)
        x = torch.matmul(adj_norm, x)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class GCNClassifier(nn.Module):
    """GCN分类器"""
    def __init__(self, input_size, hidden_dim=128, num_layers=2, num_classes=9, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        # 特征嵌入
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # 构建K近邻图
        batch_size, num_nodes, hidden_dim = x.shape
        adj = self.build_knn_graph(x, k=10)
        
        # GCN层
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj)
        
        # 全局池化
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # 分类
        return self.classifier(x)
    
    def build_knn_graph(self, x, k=10):
        """构建K近邻图"""
        batch_size, num_nodes, hidden_dim = x.shape
        
        # 计算相似度矩阵
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))  # (batch, num_nodes, num_nodes)
        
        # Top-K
        _, indices = torch.topk(sim, k=min(k, num_nodes), dim=-1)
        
        # 构建邻接矩阵
        adj = torch.zeros_like(sim)
        adj.scatter_(-1, indices, 1.0)
        
        # 对称化
        adj = (adj + adj.transpose(-1, -2)) / 2
        
        # 添加自环
        adj = adj + torch.eye(num_nodes, device=x.device).unsqueeze(0)
        
        return adj


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


# ==================== GAT模型 ====================

class GATLayer(nn.Module):
    """图注意力层 - 简化版"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        self.W = nn.Linear(in_dim, out_dim)
        self.attn_src = nn.Linear(out_dim, num_heads)
        self.attn_dst = nn.Linear(out_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_dim)
        adj: (batch, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape

        # 特征变换
        h = self.W(x)  # (batch, num_nodes, out_dim)

        # 计算注意力分数
        attn_src = self.attn_src(h)  # (batch, num_nodes, num_heads)
        attn_dst = self.attn_dst(h)  # (batch, num_nodes, num_heads)

        # 广播计算注意力
        attn_scores = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)  # (batch, N, N, heads)

        # 应用邻接矩阵掩码
        mask = (adj > 0).unsqueeze(-1).float()  # (batch, N, N, 1)
        attn_scores = attn_scores * mask + (1 - mask) * (-1e9)

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=2)  # (batch, N, N, heads)
        attn_weights = self.dropout(attn_weights)

        # 聚合 - 简化版：对每个头使用相同的特征
        h_expanded = h.unsqueeze(1).unsqueeze(-1)  # (batch, 1, N, out_dim, 1)
        attn_expanded = attn_weights.unsqueeze(3)  # (batch, N, N, 1, heads)

        # 简单聚合
        out = torch.matmul(attn_weights.mean(dim=-1), h)  # (batch, N, out_dim)

        return torch.relu(out)


class GATClassifier(nn.Module):
    """GAT分类器"""
    def __init__(self, input_size, hidden_dim=128, num_heads=4, num_layers=2,
                 num_classes=9, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        # 特征嵌入
        x = self.input_proj(x)

        # 构建K近邻图
        batch_size, num_nodes, hidden_dim = x.shape
        adj = self.build_knn_graph(x, k=10)

        # GAT层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adj)

        # 全局池化
        x = x.mean(dim=1)

        # 分类
        return self.classifier(x)

    def build_knn_graph(self, x, k=10):
        """构建K近邻图"""
        batch_size, num_nodes, hidden_dim = x.shape

        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))

        _, indices = torch.topk(sim, k=min(k, num_nodes), dim=-1)

        adj = torch.zeros_like(sim)
        adj.scatter_(-1, indices, 1.0)
        adj = (adj + adj.transpose(-1, -2)) / 2
        adj = adj + torch.eye(num_nodes, device=x.device).unsqueeze(0)

        return adj


# ==================== LSTM+GNN模型 ====================

class LSTMGNNClassifier(nn.Module):
    """LSTM+GNN分类器 - 串行时空建模"""
    def __init__(self, input_size, hidden_dim=128, num_layers=2, num_classes=9, dropout=0.2):
        super().__init__()
        # LSTM编码器
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)

        # GCN层
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(2)
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # 构建动态图
        batch_size, num_nodes, hidden_dim = lstm_out.shape
        adj = self.build_knn_graph(lstm_out, k=10)

        # GCN传播
        gnn_out = lstm_out
        for gcn_layer in self.gcn_layers:
            gnn_out = gcn_layer(gnn_out, adj)

        # 全局池化
        out = gnn_out.mean(dim=1)

        # 分类
        return self.classifier(out)

    def build_knn_graph(self, x, k=10):
        """构建K近邻图"""
        batch_size, num_nodes, hidden_dim = x.shape

        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))

        _, indices = torch.topk(sim, k=min(k, num_nodes), dim=-1)

        adj = torch.zeros_like(sim)
        adj.scatter_(-1, indices, 1.0)
        adj = (adj + adj.transpose(-1, -2)) / 2
        adj = adj + torch.eye(num_nodes, device=x.device).unsqueeze(0)

        return adj


# ==================== STGCN模型 ====================

class STGCNBlock(nn.Module):
    """时空图卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        # 时间卷积
        self.temporal_conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size),
                                       padding=(0, kernel_size//2))
        # 空间图卷积
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, (1, 1))
        # 时间卷积
        self.temporal_conv2 = nn.Conv2d(out_channels, out_channels, (1, kernel_size),
                                       padding=(0, kernel_size//2))

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (batch, channels, num_nodes, seq_len)
        adj: (batch, num_nodes, num_nodes)
        """
        # 时间卷积1
        x = torch.relu(self.temporal_conv1(x))

        # 空间图卷积
        batch_size, channels, num_nodes, seq_len = x.shape
        x_reshaped = x.permute(0, 3, 1, 2).reshape(batch_size * seq_len, channels, num_nodes)

        # 归一化邻接矩阵
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)

        # 图卷积
        adj_expanded = adj_norm.unsqueeze(1).expand(batch_size, seq_len, num_nodes, num_nodes)
        adj_expanded = adj_expanded.reshape(batch_size * seq_len, num_nodes, num_nodes)

        x_spatial = torch.matmul(adj_expanded, x_reshaped.transpose(1, 2))
        x_spatial = x_spatial.transpose(1, 2).reshape(batch_size, seq_len, channels, num_nodes)
        x_spatial = x_spatial.permute(0, 2, 3, 1)

        x = self.spatial_conv(x_spatial)

        # 时间卷积2
        x = self.temporal_conv2(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)

        return x


class STGCNClassifier(nn.Module):
    """STGCN分类器 - 时空图卷积网络"""
    def __init__(self, input_size, hidden_channels=[64, 128], num_classes=9, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_channels[0])

        self.stgcn_blocks = nn.ModuleList([
            STGCNBlock(hidden_channels[i], hidden_channels[i+1] if i+1 < len(hidden_channels)
                      else hidden_channels[i], dropout=dropout)
            for i in range(len(hidden_channels))
        ])

        self.output_conv = nn.Conv2d(hidden_channels[-1], num_classes, (1, 1))

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        # 特征嵌入
        x = self.input_proj(x)  # (batch, seq_len, hidden_channels[0])

        # 构建图
        batch_size, seq_len, hidden_dim = x.shape
        adj = self.build_knn_graph(x, k=10)

        # 转换为STGCN输入格式: (batch, channels, num_nodes, seq_len)
        x = x.permute(0, 2, 1).unsqueeze(2)  # (batch, channels, 1, seq_len)
        x = x.expand(-1, -1, seq_len, -1)  # (batch, channels, num_nodes, seq_len)

        # STGCN块
        for stgcn_block in self.stgcn_blocks:
            x = stgcn_block(x, adj)

        # 输出层
        x = self.output_conv(x)  # (batch, num_classes, num_nodes, seq_len)

        # 全局池化
        x = x.mean(dim=[2, 3])  # (batch, num_classes)

        return x

    def build_knn_graph(self, x, k=10):
        """构建K近邻图"""
        batch_size, num_nodes, hidden_dim = x.shape

        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))

        _, indices = torch.topk(sim, k=min(k, num_nodes), dim=-1)

        adj = torch.zeros_like(sim)
        adj.scatter_(-1, indices, 1.0)
        adj = (adj + adj.transpose(-1, -2)) / 2
        adj = adj + torch.eye(num_nodes, device=x.device).unsqueeze(0)

        return adj


# ==================== 训练和评估函数 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, device, save_path, model_name):
    """训练模型"""
    print("\n开始训练...")
    best_val_f1 = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(output.argmax(1).cpu().numpy())
            train_labels.extend(target.cpu().numpy())

        train_f1 = f1_score(train_labels, train_preds, average='macro')

        # 验证
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_preds.extend(output.argmax(1).cpu().numpy())
                val_labels.extend(target.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"学习率降低: {old_lr:.6f} -> {new_lr:.6f}")

        # 早停
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"\n早停触发 (patience={patience})")
            break

    train_time = time.time() - start_time
    print(f"\n训练完成！耗时: {train_time:.2f}秒")
    print(f"最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")

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


def save_results(preds, labels, class_names, save_dir, model_name, train_time=None):
    """保存结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    report = classification_report(labels, preds, target_names=class_names, digits=4)

    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{model_name} Evaluation Report (Comprehensive Baselines)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data Split: 70:15:15 (train:val:test)\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        if train_time:
            f.write(f"Training Time: {train_time:.2f}s\n\n")
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

    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'train_time': train_time}


def train_tree_model(model, train_X, train_y, val_X, val_y):
    """训练LightGBM"""
    print("\n开始训练...")
    start_time = time.time()

    # 展平特征: (N, 100, 25) -> (N, 2500)
    train_X_flat = train_X.reshape(train_X.shape[0], -1)
    val_X_flat = val_X.reshape(val_X.shape[0], -1)

    model.fit(
        train_X_flat, train_y,
        eval_set=[(val_X_flat, val_y)],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    val_preds = model.predict(val_X_flat)
    val_f1 = f1_score(val_y, val_preds, average='macro')

    train_time = time.time() - start_time
    print(f"\n训练完成！耗时: {train_time:.2f}秒")
    print(f"验证集F1: {val_f1:.4f}")
    print(f"最佳迭代: {model.best_iteration_}")

    return train_time


def evaluate_tree_model(model, test_X, test_y):
    """评估树模型"""
    test_X_flat = test_X.reshape(test_X.shape[0], -1)
    preds = model.predict(test_X_flat)
    return preds


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 如果只是列出模型
    if args.list:
        print("\n可用模型:")
        for key, name in AVAILABLE_MODELS.items():
            print(f"  {key:<12} - {name}")
        print("\n使用方法:")
        print("  python baseline/run_comprehensive_baselines.py --models cnn1d gru")
        print("  python baseline/run_comprehensive_baselines.py --models stgcn")
        print("  python baseline/run_comprehensive_baselines.py --models all")
        print("  python baseline/run_comprehensive_baselines.py --data 50packets --models all")
        return

    # 确定要运行的模型
    if 'all' in args.models:
        models_to_run = list(AVAILABLE_MODELS.keys())
    else:
        models_to_run = args.models

    # 确定数据集路径和输出目录
    if args.data == '50packets_no_leak':
        data_dir = 'data/processed/raw_packet_sequences_50packets_no_leak'
        output_base = args.output_dir or 'baseline/results_no_leak'
    elif args.data == '50packets':
        data_dir = 'data/processed/raw_packet_sequences_50packets'
        output_base = args.output_dir or 'baseline/results_50packets'
    else:
        data_dir = 'data/processed/raw_packet_sequences'
        output_base = args.output_dir or 'baseline/results_v2'

    print(f"\n数据集: {data_dir}")
    print(f"输出目录: {output_base}")
    print(f"\n将运行以下模型: {', '.join([AVAILABLE_MODELS[m] for m in models_to_run])}")

    print("\n" + "="*100)
    print("加载数据".center(100))
    print("="*100)

    # 加载数据
    train_X, train_y, val_X, val_y, test_X, test_y, metadata = load_data(data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    input_size = train_X.shape[2]
    seq_len = train_X.shape[1]

    # 类别权重
    class_weights = torch.FloatTensor(get_class_weights(train_y, num_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 创建DataLoader（所有模型共用）
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

    results = {}

    # ==================== A. 时序模型 ====================

    # 1. CNN1D
    if 'cnn1d' in models_to_run:
        print("\n" + "="*100)
        print("1. CNN1D - 局部时序特征".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'cnn1d'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = CNN1DClassifier(input_size, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {
            'model': 'CNN1D',
            'type': 'Temporal',
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'random_seed': RANDOM_SEED,
            'data_split': '70:15:15 (train:val:test)'
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'CNN1D')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'CNN1D', train_time)
        results['CNN1D'] = metrics

    # 2. GRU
    if 'gru' in models_to_run:
        print("\n" + "="*100)
        print("2. GRU - 全局时序依赖".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'gru'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = GRUClassifier(input_size, num_classes, hidden_size=128, num_layers=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {
            'model': 'GRU',
            'type': 'Temporal',
            'hidden_size': 128,
            'num_layers': 2,
            'batch_size': 64,
            'learning_rate': 0.001,
            'random_seed': RANDOM_SEED
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'GRU')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'GRU', train_time)
        results['GRU'] = metrics

    # 3. TCN
    if 'tcn' in models_to_run:
        print("\n" + "="*100)
        print("3. TCN - 因果时序建模".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'tcn'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = TCNClassifier(input_size, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {'model': 'TCN', 'type': 'Temporal', 'random_seed': RANDOM_SEED}
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'TCN')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'TCN', train_time)
        results['TCN'] = metrics

    # ==================== B. 非时序模型 ====================

    # 4. LightGBM
    if 'lightgbm' in models_to_run and LIGHTGBM_AVAILABLE:
        print("\n" + "="*100)
        print("4. LightGBM - 表格数据基线".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'lightgbm'
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

        config = {'model': 'LightGBM', 'type': 'Non-Temporal', 'random_seed': RANDOM_SEED}
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        train_time = train_tree_model(lgb_model, train_X, train_y, val_X, val_y)

        print("\n最终测试集评估...")
        test_preds = evaluate_tree_model(lgb_model, test_X, test_y)
        metrics = save_results(test_preds, test_y, class_names, save_dir, 'LightGBM', train_time)
        results['LightGBM'] = metrics

    # 5. XGBoost
    if 'xgboost' in models_to_run:
        print("\n" + "="*100)
        print("5. XGBoost - 梯度提升基线".center(100))
        print("="*100)

        try:
            import xgboost as xgb

            save_dir = Path(output_base) / 'xgboost'
            save_dir.mkdir(parents=True, exist_ok=True)

            # 展平时序维度 (batch, seq_len, features) -> (batch, seq_len * features)
            train_X_flat = train_X.reshape(train_X.shape[0], -1)
            val_X_flat = val_X.reshape(val_X.shape[0], -1)
            test_X_flat = test_X.reshape(test_X.shape[0], -1)

            print(f"展平后数据形状: train_X_flat={train_X_flat.shape}, val_X_flat={val_X_flat.shape}, test_X_flat={test_X_flat.shape}")

            print("训练XGBoost模型...")

            config = {'model': 'XGBoost', 'type': 'Non-Temporal', 'random_seed': RANDOM_SEED}
            with open(save_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)

            start_time = time.time()
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            xgb_model.fit(
                train_X_flat, train_y,
                eval_set=[(val_X_flat, val_y)],
                verbose=False
            )
            train_time = time.time() - start_time

            print("\n最终测试集评估...")
            test_preds = evaluate_tree_model(xgb_model, test_X_flat, test_y)
            metrics = save_results(test_preds, test_y, class_names, save_dir, 'XGBoost', train_time)
            results['XGBoost'] = metrics
        except ImportError:
            print("⚠️  XGBoost未安装，跳过此模型")

    # ==================== C. 图结构模型 ====================

    # 6. GCN
    if 'gcn' in models_to_run:
        print("\n" + "="*100)
        print("6. GCN - 图卷积基线".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'gcn'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = GCNClassifier(input_size, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {'model': 'GCN', 'type': 'Graph', 'random_seed': RANDOM_SEED}
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'GCN')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'GCN', train_time)
        results['GCN'] = metrics

    # 7. GAT
    if 'gat' in models_to_run:
        print("\n" + "="*100)
        print("7. GAT - 图注意力".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'gat'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = GATClassifier(input_size, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {'model': 'GAT', 'type': 'Graph', 'random_seed': RANDOM_SEED}
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'GAT')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'GAT', train_time)
        results['GAT'] = metrics

    # ==================== D. 时空结合模型 ====================

    # 8. LSTM+GNN
    if 'lstm_gnn' in models_to_run:
        print("\n" + "="*100)
        print("8. LSTM+GNN - 串行时空建模".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'lstm_gnn'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = LSTMGNNClassifier(input_size, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {'model': 'LSTM+GNN', 'type': 'Spatio-Temporal', 'random_seed': RANDOM_SEED}
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'LSTM+GNN')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'LSTM+GNN', train_time)
        results['LSTM+GNN'] = metrics

    # 9. STGCN
    if 'stgcn' in models_to_run:
        print("\n" + "="*100)
        print("9. STGCN - 并行时空建模".center(100))
        print("="*100)

        save_dir = Path(output_base) / 'stgcn'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = STGCNClassifier(input_size, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        config = {'model': 'STGCN', 'type': 'Spatio-Temporal', 'random_seed': RANDOM_SEED}
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        model_path = save_dir / 'best_model.pth'
        train_time, best_epoch = train_model(model, train_loader, val_loader, criterion,
                                             optimizer, 50, device, model_path, 'STGCN')

        print("\n最终测试集评估...")
        model.load_state_dict(torch.load(model_path))
        test_preds, test_labels = evaluate_model(model, test_loader, device)
        metrics = save_results(test_preds, test_labels, class_names, save_dir, 'STGCN', train_time)
        results['STGCN'] = metrics

    # 10. Hybrid (从已有结果读取)
    if 'hybrid' in models_to_run:
        print("\n" + "="*100)
        print("10. Hybrid - Transformer+GNN (主要贡献)".center(100))
        print("="*100)

        # 根据数据集选择正确的Hybrid结果路径
        if args.data == '50packets':
            hybrid_result_path = Path('baseline/results/hybrid_model_improved_50packets')
        else:
            hybrid_result_path = Path('baseline/results/hybrid_model_fixed')

        if hybrid_result_path.exists():
            print(f"从已有结果读取Hybrid模型性能: {hybrid_result_path}")
            # 优先读取JSON结果文件
            json_file = hybrid_result_path / 'test_results.json'
            report_file = hybrid_result_path / 'evaluation_report.txt'

            import re

            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    test_results = json.load(f)
                results['Hybrid'] = {
                    'accuracy': test_results.get('test_accuracy', test_results.get('accuracy', 0.0)),
                    'macro_f1': test_results.get('test_macro_f1', test_results.get('macro_f1', 0.0)),
                    'weighted_f1': test_results.get('test_weighted_f1', test_results.get('weighted_f1', 0.0)),
                    'train_time': test_results.get('train_time', None)
                }
            elif report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    acc_match = re.search(r'Accuracy:\s+([\d.]+)', content)
                    macro_f1_match = re.search(r'Macro F1:\s+([\d.]+)', content)
                    weighted_f1_match = re.search(r'Weighted F1:\s+([\d.]+)', content)
                    time_match = re.search(r'训练时间:\s+([\d.]+)秒', content)
                    if not time_match:
                        time_match = re.search(r'Training Time:\s+([\d.]+)s', content)

                    if acc_match and macro_f1_match:
                        results['Hybrid'] = {
                            'accuracy': float(acc_match.group(1)),
                            'macro_f1': float(macro_f1_match.group(1)),
                            'weighted_f1': float(weighted_f1_match.group(1)) if weighted_f1_match else 0.0,
                            'train_time': float(time_match.group(1)) if time_match else None
                        }

            if 'Hybrid' in results:
                print(f"  Accuracy:    {results['Hybrid']['accuracy']:.4f}")
                print(f"  Macro F1:    {results['Hybrid']['macro_f1']:.4f}")
                print(f"  Weighted F1: {results['Hybrid']['weighted_f1']:.4f}")
            else:
                print("⚠️  无法解析Hybrid结果文件")
        else:
            print("⚠️  Hybrid模型结果未找到，请先运行: python scripts/hybrid_model/02_train_hybrid_improved.py")

    # 11. BSTGCN (从已有结果读取)
    if 'bstgcn' in models_to_run:
        print("\n" + "="*100)
        print("11. BSTGCN - 贝叶斯时空图神经网络 (不确定性量化)".center(100))
        print("="*100)

        # 查找最新的BSTGCN结果
        bstgcn_results_dir = Path('baseline/results/bstgcn_50packets')
        if bstgcn_results_dir.exists():
            # 查找最新的结果目录
            result_dirs = sorted([d for d in bstgcn_results_dir.iterdir() if d.is_dir()],
                               key=lambda x: x.name, reverse=True)

            if result_dirs:
                latest_result = result_dirs[0]
                print(f"从已有结果读取BSTGCN模型性能: {latest_result.name}")

                # 优先读取JSON结果文件
                json_file = latest_result / 'test_results.json'
                report_file = latest_result / 'evaluation_report.txt'

                if json_file.exists():
                    # 从JSON读取
                    with open(json_file, 'r', encoding='utf-8') as f:
                        test_results = json.load(f)

                    # 读取训练历史获取训练时间
                    train_time = None
                    history_file = latest_result / 'train_history.json'
                    if history_file.exists():
                        with open(history_file, 'r', encoding='utf-8') as f:
                            history = json.load(f)
                            if 'total_time' in history:
                                train_time = history['total_time']

                    # 兼容不同的键名
                    accuracy = test_results.get('test_accuracy', test_results.get('accuracy', 0.0))
                    macro_f1 = test_results.get('test_macro_f1', test_results.get('macro_f1', 0.0))
                    weighted_f1 = test_results.get('test_weighted_f1', test_results.get('weighted_f1', 0.0))

                    results['BSTGCN'] = {
                        'accuracy': accuracy,
                        'macro_f1': macro_f1,
                        'weighted_f1': weighted_f1,
                        'train_time': train_time
                    }

                    print(f"  Accuracy:    {accuracy:.4f}")
                    print(f"  Macro F1:    {macro_f1:.4f}")
                    if weighted_f1 > 0:
                        print(f"  Weighted F1: {weighted_f1:.4f}")

                    # 显示不确定性信息
                    mean_unc = test_results.get('test_mean_uncertainty', test_results.get('mean_uncertainty', None))
                    unc_corr = test_results.get('uncertainty_error_correlation', test_results.get('uncertainty_correlation', None))

                    if mean_unc is not None:
                        print(f"  Mean Uncertainty: {mean_unc:.4f}")
                    if unc_corr is not None:
                        print(f"  Uncertainty-Error Correlation: {unc_corr:.4f}")

                elif report_file.exists():
                    # 从文本报告读取（备用方案）
                    with open(report_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 解析指标
                        import re
                        acc_match = re.search(r'Accuracy:\s+([\d.]+)', content)
                        macro_f1_match = re.search(r'Macro F1:\s+([\d.]+)', content)
                        weighted_f1_match = re.search(r'Weighted F1:\s+([\d.]+)', content)
                        time_match = re.search(r'训练时间:\s+([\d.]+)秒', content)
                        if not time_match:
                            time_match = re.search(r'Training Time:\s+([\d.]+)s', content)

                        if acc_match and macro_f1_match:
                            results['BSTGCN'] = {
                                'accuracy': float(acc_match.group(1)),
                                'macro_f1': float(macro_f1_match.group(1)),
                                'weighted_f1': float(weighted_f1_match.group(1)) if weighted_f1_match else 0.0,
                                'train_time': float(time_match.group(1)) if time_match else None
                            }
                            print(f"  Accuracy:    {results['BSTGCN']['accuracy']:.4f}")
                            print(f"  Macro F1:    {results['BSTGCN']['macro_f1']:.4f}")
                            if weighted_f1_match:
                                print(f"  Weighted F1: {results['BSTGCN']['weighted_f1']:.4f}")
                        else:
                            print("⚠️  无法解析BSTGCN结果文件")
                else:
                    print("⚠️  BSTGCN结果文件未找到（需要 test_results.json 或 evaluation_report.txt）")
            else:
                print("⚠️  未找到BSTGCN结果目录")
        else:
            print("⚠️  BSTGCN模型结果未找到，请先运行: python scripts/BSTGCN/train_bstgcn.py")

    # ==================== 总结 ====================

    print("\n" + "="*100)
    print("实验总结".center(100))
    print("="*100)

    print("\n按类别分组:")
    print("\nA. 时序模型:")
    print("| 模型 | Accuracy | Macro F1 | Weighted F1 | 训练时间(s) |")
    print("|------|----------|----------|-------------|-------------|")
    for name in ['CNN1D', 'GRU', 'TCN']:
        if name in results:
            m = results[name]
            print(f"| {name:<8} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                  f"{m['weighted_f1']:.4f} | {m['train_time']:.1f} |")

    print("\nB. 非时序模型:")
    print("| 模型 | Accuracy | Macro F1 | Weighted F1 | 训练时间(s) |")
    print("|------|----------|----------|-------------|-------------|")
    for name in ['LightGBM', 'XGBoost']:
        if name in results:
            m = results[name]
            print(f"| {name:<8} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                  f"{m['weighted_f1']:.4f} | {m['train_time']:.1f} |")

    print("\nC. 图结构模型:")
    print("| 模型 | Accuracy | Macro F1 | Weighted F1 | 训练时间(s) |")
    print("|------|----------|----------|-------------|-------------|")
    for name in ['GCN', 'GAT']:
        if name in results:
            m = results[name]
            print(f"| {name:<8} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                  f"{m['weighted_f1']:.4f} | {m['train_time']:.1f} |")

    print("\nD. 时空结合模型:")
    print("| 模型 | Accuracy | Macro F1 | Weighted F1 | 训练时间(s) |")
    print("|------|----------|----------|-------------|-------------|")
    for name in ['LSTM+GNN', 'STGCN', 'Hybrid', 'BSTGCN']:
        if name in results:
            m = results[name]
            t = m['train_time'] if m['train_time'] else 0
            print(f"| {name:<10} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                  f"{m['weighted_f1']:.4f} | {t:.1f} |")

    print("\n总体排名 (按Macro F1):")
    print("| 排名 | 模型 | 类型 | Macro F1 | Accuracy |")
    print("|------|------|------|----------|----------|")

    model_types = {
        'CNN1D': 'Temporal', 'GRU': 'Temporal', 'TCN': 'Temporal',
        'LightGBM': 'Non-Temporal', 'XGBoost': 'Non-Temporal',
        'GCN': 'Graph', 'GAT': 'Graph',
        'LSTM+GNN': 'Spatio-Temporal', 'STGCN': 'Spatio-Temporal',
        'Hybrid': 'Spatio-Temporal', 'BSTGCN': 'Bayesian-ST'
    }

    sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        model_type = model_types.get(name, 'Unknown')
        print(f"| {rank:<4} | {name:<10} | {model_type:<16} | "
              f"{metrics['macro_f1']:.4f} | {metrics['accuracy']:.4f} |")

    # 保存总结
    summary_path = Path(output_base) / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 完成！结果已保存到: {output_base}/")
    print(f"✅ 总结已保存到: {summary_path}")


if __name__ == '__main__':
    main()

