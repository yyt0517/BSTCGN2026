"""
强基线OOD检测实验 - XGBoost, GRU, LSTM+GNN, GAT, STGCN

对比:
1. XGBoost + Entropy
2. GRU + MC Dropout
3. LSTM+GNN + Entropy
4. GAT + MC Dropout
5. STGCN + MC Dropout

实验设置: Leave-One-Attack-Out (SlowrateDoS as OOD)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")


# ==================== OOD Score Functions ====================

def compute_msp_score(logits: torch.Tensor) -> np.ndarray:
    """Maximum Softmax Probability Score (higher = more likely OOD)"""
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1)[0]
    return (1 - max_probs).cpu().numpy()


def compute_entropy_score(logits: torch.Tensor) -> np.ndarray:
    """Entropy Score (higher = more likely OOD)"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy.cpu().numpy()


def mc_dropout_inference(model, x, num_samples=30):
    """MC Dropout inference"""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    probs_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

    probs_stack = torch.stack(probs_list, dim=0)
    pred_mean = probs_stack.mean(dim=0)

    epsilon = 1e-10
    entropy_mean = -torch.sum(pred_mean * torch.log(pred_mean + epsilon), dim=-1)
    entropy_samples = -torch.sum(probs_stack * torch.log(probs_stack + epsilon), dim=-1)
    mean_entropy = entropy_samples.mean(dim=0)
    epistemic_uncertainty = entropy_mean - mean_entropy

    return pred_mean, epistemic_uncertainty.cpu().numpy()


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
    """Compute OOD detection metrics"""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[idx]

    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95tpr': fpr_at_95tpr,
        'id_mean': float(np.mean(id_scores)),
        'ood_mean': float(np.mean(ood_scores))
    }


# ==================== Model Definitions ====================

class GRUClassifier(nn.Module):
    """GRU Classifier"""
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


class STGCNBlock(nn.Module):
    """STGCN Block - Temporal Conv + Spatial Graph Conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        # Temporal conv
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Spatial graph conv (simplified)
        self.spatial_fc = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj):
        # x: (B, L, C), adj: (B, L, L)
        residual = self.residual(x)

        # Temporal conv
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)  # (B, L, C)

        # Spatial graph conv: X' = AXW
        x = torch.bmm(adj, x)  # (B, L, C)
        x = self.spatial_fc(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)

        x = self.dropout(x)
        return x + residual


class GCNLayer(nn.Module):
    """Simple GCN Layer"""
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # x: (B, L, C), adj: (B, L, L)
        x = torch.bmm(adj, x)  # Graph conv
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class LSTMGNNClassifier(nn.Module):
    """LSTM+GNN Classifier"""
    def __init__(self, input_size, num_classes, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout) for _ in range(2)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def build_knn_graph(self, x, k=10):
        B, L, H = x.shape
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))
        _, indices = torch.topk(sim, k=min(k, L), dim=-1)
        adj = torch.zeros_like(sim)
        adj.scatter_(-1, indices, 1.0)
        adj = (adj + adj.transpose(-1, -2)) / 2
        adj = adj + torch.eye(L, device=x.device).unsqueeze(0)
        # Normalize
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg
        return adj

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        adj = self.build_knn_graph(lstm_out, k=10)
        gnn_out = lstm_out
        for gcn in self.gcn_layers:
            gnn_out = gcn(gnn_out, adj)
        out = gnn_out.mean(dim=1)
        return self.classifier(out)


class STGCNClassifier(nn.Module):
    """STGCN Classifier"""
    def __init__(self, input_size, num_classes, hidden_channels=[64, 128], k_neighbors=10, dropout=0.2):
        super().__init__()
        self.k_neighbors = k_neighbors

        self.input_proj = nn.Linear(input_size, hidden_channels[0])

        self.stgcn_blocks = nn.ModuleList()
        for i in range(len(hidden_channels)):
            in_ch = hidden_channels[i-1] if i > 0 else hidden_channels[0]
            out_ch = hidden_channels[i]
            self.stgcn_blocks.append(STGCNBlock(in_ch, out_ch, dropout=dropout))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1], num_classes)
        )

    def build_knn_graph(self, x, k):
        # x: (B, L, C)
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        dist = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
        _, indices = dist.topk(k + 1, dim=-1, largest=False)
        B, L, _ = x.shape
        adj = torch.zeros(B, L, L, device=x.device)
        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1).expand(-1, L, k + 1)
        row_idx = torch.arange(L, device=x.device).view(1, -1, 1).expand(B, -1, k + 1)
        adj[batch_idx, row_idx, indices] = 1.0
        adj = (adj + adj.transpose(1, 2)) / 2
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg
        return adj

    def forward(self, x):
        x = self.input_proj(x)
        adj = self.build_knn_graph(x, self.k_neighbors)
        for block in self.stgcn_blocks:
            x = block(x, adj)
        x = x.mean(dim=1)
        return self.classifier(x)


class GATLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        self.W = nn.Linear(in_dim, out_dim)
        self.attn = nn.Parameter(torch.Tensor(num_heads, self.head_dim * 2))
        nn.init.xavier_uniform_(self.attn)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, N, in_dim), adj: (B, N, N)
        B, N, _ = x.shape
        h = self.W(x)  # (B, N, out_dim)
        h = h.view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)

        # Attention
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, N, N, H, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)  # (B, N, N, H, D)
        concat = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, H, 2D)

        attn_scores = (concat * self.attn).sum(dim=-1)  # (B, N, N, H)
        attn_scores = self.leaky_relu(attn_scores)

        # Mask with adjacency
        mask = (adj == 0).unsqueeze(-1)  # (B, N, N, 1)
        attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=2)  # (B, N, N, H)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        out = torch.einsum('bnjh,bnjhd->bnhd', attn_weights, h_j)  # (B, N, H, D)
        out = out.reshape(B, N, self.out_dim)  # (B, N, out_dim)

        return F.relu(out)


class GATClassifier(nn.Module):
    """GAT Classifier"""
    def __init__(self, input_size, num_classes, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def build_knn_graph(self, x, k=10):
        B, L, H = x.shape
        x_norm = F.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))
        _, indices = torch.topk(sim, k=min(k, L), dim=-1)
        adj = torch.zeros_like(sim)
        adj.scatter_(-1, indices, 1.0)
        adj = (adj + adj.transpose(-1, -2)) / 2
        adj = adj + torch.eye(L, device=x.device).unsqueeze(0)
        return adj

    def forward(self, x):
        x = self.input_proj(x)
        adj = self.build_knn_graph(x, k=10)
        for gat in self.gat_layers:
            x = gat(x, adj)
        out = x.mean(dim=1)
        return self.classifier(out)


# ==================== Data Loading ====================

def load_ood_data(data_dir: str, batch_size: int = 512):
    """Load OOD experiment data"""
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

    return {
        'train_seq': train_seq, 'train_labels': train_labels,
        'val_seq': val_seq, 'val_labels': val_labels,
        'test_id_seq': test_id_seq, 'test_id_labels': test_id_labels,
        'test_ood_seq': test_ood_seq,
        'metadata': metadata
    }


# ==================== Training Functions ====================

def train_xgboost(data, num_classes):
    """Train XGBoost model"""
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)

    train_X = data['train_seq'].reshape(data['train_seq'].shape[0], -1)
    val_X = data['val_seq'].reshape(data['val_seq'].shape[0], -1)
    test_id_X = data['test_id_seq'].reshape(data['test_id_seq'].shape[0], -1)
    test_ood_X = data['test_ood_seq'].reshape(data['test_ood_seq'].shape[0], -1)

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )

    model.fit(train_X, data['train_labels'],
              eval_set=[(val_X, data['val_labels'])],
              verbose=False)

    # Get predictions and probabilities
    test_id_probs = model.predict_proba(test_id_X)
    test_ood_probs = model.predict_proba(test_ood_X)
    test_id_preds = model.predict(test_id_X)

    id_acc = accuracy_score(data['test_id_labels'], test_id_preds)
    id_f1 = f1_score(data['test_id_labels'], test_id_preds, average='macro')

    print(f"  ID Accuracy: {id_acc:.4f}")
    print(f"  ID F1: {id_f1:.4f}")

    # Compute OOD scores
    results = {'id_accuracy': id_acc, 'id_f1': id_f1}

    # Entropy
    id_entropy = -np.sum(test_id_probs * np.log(test_id_probs + 1e-10), axis=1)
    ood_entropy = -np.sum(test_ood_probs * np.log(test_ood_probs + 1e-10), axis=1)
    results['Entropy'] = compute_ood_metrics(id_entropy, ood_entropy)

    return results


def train_gru(data, num_classes, device, batch_size=512, epochs=50):
    """Train GRU model"""
    print("\n" + "="*60)
    print("Training GRU")
    print("="*60)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['train_seq']),
                     torch.LongTensor(data['train_labels'])),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['val_seq']),
                     torch.LongTensor(data['val_labels'])),
        batch_size=batch_size, shuffle=False
    )

    input_size = data['train_seq'].shape[2]
    model = GRUClassifier(input_size, num_classes, hidden_size=128,
                          num_layers=2, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_f1 = 0
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Val F1 = {val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best Val F1: {best_val_f1:.4f}")

    return model


def evaluate_gru_ood(model, data, device, batch_size=512, num_mc_samples=30):
    """Evaluate GRU for OOD detection"""
    model.eval()

    test_id_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_id_seq']),
                     torch.LongTensor(data['test_id_labels'])),
        batch_size=batch_size, shuffle=False
    )
    test_ood_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_ood_seq']),
                     torch.zeros(len(data['test_ood_seq']), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Collect ID logits
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

    # Collect OOD logits
    ood_logits_list = []
    with torch.no_grad():
        for batch_x, _ in test_ood_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            ood_logits_list.append(logits)

    ood_logits = torch.cat(ood_logits_list, dim=0)

    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    print(f"  ID Accuracy: {id_acc:.4f}")
    print(f"  ID F1: {id_f1:.4f}")

    results = {'id_accuracy': id_acc, 'id_f1': id_f1}

    # MSP
    id_msp = compute_msp_score(id_logits)
    ood_msp = compute_msp_score(ood_logits)
    results['MSP'] = compute_ood_metrics(id_msp, ood_msp)

    # Entropy
    id_entropy = compute_entropy_score(id_logits)
    ood_entropy = compute_entropy_score(ood_logits)
    results['Entropy'] = compute_ood_metrics(id_entropy, ood_entropy)

    # MC Dropout
    print("  Running MC Dropout inference...")
    id_mc_list, ood_mc_list = [], []

    for batch_x, _ in test_id_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        id_mc_list.append(unc)

    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        ood_mc_list.append(unc)

    id_mc = np.concatenate(id_mc_list)
    ood_mc = np.concatenate(ood_mc_list)
    results['MC_Dropout'] = compute_ood_metrics(id_mc, ood_mc)

    return results


def train_lstm_gnn(data, num_classes, device, batch_size=256, epochs=100):
    """Train LSTM+GNN model"""
    print("\n" + "="*60)
    print("Training LSTM+GNN")
    print("="*60)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['train_seq']),
                     torch.LongTensor(data['train_labels'])),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['val_seq']),
                     torch.LongTensor(data['val_labels'])),
        batch_size=batch_size, shuffle=False
    )

    input_size = data['train_seq'].shape[2]
    model = LSTMGNNClassifier(input_size, num_classes, hidden_dim=128,
                              num_layers=2, dropout=0.2).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
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

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Val F1 = {val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best Val F1: {best_val_f1:.4f}")

    return model


def evaluate_lstm_gnn_ood(model, data, device, batch_size=256, num_mc_samples=30):
    """Evaluate LSTM+GNN for OOD detection"""
    model.eval()

    test_id_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_id_seq']),
                     torch.LongTensor(data['test_id_labels'])),
        batch_size=batch_size, shuffle=False
    )
    test_ood_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_ood_seq']),
                     torch.zeros(len(data['test_ood_seq']), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Collect ID logits
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

    # Collect OOD logits
    ood_logits_list = []
    with torch.no_grad():
        for batch_x, _ in test_ood_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            ood_logits_list.append(logits)

    ood_logits = torch.cat(ood_logits_list, dim=0)

    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    print(f"  ID Accuracy: {id_acc:.4f}")
    print(f"  ID F1: {id_f1:.4f}")

    results = {'id_accuracy': id_acc, 'id_f1': id_f1}

    # MSP
    id_msp = compute_msp_score(id_logits)
    ood_msp = compute_msp_score(ood_logits)
    results['MSP'] = compute_ood_metrics(id_msp, ood_msp)

    # Entropy
    id_entropy = compute_entropy_score(id_logits)
    ood_entropy = compute_entropy_score(ood_logits)
    results['Entropy'] = compute_ood_metrics(id_entropy, ood_entropy)

    # MC Dropout
    print("  Running MC Dropout inference...")
    id_mc_list, ood_mc_list = [], []

    for batch_x, _ in test_id_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        id_mc_list.append(unc)

    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        ood_mc_list.append(unc)

    id_mc = np.concatenate(id_mc_list)
    ood_mc = np.concatenate(ood_mc_list)
    results['MC_Dropout'] = compute_ood_metrics(id_mc, ood_mc)

    return results


def train_stgcn(data, num_classes, device, batch_size=256, epochs=100):
    """Train STGCN model"""
    print("\n" + "="*60)
    print("Training STGCN")
    print("="*60)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['train_seq']),
                     torch.LongTensor(data['train_labels'])),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['val_seq']),
                     torch.LongTensor(data['val_labels'])),
        batch_size=batch_size, shuffle=False
    )

    input_size = data['train_seq'].shape[2]
    model = STGCNClassifier(input_size, num_classes, hidden_channels=[64, 128],
                            k_neighbors=10, dropout=0.2).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
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

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Val F1 = {val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best Val F1: {best_val_f1:.4f}")

    return model


def evaluate_stgcn_ood(model, data, device, batch_size=256, num_mc_samples=30):
    """Evaluate STGCN for OOD detection"""
    model.eval()

    test_id_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_id_seq']),
                     torch.LongTensor(data['test_id_labels'])),
        batch_size=batch_size, shuffle=False
    )
    test_ood_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_ood_seq']),
                     torch.zeros(len(data['test_ood_seq']), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Collect ID logits
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

    # Collect OOD logits
    ood_logits_list = []
    with torch.no_grad():
        for batch_x, _ in test_ood_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            ood_logits_list.append(logits)

    ood_logits = torch.cat(ood_logits_list, dim=0)

    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    print(f"  ID Accuracy: {id_acc:.4f}")
    print(f"  ID F1: {id_f1:.4f}")

    results = {'id_accuracy': id_acc, 'id_f1': id_f1}

    # MSP
    id_msp = compute_msp_score(id_logits)
    ood_msp = compute_msp_score(ood_logits)
    results['MSP'] = compute_ood_metrics(id_msp, ood_msp)

    # Entropy
    id_entropy = compute_entropy_score(id_logits)
    ood_entropy = compute_entropy_score(ood_logits)
    results['Entropy'] = compute_ood_metrics(id_entropy, ood_entropy)

    # MC Dropout
    print("  Running MC Dropout inference...")
    id_mc_list, ood_mc_list = [], []

    for batch_x, _ in test_id_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        id_mc_list.append(unc)

    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        ood_mc_list.append(unc)

    id_mc = np.concatenate(id_mc_list)
    ood_mc = np.concatenate(ood_mc_list)
    results['MC_Dropout'] = compute_ood_metrics(id_mc, ood_mc)

    return results


def train_gat(data, num_classes, device, batch_size=256, epochs=100):
    """Train GAT model"""
    print("\n" + "="*60)
    print("Training GAT")
    print("="*60)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['train_seq']),
                     torch.LongTensor(data['train_labels'])),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['val_seq']),
                     torch.LongTensor(data['val_labels'])),
        batch_size=batch_size, shuffle=False
    )

    input_size = data['train_seq'].shape[2]
    model = GATClassifier(input_size, num_classes, hidden_dim=128,
                          num_heads=4, num_layers=2, dropout=0.2).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
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

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Val F1 = {val_f1:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best Val F1: {best_val_f1:.4f}")

    return model


def evaluate_gat_ood(model, data, device, batch_size=256, num_mc_samples=30):
    """Evaluate GAT for OOD detection"""
    model.eval()

    test_id_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_id_seq']),
                     torch.LongTensor(data['test_id_labels'])),
        batch_size=batch_size, shuffle=False
    )
    test_ood_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['test_ood_seq']),
                     torch.zeros(len(data['test_ood_seq']), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # Collect ID logits
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

    # Collect OOD logits
    ood_logits_list = []
    with torch.no_grad():
        for batch_x, _ in test_ood_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            ood_logits_list.append(logits)

    ood_logits = torch.cat(ood_logits_list, dim=0)

    id_acc = accuracy_score(id_labels, id_preds)
    id_f1 = f1_score(id_labels, id_preds, average='macro')

    print(f"  ID Accuracy: {id_acc:.4f}")
    print(f"  ID F1: {id_f1:.4f}")

    results = {'id_accuracy': id_acc, 'id_f1': id_f1}

    # Entropy
    id_entropy = compute_entropy_score(id_logits)
    ood_entropy = compute_entropy_score(ood_logits)
    results['Entropy'] = compute_ood_metrics(id_entropy, ood_entropy)

    # MC Dropout
    print("  Running MC Dropout inference...")
    id_mc_list, ood_mc_list = [], []

    for batch_x, _ in test_id_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        id_mc_list.append(unc)

    for batch_x, _ in test_ood_loader:
        batch_x = batch_x.to(device)
        _, unc = mc_dropout_inference(model, batch_x, num_mc_samples)
        ood_mc_list.append(unc)

    id_mc = np.concatenate(id_mc_list)
    ood_mc = np.concatenate(ood_mc_list)
    results['MC_Dropout'] = compute_ood_metrics(id_mc, ood_mc)

    return results


# ==================== Main Function ====================

def main():
    print("="*70)
    print("Strong Baselines OOD Detection Experiment")
    print("OOD Class: SlowrateDoS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    data_dir = 'data/processed/ood_slowrate'
    print(f"\nLoading data from {data_dir}...")
    data = load_ood_data(data_dir)

    metadata = data['metadata']
    num_classes = metadata['num_id_classes']
    num_features = metadata['num_features']

    print(f"  ID Classes: {num_classes}")
    print(f"  Features: {num_features}")
    print(f"  Train samples: {len(data['train_seq']):,}")
    print(f"  ID Test samples: {len(data['test_id_seq']):,}")
    print(f"  OOD Test samples: {len(data['test_ood_seq']):,}")

    all_results = {}

    # 1. XGBoost (Non-temporal baseline)
    if XGBOOST_AVAILABLE:
        xgb_results = train_xgboost(data, num_classes)
        all_results['XGBoost'] = xgb_results
        print(f"  XGBoost Entropy: AUROC={xgb_results['Entropy']['auroc']:.4f}")

    # 2. GRU (Temporal model)
    gru_model = train_gru(data, num_classes, device, batch_size=512, epochs=50)
    gru_results = evaluate_gru_ood(gru_model, data, device, batch_size=512, num_mc_samples=30)
    all_results['GRU'] = gru_results
    print(f"  GRU MC Dropout: AUROC={gru_results['MC_Dropout']['auroc']:.4f}")

    # 3. LSTM+GNN (Spatiotemporal - serial)
    lstm_gnn_model = train_lstm_gnn(data, num_classes, device, batch_size=256, epochs=100)
    lstm_gnn_results = evaluate_lstm_gnn_ood(lstm_gnn_model, data, device, batch_size=256, num_mc_samples=30)
    all_results['LSTM_GNN'] = lstm_gnn_results
    print(f"  LSTM+GNN Entropy: AUROC={lstm_gnn_results['Entropy']['auroc']:.4f}")

    # 4. GAT (Graph model)
    gat_model = train_gat(data, num_classes, device, batch_size=256, epochs=100)
    gat_results = evaluate_gat_ood(gat_model, data, device, batch_size=256, num_mc_samples=30)
    all_results['GAT'] = gat_results
    print(f"  GAT MC Dropout: AUROC={gat_results['MC_Dropout']['auroc']:.4f}")

    # 5. STGCN (Spatiotemporal - parallel)
    stgcn_model = train_stgcn(data, num_classes, device, batch_size=256, epochs=100)
    stgcn_results = evaluate_stgcn_ood(stgcn_model, data, device, batch_size=256, num_mc_samples=30)
    all_results['STGCN'] = stgcn_results
    print(f"  STGCN MC Dropout: AUROC={stgcn_results['MC_Dropout']['auroc']:.4f}")

    # Print summary table (simplified - one method per model)
    print("\n" + "="*85)
    print("SUMMARY: OOD Detection Comparison (Leave-One-Attack-Out: SlowrateDoS)")
    print("="*85)
    print(f"{'Method':<15} {'Model':<12} {'AUROC':>8} {'AUPR':>8} {'FPR@95':>8} {'ID Acc':>10} {'ID F1':>10}")
    print("-"*85)

    # XGBoost - Entropy
    if XGBOOST_AVAILABLE:
        r = all_results['XGBoost']['Entropy']
        print(f"{'Entropy':<15} {'XGBoost':<12} {r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['fpr_at_95tpr']:>8.4f} "
              f"{all_results['XGBoost']['id_accuracy']:>10.4f} {all_results['XGBoost']['id_f1']:>10.4f}")

    # GRU - MC Dropout
    r = all_results['GRU']['MC_Dropout']
    print(f"{'MC Dropout':<15} {'GRU':<12} {r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['fpr_at_95tpr']:>8.4f} "
          f"{all_results['GRU']['id_accuracy']:>10.4f} {all_results['GRU']['id_f1']:>10.4f}")

    # LSTM+GNN - Entropy
    r = all_results['LSTM_GNN']['Entropy']
    print(f"{'Entropy':<15} {'LSTM+GNN':<12} {r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['fpr_at_95tpr']:>8.4f} "
          f"{all_results['LSTM_GNN']['id_accuracy']:>10.4f} {all_results['LSTM_GNN']['id_f1']:>10.4f}")

    # GAT - MC Dropout
    r = all_results['GAT']['MC_Dropout']
    print(f"{'MC Dropout':<15} {'GAT':<12} {r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['fpr_at_95tpr']:>8.4f} "
          f"{all_results['GAT']['id_accuracy']:>10.4f} {all_results['GAT']['id_f1']:>10.4f}")

    # STGCN - MC Dropout
    r = all_results['STGCN']['MC_Dropout']
    print(f"{'MC Dropout':<15} {'STGCN':<12} {r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['fpr_at_95tpr']:>8.4f} "
          f"{all_results['STGCN']['id_accuracy']:>10.4f} {all_results['STGCN']['id_f1']:>10.4f}")

    print("-"*85)
    print("Reference (BSTGCN):")
    print(f"{'Epistemic':<15} {'BSTGCN':<12} {'0.833':>8} {'0.813':>8} {'0.795':>8} {'0.9660':>10} {'0.9030':>10}")
    print("="*85)

    # Save results
    output_dir = Path('results/ood_strong_baselines')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_dir / f'results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nResults saved to {output_dir}/results_{timestamp}.pkl")

    return all_results


if __name__ == '__main__':
    main()

