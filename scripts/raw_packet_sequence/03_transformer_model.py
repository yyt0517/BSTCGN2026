"""
基于Transformer的原始包序列分类模型
使用多头自注意力机制捕捉包之间的依赖关系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PacketTransformer(nn.Module):
    """
    基于Transformer的包序列分类模型
    
    Architecture:
        Input: (batch, seq_len, num_features)
        ↓ Feature Embedding
        (batch, seq_len, d_model)
        ↓ Positional Encoding
        ↓ Transformer Encoder (N layers)
        ↓ Global Average Pooling
        (batch, d_model)
        ↓ Classifier
        Output: (batch, num_classes)
    """
    def __init__(self, num_features, num_classes, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        # Feature embedding
        x = self.feature_embedding(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits

class EnhancedPacketTransformer(nn.Module):
    """
    增强版Transformer模型
    添加了多尺度特征提取和注意力池化
    """
    def __init__(self, num_features, num_classes, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Feature embedding (deeper)
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier (deeper)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        # Feature embedding
        x = self.feature_embedding(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Attention pooling
        attention_weights = self.attention_pooling(x)  # (batch, seq_len, 1)
        x = (x * attention_weights).sum(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits

class LSTMAttentionModel(nn.Module):
    """
    LSTM + Attention模型（作为对比基线）
    """
    def __init__(self, num_features, num_classes, hidden_dim=128, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Attention
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = (lstm_out * attention_weights).sum(dim=1)  # (batch, hidden_dim*2)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits

def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test models
    batch_size = 32
    seq_len = 100
    num_features = 25  # Updated to 25 core features
    num_classes = 9

    x = torch.randn(batch_size, seq_len, num_features)

    print("="*100)
    print("Model Architecture Test".center(100))
    print("="*100)

    # Test PacketTransformer
    print("\n1. PacketTransformer")
    model1 = PacketTransformer(num_features, num_classes, d_model=128, nhead=8, num_layers=4)
    out1 = model1(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out1.shape}")
    print(f"   Parameters: {count_parameters(model1):,}")

    # Test EnhancedPacketTransformer
    print("\n2. EnhancedPacketTransformer")
    model2 = EnhancedPacketTransformer(num_features, num_classes, d_model=128, nhead=8, num_layers=4)
    out2 = model2(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {count_parameters(model2):,}")

    # Test LSTMAttentionModel
    print("\n3. LSTMAttentionModel")
    model3 = LSTMAttentionModel(num_features, num_classes, hidden_dim=128, num_layers=2)
    out3 = model3(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out3.shape}")
    print(f"   Parameters: {count_parameters(model3):,}")

    print("\n" + "="*100)
    print("All models tested successfully!".center(100))
    print("="*100)

