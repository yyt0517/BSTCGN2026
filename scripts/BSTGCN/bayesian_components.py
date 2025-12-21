"""
贝叶斯时空组件实现

包含:
- BayesianGraphSAGELayer: 贝叶斯GraphSAGE层
- BayesianDepthwiseSeparableTCN: 贝叶斯深度可分离时间卷积
- BayesianEfficientSTBlock: 贝叶斯时空并行交互块
- BayesianGatedAttentionPooling: 贝叶斯门控注意力池化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bayesian_layers import BayesianLinear, BayesianConv1d


class BayesianGraphSAGELayer(nn.Module):
    """
    贝叶斯GraphSAGE层
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        aggr: 聚合方式 ('mean' or 'max')
        dropout: Dropout概率
        prior_std: 先验标准差
    """
    def __init__(self, in_dim, out_dim, aggr='mean', dropout=0.1, prior_std=1.0):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr
        
        # 贝叶斯线性层
        self.neigh_linear = BayesianLinear(in_dim, out_dim, bias=False, prior_std=prior_std)
        self.self_linear = BayesianLinear(in_dim, out_dim, bias=False, prior_std=prior_std)
        
        # 归一化和dropout保持确定性
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj, sample=True):
        """
        前向传播
        
        Args:
            x: 节点特征 (batch, num_nodes, in_dim)
            adj: 邻接矩阵 (batch, num_nodes, num_nodes)
            sample: 是否采样权重
        
        Returns:
            输出 (batch, num_nodes, out_dim)
        """
        B, N, _ = x.shape
        
        # 归一化邻接矩阵 (确定性)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        
        # 聚合邻居特征 (确定性)
        if self.aggr == 'mean':
            neigh_feat = torch.bmm(adj_norm, x)
        elif self.aggr == 'max':
            adj_expanded = adj.unsqueeze(-1)  # (B, N, N, 1)
            x_expanded = x.unsqueeze(1)  # (B, 1, N, D)
            masked_feat = x_expanded * adj_expanded  # (B, N, N, D)
            neigh_feat = masked_feat.max(dim=2)[0]  # (B, N, D)
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggr}")
        
        # 贝叶斯变换
        neigh_h = self.neigh_linear(neigh_feat, sample=sample)
        self_h = self.self_linear(x, sample=sample)
        
        # 组合
        h = neigh_h + self_h
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h
    
    def kl_divergence(self):
        """计算KL散度"""
        return self.neigh_linear.kl_divergence() + self.self_linear.kl_divergence()


class BayesianDepthwiseSeparableTCN(nn.Module):
    """
    贝叶斯深度可分离时间卷积
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        dilation: 膨胀率
        dropout: Dropout概率
        prior_std: 先验标准差
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 dilation=1, dropout=0.1, prior_std=1.0):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        # Depthwise convolution (贝叶斯)
        self.depthwise = BayesianConv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, groups=in_channels,
            prior_std=prior_std
        )
        
        # Pointwise convolution (贝叶斯)
        self.pointwise = BayesianConv1d(
            in_channels, out_channels, 1,
            prior_std=prior_std
        )
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sample=True):
        """
        前向传播
        
        Args:
            x: 输入 (batch, seq_len, channels)
            sample: 是否采样权重
        
        Returns:
            输出 (batch, seq_len, channels)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (B, C, L)
        
        x = self.depthwise(x, sample=sample)
        x = self.pointwise(x, sample=sample)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (B, L, C)
        return x
    
    def kl_divergence(self):
        """计算KL散度"""
        return self.depthwise.kl_divergence() + self.pointwise.kl_divergence()


class BayesianEfficientSTBlock(nn.Module):
    """
    贝叶斯时空并行交互块

    Args:
        hidden_dim: 隐藏维度
        kernel_size: TCN卷积核大小
        dilation: TCN膨胀率
        aggr: GraphSAGE聚合方式
        dropout: Dropout概率
        prior_std: 先验标准差
    """
    def __init__(self, hidden_dim, kernel_size=3, dilation=1,
                 aggr='mean', dropout=0.1, prior_std=1.0):
        super().__init__()

        # Temporal branch (贝叶斯TCN)
        self.temporal_conv = BayesianDepthwiseSeparableTCN(
            hidden_dim, hidden_dim, kernel_size, dilation, dropout, prior_std
        )

        # Spatial branch (贝叶斯GraphSAGE)
        self.spatial_sage = BayesianGraphSAGELayer(
            hidden_dim, hidden_dim, aggr=aggr, dropout=dropout, prior_std=prior_std
        )

        # Gated fusion (贝叶斯)
        self.gate_temporal = BayesianLinear(hidden_dim, hidden_dim, prior_std=prior_std)
        self.gate_spatial = BayesianLinear(hidden_dim, hidden_dim, prior_std=prior_std)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, sample=True):
        """
        前向传播

        Args:
            x: 输入 (batch, seq_len, hidden_dim)
            adj: 邻接矩阵 (batch, seq_len, seq_len)
            sample: 是否采样权重

        Returns:
            输出 (batch, seq_len, hidden_dim)
        """
        residual = x

        # Parallel temporal and spatial processing
        temporal_out = self.temporal_conv(x, sample=sample)
        spatial_out = self.spatial_sage(x, adj, sample=sample)

        # Gated fusion
        gate_t = torch.sigmoid(self.gate_temporal(x, sample=sample))
        gate_s = torch.sigmoid(self.gate_spatial(x, sample=sample))

        # Normalize gates
        gate_sum = gate_t + gate_s + 1e-8
        gate_t = gate_t / gate_sum
        gate_s = gate_s / gate_sum

        # Fused output
        out = gate_t * temporal_out + gate_s * spatial_out
        out = self.norm(out + residual)

        return out

    def kl_divergence(self):
        """计算KL散度"""
        kl = self.temporal_conv.kl_divergence()
        kl += self.spatial_sage.kl_divergence()
        kl += self.gate_temporal.kl_divergence()
        kl += self.gate_spatial.kl_divergence()
        return kl


class BayesianGatedAttentionPooling(nn.Module):
    """
    贝叶斯门控注意力池化

    Args:
        hidden_dim: 隐藏维度
        dropout: Dropout概率
        prior_std: 先验标准差
    """
    def __init__(self, hidden_dim, dropout=0.1, prior_std=1.0):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 注意力机制 (贝叶斯)
        self.attention = BayesianLinear(hidden_dim, 1, prior_std=prior_std)

        # 门控机制 (贝叶斯)
        self.gate = nn.Sequential(
            BayesianLinear(hidden_dim, hidden_dim, prior_std=prior_std),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sample=True):
        """
        前向传播

        Args:
            x: 输入 (batch, seq_len, hidden_dim)
            sample: 是否采样权重

        Returns:
            输出 (batch, hidden_dim)
        """
        # 计算注意力分数
        attn_scores = self.attention(x, sample=sample)  # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L, 1)

        # 加权聚合
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, D)

        # 门控
        gate_values = self.gate[0](pooled, sample=sample)
        gate_values = self.gate[1](gate_values)

        output = gate_values * pooled
        output = self.dropout(output)

        return output

    def kl_divergence(self):
        """计算KL散度"""
        return self.attention.kl_divergence() + self.gate[0].kl_divergence()

