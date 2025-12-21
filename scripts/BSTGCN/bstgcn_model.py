"""
BSTGCN完整模型实现

贝叶斯时空图神经网络，用于网络流量分类

================================================================================
命名映射表 (代码 -> 论文)
================================================================================
| 代码名称   | 论文名称    | 说明                                           |
|------------|-------------|------------------------------------------------|
| STGCN      | Seq-STGCN   | 串行时空建模 (先temporal后spatial)              |
| Hybrid     | STGCN       | 并行时空建模 (temporal和spatial并行处理)        |
| BSTGCN     | BSTGCN      | 贝叶斯并行时空建模 = Bayesian + STGCN(论文)     |
================================================================================
论文命名逻辑: BSTGCN = Bayesian + STGCN, 所以STGCN(论文)是BSTGCN的确定性版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bayesian_layers import BayesianLinear
from .bayesian_components import (
    BayesianGraphSAGELayer,
    BayesianDepthwiseSeparableTCN,
    BayesianEfficientSTBlock,
    BayesianGatedAttentionPooling
)


class KNNGraphBuilder(nn.Module):
    """KNN图构建器 (确定性)"""
    def __init__(self, k=10):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        """
        构建KNN图
        
        Args:
            x: 输入特征 (batch, seq_len, hidden_dim)
        
        Returns:
            邻接矩阵 (batch, seq_len, seq_len)
        """
        B, L, D = x.shape
        
        # 计算余弦相似度
        x_norm = F.normalize(x, p=2, dim=-1)
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, L, L)
        
        # 选择top-k邻居
        _, topk_indices = torch.topk(sim_matrix, k=min(self.k, L), dim=-1)
        
        # 构建邻接矩阵
        adj = torch.zeros_like(sim_matrix)
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, min(self.k, L))
        node_indices = torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, min(self.k, L))
        adj[batch_indices, node_indices, topk_indices] = 1.0
        
        # 对称化
        adj = (adj + adj.transpose(1, 2)) / 2.0
        
        return adj


class BSTGCNModel(nn.Module):
    """
    贝叶斯时空图神经网络模型 (支持Heteroscedastic建模)

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏维度
        num_heads: Transformer注意力头数
        num_st_blocks: ST块数量
        num_classes: 分类类别数
        dropout: Dropout概率
        k_neighbors: KNN邻居数
        sage_aggr: GraphSAGE聚合方式
        prior_std: 先验标准差
        heteroscedastic: 是否启用heteroscedastic建模 (学习数据不确定性)
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_heads=8,
        num_st_blocks=3,
        num_classes=10,
        dropout=0.15,
        k_neighbors=10,
        sage_aggr='mean',
        prior_std=1.0,
        heteroscedastic=False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_st_blocks = num_st_blocks
        self.num_classes = num_classes
        self.heteroscedastic = heteroscedastic

        # 输入投影 (贝叶斯)
        self.input_proj = BayesianLinear(input_dim, hidden_dim, prior_std=prior_std)

        # Transformer编码器 (保持确定性以降低计算成本)
        # 增加到2层以提高表达能力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # KNN图构建器 (确定性)
        self.graph_builder = KNNGraphBuilder(k=k_neighbors)

        # 贝叶斯ST块
        self.st_blocks = nn.ModuleList([
            BayesianEfficientSTBlock(
                hidden_dim=hidden_dim,
                kernel_size=3,
                dilation=2**i,
                aggr=sage_aggr,
                dropout=dropout,
                prior_std=prior_std
            )
            for i in range(num_st_blocks)
        ])

        # 贝叶斯门控注意力池化
        self.pooling = BayesianGatedAttentionPooling(
            hidden_dim=hidden_dim,
            dropout=dropout,
            prior_std=prior_std
        )

        # 贝叶斯分类器 (输出logits)
        self.classifier = nn.Sequential(
            BayesianLinear(hidden_dim, hidden_dim // 2, prior_std=prior_std),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            BayesianLinear(hidden_dim // 2, num_classes, prior_std=prior_std)
        )

        # Heteroscedastic: 方差预测头 (学习每个样本的数据不确定性)
        if heteroscedastic:
            self.variance_head = nn.Sequential(
                BayesianLinear(hidden_dim, hidden_dim // 2, prior_std=prior_std),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                BayesianLinear(hidden_dim // 2, num_classes, prior_std=prior_std)
            )
    
    def forward(self, x, sample=True, return_variance=False):
        """
        前向传播

        Args:
            x: 输入 (batch, seq_len, input_dim)
            sample: 是否采样权重 (训练时True，MAP推理时False)
            return_variance: 是否返回学习到的方差 (仅heteroscedastic模式有效)

        Returns:
            如果 return_variance=False 或 heteroscedastic=False:
                logits (batch, num_classes)
            如果 return_variance=True 且 heteroscedastic=True:
                logits (batch, num_classes), log_variance (batch, num_classes)
        """
        # 输入投影
        x = self.input_proj(x, sample=sample)  # (B, L, D)

        # Transformer编码 (确定性)
        x = self.transformer(x)  # (B, L, D)

        # 构建KNN图 (确定性)
        adj = self.graph_builder(x)  # (B, L, L)

        # 贝叶斯ST块
        for st_block in self.st_blocks:
            x = st_block(x, adj, sample=sample)  # (B, L, D)

        # 贝叶斯池化
        pooled = self.pooling(x, sample=sample)  # (B, D)

        # 贝叶斯分类
        h = self.classifier[0](pooled, sample=sample)  # (B, D//2)
        h = self.classifier[1](h)  # LayerNorm
        h = self.classifier[2](h)  # GELU
        h = self.classifier[3](h)  # Dropout
        logits = self.classifier[4](h, sample=sample)  # (B, num_classes)

        # Heteroscedastic: 返回方差
        if self.heteroscedastic and return_variance:
            h_var = self.variance_head[0](pooled, sample=sample)  # (B, D//2)
            h_var = self.variance_head[1](h_var)  # LayerNorm
            h_var = self.variance_head[2](h_var)  # GELU
            h_var = self.variance_head[3](h_var)  # Dropout
            log_variance = self.variance_head[4](h_var, sample=sample)  # (B, num_classes)
            return logits, log_variance

        return logits

    def kl_divergence(self):
        """计算所有贝叶斯层的KL散度"""
        kl = 0.0

        # 输入投影
        kl += self.input_proj.kl_divergence()

        # ST块
        for st_block in self.st_blocks:
            kl += st_block.kl_divergence()

        # 池化
        kl += self.pooling.kl_divergence()

        # 分类器
        kl += self.classifier[0].kl_divergence()
        kl += self.classifier[4].kl_divergence()

        # 方差预测头 (如果启用heteroscedastic)
        if self.heteroscedastic:
            kl += self.variance_head[0].kl_divergence()
            kl += self.variance_head[4].kl_divergence()

        return kl

    def get_num_params(self):
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_from_hybrid(self, hybrid_state_dict, init_rho=-3.0):
        """
        从Hybrid模型加载权重，初始化贝叶斯层的均值参数

        Args:
            hybrid_state_dict: Hybrid模型的state_dict
            init_rho: 初始化rho参数的值 (sigma = softplus(rho) ≈ 0.05 when rho=-3)

        映射关系:
            Hybrid Linear层 -> BSTGCN BayesianLinear层
            weight -> weight_mu
            bias -> bias_mu
            rho参数初始化为init_rho (对应小方差)
        """
        own_state = self.state_dict()
        loaded_count = 0

        # 创建映射表: Hybrid参数名 -> BSTGCN参数名
        mapping = {}
        for name in hybrid_state_dict.keys():
            # 将 .weight 映射到 .weight_mu, .bias 映射到 .bias_mu
            if '.weight' in name and '_mu' not in name and '_rho' not in name:
                mu_name = name.replace('.weight', '.weight_mu')
                if mu_name in own_state:
                    mapping[name] = mu_name
            elif '.bias' in name and '_mu' not in name and '_rho' not in name:
                mu_name = name.replace('.bias', '.bias_mu')
                if mu_name in own_state:
                    mapping[name] = mu_name

        # 加载权重
        for hybrid_name, bstgcn_name in mapping.items():
            if hybrid_state_dict[hybrid_name].shape == own_state[bstgcn_name].shape:
                own_state[bstgcn_name].copy_(hybrid_state_dict[hybrid_name])
                loaded_count += 1

                # 对应的rho参数初始化为小值
                rho_name = bstgcn_name.replace('_mu', '_rho')
                if rho_name in own_state:
                    own_state[rho_name].fill_(init_rho)

        # 加载非贝叶斯层的权重 (Transformer, LayerNorm等)
        for name, param in hybrid_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                if name not in mapping:  # 还没加载过
                    own_state[name].copy_(param)
                    loaded_count += 1

        self.load_state_dict(own_state)
        return loaded_count

