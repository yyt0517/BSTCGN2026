# Hybrid 模型改进说明

## 问题诊断

原始 Hybrid 模型表现不如 STGCN 的主要原因：

### 1. 串行信息流导致时空瓶颈
- **问题**：Transformer 先处理时序，GNN 后处理空间，信息流是串行的
- **影响**：空间邻域无法在时间卷积的每一层即时作用，错过了并行耦合的增益
- **证据**：STGCN 的核心优势在于层内并行（Temporal Conv → Spatial Conv → Temporal Conv）

### 2. GATv2 实现过于简化
- **问题**：
  - 使用 `attn(h)` 生成节点打分，再广播成 (N,N,heads)
  - 提前把多头 `mean(dim=-1)` 掉了，没有 per-head V，没有 concat
  - 等价于"节点打分的广播注意力"
- **影响**：表达力被明显削弱，与标准 GATv2 差异很大
- **位置**：`hybrid_model_01.py` 第 127-197 行

### 3. "静态 KNN 图"实际上每次前向都重建
- **问题**：
  - 文档写"使用静态 KNN 图"，注释写"只构建一次"
  - 但 `forward()` 里每个 batch 都调用 `build_knn_graph()` 重算邻接
- **影响**：
  - 邻接随表征漂移，不稳定
  - 计算开销大
  - 与"静态"相悖

### 4. 读出机制过于简化
- **问题**：
  - `__init__` 里定义了 `AdaptiveAvgPool1d(1)` 但未使用
  - 前向直接 `mean(dim=1)` 均值池化
- **影响**：忽略了可学习/稳健读出，对长尾类别不友好

### 5. 训练侧可优化空间
- **现状**：已有类别权重、Plateau LR、早停
- **可改进**：Label Smoothing、Cosine LR with Warmup、更稳定的调度

---

## 改进方案

### A. 并行时空交互（ST-Block）✅

**实现**：
```python
class STBlock(nn.Module):
    """时空并行交互块"""
    def forward(self, x, adj):
        # 并行处理
        temporal_out = self.temporal_conv(x)  # TCN
        spatial_out = self.spatial_gat(x, adj)  # GATv2
        
        # 门控融合
        gate_t = self.gate_temporal(x)
        gate_s = self.gate_spatial(x)
        
        # 归一化门控
        gate_sum = gate_t + gate_s + 1e-8
        gate_t = gate_t / gate_sum
        gate_s = gate_s / gate_sum
        
        # 融合输出
        out = gate_t * temporal_out + gate_s * spatial_out
        out = self.norm(out + residual)
        
        return out
```

**优势**：
- 让空间上下文在每一层就影响时间特征提取
- 复现 STGCN 的核心优势（层内并行）
- 保留 Hybrid 的灵活性

**预期收益**：+2%~+3% Macro-F1

---

### B. 修正的 GATv2 实现 ✅

**关键改进**：
1. **Per-head 线性变换**：`W` 映射到 `head_dim * num_heads`
2. **Per-head 注意力参数**：`attn_src` 和 `attn_dst` 都是 `(num_heads, head_dim)`
3. **Per-head 聚合**：每个头独立计算注意力和聚合
4. **Concat 多头**：最后 concat 所有头，而不是提前 mean

**实现**：
```python
class CorrectedGATv2Layer(nn.Module):
    def forward(self, x, adj):
        # Linear transformation and split into heads
        h = self.W(x).view(B, N, self.num_heads, self.head_dim)
        
        # Per-head attention: e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        attn_src = (h * self.attn_src).sum(dim=-1)  # (B, N, H)
        attn_dst = (h * self.attn_dst).sum(dim=-1)  # (B, N, H)
        attn_scores = self.leaky_relu(attn_src.unsqueeze(2) + attn_dst.unsqueeze(1))
        
        # Apply mask and softmax
        attn_weights = F.softmax(attn_scores * mask, dim=2)
        
        # Per-head aggregation
        out = torch.matmul(attn_weights, h)  # (B, H, N, D)
        
        # Concat heads
        out = out.reshape(B, N, self.out_dim)
        
        return out
```

**预期收益**：+1%~+2% Macro-F1

---

### C. 静态+动态混合图（带缓存）✅

**实现**：
```python
class HybridGraphBuilder(nn.Module):
    def __init__(self):
        # Cache for adjacency matrix
        self.register_buffer('cached_adj', None)
        self.cache_valid = False
    
    def forward(self, x, use_cache=True):
        # Build dynamic KNN graph
        if use_cache and self.cache_valid:
            adj_dynamic = self.cached_adj
        else:
            adj_dynamic = self.build_knn_graph(x, self.k_neighbors)
            if use_cache:
                self.cached_adj = adj_dynamic.detach()
                self.cache_valid = True
        
        return adj_dynamic
    
    def reset_cache(self):
        """每个 epoch 开始时调用"""
        self.cache_valid = False
```

**优势**：
- 训练时缓存邻接矩阵，每个 epoch 只构建一次
- 推理时完全复用缓存
- 更稳定、更高效

**预期收益**：+0.5%~+1% Macro-F1，方差下降

---

### D. 门控注意力池化 ✅

**实现**：
```python
class GatedAttentionPooling(nn.Module):
    def forward(self, x):
        # Attention scores
        attn_scores = self.attention(x)  # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (x * attn_weights).sum(dim=1)  # (B, H)
        
        # Gate
        gate = self.gate(context)  # (B, H)
        mean_pool = x.mean(dim=1)  # (B, H)
        
        # Gated combination
        out = gate * context + (1 - gate) * mean_pool
        
        return out
```

**优势**：
- 结合注意力池化和均值池化
- 门控机制自适应选择
- 对长尾类别更友好

**预期收益**：+0.3%~+0.8% Macro-F1

---

### E. 训练侧优化 ✅

#### 1. Label Smoothing
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def forward(self, logits, target):
        # One-hot with smoothing
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(epsilon / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - epsilon)
        
        # Apply class weights
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)
        
        loss = -torch.sum(true_dist * log_probs, dim=-1).mean()
        return loss
```

**优势**：
- 减轻过拟合和过自信
- 与类别权重兼容

#### 2. Cosine LR with Warmup
```python
class CosineWarmupScheduler:
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
                    for base_lr in self.base_lrs]
```

**优势**：
- 更平滑的收敛
- Warmup 避免初期不稳定
- Cosine 退火在后期精细调整

**预期收益**：+0.5%~+1% Macro-F1

---

### F. 轻量 TCN 替代部分 Transformer ✅

**实现**：
- 保留 1 层 Transformer 用于全局依赖
- 主干使用深度可分离 TCN（dilation 1/2/4）
- 在 ST-Block 中并行使用

**优势**：
- 对网络流量这类近邻模式强的任务，卷积更契合
- 参数更少，训练更快
- 多尺度感受野（通过 dilation）

---

## 架构对比

### 原始 Hybrid 模型
```
Input → Transformer (3层) → Graph Builder → GNN (2层) → Attention Pool → Pattern Memory → Fusion → Classifier
```

**问题**：
- 串行时空处理
- 简化的 GATv2
- 每次前向重建图
- 简单均值池化

---

### 改进版 Hybrid 模型
```
Input → Projection → Transformer (1层) → [ST-Block × 3] → Gated Attention Pool → Classifier
                                            ↓
                                    (Temporal TCN ‖ Spatial GATv2)
                                            ↓
                                      Gated Fusion
```

**改进**：
- ✅ 并行时空交互（ST-Block）
- ✅ 修正的 GATv2（真正的多头注意力）
- ✅ 缓存的混合图
- ✅ 门控注意力池化
- ✅ Label Smoothing + Cosine LR

---

## 预期性能提升

| 改进项 | 预期收益 | 复杂度 |
|--------|----------|--------|
| 并行时空交互（ST-Block） | +2%~+3% | 中 |
| 修正的 GATv2 | +1%~+2% | 低 |
| 混合图+缓存 | +0.5%~+1% | 低 |
| 门控注意力池化 | +0.3%~+0.8% | 低 |
| Label Smoothing + Cosine LR | +0.5%~+1% | 低 |
| **总计** | **+4.3%~+7.8%** | - |

**保守估计**：如果原始 Hybrid 的 Macro-F1 为 0.90，改进后应该能达到 **0.94~0.96**，接近或超过 STGCN 的 0.93~0.94。

---

## 使用方法

### 1. 训练改进版模型
```bash
conda activate gnn_py312
python scripts/hybrid_model/02_train_hybrid_improved.py
```

### 2. 配置参数
在 `02_train_hybrid_improved.py` 中修改：
```python
# 模型参数
HIDDEN_DIM = 128
NUM_HEADS = 8
NUM_ST_BLOCKS = 3  # ST-Block 层数
DROPOUT = 0.2
K_NEIGHBORS = 10

# 训练参数
LEARNING_RATE = 0.001
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.05
```

### 3. 查看结果
结果保存在 `baseline/results/hybrid_model_improved/`：
- `best_model.pth`：最佳模型权重
- `evaluation_report.txt`：详细评估报告
- `config.json`：训练配置
- `predictions.npy`：测试集预测

---

## 下一步优化方向

如果改进后仍有差距，可以考虑：

1. **引入 FDG 先验图**：
   - 使用 FDG 的关系类型作为静态先验
   - 与动态 KNN 图混合

2. **SWA/EMA**：
   - 最后 10~20 epoch 开启
   - 稳定泛化性能

3. **ArcFace/AM-Softmax**：
   - 替换分类器
   - 增强类别间区分度

4. **更深的 ST-Block**：
   - 增加到 4~5 层
   - 配合残差连接

5. **多尺度特征融合**：
   - 不同 dilation 的 TCN 输出融合
   - 类似 Inception 结构

---

## 参考

- **STGCN 优化版**：`baseline/stgcn_optimized.py`
- **原始 Hybrid 模型**：`scripts/hybrid_model/hybrid_model_01.py`
- **改进版 Hybrid 模型**：`scripts/hybrid_model/hybrid_model_improved.py`

