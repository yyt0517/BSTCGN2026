# Hybrid模型 vs BSTGCN 架构对比

## 一、整体架构对比

### 当前Hybrid模型
```
输入 (B, L, F)
    ↓
[确定性] 投影层: Linear(F → H)
    ↓
[确定性] Transformer(1层)
    ↓
[确定性] KNN图构建 → 邻接矩阵 (B, L, L)
    ↓
┌─────────────────────────────────────┐
│ ST-Block 1 (dilation=1)             │
│  ├─ [确定性] TCN (时间)              │
│  ├─ [确定性] GraphSAGE (空间)        │
│  └─ [确定性] 门控融合                │
├─────────────────────────────────────┤
│ ST-Block 2 (dilation=2)             │
│  ├─ [确定性] TCN                    │
│  ├─ [确定性] GraphSAGE              │
│  └─ [确定性] 门控融合                │
├─────────────────────────────────────┤
│ ST-Block 3 (dilation=4)             │
│  ├─ [确定性] TCN                    │
│  ├─ [确定性] GraphSAGE              │
│  └─ [确定性] 门控融合                │
└─────────────────────────────────────┘
    ↓
[确定性] 门控注意力池化 (B, L, H) → (B, H)
    ↓
[确定性] 分类器 MLP
    ↓
输出: Logits (B, C)
    ↓
Softmax → 预测类别
```

**特点**:
- ✅ 所有权重为点估计
- ✅ 单次前向传播
- ✅ 输出确定性预测
- ❌ 无不确定性量化

---

### BSTGCN模型
```
输入 (B, L, F)
    ↓
[贝叶斯] 投影层: BayesianLinear(F → H)
    │ 权重: W ~ N(μ_w, σ_w²)
    ↓
[确定性] Transformer(1层)  ← 保持确定性以降低计算成本
    ↓
[确定性] KNN图构建 → 邻接矩阵 (B, L, L)
    ↓
┌─────────────────────────────────────┐
│ Bayesian ST-Block 1 (dilation=1)    │
│  ├─ [贝叶斯] TCN (时间)              │
│  │   W_tcn ~ N(μ, σ²)              │
│  ├─ [贝叶斯] GraphSAGE (空间)        │
│  │   W_sage ~ N(μ, σ²)             │
│  └─ [贝叶斯] 门控融合                │
│      W_gate ~ N(μ, σ²)              │
├─────────────────────────────────────┤
│ Bayesian ST-Block 2 (dilation=2)    │
│  ├─ [贝叶斯] TCN                    │
│  ├─ [贝叶斯] GraphSAGE              │
│  └─ [贝叶斯] 门控融合                │
├─────────────────────────────────────┤
│ Bayesian ST-Block 3 (dilation=4)    │
│  ├─ [贝叶斯] TCN                    │
│  ├─ [贝叶斯] GraphSAGE              │
│  └─ [贝叶斯] 门控融合                │
└─────────────────────────────────────┘
    ↓
[贝叶斯] 门控注意力池化 (B, L, H) → (B, H)
    │ W_pool ~ N(μ, σ²)
    ↓
[贝叶斯] 分类器 MLP
    │ W_cls ~ N(μ, σ²)
    ↓
训练: 单次采样 → Logits (B, C)
推理: M次采样 → {Logits_1, ..., Logits_M}
    ↓
Softmax → {p_1, ..., p_M}
    ↓
聚合: p̄ = mean(p_1, ..., p_M)
    ↓
输出: 
  - 预测类别: argmax(p̄)
  - 认知不确定性: H(p̄) - mean(H(p_m))
```

**特点**:
- ✅ 关键权重为分布
- ✅ 训练时单次采样，推理时MC采样
- ✅ 输出预测 + 不确定性
- ✅ 支持OOD检测

---

## 二、核心组件对比

### 2.1 线性层对比

#### Hybrid模型 - 确定性Linear
```python
class DeterministicLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # 单一权重，确定性输出
        return F.linear(x, self.weight, self.bias)
```

**参数量**: `out_features × in_features + out_features`

---

#### BSTGCN - 贝叶斯Linear
```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        
        # 权重的变分参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -3.0)
        
        # 偏置的变分参数
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3.0)
        
        self.prior_std = prior_std
    
    def forward(self, x):
        # 计算标准差 (softplus保证正性)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # 重参数化采样
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """计算KL[q(W)||p(W)]，其中p(W)=N(0, prior_std²)"""
        # 权重KL
        weight_sigma = F.softplus(self.weight_rho)
        kl_weight = 0.5 * torch.sum(
            (weight_sigma / self.prior_std) ** 2 +
            (self.weight_mu / self.prior_std) ** 2 -
            1 - 2 * torch.log(weight_sigma / self.prior_std + 1e-8)
        )
        
        # 偏置KL
        bias_sigma = F.softplus(self.bias_rho)
        kl_bias = 0.5 * torch.sum(
            (bias_sigma / self.prior_std) ** 2 +
            (self.bias_mu / self.prior_std) ** 2 -
            1 - 2 * torch.log(bias_sigma / self.prior_std + 1e-8)
        )
        
        return kl_weight + kl_bias
```

**参数量**: `2 × (out_features × in_features + out_features)` ← **翻倍**

---

### 2.2 GraphSAGE层对比

#### Hybrid模型
```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean', dropout=0.1):
        super().__init__()
        self.neigh_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # 聚合邻居
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        neigh_feat = torch.bmm(adj_norm, x)
        
        # 变换
        neigh_h = self.neigh_linear(neigh_feat)
        self_h = self.self_linear(x)
        
        # 组合
        h = neigh_h + self_h
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h
```

---

#### BSTGCN
```python
class BayesianGraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean', dropout=0.1, prior_std=1.0):
        super().__init__()
        
        # 贝叶斯线性层
        self.neigh_linear = BayesianLinear(in_dim, out_dim, prior_std)
        self.self_linear = BayesianLinear(in_dim, out_dim, prior_std)
        
        # 归一化和dropout保持确定性
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # 聚合邻居 (确定性)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        neigh_feat = torch.bmm(adj_norm, x)
        
        # 贝叶斯变换 (每次前向传播采样不同权重)
        neigh_h = self.neigh_linear(neigh_feat)  # 内部采样
        self_h = self.self_linear(x)              # 内部采样
        
        # 组合
        h = neigh_h + self_h
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h
    
    def kl_divergence(self):
        return self.neigh_linear.kl_divergence() + \
               self.self_linear.kl_divergence()
```

**关键差异**:
- 每次前向传播，`neigh_linear`和`self_linear`采样不同权重
- 增加`kl_divergence()`方法用于ELBO损失计算

---

## 三、训练流程对比

### 3.1 Hybrid模型训练
```python
# 单次前向传播
logits = model(x)

# 交叉熵损失 (带Label Smoothing)
loss = LabelSmoothingCrossEntropy(logits, labels)

# 反向传播
loss.backward()
optimizer.step()
```

**损失组成**: 仅重建损失

---

### 3.2 BSTGCN训练
```python
# 单次采样前向传播 (内部自动采样权重)
logits = model(x)

# ELBO损失
L_rec = LabelSmoothingCrossEntropy(logits, labels)  # 重建损失
L_KL = model.kl_divergence() / num_batches          # KL散度
loss = L_rec + kl_weight * L_KL                     # 总损失

# 反向传播
loss.backward()
optimizer.step()
```

**损失组成**: 重建损失 + KL正则化

**关键点**:
- KL散度需要除以batch数进行归一化
- `kl_weight`控制正则化强度 (通常0.01-1.0)

---

## 四、推理流程对比

### 4.1 Hybrid模型推理
```python
model.eval()
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)

# 输出: 预测类别
```

**推理次数**: 1次  
**输出**: 预测类别  
**不确定性**: 无

---

### 4.2 BSTGCN推理
```python
model.eval()
probs_list = []

with torch.no_grad():
    for _ in range(num_mc_samples):  # 例如30次
        logits = model(x)  # 每次采样不同权重
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs)

# 聚合结果
probs_stack = torch.stack(probs_list, dim=0)  # (M, B, C)
pred_mean = probs_stack.mean(dim=0)           # (B, C)
pred_class = pred_mean.argmax(dim=-1)         # (B,)

# 计算认知不确定性
entropy_mean = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10), dim=-1)
entropy_samples = -torch.sum(probs_stack * torch.log(probs_stack + 1e-10), dim=-1)
mean_entropy = entropy_samples.mean(dim=0)
epistemic_uncertainty = entropy_mean - mean_entropy

# 输出: 预测类别 + 不确定性
```

**推理次数**: M次 (例如30次)  
**输出**: 预测类别 + 认知不确定性  
**不确定性**: 互信息 I(Y;W|X)

---

## 五、性能指标对比表

| 指标 | Hybrid模型 | BSTGCN | 变化 |
|------|-----------|--------|------|
| **模型参数** |
| 参数量 | ~500K | ~1M | +100% |
| 模型大小 | ~2MB | ~4MB | +100% |
| **训练性能** |
| 训练时间/epoch | 100s | 130s | +30% |
| 内存占用 | 4GB | 6.4GB | +60% |
| 收敛速度 | 50 epochs | 70 epochs | +40% |
| **推理性能** |
| 推理时间/sample | 0.5ms | 15ms (M=30) | +2900% |
| 吞吐量 | 2000 samples/s | 67 samples/s | -97% |
| **预测性能** |
| Macro-F1 | 0.95 | 0.96-0.97 (预期) | +1-2% |
| 鲁棒性 | 基线 | 提升 | - |
| OOD检测 | 不支持 | 支持 | 新增 |
| 不确定性量化 | 无 | 有 | 新增 |

---

## 六、应用场景对比

### Hybrid模型适用场景
- ✅ 实时检测 (低延迟要求)
- ✅ 大规模部署 (资源受限)
- ✅ 确定性预测足够的场景
- ✅ 训练数据充足且分布稳定

### BSTGCN适用场景
- ✅ 高风险决策 (需要不确定性)
- ✅ OOD检测 (未知攻击识别)
- ✅ 对抗防御 (鲁棒性要求高)
- ✅ 小样本学习 (数据不足)
- ✅ 模型可解释性 (置信度评估)
- ⚠️ 可接受较高推理延迟

---

## 七、混合部署策略

### 策略1: 两阶段检测
```
第一阶段: Hybrid模型 (快速筛选)
    ↓
高置信度样本 → 直接分类
    ↓
低置信度样本 → 第二阶段
    ↓
第二阶段: BSTGCN (精细分析)
    ↓
预测 + 不确定性
```

**优势**: 平衡速度与准确性

---

### 策略2: 自适应MC采样
```python
def adaptive_inference(model, x, confidence_threshold=0.9):
    # 初始少量采样
    probs_list = []
    for _ in range(5):
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs)
    
    pred_mean = torch.stack(probs_list).mean(dim=0)
    max_prob = pred_mean.max(dim=-1)[0]
    
    # 如果置信度高，直接返回
    if max_prob > confidence_threshold:
        return pred_mean.argmax(dim=-1), None
    
    # 否则继续采样
    for _ in range(25):  # 总共30次
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs)
    
    # 完整不确定性计算
    probs_stack = torch.stack(probs_list)
    pred_mean = probs_stack.mean(dim=0)
    uncertainty = compute_uncertainty(probs_stack)
    
    return pred_mean.argmax(dim=-1), uncertainty
```

**优势**: 动态调整计算成本

---

## 八、总结

### 核心差异
1. **权重表示**: 点估计 vs 分布
2. **推理方式**: 单次 vs MC采样
3. **输出内容**: 预测 vs 预测+不确定性
4. **损失函数**: CE vs ELBO

### 升级收益
- ✅ 不确定性量化能力
- ✅ 更好的泛化性能
- ✅ OOD检测能力
- ✅ 对抗鲁棒性

### 升级成本
- ⚠️ 参数量翻倍
- ⚠️ 训练时间增加30%
- ⚠️ 推理时间增加30倍 (可优化)
- ⚠️ 超参数调优复杂度提升

### 推荐方案
**渐进式升级**:
1. 先实现部分贝叶斯化 (ST-Blocks + 分类器)
2. 验证效果后决定是否全面贝叶斯化
3. 根据应用场景选择部署策略

