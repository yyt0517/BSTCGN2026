# GNN模型改进说明

## 改进动机

树模型（LightGBM/XGBoost）在时间划分数据集上的SlowrateDoS召回率达到**70.61%**，而GNN只有**48.33%**。

**核心问题**：GNN理论上应该能够学习到稳定特征，但当前实现没有充分利用：
1. **边权重信息被丢弃**：FDG中的边权重（0.3-1.0）没有被使用
2. **所有特征被平等对待**：没有区分稳定特征和不稳定特征
3. **图卷积方式过于简单**：只是简单的邻居聚合

---

## 改进方案

### 1. 特征感知GNN架构

#### 核心思想

让GNN能够：
- **识别稳定特征**：Rate、SrcRate、DstRate等速率特征
- **利用边权重**：FDG中的边权重反映了特征之间的依赖强度
- **学习特征重要性**：通过可学习的权重参数

#### 实现细节

```python
class FeatureAwareGNN(nn.Module):
    def __init__(self, ..., stable_feature_indices, stable_feature_boost=2.0):
        # 1. 特征重要性权重（可学习）
        self.feature_importance = nn.Parameter(torch.ones(num_feat_nodes))
        
        # 2. 初始化稳定特征的权重更高
        for idx in stable_feature_indices:
            self.feature_importance[idx] = stable_feature_boost
        
        # 3. 在池化时使用特征重要性
        importance_weights = F.softmax(self.feature_importance, dim=0)
        H_graph = (H_feat * importance_weights).sum(dim=1)
```

### 2. 稳定特征定义

基于树模型的启发和领域知识，以下特征在时间划分下相对稳定：

#### 速率类特征（最重要！）
- **Rate**: 总体速率
- **SrcRate**: 源端速率
- **DstRate**: 目标端速率

**原因**：SlowrateDoS的本质是"慢速"，速率特征不随时间变化

#### 协议类特征
- **Proto**: 协议类型
- **Sport**: 源端口
- **Dport**: 目标端口

**原因**：攻击目标固定，协议特征稳定

#### 状态类特征
- **State**: 连接状态
- **Dir**: 方向

**原因**：连接模式固定

#### 平均包大小
- **sMeanPktSz**: 源端平均包大小
- **dMeanPktSz**: 目标端平均包大小

**原因**：攻击特征固定，包大小相对稳定

#### 其他稳定特征
- **pLoss**: 丢包率
- **sTtl**: 源端TTL
- **dTtl**: 目标端TTL

### 3. 边权重利用

#### 当前问题

```python
# 旧实现：所有边权重都是1.0
weights = np.ones(len(src))
A = sp.coo_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes))
```

#### 改进方案

```python
# 新实现：使用FDG中的边权重
edge_weight_dict = {}
for _, row in edges_df.iterrows():
    src_name = row['src']
    dst_name = row['dst']
    weight = row.get('weight', 1.0)  # 0.3-1.0
    edge_weight_dict[(src_name, dst_name)] = weight

# 为每条边分配权重
weights = [edge_weight_dict.get((src_name, dst_name), 1.0) 
           for src_name, dst_name in edges]
A = sp.coo_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes))
```

**效果**：
- 高权重边（1.0）：强依赖关系，信息传递更多
- 低权重边（0.3）：弱依赖关系，信息传递较少

---

## 预期效果

### 1. 提升SlowrateDoS召回率

**目标**：从48.33%提升到接近70%（接近树模型）

**原理**：
- 稳定特征（Rate等）的权重更高
- GNN会更关注这些特征
- 即使Duration、Packet_Count等特征变化，Rate仍然稳定
- 模型能够识别更多的SlowrateDoS样本

### 2. 学习到的特征重要性

训练结束后，模型会输出学习到的特征重要性：

```
学习到的特征重要性（Top 15）:
 1. [✓] Rate                : 0.085432
 2. [✓] SrcRate             : 0.078123
 3. [✓] DstRate             : 0.072456
 4. [ ] Duration            : 0.065234
 5. [✓] sMeanPktSz          : 0.058912
 ...
```

**验证**：
- 如果Rate、SrcRate、DstRate排名靠前 → 模型学到了稳定特征 ✓
- 如果Duration、Packet_Count排名靠前 → 模型仍然依赖不稳定特征 ✗

### 3. 更好的泛化能力

**训练集**（0-12.7分钟）：
- 模型学习到Rate < 10的规则
- 而不是Duration ∈ [5, 15]的规则

**测试集**（32.4-60分钟）：
- Rate仍然 < 10 → 能够识别 ✓
- Duration变成 [30, 50] → 但不影响识别

---

## 实现对比

### 旧实现（简化GNN）

```python
class OptimizedGATv2Classifier(nn.Module):
    def forward(self, node_feats):
        H = self.input_proj(node_feats)
        
        for layer in self.layers:
            H_neigh = self.spmm(H)  # 简单邻居聚合
            H = layer(H_neigh)
            H = F.relu(H)
        
        H_feat = H[:, :41, :]
        H_graph = H_feat.mean(dim=1)  # 简单平均池化
        
        logits = self.head(H_graph)
        return logits
```

**问题**：
- 所有特征权重相同
- 边权重被忽略
- 无法区分稳定和不稳定特征

### 新实现（特征感知GNN）

```python
class FeatureAwareGNN(nn.Module):
    def __init__(self, ..., stable_feature_indices, stable_feature_boost):
        # 特征重要性权重（可学习）
        self.feature_importance = nn.Parameter(torch.ones(num_feat_nodes))
        
        # 初始化稳定特征权重更高
        for idx in stable_feature_indices:
            self.feature_importance[idx] = stable_feature_boost
    
    def forward(self, node_feats):
        H = self.input_proj(node_feats)
        
        for layer in self.layers:
            H_neigh = self.spmm(H)  # 使用边权重的邻居聚合
            H = layer(H_neigh)
            H = F.relu(H)
        
        H_feat = H[:, :41, :]
        
        # 加权池化（使用特征重要性）
        importance_weights = F.softmax(self.feature_importance, dim=0)
        H_graph = (H_feat * importance_weights).sum(dim=1)
        
        logits = self.head(H_graph)
        return logits
```

**改进**：
- ✅ 稳定特征权重更高（初始化为2.0x）
- ✅ 使用边权重进行邻居聚合
- ✅ 加权池化，关注重要特征
- ✅ 可学习的特征重要性

---

## 配置参数

### 新增参数

```python
@dataclass
class Config:
    # 特征稳定性感知
    use_edge_weights: bool = True  # 使用边权重
    use_feature_importance: bool = True  # 使用特征重要性
    stable_feature_boost: float = 2.0  # 稳定特征的权重提升
    
    # 稳定特征列表
    stable_features: list = [
        'Rate', 'SrcRate', 'DstRate',  # 速率特征（最重要！）
        'Proto', 'Sport', 'Dport',  # 协议特征
        'State', 'Dir',  # 状态特征
        'sMeanPktSz', 'dMeanPktSz',  # 平均包大小
        'pLoss', 'sTtl', 'dTtl',  # 其他稳定特征
    ]
```

### 调优建议

1. **stable_feature_boost**：
   - 当前值：2.0
   - 如果效果不好，可以尝试：3.0、5.0、10.0
   - 更高的值 → 更关注稳定特征

2. **stable_features列表**：
   - 当前基于领域知识和树模型启发
   - 可以根据训练后的特征重要性调整
   - 添加或删除特征

---

## 验证方法

### 1. 性能指标

| 指标 | 旧模型 | 新模型 | 目标 |
|------|--------|--------|------|
| SlowrateDoS召回率 | 48.33% | ? | >60% |
| 整体准确率 | 91.98% | ? | >92% |
| Macro-F1 | 82.80% | ? | >85% |

### 2. 特征重要性分析

**期望**：
- Rate、SrcRate、DstRate排名Top 5
- 稳定特征的平均重要性 > 不稳定特征

**如果不符合期望**：
- 增加stable_feature_boost
- 调整stable_features列表
- 检查边权重是否正确加载

### 3. 消融实验

可以尝试以下变体来验证每个改进的贡献：

1. **只使用边权重**：
   - stable_feature_boost = 1.0
   - 验证边权重的作用

2. **只使用特征重要性**：
   - 不使用边权重（所有边权重=1.0）
   - 验证特征重要性的作用

3. **完整模型**：
   - 边权重 + 特征重要性
   - 验证组合效果

---

## 总结

### 核心改进

1. **特征重要性感知**：可学习的特征权重，稳定特征初始权重更高
2. **边权重利用**：使用FDG中的边权重进行邻居聚合
3. **加权池化**：根据特征重要性进行加权平均

### 理论优势

- **更接近树模型的思想**：关注稳定特征
- **保留GNN的优势**：建模特征依赖关系
- **可解释性**：输出特征重要性

### 预期结果

- SlowrateDoS召回率：48.33% → 60-70%
- 整体性能：保持或提升
- 学习到的特征重要性：验证稳定特征的作用

---

**实验时间**: 2025-11-07  
**改进者**: Augment Agent  
**目标**: 让GNN能够像树模型一样找到稳定特征

