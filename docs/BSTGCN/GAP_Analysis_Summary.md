# BSTGCN GAP分析总结

## 执行摘要

本文档提供从当前Hybrid模型升级到BSTGCN的GAP分析总结表格，便于快速理解差异和实施要点。

---

## 一、核心GAP对比表

| GAP编号 | 维度 | 当前Hybrid模型 | BSTGCN目标 | 优先级 | 实施难度 | 影响范围 |
|---------|------|---------------|-----------|--------|---------|---------|
| **GAP-1** | 权重表示 | 点估计 W | 分布 q(W;μ,ρ) | 🔴 最高 | ⭐⭐⭐⭐ | 所有层 |
| **GAP-2** | 推理方式 | 单次前向 | MC采样 M次 | 🔴 最高 | ⭐⭐⭐ | 推理流程 |
| **GAP-3** | 损失函数 | CrossEntropy | ELBO (CE+KL) | 🔴 最高 | ⭐⭐⭐ | 训练流程 |
| **GAP-4** | 输出内容 | 预测类别 | 预测+不确定性 | 🟡 高 | ⭐⭐ | 评估流程 |

---

## 二、详细GAP分析

### GAP-1: 权重表示 (确定性 → 贝叶斯)

#### 当前状态
```python
# 标准PyTorch层
self.linear = nn.Linear(in_dim, out_dim)
# 权重: W ∈ R^(out_dim × in_dim)
# 单一点估计
```

#### 目标状态
```python
# 贝叶斯层
self.linear = BayesianLinear(in_dim, out_dim)
# 权重参数: μ_W, ρ_W ∈ R^(out_dim × in_dim)
# 分布: W ~ N(μ_W, softplus(ρ_W)²)
```

#### 需要修改的组件

| 组件 | 当前实现 | 需要替换为 | 优先级 |
|------|---------|-----------|--------|
| 输入投影 | `nn.Linear` | `BayesianLinear` | P1 |
| GraphSAGE层 | `nn.Linear` × 2 | `BayesianLinear` × 2 | P0 |
| TCN层 | `nn.Conv1d` × 2 | `BayesianConv1d` × 2 | P0 |
| 门控网络 | `nn.Linear` | `BayesianLinear` | P1 |
| 注意力池化 | `nn.Linear` × 2 | `BayesianLinear` × 2 | P1 |
| 分类器 | `nn.Linear` × 2 | `BayesianLinear` × 2 | P0 |

**实施要点**:
- 实现`BayesianLinear`和`BayesianConv1d`基础类
- 每个贝叶斯层需要实现`kl_divergence()`方法
- 参数量翻倍 (μ和ρ)
- 初始化策略: μ用Kaiming, ρ=-3.0

**预期影响**:
- ✅ 参数量: +100%
- ✅ 内存: +50-80%
- ⚠️ 训练速度: -20-30%

---

### GAP-2: 推理方式 (单次 → MC采样)

#### 当前状态
```python
# 训练
logits = model(x)  # 单次前向

# 推理
with torch.no_grad():
    logits = model(x)  # 单次前向
    pred = logits.argmax(dim=-1)
```

#### 目标状态
```python
# 训练 (单次采样)
logits = model(x)  # 内部采样一次

# 推理 (MC采样)
probs_list = []
for _ in range(M):  # M=30
    logits = model(x)  # 每次采样不同权重
    probs_list.append(F.softmax(logits, dim=-1))

pred_mean = torch.stack(probs_list).mean(dim=0)
pred = pred_mean.argmax(dim=-1)
```

#### 实施要点
- 贝叶斯层的`forward()`需要支持采样控制
- 训练时: `sample=True` (单次)
- 推理时: 循环调用M次
- 实现`mc_inference()`函数
- 可选: 实现自适应采样策略

**预期影响**:
- ⚠️ 推理时间: +2900% (M=30)
- ✅ 预测质量: 提升
- ✅ 新增不确定性量化

**优化方案**:
1. 批量MC推理 (并行化)
2. 减少M (10-15)
3. 自适应采样
4. MAP推理模式 (仅用μ)

---

### GAP-3: 损失函数 (CE → ELBO)

#### 当前状态
```python
# Label Smoothing交叉熵
criterion = LabelSmoothingCrossEntropy(
    epsilon=0.05,
    weight=class_weights
)

loss = criterion(logits, labels)
```

#### 目标状态
```python
# ELBO损失
elbo_loss = ELBOLoss(
    model=model,
    kl_weight=0.1,
    label_smoothing=0.05,
    class_weights=class_weights,
    num_batches=len(train_loader)
)

total_loss, rec_loss, kl_loss = elbo_loss(logits, labels)

# 其中:
# rec_loss = LabelSmoothingCrossEntropy(logits, labels)
# kl_loss = Σ layer.kl_divergence() / num_batches
# total_loss = rec_loss + kl_weight * kl_loss
```

#### 实施要点
- 实现`ELBOLoss`类
- 收集所有贝叶斯层的KL散度
- KL散度需要归一化 (除以batch数)
- 保留Label Smoothing和类别权重
- 新增超参数: `kl_weight`

**KL散度计算** (解析解):
```python
def kl_divergence(self):
    σ = F.softplus(self.rho)
    kl = 0.5 * torch.sum(
        (σ / prior_std)² + (μ / prior_std)² - 1 - 2*log(σ / prior_std)
    )
    return kl
```

**预期影响**:
- ✅ 增加正则化效果
- ✅ 提升泛化能力
- ⚠️ 需要调节`kl_weight`

---

### GAP-4: 输出内容 (预测 → 预测+不确定性)

#### 当前状态
```python
# 仅输出预测类别
pred_class = logits.argmax(dim=-1)
```

#### 目标状态
```python
# 输出预测类别 + 认知不确定性
pred_mean, pred_class, epistemic_uncertainty = mc_inference(model, x, M=30)

# 认知不确定性 (互信息)
# I(Y;W|X) = H(p̄) - E[H(p)]
```

#### 不确定性计算
```python
# 预测均值的熵
H_mean = -torch.sum(pred_mean * torch.log(pred_mean + 1e-10), dim=-1)

# 预测熵的均值
H_samples = -torch.sum(probs_stack * torch.log(probs_stack + 1e-10), dim=-1)
mean_H = H_samples.mean(dim=0)

# 认知不确定性 (互信息)
epistemic_uncertainty = H_mean - mean_H
```

#### 实施要点
- 修改评估函数返回不确定性
- 实现不确定性可视化工具
- 分析不确定性与预测错误的相关性
- 支持OOD检测

**预期影响**:
- ✅ 新增不确定性量化能力
- ✅ 支持OOD检测
- ✅ 提升模型可解释性

---

## 三、组件级修改清单

### 3.1 需要新增的组件

| 组件名称 | 功能 | 代码量估计 | 优先级 |
|---------|------|-----------|--------|
| `BayesianLinear` | 贝叶斯线性层 | ~100行 | P0 |
| `BayesianConv1d` | 贝叶斯1D卷积 | ~80行 | P0 |
| `BayesianGraphSAGELayer` | 贝叶斯GraphSAGE | ~60行 | P0 |
| `BayesianDepthwiseSeparableTCN` | 贝叶斯TCN | ~50行 | P0 |
| `BayesianEfficientSTBlock` | 贝叶斯ST块 | ~80行 | P0 |
| `BayesianGatedAttentionPooling` | 贝叶斯池化 | ~40行 | P1 |
| `ELBOLoss` | ELBO损失函数 | ~60行 | P0 |
| `mc_inference()` | MC推理函数 | ~50行 | P0 |
| `BSTGCNModel` | 完整模型 | ~150行 | P0 |

**总计**: ~670行核心代码

---

### 3.2 需要修改的组件

| 组件名称 | 修改内容 | 修改量 | 优先级 |
|---------|---------|--------|--------|
| `train_epoch()` | 使用ELBO损失 | 中 | P0 |
| `evaluate()` | 支持MC推理 | 大 | P0 |
| `02_train_hybrid_improved.py` | 训练脚本改造 | 大 | P0 |
| `save_results()` | 保存不确定性 | 小 | P1 |

---

## 四、超参数对比

### 4.1 保持不变的超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_dim` | 128 | 隐藏维度 |
| `num_heads` | 8 | 注意力头数 |
| `num_st_blocks` | 3 | ST块数量 |
| `k_neighbors` | 10 | KNN邻居数 |
| `sage_aggr` | 'mean' | GraphSAGE聚合方式 |
| `batch_size` | 128 | 批大小 |

---

### 4.2 需要调整的超参数

| 参数 | Hybrid值 | BSTGCN建议值 | 理由 |
|------|---------|-------------|------|
| `learning_rate` | 0.001 | 0.0005 | 贝叶斯模型需要更小LR |
| `weight_decay` | 1e-4 | 1e-5 | KL已提供正则化 |
| `dropout` | 0.2 | 0.15 | 贝叶斯层有正则化效果 |
| `num_epochs` | 100 | 150 | 可能需要更多训练 |
| `patience` | 20 | 25 | 收敛可能更慢 |
| `warmup_epochs` | 5 | 10 | 更长的warmup |

---

### 4.3 新增超参数

| 参数 | 推荐初始值 | 调优范围 | 说明 |
|------|-----------|---------|------|
| `kl_weight` | 0.1 | [0.01, 1.0] | KL散度权重 |
| `prior_std` | 1.0 | [0.1, 10.0] | 先验标准差 |
| `num_mc_samples` | 30 | [10, 100] | MC采样次数 |
| `init_rho` | -3.0 | [-5.0, -1.0] | ρ初始化值 |

---

## 五、性能指标对比

### 5.1 计算资源

| 指标 | Hybrid | BSTGCN | 变化率 |
|------|--------|--------|--------|
| 参数量 | 500K | 1M | +100% |
| 模型大小 | 2MB | 4MB | +100% |
| 训练内存 | 4GB | 6.4GB | +60% |
| 训练时间/epoch | 100s | 130s | +30% |
| 推理时间/sample | 0.5ms | 15ms | +2900% |
| 推理吞吐量 | 2000/s | 67/s | -97% |

---

### 5.2 预测性能

| 指标 | Hybrid | BSTGCN (预期) | 变化 |
|------|--------|--------------|------|
| Macro-F1 | 0.95 | 0.96-0.97 | +1-2% |
| Accuracy | 0.95 | 0.96-0.97 | +1-2% |
| 鲁棒性 | 基线 | 提升 | - |
| OOD检测 | 不支持 | 支持 | 新增 |
| 不确定性量化 | 无 | 有 | 新增 |

---

## 六、风险评估矩阵

| 风险 | 概率 | 影响 | 等级 | 缓解策略 |
|------|------|------|------|---------|
| 训练不稳定 | 中 | 高 | 🔴 | KL权重退火, 降低LR |
| 性能下降 | 低 | 高 | 🟡 | 调整kl_weight, 部分贝叶斯化 |
| 推理过慢 | 高 | 中 | 🟡 | 减少MC样本, 批量推理 |
| 内存溢出 | 低 | 中 | 🟢 | 减小batch size, 梯度检查点 |
| 超参难调 | 中 | 中 | 🟡 | 网格搜索, 参考文献 |

---

## 七、实施优先级

### P0 (必须实现)
- [ ] `BayesianLinear`和`BayesianConv1d`
- [ ] `BayesianGraphSAGELayer`和`BayesianDepthwiseSeparableTCN`
- [ ] `BayesianEfficientSTBlock`
- [ ] `ELBOLoss`
- [ ] `mc_inference()`
- [ ] 训练和评估流程改造

### P1 (重要但可延后)
- [ ] `BayesianGatedAttentionPooling`
- [ ] 不确定性可视化工具
- [ ] OOD检测评估
- [ ] 自适应MC采样

### P2 (可选优化)
- [ ] 推理加速优化
- [ ] 内存优化
- [ ] KL权重退火策略
- [ ] 对抗样本评估

---

## 八、成功标准

### 必须达成 (P0)
- ✅ 模型能够成功训练并收敛
- ✅ 提供不确定性量化输出
- ✅ Macro-F1 ≥ Hybrid模型 - 1%

### 期望达成 (P1)
- ✅ Macro-F1 > Hybrid模型
- ✅ 不确定性与错误相关性 > 0.5
- ✅ OOD样本不确定性显著高于ID

### 理想达成 (P2)
- ✅ Macro-F1 提升 > 2%
- ✅ 推理时间 < 10ms/sample
- ✅ 对抗样本表现优于确定性模型

---

## 九、时间估计

| 阶段 | 任务 | 估计时间 | 依赖 |
|------|------|---------|------|
| 第1周 | 基础设施 | 5天 | - |
| 第2周 | 模型改造 | 5天 | 第1周 |
| 第3周 | 训练流程 | 5天 | 第2周 |
| 第4周 | 实验验证 | 5天 | 第3周 |
| 第5周 | 优化文档 | 5天 | 第4周 |

**总计**: 25个工作日 (约5周)

---

## 十、总结

### 核心差异
1. **权重**: 点估计 → 分布
2. **推理**: 单次 → MC采样
3. **损失**: CE → ELBO
4. **输出**: 预测 → 预测+不确定性

### 主要收益
- ✅ 不确定性量化
- ✅ 更好的泛化
- ✅ OOD检测
- ✅ 对抗鲁棒性

### 主要成本
- ⚠️ 参数量翻倍
- ⚠️ 训练时间+30%
- ⚠️ 推理时间+30倍
- ⚠️ 超参调优复杂

### 推荐策略
**渐进式升级**: 先部分贝叶斯化 → 验证效果 → 全面升级

---

**文档版本**: v1.0  
**创建日期**: 2025-11-15

