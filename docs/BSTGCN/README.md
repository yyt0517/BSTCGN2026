# BSTGCN升级计划 - 文档索引

本目录包含从当前Hybrid模型升级到贝叶斯时空图神经网络(BSTGCN)的完整技术文档。

---

## 📚 文档列表

### 1. 核心文档

#### 📄 [贝叶斯图神经网络.pdf](./贝叶斯图神经网络.pdf)
- **内容**: BSTGCN的理论框架和数学推导
- **关键章节**:
  - 问题定义与符号表示
  - 贝叶斯时空图神经网络架构
  - 变分推断训练流程 (ELBO优化)
  - 蒙特卡洛推理与不确定性量化
- **适合**: 理解理论基础

#### 📋 [BSTGCN_Upgrade_Plan.md](./BSTGCN_Upgrade_Plan.md) ⭐ **核心文档**
- **内容**: 完整的GAP分析和实施计划
- **关键章节**:
  - 当前Hybrid模型架构回顾
  - BSTGCN核心要求分析
  - 4个关键GAP识别
  - 5个阶段的实施计划
  - 超参数调优建议
  - 性能影响评估
  - 风险与缓解策略
  - 成功标准定义
- **适合**: 项目规划和管理

#### 🔄 [architecture_comparison.md](./architecture_comparison.md)
- **内容**: Hybrid模型与BSTGCN的详细对比
- **关键章节**:
  - 整体架构对比图
  - 核心组件对比 (Linear, GraphSAGE, TCN)
  - 训练流程对比
  - 推理流程对比
  - 性能指标对比表
  - 应用场景分析
  - 混合部署策略
- **适合**: 理解架构差异

#### 💻 [implementation_examples.md](./implementation_examples.md)
- **内容**: 详细的代码实现示例
- **关键章节**:
  - BayesianLinear完整实现
  - BayesianConv1d实现
  - BayesianGraphSAGELayer实现
  - BayesianDepthwiseSeparableTCN实现
  - BayesianEfficientSTBlock实现
  - ELBOLoss实现
  - MC推理函数
  - 完整训练脚本框架
- **适合**: 代码开发参考

---

## 🎯 快速导航

### 如果你想...

**了解理论基础** → 阅读 [贝叶斯图神经网络.pdf](./贝叶斯图神经网络.pdf)

**制定项目计划** → 阅读 [BSTGCN_Upgrade_Plan.md](./BSTGCN_Upgrade_Plan.md)

**理解架构差异** → 阅读 [architecture_comparison.md](./architecture_comparison.md)

**开始编码实现** → 阅读 [implementation_examples.md](./implementation_examples.md)

---

## 📊 核心GAP总结

根据详细分析，从Hybrid模型升级到BSTGCN需要解决以下4个核心GAP：

### GAP 1: 确定性权重 → 贝叶斯权重分布 🔴
**当前**: 所有权重为点估计 `W`  
**目标**: 权重为分布 `q(W; μ, ρ)`  
**影响**: 参数量翻倍，需要实现贝叶斯层

### GAP 2: 单次前向传播 → 蒙特卡洛采样 🔴
**当前**: `logits = model(x)` (单次)  
**目标**: 训练时单次采样，推理时M次采样  
**影响**: 推理时间增加M倍

### GAP 3: 交叉熵损失 → ELBO损失 🔴
**当前**: `loss = CE(logits, labels)`  
**目标**: `loss = L_rec + λ * L_KL`  
**影响**: 需要收集所有贝叶斯层的KL散度

### GAP 4: 点预测 → 不确定性量化 🟡
**当前**: 仅输出预测类别  
**目标**: 输出预测 + 认知不确定性  
**影响**: 新增能力，支持OOD检测

---

## 🚀 实施路线图

### 第1周: 基础设施
- [ ] 实现`BayesianLinear`和`BayesianConv1d`
- [ ] 单元测试贝叶斯层
- [ ] 实现`ELBOLoss`

### 第2周: 模型改造
- [ ] 实现`BayesianGraphSAGELayer`
- [ ] 实现`BayesianDepthwiseSeparableTCN`
- [ ] 实现`BayesianEfficientSTBlock`
- [ ] 构建完整`BSTGCNModel`

### 第3周: 训练流程
- [ ] 修改训练脚本
- [ ] 实现MC推理函数
- [ ] 修改评估函数
- [ ] 添加不确定性可视化

### 第4周: 实验验证
- [ ] 基线实验 (与Hybrid模型对比)
- [ ] 超参数调优
- [ ] 消融实验
- [ ] OOD检测实验

### 第5周: 优化与文档
- [ ] 性能优化
- [ ] 内存优化
- [ ] 完善文档
- [ ] 代码审查

---

## ⚖️ 性能影响预估

| 指标 | Hybrid模型 | BSTGCN | 变化 |
|------|-----------|--------|------|
| 参数量 | ~500K | ~1M | +100% |
| 训练时间/epoch | 100s | 130s | +30% |
| 推理时间/sample | 0.5ms | 15ms | +2900% |
| Macro-F1 | 0.95 | 0.96-0.97 | +1-2% |
| 不确定性量化 | ❌ | ✅ | 新增 |
| OOD检测 | ❌ | ✅ | 新增 |

---

## 🎓 关键技术点

### 1. 重参数化技巧
```python
# 使得采样过程可微
σ = softplus(ρ)
W = μ + σ ⊙ ε, where ε ~ N(0, I)
```

### 2. ELBO损失
```python
L_ELBO = -E_q[log p(Y|X,W)] + KL[q(W)||p(W)]
       = L_rec + λ * L_KL
```

### 3. 认知不确定性
```python
# 互信息: 预测均值的熵 - 预测熵的均值
I(Y;W|X) = H(p̄) - E[H(p)]
```

### 4. KL散度 (解析解)
```python
KL[N(μ,σ²)||N(0,1)] = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
```

---

## 📈 预期收益

### 准确性提升
- ✅ Macro-F1: +1-2%
- ✅ 泛化能力: 显著提升
- ✅ 鲁棒性: 提升

### 新增能力
- ✅ 不确定性量化
- ✅ OOD检测
- ✅ 对抗样本识别
- ✅ 置信度评估

### 应用价值
- ✅ 高风险决策支持
- ✅ 未知攻击识别
- ✅ 模型可解释性
- ✅ 小样本学习

---

## ⚠️ 主要挑战

### 计算成本
- 参数量翻倍
- 训练时间增加30%
- 推理时间增加30倍 (可优化)

### 超参数调优
- 新增`kl_weight`, `prior_std`, `num_mc_samples`等
- 需要更多实验验证

### 工程实现
- 需要实现多个贝叶斯层
- 需要修改训练和推理流程
- 需要添加不确定性评估工具

---

## 🔧 推荐工具和库

### PyTorch贝叶斯库
- **Blitz**: https://github.com/piEsposito/blitz-bayesian-deep-learning
- **BayesianTorch**: https://github.com/IntelLabs/bayesian-torch
- **Pyro**: https://pyro.ai/

### 参考论文
- Weight Uncertainty in Neural Networks (Bayes by Backprop)
- Dropout as a Bayesian Approximation
- Uncertainty in Deep Learning (Yarin Gal's thesis)

---

## 💡 优化建议

### 推理加速
1. 批量MC采样
2. 减少MC样本数 (M=10-15)
3. 自适应采样策略
4. MAP推理模式 (无不确定性)

### 内存优化
1. 梯度检查点
2. 部分贝叶斯化 (仅关键层)
3. 混合精度训练

### 训练加速
1. 训练时单次采样
2. KL权重退火
3. 更大的batch size

---

## 📞 联系与支持

如有问题或建议，请参考以下资源：
- 详细实施计划: [BSTGCN_Upgrade_Plan.md](./BSTGCN_Upgrade_Plan.md)
- 代码示例: [implementation_examples.md](./implementation_examples.md)
- 架构对比: [architecture_comparison.md](./architecture_comparison.md)

---

**文档版本**: v1.0  
**创建日期**: 2025-11-15  
**状态**: 待审核

