# BSTGCN Val F1下降问题 - 快速修复参考

## 🔴 问题症状

- Val F1从0.86急速下降到0.14（84%下降）
- 频繁出现梯度NaN警告
- Train F1持续上升但Val F1下降（过拟合）
- KL散度持续很大（1000+）

## 🎯 核心原因

**KL权重过大导致过度正则化**

```
实际KL惩罚 = 1e-3 × 1000 / 1173 ≈ 0.85
重建损失 ≈ 0.1
比例 = 8.5:1  ❌ 太大了！
```

## ✅ 修复清单

### 1. 降低KL权重 (最重要)
```python
# scripts/BSTGCN/train_bstgcn.py
'kl_weight_max': 2e-4  # 从1e-3改为2e-4
```

### 2. 更保守的初始化
```python
# scripts/BSTGCN/bayesian_layers.py
nn.init.constant_(self.weight_rho, -9.0)  # 从-7.0改为-9.0
nn.init.constant_(self.bias_rho, -9.0)
```

### 3. 限制sigma上限
```python
# scripts/BSTGCN/bayesian_layers.py (BayesianLinear和BayesianConv1d)
weight_sigma = torch.clamp(weight_sigma, min=1e-8, max=2.0)  # 从10.0改为2.0
```

### 4. 裁剪采样权重
```python
# scripts/BSTGCN/bayesian_layers.py (forward方法)
weight = self.weight_mu + weight_sigma * weight_eps
weight = torch.clamp(weight, min=-15.0, max=15.0)  # 新增
```

### 5. 严格限制rho
```python
# scripts/BSTGCN/train_bstgcn.py (train_epoch_bayesian)
param.clamp_(-10.0, -1.0)  # 从(-10.0, 2.0)改为(-10.0, -1.0)
```

### 6. 添加健康检查
```python
# scripts/BSTGCN/train_bstgcn.py
def check_model_health(model):
    for name, param in model.named_parameters():
        if 'weight_rho' in name or 'bias_rho' in name:
            sigma = F.softplus(param)
            if (sigma > 5.0).any():
                return False, f"参数 {name} sigma过大"
    return True, "健康"
```

## 📊 修复效果对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| KL权重 | 1e-3 | 2e-4 |
| 初始sigma | 0.001 | 0.0001 |
| 最大sigma | 10.0 | 0.31 |
| KL/重建比 | 8.5:1 | 1.7:1 |

## 🧪 验证步骤

1. **运行稳定性测试**：
   ```bash
   python scripts/BSTGCN/test_bayesian_stability.py
   ```
   预期：所有测试通过，无NaN/Inf

2. **重新训练**：
   ```bash
   python scripts/BSTGCN/train_bstgcn.py
   ```

3. **监控指标**：
   - ✅ 梯度NaN警告减少
   - ✅ Val F1稳定增长
   - ✅ KL散度逐渐收敛
   - ✅ Train/Val F1差距缩小

## 🔧 如果问题仍存在

### 方案A: 进一步降低KL权重
```python
'kl_weight_max': 1e-4  # 再降低一半
```

### 方案B: 增加KL warmup
```python
'kl_warmup_epochs': 80  # 从60增加到80
```

### 方案C: 降低学习率
```python
'learning_rate': 3e-4  # 从5e-4降低到3e-4
```

### 方案D: 增强梯度裁剪
```python
'grad_clip': 0.3  # 从0.5降低到0.3
```

## 📚 详细文档

- `docs/BSTGCN/fix_summary.md` - 完整修复总结
- `docs/BSTGCN/val_f1_drop_fix.md` - 详细问题分析
- `scripts/BSTGCN/test_bayesian_stability.py` - 稳定性测试

## 💡 关键理解

**贝叶斯神经网络的平衡**：
```
总损失 = 重建损失 + λ × KL散度
         ↑           ↑
      预测能力    正则化

λ太大 → 过度正则化 → 预测能力崩溃 ❌
λ太小 → 欠正则化 → 过拟合 ❌
λ适中 → 平衡 → 良好泛化 ✅
```

**sigma的控制**：
```
rho → softplus → sigma → 采样权重
-9.0 → 0.0001 → 小方差 → 稳定 ✅
-1.0 → 0.31 → 中方差 → 可接受 ✅
2.0 → 1.3 → 大方差 → 不稳定 ❌
10.0 → 10.0 → 极大方差 → 崩溃 ❌
```

