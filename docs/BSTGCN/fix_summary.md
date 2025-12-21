# BSTGCN Val F1急速下降问题修复总结

## 问题现象

训练过程中出现以下问题：

1. **Val F1急速下降**：
   - Epoch 13: 0.8600 → Epoch 20: 0.1399
   - 下降幅度：84%

2. **频繁的梯度NaN警告**：
   - 几乎每个epoch都有`input_proj.weight_mu`梯度NaN
   - 跳过的异常batch数量：1-2个/epoch

3. **训练集过拟合**：
   - Train F1持续上升到0.88
   - Val F1却急速下降

4. **KL散度异常大**：
   - 持续在1000+的水平
   - 未能有效收敛

## 根本原因

### 原因1: KL权重过大 (主要原因)

```
实际KL惩罚 = kl_weight * kl_divergence / num_batches
           = 1e-3 * 1000 / 1173
           ≈ 0.85

重建损失 ≈ 0.1
```

**结论**：KL惩罚项是重建损失的8.5倍，导致模型被过度正则化，牺牲预测能力来降低KL散度。

### 原因2: 贝叶斯参数不稳定

- `weight_rho`初始化为-7，但训练中可能增长过快
- `weight_sigma`上限为10.0，允许过大的方差
- 采样的权重可能出现极端值

### 原因3: 参数约束不够严格

- `weight_rho`上限为2.0（对应sigma≈1.3，仍然较大）
- 没有对采样后的权重进行裁剪
- 参数裁剪在梯度更新后才执行，可能已经污染了参数

## 修复方案

### 修复1: 降低KL权重 ⭐⭐⭐

**文件**：`scripts/BSTGCN/train_bstgcn.py`

```python
'kl_weight_max': 2e-4  # 从1e-3降低到2e-4
```

**效果**：将KL惩罚降低到与重建损失相当的水平。

### 修复2: 更保守的初始化 ⭐⭐

**文件**：`scripts/BSTGCN/bayesian_layers.py`

```python
nn.init.constant_(self.weight_rho, -9.0)  # 从-7.0改为-9.0
# sigma从0.001降低到0.0001
```

**效果**：模型从接近确定性的状态开始，训练更稳定。

### 修复3: 严格限制sigma上限 ⭐⭐⭐

**文件**：`scripts/BSTGCN/bayesian_layers.py`

```python
weight_sigma = torch.clamp(weight_sigma, min=1e-8, max=2.0)  # 从10.0改为2.0
```

**效果**：防止方差过大导致的数值不稳定。

### 修复4: 裁剪采样后的权重 ⭐⭐

**文件**：`scripts/BSTGCN/bayesian_layers.py`

```python
weight = self.weight_mu + weight_sigma * weight_eps
weight = torch.clamp(weight, min=-15.0, max=15.0)  # 新增
```

**效果**：即使sigma被限制，极端随机采样仍可能产生极端值，需要额外保护。

### 修复5: 更严格的rho裁剪 ⭐⭐⭐

**文件**：`scripts/BSTGCN/train_bstgcn.py`

```python
param.clamp_(-10.0, -1.0)  # 从(-10.0, 2.0)改为(-10.0, -1.0)
# 对应sigma范围：[0.0001, 0.31]
```

**效果**：强制rho保持负值，确保sigma始终较小。

### 修复6: 添加健康检查 ⭐

**文件**：`scripts/BSTGCN/train_bstgcn.py`

```python
def check_model_health(model):
    """检查模型参数健康状态"""
    for name, param in model.named_parameters():
        if 'weight_rho' in name or 'bias_rho' in name:
            sigma = F.softplus(param)
            if (sigma > 5.0).any():
                return False, f"参数 {name} 对应的sigma过大"
    return True, "健康"
```

**效果**：及早发现参数异常，便于调试。

## 修复验证

### 稳定性测试结果

运行`scripts/BSTGCN/test_bayesian_stability.py`：

```
✓ BayesianLinear 测试通过
  - 初始KL散度: 64456 (合理范围)
  - 前向传播无NaN/Inf
  - 反向传播梯度正常
  - weight_sigma范围: [0.000123, 0.000123]

✓ BayesianConv1d 测试通过
  - 初始KL散度: 384728 (合理范围)
  - 前向传播无NaN/Inf
  - 反向传播梯度正常

✓ 参数裁剪测试通过
  - 极端值被正确裁剪
  - 输出保持有限
```

## 预期改进

1. **训练稳定性**：
   - 梯度NaN警告减少或消失
   - 跳过的异常batch数量减少

2. **验证性能**：
   - Val F1不再急速下降
   - Val F1曲线更平滑

3. **KL散度**：
   - KL散度逐渐收敛
   - KL惩罚项与重建损失平衡

4. **泛化能力**：
   - 训练和验证F1差距缩小
   - 测试集性能提升

## 下一步

1. **重新训练**：
   ```bash
   conda activate gnn_py312
   python scripts/BSTGCN/train_bstgcn.py
   ```

2. **监控指标**：
   - 梯度NaN警告频率
   - Val F1稳定性
   - KL散度收敛情况
   - Train/Val F1差距

3. **如果问题仍存在**：
   - 进一步降低KL权重（1e-4）
   - 增加KL warmup轮数（从60到80）
   - 降低学习率（从5e-4到3e-4）
   - 增加梯度裁剪强度（从0.5到0.3）

## 修复前后对比

| 参数 | 修复前 | 修复后 | 影响 |
|------|--------|--------|------|
| `kl_weight_max` | 1e-3 | 2e-4 | KL惩罚降低5倍 |
| `weight_rho`初始值 | -7.0 | -9.0 | 初始sigma降低10倍 |
| `weight_sigma`上限 | 10.0 | 2.0 | 最大方差降低5倍 |
| `weight_rho`上限 | 2.0 | -1.0 | 强制sigma<0.31 |
| 采样权重裁剪 | 无 | [-15, 15] | 防止极端值 |
| 健康检查 | 无 | 每100 batch | 及早发现异常 |

**数值稳定性改进**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 初始sigma | ~0.001 | ~0.0001 |
| 最大sigma | 10.0 | 0.31 |
| KL惩罚/重建损失比 | ~8.5 | ~1.7 |
| 梯度NaN频率 | 几乎每epoch | 预期大幅减少 |

## 参考文档

- `docs/BSTGCN/val_f1_drop_fix.md` - 详细修复说明
- `scripts/BSTGCN/test_bayesian_stability.py` - 稳定性测试脚本

