# Val F1急速下降问题修复

## 问题描述

在训练BSTGCN模型时，验证集F1分数在epoch 17后急速下降：

- Epoch 13: Val F1 = 0.8600 (最高)
- Epoch 17: Val F1 = 0.6683
- Epoch 18: Val F1 = 0.3917
- Epoch 19: Val F1 = 0.2028
- Epoch 20: Val F1 = 0.1399

同时伴随以下现象：
1. 频繁出现梯度NaN警告（`input_proj.weight_mu`）
2. KL散度持续很大（~1000+）
3. 训练集F1持续上升（过拟合）

## 根本原因分析

### 1. KL权重过大导致过度正则化

**问题**：
- 配置：`kl_weight_max = 1e-3`
- KL散度：~1000
- 实际KL惩罚：`1e-3 * 1000 / num_batches ≈ 0.85`
- 重建损失：~0.1

**结果**：KL惩罚项远大于重建损失，模型被过度正则化，导致预测能力崩溃。

### 2. 贝叶斯参数初始化和约束不够保守

**问题**：
- `weight_rho`初始化为-7（对应sigma ≈ 0.001）
- `weight_sigma`上限为10.0（过大）
- 训练过程中rho可能增长过快

**结果**：采样的权重方差过大，导致数值不稳定和梯度爆炸。

### 3. 参数裁剪不够严格

**问题**：
- `weight_rho`裁剪上限为2.0（对应sigma ≈ 1.3，仍然较大）
- 采样后的权重没有裁剪

**结果**：极端权重值导致前向传播和反向传播中的数值溢出。

## 修复方案

### 修复1: 降低KL权重

**文件**：`scripts/BSTGCN/train_bstgcn.py`

```python
# 修改前
'Exp2-KL-Mid-L2-Low': {
    'kl_weight_max': 1e-3,
    'weight_decay': 5e-5,
}

# 修改后
'Exp2-KL-Mid-L2-Low': {
    'kl_weight_max': 2e-4,  # 降低到基线水平
    'weight_decay': 5e-5,
}
```

**理由**：将KL权重降低到基线水平（2e-4），避免过度正则化。

### 修复2: 更保守的贝叶斯参数初始化

**文件**：`scripts/BSTGCN/bayesian_layers.py`

```python
# 修改前
nn.init.constant_(self.weight_rho, -7.0)  # sigma ≈ 0.001
nn.init.constant_(self.bias_rho, -7.0)

# 修改后
nn.init.constant_(self.weight_rho, -9.0)  # sigma ≈ 0.0001
nn.init.constant_(self.bias_rho, -9.0)
```

**理由**：更小的初始方差，使模型从接近确定性的状态开始训练，更稳定。

### 修复3: 严格限制sigma上限

**文件**：`scripts/BSTGCN/bayesian_layers.py`

```python
# 修改前
weight_sigma = torch.clamp(weight_sigma, min=1e-8, max=10.0)

# 修改后
weight_sigma = torch.clamp(weight_sigma, min=1e-8, max=2.0)
```

**理由**：限制方差上限，防止采样的权重过于极端。

### 修复4: 裁剪采样后的权重

**文件**：`scripts/BSTGCN/bayesian_layers.py`

```python
# 新增
weight = self.weight_mu + weight_sigma * weight_eps
weight = torch.clamp(weight, min=-15.0, max=15.0)  # 裁剪采样后的权重

bias = self.bias_mu + bias_sigma * bias_eps
bias = torch.clamp(bias, min=-15.0, max=15.0)
```

**理由**：即使sigma被限制，极端的随机采样仍可能产生极端值，需要额外裁剪。

### 修复5: 更严格的rho参数裁剪

**文件**：`scripts/BSTGCN/train_bstgcn.py`

```python
# 修改前
elif 'weight_rho' in name or 'bias_rho' in name:
    param.clamp_(-10.0, 2.0)  # rho不应该太大

# 修改后
elif 'weight_rho' in name or 'bias_rho' in name:
    # rho限制在[-10, -1]，对应sigma在[0.0001, 0.31]
    param.clamp_(-10.0, -1.0)
```

**理由**：强制rho保持在负值范围，确保sigma始终较小。

### 修复6: 添加模型健康检查

**文件**：`scripts/BSTGCN/train_bstgcn.py`

```python
def check_model_health(model):
    """检查模型参数健康状态"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.isfinite(param).all():
                return False, f"参数 {name} 包含NaN/Inf"
            if 'weight_rho' in name or 'bias_rho' in name:
                sigma = torch.nn.functional.softplus(param)
                if (sigma > 5.0).any():
                    return False, f"参数 {name} 对应的sigma过大"
    return True, "健康"
```

**理由**：定期检查参数状态，及早发现异常。

## 预期效果

1. **训练稳定性提升**：减少或消除梯度NaN警告
2. **验证性能稳定**：Val F1不再急速下降
3. **KL散度可控**：KL散度保持在合理范围
4. **泛化能力改善**：减少过拟合，提高测试集性能

## 验证方法

重新运行训练：

```bash
conda activate gnn_py312
python scripts/BSTGCN/train_bstgcn.py
```

观察指标：
1. 梯度NaN警告是否减少
2. Val F1是否稳定增长
3. KL散度是否在合理范围
4. 训练和验证F1的差距是否缩小

