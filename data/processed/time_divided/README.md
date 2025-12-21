# 基于时间划分的5G-NIDD数据集

## 概述

本目录包含基于时间顺序划分的5G-NIDD数据集，用于评估模型在真实场景下的泛化能力。

## 划分策略

### 核心思想

对每个攻击类型分别按时间排序后划分，确保：
1. **时间顺序**: 训练集样本时间 < 验证集样本时间 < 测试集样本时间
2. **类别完整**: 每个集合都包含全部9个攻击类型
3. **避免泄露**: 同一攻击事件的样本不会分散到不同集合

### 具体步骤

1. 解析StartTime字段（格式：MM:SS.s）为秒数
2. 对每个攻击类型，按时间排序
3. 对每个类型分别取：
   - 前70%样本 → 训练集
   - 中间15%样本 → 验证集
   - 后15%样本 → 测试集
4. 合并所有类型的划分结果

## 数据统计

### 总体分布

| 集合 | 样本数 | 占比 | 类别数 |
|------|--------|------|--------|
| 训练集 | 851,119 | 70.0% | 9 |
| 验证集 | 182,379 | 15.0% | 9 |
| 测试集 | 182,392 | 15.0% | 9 |
| **总计** | **1,215,890** | **100%** | **9** |

### 各类别时间划分详情

#### Benign（良性流量）
- 训练集: 334,415样本 (70.0%), 时间范围: 0.0-25.2分钟
- 验证集: 71,660样本 (15.0%), 时间范围: 25.2-47.2分钟
- 测试集: 71,662样本 (15.0%), 时间范围: 47.2-60.0分钟

#### UDPFlood
- 训练集: 320,138样本 (70.0%), 时间范围: 0.0-58.4分钟
- 验证集: 68,601样本 (15.0%), 时间范围: 58.4-58.6分钟
- 测试集: 68,601样本 (15.0%), 时间范围: 58.6-60.0分钟

#### HTTPFlood
- 训练集: 98,568样本 (70.0%), 时间范围: 28.6-43.1分钟
- 验证集: 21,121样本 (15.0%), 时间范围: 43.1-44.3分钟
- 测试集: 21,123样本 (15.0%), 时间范围: 44.3-45.9分钟

#### SlowrateDoS
- 训练集: 51,186样本 (70.0%), 时间范围: 0.0-12.7分钟
- 验证集: 10,968样本 (15.0%), 时间范围: 12.7-32.4分钟
- 测试集: 10,970样本 (15.0%), 时间范围: 32.4-60.0分钟

#### TCPConnectScan
- 训练集: 14,036样本 (70.0%), 时间范围: 0.1-57.2分钟
- 验证集: 3,007样本 (15.0%), 时间范围: 57.2-58.7分钟
- 测试集: 3,009样本 (15.0%), 时间范围: 58.7-59.9分钟

#### SYNScan
- 训练集: 14,030样本 (70.0%), 时间范围: 30.3-35.2分钟
- 验证集: 3,006样本 (15.0%), 时间范围: 35.2-36.1分钟
- 测试集: 3,007样本 (15.0%), 时间范围: 36.1-37.0分钟

#### UDPScan
- 训练集: 11,134样本 (70.0%), 时间范围: 28.8-35.3分钟
- 验证集: 2,385样本 (15.0%), 时间范围: 35.3-36.9分钟
- 测试集: 2,387样本 (15.0%), 时间范围: 36.9-38.1分钟

#### SYNFlood
- 训练集: 6,804样本 (70.0%), 时间范围: 44.5-53.7分钟
- 验证集: 1,458样本 (15.0%), 时间范围: 53.7-56.1分钟
- 测试集: 1,459样本 (15.0%), 时间范围: 56.1-58.6分钟

#### ICMPFlood
- 训练集: 808样本 (70.0%), 时间范围: 0.0-9.0分钟
- 验证集: 173样本 (15.0%), 时间范围: 9.0-11.9分钟
- 测试集: 174样本 (15.1%), 时间范围: 11.9-60.0分钟

### 类别权重

由于训练集中类别不平衡，计算的类别权重为：

```python
{
    'Benign':         0.283,
    'HTTPFlood':      0.959,
    'ICMPFlood':    117.041,  # ⚠️ 极少样本，需要特别关注
    'SYNFlood':      13.899,
    'SYNScan':        6.740,
    'SlowrateDoS':    1.848,
    'TCPConnectScan': 6.738,
    'UDPFlood':       0.295,
    'UDPScan':        8.494
}
```

## 文件说明

```
time_divided/
├── README.md                  # 本文件
├── split_indices.pkl          # 划分索引（train/val/test的样本索引）
├── train_features.npy         # 训练集特征 (851,119, 41)
├── train_labels.npy           # 训练集标签 (851,119,)
├── val_features.npy           # 验证集特征 (182,379, 41)
├── val_labels.npy             # 验证集标签 (182,379,)
├── test_features.npy          # 测试集特征 (182,392, 41)
├── test_labels.npy            # 测试集标签 (182,392,)
├── encoders.pkl               # 特征编码器（StandardScaler + LabelEncoders）
└── class_weights.npy          # 类别权重 (9,)
```

## 使用方法

### 加载数据

```python
import numpy as np
import pickle
from pathlib import Path

# 数据目录
data_dir = Path('data/processed/time_divided')

# 加载训练集
train_features = np.load(data_dir / 'train_features.npy')
train_labels = np.load(data_dir / 'train_labels.npy')

# 加载验证集
val_features = np.load(data_dir / 'val_features.npy')
val_labels = np.load(data_dir / 'val_labels.npy')

# 加载测试集
test_features = np.load(data_dir / 'test_features.npy')
test_labels = np.load(data_dir / 'test_labels.npy')

# 加载编码器
with open(data_dir / 'encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 加载类别权重
class_weights = np.load(data_dir / 'class_weights.npy')

print(f"训练集: {train_features.shape}, {train_labels.shape}")
print(f"验证集: {val_features.shape}, {val_labels.shape}")
print(f"测试集: {test_features.shape}, {test_labels.shape}")
```

### 标签映射

```python
# 获取标签编码器
label_encoder = encoders['label_encoder']

# 标签到类别名的映射
label_to_name = {i: name for i, name in enumerate(label_encoder.classes_)}
print("标签映射:", label_to_name)

# 输出:
# {
#     0: 'Benign',
#     1: 'HTTPFlood',
#     2: 'ICMPFlood',
#     3: 'SYNFlood',
#     4: 'SYNScan',
#     5: 'SlowrateDoS',
#     6: 'TCPConnectScan',
#     7: 'UDPFlood',
#     8: 'UDPScan'
# }
```

### 在训练中使用类别权重

```python
import torch
import torch.nn as nn

# 加载类别权重
class_weights = np.load(data_dir / 'class_weights.npy')
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 使用加权交叉熵损失
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

## 与随机划分的对比

| 维度 | 随机分层划分 | 基于时间划分（本数据） |
|------|-------------|---------------------|
| **数据位置** | `data/processed/v1_gnn_baseline/` | `data/processed/time_divided/` |
| **数据泄露风险** | 中等 | 低 |
| **真实场景符合度** | 中等 | 高 |
| **类别平衡性** | 优秀（完全一致） | 良好（都包含所有类别） |
| **模型性能** | 通常较高（99.994%） | 通常较低（更具挑战性） |
| **适用场景** | 算法对比、消融实验 | 实际部署评估、泛化能力测试 |

## 训练建议

### 1. 处理类别不平衡

由于ICMPFlood样本极少（训练集仅808个），建议：
- 使用类别权重（已提供）
- 使用Focal Loss
- 考虑过采样或数据增强

### 2. 监控验证集性能

由于时间划分的特性，验证集和测试集的分布可能与训练集有差异，建议：
- 密切监控验证集性能
- 使用早停（early stopping）防止过拟合
- 分析各类别的性能差异

### 3. 评估时间泛化能力

建议额外评估：
- 各时间段的性能变化
- 不同攻击类型的时间泛化能力
- 模型对新出现攻击模式的检测能力

## 生成脚本

本数据集由以下脚本生成：
```bash
python scripts/01_data_preparation_time_based.py
```

详细日志位于：`logs/data_preparation_time_based.log`

## 相关文档

- [数据划分对比文档](../../../docs/dataset/data_split_comparison.md)
- [总体方案](../../../docs/project_overview/总体方案.md)
- [时间分布分析](../../../docs/dataset/time_distribution_analysis.png)

## 注意事项

⚠️ **ICMPFlood类别样本极少**：训练集仅808个样本，可能导致该类别的检测性能不稳定。

⚠️ **时间集中的攻击**：某些攻击类型（如HTTPFlood、SYNScan、UDPScan）在时间上较为集中，可能影响模型的泛化能力。

⚠️ **类别权重差异大**：ICMPFlood的权重高达117.041，训练时需要注意权重的影响。

---

**生成时间**: 2025-11-06  
**生成脚本**: `scripts/01_data_preparation_time_based.py`  
**数据版本**: v1.0

