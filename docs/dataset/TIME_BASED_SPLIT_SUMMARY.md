# 基于时间的数据集划分 - 完成总结

## 任务完成情况 ✅

已成功实现基于时间的数据集划分策略，并完成以下工作：

### 1. 数据准备脚本 ✅
- **文件**: `scripts/01_data_preparation_time_based.py`
- **功能**: 对每个攻击类型分别按时间排序后划分70%-15%-15%
- **特点**: 确保每个集合都包含全部9个类别，避免数据泄露

### 2. 数据集生成 ✅
- **位置**: `data/processed/time_divided/`
- **样本数**: 
  - 训练集: 851,119 (70.0%)
  - 验证集: 182,379 (15.0%)
  - 测试集: 182,392 (15.0%)
- **类别完整性**: 所有集合均包含全部9个类别

### 3. 文档更新 ✅
- **总体方案**: `docs/project_overview/总体方案.md` - 已更新数据划分策略章节
- **对比文档**: `docs/dataset/data_split_comparison.md` - 详细对比两种划分方式
- **数据说明**: `data/processed/time_divided/README.md` - 使用指南和注意事项

### 4. 分析工具 ✅
- **时间分布分析**: `scripts/analyze_time_distribution.py` - 分析数据集时间特征
- **划分对比**: `scripts/compare_data_splits.py` - 对比两种划分方式的统计信息

---

## 核心改进

### 问题：原随机划分的局限性

1. **数据泄露风险**: 同一攻击事件的样本可能分散到训练集和测试集
2. **过于乐观**: 测试集性能可能高估模型在真实场景的表现
3. **时间无关**: 没有考虑时间顺序，不符合实际部署场景

### 解决方案：基于时间的划分策略

#### 核心思想
对每个攻击类型分别按时间排序后划分，确保：
- ✅ 训练集样本时间 < 验证集样本时间 < 测试集样本时间
- ✅ 每个集合都包含全部9个攻击类型
- ✅ 同一攻击事件的样本不会分散到不同集合

#### 实现步骤
1. 解析StartTime字段（格式：MM:SS.s）为秒数
2. 对每个攻击类型，按时间排序
3. 对每个类型分别取前70%、中间15%、后15%
4. 合并所有类型的划分结果

---

## 数据统计对比

### 样本数对比

| 集合 | 随机划分 | 时间划分 | 差异 |
|------|---------|---------|------|
| 训练集 | 851,122 | 851,119 | 3 |
| 验证集 | 182,384 | 182,379 | 5 |
| 测试集 | 182,384 | 182,392 | -8 |

**结论**: 两种划分方式的样本数几乎完全一致，差异可忽略不计。

### 训练集类别分布对比

| 类别 | 随机划分 | 时间划分 | 差异 |
|------|---------|---------|------|
| Benign | 334,415 | 334,415 | 0 |
| UDPFlood | 320,138 | 320,138 | 0 |
| HTTPFlood | 98,568 | 98,568 | 0 |
| SlowrateDoS | 51,186 | 51,186 | 0 |
| TCPConnectScan | 14,036 | 14,036 | 0 |
| SYNScan | 14,031 | 14,030 | 1 |
| UDPScan | 11,134 | 11,134 | 0 |
| SYNFlood | 6,805 | 6,804 | 1 |
| ICMPFlood | 809 | 808 | 1 |

**结论**: 训练集类别分布几乎完全相同，最大差异仅1个样本。

### 类别权重对比

| 类别 | 随机划分 | 时间划分 |
|------|---------|---------|
| Benign | 0.283 | 0.283 |
| HTTPFlood | 0.959 | 0.959 |
| ICMPFlood | 116.896 | 117.041 |
| SYNFlood | 13.897 | 13.899 |
| SYNScan | 6.740 | 6.740 |
| SlowrateDoS | 1.848 | 1.848 |
| TCPConnectScan | 6.738 | 6.738 |
| UDPFlood | 0.295 | 0.295 |
| UDPScan | 8.494 | 8.494 |

**结论**: 类别权重几乎完全一致，可以使用相同的训练策略。

---

## 时间划分详情

### 各类别时间范围

| 攻击类型 | 训练集时间范围 | 验证集时间范围 | 测试集时间范围 |
|---------|--------------|--------------|--------------|
| Benign | 0.0-25.2分钟 | 25.2-47.2分钟 | 47.2-60.0分钟 |
| UDPFlood | 0.0-58.4分钟 | 58.4-58.6分钟 | 58.6-60.0分钟 |
| HTTPFlood | 28.6-43.1分钟 | 43.1-44.3分钟 | 44.3-45.9分钟 |
| SlowrateDoS | 0.0-12.7分钟 | 12.7-32.4分钟 | 32.4-60.0分钟 |
| TCPConnectScan | 0.1-57.2分钟 | 57.2-58.7分钟 | 58.7-59.9分钟 |
| SYNScan | 30.3-35.2分钟 | 35.2-36.1分钟 | 36.1-37.0分钟 |
| UDPScan | 28.8-35.3分钟 | 35.3-36.9分钟 | 36.9-38.1分钟 |
| SYNFlood | 44.5-53.7分钟 | 53.7-56.1分钟 | 56.1-58.6分钟 |
| ICMPFlood | 0.0-9.0分钟 | 9.0-11.9分钟 | 11.9-60.0分钟 |

### 关键观察

1. **时间分布广泛**: Benign、UDPFlood、SlowrateDoS、ICMPFlood、TCPConnectScan跨越整个60分钟
2. **时间集中攻击**: HTTPFlood、SYNScan、UDPScan、SYNFlood在特定时间段集中出现
3. **严格时间顺序**: 每个类别的训练集时间 < 验证集时间 < 测试集时间

---

## 使用建议

### 何时使用时间划分？

✅ **推荐场景**:
- 评估模型在真实场景的表现
- 测试模型对未见时间段的泛化能力
- 论文发表（提供更可信的性能评估）
- 生产环境模拟

⚠️ **注意事项**:
- 性能可能低于随机划分（更具挑战性）
- ICMPFlood样本极少（训练集仅808个），需要特别关注
- 某些攻击类型时间集中，可能影响泛化能力

### 训练建议

1. **处理类别不平衡**:
   - 使用提供的类别权重
   - 考虑Focal Loss
   - 监控少样本类别（ICMPFlood）的性能

2. **监控验证集性能**:
   - 密切监控验证集性能变化
   - 使用早停防止过拟合
   - 分析各类别的性能差异

3. **评估时间泛化能力**:
   - 评估各时间段的性能变化
   - 分析不同攻击类型的时间泛化能力
   - 测试对新攻击模式的检测能力

---

## 文件清单

### 生成的数据文件
```
data/processed/time_divided/
├── README.md                  # 使用指南
├── split_indices.pkl          # 划分索引
├── train_features.npy         # 训练集特征 (851,119, 41)
├── train_labels.npy           # 训练集标签 (851,119,)
├── val_features.npy           # 验证集特征 (182,379, 41)
├── val_labels.npy             # 验证集标签 (182,379,)
├── test_features.npy          # 测试集特征 (182,392, 41)
├── test_labels.npy            # 测试集标签 (182,392,)
├── encoders.pkl               # 特征编码器
└── class_weights.npy          # 类别权重 (9,)
```

### 脚本文件
```
scripts/
├── 01_data_preparation_time_based.py  # 数据准备脚本
├── analyze_time_distribution.py       # 时间分布分析
└── compare_data_splits.py             # 划分对比
```

### 文档文件
```
docs/
├── project_overview/
│   └── 总体方案.md                    # 已更新数据划分策略
├── dataset/
│   ├── data_split_comparison.md       # 详细对比文档
│   ├── TIME_BASED_SPLIT_SUMMARY.md    # 本文件
│   ├── time_distribution_analysis.png # 时间分布可视化
│   └── time_distribution_stats.csv    # 时间统计数据
└── data/processed/time_divided/
    └── README.md                       # 数据使用指南
```

### 日志文件
```
logs/
└── data_preparation_time_based.log    # 数据准备日志
```

---

## 下一步工作

### 立即可做
- [ ] 使用时间划分数据训练GATv2模型
- [ ] 对比两种划分方式的模型性能
- [ ] 分析性能差异的原因

### 后续研究
- [ ] 评估模型的时间泛化能力
- [ ] 研究攻击类型的时间演化模式
- [ ] 探索时间感知的模型架构
- [ ] 撰写实验报告和论文

---

## 快速开始

### 加载数据
```python
import numpy as np
from pathlib import Path

data_dir = Path('data/processed/time_divided')

# 加载训练集
train_features = np.load(data_dir / 'train_features.npy')
train_labels = np.load(data_dir / 'train_labels.npy')

print(f"训练集: {train_features.shape}, {train_labels.shape}")
```

### 运行对比分析
```bash
# 对比两种划分方式
python scripts/compare_data_splits.py

# 分析时间分布
python scripts/analyze_time_distribution.py
```

---

**完成时间**: 2025-11-06  
**数据版本**: v1.0  
**状态**: ✅ 已完成并验证

