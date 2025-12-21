# 实验文档总览

## 🎯 实验目标

对比50包和100包两种序列长度，验证：
1. 更少的包是否能达到更好的性能？
2. 训练和推理速度如何？
3. 与参考论文的对比

## 🎉 核心结论

**50包版本在所有维度上都显著优于100包版本！** ⭐

| 维度 | 100包 | 50包 | 提升 |
|------|-------|------|------|
| **Macro F1** | 0.9291 | **0.9543** | **+2.52%** ⭐ |
| **训练时间** | 2.5小时 | **34分钟** | **4.4×加速** ⭐ |
| **推理速度** | ~100/s | **24,017/s** | **240×加速** ⭐ |

## 📚 文档结构

### 1. 实验结果总结
**文件**: `EXPERIMENT_RESULTS_SUMMARY.md`

**内容**：
- 核心结论和对比表
- 详细性能分析
- 与参考论文对比
- 论文撰写要点

**关键数据**：
- Macro F1: 0.9543
- Accuracy: 0.9841
- 推理延迟: 0.0416ms
- 推理吞吐量: 24,017 samples/s

### 2. 50包 vs 100包详细对比
**文件**: `50_VS_100_PACKETS_COMPARISON.md`

**内容**：
- 实验设置对比
- 数据集规模对比
- 性能结果对比
- 关键问题分析（为什么训练慢？为什么推理快？）

**关键发现**：
- 数据量增加13倍（步长减半）
- 训练反而更快（4.4倍加速）
- 推理速度惊人（240倍加速）

### 3. 性能优化分析
**文件**: `PERFORMANCE_OPTIMIZATION.md`

**内容**：
- 数据量差异分析
- 优化策略详解
- 训练时间分析
- 推理速度分析

**优化策略**：
1. 增大Batch Size（128 vs 64）
2. 混合精度训练（AMP）
3. 多进程数据加载
4. 序列长度优势

### 4. 优化总结
**文件**: `OPTIMIZATION_SUMMARY.md`

**内容**：
- 实验目标
- 关键发现
- 优化策略
- 实际性能结果
- 常见问题解答

**关键问题**：
- Q: 为什么训练变慢了？A: 数据量增加13倍
- Q: 为什么推理更快？A: 序列短 + 优化策略

### 5. 各攻击类型性能分析
**文件**: `PER_CLASS_PERFORMANCE_ANALYSIS.md`

**内容**：
- 各攻击类型详细性能
- 性能分级
- 难点分析
- 优势分析

**性能分级**：
- 优秀（F1 ≥ 0.99）：UDPFlood, HTTPFlood, SYNFlood
- 良好（0.97 ≤ F1 < 0.99）：SlowrateDoS, ICMPFlood, Benign
- 中等（0.90 ≤ F1 < 0.97）：UDPScan
- 需改进（F1 < 0.90）：TCPConnectScan, SYNScan

## 📊 快速查看关键数据

### 总体性能

```
Accuracy:     0.9841
Macro F1:     0.9543
Weighted F1:  0.9841
```

### 训练效率

```
训练时间:     2060秒（34分钟）
训练样本:     150,154
最佳Epoch:    88
```

### 推理效率

```
延迟:         0.0416 ms/sample
吞吐量:       24,017 samples/sec
```

### 各攻击类型F1分数

```
UDPFlood:       0.9985 ⭐
HTTPFlood:      0.9980 ⭐
SYNFlood:       0.9955 ⭐
SlowrateDoS:    0.9944
ICMPFlood:      0.9723
Benign:         0.9667
UDPScan:        0.9147
TCPConnectScan: 0.8917
SYNScan:        0.8567 ⚠️
```

## 🎯 论文撰写要点

### 应该强调的优势

1. **包级细粒度建模**
   - 50个包节点 vs Burst图的5-20个节点
   - 保留更多细节

2. **端到端学习**
   - 无需手工定义规则
   - 图构建与分类联合优化

3. **极快推理速度**
   - 0.0416ms延迟
   - 满足实时检测需求

4. **性能优异**
   - Macro F1: 0.9543
   - 9类细粒度分类

5. **多流聚合检测**
   - 无需维护流状态表
   - 在多流攻击场景下更快

### 不应过度声称

1. ❌ "比参考论文更快"（单流场景下他们确实更快）
2. ❌ "使用更少的包"（Graph2vec也用50包）
3. ❌ "检测混合攻击类型"（每个序列仍是单一标签）

## 📁 实验结果文件

### 模型文件
- `baseline/results/hybrid_model_improved_50packets/best_model.pth`

### 评估报告
- `baseline/results/hybrid_model_improved_50packets/evaluation_report.txt`

### 预测结果
- `baseline/results/hybrid_model_improved_50packets/predictions.npy`

### 配置文件
- `baseline/results/hybrid_model_improved_50packets/config.json`

## 🔍 如何使用这些文档

1. **快速了解结果**：阅读本文档（README_EXPERIMENTS.md）
2. **详细性能分析**：阅读 EXPERIMENT_RESULTS_SUMMARY.md
3. **理解优化策略**：阅读 PERFORMANCE_OPTIMIZATION.md
4. **对比实验细节**：阅读 50_VS_100_PACKETS_COMPARISON.md
5. **各类别性能**：阅读 PER_CLASS_PERFORMANCE_ANALYSIS.md

## 📝 下一步工作

1. ✅ 完成实验和文档
2. ⬜ 可视化结果（混淆矩阵、ROC曲线）
3. ⬜ 撰写论文实验部分
4. ⬜ 准备演示材料

