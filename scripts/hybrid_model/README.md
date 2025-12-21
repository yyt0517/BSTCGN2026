# Hybrid Model (GraphSAGE + TCN)

## 📋 概述

本目录包含改进版混合模型的实现，使用 **GraphSAGE + TCN** 进行高效的时空图神经网络建模，用于网络流量分类任务。

### 🎯 核心特性

- ✅ **GraphSAGE 图卷积** - 替代 GATv2，计算效率提升 2-3倍
- ✅ **并行时空交互** - ST-Block 同时处理时间和空间特征
- ✅ **轻量 TCN** - 深度可分离卷积进行时间建模
- ✅ **门控融合** - 自适应融合时间和空间特征
- ✅ **高效推理** - 推理速度提升 3-4倍

---

## 🏗️ 模型架构

### 整体流程

```
输入 (batch, 50, 25)
    ↓
输入投影 (batch, 50, 128)
    ↓
Transformer (1层) - 全局上下文
    ↓
KNN 图构建 (batch, 50, 50)
    ↓
ST-Block 1 ──┬── TCN (时间) ──┬── 门控融合
             └── GraphSAGE (空间) ─┘
    ↓
ST-Block 2 (同上)
    ↓
ST-Block 3 (同上)
    ↓
门控注意力池化 (batch, 128)
    ↓
分类器 (batch, 9)
```

### 核心组件

1. **GraphSAGE 图卷积** - 邻居采样聚合，降低计算复杂度
2. **深度可分离 TCN** - 轻量时间建模
3. **ST-Block** - 并行时空交互 + 门控融合
4. **门控注意力池化** - 自适应序列聚合

---

## 📊 性能对比

### vs GATv2

| 指标 | GATv2 | GraphSAGE | 提升 |
|------|-------|-----------|------|
| **参数量** | 463,499 | 421,130 | **9.1% ↓** |
| **训练时间** | 2060秒 | 1558秒 | **1.32x ↑** |
| **推理吞吐量** | 24,018 samples/s | 41,174 samples/s | **1.71x ↑** |
| **推理延迟** | 0.0416 ms | 0.0243 ms | **41.6% ↓** |
| **测试集 F1** | 95.43% | 95.32% | **-0.11%** |

**结论**：GraphSAGE 在保持性能的同时大幅提升效率！

---

## 📁 文件说明

### 核心文件

- **`hybrid_model_improved.py`** - GraphSAGE + TCN 混合模型
- **`02_train_hybrid_improved.py`** - 训练脚本

### 辅助文件

- `03_compare_results.py` - 结果对比
- `check_data_leakage.py` - 数据泄漏检查
- `explain_data_structure.py` - 数据结构说明
- `visualize_graph_structure.py` - 图结构可视化

### 备份文件

- `hybrid_model_improved_gatv2_backup.py` - GATv2 版本备份
- `02_train_hybrid_improved_gatv2_backup.py` - GATv2 训练脚本备份

### 文档

- `README.md` - 本文档
- `QUICKSTART.md` - 快速开始指南
- `ALGORITHM_COMPARISON.md` - 算法详细对比
- `IMPROVEMENTS.md` - 改进说明
- `PERFORMANCE_ANALYSIS.md` - 性能分析

---

## 🚀 快速开始

### 训练模型

```bash
# 激活环境
conda activate gnn_py312

# 训练模型
python scripts/hybrid_model/02_train_hybrid_improved.py
```

**预期输出**：
- 训练时间：~26 分钟
- 最佳验证 F1：~0.95
- 测试集 F1：~0.95
- 推理吞吐量：~41,000 samples/sec

### 查看结果

结果保存在：`baseline/results/hybrid_model_improved_50packets/`

---

## 🔧 配置参数

### 模型参数

```python
HIDDEN_DIM = 128        # 隐藏层维度
NUM_HEADS = 8           # Transformer 注意力头数
NUM_ST_BLOCKS = 3       # ST-Block 层数
DROPOUT = 0.2           # Dropout 比例
K_NEIGHBORS = 10        # KNN 图的邻居数
SAGE_AGGR = 'mean'      # GraphSAGE 聚合方式
```

### 训练参数

```python
BATCH_SIZE = 128        # 批次大小
NUM_EPOCHS = 100        # 最大训练轮数
LEARNING_RATE = 0.001   # 学习率
PATIENCE = 20           # 早停耐心值
LABEL_SMOOTHING = 0.05  # 标签平滑系数
USE_AMP = True          # 混合精度训练
```

---

## 📈 实验结果

### 在 5G-NIDD 数据集上（50包序列）

- **测试集 Accuracy**: 98.40%
- **测试集 Macro F1**: 95.32%
- **训练时间**: 1558 秒（26 分钟）
- **推理吞吐量**: 41,174 samples/sec

---

## 📚 参考文献

1. **GraphSAGE**: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
2. **TCN**: Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" (2018)

---

## 📞 常见问题

**Q: 为什么使用 GraphSAGE 而不是 GATv2？**
A: GraphSAGE 计算效率更高（训练快32%，推理快71%），性能几乎相同（仅下降0.11%）。

**Q: 可以进一步加速吗？**
A: 可以减少层数、隐藏维度或邻居数。

**Q: 如何选择聚合方式？**
A: `mean` 适合大多数场景，`max` 适合稀疏特征。

---

更多详细信息，请参阅：
- `QUICKSTART.md` - 快速开始指南
- `ALGORITHM_COMPARISON.md` - 算法详细对比
- `GRAPHSAGE_README.md` - GraphSAGE 详细文档

