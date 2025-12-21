# Baseline Models

本目录包含所有基线模型的训练脚本和结果。

---

## 🏆 最新排名（包含改进版 Hybrid）

| 排名 | 模型 | 类型 | Macro F1 | Accuracy | 训练时间 | 状态 |
|------|------|------|----------|----------|----------|------|
| 🥇 **1** | **Hybrid (改进版)** | 时空图 | **0.9544** | **0.9500** | **6.6分钟** | ✅ **NEW** |
| 🥈 **2** | **STGCN (优化版)** | 时空图 | **0.9341** | **0.9276** | **45.1分钟** | ✅ |
| 🥉 3 | Hybrid (原始) | 时空图 | 0.9184 | 0.9089 | ~5分钟 | ✅ |
| 4 | CNN1D | 时序 | 0.8977 | 0.8881 | 27.7秒 | ✅ |
| 5 | LightGBM | 非时序 | 0.8965 | 0.8865 | 67.3秒 | ✅ |
| 6 | LSTM+GNN | 时空图 | 0.8941 | 0.8849 | 56.4秒 | ✅ |
| 7 | GAT | 图 | 0.8840 | 0.8747 | 41.8秒 | ✅ |
| 8 | GCN | 图 | 0.8810 | 0.8706 | 28.4秒 | ✅ |
| 9 | GRU | 时序 | 0.8792 | 0.8678 | 32.4秒 | ✅ |
| 10 | TCN | 时序 | 0.8386 | 0.8222 | 23.4秒 | ✅ |
| 11 | MLP | 非时序 | 0.5411 | 0.5024 | 26.2秒 | ✅ |

**🏆 最佳模型**: Hybrid (改进版) - Macro F1: 0.9544
- 相比 STGCN：+2.03% Macro F1
- 训练时间：6.6分钟（STGCN：45.1分钟）
- 加速比：6.8倍

**详见**：`../PROJECT_RESULTS_SUMMARY.md`

---

## 🚀 快速开始

### 1. 准备数据

```bash
# 确保已经运行数据准备脚本
python scripts/raw_packet_sequence/02_prepare_data_with_val.py
```

### 2. 运行基线模型

#### 方式1：运行综合基线（10个模型）

```bash
# 运行所有10个模型（推荐）
python baseline/run_comprehensive_baselines.py
```

包含模型：
- **时序模型**: CNN1D, GRU, TCN
- **非时序模型**: MLP, LightGBM
- **图模型**: GCN, GAT
- **时空图模型**: LSTM+GNN, STGCN, Hybrid

#### 方式2：运行传统基线（7个模型）

```bash
# 运行所有模型
python baseline/run_baselines_fixed.py

# 运行单个模型
python baseline/run_baselines_fixed.py --models lstm

# 运行多个模型
python baseline/run_baselines_fixed.py --models lstm gru cnn1d

# 列出所有可用模型
python baseline/run_baselines_fixed.py --list
```

可用模型：
- `lstm` - LSTM
- `gru` - GRU
- `bilstm` - BiLSTM
- `cnn1d` - CNN1D
- `rf` - Random Forest
- `xgboost` - XGBoost
- `lightgbm` - LightGBM

### 3. 查看结果

#### 综合基线结果
```
baseline/results_v2/
├── cnn1d/
├── gru/
├── tcn/
├── mlp/
├── lightgbm/
├── gcn/
├── gat/
├── lstm_gnn/
├── stgcn/
└── hybrid/
```

#### 传统基线结果
```
baseline/results/
├── lstm_raw_fixed/
├── gru_raw_fixed/
├── bilstm_raw_fixed/
├── cnn1d_raw_fixed/
├── rf_raw_fixed/
├── xgboost_raw_fixed/
└── lightgbm_raw_fixed/
```

每个目录包含：
- `best_model.pth` - 最佳模型权重
- `config.json` - 训练配置
- `evaluation_report.txt` - 详细评估报告
- `confusion_matrix.png` - 混淆矩阵

---

## 📁 文件说明

### 核心脚本

- **`run_comprehensive_baselines.py`** - 综合基线脚本（推荐）
  - 包含10个模型（时序、图、时空图）
  - 统一的训练和评估流程
  - 固定随机种子（42）
  - 70:15:15数据划分

- **`run_baselines_fixed.py`** - 传统基线脚本（支持单独运行）
  - 包含7个传统模型
  - 支持命令行参数选择模型
  - 修复了测试集泄露问题

- **`stgcn_optimized.py`** - 优化版STGCN模型
  - 前向传播加速216倍
  - 内存占用减少98%
  - 训练时间减少80-90%

- **`test_optimized_models.py`** - 测试优化模型
  - 简化Hybrid vs 原始Hybrid
  - 优化STGCN vs 原始STGCN

- **`visualize_results.py`** - 可视化结果
  - 生成对比图表
  - 性能分析

- **`utils.py`** - 工具函数
  - 数据加载
  - 评估函数
  - 报告生成

### 核心文档

- **`LESSONS_LEARNED.md`** - 经验教训（必读）
  - STGCN vs Hybrid 深度分析
  - 优化经验总结
  - 实践建议

- **`COMPREHENSIVE_RESULTS_SUMMARY.md`** - 综合结果总结
  - 10个模型完整结果
  - 关键发现
  - 性能对比

---

## 🔧 配置说明

所有模型使用相同的配置：

```python
# 数据划分
train_size = 0.70  # 70%
val_size = 0.15    # 15%
test_size = 0.15   # 15%

# 随机种子
RANDOM_SEED = 42

# 训练参数
batch_size = 64
learning_rate = 0.001
max_epochs = 100
early_stopping_patience = 15

# 早停和学习率调度
monitor = 'val_f1'  # 基于验证集
```

---

## 📊 模型详情

### 时序模型

**LSTM**
- 2层LSTM，隐藏维度128
- Dropout 0.3
- 训练时间: ~49s

**GRU**
- 2层GRU，隐藏维度128
- Dropout 0.3
- 训练时间: ~35s

**BiLSTM**
- 2层双向LSTM，隐藏维度128
- Dropout 0.3
- 训练时间: ~83s

**CNN1D** ⭐
- 3层1D卷积，通道数[64, 128, 256]
- 全局平均池化
- 训练时间: ~32s
- **最佳基线模型**

### 树模型

**Random Forest**
- 100棵树
- 最大深度20
- 训练时间: ~1.8s

**XGBoost**
- 100棵树
- 学习率0.1
- 训练时间: ~89s

**LightGBM**
- 100棵树
- 学习率0.1
- 训练时间: ~76s

---

## 🎯 关键发现

### 1. 时空图模型表现最佳

**STGCN (F1: 0.9399)** 🥇
- 并行时空建模
- 简单有效
- 训练时间长（43.4分钟）

**Hybrid (F1: 0.9184)** 🥈
- 串行时空建模
- 架构复杂
- 训练时间短（~5分钟）

**关键洞察**：
- ✅ 并行 > 串行：STGCN的并行时空建模比Hybrid的串行建模好2.15%
- ✅ 简单 > 复杂：STGCN参数少（200K）但性能更好
- ✅ 卷积 > 注意力：对于网络流量，卷积的局部性更适合

### 2. 时序建模的重要性

- MLP (无时序): F1 = 0.5411
- GRU (时序): F1 = 0.8792
- **提升**: +34.8%

**结论**：时序信息对网络入侵检测至关重要

### 3. 图结构的贡献

- GRU (无图): F1 = 0.8792
- LSTM+GNN (有图): F1 = 0.8941
- **提升**: +1.5%

**结论**：图结构能捕获包之间的关系，但贡献有限

### 4. CNN1D是最佳传统基线

- 性能高（F1: 0.8977）
- 训练快（25.35秒）
- 适合实时检测

### 5. 优化的重要性

**优化STGCN**：
- 前向传播加速：216倍
- 内存占用减少：98%
- 训练时间减少：80-90%

**简化Hybrid**：
- 参数量减少：9.4%
- 预期性能提升：0.5-1.5%

---

## 📝 注意事项

### 测试集泄露修复

所有模型已修复测试集泄露问题：
- ✅ 早停监控: `val_f1`（验证集）
- ✅ 学习率调度: `val_loss`（验证集）
- ✅ 模型选择: 基于验证集F1
- ✅ 测试集: 只在最后评估一次

### 可复现性

所有模型固定随机种子：
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
```

### 数据要求

确保数据已准备好：
```
data/processed/raw_packet_sequences/
├── train_sequences.npy  # (11467, 100, 25)
├── train_labels.npy
├── val_sequences.npy    # (2458, 100, 25)
├── val_labels.npy
├── test_sequences.npy   # (2458, 100, 25)
├── test_labels.npy
└── metadata.pkl
```

---

## 🔗 相关文档

### 核心文档
- **[LESSONS_LEARNED.md](LESSONS_LEARNED.md)** - 经验教训（必读）
  - STGCN vs Hybrid 深度分析
  - 为什么简单模型更好
  - 优化经验总结
  - 实践建议

- **[COMPREHENSIVE_RESULTS_SUMMARY.md](COMPREHENSIVE_RESULTS_SUMMARY.md)** - 综合结果总结
  - 10个模型完整结果
  - 性能对比分析
  - 关键发现

### 项目文档
- [主README](../README.md) - 项目总览
- [Hybrid模型](../scripts/hybrid_model/) - Hybrid模型实现

---

## 📊 快速对比

| 维度 | STGCN | Hybrid | CNN1D |
|------|-------|--------|-------|
| **F1** | 0.9399 🥇 | 0.9184 🥈 | 0.8977 🥉 |
| **训练时间** | 43.4分钟 | ~5分钟 | 25.35秒 |
| **参数量** | ~200K | ~500K | ~150K |
| **架构** | 并行时空 | 串行时空 | 时序 |
| **适用场景** | 高精度 | 平衡 | 实时 |

---

*最后更新: 2025-11-12*

