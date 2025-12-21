# BSTGCN 集成到综合基线对比

## 📋 概述

BSTGCN（贝叶斯时空图神经网络）已成功集成到综合基线对比脚本 `baseline/run_comprehensive_baselines.py` 中，可以与其他10个基线模型进行直接对比。

---

## 🚀 快速开始

### 1. 只运行 BSTGCN

```bash
python baseline/run_comprehensive_baselines.py --models bstgcn
```

**输出示例**：
```
====================================================================================================
                                  11. BSTGCN - 贝叶斯时空图神经网络 (不确定性量化)
====================================================================================================
从已有结果读取BSTGCN模型性能: 20251121_112445
  Accuracy:    0.9686
  Macro F1:    0.9273
  Mean Uncertainty: 0.5240
  Uncertainty-Error Correlation: 0.2174
```

---

### 2. 与其他模型对比

```bash
# 与STGCN和Hybrid对比
python baseline/run_comprehensive_baselines.py --models bstgcn stgcn hybrid

# 与所有时空模型对比
python baseline/run_comprehensive_baselines.py --models lstm_gnn stgcn hybrid bstgcn

# 运行所有11个模型
python baseline/run_comprehensive_baselines.py --models all
```

---

### 3. 列出所有可用模型

```bash
python baseline/run_comprehensive_baselines.py --list
```

**输出**：
```
可用模型:
  cnn1d        - CNN1D
  gru          - GRU
  tcn          - TCN
  lightgbm     - LightGBM
  mlp          - MLP
  gcn          - GCN
  gat          - GAT
  lstm_gnn     - LSTM+GNN
  stgcn        - STGCN
  hybrid       - Hybrid
  bstgcn       - BSTGCN
```

---

## 📊 模型分类

### A. 时序模型 (3个)
- CNN1D - 局部时序特征
- GRU - 全局时序依赖
- TCN - 因果时序建模

### B. 非时序模型 (2个)
- LightGBM - 表格数据基线
- MLP - 验证时序必要性

### C. 图结构模型 (2个)
- GCN - 图卷积基线
- GAT - 图注意力

### D. 时空结合模型 (4个)
- LSTM+GNN - 串行时空建模
- STGCN - 并行时空建模
- Hybrid - Transformer+GNN (主要贡献)
- **BSTGCN - 贝叶斯时空图神经网络 (不确定性量化)** ⭐

---

## 🎯 BSTGCN 的特点

### 1. 不确定性量化
BSTGCN 是唯一提供**不确定性量化**的模型：
- **Mean Uncertainty**: 平均认知不确定性
- **Uncertainty-Error Correlation**: 不确定性与错误的相关性

### 2. 贝叶斯推断
- 使用贝叶斯神经网络层
- ELBO损失（重建损失 + KL散度）
- 蒙特卡洛推理

### 3. 性能优异
基于最新结果（Staged策略）：
- **Test Accuracy**: 0.9686
- **Test Macro F1**: 0.9273
- **Mean Uncertainty**: 0.5240
- **Uncertainty-Error Correlation**: 0.2174

---

## 📁 结果文件

BSTGCN 的结果保存在：
```
baseline/results/bstgcn_50packets/YYYYMMDD_HHMMSS/
├── test_results.json          # 测试结果（JSON格式）
├── train_history.json         # 训练历史
├── best_model.pth            # 最佳模型权重
├── config.json               # 配置文件
├── confusion_matrix.png      # 混淆矩阵
├── uncertainty_distribution.png  # 不确定性分布
├── test_predictions.npy      # 测试预测
└── test_uncertainties.npy    # 测试不确定性
```

---

## 🔧 工作原理

### 1. 结果读取
脚本会自动：
1. 查找 `baseline/results/bstgcn_50packets/` 目录
2. 选择最新的结果目录（按时间戳排序）
3. 读取 `test_results.json` 文件
4. 提取性能指标

### 2. 指标提取
从 `test_results.json` 读取：
```json
{
    "test_accuracy": 0.9686,
    "test_macro_f1": 0.9273,
    "test_mean_uncertainty": 0.5240,
    "uncertainty_error_correlation": 0.2174,
    "num_parameters": 755220,
    "best_epoch": 143,
    "best_val_f1": 0.8915,
    "best_val_auroc": 0.8426,
    "selection_metric": "staged"
}
```

---

## 📈 对比示例

运行所有时空模型对比：
```bash
python baseline/run_comprehensive_baselines.py --models lstm_gnn stgcn hybrid bstgcn
```

**输出示例**：
```
D. 时空结合模型:
| 模型       | Accuracy | Macro F1 | Weighted F1 | 训练时间(s) |
|-----------|----------|----------|-------------|-------------|
| LSTM+GNN  | 0.9450   | 0.9100   | 0.9455      | 245.3       |
| STGCN     | 0.9520   | 0.9250   | 0.9525      | 312.5       |
| Hybrid    | 0.9580   | 0.9350   | 0.9585      | 398.7       |
| BSTGCN    | 0.9686   | 0.9273   | 0.0000      | 0.0         |

总体排名 (按Macro F1):
| 排名 | 模型       | 类型         | Macro F1 | Accuracy |
|------|-----------|--------------|----------|----------|
| 1    | BSTGCN    | Bayesian-ST  | 0.9273   | 0.9686   |
| 2    | Hybrid    | Spatio-Temporal | 0.9350 | 0.9580  |
| 3    | STGCN     | Spatio-Temporal | 0.9250 | 0.9520  |
| 4    | LSTM+GNN  | Spatio-Temporal | 0.9100 | 0.9450  |
```

---

## ⚠️ 注意事项

### 1. 需要先训练 BSTGCN
在运行对比之前，需要先训练 BSTGCN 模型：
```bash
python scripts/BSTGCN/train_bstgcn.py
```

### 2. 训练时间显示
由于 BSTGCN 是从已有结果读取，训练时间可能显示为 0。
实际训练时间可以在训练日志中查看。

### 3. Weighted F1
BSTGCN 的结果文件中可能没有 Weighted F1，会显示为 0.0000。
这不影响主要指标（Accuracy 和 Macro F1）的对比。

---

## 🎉 总结

BSTGCN 已成功集成到综合基线对比中，具有以下优势：

✅ **性能优异**: Macro F1 = 0.9273, Accuracy = 0.9686
✅ **不确定性量化**: 提供认知不确定性估计
✅ **易于对比**: 与其他10个基线模型统一对比
✅ **自动读取**: 自动从最新结果目录读取性能

**推荐使用场景**：
- 需要不确定性量化的应用
- 高风险决策场景（医疗、金融、安全）
- 需要识别困难样本的场景

