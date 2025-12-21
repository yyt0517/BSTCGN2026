# 文档索引

本目录包含项目的所有文档，按功能分类。

---

## 📁 目录结构

```
docs/
├── README.md                          # 本文件（文档索引）
├── timeseries_construction/           # 时序数据构建文档
├── analysis/                          # 数据分析文档
├── baseline/                          # 基线模型文档
├── gnn/                               # GNN模型文档
└── dataset/                           # 数据集文档
```

---

## 📖 文档分类

### 1. 时序数据构建 (`timeseries_construction/`)

**核心文档**：
- **`BENIGN_RATIO_ANALYSIS.md`** ⭐ - 良性流量占比分析
  - 为什么良性流量从39.2%降到9.59%？
  - 流数量 vs 实际流量
  - 时序数据的准确性

**相关主题**：
- PCAP文件处理
- 时间窗口聚合
- 特征提取
- 数据质量验证

---

### 2. 数据分析 (`analysis/`)

**核心文档**：
- **`5G_NIDD_TIME_SERIES_Analysis_Report.md`** - 时序数据分析报告
  - 数据集统计
  - 特征分析
  - 类别分布
  
- **`TEMPORAL_DISTRIBUTION_ANALYSIS.md`** - 时间分布分析
  - 攻击类型的时间模式
  - 时间窗口分析
  
- **`README.md`** - 分析文档索引

**相关主题**：
- 数据可视化
- 统计分析
- 特征重要性
- 相关性分析

---

### 3. 基线模型 (`baseline/`)

**核心文档**：
- **`BASELINE_MODELS_COMPLETED.md`** ⭐ - 基线模型完成报告
  - 8种基线模型的实现
  - 性能对比
  - 最佳模型推荐
  
- **`WHY_TREE_MODELS_WORK_BETTER.md`** - 为什么树模型表现更好
  - 树模型 vs 神经网络
  - 特征重要性分析
  - 模型选择建议

**相关主题**：
- Random Forest
- XGBoost
- LightGBM
- 模型评估
- 超参数调优

---

### 4. GNN模型 (`gnn/`)

**核心文档**：
- **`GNN_IMPROVEMENTS.md`** ⭐ - GNN模型改进
  - 模型架构
  - 性能优化
  - 实验结果

**相关主题**：
- Graph Attention Networks (GAT)
- GraphSAGE
- GATv2
- 时序图神经网络
- 模型训练技巧

---

### 5. 数据集 (`dataset/`)

**核心文档**：
- **`5G-NIDD/`** - 5G-NIDD数据集文档
  - 数据集介绍
  - 文件结构
  - 使用说明

**相关主题**：
- PCAP文件
- 流聚合数据
- 攻击类型
- 数据预处理

---

## 🎯 快速导航

### 我想了解...

#### 数据集相关

- **数据集基本信息** → [README.md](../README.md#数据集)
- **为什么良性流量占比这么低？** → [timeseries_construction/BENIGN_RATIO_ANALYSIS.md](timeseries_construction/BENIGN_RATIO_ANALYSIS.md)
- **数据集统计分析** → [analysis/5G_NIDD_TIME_SERIES_Analysis_Report.md](analysis/5G_NIDD_TIME_SERIES_Analysis_Report.md)
- **时间分布分析** → [analysis/TEMPORAL_DISTRIBUTION_ANALYSIS.md](analysis/TEMPORAL_DISTRIBUTION_ANALYSIS.md)

#### 模型相关

- **基线模型性能** → [baseline/BASELINE_MODELS_COMPLETED.md](baseline/BASELINE_MODELS_COMPLETED.md)
- **为什么树模型表现好？** → [baseline/WHY_TREE_MODELS_WORK_BETTER.md](baseline/WHY_TREE_MODELS_WORK_BETTER.md)
- **GNN模型改进** → [gnn/GNN_IMPROVEMENTS.md](gnn/GNN_IMPROVEMENTS.md)

#### 实现相关

- **如何构建时序数据？** → [timeseries_construction/](timeseries_construction/)
- **如何训练基线模型？** → [../baseline/README.md](../baseline/README.md)
- **如何训练GNN模型？** → [../scripts/gnn/](../scripts/gnn/)

---

## 📊 文档统计

| 类别 | 文档数 | 主要内容 |
|------|--------|---------|
| 时序数据构建 | 1+ | PCAP处理、特征提取、数据验证 |
| 数据分析 | 3+ | 统计分析、可视化、特征分析 |
| 基线模型 | 2+ | 模型实现、性能对比、优化 |
| GNN模型 | 1+ | 模型架构、训练技巧、改进 |
| 数据集 | 1+ | 数据集介绍、使用说明 |

---

## 🔍 搜索建议

### 按关键词搜索

- **良性流量** → `BENIGN_RATIO_ANALYSIS.md`
- **ICMPFlood** → `BENIGN_RATIO_ANALYSIS.md`, `5G_NIDD_TIME_SERIES_Analysis_Report.md`
- **基线模型** → `baseline/`
- **GNN** → `gnn/`
- **时序数据** → `timeseries_construction/`, `analysis/`
- **性能对比** → `BASELINE_MODELS_COMPLETED.md`, `GNN_IMPROVEMENTS.md`

### 按问题搜索

- **数据集有问题吗？** → `BENIGN_RATIO_ANALYSIS.md`
- **哪个模型最好？** → `BASELINE_MODELS_COMPLETED.md`
- **如何提高性能？** → `WHY_TREE_MODELS_WORK_BETTER.md`, `GNN_IMPROVEMENTS.md`
- **如何处理类别不平衡？** → `BENIGN_RATIO_ANALYSIS.md`, `BASELINE_MODELS_COMPLETED.md`

---

## 📝 文档规范

### 文档命名

- 使用大写字母和下划线：`DOCUMENT_NAME.md`
- 使用描述性名称：`BENIGN_RATIO_ANALYSIS.md` 而不是 `analysis.md`

### 文档结构

1. **标题**：清晰的文档标题
2. **概述**：简短的文档概述
3. **目录**：长文档需要目录
4. **正文**：分节组织内容
5. **总结**：关键结论和下一步
6. **参考**：相关文档链接

### 文档更新

- 每次更新添加更新时间
- 重大更新添加版本号
- 保留历史版本在 `archive/`

---

## 🚀 贡献文档

### 添加新文档

1. 确定文档类别（时序数据构建、分析、基线模型、GNN）
2. 在相应目录创建文档
3. 更新本索引文件
4. 在主README中添加链接（如果是核心文档）

### 更新现有文档

1. 在文档末尾添加更新时间
2. 如果是重大更新，更新版本号
3. 更新本索引文件（如果文档名称或位置改变）

---

## 📧 联系

如果您有任何问题或建议，请：
1. 查看相关文档
2. 查看 [FAQ](../README.md#常见问题)
3. 提交Issue

---

*最后更新: 2025-11-10*
*文档总数: 7+*

