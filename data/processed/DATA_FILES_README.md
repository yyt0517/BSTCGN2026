# 数据文件说明

## 重要提示

由于GitHub对单个文件大小的限制（100MB），本仓库**不包含**以下大型数据文件：

- 所有 `.npy` 文件（NumPy数组）
- 部分 `.pkl` 文件（Python pickle文件）
- `.bin` 文件（二进制图数据）

这些文件已通过 `.gitignore` 排除，不会被推送到远程仓库。

## 如何获取数据文件

### 方法1：从原始数据集重新生成

1. 下载5G-NIDD原始数据集
2. 运行相应的数据预处理脚本：
   - `scripts/raw_packet_sequence/01_extract_raw_sequences.py` - 生成原始包序列数据
   - `scripts/raw_packet_sequence/02_prepare_data_with_val.py` - 准备训练/验证/测试集
   - 其他预处理脚本（根据需要）

### 方法2：联系项目维护者

如果你需要已处理好的数据文件，请联系项目维护者获取。

## 数据目录结构

```
data/processed/
├── raw_packet_sequences/      # 原始包序列数据（用于Transformer和混合模型）
│   ├── sequences.npy          # 完整序列数据 (156.24 MB) - 未包含
│   ├── train_sequences.npy    # 训练集序列 (109.36 MB) - 未包含
│   ├── val_sequences.npy      # 验证集序列 - 未包含
│   ├── test_sequences.npy     # 测试集序列 - 未包含
│   ├── *_labels.npy           # 标签文件 - 未包含
│   └── metadata.pkl           # 元数据（已包含）
│
├── time_divided/              # 基于时间划分的数据
│   ├── train_features.npy     # 训练特征 (133.12 MB) - 未包含
│   ├── val_features.npy       # 验证特征 - 未包含
│   ├── test_features.npy      # 测试特征 - 未包含
│   ├── *_labels.npy           # 标签文件 - 未包含
│   └── encoders.pkl           # 编码器（已包含）
│
├── v1_gnn_baseline/           # GNN基线数据
│   ├── train_features.npy     # 训练特征 (133.12 MB) - 未包含
│   ├── dgl_graph.bin          # DGL图数据 - 未包含
│   └── graph_metadata.pkl     # 图元数据（已包含）
│
├── packet_based/              # 基于包的特征
├── packet_based_window5/      # 窗口大小为5的包特征
├── stgnn/                     # 时空图神经网络数据
└── kg_stgnn/                  # 知识图谱增强的STGNN数据
```

## 已包含的文件

以下文件**已包含**在仓库中：

- `metadata.pkl` - 数据集元数据
- `encoders.pkl` - 特征编码器
- `graph_metadata.pkl` - 图结构元数据
- `split_indices.pkl` - 数据划分索引
- `README.md` - 各目录说明文档
- `.txt` 文件 - 数据集信息

## 数据文件大小参考

| 文件 | 大小 | 状态 |
|------|------|------|
| `raw_packet_sequences/sequences.npy` | 156.24 MB | 未包含 |
| `time_divided/train_features.npy` | 133.12 MB | 未包含 |
| `v1_gnn_baseline/train_features.npy` | 133.12 MB | 未包含 |
| `raw_packet_sequences/train_sequences.npy` | 109.36 MB | 未包含 |

## 注意事项

1. **不要尝试将大文件添加到Git**：这些文件已在 `.gitignore` 中排除
2. **本地保留数据文件**：在本地开发时，这些文件应该存在于相应目录中
3. **数据文件是可重现的**：通过运行预处理脚本可以重新生成所有数据文件

## 相关文档

- [数据集构建指南](../../docs/DATASET_CONSTRUCTION_GUIDE.md)
- [原始包序列实验总结](../../docs/RAW_PACKET_EXPERIMENT_SUMMARY.md)
- [时间序列构建说明](../../docs/SEQUENCE_CONSTRUCTION_EXPLAINED.md)

