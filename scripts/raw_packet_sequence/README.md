# 原始包序列分类方案

## 📋 概述

本方案直接从PCAP文件提取**原始包级别的时序数据**，不进行时间窗口聚合，使用Transformer模型进行分类。

### 与聚合特征方案的对比

| 特性 | 聚合特征方案 | 原始包序列方案 |
|------|-------------|---------------|
| **数据类型** | 1秒窗口聚合统计特征 | 原始包级别特征 |
| **特征示例** | avg_packet_size, std_packet_size, packet_rate | 每个包的packet_length, ttl, tcp_flags |
| **时序信息** | 弱（聚合后丢失细节） | 强（保留完整时序） |
| **数据量** | 小（22,822条流量记录） | 大（数百万个包） |
| **最佳模型** | LightGBM (Macro F1: 95.17%) | Transformer (待测试) |
| **训练时间** | 快（9.54秒） | 慢（预计数小时） |
| **适用场景** | 实时检测、资源受限 | 离线分析、高精度需求 |

---

## 🎯 核心思想

### 问题：为什么聚合特征模型性能有上限？

当前的聚合特征方案：
```
原始PCAP包流 → 1秒窗口聚合 → 统计特征 (avg, std, count等)
                    ↓
            丢失了细粒度时序信息
                    ↓
        5秒窗口再次聚合 → 序列数据
```

**信息损失**：
- 包到达的精确时间
- 包之间的微观模式
- 短时突发行为
- 细粒度的协议交互

### 解决方案：使用原始包序列

```
原始PCAP包流 → 提取每个包的特征 → 固定长度序列 (100个包)
                    ↓
            保留完整时序信息
                    ↓
        Transformer建模 → 分类
```

**优势**：
- ✅ 保留细粒度时序信息
- ✅ 捕捉包之间的依赖关系
- ✅ 学习微观攻击模式
- ✅ 没有信息损失

---

## 📁 文件结构

```
scripts/raw_packet_sequence/
├── README.md                      # 本文档
├── 01_extract_raw_sequences.py   # 从PCAP提取原始包序列
├── 02_prepare_data.py             # 数据预处理（标准化、划分）
├── 03_transformer_model.py        # Transformer模型定义
├── 04_train_transformer.py        # 训练脚本
├── 05_generate_report.py          # 生成对比报告
└── run_all.py                     # 运行完整流程
```

---

## 🚀 使用方法

### 方法1: 运行完整流程（推荐）

```bash
python scripts/raw_packet_sequence/run_all.py
```

这会自动执行：
1. 提取原始包序列
2. 数据预处理
3. 训练Transformer模型
4. 生成对比报告

### 方法2: 分步执行

#### Step 1: 提取原始包序列

```bash
python scripts/raw_packet_sequence/01_extract_raw_sequences.py
```

**配置参数**：
- `SEQUENCE_LENGTH`: 每个序列包含的包数量（默认100）
- `STRIDE`: 滑动窗口步长（默认50）
- `MAX_PACKETS`: 每个文件最多读取的包数量（测试用，默认50000）

**输出**：
- `data/processed/raw_packet_sequences/sequences.npy`: (N, 100, 14)
- `data/processed/raw_packet_sequences/labels.npy`: (N,)
- `data/processed/raw_packet_sequences/metadata.pkl`

#### Step 2: 数据预处理

```bash
python scripts/raw_packet_sequence/02_prepare_data.py
```

**功能**：
- 划分训练/测试集（70/30）
- 标准化特征
- 保存处理后的数据

**输出**：
- `train_sequences.npy`, `test_sequences.npy`
- `train_labels.npy`, `test_labels.npy`
- `metadata.pkl` (包含scaler)

#### Step 3: 训练Transformer模型

```bash
python scripts/raw_packet_sequence/04_train_transformer.py
```

**配置参数**：
- `MODEL_TYPE`: 'basic', 'enhanced', 'lstm'
- `BATCH_SIZE`: 64
- `NUM_EPOCHS`: 100
- `LEARNING_RATE`: 0.0001
- `D_MODEL`: 128
- `NHEAD`: 8
- `NUM_LAYERS`: 4

**输出**：
- `baseline/results/transformer_raw/best_model.pth`
- `baseline/results/transformer_raw/evaluation_report.txt`
- `baseline/results/transformer_raw/confusion_matrix.npy`

#### Step 4: 生成对比报告

```bash
python scripts/raw_packet_sequence/05_generate_report.py
```

**输出**：
- `baseline/results/RAW_PACKET_COMPARISON.txt`

---

## 🏗️ 模型架构

### 1. PacketTransformer (基础版)

```
Input: (batch, seq_len=100, num_features=14)
  ↓ Feature Embedding (Linear + LayerNorm + ReLU)
(batch, seq_len, d_model=128)
  ↓ Positional Encoding
  ↓ Transformer Encoder (4 layers, 8 heads)
  ↓ Global Average Pooling
(batch, d_model)
  ↓ Classifier (2-layer MLP)
Output: (batch, num_classes=9)
```

**参数量**: ~200K

### 2. EnhancedPacketTransformer (增强版)

**改进**：
- 更深的特征嵌入（2层）
- 注意力池化（替代平均池化）
- 更深的分类器（3层）
- GELU激活函数

**参数量**: ~300K

### 3. LSTMAttentionModel (对比基线)

```
Input: (batch, seq_len, num_features)
  ↓ Bidirectional LSTM (2 layers)
(batch, seq_len, hidden_dim*2)
  ↓ Attention Mechanism
(batch, hidden_dim*2)
  ↓ Classifier
Output: (batch, num_classes)
```

**参数量**: ~150K

---

## 📊 特征说明

### 提取的14个包级别特征

| 特征 | 说明 | 取值范围 |
|------|------|---------|
| `timestamp` | 相对时间戳（秒） | [0, +∞) |
| `packet_length` | 包长度（字节） | [0, 65535] |
| `protocol` | 协议类型 | 0=OTHER, 1=TCP, 2=UDP, 3=ICMP |
| `src_port` | 源端口 | [0, 65535] |
| `dst_port` | 目标端口 | [0, 65535] |
| `tcp_flags_syn` | SYN标志 | 0/1 |
| `tcp_flags_ack` | ACK标志 | 0/1 |
| `tcp_flags_fin` | FIN标志 | 0/1 |
| `tcp_flags_rst` | RST标志 | 0/1 |
| `tcp_flags_psh` | PSH标志 | 0/1 |
| `tcp_window` | TCP窗口大小 | [0, 65535] |
| `ttl` | 生存时间 | [0, 255] |
| `payload_size` | 载荷大小（字节） | [0, 65535] |
| `ip_len` | IP包长度 | [0, 65535] |

---

## 🔍 预期结果

### 场景1: 原始包序列 > 聚合特征

如果Transformer模型的Macro F1 > 95.17% (LightGBM)：

**结论**: ✅ 原始包序列方案更好！

**原因**:
- 保留了细粒度时序信息
- Transformer捕捉到了包之间的复杂依赖
- 微观攻击模式被学习到

**推荐**: 使用原始包序列 + Transformer

### 场景2: 原始包序列 ≈ 聚合特征

如果Transformer模型的Macro F1 ≈ 95% (差距<5%)：

**结论**: ⚖️ 两种方案性能相当

**原因**:
- 聚合特征已经包含了足够信息
- 原始包序列的额外信息贡献有限

**推荐**: 使用聚合特征 + LightGBM（更简单、更快）

### 场景3: 原始包序列 < 聚合特征

如果Transformer模型的Macro F1 < 90%：

**结论**: ❌ 原始包序列方案不如聚合特征

**可能原因**:
- 模型容量不足
- 训练数据不足
- 超参数未优化
- 序列长度不合适
- 过拟合

**改进方向**:
1. 增加模型容量（更多层、更大hidden_dim）
2. 增加训练数据（读取更多包）
3. 调整序列长度（50, 200, 500）
4. 使用预训练模型
5. 数据增强

---

## ⚙️ 配置调优

### 序列长度选择

| 序列长度 | 优势 | 劣势 | 适用场景 |
|---------|------|------|---------|
| 50 | 训练快，内存小 | 信息少 | 快速测试 |
| 100 | 平衡 | - | 默认推荐 |
| 200 | 信息丰富 | 训练慢 | 高精度需求 |
| 500 | 完整攻击模式 | 内存大，训练很慢 | 离线分析 |

### 模型容量选择

| 配置 | d_model | num_layers | nhead | 参数量 | 训练时间 |
|------|---------|-----------|-------|--------|---------|
| 小 | 64 | 2 | 4 | ~50K | 快 |
| 中 | 128 | 4 | 8 | ~200K | 中 |
| 大 | 256 | 6 | 8 | ~800K | 慢 |
| 超大 | 512 | 8 | 16 | ~3M | 很慢 |

---

## 💡 关键洞察

### 1. 数据特性决定模型选择

- **聚合统计特征** → 树模型 (LightGBM/XGBoost)
- **原始时序数据** → 时序模型 (Transformer/LSTM)

### 2. 信息 vs 计算成本

- 原始包序列：信息丰富，但计算成本高
- 聚合特征：信息损失，但计算成本低

### 3. 实际应用考虑

**实时检测**：
- 使用聚合特征 + LightGBM
- 推理速度快 (<1ms)
- 资源占用小

**离线分析**：
- 使用原始包序列 + Transformer
- 追求最高精度
- 资源充足

---

## 📈 改进方向

### 短期（1周）

1. **测试不同序列长度**: 50, 100, 200, 500
2. **测试不同模型容量**: 小、中、大
3. **超参数调优**: learning rate, batch size, dropout

### 中期（1个月）

1. **多尺度建模**: 结合不同序列长度
2. **预训练**: 在大规模数据上预训练
3. **集成学习**: 多个Transformer模型投票

### 长期（3个月）

1. **流级别建模**: 按流分组，建模流内包序列
2. **图神经网络**: 包之间构建图，使用GNN
3. **自监督学习**: 利用未标注数据

---

## 🎉 总结

本方案提供了一个完整的原始包序列分类流程，从PCAP提取到模型训练再到结果对比。

**核心优势**：
- ✅ 保留完整时序信息
- ✅ 使用先进的Transformer架构
- ✅ 完整的实验流程
- ✅ 与基线模型对比

**适用场景**：
- 需要最高精度的离线分析
- 研究攻击的微观模式
- 探索深度学习在网络安全的潜力

**下一步**：
运行 `python scripts/raw_packet_sequence/run_all.py` 开始实验！

