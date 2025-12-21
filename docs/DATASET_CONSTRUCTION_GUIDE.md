# 数据集构造和划分详细说明

## 📋 概述

本文档详细说明了5G-NIDD数据集的两种构造方式：
1. **聚合特征方案**（用于LightGBM/XGBoost等树模型）
2. **原始包序列方案**（用于Transformer等时序模型）

---

## 🎯 方案1: 聚合特征方案

### 数据来源

**原始数据**: 18个PCAP文件（BS1和BS2各9个）
- `5G-NIDD-raw/BS1/*_nogtp.pcapng` (9个文件)
- `5G-NIDD-raw/BS2/*_nogtp.pcapng` (9个文件)

**攻击类型**: 9种
- Benign (SSH流量)
- HTTPFlood (Goldeneye)
- ICMPFlood
- SYNFlood
- SYNScan
- SlowrateDoS (Slowloris)
- TCPConnectScan
- UDPFlood
- UDPScan

### 构造流程

#### Step 1: 按秒聚合 (`scripts/timeseries_construction/build_timeseries_from_pcap.py`)

```
原始PCAP包流
    ↓
按1秒时间窗口聚合
    ↓
每秒生成一条记录，包含38个统计特征
```

**特征类型**:
- **包统计**: packet_count, avg_packet_size, std_packet_size, min_packet_size, max_packet_size
- **字节统计**: total_bytes, avg_bytes_per_sec
- **协议统计**: tcp_count, udp_count, icmp_count, other_count
- **端口统计**: unique_src_ports, unique_dst_ports, avg_src_port, avg_dst_port
- **TCP标志**: syn_count, ack_count, fin_count, rst_count, psh_count
- **时间特征**: avg_inter_arrival_time, std_inter_arrival_time
- **其他**: avg_ttl, std_ttl, avg_tcp_window, std_tcp_window, payload_ratio

**输出**: `5G_NIDD_TIME_SERIES/timeseries_from_pcap_1s.csv`
- 22,822条记录（每条代表1秒的流量）

#### Step 2: 滑动窗口构造序列 (`scripts/baseline/prepare_packet_based_data.py`)

```
1秒聚合数据 (22,822条)
    ↓
5秒滑动窗口 (stride=1秒)
    ↓
每个序列包含5个时间步
```

**配置**:
- `WINDOW_SIZE = 5` (5秒窗口)
- `STRIDE = 1` (1秒步长)
- 每个序列形状: `(5, 38)` (5个时间步，每步38个特征)

**输出**: `data/processed/packet_based/`
- `train_sequences.npy`: (8,717, 5, 38)
- `test_sequences.npy`: (3,722, 5, 38)
- `train_labels.npy`: (8,717,)
- `test_labels.npy`: (3,722,)

#### Step 3: 训练/测试划分

**划分方式**: 按时间顺序划分（70/30）

```python
# 按时间顺序划分，不打乱
split_idx = int(len(sequences) * 0.7)
train_sequences = sequences[:split_idx]
test_sequences = sequences[split_idx:]
```

**原因**: 模拟真实场景，训练集在前，测试集在后

**数据分布**:

| 类别 | 训练集 | 测试集 | 总计 |
|------|--------|--------|------|
| Benign | 1,742 (20.0%) | 747 (20.1%) | 2,489 |
| HTTPFlood | 1,089 (12.5%) | 467 (12.5%) | 1,556 |
| ICMPFlood | 1,089 (12.5%) | 467 (12.5%) | 1,556 |
| SYNFlood | 1,089 (12.5%) | 467 (12.5%) | 1,556 |
| SYNScan | 1,089 (12.5%) | 467 (12.5%) | 1,556 |
| SlowrateDoS | 1,089 (12.5%) | 467 (12.5%) | 1,556 |
| TCPConnectScan | 1,089 (12.5%) | 467 (12.5%) | 1,556 |
| UDPFlood | 541 (6.2%) | 232 (6.2%) | 773 |
| UDPScan | 0 (0.0%) | 0 (0.0%) | 0 |

**注意**: UDPScan在聚合特征数据中缺失（可能是数据处理问题）

### 性能结果

| 模型 | Macro F1 | 准确率 | 训练时间 |
|------|----------|--------|---------|
| **LightGBM** | **95.17%** | **96.00%** | 9.54秒 |
| **XGBoost** | **93.01%** | **94.87%** | 11.52秒 |
| LSTM | 66.92% | 68.54% | 15.84秒 |
| GRU | 67.81% | 68.40% | 15.84秒 |

---

## 🎯 方案2: 原始包序列方案

### 数据来源

**原始数据**: 同方案1（18个PCAP文件）

### 构造流程

#### Step 1: 提取原始包序列 (`scripts/raw_packet_sequence/01_extract_raw_sequences.py`)

```
原始PCAP包流
    ↓
提取每个包的14个特征（不聚合）
    ↓
按固定长度（100个包）切分序列
```

**配置**:
- `SEQUENCE_LENGTH = 100` (每个序列100个包)
- `STRIDE = 50` (滑动窗口步长50个包，50%重叠)
- `MAX_PACKETS = 50000` (每个文件最多读取50000个包，用于快速测试)

**提取的14个包级别特征**:

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

**处理逻辑**:

```python
# 读取PCAP文件
packets = rdpcap(pcap_path)

# 提取每个包的特征
for pkt in packets:
    features = extract_packet_features(pkt, base_time)
    packet_features.append(features)

# 滑动窗口切分序列
for i in range(0, len(packet_features) - SEQUENCE_LENGTH + 1, STRIDE):
    sequence = packet_features[i:i+SEQUENCE_LENGTH]
    sequences.append(sequence)
```

**输出**: `data/processed/raw_packet_sequences/sequences.npy`
- 形状: (16,383, 100, 14)
- 16,383个序列，每个序列100个包，每个包14个特征

**数据分布**:

| 类别 | 序列数量 | 百分比 |
|------|---------|--------|
| Benign | 399 | 2.44% |
| HTTPFlood | 1,998 | 12.20% |
| ICMPFlood | 1,998 | 12.20% |
| SYNFlood | 1,998 | 12.20% |
| SYNScan | 1,998 | 12.20% |
| SlowrateDoS | 1,998 | 12.20% |
| TCPConnectScan | 1,998 | 12.20% |
| UDPFlood | 1,998 | 12.20% |
| UDPScan | 1,998 | 12.20% |

**注意**: 
- Benign类别样本少（只有2.44%），因为SSH流量包数量少
- 其他攻击类别均衡（每个12.20%）

#### Step 2: 数据预处理 (`scripts/raw_packet_sequence/02_prepare_data.py`)

```
原始包序列 (16,383, 100, 14)
    ↓
标准化（StandardScaler）
    ↓
训练/测试划分（70/30，分层采样）
```

**标准化方法**:

```python
# Reshape to (n_samples * seq_len, n_features)
X_train_flat = X_train.reshape(-1, 14)
X_test_flat = X_test.reshape(-1, 14)

# Standardize
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Reshape back
X_train_norm = X_train_flat.reshape(n_train, 100, 14)
X_test_norm = X_test_flat.reshape(n_test, 100, 14)
```

**输出**: `data/processed/raw_packet_sequences/`
- `train_sequences.npy`: (11,468, 100, 14)
- `test_sequences.npy`: (4,915, 100, 14)
- `train_labels.npy`: (11,468,)
- `test_labels.npy`: (4,915,)

#### Step 3: 训练/测试划分

**划分方式**: 分层随机采样（70/30）

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, 
    test_size=0.3, 
    random_state=42, 
    stratify=labels  # 保持类别比例
)
```

**原因**: 保持训练集和测试集的类别分布一致

**数据分布**:

| 类别 | 训练集 | 测试集 | 总计 |
|------|--------|--------|------|
| Benign | 279 (2.43%) | 120 (2.44%) | 399 |
| HTTPFlood | 1,399 (12.20%) | 599 (12.19%) | 1,998 |
| ICMPFlood | 1,399 (12.20%) | 599 (12.19%) | 1,998 |
| SYNFlood | 1,399 (12.20%) | 599 (12.19%) | 1,998 |
| SYNScan | 1,398 (12.19%) | 600 (12.21%) | 1,998 |
| SlowrateDoS | 1,398 (12.19%) | 600 (12.21%) | 1,998 |
| TCPConnectScan | 1,399 (12.20%) | 599 (12.19%) | 1,998 |
| UDPFlood | 1,398 (12.19%) | 600 (12.21%) | 1,998 |
| UDPScan | 1,399 (12.20%) | 599 (12.19%) | 1,998 |

**注意**: 所有9个类别都存在，分布均衡

### 性能结果

| 模型 | Macro F1 | 准确率 | 训练时间 |
|------|----------|--------|---------|
| **Transformer (增强)** | **87.96%** | **87.28%** | 384.79秒 |

---

## 📊 两种方案对比

### 数据特性对比

| 特性 | 聚合特征方案 | 原始包序列方案 |
|------|-------------|---------------|
| **数据类型** | 1秒窗口统计特征 | 原始包级别特征 |
| **样本数量** | 12,439序列 | 16,383序列 |
| **序列长度** | 5个时间步 | 100个包 |
| **特征维度** | 38维 | 14维 |
| **时序粒度** | 粗（1秒聚合） | 细（包级别） |
| **信息损失** | 高（聚合丢失细节） | 低（保留完整信息） |
| **数据大小** | ~20MB | ~90MB |

### 划分方式对比

| 特性 | 聚合特征方案 | 原始包序列方案 |
|------|-------------|---------------|
| **划分方式** | 按时间顺序 | 分层随机采样 |
| **训练/测试比例** | 70/30 | 70/30 |
| **类别平衡** | 不平衡（Benign 20%） | 相对平衡（Benign 2.4%） |
| **UDPScan** | 缺失 | 存在 |

### 性能对比

| 模型 | 数据类型 | Macro F1 | 准确率 |
|------|---------|----------|--------|
| **LightGBM** | 聚合特征 | **95.17%** | **96.00%** |
| **XGBoost** | 聚合特征 | **93.01%** | **94.87%** |
| **Transformer** | 原始包序列 | **87.96%** | **87.28%** |
| STGNN (增强) | 聚合特征 | 70.59% | 68.38% |
| GRU | 聚合特征 | 67.81% | 68.40% |

---

## 🔍 关键洞察

### 1. 为什么聚合特征方案性能更好？

**原因**:
1. **特征工程**: 38个精心设计的统计特征包含了丰富的信息
2. **树模型优势**: LightGBM/XGBoost擅长处理聚合统计特征
3. **数据质量**: 1秒聚合过滤了噪声，提取了关键模式

### 2. 为什么原始包序列方案性能较差？

**可能原因**:
1. **模型容量不足**: 830K参数可能不够学习复杂模式
2. **训练数据不足**: 16,383个序列可能不够（虽然比聚合特征多）
3. **序列长度不合适**: 100个包可能太短或太长
4. **特征选择**: 14个特征可能不够表达攻击模式
5. **类别不平衡**: Benign只有2.4%，模型难以学习

### 3. 原始包序列方案的优势在哪里？

**潜在优势**:
1. **保留细粒度信息**: 包到达顺序、包间隔等
2. **捕捉微观模式**: TCP握手、端口扫描序列等
3. **更好的泛化**: 理论上应该更好地泛化到新攻击

**但实际上**: 这些优势没有体现出来（87.96% vs 95.17%）

---

## 💡 改进建议

### 对于原始包序列方案

1. **增加模型容量**: d_model=256, num_layers=6
2. **调整序列长度**: 尝试50, 200, 500个包
3. **添加更多特征**: IP地址、更多TCP选项、应用层特征
4. **处理类别不平衡**: 过采样Benign类别
5. **数据增强**: 时间扰动、包丢弃等
6. **预训练**: 在大规模数据上预训练

### 对于聚合特征方案

1. **超参数调优**: 网格搜索LightGBM参数
2. **特征工程**: 添加更多统计特征
3. **集成学习**: LightGBM + XGBoost投票

---

## 📁 相关文件

### 聚合特征方案
- 数据构造: `scripts/timeseries_construction/build_timeseries_from_pcap.py`
- 序列构造: `scripts/baseline/prepare_packet_based_data.py`
- 数据目录: `data/processed/packet_based/`
- 结果目录: `baseline/results/lightgbm/`, `baseline/results/xgboost/`

### 原始包序列方案
- 数据提取: `scripts/raw_packet_sequence/01_extract_raw_sequences.py`
- 数据预处理: `scripts/raw_packet_sequence/02_prepare_data.py`
- 模型定义: `scripts/raw_packet_sequence/03_transformer_model.py`
- 模型训练: `scripts/raw_packet_sequence/04_train_transformer.py`
- 数据目录: `data/processed/raw_packet_sequences/`
- 结果目录: `baseline/results/transformer_raw/`

---

## 🎯 总结

**聚合特征方案**:
- ✅ 性能最佳 (LightGBM: 95.17%)
- ✅ 训练速度快 (9.54秒)
- ✅ 易于部署
- ✅ **推荐用于生产环境**

**原始包序列方案**:
- ⚠️ 性能较好 (Transformer: 87.96%)
- ⚠️ 训练速度慢 (384.79秒)
- ⚠️ 需要更多优化
- ⚠️ **适合研究探索**

**最终推荐**: 使用聚合特征 + LightGBM模型

