# 任务定义：序列分类 vs 包分类

## 🎯 核心问题：任务是什么？

**问题**: 是预测每个包的类型吗？

**答案**: ❌ **不是！** 是预测**整个序列**的类型。

---

## 📊 任务对比

### ❌ 错误理解：包分类（Packet Classification）

```
输入: 单个数据包
    ↓
[包1的14个特征]
    ↓
模型预测
    ↓
输出: 这个包的类型
    ↓
预测: HTTPFlood
```

**这不是我们的任务！**

**问题**:
- 单个包无法判断攻击类型
- 需要看一段时间内的流量模式
- 例如：一个SYN包可能是正常的TCP握手，也可能是SYN Flood的一部分

### ✅ 正确理解：序列分类（Sequence Classification）

```
输入: 一段时间内的流量（序列）
    ↓
[包1, 包2, 包3, ..., 包100]
或
[时刻1, 时刻2, 时刻3, 时刻4, 时刻5]
    ↓
模型预测
    ↓
输出: 这段流量的类型
    ↓
预测: HTTPFlood
```

**这才是我们的任务！**

---

## 🔍 详细说明

### 方案1: 聚合特征方案

#### 输入

**一个序列 = 5秒的流量**

```
时刻1 (第1秒):
  - packet_count: 1000
  - syn_count: 10
  - avg_packet_size: 500
  - unique_dst_ports: 5
  ... (共38个特征)

时刻2 (第2秒):
  - packet_count: 1050
  - syn_count: 12
  - avg_packet_size: 480
  - unique_dst_ports: 6
  ... (共38个特征)

时刻3 (第3秒):
  - packet_count: 980
  - syn_count: 8
  - avg_packet_size: 520
  - unique_dst_ports: 4
  ... (共38个特征)

时刻4 (第4秒):
  - packet_count: 1020
  - syn_count: 11
  - avg_packet_size: 490
  - unique_dst_ports: 5
  ... (共38个特征)

时刻5 (第5秒):
  - packet_count: 990
  - syn_count: 9
  - avg_packet_size: 510
  - unique_dst_ports: 5
  ... (共38个特征)
```

**形状**: (5, 38)
- 5个时间步
- 每个时间步38个特征

#### 输出

**这5秒流量的类型**: Benign

**标签**: 0

#### 训练数据

```
训练集: 8,717个序列
  - 每个序列代表5秒的流量
  - 每个序列有一个标签（0-8）

测试集: 3,722个序列
  - 每个序列代表5秒的流量
  - 每个序列有一个标签（0-8）
```

#### 模型

```python
输入: (batch_size, 5, 38)
    ↓
LightGBM / LSTM / GRU / ...
    ↓
输出: (batch_size, 9)  # 9个类别的概率
    ↓
预测: argmax → 类别ID
```

---

### 方案2: 原始包序列方案

#### 输入

**一个序列 = 100个连续的数据包**

```
包1:
  - timestamp: 0.000
  - packet_length: 60
  - protocol: 1 (TCP)
  - src_port: 52341
  - dst_port: 80
  - tcp_flags_syn: 1
  - tcp_flags_ack: 0
  ... (共14个特征)

包2:
  - timestamp: 0.001
  - packet_length: 60
  - protocol: 1 (TCP)
  - src_port: 52341
  - dst_port: 80
  - tcp_flags_syn: 0
  - tcp_flags_ack: 1
  ... (共14个特征)

包3:
  - timestamp: 0.002
  - packet_length: 1500
  - protocol: 1 (TCP)
  - src_port: 52341
  - dst_port: 80
  - tcp_flags_syn: 0
  - tcp_flags_ack: 1
  - tcp_flags_psh: 1
  ... (共14个特征)

...

包100:
  - timestamp: 0.150
  - packet_length: 60
  - protocol: 1 (TCP)
  - src_port: 52341
  - dst_port: 80
  - tcp_flags_syn: 0
  - tcp_flags_ack: 1
  - tcp_flags_fin: 1
  ... (共14个特征)
```

**形状**: (100, 14)
- 100个包
- 每个包14个特征

#### 输出

**这100个包所属的流量类型**: HTTPFlood

**标签**: 1

#### 训练数据

```
训练集: 11,468个序列
  - 每个序列包含100个连续的包
  - 每个序列有一个标签（0-8）

测试集: 4,915个序列
  - 每个序列包含100个连续的包
  - 每个序列有一个标签（0-8）
```

#### 模型

```python
输入: (batch_size, 100, 14)
    ↓
Transformer / LSTM / ...
    ↓
输出: (batch_size, 9)  # 9个类别的概率
    ↓
预测: argmax → 类别ID
```

---

## 🔑 关键点

### 1. 任务是序列分类，不是包分类

**序列分类**:
- 输入: 一段流量（多个包或多个时间步）
- 输出: 这段流量的类型
- 例子: 这5秒的流量是HTTPFlood

**包分类**（不是我们的任务）:
- 输入: 单个包
- 输出: 这个包的类型
- 例子: 这个包是HTTPFlood的一部分

### 2. 为什么不能做包分类？

**原因**: 单个包无法判断攻击类型

**例子1: SYN包**
```
单个SYN包:
  - tcp_flags_syn=1
  - tcp_flags_ack=0

可能是:
  - 正常的TCP握手 (Benign)
  - SYN Flood攻击 (SYNFlood)
  - 端口扫描 (SYNScan)

无法判断！需要看一段时间内的模式。
```

**例子2: HTTP请求包**
```
单个HTTP请求包:
  - dst_port=80
  - packet_length=500
  - tcp_flags_psh=1

可能是:
  - 正常的HTTP请求 (Benign)
  - HTTP Flood攻击 (HTTPFlood)

无法判断！需要看请求频率、模式等。
```

### 3. 序列长度的选择

**聚合特征方案**: 5秒窗口
- 5个时间步（每步1秒）
- 足够捕捉攻击模式
- 不会太长（实时性好）

**原始包序列方案**: 100个包
- 约0.1-1秒的流量（取决于流量速率）
- 足够捕捉包级别的模式
- 不会太长（计算成本可控）

---

## 📊 数据标注方式

### 标注粒度

**不是**: 每个包都有标签 ❌

**而是**: 每个序列有一个标签 ✅

### 标注来源

**原始PCAP文件名决定标签**:

```
文件: Goldeneye_BS1_nogtp.pcapng
    ↓
标签: HTTPFlood (类别1)
    ↓
从这个文件提取的所有序列都标记为HTTPFlood
```

**例子**:

| PCAP文件 | 标签 | 类别ID |
|---------|------|--------|
| SSH_BS1_nogtp.pcapng | Benign | 0 |
| Goldeneye_BS1_nogtp.pcapng | HTTPFlood | 1 |
| ICMPflood_BS1_nogtp.pcapng | ICMPFlood | 2 |
| SYNflood_BS1_nogtp.pcapng | SYNFlood | 3 |
| SYNScan_BS1_nogtp.pcapng | SYNScan | 4 |
| Slowloris_BS1_nogtp.pcapng | SlowrateDoS | 5 |
| TCPConnect_BS1_nogtp.pcapng | TCPConnectScan | 6 |
| UDPflood_BS1_nogtp.pcapng | UDPFlood | 7 |
| UDPScan_BS1_nogtp.pcapng | UDPScan | 8 |

### 标注假设

**假设**: 一个PCAP文件中的所有流量都是同一类型

**合理性**:
- ✅ 数据集是专门构造的（每个文件只包含一种攻击）
- ✅ 符合实验室环境的数据收集方式

**局限性**:
- ⚠️ 真实环境中，流量是混合的
- ⚠️ 需要更细粒度的标注（流级别或包级别）

---

## 🎯 模型的工作方式

### LightGBM (聚合特征)

```python
# 输入: 一个5秒的序列
sequence = [
    [时刻1的38个特征],
    [时刻2的38个特征],
    [时刻3的38个特征],
    [时刻4的38个特征],
    [时刻5的38个特征],
]  # 形状: (5, 38)

# LightGBM需要展平
sequence_flat = sequence.flatten()  # 形状: (190,)

# 预测
prediction = lightgbm.predict(sequence_flat)
# 输出: HTTPFlood (类别1)
```

**LightGBM看到的**: 190个数值（5×38）

**LightGBM学习的**: 这190个数值的模式 → 攻击类型

### Transformer (原始包序列)

```python
# 输入: 一个100包的序列
sequence = [
    [包1的14个特征],
    [包2的14个特征],
    ...
    [包100的14个特征],
]  # 形状: (100, 14)

# Transformer处理序列
# 1. 特征嵌入
embedded = feature_embedding(sequence)  # (100, 128)

# 2. 位置编码
embedded = embedded + positional_encoding  # (100, 128)

# 3. Transformer编码器
encoded = transformer_encoder(embedded)  # (100, 128)

# 4. 全局池化
pooled = global_average_pooling(encoded)  # (128,)

# 5. 分类器
prediction = classifier(pooled)  # (9,)
# 输出: HTTPFlood (类别1)
```

**Transformer看到的**: 100个包的序列

**Transformer学习的**: 包之间的依赖关系 → 攻击类型

---

## 💡 总结

### 任务定义

**任务**: 序列分类（Sequence Classification）

**输入**: 一段流量（序列）
- 聚合特征: 5秒的流量（5个时间步）
- 原始包序列: 100个连续的包

**输出**: 这段流量的类型
- 9个类别: Benign, HTTPFlood, ICMPFlood, SYNFlood, SYNScan, SlowrateDoS, TCPConnectScan, UDPFlood, UDPScan

**不是**: 预测每个包的类型 ❌

### 为什么是序列分类？

**原因**: 攻击是一种**模式**，需要看一段时间内的流量

**例子**:
- SYN Flood: 短时间内大量SYN包
- HTTP Flood: 短时间内大量HTTP请求
- 端口扫描: 访问多个不同的端口

**单个包无法判断这些模式！**

### 数据标注

**粒度**: 序列级别（不是包级别）

**来源**: PCAP文件名

**假设**: 一个文件中的所有流量都是同一类型

---

希望这个详细的说明回答了你的问题！🎉

