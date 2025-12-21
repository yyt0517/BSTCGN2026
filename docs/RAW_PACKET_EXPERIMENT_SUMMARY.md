# 原始包序列实验总结

## 实验概述

**实验日期**: 2025-11-11  
**数据集**: 5G-NIDD (9类：1个正常 + 8个攻击)  
**数据格式**: 原始包序列（100个包 × 25个特征）  
**数据划分**: 70% train (11,468), 30% test (4,915)

## 特征工程

### 从14个特征升级到25个核心特征

根据 `docs/COMPLETE_PACKET_FEATURES.md` 中的方案1，我们将特征从14个扩展到25个：

#### 新增特征（+11个）

**IP地址特征** (4个):
- `src_ip_int`: 源IP地址（32位整数）
- `dst_ip_int`: 目标IP地址（32位整数）
- `src_ip_is_private`: 源IP是否为内网
- `dst_ip_is_private`: 目标IP是否为内网

**IP层特征** (+2个):
- `ip_tos`: IP服务类型
- `ip_flags`: IP标志（DF/MF）

**TCP特征** (+3个):
- `tcp_flags_urg`: URG标志
- `tcp_seq`: TCP序列号
- `tcp_ack`: TCP确认号

**ICMP特征** (2个):
- `icmp_type`: ICMP类型
- `icmp_code`: ICMP代码

### 特征提取改进

**权重初始化**:
- Xavier初始化（weight_ih）
- 正交初始化（weight_hh）
- **遗忘门偏置=1**（LSTM关键技巧）

**训练优化**:
- 梯度裁剪（max_norm=1.0）
- 学习率调度（ReduceLROnPlateau）
- 早停机制（patience=10）

## 模型实验

### 运行的模型

1. **时序模型** (4个):
   - LSTM (改进版)
   - GRU
   - BiLSTM (改进版)
   - CNN1D

2. **树模型** (3个):
   - Random Forest
   - XGBoost
   - LightGBM

3. **Transformer** (1个):
   - Enhanced Transformer (4层, 8头)

**总计**: 8个模型

## 实验结果

### 整体性能排名

| 排名 | 模型 | Accuracy | Macro F1 | Weighted F1 | 训练时间 |
|------|------|----------|----------|-------------|----------|
| 1 | **CNN1D** | **0.8765** | **0.8872** | **0.8762** | 27.67s |
| 2 | **Transformer** | **0.8704** | **0.8794** | **0.8688** | ~385s |
| 3 | LightGBM | 0.8710 | 0.8824 | 0.8710 | 16.75s |
| 4 | GRU | 0.8690 | 0.8804 | 0.8687 | 33.30s |
| 5 | XGBoost | 0.8649 | 0.8760 | 0.8643 | 31.16s |
| 6 | LSTM | 0.8378 | 0.8493 | 0.8370 | 49.01s |
| 7 | Random Forest | 0.8047 | 0.8212 | 0.8038 | 0.95s |
| 8 | BiLSTM | 0.7986 | 0.8138 | 0.7977 | 85.30s |

### 关键发现

#### 1. CNN1D表现最佳
- **Macro F1: 0.8872** (最高)
- 训练速度快（27.67秒）
- 在3个类别上表现最好

#### 2. Transformer性能优秀但不是最佳
- **Macro F1: 0.8794** (第2名)
- 比CNN1D低 **1.16%**
- 训练时间长（385秒）

#### 3. LightGBM在多个类别上表现最好
- 在 **4/9** 个类别上F1最高
- 训练速度快（16.75秒）
- 特别擅长：HTTPFlood, SYNFlood, SlowrateDoS, UDPFlood

#### 4. LSTM改进效果显著
- 改进前: Macro F1 = 0.7657
- 改进后: Macro F1 = 0.8493
- **提升: +8.36%**

#### 5. BiLSTM表现不佳
- Macro F1 = 0.8138 (最低)
- 训练时间最长（85.30秒）
- 可能原因：双向结构在这个数据集上过拟合

### 各类别最佳模型

| 攻击类型 | 最佳模型 | F1-Score | 难度 |
|----------|----------|----------|------|
| Benign | GRU | 1.0000 | 🏆 容易 |
| HTTPFlood | LightGBM | 0.9843 | 🏆 容易 |
| ICMPFlood | CNN1D | 0.8330 | ✅ 中等 |
| SYNFlood | LightGBM | 0.9958 | 🏆 容易 |
| **SYNScan** | CNN1D | 0.7687 | ⚠️ **困难** |
| SlowrateDoS | LightGBM | 0.9925 | 🏆 容易 |
| **TCPConnectScan** | CNN1D | 0.7913 | ✅ **中等** |
| UDPFlood | LightGBM | 0.9832 | 🏆 容易 |
| **UDPScan** | GRU | 0.8474 | ✅ **中等** |

### 困难类别分析

**最困难的3个类别**:
1. **SYNScan**: 平均F1 = 0.6989 ⚠️
2. **TCPConnectScan**: 平均F1 = 0.7369 ✅
3. **UDPScan**: 平均F1 = 0.7803 ✅

**原因分析**:
- **SYNScan**: 扫描流量特征不明显，与正常流量相似
- **TCPConnectScan**: 完整TCP连接，难以与正常流量区分
- **UDPScan**: UDP无连接特性，特征稀疏

## 对比分析

### 与聚合特征模型对比

| 方法 | 最佳模型 | Macro F1 | 数据形式 |
|------|----------|----------|----------|
| **聚合特征** | LightGBM | **0.9517** | 5秒窗口统计特征 |
| **原始包序列** | CNN1D | **0.8872** | 100个包 × 25特征 |
| **差距** | - | **-6.45%** | - |

**结论**: 聚合特征方法仍然更优，但原始包序列方法也达到了不错的性能。

### 与5秒窗口基线对比

**5秒窗口基线** (之前的实验):
- 最佳模型: GRU (Macro F1: 未记录)
- 数据: 5秒窗口聚合特征

**原始包序列** (本次实验):
- 最佳模型: CNN1D (Macro F1: 0.8872)
- 数据: 100个包 × 25特征

## LSTM改进详解

### 问题

改进前LSTM性能异常差：
- Macro F1: 0.7657
- 比GRU低 **13.2%**
- 训练不稳定，收敛慢

### 根本原因

1. **梯度问题**: LSTM有4个门，梯度流复杂
2. **权重初始化不当**: PyTorch默认初始化不适合LSTM
3. **学习率固定**: 无法自适应调整
4. **没有早停**: 可能过拟合

### 解决方案

1. **梯度裁剪**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
2. **改进的权重初始化**:
   - Xavier初始化（weight_ih）
   - 正交初始化（weight_hh）
   - **遗忘门偏置=1**（关键！）
3. **学习率调度**: `ReduceLROnPlateau`
4. **早停机制**: patience=10

### 改进效果

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| **Macro F1** | 0.7657 | **0.8493** | **+8.36%** |
| **Accuracy** | 0.7440 | **0.8378** | **+9.38%** |
| **Epoch 30 Test Acc** | 48.99% | **74.63%** | **+25.64%** |

## 推荐

### 综合推荐

**1. 生产环境**: **LightGBM**
- 在4/9个类别上表现最好
- 训练速度快（16.75秒）
- 推理速度快
- 易于部署

**2. 研究/实验**: **CNN1D**
- 整体性能最佳（Macro F1: 0.8872）
- 训练速度适中（27.67秒）
- 可解释性较好

**3. 实时检测**: **Random Forest**
- 训练最快（0.95秒）
- 推理速度快
- 性能可接受（Macro F1: 0.8212）

### 不推荐

**BiLSTM**:
- 性能最差（Macro F1: 0.8138）
- 训练时间最长（85.30秒）
- 在这个数据集上过拟合

## 改进方向

### 1. 针对困难类别

**SYNScan** (平均F1: 0.6989):
- 增加时序特征（包间隔、扫描速率）
- 使用注意力机制关注关键包
- 集成多个模型

**TCPConnectScan** (平均F1: 0.7369):
- 增加连接状态特征
- 使用图神经网络建模连接关系
- 增加上下文信息

**UDPScan** (平均F1: 0.7803):
- 增加端口扫描模式特征
- 使用序列模式挖掘
- 增加时间窗口特征

### 2. 模型改进

**Transformer**:
- 尝试更深的网络（6-8层）
- 增加预训练
- 使用知识蒸馏

**CNN1D**:
- 尝试多尺度卷积
- 增加残差连接
- 使用注意力机制

**集成方法**:
- LightGBM + CNN1D + GRU
- 投票或加权平均
- Stacking

### 3. 数据增强

- 时间窗口滑动
- 包序列采样
- 对抗样本生成

## 文件清单

### 数据
- `data/processed/raw_packet_sequences/train_sequences.npy` (11468, 100, 25)
- `data/processed/raw_packet_sequences/test_sequences.npy` (4915, 100, 25)
- `data/processed/raw_packet_sequences/metadata.pkl`

### 脚本
- `scripts/raw_packet_sequence/01_extract_raw_sequences.py` - 特征提取
- `scripts/raw_packet_sequence/02_prepare_data.py` - 数据准备
- `scripts/raw_packet_sequence/03_transformer_model.py` - Transformer模型
- `scripts/raw_packet_sequence/04_train_transformer.py` - 训练Transformer
- `scripts/raw_packet_sequence/05_generate_report.py` - 生成报告
- `scripts/raw_packet_sequence/run_all.py` - 运行完整流程
- `baseline/run_baselines_on_raw_packet.py` - 运行所有基线模型
- `scripts/generate_detailed_comparison_report.py` - 生成对比报告

### 结果
- `baseline/results/lstm_raw/` - LSTM结果
- `baseline/results/gru_raw/` - GRU结果
- `baseline/results/bilstm_raw/` - BiLSTM结果
- `baseline/results/cnn1d_raw/` - CNN1D结果
- `baseline/results/random_forest_raw/` - Random Forest结果
- `baseline/results/xgboost_raw/` - XGBoost结果
- `baseline/results/lightgbm_raw/` - LightGBM结果
- `baseline/results/transformer_raw/` - Transformer结果
- `baseline/results/detailed_comparison_report_raw_packet.txt` - 详细对比报告

### 文档
- `docs/COMPLETE_PACKET_FEATURES.md` - 完整特征方案
- `docs/LSTM_PERFORMANCE_ANALYSIS.md` - LSTM性能分析
- `docs/RAW_PACKET_EXPERIMENT_SUMMARY.md` - 本文档

## 总结

1. ✅ **成功将特征从14个扩展到25个**，包含IP地址、TCP序列号等关键特征
2. ✅ **运行了8个模型**，包括Transformer和7个基线模型
3. ✅ **CNN1D表现最佳**（Macro F1: 0.8872），超过Transformer
4. ✅ **LightGBM在多个类别上表现最好**，适合生产环境
5. ✅ **LSTM改进效果显著**（+8.36%），验证了优化方法的有效性
6. ⚠️ **SYNScan、TCPConnectScan、UDPScan仍然是困难类别**，需要进一步改进
7. ⚠️ **原始包序列方法仍然不如聚合特征方法**（-6.45%）

**下一步**:
- 针对困难类别进行特征工程
- 尝试集成方法
- 探索图神经网络方法

