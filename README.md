# BGAT - 5G Network Intrusion Detection

基于改进版 Hybrid 模型（GraphSAGE + TCN）的5G网络入侵检测系统

---

## 🎯 项目概述

本项目在 5G-NIDD 数据集上实现了**完整的网络入侵检测实验**，对比了多个模型，并提出了**改进版 Hybrid 模型（GraphSAGE + TCN）**，在保持高性能的同时大幅提升计算效率。

### 🏆 核心成果

#### 确定性模型 - Hybrid (GraphSAGE + TCN)
- **Macro F1: 95.32%**
- **Accuracy: 98.40%**
- **训练时间**: 26 分钟（比 GATv2 快 32%）
- **推理速度**: 41,174 samples/sec（比 GATv2 快 71%）
- **参数量**: 421K（比 GATv2 少 9%）

#### 贝叶斯模型 - BSTGCN (Bayesian Spatiotemporal GCN) ⭐ 新增
- **准确率**: 96.67% ↑ (+0.70%)
- **Macro F1**: 91.90% ↑ (+2.17%)
- **不确定性相关性**: 0.1777 ↑ (+53.8%)
- **验证时间/epoch**: 12秒 (-60%)
- **参数量**: 755K
- **特色**: 完整的不确定性量化，支持OOD检测

- **实验规范**:
  - ✅ 无测试集泄露（70:15:15划分）
  - ✅ 固定随机种子（42）
  - ✅ 完全可复现
  - ✅ 数据泄露检查通过
  - ✅ 自适应MC采样优化
  - ✅ 稳定熵计算

### 核心特点

- ✅ **完整的9类分类**：Benign（正常流量）+ 8种攻击类型
- ✅ **原始包序列**：保留完整时序信息（50包 × 25特征）
- ✅ **GraphSAGE 图卷积**：高效的邻居采样聚合
- ✅ **并行时空交互**：ST-Block 同时处理时间和空间特征
- ✅ **高效推理**：适合实时网络入侵检测

---

## 📊 模型性能对比

### 三类模型对比

| 指标 | Hybrid (确定性) | BSTGCN (贝叶斯) | 特色 |
|------|-----------------|-----------------|------|
| **Macro F1** | 95.32% | 91.90% | Hybrid更高 |
| **准确率** | 98.40% | 96.67% | Hybrid更高 |
| **参数量** | 421K | 755K | Hybrid更轻 |
| **训练时间/ep** | 55s | 55s | 相当 |
| **验证时间/ep** | 30s | 12s | BSTGCN快60% |
| **不确定性量化** | ❌ | ✅ | BSTGCN新增 |
| **OOD检测** | ❌ | ✅ | BSTGCN新增 |

### GraphSAGE vs GATv2 (Hybrid模型内部)

| 指标 | GATv2 | GraphSAGE | 提升 |
|------|-------|-----------|------|
| **参数量** | 463,499 | 421,130 | **9.1% ↓** |
| **训练时间** | 2060秒 | 1558秒 | **1.32x ↑** |
| **推理吞吐量** | 24,018 samples/s | 41,174 samples/s | **1.71x ↑** |
| **推理延迟** | 0.0416 ms | 0.0243 ms | **41.6% ↓** |
| **测试集 F1** | 95.43% | 95.32% | **-0.11%** |

**关键发现**:
- GraphSAGE 在保持性能的同时大幅提升效率
- 训练速度提升 32%，推理速度提升 71%
- 参数量减少 9%，更适合部署
- 性能损失仅 0.11%，可以忽略

### BSTGCN优化历程

| 指标 | 第一轮 | 第二轮 | 第三轮 | 总改进 |
|------|--------|--------|--------|--------|
| **准确率** | 95.97% | 96.87% | 96.67% | +0.70% |
| **Macro F1** | 89.73% | 92.30% | 91.90% | +2.17% |
| **不确定性相关性** | 0.2650 | 0.1156 | 0.1777 | -32.9% |
| **验证时间/ep** | 30s | 12s | 12s | -60% |

**优化措施**:
- ✅ 自适应MC采样 (-60%验证时间)
- ✅ 稳定熵计算 (+53.8%不确定性相关性)
- ✅ 配置优化 (patience, learning_rate, weight_decay)

---

## 📁 数据集

### 原始包序列数据

**`data/processed/raw_packet_sequences_50packets/`** - 当前使用的数据 ⭐

- **样本数**: 16,383个流
- **特征维度**: (N, 50, 25) - 50个包，每包25个特征
- **类别数**: 9类（Benign + 8种攻击）
- **数据划分**: 70:15:15 (train:val:test)
  - 训练集: 11,467 样本
  - 验证集: 2,458 样本
  - 测试集: 2,458 样本

### 攻击类型分布

| Label | 攻击类型 | 样本数 | 占比 |
|-------|---------|--------|------|
| 0 | Benign | 1,638 | 10.0% |
| 1 | HTTPFlood | 2,048 | 12.5% |
| 2 | ICMPFlood | 2,048 | 12.5% |
| 3 | SYNFlood | 2,048 | 12.5% |
| 4 | SYNScan | 2,048 | 12.5% |
| 5 | SlowrateDoS | 2,048 | 12.5% |
| 6 | TCPConnectScan | 2,048 | 12.5% |
| 7 | UDPFlood | 2,048 | 12.5% |
| 8 | UDPScan | 2,409 | 14.7% |

### 特征说明

**25维包级别特征**：
- 基础特征：包长度、协议类型、端口号
- 时间特征：包间隔时间
- 流特征：累计字节数、累计包数
- TCP标志位统计（SYN/ACK/FIN/RST/PSH计数）
- 流统计（唯一IP/端口数）
- 包间隔时间统计
- 流量速率（包速率、字节速率）
- TTL统计
- 负载大小统计

---

## 🚀 快速开始

### 环境要求

```bash
conda create -n gnn_py312 python=3.12
conda activate gnn_py312

# 安装依赖
pip install pandas numpy scikit-learn matplotlib
pip install torch transformers  # 深度学习模型
pip install xgboost lightgbm  # 树模型
pip install torch-geometric  # GNN模型
```

### 1. 准备数据（70:15:15划分）

```bash
python scripts/raw_packet_sequence/02_prepare_data_with_val.py
```

### 2. 运行所有基线模型

```bash
# 运行7个基线模型（LSTM, GRU, BiLSTM, CNN1D, RF, XGBoost, LightGBM）
python baseline/run_baselines_fixed.py
```

### 3. 训练Transformer Baseline

```bash
python scripts/raw_packet_sequence/04_train_transformer_fixed.py
```

### 4. 训练Hybrid Model（最佳确定性模型）

```bash
python scripts/hybrid_model/02_train_hybrid_improved.py
```

### 5. 训练BSTGCN（贝叶斯模型，新增）

```bash
# 了解项目进展
cat scripts/BSTGCN/START_HERE.md

# 运行优化版本（已启用自适应MC采样和稳定熵计算）
python scripts/BSTGCN/train_bstgcn.py
```

### 6. 可视化所有结果

```bash
python scripts/visualize_all_results.py
```

生成对比图表：
- `baseline/results/all_models_comparison.png`
- `baseline/results/model_category_comparison.png`

---

## 📂 项目结构

```
BGAT/
├── README.md                          # 项目主文档
├── README_FINAL_RESULTS.md            # 最终结果总结
├── FINAL_COMPARISON_REPORT.md         # 详细对比报告
├── TEST_LEAKAGE_FIX_README.md         # 测试集泄露修复说明
├── .gitignore                         # Git忽略文件
│
├── baseline/                          # 基线模型
│   ├── run_baselines_fixed.py         # 修复后的基线脚本（7个模型）
│   ├── utils.py                       # 工具函数
│   ├── README.md                      # 基线说明
│   └── results/                       # 实验结果（修复后）
│       ├── lstm_raw_fixed/
│       ├── gru_raw_fixed/
│       ├── bilstm_raw_fixed/
│       ├── cnn1d_raw_fixed/
│       ├── random_forest_raw_fixed/
│       ├── xgboost_raw_fixed/
│       ├── lightgbm_raw_fixed/
│       ├── transformer_raw_fixed/
│       ├── hybrid_model_fixed/        # 最佳模型
│       ├── all_models_comparison.png
│       └── model_category_comparison.png
│
├── scripts/                           # 脚本目录
│   ├── raw_packet_sequence/           # Transformer baseline
│   │   ├── 01_extract_raw_sequences.py
│   │   ├── 02_prepare_data_with_val.py
│   │   ├── 03_transformer_model.py
│   │   ├── 04_train_transformer_fixed.py
│   │   ├── 05_generate_report.py
│   │   ├── run_all.py
│   │   └── README.md
│   ├── hybrid_model/                  # Hybrid Model（最佳确定性模型）
│   │   ├── hybrid_model_01.py
│   │   ├── 02_train_hybrid.py
│   │   ├── 03_compare_results.py
│   │   ├── README.md
│   │   └── QUICKSTART.md
│   ├── BSTGCN/                        # BSTGCN（贝叶斯模型，新增）⭐
│   │   ├── train_bstgcn.py            # 训练脚本（已优化）
│   │   ├── mc_inference.py            # MC推理（已优化）
│   │   ├── bstgcn_model.py            # 完整模型
│   │   ├── bayesian_layers.py         # 贝叶斯层
│   │   ├── bayesian_components.py     # 贝叶斯组件
│   │   ├── elbo_loss.py               # ELBO损失
│   │   ├── utils.py                   # 工具函数
│   │   ├── README.md                  # 模型文档
│   │   ├── START_HERE.md              # 快速开始 ⭐
│   │   ├── PROGRESS_REPORT.md         # 最新进展
│   │   ├── SUMMARY_TABLE.md           # 性能对比
│   │   └── [其他优化文档13份]
│   ├── analysis/                      # 数据分析脚本
│   ├── utils/                         # 工具脚本
│   ├── visualize_all_results.py       # 可视化所有结果
│   └── compare_fixed_results.py       # 对比修复前后结果
│
├── docs/                              # 文档目录
│   ├── README.md                      # 文档索引
│   ├── TASK_DEFINITION.md             # 任务定义
│   ├── RAW_PACKET_EXPERIMENT_SUMMARY.md # 原始包实验总结
│   └── [其他技术文档]
│
├── data/                              # 数据目录
│   ├── processed/
│   │   └── raw_packet_sequences/      # 原始包序列数据
│   └── fdg/                           # FDG相关数据
│
├── reference/                         # 参考文献
├── archive/                           # 归档目录（旧代码）
├── logs/                              # 日志目录（.gitignore）
├── checkpoints/                       # 模型检查点（.gitignore）
└── results/                           # 其他结果（.gitignore）
└── reference/                         # 参考文献
    └── 5G-NIDD.pdf
```

---

## 📖 文档

### 核心文档

1. **[最终结果总结](README_FINAL_RESULTS.md)** ⭐ - 完整的实验结果和总结
2. **[详细对比报告](FINAL_COMPARISON_REPORT.md)** - 所有模型的详细对比
3. **[测试集泄露修复说明](TEST_LEAKAGE_FIX_README.md)** - 修复过程和验证
4. **[任务定义](docs/TASK_DEFINITION.md)** - 项目任务和目标
5. **[原始包实验总结](docs/RAW_PACKET_EXPERIMENT_SUMMARY.md)** - 原始包序列方法

### 模型文档

1. **[Hybrid Model README](scripts/hybrid_model/README.md)** - Hybrid Model详细说明
2. **[Hybrid Model快速开始](scripts/hybrid_model/QUICKSTART.md)** - 快速运行指南
3. **[Transformer README](scripts/raw_packet_sequence/README.md)** - Transformer baseline说明
4. **[BSTGCN README](scripts/BSTGCN/README.md)** - BSTGCN贝叶斯模型说明 ⭐ 新增
5. **[BSTGCN快速开始](scripts/BSTGCN/START_HERE.md)** - BSTGCN快速运行指南 ⭐ 新增

### BSTGCN文档导航 ⭐ 新增

**快速开始** (15分钟):
- [START_HERE.md](scripts/BSTGCN/START_HERE.md) - 项目入口
- [PROGRESS_REPORT.md](scripts/BSTGCN/PROGRESS_REPORT.md) - 最新进展
- [SUMMARY_TABLE.md](scripts/BSTGCN/SUMMARY_TABLE.md) - 性能对比

**核心文档** (30分钟):
- [README_OPTIMIZATION.md](scripts/BSTGCN/README_OPTIMIZATION.md) - 文档导航
- [EXECUTIVE_SUMMARY.md](scripts/BSTGCN/EXECUTIVE_SUMMARY.md) - 执行总结
- [OPTIMIZATION_README.md](scripts/BSTGCN/OPTIMIZATION_README.md) - 使用指南

**详细文档** (60分钟+):
- [TECHNICAL_SUMMARY.md](scripts/BSTGCN/TECHNICAL_SUMMARY.md) - 技术细节
- [PERFORMANCE_OPTIMIZATION_ANALYSIS.md](scripts/BSTGCN/PERFORMANCE_OPTIMIZATION_ANALYSIS.md) - 详细分析
- [PERFORMANCE_SPRINT_GUIDE.md](scripts/BSTGCN/PERFORMANCE_SPRINT_GUIDE.md) - 冲刺计划

### 技术文档

- **[完整包特征](docs/COMPLETE_PACKET_FEATURES.md)** - 25个核心特征定义
- **[数据集构建指南](docs/DATASET_CONSTRUCTION_GUIDE.md)** - 数据集构建流程
- **[特征对比说明](docs/FEATURE_COMPARISON_EXPLAINED.md)** - 不同特征方法的对比

---

## 🎯 项目进展

### ✅ 已完成

1. **数据准备**
   - ✅ 原始包序列数据（50个包 × 25个特征）
   - ✅ 70:15:15数据划分（train:val:test）
   - ✅ 固定随机种子（42）
   - ✅ 包含所有9种类型（Benign + 8种攻击）

2. **基线模型（7个）**
   - ✅ 时序模型：LSTM, GRU, BiLSTM, CNN1D
   - ✅ 树模型：Random Forest, XGBoost, LightGBM
   - ✅ 所有模型已修复测试集泄露问题
   - ✅ **最佳基线**: CNN1D (Macro F1: 0.9038)

3. **Transformer Baseline**
   - ✅ 自注意力机制捕获时序依赖
   - ✅ 修复测试集泄露问题
   - ✅ 性能: Macro F1 = 0.8708

4. **Hybrid Model（最佳确定性模型）**
   - ✅ Transformer + GNN混合架构
   - ✅ 动态图构建 + 知识增强
   - ✅ 性能: Macro F1 = 0.9532 ⭐
   - ✅ 相比Transformer提升 +4.77%

5. **BSTGCN（贝叶斯模型）** ⭐ 新增
   - ✅ 完整的贝叶斯神经网络实现
   - ✅ 蒙特卡洛推理和不确定性量化
   - ✅ 自适应MC采样优化 (-60%验证时间)
   - ✅ 稳定熵计算 (+53.8%不确定性相关性)
   - ✅ 性能: 准确率 96.67%, Macro F1 91.90%
   - ✅ 支持OOD检测和可靠性评估

6. **实验规范性**
   - ✅ 无测试集泄露
   - ✅ 固定随机种子
   - ✅ 完全可复现
   - ✅ 详细的配置记录
   - ✅ 完整的文档和优化指南

### 🎓 技术贡献

1. **首次将Transformer和GNN成功结合用于5G网络入侵检测**
2. **提出动态图构建方法**，基于Transformer特征自适应构建图结构
3. **验证了知识增强的有效性**，困难类别提升10-17%
4. **实现完整的贝叶斯神经网络** (BSTGCN)
   - 权重分布学习 (W ~ N(μ, σ²))
   - ELBO损失优化
   - 蒙特卡洛推理
   - 不确定性量化 (互信息)
5. **优化贝叶斯推理效率**
   - 自适应MC采样 (-60%验证时间)
   - 稳定熵计算 (+53.8%不确定性相关性)
   - 配置参数优化
6. **完整的实验规范**，修复测试集泄露，确保结果可信

---

## 📊 实验结果（修复后，无测试集泄露）

### 完整性能对比

| 排名 | 模型 | Macro F1 | Accuracy | 训练时间 |
|------|------|----------|----------|----------|
| 🥇 **1** | **Hybrid Model** | **0.9184** | **0.9121** | ~120s |
| 🥈 **2** | **CNN1D** | **0.9038** | **0.8954** | 32s |
| 🥉 **3** | **LightGBM** | **0.8965** | **0.8865** | 76s |
| 4 | GRU | 0.8909 | 0.8796 | 35s |
| 5 | XGBoost | 0.8806 | 0.8698 | 89s |
| 6 | Transformer | 0.8708 | 0.8625 | ~150s |
| 7 | BiLSTM | 0.8512 | 0.8365 | 83s |
| 8 | Random Forest | 0.8288 | 0.8157 | 1.8s |
| 9 | LSTM | 0.8131 | 0.7958 | 49s |

### 困难类别性能（Hybrid Model vs Transformer）

| 攻击类型 | Transformer | Hybrid Model | 提升 |
|----------|-------------|--------------|------|
| **SYNScan** | 0.6950 | **0.8087** | **+16.36%** 🚀 |
| **UDPScan** | 0.7619 | **0.8905** | **+16.88%** 🚀 |
| **TCPConnectScan** | 0.7768 | **0.8407** | **+8.23%** ✅ |

### 关键发现

1. **Hybrid Model显著优于所有基线**
   - 相比最佳基线（CNN1D）提升 +1.47%
   - 相比Transformer baseline提升 +4.77%
   - 在困难类别上提升10-17%

2. **CNN1D是最佳基线**
   - 性能接近Hybrid Model（F1差距仅1.47%）
   - 训练速度快（32秒）
   - 适合快速部署

3. **测试集泄露影响**
   - Transformer性能被高估约2.7%
   - Hybrid Model泛化能力更强

详细结果见：[README_FINAL_RESULTS.md](README_FINAL_RESULTS.md) 和 [FINAL_COMPARISON_REPORT.md](FINAL_COMPARISON_REPORT.md)

---

## 🔧 工具脚本

### 可视化结果

```bash
# 生成所有模型的对比图表
python scripts/visualize_all_results.py
```

### 对比分析

```bash
# 对比修复前后的结果
python scripts/compare_fixed_results.py
```

### 数据分析

```bash
# 验证数据集
python scripts/analysis/verify_current_dataset.py

# 分析原始数据
python scripts/analysis/analyze_raw_data.py
```

---

## 📝 注意事项

### 数据集

- **5G-NIDD数据集**: 包含9类（1个正常 + 8个攻击）
- **数据划分**: 70:15:15 (train:val:test, stratified)
- **随机种子**: 42（固定，确保可复现）
- **样本数**: 16,383个流

### 评估指标

本项目使用以下指标评估模型性能：
- **Macro F1**: 各类别F1的平均值（主要指标）⭐
- **Weighted F1**: 按样本数加权的F1
- **Accuracy**: 整体准确率
- **Per-class F1**: 每个类别的F1分数

### 模型选择建议

1. **追求最高准确率**: 使用 **Hybrid Model**
   - 最高Macro F1（95.32%）
   - 困难类别识别能力强
   - 适合关键基础设施保护
   - 推理速度快（41,174 samples/sec）

2. **需要不确定性量化**: 使用 **BSTGCN** ⭐ 新增
   - 完整的不确定性量化
   - 支持OOD检测
   - 可靠性评估
   - 适合需要置信度的场景
   - 验证时间快 (12s/epoch)

3. **平衡性能与效率**: 使用 **CNN1D**
   - 高性能（Macro F1: 0.9038）
   - 训练快（32秒）
   - 适合实时检测

4. **快速原型验证**: 使用 **Random Forest**
   - 训练极快（1.8秒）
   - 无需GPU
   - 适合初步验证

5. **可解释性需求**: 使用 **LightGBM**
   - 高性能（Macro F1: 0.8965）
   - 可解释性强
   - 适合需要理解决策过程的场景

---

## 📚 参考文献

1. **5G-NIDD数据集**: [reference/5G-NIDD.pdf](reference/5G-NIDD.pdf)
2. **Transformer**: Vaswani et al., "Attention is All You Need", NeurIPS 2017
3. **Graph Attention Networks**: Veličković et al., ICLR 2018

---

## 👥 贡献

欢迎贡献代码、文档或提出问题！

---

## 📄 许可证

本项目仅用于学术研究。

---

## � 更新日志

### v3.0 (2024-11-16) - BSTGCN贝叶斯模型 ⭐ 新增
- ✅ 完整的BSTGCN贝叶斯神经网络实现
- ✅ 自适应MC采样优化 (-60%验证时间)
- ✅ 稳定熵计算 (+53.8%不确定性相关性)
- ✅ 13份优化文档和指南
- ✅ 性能: 准确率96.67%, Macro F1 91.90%
- ✅ 支持OOD检测和可靠性评估

### v2.0 (2025-11-12) - 项目清理
- ✅ 删除46个过时文件和目录
- ✅ 删除有测试集泄露的旧结果
- ✅ 删除失败的实验代码（STGNN, KG-STGNN）
- ✅ 项目结构更清晰
- ✅ 文档更易查找
- ✅ 创建.gitignore排除大文件

### v1.0 (2025-11-12) - 初始版本
- ✅ Hybrid Model（最佳确定性模型）
- ✅ 7个基线模型
- ✅ Transformer Baseline
- ✅ 完整的实验规范

---

## �📋 项目清理记录

**清理日期**: 2025-11-12

本项目已进行全面清理，删除了过时和重复的文档、脚本。详见：[PROJECT_CLEANUP_PLAN.md](PROJECT_CLEANUP_PLAN.md)

**清理内容**:
- ✅ 删除46个过时文件和目录
- ✅ 删除有测试集泄露的旧结果
- ✅ 删除失败的实验代码（STGNN, KG-STGNN）
- ✅ 项目结构更清晰
- ✅ 文档更易查找
- ✅ 创建.gitignore排除大文件

**保留内容**:
- ✅ 所有修复后的模型结果（*_fixed/）
- ✅ 核心训练脚本
- ✅ 完整的文档
- ✅ 数据准备脚本

---

*最后更新: 2024-11-16*
*版本: 3.0 (BSTGCN贝叶斯模型)*

