# 5G-NIDD FDG版本对比文档

## 📋 概述

本文档详细对比5G-NIDD特征依赖图(FDG)的不同版本，帮助用户选择适合的版本进行研究和应用。

---

## 🔄 版本总览

| 版本 | 文件 | 节点数 | 边数 | 孤立节点 | 主要用途 |
|------|------|--------|------|----------|----------|
| 核心版本 | `fdg_*.csv` | 41 | 46 | 10个 | 分析、解释 |
| 增强版本 | `fdg_*_v2.csv` | 57 | 136 | 0个 | GNN训练 |
| 验证版本 | `fdg_*_verified.csv` | 57 | 136 | 0个 | 研究验证 |

---

## 📊 详细对比

### 1. 核心版本 (Core Version)

#### 文件
- `5G-NIDD-cleaned/fdg_nodes.csv`
- `5G-NIDD-cleaned/fdg_edges.csv`

#### 特点
- ✅ **高质量关系**: 所有边都经过严格的统计验证
- ✅ **易于解释**: 关系纯粹，符合领域知识
- ✅ **计算高效**: 边数较少，计算开销小
- ⚠️ **孤立节点**: 10个节点无连接，影响GNN效果

#### 边类型分布
```
MATHEMATICAL_DIV     : 10 (21.7%)  - 数学除法关系
STATE_BEHAVIOR       : 9  (19.6%)  - 状态行为关系  
CORRELATED           : 7  (15.2%)  - 强相关关系
CROSS_LAYER          : 6  (13.0%)  - 跨层关系
PROTOCOL_FEATURE     : 4  (8.7%)   - 协议特征关系
MATHEMATICAL_SUM     : 4  (8.7%)   - 数学加法关系
PROTOCOL_QOS         : 4  (8.7%)   - 协议QoS关系
PROTOCOL_CONSTRAINT  : 2  (4.3%)   - 协议约束关系
```

#### 孤立节点
```
AckDat, DstAddr, Flgs, IdleTime, LastTime, 
SrcAddr, SrcGap, StartTime, SynAck, sIpId
```

#### 适用场景
- 🔍 **特征关系分析**: 理解特征间的因果关系
- 📊 **统计建模**: 基于相关性的传统机器学习
- 📝 **论文写作**: 关系清晰，易于解释和可视化
- 🎯 **特征选择**: 基于边权重进行特征筛选

---

### 2. 增强版本 (Enhanced Version)

#### 文件
- `5G-NIDD-cleaned/fdg_nodes_v2.csv`
- `5G-NIDD-cleaned/fdg_edges_v2.csv`

#### 特点
- ✅ **完全连通**: 所有节点都有连接，无孤立节点
- ✅ **GNN友好**: 支持完整的消息传递机制
- ✅ **层次结构**: 引入类别节点，支持多层次建模
- ⚠️ **复杂度高**: 边数增加3倍，计算开销较大

#### 边类型分布
```
BELONGS_TO           : 41 (30.1%)  - 特征归属类别
CATEGORY_INFLUENCE   : 41 (30.1%)  - 类别影响特征
MATHEMATICAL_DIV     : 10 (7.4%)   - 数学除法关系
STATE_BEHAVIOR       : 9  (6.6%)   - 状态行为关系
CORRELATED           : 7  (5.1%)   - 强相关关系
PORT_CLASSIFICATION  : 6  (4.4%)   - 端口分类关系
CROSS_LAYER          : 6  (4.4%)   - 跨层关系
其他                 : 16 (11.8%)  - 协议和数学关系
```

#### 新增节点类型
```
类别节点 (13个): CAT_TCP, CAT_NETWORK_ID, CAT_TIME, ...
端口类节点 (3个): PORT_WELLKNOWN, PORT_REGISTERED, PORT_EPHEMERAL
```

#### 连接策略
1. **类别超边**: 每个特征连接到其类别节点
2. **端口分箱**: 端口特征连接到端口类别
3. **协议关联**: TCP标志位与协议建立弱连接

#### 适用场景
- 🧠 **图神经网络**: GCN, GAT, GraphSAGE等模型训练
- 🔄 **消息传递**: 需要完整邻域信息的算法
- 📈 **表示学习**: 节点嵌入和图嵌入
- 🎲 **不确定性量化**: 贝叶斯图神经网络

---

### 3. 验证版本 (Verified Version)

#### 文件
- `5G-NIDD-cleaned/fdg_edges_v2_verified.csv`
- `5G-NIDD-cleaned/edge_verification_report.csv`

#### 特点
- ✅ **统计验证**: 包含每条边的验证结果
- ✅ **可信度量**: 提供verification_score字段
- ✅ **研究友好**: 适合学术研究和方法验证
- ✅ **透明度高**: 验证过程和结果完全可追溯

#### 额外字段
```
verified           : 边关系是否通过验证 (True/False)
verification_score : 验证得分 (0.0-1.0)
```

#### 验证结果统计
```
✅ 通过验证: 25条边
⚠️ 需要注意: 5条边
验证覆盖率: 30条边/136条边 (22.1%)
```

#### 适用场景
- 🔬 **方法验证**: 验证图构建方法的有效性
- 📊 **质量评估**: 评估边关系的统计可靠性
- 📝 **学术研究**: 提供验证依据和统计支撑
- 🛠️ **调试分析**: 识别问题边关系

---

## 🎯 选择指南

### 根据应用场景选择

#### 传统机器学习
```
推荐: 核心版本
原因: 关系纯粹，特征选择清晰
模型: XGBoost, Random Forest, SVM
```

#### 图神经网络
```
推荐: 增强版本
原因: 无孤立节点，消息传递完整
模型: GCN, GAT, GraphSAGE
```

#### 贝叶斯建模
```
推荐: 验证版本
原因: 包含不确定性信息
模型: Bayesian GNN, MC Dropout
```

#### 特征工程
```
推荐: 核心版本
原因: 高权重边指导特征构造
应用: 特征选择, 特征组合
```

#### 可解释AI
```
推荐: 核心版本
原因: 关系语义清晰
应用: 注意力可视化, 路径分析
```

### 根据计算资源选择

#### 资源充足
```
推荐: 增强版本或验证版本
优势: 完整功能，最佳效果
```

#### 资源受限
```
推荐: 核心版本
优势: 计算高效，内存占用小
```

---

## 🔄 版本迁移

### 从核心版本到增强版本
```python
# 加载核心版本
core_nodes = pd.read_csv('fdg_nodes.csv')
core_edges = pd.read_csv('fdg_edges.csv')

# 加载增强版本
enhanced_nodes = pd.read_csv('fdg_nodes_v2.csv')
enhanced_edges = pd.read_csv('fdg_edges_v2.csv')

# 提取核心关系
core_relations = enhanced_edges[
    enhanced_edges['relation'].isin([
        'MATHEMATICAL_SUM', 'MATHEMATICAL_DIV', 
        'CORRELATED', 'PROTOCOL_QOS', 'PROTOCOL_FEATURE',
        'PROTOCOL_CONSTRAINT', 'STATE_BEHAVIOR', 'CROSS_LAYER'
    ])
]
```

### 从增强版本到核心版本
```python
# 过滤原始特征节点
original_features = enhanced_nodes[
    ~enhanced_nodes['category'].isin(['meta_category', 'port_class'])
]

# 过滤核心边关系
core_edges = enhanced_edges[
    (enhanced_edges['src'].isin(original_features['id'])) &
    (enhanced_edges['dst'].isin(original_features['id']))
]
```

---

## 📝 最佳实践

### 1. 版本选择原则
- **分析优先**: 选择核心版本，关系更纯粹
- **训练优先**: 选择增强版本，效果更好
- **研究优先**: 选择验证版本，证据更充分

### 2. 混合使用策略
- **特征分析**: 使用核心版本理解关系
- **模型训练**: 使用增强版本训练模型
- **结果验证**: 使用验证版本确认可靠性

### 3. 性能优化建议
- **内存优化**: 按需加载边类型
- **计算优化**: 缓存邻接矩阵
- **并行优化**: 利用图的稀疏性

---

## 📊 性能对比

| 指标 | 核心版本 | 增强版本 | 验证版本 |
|------|----------|----------|----------|
| 加载时间 | 快 | 中等 | 中等 |
| 内存占用 | 小 | 大 | 大 |
| 训练速度 | 快 | 慢 | 慢 |
| GNN效果 | 中等 | 最佳 | 最佳 |
| 可解释性 | 最佳 | 中等 | 最佳 |

---

## 🔮 未来规划

### 版本演进路线
1. **v3.0**: 动态图支持，时序关系建模
2. **v4.0**: 多模态特征融合
3. **v5.0**: 自适应图结构学习

### 功能扩展
- 条件化边的动态激活
- 边权重的自适应调整
- 图结构的在线学习

---

**版本**: v1.0 | **更新**: 2024-12-28 | **状态**: ✅ 已同步
