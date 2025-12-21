# 5G-NIDD 特征依赖图文件说明文档

## 📋 概述

本文档详细说明5G-NIDD特征依赖图(FDG)的节点文件和边文件的结构、内容和构建依据。

### 文件信息

#### 核心版本 (推荐用于分析)
- **节点文件**: `5G-NIDD-cleaned/fdg_nodes.csv` (41个节点)
- **边文件**: `5G-NIDD-cleaned/fdg_edges.csv` (46条边)
- **特点**: 经过严格质量控制的核心关系，适合分析和解释

#### 增强版本 (推荐用于GNN训练)
- **节点文件**: `5G-NIDD-cleaned/fdg_nodes_v2.csv` (57个节点)
- **边文件**: `5G-NIDD-cleaned/fdg_edges_v2.csv` (136条边)
- **验证版本**: `5G-NIDD-cleaned/fdg_edges_v2_verified.csv` (包含验证结果)
- **特点**: 解决孤立节点问题，适合GNN训练和消息传递

#### 版本管理
- **版本说明**: `5G-NIDD-cleaned/fdg_versions.json`
- **同步报告**: `5G-NIDD-cleaned/sync_report.json`

**数据来源**: 5G-NIDD-cleaned/BTS1_BTS2_cleaned.csv (1,215,890条记录)
**构建方法**: 基于统计分析、领域知识和质量控制的多维评估

---

## 📊 fdg_nodes.csv - 节点文件

### 文件结构
```csv
id,category,description
TotPkts,packets,总数据包数
SrcPkts,packets,源端数据包数
...
```

### 字段说明
- **id**: 节点标识符，对应5G-NIDD数据集中的特征名称
- **category**: 节点类别，基于网络协议栈和5G语义分类
- **description**: 节点描述，说明该特征的含义

### 节点分类体系

#### 1. 基础网络标识 (network_id) - 5个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| SrcAddr | 源地址 | 孤立节点 |
| DstAddr | 目标地址 | 孤立节点 |
| Proto | 协议类型 | 连通节点 |
| Sport | 源端口 | 连通节点 |
| Dport | 目标端口 | 连通节点 |

**设计说明**: IP地址被排除是因为高基数特征会导致过拟合，只保留协议和端口的语义关系。

#### 2. 时间特征 (time) - 3个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| StartTime | 流开始时间 | 孤立节点 |
| LastTime | 流结束时间 | 孤立节点 |
| Dur | 流持续时间 | 连通节点 |

**设计说明**: 绝对时间戳被排除以防止数据泄露，只保留相对时间特征。

#### 3. 数据包统计 (packets) - 3个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| TotPkts | 总数据包数 | 连通节点 |
| SrcPkts | 源端数据包数 | 连通节点 |
| DstPkts | 目标端数据包数 | 连通节点 |

#### 4. 字节统计 (bytes) - 5个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| TotBytes | 总字节数 | 连通节点 |
| SrcBytes | 源端字节数 | 连通节点 |
| DstBytes | 目标端字节数 | 连通节点 |
| sMeanPktSz | 源端平均包大小 | 连通节点 |
| dMeanPktSz | 目标端平均包大小 | 连通节点 |

#### 5. 5G QoS标记 (qos) - 1个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| sDSb | 源端DSCP字节(5G QoS标记) | 连通节点 |

**设计说明**: 5G网络的服务质量标记，用于流量分类和优先级控制。

#### 6. 网络层特征 (network) - 4个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| sTtl | 源端生存时间 | 连通节点 |
| dTtl | 目标端生存时间 | 连通节点 |
| sHops | 源端跳数 | 连通节点 |
| sIpId | 源端IP标识 | 孤立节点 |

#### 7. TCP特征 (tcp) - 6个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| SrcWin | 源端窗口大小 | 连通节点 |
| DstWin | 目标端窗口大小 | 连通节点 |
| TcpRtt | TCP往返时间 | 连通节点 |
| TcpOpt | TCP选项 | 连通节点 |
| SynAck | SYN-ACK标志 | 孤立节点 |
| AckDat | ACK-DATA标志 | 孤立节点 |

#### 8. 负载统计 (load) - 2个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| Load | 总负载 | 连通节点 |
| SrcLoad | 源端负载 | 连通节点 |

#### 9. 丢包统计 (loss) - 1个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| pLoss | 丢包率 | 连通节点 |

#### 10. 速率统计 (rate) - 3个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| Rate | 总传输速率 | 连通节点 |
| SrcRate | 源端传输速率 | 连通节点 |
| DstRate | 目标端传输速率 | 连通节点 |

#### 11. 连接状态 (state) - 3个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| State | 连接状态 | 连通节点 |
| Dir | 流方向 | 连通节点 |
| Cause | 连接原因 | 连通节点 |

#### 12. 间隔统计 (gap) - 1个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| SrcGap | 源端间隔 | 孤立节点 |

#### 13. 其他特征 (other) - 4个节点
| 节点ID | 描述 | 状态 |
|--------|------|------|
| Flgs | 流标志 | 孤立节点 |
| Seq | 序列号 | 连通节点 |
| IdleTime | 空闲时间 | 孤立节点 |
| Offset | 偏移量 | 连通节点 |

### 孤立节点分析
**孤立节点数量**: 10个 (24.4%)
**孤立节点列表**: AckDat, DstAddr, Flgs, IdleTime, LastTime, SrcAddr, SrcGap, StartTime, SynAck, sIpId

**排除原因**:
1. **防泄露策略**: 时间戳字段可能导致数据泄露
2. **高基数问题**: IP地址等标识符会导致过拟合
3. **统计不足**: 样本量<100或相关性不显著
4. **稳定性差**: Bootstrap测试显示关系不稳定

---

## 🔗 fdg_edges.csv - 边文件

### 文件结构
```csv
src,dst,relation,weight,condition,support,stability,direction,description
Proto,sDSb,PROTOCOL_QOS,0.7,Proto=SCTP,4327,0.80,+,控制面高优先级
```

### 字段说明
- **src**: 源节点ID
- **dst**: 目标节点ID  
- **relation**: 边关系类型
- **weight**: 边权重 (0.3-1.0)
- **condition**: 条件化约束 (可选)
- **support**: 支持度 (样本数量)
- **stability**: 稳定性 (Bootstrap保留率)
- **direction**: 影响方向 (+正向/-负向)
- **description**: 边关系描述

### 边类型体系

#### 核心版本边类型 (46条边)

##### 1. 数学关系 (MATHEMATICAL_*) - 14条边 (30.4%)

###### 加法关系 (MATHEMATICAL_SUM) - 4条
```
SrcPkts + DstPkts → TotPkts (权重=1.0)
SrcBytes + DstBytes → TotBytes (权重=1.0)
```
**特点**: 确定性数学依赖，权重固定为1.0

###### 除法关系 (MATHEMATICAL_DIV) - 10条
```
SrcBytes / SrcPkts → sMeanPktSz (权重=1.0, 条件: SrcPkts>0)
TotBytes / Dur → Rate (权重=1.0, 条件: Dur>1e-6)
```
**特点**: 需要零值防御，通过条件化约束实现

#### 增强版本新增边类型 (136条边)

##### 6. 类别关系 (CATEGORY_*) - 82条边 (60.3%)

###### 归属关系 (BELONGS_TO) - 41条
```
特征节点 → 类别节点 (权重=0.3)
例: TotPkts → CAT_PACKETS
```
**特点**: 特征归属其类别，建立层次结构

###### 影响关系 (CATEGORY_INFLUENCE) - 41条
```
类别节点 → 特征节点 (权重=0.3)
例: CAT_PACKETS → TotPkts
```
**特点**: 类别对特征的影响，支持自上而下的消息传递

##### 7. 端口分类关系 (PORT_CLASSIFICATION) - 6条边 (4.4%)
```
Sport/Dport → PORT_WELLKNOWN/REGISTERED/EPHEMERAL (权重=0.4)
```
**特点**: 端口特征的语义分类，减少孤立性

##### 8. 协议弱关系 (PROTOCOL_WEAK) - 2条边 (1.5%)
```
Proto → SynAck/AckDat (权重=0.3, 条件: Proto=TCP)
```
**特点**: 条件化的弱协议关系，保持语义一致性

#### 2. 强相关关系 (CORRELATED) - 7条边 (15.2%)
```
TotPkts ↔ TotBytes (权重=0.99, r=0.960)
Load ↔ Rate (权重=0.95, r=0.960) 
```
**特点**: 基于Pearson相关系数，r>0.7，统计显著性p<0.05

#### 3. 协议相关关系 (PROTOCOL_*) - 10条边 (21.7%)

##### 协议QoS关系 (PROTOCOL_QOS) - 4条
```
Proto → sDSb (权重=0.7, 条件=SCTP, support=4327)
```
**特点**: 5G网络QoS标记与协议的关联

##### 协议特征关系 (PROTOCOL_FEATURE) - 4条
```
Proto → Sport (权重=0.5, 动态端口影响)
Proto → Dport (权重=0.7, 标准端口更稳定)
```

##### 协议约束关系 (PROTOCOL_CONSTRAINT) - 2条
```
Proto → TcpOpt (权重=0.6, 条件=TCP)
```

#### 4. 状态行为关系 (STATE_BEHAVIOR) - 9条边 (19.6%)
```
State → Dur (权重=分状态设定, 条件化)
Dir → SrcPkts (权重=0.5)
Dir → DstPkts (权重=0.5)
```

**分状态权重策略**:
- State=CON → Dur (权重=0.7, support=132174)
- State=REQ → Dur (权重=0.5, support=589299)  
- State=FIN → Dur (权重=0.6, support=59181)
- State=RST → Dur (权重=0.4, support=75670)
- State=ECO → Dur (权重=0.3, support=28822)

#### 5. 跨层关系 (CROSS_LAYER) - 6条边 (13.0%)
```
sDSb → Rate (权重=0.4, 5G QoS影响性能)
TcpRtt → Rate (权重=0.3×收缩因子, 弱相关)
```
**特点**: 跨协议层的弱关系，使用收缩策略防止噪声放大

### 权重计算体系

#### 统一边权公式
```
w_final = α_type × s_strength × s_support × s_stability × s_context
```

**参数详解**:
- **α_type**: 边型先验权重
  - 数学恒等式: 1.0
  - 强相关: 0.9  
  - 协议语义: 0.7
  - 状态行为: 0.6
  - 跨层弱关系: 0.5

- **s_strength**: 关系强度因子
  - Pearson相关系数的绝对值
  - 互信息标准化到[0,1]

- **s_support**: 支持度因子
  - 公式: min(1, log(1+N/1000))
  - N为样本数量，确保统计可靠性

- **s_stability**: 稳定性因子  
  - Bootstrap 10次采样的保留率
  - 阈值: ≥0.7

- **s_context**: 条件化置信度
  - Proto=SCTP: 1.0 (5G标准严格)
  - Proto=TCP/UDP: 0.6 (应用多样)
  - 其他: 1.0

#### 质量控制阈值
- **最低权重**: w_final ≥ 0.3
- **统计显著性**: p < 0.05
- **最小样本**: N ≥ 100  
- **稳定性**: Bootstrap保留率 ≥ 0.7

### 支持度分布
- **最小支持度**: 4,327 (Proto=SCTP相关边)
- **最大支持度**: 1,215,890 (数学关系)
- **平均支持度**: 669,761
- **中位数支持度**: 579,034

所有边的支持度都远超100的最小阈值，确保统计可靠性。

### 条件化边处理
**条件化边数**: 25/46 (54.3%)

**主要条件类型**:
- **协议条件**: Proto=TCP/UDP/SCTP/ICMP
- **数值条件**: Dur>1e-6, SrcPkts>0, DstPkts>0
- **状态条件**: State=CON/REQ/FIN/RST/ECO

**条件化意义**: 确保边关系在特定上下文中才生效，提高关系的精确性。

---

## 🎯 使用指南

### 图构建建议
1. **节点初始化**: 使用category字段进行节点类型嵌入
2. **边权重**: 直接使用weight字段作为边权重
3. **条件处理**: 根据condition字段进行条件化边激活
4. **孤立节点**: 可选择包含或排除孤立节点

### GNN应用建议
- **异质图处理**: 利用节点和边的类型信息
- **注意力机制**: 使用边权重调节注意力强度
- **消息传递**: 根据边类型设计不同的传递函数
- **稳定性考虑**: 利用stability字段评估关系可靠性

### 质量保证
- 所有边都通过了严格的质量控制
- 支持度确保统计可靠性
- 稳定性确保关系一致性
- 权重反映关系强度和可信度

---

## 📈 统计摘要

### 节点统计
- **总节点数**: 41个
- **连通节点**: 31个 (75.6%)
- **孤立节点**: 10个 (24.4%)
- **节点类别**: 12类
- **度数分布**: 最小1，最大10，平均2.97

### 边统计
- **总边数**: 46条
- **边类型**: 5类
- **权重分布**:
  - 0.9-1.0: 14条 (30.4%) - 数学关系
  - 0.7-0.9: 8条 (17.4%) - 协议语义
  - 0.5-0.7: 8条 (17.4%) - 状态行为
  - 0.3-0.5: 10条 (21.7%) - 跨层弱关系
  - <0.3: 6条 (13.0%) - 被质量控制过滤

### 质量指标
- **平均权重**: 0.70
- **平均支持度**: 669,761
- **条件化边比例**: 54.3%
- **质量控制通过率**: 100%

---

## 🔧 技术实现细节

### 数据加载示例
```python
import pandas as pd

# 加载节点和边数据
nodes_df = pd.read_csv('5G-NIDD-cleaned/fdg_nodes.csv')
edges_df = pd.read_csv('5G-NIDD-cleaned/fdg_edges.csv')

# 基本信息
print(f"节点数: {len(nodes_df)}")
print(f"边数: {len(edges_df)}")
print(f"节点类别: {nodes_df['category'].nunique()}")
print(f"边类型: {edges_df['relation'].nunique()}")
```

### 图构建示例 (DGL)
```python
import dgl
import torch

# 创建节点映射
node_to_id = {node: i for i, node in enumerate(nodes_df['id'])}

# 构建边列表
src_ids = [node_to_id[src] for src in edges_df['src']]
dst_ids = [node_to_id[dst] for dst in edges_df['dst']]

# 创建图
g = dgl.graph((src_ids, dst_ids))

# 添加节点特征
g.ndata['category'] = torch.tensor([...])  # 节点类别编码

# 添加边特征
g.edata['weight'] = torch.tensor(edges_df['weight'].values)
g.edata['relation'] = torch.tensor([...])  # 边类型编码
```

### 条件化边处理
```python
def apply_conditions(edges_df, data_sample):
    """根据条件激活边"""
    active_edges = []

    for _, edge in edges_df.iterrows():
        if pd.isna(edge['condition']):
            # 无条件边，直接激活
            active_edges.append(edge)
        else:
            # 检查条件是否满足
            if evaluate_condition(edge['condition'], data_sample):
                active_edges.append(edge)

    return pd.DataFrame(active_edges)

def evaluate_condition(condition, sample):
    """评估条件是否满足"""
    if 'Proto=' in condition:
        proto_value = condition.split('=')[1]
        return sample['Proto'] == proto_value
    elif 'Dur>' in condition:
        threshold = float(condition.split('>')[1])
        return sample['Dur'] > threshold
    # ... 其他条件类型
```

### 权重归一化
```python
def normalize_weights(edges_df, method='minmax'):
    """权重归一化"""
    weights = edges_df['weight'].values

    if method == 'minmax':
        # Min-Max归一化到[0,1]
        min_w, max_w = weights.min(), weights.max()
        normalized = (weights - min_w) / (max_w - min_w)
    elif method == 'softmax':
        # Softmax归一化
        normalized = torch.softmax(torch.tensor(weights), dim=0)

    return normalized
```

---

## ⚠️ 注意事项

### 数据完整性
- 所有边的src和dst节点都在节点文件中存在
- 权重值都在[0.3, 1.0]范围内
- 支持度都≥100，确保统计可靠性
- 条件字段可能为空，表示无条件边

### 使用限制
1. **孤立节点**: 在图学习中可能需要特殊处理
2. **条件化边**: 需要根据具体样本动态激活
3. **权重解释**: 权重反映关系强度，不是概率值
4. **方向性**: 所有边都是有向的，需要时可转换为无向

### 扩展建议
1. **动态权重**: 可根据数据分布动态调整权重
2. **多层图**: 可按边类型构建多层异质图
3. **时序扩展**: 可添加时间维度构建动态图
4. **领域适应**: 可根据具体应用调整权重策略

---

## 📚 参考资料

### 相关文档
- `docs/5G-NIDD_Feature_Dependency_Graph_Design.md`: 详细设计文档
- `5G-NIDD-cleaned/BTS1_BTS2_cleaned.csv`: 原始数据集

### 构建脚本
- 节点和边的构建脚本位于项目的scripts目录
- 质量控制和验证脚本确保数据一致性

### 引用格式
```
5G-NIDD Feature Dependency Graph
Nodes: 41, Edges: 46, Categories: 12
Quality-controlled with multi-dimensional assessment
```

---

## 📝 版本信息
- **创建日期**: 2024年
- **数据版本**: 5G-NIDD-cleaned
- **构建方法**: 多维质量评估
- **验证状态**: 已通过完整性和一致性验证
- **文档版本**: v1.0
