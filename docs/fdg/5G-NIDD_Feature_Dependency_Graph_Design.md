# 5G-NIDD 特征依赖图 (FDG) 设计文档

## 📋 概述

本文档详细描述了5G-NIDD数据集的特征依赖图(Feature Dependency Graph, FDG)构建设计，基于数据驱动的分析结果和5G网络语义知识。

### 基本信息
- **数据集**: 5G-NIDD-cleaned/BTS1_BTS2_cleaned.csv
- **特征数量**: 41个 (从原始112个压缩而来)
- **数据记录**: 1,215,890条
- **图类型**: 有向异质图

#### 当前实现 (fdg_*.csv)
- **节点数**: 57个 (41个原始特征 + 16个辅助节点)
- **边数**: 136条 (已解决孤立节点问题)
- **连通性**: 100% (所有节点连通)
- **字段数**: 10个 (已整合notes到description)
- **用途**: 支持完整的GNN训练和分析

---

## 🗂️ 特征分类

### 1. 基础网络标识 (5个)
- `SrcAddr`, `DstAddr`, `Proto`, `Sport`, `Dport`

### 2. 时间特征 (3个)
- `StartTime`, `LastTime`, `Dur`
- **注**: 已删除冗余时间特征 `RunTime`, `Mean`, `Sum`, `Min`, `Max`

### 3. 数据包统计 (3个)
- `TotPkts`, `SrcPkts`, `DstPkts`

### 4. 字节统计 (5个)
- `TotBytes`, `SrcBytes`, `DstBytes`, `sMeanPktSz`, `dMeanPktSz`

### 5. 5G QoS标记 (1个)
- `sDSb`
- **注**: 已删除低方差特征 `sTos`

### 6. 网络层特征 (4个)
- `sTtl`, `dTtl`, `sHops`, `sIpId`

### 7. TCP标志位 (4个)
- `TcpOpt`, `TcpRtt`, `SynAck`, `AckDat`

### 8. 负载统计 (2个)
- `SrcLoad`, `Load`
- **注**: 已删除冗余特征 `DstLoad`

### 9. 丢包统计 (1个)
- `pLoss`
- **注**: 已删除低方差特征 `Loss`, `SrcLoss`, `DstLoss`

### 10. 速率统计 (3个)
- `SrcRate`, `DstRate`, `Rate`

### 11. 窗口统计 (2个)
- `SrcWin`, `DstWin`

### 12. 间隔统计 (1个)
- `SrcGap`

### 13. 连接状态 (3个)
- `State`, `Dir`, `Cause`

### 14. 其他特征 (4个)
- `Flgs`, `Seq`, `IdleTime`, `Offset`

### 孤立节点处理策略

#### 核心版本中的孤立节点
以下10个节点在核心FDG中为孤立节点（无边连接）：
`AckDat`, `DstAddr`, `Flgs`, `IdleTime`, `LastTime`, `SrcAddr`, `SrcGap`, `StartTime`, `SynAck`, `sIpId`

**排除原因**：
- **时间字段**: 仅用于过滤/采样，不进入可学习特征（防泄露策略）
- **基础网络标识**: 高基数特征，直接建立边关系会导致过拟合
- **样本量不足**: 相关关系的样本数<100或统计不显著
- **稳定性不足**: Bootstrap测试显示关系不稳定

#### 增强版本的连接策略
为解决GNN训练中的孤立节点问题，增强版本采用了三种连接策略：

##### 1. 类别超边策略 (Category Supernodes)
- **原理**: 为每个特征类别创建元节点，作为该类别特征的"代表"
- **实现**:
  - 创建13个类别节点 (如`CAT_TCP`, `CAT_NETWORK_ID`, `CAT_TIME`等)
  - 每个特征通过`BELONGS_TO`关系连接到其类别节点
  - 类别节点通过`CATEGORY_INFLUENCE`关系影响特征
- **权重**: 0.3 (弱连接，不干扰原有强关系)
- **优势**: 保持语义清晰，支持层次化消息传递

##### 2. 端口分箱策略 (Port Classification)
- **原理**: 将端口特征连接到端口类别节点
- **实现**:
  - 创建3个端口类节点: `PORT_WELLKNOWN`, `PORT_REGISTERED`, `PORT_EPHEMERAL`
  - `Sport`和`Dport`通过`PORT_CLASSIFICATION`关系连接
- **权重**: 0.4 (中等强度，反映端口分类的重要性)
- **优势**: 符合网络协议语义，减少端口特征的孤立性

##### 3. 协议关联策略 (Protocol Weak Links)
- **原理**: 为协议相关的孤立节点建立弱连接
- **实现**:
  - TCP标志位(`SynAck`, `AckDat`)与`Proto`建立条件化弱连接
  - 条件: `Proto=TCP`时激活
- **权重**: 0.3 (弱连接，条件化激活)
- **优势**: 保持协议语义一致性

---

## 🔗 边关系设计

### 第1层: 确定性数学依赖 (权重=1.0)

#### 加法关系
```
SrcPkts + DstPkts → TotPkts
SrcBytes + DstBytes → TotBytes
```

#### 除法关系 (需要零值防御)
```
SrcBytes / SrcPkts → sMeanPktSz
DstBytes / DstPkts → dMeanPktSz
TotBytes / Dur → Rate (条件: Dur > ε, ε=1e-6)
SrcBytes / Dur → SrcRate (条件: Dur > ε)
DstBytes / Dur → DstRate (条件: Dur > ε)
```

**验证结果**:
- `Rate = TotBytes/Dur`: r = 0.970 ✅
- `SrcRate = SrcBytes/Dur`: r = 0.787 ✅

**实现注意**: 速率计算需要ε-截断防御，避免Dur≈0导致的异常值污染

### 第2层: 强相关关系 (权重=0.8-0.99, 去冗余优化)

```
TotPkts ↔ TotBytes (r=0.991, 权重=0.85) # 降权避免与恒等式冲突
SrcPkts ↔ SrcBytes (r=0.984, 权重=0.80) # 保留更强的分向关系
DstPkts ↔ DstBytes (r=0.986, 权重=0.82) # 保留更强的分向关系
Load ↔ SrcLoad (r=0.774, 权重=0.77)    # 从数学依赖移入
Load ↔ Rate (r=0.960, 权重=0.90)
sTtl ↔ dTtl (r=0.819, 权重=0.82)
Seq ↔ Offset (r=0.808, 权重=0.75)      # 降权避免协议语义混淆
```

**去冗余策略**: 强相关边权重适当降低，避免与数学恒等式形成信息重复的三角闭环

### 第3层: 协议条件化关系 (权重=0.3-0.8)

#### 协议与端口
```
Proto → Sport (权重=0.5)  # 动态端口影响，源端口通常是临时端口，协议弱关联，统一协议的不同连接可能使用完全不同的源端口
Proto → Dport (权重=0.7)  # 标准端口更稳定，目标端口相对可预测，协议强相关，特定协议通常对应特定的知名端口
```

#### 协议与QoS (条件化权重 + 语义先验)
```
Proto=SCTP → sDSb (权重=0.7, context="control_plane", support=4327)
Proto=ICMP → sDSb (权重=0.4, context="management", support=29636)
Proto=TCP → sDSb (权重=0.3, context="data_plane", support=278671)
Proto=UDP → sDSb (权重=0.3, context="data_plane", support=903040)
```

**条件化实现**:
- `context` 属性区分控制面/数据面/管理面
- `support` 属性记录样本支持度
- 弱相关但语义重要的边保持权重≥0.3

#### 协议与TCP特征
```
Proto=TCP → TcpOpt (权重=0.8)
Proto=TCP → TcpRtt (权重=0.6)
Proto=TCP → SrcWin (权重=0.7)
Proto=TCP → DstWin (权重=0.7)
```

### 第4层: 状态行为关系 (权重=0.4-0.6)

#### 连接状态影响 (定向 + 分状态权重)
```
State=CON → Dur (权重=0.7, support=132174, mean_dur=2.995)
State=REQ → Dur (权重=0.5, support=589299, mean_dur=1.295)
State=FIN → Dur (权重=0.6, support=59181, mean_dur=1.244)
State=RST → Dur (权重=0.4, support=75670, mean_dur=0.613)
State=ECO → Dur (权重=0.3, support=28822, mean_dur=0.011)
State → TcpRtt (权重=0.6, 条件: Proto=TCP)
```
URP, RSP, TST, NRS样本量不足100（共计889个样本，占总数据的0.0731%，且都是网络诊断或错误状态，不是核心业务流量），暂时不纳入考虑。目的是减少噪声，提高泛化能力和计算效率。

**分状态权重策略**:
- 根据状态语义和持续时间差异设定不同权重
- 记录支持度(support)作为边属性
- 状态影响时长的因果方向明确

#### 方向影响
```
Dir → SrcPkts (权重=0.5)
Dir → DstPkts (权重=0.5)
```

**验证结果**:
- `<->` (双向): SrcPkts=54.1, DstPkts=17.2
- `->` (单向): SrcPkts=1.9, DstPkts=0.9
流方向对包数有极强的影响。
从样本分布可以看出：
96.3%是单向流: 大部分网络流量是单向的（可能是失败连接、探测等）
3.5%是双向流: 少数但重要的完整交互
其他方向极少: 特殊情况，如服务端推送等

#### 连接原因
```
Cause ↔ Dur (权重=0.4)
```

### 第5层: 跨层弱关系 (权重=0.3-0.5)

#### 网络层与时间
```
sTtl ↔ Dur (权重=0.5)  # r=0.056, 互信息=0.229
sHops ↔ Rate (权重=0.4)  # r=0.007
```

#### 5G QoS与性能
```
sDSb ↔ Rate (权重=0.4)  # r=0.012, 但语义重要
sDSb ↔ pLoss (权重=0.3)  # QoS影响丢包
```

**验证结果** (主要QoS标记的平均速率):
- `ef` (Expedited Forwarding): 30,407 bps
- `af11` (Assured Forwarding): 34,649 bps
- `cs7` (Class Selector): 4,834 bps
- `cs0` (Best Effort): 1,795 bps

#### TCP性能关系 (收缩策略)
```
TcpRtt ↔ Rate (权重=0.3×s_shrink, r=0.003, 语义先验)
pLoss ↔ Rate (权重=0.3×s_shrink, r=-0.006, MI=0.052)
```

**收缩公式**: `w_final = w_base × s_support × s_stability`
- `s_support = min(1, log(1+N/1000))`
- `s_stability = bootstrap保留率`
- 防止小样本/偶然性放大弱相关关系
理论期望：
TCP RTT ↑ → Rate ↓ (延迟高，速率低)
pLoss ↑ → Rate ↓ (丢包多，速率低)
数据现实：
相关性极弱：r≈0.003 (几乎为0)
统计不显著：p>0.05
可能原因：网络复杂性、数据预处理、测量误差
---

## ⚙️ 统一边权公式

### 边权计算公式
对每条非恒等式边：
```
w_final = α_type × s_strength × s_support × s_stability × s_context
```

**参数说明**:
- `α_type`: 边型先验权重
  - 数学恒等式: 1.0 #绝对可靠，如 A+B=C
  - 强相关: 0.9 #统计验证的强关系
  - 协议语义: 0.7 #基于网络协议理论
  - 状态行为: 0.6 #基于状态机逻辑
  - 跨层弱关系: 0.5 #跨协议层的弱关系
- `s_strength`: 关系强度因子，|r|或MI标准化到[0,1]
- `s_support`: 支持度因子，min(1, log(1+N/1000)), N为样本数
- `s_stability`: bootstrap保留比例(10次采样)
- `s_context`: 条件化置信度
  - Proto=SCTP时: 1.0 #5G标准严格，行为可预测
  - Proto=TCP/UDP时: 0.5-0.8 #应用多样，行为变化大
  - 其他: 1.0

### 质量控制阈值
- **最低权重**: w_final ≥ 0.3 #过滤噪声关系
- **显著性**: p < 0.05 (适用时) #过滤偶然关系
- **最小样本**: N ≥ 100 #保证统计基础
- **稳定性**: bootstrap保留率 ≥ 0.7 #确保重采样一致性
多维评估: 从5个不同角度评估关系质量
统计严谨: 基于扎实的统计学原理
领域智慧: 融入5G网络的专业知识
质量控制: 四重阈值确保关系可靠性

---

## 📊 实际实现统计

### 边类型分布
基于实际构建的FDG，边类型分布如下：

| 边类型 | 中文名称 | 边数 | 占比 |
|--------|----------|------|------|
| MATHEMATICAL_DIV | 数学除法关系 | 10 | 21.7% |
| STATE_BEHAVIOR | 状态行为关系 | 9 | 19.6% |
| CORRELATED | 强相关关系 | 7 | 15.2% |
| CROSS_LAYER | 跨层关系 | 6 | 13.0% |
| PROTOCOL_FEATURE | 协议特征关系 | 4 | 8.7% |
| MATHEMATICAL_SUM | 数学加法关系 | 4 | 8.7% |
| PROTOCOL_QOS | 协议QoS关系 | 4 | 8.7% |
| PROTOCOL_CONSTRAINT | 协议约束关系 | 2 | 4.3% |

### 权重分布
- **权重范围**: 0.3-1.0
- **平均权重**: 0.70
- **权重≥0.3的边**: 100% (质量控制生效)

**权重区间分布**:
- 0.9-1.0: 14条 (30.4%) - 数学关系为主
- 0.7-0.9: 8条 (17.4%) - 协议语义关系
- 0.5-0.7: 8条 (17.4%) - 状态行为关系
- 0.3-0.5: 10条 (21.7%) - 跨层弱关系
- <0.3: 6条 (13.0%) - 被质量控制过滤

### 支持度分布
- **最小支持度**: 4,327
- **最大支持度**: 1,215,890
- **平均支持度**: 669,761
- **中位数支持度**: 579,034

所有边的支持度都远超100的最小阈值，确保了统计可靠性。

### 条件化边统计
- **条件化边数**: 25/46 (54.3%)
- **主要条件类型**:
  - Proto=TCP: 7条
  - Dur>1e-6: 6条
  - 各种状态条件: 5条
  - 其他协议条件: 7条

---

## 📊 图结构特性

### 节点特性
- **总节点数**: 41
- **连通节点数**: 31 (75.6%)
- **孤立节点数**: 10 (24.4%)
- **节点类别**: 12类
- **度数分布**:
  - 最小度数: 1
  - 最大度数: 10
  - 平均度数: 2.97

### 边特性
- **总边数**: 136条
- **边类型**: 12种 (包含核心关系和辅助连接)
- **权重范围**: 0.3-1.0 (平均0.44)
- **有向性**: 所有边都是有向边
- **字段数**: 10个 (src, dst, relation, weight, condition, support, stability, sign, direction, description)

#### 边类型分布
- **BELONGS_TO**: 41条 (30.1%) - 特征归属类别
- **CATEGORY_INFLUENCE**: 41条 (30.1%) - 类别影响特征
- **MATHEMATICAL_DIV**: 10条 (7.4%) - 数学除法关系
- **STATE_BEHAVIOR**: 9条 (6.6%) - 状态行为关系
- **CORRELATED**: 7条 (5.1%) - 强相关关系
- **CROSS_LAYER**: 6条 (4.4%) - 跨层关系
- **PORT_CLASSIFICATION**: 6条 (4.4%) - 端口分类关系
- **其他类型**: 16条 (11.8%) - 协议和数学关系

### 异质性
- **高异质性**: 多种节点类型和12种边类型
- **层次结构**: 核心关系 + 辅助连接的双层结构
- **领域特化**: 包含5G网络的QoS和协议关系

---

## 🎯 构建策略

### 分阶段构建
1. **第1阶段**: 构建数学依赖关系 (权重=1.0) - 14条边
2. **第2阶段**: 添加强相关关系 (权重=0.8-0.99) - 7条边
3. **第3阶段**: 添加协议相关关系 (权重=0.3-0.8) - 10条边
4. **第4阶段**: 添加状态行为关系 (权重=0.4-0.6) - 9条边
5. **第5阶段**: 添加跨层弱关系 (权重=0.3-0.5) - 6条边
6. **第6阶段**: 添加辅助连接关系 (权重=0.3-0.4) - 90条边
   - 类别归属关系 (BELONGS_TO): 41条
   - 类别影响关系 (CATEGORY_INFLUENCE): 41条
   - 端口分类关系 (PORT_CLASSIFICATION): 6条
   - 协议弱连接 (PROTOCOL_WEAK): 2条

### 权重验证
- **数学关系**: 通过计算验证 (相关性>0.7)
- **强相关**: 通过Pearson相关系数验证
- **语义关系**: 通过互信息和领域知识验证
- **协议条件**: 通过分组统计验证

### 质量控制
- **边权重阈值**: 最低0.3，确保关系有意义
- **统计显著性**: p值<0.05 (适用时)
- **样本量要求**: 每个关系至少100个样本
- **领域验证**: 符合5G网络和TCP/IP协议语义

### 字段结构
FDG边文件包含10个字段：

#### 必需字段 (4个)
- `src`: 源节点ID
- `dst`: 目标节点ID
- `relation`: 边关系类型 (12种)
- `weight`: 边权重 (0.3-1.0)

#### 质量字段 (3个)
- `condition`: 条件约束 (19.9%使用，主要用于数学关系)
- `support`: 支持度/样本数量 (89.7%使用，1,000-1,215,890)
- `stability`: 稳定性得分 (89.7%使用，0.3-1.0，平均0.92)

#### 方向字段 (2个)
- `sign`: 核心关系的正负符号 (33.8%使用，包含6条重要负向关系)
- `direction`: 辅助连接的方向标识 (66.2%使用，专用于低权重辅助关系)

#### 描述字段 (1个)
- `description`: 边的详细描述 (100%覆盖，已整合notes内容)

---

## ⚠️ 实现说明

### 最新实现特点
本文档反映最新的FDG实现，具有以下特点：

1. **完全连通性**:
   - 通过辅助连接策略解决了孤立节点问题
   - 所有57个节点(41个原始+16个辅助)都已连通
   - 支持完整的图神经网络消息传递

2. **边类型扩展**:
   - 从5类核心关系扩展到12种边类型
   - 新增辅助连接类型：BELONGS_TO, CATEGORY_INFLUENCE, PORT_CLASSIFICATION, PROTOCOL_WEAK
   - 保持核心关系的语义纯粹性

3. **字段优化**:
   - 从11个字段精简到10个字段
   - 移除冗余的notes字段，内容整合到description
   - 保留重要的sign和direction字段，各司其职

4. **连接策略**:
   - 类别超边：通过13个类别节点连接同类特征
   - 端口分箱：通过3个端口类节点连接端口特征
   - 协议关联：为TCP相关特征建立弱连接

### 方向性信息处理
实现中采用双字段方案处理不同类型的方向性信息：

#### sign字段 (核心关系正负性)
- **用途**: 标识核心关系的正负影响
- **使用范围**: 数学、相关、协议、状态关系 (46条边)
- **重要负向关系** (6条):
  - 数学分母: `Dur → Rate/SrcRate/DstRate` (分母关系)
  - 数学分母: `SrcPkts/DstPkts → MeanPktSz` (分母关系)
  - 负相关: `pLoss → Rate` (丢包率与速率负相关)

#### direction字段 (辅助连接标识)
- **用途**: 标识辅助连接的层次方向
- **使用范围**: 归属、影响、分类、弱连接关系 (90条边)
- **特征**: 低权重(0.3-0.4)，表示辅助性连接
- **专用关系**: BELONGS_TO, CATEGORY_INFLUENCE, PORT_CLASSIFICATION, PROTOCOL_WEAK

### 5G特征的实际归类
- **5G QoS关系**: `Proto → sDSb` 归类为 `PROTOCOL_QOS`
- **5G性能关系**: `sDSb → Rate/pLoss` 归类为 `CROSS_LAYER`
- **5G特征连接**: 通过类别节点与其他特征建立层次关系

这种归类更符合网络语义，避免了模糊的"5G特有"分类。

---

## 📝 使用建议

### 图神经网络应用
- **节点嵌入**: 考虑节点类别的异质性，支持原始特征和辅助节点
- **边权重**: 作为注意力机制的权重，区分核心关系(高权重)和辅助连接(低权重)
- **消息传递**: 按边类型设计不同的传递函数，支持12种边关系
- **层次建模**: 利用类别节点进行自上而下的消息传递
- **完全连通**: 所有节点都参与消息传递，无孤立节点问题

### 特征选择
- **中心性分析**: 基于度中心性和介数中心性
- **社区检测**: 识别紧密相关的特征群
- **路径分析**: 分析特征间的依赖路径

### 异常检测
- **图结构异常**: 检测不符合依赖关系的样本
- **节点异常**: 基于邻居节点的异常检测
- **边异常**: 检测权重异常的关系

---

## � 异质图实现策略

### DGL/PyG友好的图构建
```python
# 节点设计 (57个节点)
nodes = {
    'feature': 41个原始特征节点,
    'meta_category': 13个类别节点,  # CAT_TCP, CAT_NETWORK_ID等
    'port_class': 3个端口类节点   # PORT_WELLKNOWN, PORT_REGISTERED, PORT_EPHEMERAL
}

# 边设计 (136条边，12种类型)
edges = {
    'mathematical': 数学关系边 (14条),
    'correlation': 强相关边 (7条),
    'protocol': 协议相关边 (10条),
    'state': 状态行为边 (9条),
    'cross_layer': 跨层语义边 (6条),
    'belongs_to': 特征归属边 (41条),
    'category_influence': 类别影响边 (41条),
    'port_classification': 端口分类边 (6条),
    'protocol_weak': 协议弱连接边 (2条)
}

# 边属性 (10个字段)
edge_attrs = {
    'weight': 边权重 (0.3-1.0),
    'condition': 条件约束 (可选),
    'support': 样本支持度,
    'stability': bootstrap稳定性,
    'sign': 核心关系正负符号,
    'direction': 辅助连接方向标识,
    'description': 边描述信息
}
```

### R-GAT注意力机制
- **边权重**: 作为attention bias输入edge_softmax
- **条件化**: 通过gate机制(0/1或连续值)控制消息传递
- **多头注意力**: 不同头关注不同类型的边关系

### GNN→BNN不确定性传递
```python
# 图级表示
graph_repr = readout(node_embeddings, edge_weights)

# 结构不确定性
struct_uncertainty = bootstrap_variance(w_final)

# BNN输入
bnn_input = concat(graph_repr, struct_uncertainty)

# 输出: (mean, var) + 结构不确定性
output = bnn_forward(bnn_input)  # MC Dropout或Bayes-by-Backprop
```

### 防泄露策略
- **特征去重**: Tot*与Src*/Dst*二选一进入BNN
- **端口处理**: 使用分箱嵌入替代高基数字符串
- **QoS编码**: 类别嵌入而非原始字符串
- **时间字段**: 仅用于过滤/采样，不进入可学习特征

---

## �📚 参考信息

### 数据来源
- **原始数据**: 5G-NIDD-raw/BTS1_BTS2_fields_preserved.csv
- **清理后数据**: 5G-NIDD-cleaned/BTS1_BTS2_cleaned.csv
- **特征列表**: 5G-NIDD-cleaned/feature_list.txt

### 验证脚本
- **特征分析**: scripts/analyze_features_notnull.py
- **关系验证**: scripts/validate_inferred_relationships.py
- **数据保存**: scripts/save_cleaned_dataset.py

### 更新记录
- **2025-9-22**: 初始版本，基于数据驱动分析
- **2025-9-28**: 解决孤立节点问题，添加辅助连接策略
- **2025-9-29**: 字段优化，整合notes到description，完善方向性处理
- **验证状态**: 已通过数学关系、相关性和互信息验证，支持完整GNN训练

---

*本文档将随着进一步的分析和验证结果持续更新*
