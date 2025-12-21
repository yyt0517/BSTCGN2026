# 项目演进历程

> **文档版本**: V2.0.0-dev  
> **更新日期**: 2025-10-29

---

## 📖 项目简介

RiskNICLP是一个基于图神经网络的5G网络入侵检测项目，在5G-NIDD数据集上实现高精度多分类检测，并通过贝叶斯方法增强不确定性量化和可解释性。

---

## 🚀 演进时间线

### V0.1.0 - 项目启动（2025-10-17）

#### 初始目标
- 在5G-NIDD数据集上实现网络入侵检测
- 利用特征依赖图（FDG）先验知识
- 探索图神经网络在网络安全领域的应用

#### 主要工作
- **数据准备**: 处理1,215,890条样本，提取41个特征
- **FDG设计**: 构建57节点、193边的特征依赖图
- **技术选型**: PyTorch + DGL + GATv2

#### 关键决策
- 选择GATv2作为基础架构（注意力机制）
- 采用异构图建模（12种边类型）
- 分层划分数据集（70/15/15）

---

### V1.0.0 - GNN基线完成（2025-10-21）

#### 核心成果
- ✅ **测试集Macro-F1**: 99.95%
- ✅ **准确率**: 99.994%
- ✅ **误分类**: 11/182,384（0.006%）
- ✅ **长尾类别**: ICMPFlood（仅173样本）F1达到99.7%

#### 技术突破

##### 1. 样本→节点值注入（最关键）
**问题**: 初始实现中，所有样本共享同一个图表示，导致模型无法区分不同样本。

**解决方案**: 为每个样本创建独立的图副本，将样本的41个特征值注入到对应节点的`value_channel`。

**效果**: Macro-F1提升到0.98+（**15倍提升**）

<augment_code_snippet path="archive/gnn_baseline_v1/scripts/model_training.py" mode="EXCERPT">
````python
# 关键代码片段
for i, feature_name in enumerate(feature_names):
    node_idx = node_name_to_idx[feature_name]
    graph.nodes[node_idx].data['value'][0, i] = sample_features[i]
````
</augment_code_snippet>

##### 2. 特征标准化
**问题**: 特征值范围差异大（如Bytes可达10^9），导致NaN loss。

**解决方案**: 使用StandardScaler标准化所有特征。

**效果**: 训练稳定，无NaN loss。

##### 3. 类别不平衡处理
**问题**: 类别分布极不平衡（最大类占50%，最小类仅0.09%）。

**解决方案**: 使用加权CrossEntropyLoss，权重与类别频率成反比。

**效果**: 长尾类别F1达到99.7%，无明显偏向。

##### 4. 早停机制
**问题**: 训练200个epoch耗时过长。

**解决方案**: 监控验证集F1，patience=10。

**效果**: 在Epoch 18达到最佳，节省约172小时训练时间。

#### 架构细节

```
GNN基线架构（V1）:
输入: 样本特征 [41]
  ↓
样本→节点值注入
  ↓
FDG图 (57节点, 193边, 12边类型)
  ↓
GATv2 Layer 1 (74→128, 4 heads)
  ↓
GATv2 Layer 2 (128→64, 4 heads)
  ↓
全局池化 (mean)
  ↓
分类器 (64→9)
  ↓
输出: 类别概率 [9]
```

#### 局限性

尽管V1取得了优异的分类性能，但存在以下局限：

1. **无不确定性量化**: 无法评估预测的置信度
2. **无OOD检测**: 无法识别未知攻击（零日攻击）
3. **可解释性有限**: 仅有注意力权重，缺乏先验-后验对齐分析
4. **无时序建模**: 未利用时间序列信息
5. **无校准评估**: 无法评估预测概率的可靠性

---



## 📚 参考文献

### V1相关
- **V1主文档**: [`../../archive/gnn_baseline_v1/README_V1.md`](../../archive/gnn_baseline_v1/README_V1.md)
- **V1实现计划**: [`../../archive/gnn_baseline_v1/docs/GNN_BASELINE_IMPLEMENTATION_PLAN.md`](../../archive/gnn_baseline_v1/docs/GNN_BASELINE_IMPLEMENTATION_PLAN.md)
- **V1技术细节**: [`../../archive/gnn_baseline_v1/docs/GNN_TECHNICAL_DETAILS.md`](../../archive/gnn_baseline_v1/docs/GNN_TECHNICAL_DETAILS.md)



### 数据集与FDG
- **5G-NIDD数据简述**: [`../dataset/5G-NIDD数据简述/`](../dataset/5G-NIDD数据简述/)
- **FDG设计**: [`../fdg/5G-NIDD_Feature_Dependency_Graph_Design.md`](../fdg/5G-NIDD_Feature_Dependency_Graph_Design.md)



