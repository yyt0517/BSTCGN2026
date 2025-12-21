# FDG先验权重分析

> **版本**: V2.0.0-dev  
> **更新日期**: 2025-10-29  
> **状态**: 待分析

---

## 📋 目标

为BSTGNF的贝叶斯层提供**关系先验强度 σ_r**，基于FDG边的质量评分。

### 核心思想

在贝叶斯神经网络中，先验分布 p(W) 控制参数的不确定性：
- **高质量关系**（强统计/硬规则）→ **小σ_r**（窄先验，强约束）
- **低质量关系**（弱相关/噪声）→ **大σ_r**（宽先验，自适应学习）

这样，模型会：
1. 对高质量关系保持接近先验的权重（信任FDG知识）
2. 对低质量关系自由学习（从数据中发现模式）

---

## 📊 分析维度

### 1. 边权重（weight）
- **范围**: 0.3 - 1.0
- **含义**: 关系强度，由相关系数/专家规则/统计显著性决定
- **来源**: FDG构建时计算

### 2. 支持度（support）
- **范围**: 0.0 - 1.0
- **含义**: 样本覆盖率，有多少样本激活了这条边
- **来源**: FDG构建时统计

### 3. 稳定性（stability）
- **范围**: 0.0 - 1.0
- **含义**: 跨数据集一致性，关系在不同数据子集上的稳定程度
- **来源**: FDG构建时验证

### 4. 关系类型（edge_type）
- **类型数**: 12种
- **含义**: 关系的语义类别
- **类型列表**:
  1. `MATHEMATICAL_IDENTITY` - 数学恒等关系（如 Bytes = Pkts × AvgPktSize）
  2. `MATHEMATICAL_DERIVED` - 数学派生关系
  3. `PROTOCOL_CONSTRAINT` - 协议硬约束（如 Proto=TCP → SrcPort有效）
  4. `PROTOCOL_RELATED` - 协议相关
  5. `CORRELATED_HIGH` - 高相关性（|r| > 0.7）
  6. `CORRELATED_MEDIUM` - 中等相关性（0.4 < |r| ≤ 0.7）
  7. `CORRELATED_LOW` - 低相关性（0.2 < |r| ≤ 0.4）
  8. `SEMANTIC_GROUP` - 语义分组
  9. `TEMPORAL_COHORT` - 时序共现
  10. `STATISTICAL_DEPENDENCY` - 统计依赖
  11. `DOMAIN_KNOWLEDGE` - 领域知识
  12. `CONDITIONAL` - 条件化关系

---

## 🎯 先验映射策略

### 方法1: 综合质量评分（推荐）

```python
# 1. 计算综合质量评分
quality_score_r = w1 * weight + w2 * support + w3 * stability

# 权重建议：
w1 = 0.5  # 边权重最重要
w2 = 0.3  # 支持度次之
w3 = 0.2  # 稳定性

# 2. 映射到先验强度
log(σ_r) = a - b * quality_score_r

# 参数建议：
a = 0.0   # 截距（对应quality_score=0时的σ）
b = 2.0   # 斜率（控制映射陡峭程度）

# 结果范围：
# quality_score = 1.0 → σ_r ≈ 0.14
# quality_score = 0.5 → σ_r ≈ 0.37
# quality_score = 0.3 → σ_r ≈ 0.55
```

### 方法2: 关系类型分级（简化版）

| 关系类型 | 语义强度 | 建议σ_r | 说明 |
|---------|---------|---------|------|
| MATHEMATICAL_IDENTITY | 极强 | **0.01** | 数学恒等，几乎确定 |
| PROTOCOL_CONSTRAINT | 强 | **0.05** | 协议硬约束，高度可信 |
| CORRELATED_HIGH | 中强 | **0.1** | 高相关性，较可信 |
| MATHEMATICAL_DERIVED | 中强 | **0.1** | 数学派生，较可信 |
| PROTOCOL_RELATED | 中 | **0.2** | 协议相关，中等可信 |
| CORRELATED_MEDIUM | 中 | **0.2** | 中等相关性 |
| STATISTICAL_DEPENDENCY | 中弱 | **0.3** | 统计依赖，需验证 |
| DOMAIN_KNOWLEDGE | 中弱 | **0.3** | 领域知识，需验证 |
| CORRELATED_LOW | 弱 | **0.5** | 低相关性，弱信号 |
| SEMANTIC_GROUP | 弱 | **0.5** | 语义分组，弱约束 |
| TEMPORAL_COHORT | 弱 | **0.5** | 时序共现，弱信号 |
| CONDITIONAL | 弱 | **0.5** | 条件化，依赖条件 |

### 方法3: 混合策略（最佳）

```python
# 结合方法1和方法2
def compute_prior_sigma(edge):
    # 1. 基础σ（基于关系类型）
    sigma_base = TYPE_TO_SIGMA[edge.edge_type]
    
    # 2. 质量调整因子
    quality_score = 0.5*edge.weight + 0.3*edge.support + 0.2*edge.stability
    quality_factor = np.exp(-2.0 * quality_score)  # [0.14, 1.0]
    
    # 3. 最终σ
    sigma_r = sigma_base * quality_factor
    
    # 4. 限制范围
    sigma_r = np.clip(sigma_r, 0.01, 1.0)
    
    return sigma_r
```

---

## 📈 实现步骤

### Step 1: 加载FDG数据

```python
import pandas as pd
import numpy as np

# 加载边数据
edges_df = pd.read_csv('data/fdg/fdg_edges.csv')

# 查看数据结构
print(edges_df.columns)
# ['source', 'target', 'edge_type', 'weight', 'support', 'stability', ...]

# 统计
print(f"总边数: {len(edges_df)}")
print(f"关系类型: {edges_df['edge_type'].nunique()}")
print(f"权重范围: [{edges_df['weight'].min()}, {edges_df['weight'].max()}]")
```

### Step 2: 分析质量维度

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 边权重分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(edges_df['weight'], bins=20)
plt.title('Edge Weight Distribution')

# 2. 支持度分布
plt.subplot(1, 3, 2)
sns.histplot(edges_df['support'], bins=20)
plt.title('Support Distribution')

# 3. 稳定性分布
plt.subplot(1, 3, 3)
sns.histplot(edges_df['stability'], bins=20)
plt.title('Stability Distribution')

plt.tight_layout()
plt.savefig('experiments/v2_bstgnf/fdg_quality_distributions.png')
```

### Step 3: 关系类型分析

```python
# 按关系类型统计
type_stats = edges_df.groupby('edge_type').agg({
    'weight': ['mean', 'std', 'min', 'max'],
    'support': ['mean', 'std'],
    'stability': ['mean', 'std'],
    'edge_type': 'count'
}).round(3)

type_stats.columns = ['weight_mean', 'weight_std', 'weight_min', 'weight_max',
                      'support_mean', 'support_std', 
                      'stability_mean', 'stability_std', 'count']

print(type_stats)
```

### Step 4: 计算先验强度

```python
# 定义关系类型基础σ
TYPE_TO_SIGMA = {
    'MATHEMATICAL_IDENTITY': 0.01,
    'PROTOCOL_CONSTRAINT': 0.05,
    'CORRELATED_HIGH': 0.1,
    'MATHEMATICAL_DERIVED': 0.1,
    'PROTOCOL_RELATED': 0.2,
    'CORRELATED_MEDIUM': 0.2,
    'STATISTICAL_DEPENDENCY': 0.3,
    'DOMAIN_KNOWLEDGE': 0.3,
    'CORRELATED_LOW': 0.5,
    'SEMANTIC_GROUP': 0.5,
    'TEMPORAL_COHORT': 0.5,
    'CONDITIONAL': 0.5,
}

# 计算综合质量评分
edges_df['quality_score'] = (
    0.5 * edges_df['weight'] + 
    0.3 * edges_df['support'] + 
    0.2 * edges_df['stability']
)

# 计算先验强度（混合策略）
def compute_sigma(row):
    sigma_base = TYPE_TO_SIGMA.get(row['edge_type'], 0.5)
    quality_factor = np.exp(-2.0 * row['quality_score'])
    sigma_r = sigma_base * quality_factor
    return np.clip(sigma_r, 0.01, 1.0)

edges_df['sigma_r'] = edges_df.apply(compute_sigma, axis=1)

# 查看结果
print(edges_df[['edge_type', 'weight', 'support', 'stability', 
                'quality_score', 'sigma_r']].head(20))
```

### Step 5: 可视化先验分布

```python
# 1. σ_r分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(edges_df['sigma_r'], bins=30)
plt.title('Prior Sigma Distribution')
plt.xlabel('σ_r')

# 2. 按关系类型的σ_r
plt.subplot(1, 2, 2)
sns.boxplot(data=edges_df, x='edge_type', y='sigma_r')
plt.xticks(rotation=45, ha='right')
plt.title('Prior Sigma by Edge Type')

plt.tight_layout()
plt.savefig('experiments/v2_bstgnf/prior_sigma_distributions.png')
```

### Step 6: 保存结果

```python
# 选择需要的列
output_df = edges_df[[
    'source', 'target', 'edge_type', 
    'weight', 'support', 'stability',
    'quality_score', 'sigma_r'
]]

# 保存到CSV
output_df.to_csv('data/fdg/edge_quality_scores.csv', index=False)

print(f"✅ 已保存 {len(output_df)} 条边的先验强度到 data/fdg/edge_quality_scores.csv")
```

---

## 🔍 验证与调优

### 1. 先验强度分布检查

```python
# 统计
print(f"σ_r 范围: [{edges_df['sigma_r'].min():.3f}, {edges_df['sigma_r'].max():.3f}]")
print(f"σ_r 均值: {edges_df['sigma_r'].mean():.3f}")
print(f"σ_r 中位数: {edges_df['sigma_r'].median():.3f}")

# 按区间统计
bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
labels = ['Very Strong', 'Strong', 'Medium', 'Weak', 'Very Weak']
edges_df['prior_strength'] = pd.cut(edges_df['sigma_r'], bins=bins, labels=labels)
print(edges_df['prior_strength'].value_counts())
```

### 2. 关系类型一致性检查

```python
# 检查同类型边的σ_r是否一致
for edge_type in edges_df['edge_type'].unique():
    subset = edges_df[edges_df['edge_type'] == edge_type]
    print(f"{edge_type}: σ_r ∈ [{subset['sigma_r'].min():.3f}, {subset['sigma_r'].max():.3f}]")
```

### 3. 极端值检查

```python
# 最强先验（最小σ_r）
print("Top 10 strongest priors:")
print(edges_df.nsmallest(10, 'sigma_r')[['source', 'target', 'edge_type', 'sigma_r']])

# 最弱先验（最大σ_r）
print("\nTop 10 weakest priors:")
print(edges_df.nlargest(10, 'sigma_r')[['source', 'target', 'edge_type', 'sigma_r']])
```

---

## 📝 输出文件格式

### `data/fdg/edge_quality_scores.csv`

| 列名 | 类型 | 说明 |
|------|------|------|
| source | str | 源节点名称 |
| target | str | 目标节点名称 |
| edge_type | str | 关系类型 |
| weight | float | 边权重（0.3-1.0） |
| support | float | 支持度（0.0-1.0） |
| stability | float | 稳定性（0.0-1.0） |
| quality_score | float | 综合质量评分（0.0-1.0） |
| sigma_r | float | 先验强度（0.01-1.0） |

---

## 🎯 在BSTGNF中使用

### 1. 加载先验配置

```python
# bstgnf/data/fdg_prior_scheduler.py
class FDGPriorScheduler:
    def __init__(self, edge_quality_file):
        self.edge_quality = pd.read_csv(edge_quality_file)
        self.edge_to_sigma = {}
        
        for _, row in self.edge_quality.iterrows():
            edge_key = (row['source'], row['target'], row['edge_type'])
            self.edge_to_sigma[edge_key] = row['sigma_r']
    
    def get_prior_sigma(self, source, target, edge_type):
        edge_key = (source, target, edge_type)
        return self.edge_to_sigma.get(edge_key, 0.5)  # 默认0.5
```

### 2. 在贝叶斯层中使用

```python
# bstgnf/models/bayesian_hetero_gat.py
class BayesianHeteroGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rel_types, fdg_prior_scheduler):
        super().__init__()
        self.fdg_prior = fdg_prior_scheduler
        
        # 为每种关系创建贝叶斯参数
        for rel_type in rel_types:
            # 获取该关系类型的平均先验强度
            sigma_r = self.fdg_prior.get_avg_sigma_for_type(rel_type)
            
            # 创建贝叶斯线性层，使用先验强度
            self.register_module(
                f'W_{rel_type}',
                BayesianLinear(in_dim, out_dim, prior_sigma=sigma_r)
            )
```

---

## 📚 参考资料

- **BSTGNF算法框架**: [`../bstgnf/BSTGNF算法框架.md`](../bstgnf/BSTGNF算法框架.md)
- **FDG设计**: [`5G-NIDD_Feature_Dependency_Graph_Design.md`](5G-NIDD_Feature_Dependency_Graph_Design.md)
- **实现指南**: [`../bstgnf/implementation_guide.md`](../bstgnf/implementation_guide.md)

---

**维护者**: RiskNICLP团队  
**最后更新**: 2025-10-29  
**状态**: 待实现

