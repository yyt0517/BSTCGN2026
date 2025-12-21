# 不确定性量化可视化指南

## 📊 核心可视化指标

### 1. 不确定性分布图

**显示内容**:
- 正确预测的不确定性分布
- 错误预测的不确定性分布
- 两个分布的重叠程度

**解读**:
- 分布分离好 → 不确定性量化有效
- 分布重叠多 → 不确定性量化效果差

**代码**:
```python
import matplotlib.pyplot as plt

correct_mask = (labels == predictions)
correct_unc = uncertainties[correct_mask]
error_unc = uncertainties[~correct_mask]

plt.hist(correct_unc, bins=50, alpha=0.5, label='Correct')
plt.hist(error_unc, bins=50, alpha=0.5, label='Error')
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.legend()
```

### 2. 不确定性与错误的散点图

**显示内容**:
- X轴: 不确定性值
- Y轴: 错误标记 (0=正确, 1=错误)
- 点的分布

**解读**:
- 向上倾斜 → 高不确定性对应高错误率
- 水平分布 → 不确定性与错误无关

**代码**:
```python
errors = (labels != predictions).astype(float)
plt.scatter(uncertainties, errors, alpha=0.3)
plt.xlabel('Uncertainty')
plt.ylabel('Error (0=Correct, 1=Error)')
```

### 3. 不确定性与准确率的关系

**显示内容**:
- 按不确定性分组
- 计算每组的准确率
- 绘制准确率曲线

**解读**:
- 曲线向下 → 高不确定性对应低准确率
- 曲线平坦 → 不确定性与准确率无关

**代码**:
```python
bins = np.percentile(uncertainties, np.linspace(0, 100, 11))
for i in range(len(bins)-1):
    mask = (uncertainties >= bins[i]) & (uncertainties < bins[i+1])
    acc = (labels[mask] == predictions[mask]).mean()
    plt.plot(bins[i], acc, 'o')
```

## 📈 三轮优化的不确定性变化

### 第一轮 (基线)
- 平均不确定性: 0.0184 (极低)
- 相关性: 0.2650 (中等)
- 问题: 模型未充分学习贝叶斯先验

### 第二轮 (早停优化)
- 平均不确定性: 0.4428 (高)
- 相关性: 0.1156 (低)
- 问题: 早停过早，贝叶斯学习不足

### 第三轮 (稳定熵优化)
- 平均不确定性: 0.3887 (中等)
- 相关性: 0.1777 (低但改善)
- 改进: +53.8%相关性提升

## 🎯 不确定性的实际应用

### 应用1: OOD检测

```python
# 设置阈值
ood_threshold = np.percentile(uncertainties, 95)

# 检测OOD样本
ood_mask = uncertainties > ood_threshold
print(f"检测到{ood_mask.sum()}个OOD样本")
```

### 应用2: 置信度排序

```python
# 按不确定性排序
sorted_idx = np.argsort(uncertainties)

# 高置信度预测 (低不确定性)
high_conf_idx = sorted_idx[:100]

# 低置信度预测 (高不确定性)
low_conf_idx = sorted_idx[-100:]
```

### 应用3: 可靠性评估

```python
# 分类
low_unc = uncertainties < np.percentile(uncertainties, 33)
mid_unc = (uncertainties >= np.percentile(uncertainties, 33)) & \
          (uncertainties < np.percentile(uncertainties, 67))
high_unc = uncertainties >= np.percentile(uncertainties, 67)

# 计算各类的准确率
print(f"低不确定性准确率: {(labels[low_unc] == predictions[low_unc]).mean():.4f}")
print(f"中不确定性准确率: {(labels[mid_unc] == predictions[mid_unc]).mean():.4f}")
print(f"高不确定性准确率: {(labels[high_unc] == predictions[high_unc]).mean():.4f}")
```

## 📊 关键指标解读

### 平均不确定性

| 值 | 含义 |
|------|------|
| <0.1 | 模型过于自信，可能未充分学习 |
| 0.1-0.5 | 正常范围，模型学到了适度的不确定性 |
| >0.5 | 模型过度不确定，可能需要调整 |

### 不确定性-错误相关性

| 值 | 含义 |
|------|------|
| <0.1 | 不确定性与错误无关，量化效果差 |
| 0.1-0.3 | 弱相关，量化效果一般 |
| 0.3-0.5 | 中等相关，量化效果良好 |
| >0.5 | 强相关，量化效果优秀 |

## 🔍 诊断问题

### 问题1: 不确定性过低

**症状**: 平均不确定性 <0.1

**原因**:
- 贝叶斯先验学习不足
- KL权重过小
- 训练轮数不足

**解决**:
- 增加训练轮数
- 提高KL权重
- 延长KL预热

### 问题2: 相关性低

**症状**: 不确定性-错误相关性 <0.2

**原因**:
- 互信息计算不敏感
- 数据集特性 (容易分类)
- 模型容量不足

**解决**:
- 多度量融合
- 增加困难样本
- 提升模型容量

## 💡 最佳实践

1. **定期监控**
   - 每个epoch记录平均不确定性
   - 每个epoch计算相关性

2. **可视化分析**
   - 绘制不确定性分布
   - 绘制相关性散点图

3. **应用验证**
   - 测试OOD检测效果
   - 验证置信度排序

---

**更新时间**: 2024-11-16  
**推荐指数**: ⭐⭐⭐⭐

