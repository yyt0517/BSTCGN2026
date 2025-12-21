# 5G NIDD 时间序列数据分析 - 完整文档

## 📁 生成的文件清单

### 📊 分析报告
1. **`5G_NIDD_TIME_SERIES_Analysis_Report.md`** - 完整的数据分析报告
   - 数据集概况（多分类和二分类）
   - 特征分析
   - 类别分布
   - 机器学习建议
   - 评估指标建议

2. **`ATTACK_TYPE_SUMMARY.md`** - 攻击类型详细总结（⭐ 推荐阅读）
   - 每种攻击类型的详细说明
   - 攻击特征对比
   - 类别不平衡分析
   - 机器学习实施路线图
   - 预期模型性能

3. **`README_ANALYSIS.md`** - 本文件，文档导航

### 📈 可视化图表
4. **`time_series_analysis.png`** - 基础数据分析可视化
   - 攻击类型分布（多分类）
   - 标签分布（二分类）
   - 流量计数时间序列
   - 关键特征分布对比

5. **`attack_type_detailed_analysis.png`** - 详细攻击类型分析（⭐ 推荐查看）
   - 攻击类型分布（带颜色编码）
   - 攻击类别饼图
   - 8个关键特征的箱线图对比
   - 攻击发生时间序列
   - 类别不平衡比例可视化

6. **`attack_type_mapping_table.png`** - 攻击类型映射表（⭐ 快速参考）
   - 标签到攻击类型的映射
   - 样本数和百分比
   - 不平衡比例
   - 关键特征统计

### 📄 数据文件
7. **`attack_type_mapping.csv`** - 攻击类型映射CSV
   - 便于程序读取的映射表
   - 包含标签、攻击类型、类别、样本数、百分比、不平衡比例

### 🐍 Python脚本
8. **`analyze_timeseries.py`** - 基础分析脚本
   - 数据加载和统计
   - 基础可视化
   - 可重复运行

9. **`attack_type_detailed_analysis.py`** - 详细分析脚本
   - 每种攻击类型的详细统计
   - 特征对比分析
   - 高级可视化

10. **`create_attack_mapping_table.py`** - 映射表生成脚本
    - 创建攻击类型映射表
    - 生成CSV文件

---

## 🎯 快速开始指南

### 1. 查看数据概况
**推荐阅读顺序**：
1. 先看 `attack_type_mapping_table.png` - 快速了解攻击类型映射
2. 再看 `ATTACK_TYPE_SUMMARY.md` - 详细了解每种攻击
3. 最后看 `5G_NIDD_TIME_SERIES_Analysis_Report.md` - 完整分析报告

### 2. 查看可视化
**推荐查看顺序**：
1. `attack_type_mapping_table.png` - 映射表和统计
2. `attack_type_detailed_analysis.png` - 详细特征对比
3. `time_series_analysis.png` - 时间序列分析

### 3. 使用数据
```python
import pandas as pd

# 加载攻击类型映射
mapping = pd.read_csv('attack_type_mapping.csv')
print(mapping)

# 加载数据集
df_multiclass = pd.read_csv('5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_attacktype.csv')
df_binary = pd.read_csv('5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_final.csv')

# 创建攻击类型字典
attack_labels = dict(zip(mapping['Label'], mapping['Attack Type']))
print(attack_labels)
```

---

## 📊 数据集核心信息

### 攻击类型映射（正确版本）

| 标签 | 攻击类型 | 类别 | 样本数 | 占比 | 不平衡比例 |
|------|---------|------|--------|------|-----------|
| 0 | Benign | Normal | 2,563 | 71.17% | Baseline |
| 1 | HTTPFlood | Flood Attack | 269 | 7.47% | 1:9.53 |
| 2 | ICMPFlood | Flood Attack | 0 | 0.00% | N/A |
| 3 | SYNFlood | Flood Attack | 3 | 0.08% | 1:854.33 ⚠️ |
| 4 | SYNScan | Scan Attack | 38 | 1.06% | 1:67.45 |
| 5 | SlowrateDoS | Flood Attack | 601 | 16.69% | 1:4.26 |
| 6 | UDPLagScan | Scan Attack | 41 | 1.14% | 1:62.51 |
| 7 | UDPFlood | Flood Attack | 25 | 0.69% | 1:102.52 |
| 8 | UDPScan | Scan Attack | 61 | 1.69% | 1:42.02 |

### 关键统计

**数据集规模**：
- 总样本数: 3,601
- 时间跨度: 1小时（3,600秒）
- 采样频率: 1秒/样本
- 特征数: 86（83个网络流量特征 + 3个元数据）

**类别分布**：
- 正常流量: 2,563样本 (71.17%)
- 洪水攻击: 898样本 (24.94%)
  - SlowrateDoS: 601 (最常见)
  - HTTPFlood: 269
  - UDPFlood: 25
  - SYNFlood: 3 (极稀少)
- 扫描攻击: 140样本 (3.89%)
  - UDPScan: 61
  - UDPLagScan: 41
  - SYNScan: 38

**数据质量**：
- ✅ 无缺失值
- ✅ 时间连续性完整
- ⚠️ 严重的类别不平衡
- ⚠️ ICMPFlood类别完全缺失
- ⚠️ SYNFlood仅3个样本

---

## 🔍 关键发现

### 1. 攻击特征模式

**洪水攻击（Flood Attacks）的特征**：
- ✓ TCP RTT显著更高（网络延迟）
- ✓ 持续时间更长
- ✓ 丢包率更低
- ✓ 数据包和字节数较少

**扫描攻击（Scan Attacks）的特征**：
- ✓ 持续时间极短（0.5-1.2秒）
- ✓ 丢包率极低
- ✓ TCP RTT较低
- ✓ 快速、短暂的连接

### 2. 最具区分性的特征

根据分析，以下特征对攻击检测最有价值：

1. **持续时间 (Dur)**: 
   - 正常: 2.51秒
   - 扫描攻击: 0.5-1.2秒 (↓50-80%)
   - 洪水攻击: 2.9-3.5秒 (↑15-40%)

2. **TCP RTT (TcpRtt)**:
   - 正常: 0.0027秒
   - HTTPFlood: 0.0251秒 (↑829%)
   - SlowrateDoS: 0.0131秒 (↑385%)

3. **数据包数 (TotPkts)**:
   - 正常: 71.72
   - UDPFlood: 2.54 (↓96.5%)
   - 其他攻击: 14-39 (↓45-80%)

4. **字节数 (TotBytes)**:
   - 正常: 84,989
   - UDPFlood: 473 (↓99.4%)
   - 其他攻击: 13,000-46,000 (↓45-85%)

5. **丢包率 (Loss)**:
   - 正常: 0.198
   - 攻击: 0.001-0.181 (↓0.5-99.9%)

### 3. 类别不平衡挑战

**严重不平衡的类别**：
- SYNFlood: 1:854.33 (仅3个样本) ⚠️⚠️⚠️
- UDPFlood: 1:102.52 (25个样本) ⚠️⚠️
- SYNScan: 1:67.45 (38个样本) ⚠️
- UDPLagScan: 1:62.51 (41个样本) ⚠️

**建议**：
- 对SYNFlood: 使用SMOTE生成至少100个样本
- 对其他稀有类别: 考虑合并或使用ADASYN
- 使用类别权重或Focal Loss

---

## 💡 机器学习建议

### 数据预处理

```python
# 1. 标准化特征
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 处理类别不平衡
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 3. 特征选择
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X_resampled, y_resampled)
```

### 推荐模型

**1. XGBoost（推荐）**：
```python
import xgboost as xgb

# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_weights,
    random_state=42
)
```

**2. Random Forest**：
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

**3. LSTM（时间序列）**：
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
```

### 评估指标

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 预测
y_pred = model.predict(X_test)

# 评估
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=attack_labels.values()))

# Macro F1-Score（平等对待所有类别）
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1-Score: {macro_f1:.4f}")

# Weighted F1-Score（考虑类别分布）
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Weighted F1-Score: {weighted_f1:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

---

## 🎯 实施路线图

### Phase 1: 基线模型（1-2天）
- [ ] 数据加载和探索
- [ ] 特征标准化
- [ ] 二分类模型（Benign vs Attack）
- [ ] XGBoost基线模型
- [ ] 性能评估

### Phase 2: 多分类模型（3-5天）
- [ ] 类别合并策略
- [ ] SMOTE过采样
- [ ] 多分类XGBoost/Random Forest
- [ ] 超参数调优
- [ ] 特征重要性分析

### Phase 3: 高级模型（5-7天）
- [ ] LSTM时间序列模型
- [ ] 图神经网络（GAT）
- [ ] 集成学习
- [ ] 模型融合

### Phase 4: 优化与部署（2-3天）
- [ ] 模型压缩
- [ ] 推理优化
- [ ] 实时检测系统
- [ ] 性能监控

---

## 📚 参考资料

### 攻击类型说明

1. **HTTPFlood**: HTTP洪水攻击，通过大量HTTP请求淹没服务器
2. **ICMPFlood**: ICMP洪水攻击（本数据集中不存在）
3. **SYNFlood**: TCP SYN洪水攻击，耗尽服务器资源
4. **SYNScan**: TCP SYN端口扫描，侦察开放端口
5. **SlowrateDoS**: 慢速DoS攻击，低速率攻击逃避检测
6. **UDPLagScan**: UDP延迟扫描，网络侦察
7. **UDPFlood**: UDP洪水攻击，大量UDP数据包淹没目标
8. **UDPScan**: UDP端口扫描，侦察活动

### 相关论文和资源

- 5G网络安全
- 入侵检测系统（IDS）
- 时间序列异常检测
- 图神经网络在网络安全中的应用
- 不平衡数据分类

---

## 🔧 重新运行分析

如果需要重新生成分析结果：

```bash
# 基础分析
python analyze_timeseries.py

# 详细攻击类型分析
python attack_type_detailed_analysis.py

# 生成映射表
python create_attack_mapping_table.py
```

---

## 📞 联系信息

如有问题或需要进一步分析，请参考：
- 完整报告: `5G_NIDD_TIME_SERIES_Analysis_Report.md`
- 攻击类型总结: `ATTACK_TYPE_SUMMARY.md`
- 可视化图表: `*.png` 文件

---

*文档创建时间: 2025-11-10*
*数据集: 5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_*.csv*
*分析工具: Python 3.12, pandas, matplotlib, seaborn*

