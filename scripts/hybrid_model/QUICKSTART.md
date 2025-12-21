# Hybrid Model 快速开始指南

## 🚀 一键运行

```bash
# 1. 激活环境
conda activate gnn_py312

# 2. 训练模型
python scripts/hybrid_model/02_train_hybrid.py

# 3. 查看对比结果
python scripts/hybrid_model/03_compare_results.py
```

## 📊 预期结果

训练完成后，你应该看到：

```
训练完成！
训练时间: ~1962秒 (~33分钟)
最佳Macro F1: 0.9242

结果已保存到: baseline\results\hybrid_model
  Accuracy: 0.9113
  Macro F1: 0.9171
  Weighted F1: 0.9115
```

对比结果：

```
🎉 Hybrid Model成功超越Transformer Baseline！
   Macro F1提升: +0.0202 (+2.25%)

困难类别分析:
  SYNScan: 0.7803 → 0.8237 (+0.0434, +5.56%) ✅ 改善
  TCPConnectScan: 0.8199 → 0.8376 (+0.0177, +2.16%) ✅ 改善
  UDPScan: 0.7933 → 0.8737 (+0.0804, +10.13%) ✅ 改善
```

## 📁 输出文件

训练完成后，以下文件会保存在 `baseline/results/hybrid_model/`:

- `hybrid_model.pth` - 训练好的模型权重
- `evaluation_report.txt` - 详细的评估报告
- `predictions.npy` - 测试集预测结果
- `pattern_weights.npy` - 攻击模式权重

## ⚙️ 配置说明

如果需要调整超参数，编辑 `scripts/hybrid_model/02_train_hybrid.py`:

```python
# 模型配置
model = HybridModel(
    input_dim=25,           # 输入特征数
    hidden_dim=128,         # 隐藏层维度
    num_heads=8,            # 注意力头数
    transformer_layers=3,   # Transformer层数
    gnn_layers=2,           # GNN层数
    num_classes=9,          # 类别数
    dropout=0.2             # Dropout率
)

# 训练配置
batch_size = 64             # 批大小
learning_rate = 0.0005      # 学习率
num_epochs = 100            # 最大训练轮数
patience = 20               # 早停耐心值
```

## 🔍 查看详细报告

```bash
# Windows
type baseline\results\hybrid_model\evaluation_report.txt

# Linux/Mac
cat baseline/results/hybrid_model/evaluation_report.txt
```

## 💡 常见问题

### Q1: 训练时间太长怎么办？

**A**: 可以减少模型大小或训练轮数：

```python
# 减小模型
hidden_dim=64  # 从128减到64
transformer_layers=2  # 从3减到2

# 减少训练轮数
num_epochs=50  # 从100减到50
```

### Q2: 显存不足怎么办？

**A**: 减小batch size：

```python
batch_size=32  # 从64减到32
# 或
batch_size=16  # 更小
```

### Q3: 如何使用训练好的模型进行预测？

**A**: 加载模型并预测：

```python
import torch
from scripts.hybrid_model.hybrid_model_01 import HybridModel

# 加载模型
model = HybridModel(input_dim=25, hidden_dim=128, ...)
model.load_state_dict(torch.load('baseline/results/hybrid_model/hybrid_model.pth'))
model.eval()

# 预测
with torch.no_grad():
    logits, pattern_weights = model(your_data)
    predictions = torch.argmax(logits, dim=1)
```

### Q4: 如何可视化模式权重？

**A**: 模式权重已保存在 `pattern_weights.npy`，可以用以下代码可视化：

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载模式权重
pattern_weights = np.load('baseline/results/hybrid_model/pattern_weights.npy')

# 计算每个类别的平均模式权重
class_names = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 
               'SYNScan', 'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

# 假设你有测试标签
test_labels = np.load('data/processed/raw_packet_sequences/test_labels.npy')

# 计算每个类别的平均权重
avg_weights = np.zeros((9, 9))
for i in range(9):
    mask = test_labels == i
    if mask.sum() > 0:
        avg_weights[i] = pattern_weights[mask].mean(axis=0)

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(avg_weights, annot=True, fmt='.3f', 
            xticklabels=class_names, yticklabels=class_names,
            cmap='YlOrRd')
plt.xlabel('Pattern')
plt.ylabel('True Class')
plt.title('Average Pattern Weights per Class')
plt.tight_layout()
plt.savefig('pattern_weights_heatmap.png', dpi=300)
plt.show()
```

## 🎯 下一步

1. **分析结果**: 查看 `evaluation_report.txt` 了解详细性能
2. **对比模型**: 运行 `03_compare_results.py` 对比不同模型
3. **调优**: 根据结果调整超参数
4. **部署**: 将训练好的模型部署到生产环境

## 📚 更多信息

- 详细文档: `scripts/hybrid_model/README.md`
- 模型代码: `scripts/hybrid_model/hybrid_model_01.py`
- 训练代码: `scripts/hybrid_model/02_train_hybrid.py`

---

**祝实验顺利！** 🎉

