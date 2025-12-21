# BSTGCN 快速参考

## 🚀 一键运行

### 训练 BSTGCN
```bash
python scripts/BSTGCN/train_bstgcn.py
```

### 运行对比
```bash
# 只对比BSTGCN
python baseline/run_comprehensive_baselines.py --models bstgcn

# 对比所有时空模型
python baseline/compare_spatiotemporal_models.py

# 对比所有11个模型
python baseline/run_comprehensive_baselines.py --models all
```

---

## 📊 最新结果 (Staged策略)

| 指标 | 值 |
|------|-----|
| **Test Accuracy** | 0.9686 |
| **Test Macro F1** | 0.9273 |
| **Test Weighted F1** | 0.9687 |
| **Mean Uncertainty** | 0.5240 |
| **Uncertainty-Error Correlation** | 0.2174 |

---

## 🎯 核心特性

### 1. 不确定性量化
- **Correct预测**: 中位数不确定性 = 0.50
- **Error预测**: 中位数不确定性 = 0.80
- **区分度**: 两个箱体几乎不重叠

### 2. 推荐阈值策略
- **保守 (0.70)**: 拒绝75%错误，保留90%正确
- **平衡 (0.80)**: 拒绝50%错误，保留95%正确 ⭐
- **激进 (0.90)**: 拒绝25%错误，保留98%正确

---

## 📁 重要文件

### 配置文件
```
configs/bstgcn_50packets.yaml
```

### 训练脚本
```
scripts/BSTGCN/train_bstgcn.py
```

### 结果目录
```
baseline/results/bstgcn_50packets/YYYYMMDD_HHMMSS/
├── test_results.json          # 测试结果
├── best_model.pth            # 最佳模型
├── confusion_matrix.png      # 混淆矩阵
└── uncertainty_distribution.png  # 不确定性分布
```

---

## 🔧 模型选择策略

### 当前使用: Staged (推荐)
```python
config = {
    'selection_metric': 'staged',
    'stage_epoch': 30,
    'auroc_weight': 0.3
}
```

**效果**:
- 前30轮: 只看F1 (保证高F1)
- 30轮后: 看F1+AUROC (改善不确定性)

### 其他策略
```bash
# F1-only
python scripts/BSTGCN/apply_strategy.py f1

# AUROC-only
python scripts/BSTGCN/apply_strategy.py auroc

# Combined
python scripts/BSTGCN/apply_strategy.py combined --auroc_weight 0.3

# Balanced
python scripts/BSTGCN/apply_strategy.py balanced --kl_threshold 0.0001
```

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| `BASELINE_INTEGRATION.md` | 集成使用说明 |
| `boxplot_interpretation.md` | 箱型图详细解释 |
| `model_selection_strategies.md` | 模型选择策略对比 |
| `STAGED_STRATEGY_SUCCESS.md` | Staged策略成功报告 |
| `INTEGRATION_SUMMARY.md` | 集成修改总结 |
| `QUICK_REFERENCE.md` | 本文档 |

---

## 🎓 关键概念

### 箱型图组成
- **箱底 (Q1)**: 25%分位数
- **橙色线 (Q2)**: 中位数
- **箱顶 (Q3)**: 75%分位数
- **须**: 延伸到非离群值
- **点**: 离群值

### ELBO损失
```
ELBO = Reconstruction Loss + β × KL Divergence
```
- β从0逐渐增加到0.0002 (60轮warmup)

### MC推理
- 采样次数: 30
- 不确定性 = 预测分布的熵

---

## ⚡ 常见问题

### Q1: 如何修改模型选择策略?
```bash
python scripts/BSTGCN/apply_strategy.py staged --stage_epoch 30
```

### Q2: 如何查看训练历史?
```python
import json
with open('baseline/results/bstgcn_50packets/YYYYMMDD_HHMMSS/train_history.json') as f:
    history = json.load(f)
```

### Q3: 如何使用不确定性进行样本拒绝?
```python
import numpy as np

# 加载不确定性
uncertainties = np.load('baseline/results/bstgcn_50packets/YYYYMMDD_HHMMSS/test_uncertainties.npy')

# 设置阈值
threshold = 0.80

# 拒绝高不确定性样本
reject_mask = uncertainties > threshold
```

### Q4: 如何对比不同策略的结果?
```bash
python scripts/BSTGCN/compare_results.py
```

---

## 🎉 快速成功检查

运行以下命令验证一切正常:

```bash
# 1. 测试集成
python baseline/test_bstgcn_integration.py

# 2. 运行BSTGCN对比
python baseline/run_comprehensive_baselines.py --models bstgcn

# 3. 查看结果
cat baseline/results_v2/summary.json
```

如果所有命令都成功运行，说明集成完成！🎉

