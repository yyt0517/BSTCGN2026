# BSTGCN 训练脚本修改总结

## 修改概述

针对您提出的两个问题进行了修改：

1. **早停策略过于严格** → 增加patience从25到40
2. **结果被覆盖** → 使用时间戳创建独立目录

---

## 详细修改

### 1. 早停策略分析与调整

#### 问题分析
从您的训练日志看：
- Epoch 82-88: Train F1稳定在0.96左右
- Val F1在0.87-0.89波动
- 最佳Val F1在Epoch 63达到0.9004
- 在Epoch 88触发早停（patience=25）

**关键观察：**
- 贝叶斯模型的验证F1波动较大（MC采样的随机性）
- 您提到非贝叶斯版本需要100+轮才达到最佳Train F1
- 当前早停可能过早终止了模型的探索

#### 解决方案
```python
# 修改前
'patience': 25,

# 修改后
'patience': 40,  # 增加早停耐心，允许更长时间探索
```

**预期效果：**
- 允许模型训练更多轮次（最多到Epoch 150）
- 给验证F1更多波动空间
- 可能找到更好的模型（Val F1 > 0.9004）

---

### 2. 结果保存方式改进

#### 问题
当前使用固定目录 `baseline/results/bstgcn_50packets`，每次运行会覆盖之前的结果。

#### 解决方案

**代码修改：**
```python
# 1. 导入datetime模块
from datetime import datetime

# 2. 修改配置
config = {
    'save_dir_base': 'baseline/results/bstgcn_50packets',  # 基础目录
    # ... 其他配置
}

# 3. 创建带时间戳的目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = Path(config['save_dir_base']) / timestamp
save_dir.mkdir(parents=True, exist_ok=True)
config['save_dir'] = str(save_dir)  # 添加完整路径到配置中

# 4. 打印保存路径
print(f"\n结果将保存到: {save_dir}\n")
```

**目录结构示例：**
```
baseline/results/bstgcn_50packets/
├── 20251120_143025/          # 第一次运行
│   ├── best_model.pth
│   ├── config.json
│   ├── train_history.json
│   ├── test_results.json
│   ├── test_predictions.npy
│   ├── test_uncertainties.npy
│   ├── confusion_matrix.png
│   └── uncertainty_distribution.png
├── 20251120_150312/          # 第二次运行
│   └── ...
└── 20251120_163548/          # 第三次运行
    └── ...
```

---

## 使用指南

### 运行训练
```bash
conda activate gnn_py312
python scripts/BSTGCN/train_bstgcn.py
```

### 查看结果
训练开始时会打印：
```
结果将保存到: baseline\results\bstgcn_50packets\20251120_184200
```

所有结果文件都在这个时间戳目录中。

### 对比不同运行
```python
import json
from pathlib import Path

base_dir = Path('baseline/results/bstgcn_50packets')

# 列出所有运行
for run_dir in sorted(base_dir.iterdir()):
    if run_dir.is_dir():
        config_file = run_dir / 'test_results.json'
        if config_file.exists():
            with open(config_file) as f:
                results = json.load(f)
            print(f"{run_dir.name}: F1={results['test_macro_f1']:.4f}")
```

---

## 预期改进

### 早停策略（patience=40）
- ✅ 允许更长时间探索
- ✅ 减少因验证波动导致的过早停止
- ✅ 可能找到更好的模型

### 结果保存（时间戳目录）
- ✅ 不会覆盖之前的结果
- ✅ 方便对比不同运行
- ✅ 保留完整的实验历史

---

## 注意事项

1. **磁盘空间**：每次运行会占用约100-200MB空间（取决于模型大小）
2. **清理旧结果**：定期删除不需要的旧目录
3. **最佳模型选择**：可以通过对比不同运行的test_results.json来选择最佳模型

---

## 测试验证

已创建测试脚本 `scripts/BSTGCN/test_save_dir.py` 验证目录创建功能正常。

测试结果：
```
基础目录: baseline/results/bstgcn_50packets
时间戳: 20251120_184200
完整保存路径: baseline\results\bstgcn_50packets\20251120_184200
目录是否存在: False
创建目录后: True
```

✅ 功能正常！

