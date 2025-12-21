# BSTGCN 训练脚本 - AUROC 模型选择功能

## 修改概述

在 `train_bstgcn.py` 中添加了基于 AUROC 的模型选择功能，用于提升不确定性估计质量。

## 主要改动

### 1. 新增配置参数

在 `config` 字典中添加了两个新参数：

```python
'selection_metric': 'combined',  # 模型选择策略: 'f1', 'auroc', 'combined'
'auroc_weight': 0.3,            # combined模式下AUROC的权重
```

### 2. 验证集 AUROC 计算

每个 epoch 的验证阶段会自动计算：
- **验证集 AUROC**：使用不确定性作为分数，预测样本是否分类错误
- 计算公式：`is_error = (pred != true_label)`，然后 `AUROC(is_error, uncertainty)`

### 3. 三种模型选择策略

#### (1) `selection_metric = 'f1'`（原始行为）
- 只根据验证集 Macro F1 选择最佳模型
- 适合只关注分类性能的场景

#### (2) `selection_metric = 'auroc'`
- 只根据验证集 AUROC 选择最佳模型
- 适合优先优化不确定性质量的场景

#### (3) `selection_metric = 'combined'`（**推荐，默认**）
- 使用 F1 和 AUROC 的加权组合：
  ```python
  normalized_auroc = (auroc - 0.5) * 2.0  # 映射到 [0, 1]
  score = (1 - auroc_weight) * f1 + auroc_weight * normalized_auroc
  ```
- 默认权重：F1 占 70%，AUROC 占 30%
- **平衡分类性能和不确定性质量**

### 4. 训练日志增强

每个 epoch 的日志现在包含：
```
Epoch [  1/150] | Time: 45.2s | Loss: 1.2345 (Rec: 1.2000, KL: 0.0345) | 
Train F1: 0.8500 | Val F1: 0.8600 | Val AUROC: 0.7200 | Val Unc: 0.4500 | 
KL_w: 0.0001 | LR: 0.000500
```

### 5. 结果保存增强

- `train_history.json` 中新增 `val_auroc` 字段
- `best_model.pth` checkpoint 中保存 `val_auroc`
- `test_results.json` 中新增：
  - `best_val_auroc`: 最佳模型的验证集 AUROC
  - `selection_metric`: 使用的选择策略

## 使用方法

### 默认使用（推荐）

直接运行，会使用 combined 模式（F1 70% + AUROC 30%）：

```bash
python scripts/BSTGCN/train_bstgcn.py
```

### 只用 AUROC 选模型

修改 `config` 中的 `selection_metric`：

```python
config = {
    ...
    'selection_metric': 'auroc',  # 改为 'auroc'
    ...
}
```

### 调整 AUROC 权重

如果想让 AUROC 占更大比重（比如 50%）：

```python
config = {
    ...
    'selection_metric': 'combined',
    'auroc_weight': 0.5,  # F1 和 AUROC 各占 50%
    ...
}
```

## 预期效果

1. **测试集 AUROC 提升**：通过选择验证集 AUROC 更高的 epoch，测试集 AUROC 通常也会提升
2. **不确定性分布分离度更好**：正确样本和错误样本的不确定性分布差异更大
3. **Coverage-Accuracy 曲线更优**：在相同覆盖率下，精度更高

## 验证效果

训练完成后，使用以下命令验证 AUROC 和 coverage-accuracy：

```bash
python scripts/BSTGCN/compute_uncertainty_metrics.py
```

查看：
- AUROC 是否从 ~0.77 提升到 ~0.78-0.80+
- Coverage-Accuracy 表格中，各覆盖率下的精度是否提升

## 注意事项

1. **AUROC 计算条件**：只有当验证集既有正确样本又有错误样本时才计算 AUROC
2. **早停机制**：patience 计数器基于选择的 metric，如果连续 40 个 epoch 没有改善会触发早停
3. **兼容性**：如果不设置新参数，默认行为与原版一致（combined 模式，AUROC 权重 0.3）

## 后续优化方向

如果 AUROC 提升仍不够明显，可以尝试：
1. 调整 `auroc_weight` 到 0.4-0.5（更重视不确定性）
2. 配合超参数调整（见主文档建议的 prior_std, kl_weight_max, dropout 等）
3. 增加 `num_mc_samples_train` 到 12-16，使验证集 AUROC 估计更稳定

