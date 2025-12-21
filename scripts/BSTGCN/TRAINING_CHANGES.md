# BSTGCN 训练脚本修改说明

## 修改日期
2025-11-20

## 修改内容

### 1. 早停策略调整

**修改前：**
- `patience = 25`

**修改后：**
- `patience = 40`

**原因：**
- 贝叶斯模型的训练过程中，验证集F1会有较大波动（由于MC采样的随机性）
- 从训练日志看，模型在100+轮后Train F1才达到最佳，需要更长的探索时间
- 增加patience可以避免过早停止，给模型更多机会找到更好的解

### 2. 结果保存方式改进

**修改前：**
```python
'save_dir': 'baseline/results/bstgcn_50packets'
```
每次运行会覆盖之前的结果。

**修改后：**
```python
'save_dir_base': 'baseline/results/bstgcn_50packets'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = Path(config['save_dir_base']) / timestamp
```

**效果：**
- 每次运行会创建一个带时间戳的子目录
- 例如：`baseline/results/bstgcn_50packets/20251120_143025/`
- 所有结果（模型、配置、可视化等）都保存在独立目录中
- 方便对比不同运行的结果

### 3. 目录结构示例

```
baseline/results/bstgcn_50packets/
├── 20251120_143025/          # 第一次运行
│   ├── best_model.pth
│   ├── config.json
│   ├── train_history.json
│   ├── test_results.json
│   ├── confusion_matrix.png
│   └── uncertainty_distribution.png
├── 20251120_150312/          # 第二次运行
│   ├── best_model.pth
│   ├── config.json
│   └── ...
└── 20251120_163548/          # 第三次运行
    └── ...
```

## 使用建议

1. **对比不同运行结果**：可以通过时间戳快速找到特定运行的结果
2. **保留历史记录**：不会丢失之前的实验结果
3. **清理旧结果**：定期删除不需要的旧目录以节省空间

## 其他说明

- 时间戳格式：`YYYYMMDD_HHMMSS`（年月日_时分秒）
- 配置文件中会同时保存 `save_dir_base` 和 `save_dir` 两个字段
- 训练开始时会打印完整的保存路径

