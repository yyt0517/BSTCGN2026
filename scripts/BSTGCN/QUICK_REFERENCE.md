# 快速参考卡

---

## 🆕 最新修改 (2025-11-20)

### 1️⃣ 早停策略调整
```python
'patience': 40  # 从25增加到40
```
**原因：** 贝叶斯模型需要更长时间探索，验证F1波动较大

### 2️⃣ 结果保存改进
```python
'save_dir_base': 'baseline/results/bstgcn_50packets'
# 自动创建时间戳子目录，例如：
# baseline/results/bstgcn_50packets/20251120_184200/
```
**原因：** 避免覆盖之前的结果，保留实验历史

**详细说明：** 参见 `TRAINING_CHANGES.md` 和 `MODIFICATION_SUMMARY.md`

---

## 当前性能 (第3轮优化)

```
准确率:        96.67%  ✓
Macro F1:      91.90%  ✓
训练时间:      ~88分钟
验证时间/ep:   ~12秒   (优化后)
不确定性相关:  0.1777  ↑ (+53.8%)
```

## 运行优化版本

```bash
conda activate gnn_py312
python scripts/BSTGCN/train_bstgcn.py
```

## 关键改进

### 1. 自适应MC采样 ✓
- 初始: 3次采样
- 最多: 8次采样
- 效果: -60%验证时间

### 2. 稳定熵计算 ✓
- 使用: torch.special.entr()
- 效果: +12%不确定性相关性

### 3. 配置参数 ✓
```python
'use_adaptive_mc': True        # 启用自适应采样
'use_stable_entropy': True     # 启用稳定熵
```

## 性能冲刺 (96.87% → 97%+)

### 最有效的优化 (按优先级)

1. **类别权重调整** (+0.3%)
   ```python
   'scan_class_boost': 1.3 → 1.5
   ```

2. **学习率微调** (+0.2%)
   ```python
   'learning_rate': 0.0005 → 0.0006
   ```

3. **多度量融合** (+0.2%)
   - 互信息 + 变分比率 + 预测熵

4. **数据增强** (+0.2%)
   - Mixup + 特征缩放

## 不确定性改进

### 当前问题
- 相关性: 0.1156 (低)
- 原因: 互信息计算不够敏感

### 改进方案
1. 多度量融合 (预期: +10-20%)
2. 校准方法 (预期: +5-10%)
3. 自适应阈值 (预期: +5%)

## 文件位置

| 文件 | 功能 |
|------|------|
| train_bstgcn.py | 主训练脚本 |
| mc_inference.py | MC推理 (已优化) |
| bstgcn_model.py | 模型架构 |
| elbo_loss.py | 损失函数 |

## 优化文档

| 文档 | 内容 |
|------|------|
| PERFORMANCE_OPTIMIZATION_ANALYSIS.md | 详细分析 |
| OPTIMIZATION_IMPLEMENTATION.md | 实现指南 |
| PERFORMANCE_SPRINT_GUIDE.md | 冲刺计划 |
| TECHNICAL_SUMMARY.md | 技术总结 |

## 监控指标

每次运行记录:
- [ ] 总训练时间
- [ ] 平均验证时间
- [ ] 最终准确率
- [ ] 最终Macro F1
- [ ] 不确定性相关性
- [ ] 平均MC采样数

## 下一步行动

### 立即 (今天)
- [ ] 运行优化版本
- [ ] 验证性能
- [ ] 记录基准

### 本周
- [ ] 实现类别权重优化
- [ ] 运行对比实验
- [ ] 选择最佳配置

### 本月
- [ ] 实现多度量融合
- [ ] 实现模型蒸馏
- [ ] 性能基准测试

## 常见问题

**Q: 优化会影响性能吗?**
A: 不会。自适应采样和稳定熵计算都是无损优化。

**Q: 如何禁用自适应采样?**
A: 在config中设置 `'use_adaptive_mc': False`

**Q: 如何调整自适应采样参数?**
A: 修改evaluate_bayesian函数中的参数:
- min_samples: 初始采样数
- max_samples: 最大采样数
- confidence_threshold: 置信度阈值
- uncertainty_threshold: 不确定性阈值

**Q: 训练时间还是很长?**
A: 可以尝试:
1. 减少验证频率
2. 减少MC采样数
3. 使用更小的batch size

## 性能目标

| 指标 | 当前 | 目标 | 优先级 |
|------|------|------|--------|
| 准确率 | 96.87% | 97%+ | 高 |
| Macro F1 | 92.30% | 93%+ | 高 |
| 训练时间 | 91分钟 | 60分钟 | 中 |
| 不确定性相关 | 0.1156 | 0.2+ | 中 |

## 联系方式

有问题? 查看:
- PERFORMANCE_OPTIMIZATION_ANALYSIS.md
- TECHNICAL_SUMMARY.md
- 代码注释

