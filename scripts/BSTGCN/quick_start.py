"""
BSTGCN快速启动脚本

用于快速测试BSTGCN模型的基本功能
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.BSTGCN import (
    BSTGCNModel,
    ELBOLoss,
    KLWeightScheduler,
    mc_inference
)


def quick_demo():
    """快速演示BSTGCN的核心功能"""
    
    print("=" * 80)
    print("BSTGCN快速演示")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n1. 创建BSTGCN模型...")
    model = BSTGCNModel(
        input_dim=10,
        hidden_dim=128,
        num_heads=8,
        num_st_blocks=3,
        num_classes=10,
        dropout=0.15,
        k_neighbors=10,
        sage_aggr='mean',
        prior_std=1.0
    ).to(device)
    
    print(f"   模型参数数量: {model.get_num_params():,}")
    
    # 创建模拟数据
    print("\n2. 创建模拟数据...")
    batch_size = 32
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 10).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    print(f"   输入形状: {x.shape}")
    print(f"   标签形状: {labels.shape}")
    
    # 前向传播 (训练模式)
    print("\n3. 前向传播 (训练模式 - 单次采样)...")
    model.train()
    logits = model(x, sample=True)
    print(f"   输出形状: {logits.shape}")
    print(f"   输出范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    # 计算KL散度
    kl = model.kl_divergence()
    print(f"   KL散度: {kl.item():.2f}")
    
    # ELBO损失
    print("\n4. 计算ELBO损失...")
    elbo_loss = ELBOLoss(
        model=model,
        kl_weight=0.1,
        label_smoothing=0.05,
        num_batches=100
    )
    
    total_loss, rec_loss, kl_loss = elbo_loss(logits, labels)
    print(f"   总损失: {total_loss.item():.4f}")
    print(f"   重建损失: {rec_loss.item():.4f}")
    print(f"   KL散度: {kl_loss.item():.4f}")
    
    # MC推理
    print("\n5. 蒙特卡洛推理 (推理模式 - 30次采样)...")
    model.eval()
    pred_mean, pred_class, uncertainty = mc_inference(model, x, num_samples=30)
    
    print(f"   预测概率形状: {pred_mean.shape}")
    print(f"   预测类别形状: {pred_class.shape}")
    print(f"   不确定性形状: {uncertainty.shape}")
    print(f"   平均不确定性: {uncertainty.mean().item():.4f}")
    print(f"   不确定性范围: [{uncertainty.min().item():.4f}, {uncertainty.max().item():.4f}]")
    
    # 分析不确定性
    print("\n6. 不确定性分析...")
    high_uncertainty_idx = uncertainty > uncertainty.median()
    low_uncertainty_idx = uncertainty <= uncertainty.median()
    
    print(f"   高不确定性样本数: {high_uncertainty_idx.sum().item()}")
    print(f"   低不确定性样本数: {low_uncertainty_idx.sum().item()}")
    print(f"   高不确定性平均值: {uncertainty[high_uncertainty_idx].mean().item():.4f}")
    print(f"   低不确定性平均值: {uncertainty[low_uncertainty_idx].mean().item():.4f}")
    
    # KL权重调度
    print("\n7. KL权重调度演示...")
    kl_scheduler = KLWeightScheduler(kl_weight_max=1.0, warmup_epochs=10, mode='linear')
    
    print("   Epoch | KL Weight")
    print("   ------|----------")
    for epoch in range(15):
        weight = kl_scheduler.get_kl_weight()
        if epoch % 3 == 0:
            print(f"   {epoch:5d} | {weight:.4f}")
        kl_scheduler.step()
    
    # 对比采样vs MAP
    print("\n8. 采样模式 vs MAP模式对比...")
    model.eval()
    
    # 采样模式
    logits_sample1 = model(x[:5], sample=True)
    logits_sample2 = model(x[:5], sample=True)
    sample_diff = (logits_sample1 - logits_sample2).abs().mean().item()
    
    # MAP模式
    logits_map1 = model(x[:5], sample=False)
    logits_map2 = model(x[:5], sample=False)
    map_diff = (logits_map1 - logits_map2).abs().mean().item()
    
    print(f"   采样模式差异: {sample_diff:.6f} (应该>0)")
    print(f"   MAP模式差异: {map_diff:.6f} (应该≈0)")
    
    print("\n" + "=" * 80)
    print("✓ 演示完成!")
    print("=" * 80)
    
    # 总结
    print("\n📊 关键特性总结:")
    print("  ✅ 贝叶斯权重分布 (μ, σ)")
    print("  ✅ ELBO损失 (重建 + KL)")
    print("  ✅ MC推理 (30次采样)")
    print("  ✅ 不确定性量化")
    print("  ✅ KL权重退火")
    print("  ✅ 采样 vs MAP模式")
    
    print("\n🚀 下一步:")
    print("  1. 运行完整训练: python scripts/BSTGCN/train_bstgcn.py")
    print("  2. 查看文档: scripts/BSTGCN/README.md")
    print("  3. 调整超参数: 编辑train_bstgcn.py中的config")


if __name__ == '__main__':
    quick_demo()

