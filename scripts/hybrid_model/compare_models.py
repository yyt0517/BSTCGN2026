"""
对比原始 Hybrid 和改进版 Hybrid 模型

快速对比两个模型的：
1. 参数数量
2. 前向传播时间
3. 内存占用
4. 架构差异
"""

import torch
import time
import numpy as np
from hybrid_model_01 import HybridModel
from hybrid_model_improved import ImprovedHybridModel


def count_parameters(model):
    """统计参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_forward_time(model, x, num_runs=100):
    """测量前向传播时间"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(x)
            end = time.time()
            times.append(end - start)
    
    return np.mean(times), np.std(times)


def measure_memory(model, x):
    """测量内存占用（需要 CUDA）"""
    if not torch.cuda.is_available():
        return None, None
    
    model = model.cuda()
    x = x.cuda()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Forward
    with torch.no_grad():
        _ = model(x)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    model = model.cpu()
    x = x.cpu()
    torch.cuda.empty_cache()
    
    return peak_memory


def print_model_structure(model, name):
    """打印模型结构"""
    print(f"\n{'='*80}")
    print(f"{name} 架构")
    print(f"{'='*80}")
    
    for name, module in model.named_children():
        print(f"{name:30s} {module.__class__.__name__}")
        
        # Print sub-modules for key components
        if hasattr(module, 'named_children'):
            for sub_name, sub_module in module.named_children():
                if sub_name in ['st_blocks', 'gnn_layers', 'stgcn_blocks']:
                    print(f"  ├─ {sub_name:28s} {len(list(sub_module))} layers")


def main():
    print("="*80)
    print("Hybrid 模型对比分析")
    print("="*80)
    
    # 配置
    batch_size = 32
    seq_len = 100
    input_dim = 25
    num_classes = 9
    hidden_dim = 128
    num_heads = 8
    
    print(f"\n测试配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num heads: {num_heads}")
    
    # 创建模型
    print(f"\n{'='*80}")
    print("创建模型...")
    print(f"{'='*80}")
    
    original_model = HybridModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        transformer_layers=3,
        gnn_layers=2,
        num_classes=num_classes,
        dropout=0.1
    )
    
    improved_model = ImprovedHybridModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_st_blocks=3,
        num_classes=num_classes,
        dropout=0.1,
        k_neighbors=10
    )
    
    print("✓ 原始 Hybrid 模型")
    print("✓ 改进版 Hybrid 模型")
    
    # 打印架构
    print_model_structure(original_model, "原始 Hybrid 模型")
    print_model_structure(improved_model, "改进版 Hybrid 模型")
    
    # 参数数量
    print(f"\n{'='*80}")
    print("参数数量对比")
    print(f"{'='*80}")
    
    orig_total, orig_trainable = count_parameters(original_model)
    impr_total, impr_trainable = count_parameters(improved_model)
    
    print(f"{'模型':<20s} {'总参数':>15s} {'可训练参数':>15s}")
    print(f"{'-'*80}")
    print(f"{'原始 Hybrid':<20s} {orig_total:>15,} {orig_trainable:>15,}")
    print(f"{'改进版 Hybrid':<20s} {impr_total:>15,} {impr_trainable:>15,}")
    print(f"{'-'*80}")
    
    diff = impr_total - orig_total
    diff_pct = (diff / orig_total) * 100
    print(f"{'差异':<20s} {diff:>15,} ({diff_pct:+.2f}%)")
    
    # 前向传播时间
    print(f"\n{'='*80}")
    print("前向传播时间对比 (CPU)")
    print(f"{'='*80}")
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("测量原始 Hybrid 模型...")
    orig_mean, orig_std = measure_forward_time(original_model, x, num_runs=50)
    
    print("测量改进版 Hybrid 模型...")
    impr_mean, impr_std = measure_forward_time(improved_model, x, num_runs=50)
    
    print(f"\n{'模型':<20s} {'平均时间 (ms)':>20s} {'标准差 (ms)':>20s}")
    print(f"{'-'*80}")
    print(f"{'原始 Hybrid':<20s} {orig_mean*1000:>20.2f} {orig_std*1000:>20.2f}")
    print(f"{'改进版 Hybrid':<20s} {impr_mean*1000:>20.2f} {impr_std*1000:>20.2f}")
    print(f"{'-'*80}")
    
    speedup = orig_mean / impr_mean
    print(f"{'加速比':<20s} {speedup:>20.2f}x")
    
    # 内存占用（如果有 GPU）
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print("内存占用对比 (GPU)")
        print(f"{'='*80}")
        
        print("测量原始 Hybrid 模型...")
        orig_memory = measure_memory(original_model, x)
        
        print("测量改进版 Hybrid 模型...")
        impr_memory = measure_memory(improved_model, x)
        
        print(f"\n{'模型':<20s} {'峰值内存 (MB)':>20s}")
        print(f"{'-'*80}")
        print(f"{'原始 Hybrid':<20s} {orig_memory:>20.2f}")
        print(f"{'改进版 Hybrid':<20s} {impr_memory:>20.2f}")
        print(f"{'-'*80}")
        
        mem_diff = impr_memory - orig_memory
        mem_diff_pct = (mem_diff / orig_memory) * 100
        print(f"{'差异':<20s} {mem_diff:>20.2f} ({mem_diff_pct:+.2f}%)")
    
    # 关键改进总结
    print(f"\n{'='*80}")
    print("关键改进总结")
    print(f"{'='*80}")
    
    improvements = [
        "1. 并行时空交互（ST-Block）",
        "   - 原始：Transformer → GNN（串行）",
        "   - 改进：TCN ‖ GATv2（并行）+ 门控融合",
        "",
        "2. 修正的 GATv2 实现",
        "   - 原始：简化版，提前 mean 多头",
        "   - 改进：真正的 per-head Q/K/V + concat",
        "",
        "3. 图构建策略",
        "   - 原始：每次前向重建 KNN 图",
        "   - 改进：缓存邻接矩阵，每 epoch 更新一次",
        "",
        "4. 读出机制",
        "   - 原始：简单均值池化",
        "   - 改进：门控注意力池化",
        "",
        "5. 训练优化",
        "   - 原始：CrossEntropy + Plateau LR",
        "   - 改进：Label Smoothing + Cosine LR with Warmup",
    ]
    
    for line in improvements:
        print(line)
    
    print(f"\n{'='*80}")
    print("预期性能提升：+4%~+8% Macro-F1")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

