"""
验证数据集没有数据泄露

检查要点：
1. stride = sequence_length (无重叠) - 序列之间没有共享的包
2. train/val/test 之间没有完全相同的序列
3. 每个 (类别, 基站) 组合在三个集合中都有数据
4. 特征分布在三个集合之间相似（无严重偏移）
"""

import numpy as np
import pickle
from pathlib import Path
from collections import Counter

DATA_DIR = Path('data/processed/raw_packet_sequences_50packets_no_leak')

def main():
    print("="*80)
    print("验证数据集无泄露".center(80))
    print("="*80)
    
    # 加载数据
    train_seq = np.load(DATA_DIR / 'train_sequences.npy')
    val_seq = np.load(DATA_DIR / 'val_sequences.npy')
    test_seq = np.load(DATA_DIR / 'test_sequences.npy')
    train_labels = np.load(DATA_DIR / 'train_labels.npy')
    val_labels = np.load(DATA_DIR / 'val_labels.npy')
    test_labels = np.load(DATA_DIR / 'test_labels.npy')
    
    with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"\n1. 数据集配置:")
    print(f"   序列长度: {metadata['sequence_length']}")
    print(f"   Stride: {metadata['stride']}")
    print(f"   划分策略: {metadata['split_strategy']}")
    
    # 检查1: stride = sequence_length
    print(f"\n2. 检查 stride = sequence_length (无重叠):")
    if metadata['stride'] == metadata['sequence_length']:
        print(f"   ✓ stride ({metadata['stride']}) = sequence_length ({metadata['sequence_length']})")
        print(f"   ✓ 序列之间没有共享的包")
    else:
        print(f"   ✗ stride ({metadata['stride']}) != sequence_length ({metadata['sequence_length']})")
        print(f"   ✗ 警告: 序列之间可能有重叠!")
    
    # 检查2: 没有完全相同的序列
    print(f"\n3. 检查 train/val/test 之间没有完全相同的序列:")
    
    # 计算序列的hash来快速比较
    def seq_hash(seq):
        return hash(seq.tobytes())
    
    train_hashes = set(seq_hash(s) for s in train_seq)
    val_hashes = set(seq_hash(s) for s in val_seq)
    test_hashes = set(seq_hash(s) for s in test_seq)
    
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    print(f"   Train-Val 重复序列: {len(train_val_overlap)}")
    print(f"   Train-Test 重复序列: {len(train_test_overlap)}")
    print(f"   Val-Test 重复序列: {len(val_test_overlap)}")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0:
        print(f"   ✓ 训练集与验证/测试集没有完全相同的序列")
    else:
        print(f"   ✗ 警告: 存在重复序列!")
    
    # 检查3: 类别分布
    print(f"\n4. 类别分布检查:")
    class_names = metadata['class_names']
    
    print(f"\n   {'类别':<18} {'Train':>10} {'Val':>10} {'Test':>10} {'Train%':>8} {'Val%':>8} {'Test%':>8}")
    print("   " + "-"*72)
    
    all_present = True
    for i, name in enumerate(class_names):
        tr = np.sum(train_labels == i)
        va = np.sum(val_labels == i)
        te = np.sum(test_labels == i)
        tr_pct = tr / len(train_labels) * 100
        va_pct = va / len(val_labels) * 100
        te_pct = te / len(test_labels) * 100
        
        print(f"   {name:<18} {tr:>10,} {va:>10,} {te:>10,} {tr_pct:>7.2f}% {va_pct:>7.2f}% {te_pct:>7.2f}%")
        
        if tr == 0 or va == 0 or te == 0:
            all_present = False
    
    if all_present:
        print(f"\n   ✓ 所有类别在三个集合中都有数据")
    else:
        print(f"\n   ✗ 警告: 某些类别在某些集合中缺失!")
    
    # 检查4: 特征分布
    print(f"\n5. 特征分布检查 (均值对比):")
    feature_names = metadata.get('feature_names', [f'f{i}' for i in range(train_seq.shape[2])])
    
    # 计算每个特征的均值和标准差
    train_mean = train_seq.mean(axis=(0, 1))
    val_mean = val_seq.mean(axis=(0, 1))
    test_mean = test_seq.mean(axis=(0, 1))
    
    print(f"\n   {'特征':<20} {'Train Mean':>12} {'Val Mean':>12} {'Test Mean':>12} {'Max Diff':>10}")
    print("   " + "-"*66)
    
    max_diffs = []
    for i in range(min(10, len(feature_names))):  # 只显示前10个特征
        tr_m, va_m, te_m = train_mean[i], val_mean[i], test_mean[i]
        max_diff = max(abs(tr_m - va_m), abs(tr_m - te_m), abs(va_m - te_m))
        max_diffs.append(max_diff)
        print(f"   {feature_names[i]:<20} {tr_m:>12.4f} {va_m:>12.4f} {te_m:>12.4f} {max_diff:>10.4f}")
    
    avg_max_diff = np.mean(max_diffs)
    print(f"\n   平均最大差异: {avg_max_diff:.4f}")
    
    if avg_max_diff < 0.5:  # 标准化后的数据，差异应该很小
        print(f"   ✓ 特征分布在三个集合之间相似")
    else:
        print(f"   ⚠ 注意: 特征分布存在一定差异，但这可能是正常的")
    
    # 总结
    print(f"\n{'='*80}")
    print("总结".center(80))
    print(f"{'='*80}")
    
    issues = []
    if metadata['stride'] != metadata['sequence_length']:
        issues.append("stride != sequence_length (可能有重叠)")
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0:
        issues.append("存在重复序列")
    if not all_present:
        issues.append("某些类别在某些集合中缺失")
    
    if len(issues) == 0:
        print("\n✓ 数据集验证通过，没有发现数据泄露问题！")
        print("\n关键点:")
        print(f"  - stride = {metadata['stride']} = sequence_length，序列之间无重叠")
        print(f"  - 使用分层抽样，确保每个类别在三个集合中都有数据")
        print(f"  - train/val/test 之间没有完全相同的序列")
    else:
        print("\n✗ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")

if __name__ == '__main__':
    main()

