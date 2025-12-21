"""
验证当前生成的时序数据集
"""
import pandas as pd
import os

print("="*100)
print("验证当前时序数据集".center(100))
print("="*100)

# Check if file exists
file_path = 'timeseries_from_pcap_1s.csv'

if not os.path.exists(file_path):
    print(f"\n✗ 文件不存在: {file_path}")
    print("\n请先运行: python quick_start.py")
    exit(1)

print(f"\n✓ 文件存在: {file_path}")
print(f"  文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")

# Load data
print("\n正在加载数据...")
df = pd.read_csv(file_path)

print(f"\n✓ 数据加载成功")
print(f"  样本数: {len(df):,}")
print(f"  特征数: {len(df.columns)}")

# Check attack type distribution
print("\n" + "="*100)
print("攻击类型分布".center(100))
print("="*100)

attack_counts = df['attack_type'].value_counts().sort_index()

print("\nLabel | 类型           | 样本数      | 占比")
print("-" * 60)

attack_names = {
    0: 'Benign',
    1: 'HTTPFlood',
    2: 'ICMPFlood',
    3: 'SYNFlood',
    4: 'SYNScan',
    5: 'SlowrateDoS',
    6: 'TCPConnectScan',
    7: 'UDPFlood',
    8: 'UDPScan'
}

for label in range(9):
    if label in attack_counts.index:
        count = attack_counts[label]
        percentage = count / len(df) * 100
        name = attack_names.get(label, 'Unknown')
        print(f"{label:5d} | {name:14s} | {count:10,d} | {percentage:5.2f}%")
    else:
        name = attack_names.get(label, 'Unknown')
        print(f"{label:5d} | {name:14s} | {'缺失':>10s} | {'0.00%':>6s}  ✗")

print("-" * 60)
print(f"总计 | {'':14s} | {len(df):10,d} | 100.00%")

# Check for missing types
print("\n" + "="*100)
print("问题检查".center(100))
print("="*100)

missing_types = []
for label in range(9):
    if label not in attack_counts.index:
        missing_types.append((label, attack_names[label]))

if missing_types:
    print("\n⚠️  发现缺失的攻击类型:")
    for label, name in missing_types:
        print(f"  - Label {label}: {name}")
    
    if 0 in [l for l, _ in missing_types]:
        print("\n✗ 严重问题: Benign（正常流量）缺失！")
        print("  模型无法学习正常行为模式")
    
    if 2 in [l for l, _ in missing_types]:
        print("\n✗ 严重问题: ICMPFlood缺失！")
        print("  这是我们要修复的主要问题")
else:
    print("\n✓ 所有9种攻击类型都存在")

# Check if BS1 and BS2 are both included
print("\n" + "="*100)
print("数据来源检查".center(100))
print("="*100)

print(f"\n当前数据集样本数: {len(df):,}")

if len(df) < 20000:
    print("\n⚠️  样本数较少，可能只包含BS1数据")
    print("  预期BS1+BS2应该有 100,000-200,000 样本")
    print("  当前只有约 10,000-20,000 样本")
    print("\n建议: 使用修复后的脚本重新生成数据")
elif len(df) < 50000:
    print("\n⚠️  样本数偏少，可能是快速测试模式")
    print("  或者只包含部分数据")
else:
    print("\n✓ 样本数充足，可能包含BS1+BS2数据")

# Feature check
print("\n" + "="*100)
print("特征检查".center(100))
print("="*100)

print(f"\n特征列表 (共{len(df.columns)}列):")
print("\n前10个特征:")
for i, col in enumerate(df.columns[:10], 1):
    print(f"  {i:2d}. {col}")

if len(df.columns) > 10:
    print(f"  ... (还有 {len(df.columns) - 10} 个特征)")

# Summary
print("\n" + "="*100)
print("总结".center(100))
print("="*100)

print("\n当前数据集状态:")
print(f"  样本数: {len(df):,}")
print(f"  特征数: {len(df.columns)}")
print(f"  类型数: {len(attack_counts)}/9")

if len(attack_counts) == 9:
    print("\n✓ 数据集完整，包含所有9种类型")
else:
    print(f"\n✗ 数据集不完整，缺少 {9 - len(attack_counts)} 种类型")

if 0 in attack_counts.index:
    print(f"✓ 包含Benign（正常流量）: {attack_counts[0]:,} 样本")
else:
    print("✗ 缺少Benign（正常流量）")

if 2 in attack_counts.index:
    print(f"✓ 包含ICMPFlood: {attack_counts[2]:,} 样本")
else:
    print("✗ 缺少ICMPFlood")

if len(df) >= 50000:
    print("✓ 样本数充足")
else:
    print("⚠️  样本数偏少")

# Recommendations
print("\n" + "="*100)
print("建议".center(100))
print("="*100)

if len(attack_counts) < 9 or 0 not in attack_counts.index or 2 not in attack_counts.index:
    print("\n⚠️  当前数据集存在问题，建议重新生成:")
    print("\n  1. 使用修复后的脚本:")
    print("     python quick_start.py")
    print("\n  2. 选择选项2（快速测试）验证修复:")
    print("     - 时间: 1-2小时")
    print("     - 输出: timeseries_from_pcap_1s_test.csv")
    print("\n  3. 如果测试成功，运行选项1（完整处理）:")
    print("     - 时间: 4-8小时")
    print("     - 输出: timeseries_from_pcap_1s.csv")
else:
    print("\n✓ 数据集质量良好，可以用于模型训练")
    print("\n下一步:")
    print("  1. 数据预处理和特征工程")
    print("  2. 划分训练集/验证集/测试集")
    print("  3. 处理类别不平衡（SMOTE等）")
    print("  4. 训练分类模型")

print("\n" + "="*100)

