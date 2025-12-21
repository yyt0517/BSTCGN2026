"""
准备50包序列数据集：划分训练/验证/测试集并标准化

对应100包版本的 02_prepare_data_with_val.py
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("准备50包序列数据集 (训练/验证/测试划分)".center(100))
print("="*100)

# 配置
INPUT_DIR = Path('data/processed/raw_packet_sequences_50packets')
OUTPUT_DIR = INPUT_DIR  # 保存到同一目录
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

print(f"\n配置:")
print(f"  输入目录: {INPUT_DIR}")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"  数据划分: {TRAIN_RATIO:.0%} 训练 / {VAL_RATIO:.0%} 验证 / {TEST_RATIO:.0%} 测试")
print(f"  随机种子: {RANDOM_SEED}")

# 加载数据
print(f"\n加载数据...")
sequences = np.load(INPUT_DIR / 'sequences.npy')
labels = np.load(INPUT_DIR / 'labels.npy')

with open(INPUT_DIR / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"  序列形状: {sequences.shape}")
print(f"  标签形状: {labels.shape}")
print(f"  类别数: {metadata['num_classes']}")

# 第一次划分：训练集 vs (验证集+测试集)
print(f"\n第一次划分: 训练集 vs (验证+测试)集...")
train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
    sequences, labels,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED,
    stratify=labels
)

print(f"  训练集: {len(train_sequences):,} 样本")
print(f"  临时集: {len(temp_sequences):,} 样本")

# 第二次划分：验证集 vs 测试集
print(f"\n第二次划分: 验证集 vs 测试集...")
val_sequences, test_sequences, val_labels, test_labels = train_test_split(
    temp_sequences, temp_labels,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED,
    stratify=temp_labels
)

print(f"  验证集: {len(val_sequences):,} 样本")
print(f"  测试集: {len(test_sequences):,} 样本")

# 标准化特征
print(f"\n标准化特征...")
print(f"  原始形状: {train_sequences.shape}")

# 重塑为 (N*L, F) 以便标准化
n_train, seq_len, n_features = train_sequences.shape
train_flat = train_sequences.reshape(-1, n_features)
val_flat = val_sequences.reshape(-1, n_features)
test_flat = test_sequences.reshape(-1, n_features)

# 使用训练集拟合 Scaler
scaler = StandardScaler()
train_flat = scaler.fit_transform(train_flat)
val_flat = scaler.transform(val_flat)
test_flat = scaler.transform(test_flat)

# 重塑回 (N, L, F)
train_sequences = train_flat.reshape(n_train, seq_len, n_features)
val_sequences = val_flat.reshape(len(val_sequences), seq_len, n_features)
test_sequences = test_flat.reshape(len(test_sequences), seq_len, n_features)

print(f"  标准化后形状: {train_sequences.shape}")

# 保存数据
print(f"\n保存数据到 {OUTPUT_DIR}...")
np.save(OUTPUT_DIR / 'train_sequences.npy', train_sequences)
np.save(OUTPUT_DIR / 'train_labels.npy', train_labels)
np.save(OUTPUT_DIR / 'val_sequences.npy', val_sequences)
np.save(OUTPUT_DIR / 'val_labels.npy', val_labels)
np.save(OUTPUT_DIR / 'test_sequences.npy', test_sequences)
np.save(OUTPUT_DIR / 'test_labels.npy', test_labels)

print(f"  ✓ train_sequences.npy: {train_sequences.nbytes / 1024**2:.2f} MB")
print(f"  ✓ val_sequences.npy: {val_sequences.nbytes / 1024**2:.2f} MB")
print(f"  ✓ test_sequences.npy: {test_sequences.nbytes / 1024**2:.2f} MB")

# 打印数据集统计
print(f"\n数据集统计:")
print(f"  训练集: {len(train_sequences):,} 样本 ({len(train_sequences)/len(sequences)*100:.1f}%)")
print(f"  验证集: {len(val_sequences):,} 样本 ({len(val_sequences)/len(sequences)*100:.1f}%)")
print(f"  测试集: {len(test_sequences):,} 样本 ({len(test_sequences)/len(sequences)*100:.1f}%)")
print(f"  总计: {len(sequences):,} 样本")

# 打印每个集合的类别分布
for split_name, split_labels in [('训练集', train_labels), ('验证集', val_labels), ('测试集', test_labels)]:
    print(f"\n{split_name}类别分布:")
    for i, class_name in enumerate(metadata['class_names']):
        count = np.sum(split_labels == i)
        print(f"  {class_name:20s}: {count:6,} ({count/len(split_labels)*100:5.2f}%)")

print("\n" + "="*100)
print("✓ 数据准备完成！".center(100))
print("="*100)

