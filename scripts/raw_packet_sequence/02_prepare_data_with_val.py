"""
准备原始包序列数据：标准化、划分训练/验证/测试集 (70:15:15)

修复测试集泄露问题：
- 训练集：70%
- 验证集：15% (用于早停和学习率调度)
- 测试集：15% (只在最后评估一次)
"""

import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子，确保可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*100)
print("Prepare Raw Packet Sequence Data with Train/Val/Test Split (70:15:15)".center(100))
print("="*100)


def load_raw_sequences(data_dir='data/processed/raw_packet_sequences'):
    """加载原始包序列数据"""
    print(f"\n加载数据: {data_dir}")
    
    X = np.load(os.path.join(data_dir, 'sequences.npy'))
    y = np.load(os.path.join(data_dir, 'labels.npy'))
    
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"  序列形状: {X.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  特征数量: {metadata['num_features']}")
    print(f"  类别数量: {metadata['num_classes']}")
    
    return X, y, metadata


def normalize_sequences(X_train, X_val, X_test, feature_names):
    """标准化序列数据（在包级别标准化）"""
    print(f"\n标准化序列数据...")
    
    # 重塑为 (N*seq_len, num_features)
    N_train, seq_len, num_features = X_train.shape
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(-1, num_features)
    X_val_flat = X_val.reshape(-1, num_features)
    X_test_flat = X_test.reshape(-1, num_features)
    
    # 只在训练集上拟合scaler
    scaler = StandardScaler()
    X_train_norm_flat = scaler.fit_transform(X_train_flat)
    X_val_norm_flat = scaler.transform(X_val_flat)
    X_test_norm_flat = scaler.transform(X_test_flat)
    
    # 重塑回 (N, seq_len, num_features)
    X_train_norm = X_train_norm_flat.reshape(N_train, seq_len, num_features)
    X_val_norm = X_val_norm_flat.reshape(N_val, seq_len, num_features)
    X_test_norm = X_test_norm_flat.reshape(N_test, seq_len, num_features)
    
    print(f"  训练集: {X_train_norm.shape}")
    print(f"  验证集: {X_val_norm.shape}")
    print(f"  测试集: {X_test_norm.shape}")
    
    # 显示标准化统计
    print(f"\n标准化统计 (训练集):")
    print(f"  均值范围: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  标准差范围: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def prepare_dataset(input_dir='data/processed/raw_packet_sequences',
                   output_dir='data/processed/raw_packet_sequences',
                   val_size=0.15,
                   test_size=0.15,
                   random_state=42):
    """
    准备数据集：划分训练/验证/测试集并标准化
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        val_size: 验证集比例 (0.15 = 15%)
        test_size: 测试集比例 (0.15 = 15%)
        random_state: 随机种子
    """
    
    # 加载数据
    X, y, metadata = load_raw_sequences(input_dir)
    
    # 第一次划分：分出测试集 (15%)
    print(f"\n第一次划分：分出测试集 (test_size={test_size})...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 第二次划分：从剩余数据中分出验证集
    # 剩余85%中的 15/85 ≈ 0.176 作为验证集，使得验证集占总数据的15%
    val_size_adjusted = val_size / (1 - test_size)
    print(f"第二次划分：从剩余数据中分出验证集 (val_size={val_size_adjusted:.3f})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"\n数据划分结果:")
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape} ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  验证集: {X_val.shape}, 标签: {y_val.shape} ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape} ({len(y_test)/len(y)*100:.1f}%)")
    
    # 检查标签分布
    print(f"\n训练集标签分布:")
    for label in range(metadata['num_classes']):
        count = (y_train == label).sum()
        print(f"  {metadata['class_names'][label]:20s}: {count:,} ({count/len(y_train)*100:.2f}%)")
    
    print(f"\n验证集标签分布:")
    for label in range(metadata['num_classes']):
        count = (y_val == label).sum()
        print(f"  {metadata['class_names'][label]:20s}: {count:,} ({count/len(y_val)*100:.2f}%)")
    
    print(f"\n测试集标签分布:")
    for label in range(metadata['num_classes']):
        count = (y_test == label).sum()
        print(f"  {metadata['class_names'][label]:20s}: {count:,} ({count/len(y_test)*100:.2f}%)")
    
    # 标准化
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_sequences(
        X_train, X_val, X_test, metadata['feature_names']
    )
    
    # 保存数据
    print(f"\n保存数据到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'train_sequences.npy'), X_train_norm)
    np.save(os.path.join(output_dir, 'val_sequences.npy'), X_val_norm)
    np.save(os.path.join(output_dir, 'test_sequences.npy'), X_test_norm)
    np.save(os.path.join(output_dir, 'train_labels.npy'), y_train)
    np.save(os.path.join(output_dir, 'val_labels.npy'), y_val)
    np.save(os.path.join(output_dir, 'test_labels.npy'), y_test)
    
    # 更新metadata
    metadata['scaler'] = scaler
    metadata['train_size'] = len(X_train_norm)
    metadata['val_size'] = len(X_val_norm)
    metadata['test_size'] = len(X_test_norm)
    metadata['random_seed'] = random_state
    metadata['split_ratio'] = {
        'train': 1 - val_size - test_size,
        'val': val_size,
        'test': test_size
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  ✓ train_sequences.npy: {X_train_norm.nbytes / (1024**2):.2f} MB")
    print(f"  ✓ val_sequences.npy: {X_val_norm.nbytes / (1024**2):.2f} MB")
    print(f"  ✓ test_sequences.npy: {X_test_norm.nbytes / (1024**2):.2f} MB")
    print(f"  ✓ train_labels.npy: {y_train.nbytes / (1024**2):.2f} MB")
    print(f"  ✓ val_labels.npy: {y_val.nbytes / (1024**2):.2f} MB")
    print(f"  ✓ test_labels.npy: {y_test.nbytes / (1024**2):.2f} MB")
    print(f"  ✓ metadata.pkl")
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"  总样本数: {len(X):,}")
    print(f"  训练样本: {len(X_train_norm):,} ({len(X_train_norm)/len(X)*100:.1f}%)")
    print(f"  验证样本: {len(X_val_norm):,} ({len(X_val_norm)/len(X)*100:.1f}%)")
    print(f"  测试样本: {len(X_test_norm):,} ({len(X_test_norm)/len(X)*100:.1f}%)")
    print(f"  序列长度: {X_train_norm.shape[1]} packets")
    print(f"  特征维度: {X_train_norm.shape[2]}")
    print(f"  类别数量: {metadata['num_classes']}")
    print(f"  随机种子: {random_state}")
    
    return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, metadata


if __name__ == "__main__":
    INPUT_DIR = 'data/processed/raw_packet_sequences'
    OUTPUT_DIR = 'data/processed/raw_packet_sequences'
    VAL_SIZE = 0.15  # 15%
    TEST_SIZE = 0.15  # 15%
    RANDOM_STATE = 42
    
    print(f"\nConfiguration:")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Train size: {1 - VAL_SIZE - TEST_SIZE:.1%} (70%)")
    print(f"  Val size: {VAL_SIZE:.1%} (15%)")
    print(f"  Test size: {TEST_SIZE:.1%} (15%)")
    print(f"  Random state: {RANDOM_STATE}")
    
    # Prepare dataset
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = prepare_dataset(
        INPUT_DIR,
        OUTPUT_DIR,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    print("\n" + "="*100)
    print("SUCCESS! Data Preparation Complete".center(100))
    print("="*100)
    print("\n⚠️  重要提示:")
    print("  - 训练集: 用于训练模型")
    print("  - 验证集: 用于早停和学习率调度")
    print("  - 测试集: 只在最后评估一次，避免测试集泄露")

