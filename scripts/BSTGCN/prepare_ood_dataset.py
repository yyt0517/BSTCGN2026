"""
Leave-One-Out OOD数据集准备脚本

实验设置:
- 训练集: 8类攻击 (排除SlowrateDoS, label=5)
- ID测试集: 8类攻击的测试样本
- OOD测试集: SlowrateDoS样本

SlowrateDoS特征:
- 低速率攻击，与其他高速率洪泛攻击模式差异大
- 占数据集约10%的包
- 是验证模型对未知攻击检测能力的理想选择
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 类别定义
CLASS_NAMES = [
    'Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 
    'SYNScan', 'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan'
]

# OOD类别: SlowrateDoS (label=5)
OOD_CLASS_LABEL = 5
OOD_CLASS_NAME = 'SlowrateDoS'


def prepare_ood_dataset(
    input_dir: str = 'data/processed/raw_packet_sequences_50packets',
    output_dir: str = 'data/processed/ood_slowrate',
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
):
    """
    准备Leave-One-Out OOD数据集

    Args:
        input_dir: 原始数据目录
        output_dir: 输出目录
        val_size: 验证集比例
        test_size: 测试集比例
        random_state: 随机种子
    """
    print("=" * 80)
    print("Prepare Leave-One-Out OOD Dataset".center(80))
    print(f"OOD Class: {OOD_CLASS_NAME} (label={OOD_CLASS_LABEL})".center(80))
    print("=" * 80)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始数据 (支持单文件或分割文件格式)
    print(f"\n[1/5] 加载原始数据: {input_dir}")

    # 检查是否存在单一的 sequences.npy 文件
    if (input_dir / 'sequences.npy').exists():
        X = np.load(input_dir / 'sequences.npy')
        y = np.load(input_dir / 'labels.npy')
    else:
        # 加载已分割的数据并合并
        print("  检测到分割数据格式，正在合并...")
        train_seq = np.load(input_dir / 'train_sequences.npy')
        train_labels = np.load(input_dir / 'train_labels.npy')
        val_seq = np.load(input_dir / 'val_sequences.npy')
        val_labels = np.load(input_dir / 'val_labels.npy')
        test_seq = np.load(input_dir / 'test_sequences.npy')
        test_labels = np.load(input_dir / 'test_labels.npy')

        X = np.concatenate([train_seq, val_seq, test_seq], axis=0)
        y = np.concatenate([train_labels, val_labels, test_labels], axis=0)
        print(f"  合并后总样本数: {len(X):,}")

    with open(input_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"  总样本数: {len(X):,}")
    print(f"  序列形状: {X.shape}")
    
    # 分离ID和OOD样本
    print(f"\n[2/5] 分离ID和OOD样本")
    id_mask = y != OOD_CLASS_LABEL
    ood_mask = y == OOD_CLASS_LABEL
    
    X_id = X[id_mask]
    y_id = y[id_mask]
    X_ood = X[ood_mask]
    y_ood = y[ood_mask]
    
    print(f"  ID样本 (8类): {len(X_id):,} ({len(X_id)/len(X)*100:.1f}%)")
    print(f"  OOD样本 ({OOD_CLASS_NAME}): {len(X_ood):,} ({len(X_ood)/len(X)*100:.1f}%)")
    
    # 重新映射标签 (移除OOD类后，label需要调整)
    # 原始: 0,1,2,3,4,5,6,7,8 -> 新: 0,1,2,3,4,-,5,6,7
    print(f"\n[3/5] 重新映射ID类别标签")
    label_mapping = {}
    new_label = 0
    for old_label in range(9):
        if old_label != OOD_CLASS_LABEL:
            label_mapping[old_label] = new_label
            new_label += 1
    
    y_id_remapped = np.array([label_mapping[l] for l in y_id])
    
    id_class_names = [name for i, name in enumerate(CLASS_NAMES) if i != OOD_CLASS_LABEL]
    print(f"  ID类别: {id_class_names}")
    print(f"  标签映射: {label_mapping}")
    
    # 划分ID数据为训练/验证/测试集
    print(f"\n[4/5] 划分ID数据 (train:val:test = 70:15:15)")
    
    # 第一次划分: 分出测试集
    X_temp, X_id_test, y_temp, y_id_test = train_test_split(
        X_id, y_id_remapped, test_size=test_size, 
        random_state=random_state, stratify=y_id_remapped
    )
    
    # 第二次划分: 分出验证集
    val_size_adj = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adj,
        random_state=random_state, stratify=y_temp
    )
    
    print(f"  训练集: {len(X_train):,}")
    print(f"  验证集: {len(X_val):,}")
    print(f"  ID测试集: {len(X_id_test):,}")
    print(f"  OOD测试集: {len(X_ood):,}")
    
    # 标准化 (只在训练集上fit)
    print(f"\n[5/5] 标准化数据")
    N_train, seq_len, num_features = X_train.shape
    
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)
    
    X_train_norm = scaler.transform(X_train_flat).reshape(N_train, seq_len, num_features)
    X_val_norm = scaler.transform(X_val.reshape(-1, num_features)).reshape(len(X_val), seq_len, num_features)
    X_id_test_norm = scaler.transform(X_id_test.reshape(-1, num_features)).reshape(len(X_id_test), seq_len, num_features)
    X_ood_norm = scaler.transform(X_ood.reshape(-1, num_features)).reshape(len(X_ood), seq_len, num_features)
    
    # 保存数据
    print(f"\n保存数据到: {output_dir}")
    
    np.save(output_dir / 'train_sequences.npy', X_train_norm.astype(np.float32))
    np.save(output_dir / 'train_labels.npy', y_train.astype(np.int64))
    np.save(output_dir / 'val_sequences.npy', X_val_norm.astype(np.float32))
    np.save(output_dir / 'val_labels.npy', y_val.astype(np.int64))
    np.save(output_dir / 'test_id_sequences.npy', X_id_test_norm.astype(np.float32))
    np.save(output_dir / 'test_id_labels.npy', y_id_test.astype(np.int64))
    np.save(output_dir / 'test_ood_sequences.npy', X_ood_norm.astype(np.float32))
    
    # 保存metadata
    ood_metadata = {
        'num_id_classes': 8,
        'id_class_names': id_class_names,
        'ood_class_name': OOD_CLASS_NAME,
        'ood_class_original_label': OOD_CLASS_LABEL,
        'label_mapping': label_mapping,
        'num_features': num_features,
        'sequence_length': seq_len,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_id_size': len(X_id_test),
        'test_ood_size': len(X_ood),
        'scaler': scaler,
        'random_seed': random_state,
        'feature_names': metadata.get('feature_names', [])
    }

    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(ood_metadata, f)

    # 打印类别分布
    print(f"\n训练集类别分布:")
    for label in range(8):
        count = (y_train == label).sum()
        print(f"  {id_class_names[label]:20s}: {count:,} ({count/len(y_train)*100:.2f}%)")

    print(f"\n" + "=" * 80)
    print("OOD数据集准备完成!".center(80))
    print("=" * 80)

    print(f"\n实验说明:")
    print(f"  1. 训练模型时使用 train_sequences.npy 和 train_labels.npy (8类)")
    print(f"  2. 验证时使用 val_sequences.npy 和 val_labels.npy")
    print(f"  3. 评估OOD检测时:")
    print(f"     - ID样本: test_id_sequences.npy (应预测为某个已知类，低不确定性)")
    print(f"     - OOD样本: test_ood_sequences.npy (应表现出高不确定性)")
    print(f"  4. OOD检测指标: 用epistemic uncertainty区分ID vs OOD")

    return ood_metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare Leave-One-Out OOD Dataset')
    parser.add_argument('--input-dir', type=str,
                        default='data/processed/raw_packet_sequences_50packets',
                        help='Input data directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed/ood_slowrate',
                        help='Output directory for OOD dataset')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation set size ratio')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set size ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    prepare_ood_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed
    )

