"""
构建无数据泄露的50包序列数据集 (V2)

问题：原始方案中使用滑动窗口（50%重叠）构建序列，然后随机划分train/val/test。
     相邻序列共享25个包，随机划分会导致训练集和测试集包含相同的包，造成数据泄露。

解决方案 V2：
    1. stride = sequence_length (无重叠) - 彻底消除数据泄露
    2. 按类别+基站分层抽样 - 确保每个类别、每个基站在三个集合中都有数据

    对于每个 (类别, 基站) 组合的序列，按比例随机划分到 train/val/test
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入特征提取函数
try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw
    print("✓ Scapy is available")
except ImportError:
    print("✗ Scapy is not installed. Please install it: conda install -c conda-forge scapy")
    exit(1)

# 类别名称映射
CLASS_NAMES = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan',
               'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

# 配置
SEQUENCE_LENGTH = 50
STRIDE = 50  # 无重叠，stride = sequence_length
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
USE_GTP_REMOVED = True
OUTPUT_DIR = 'data/processed/raw_packet_sequences_50packets_no_leak'
RANDOM_SEED = 42


def ip_to_int(ip_str):
    """将IP地址转换为整数"""
    try:
        parts = ip_str.split('.')
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except:
        return 0


def is_private_ip(ip_str):
    """判断是否为私有IP"""
    try:
        parts = [int(p) for p in ip_str.split('.')]
        if parts[0] == 10:
            return 1
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return 1
        if parts[0] == 192 and parts[1] == 168:
            return 1
        return 0
    except:
        return 0


def extract_packet_features(pkt, base_time):
    """从单个数据包中提取25个特征"""
    features = {
        'timestamp': float(pkt.time) - base_time,
        'packet_length': len(pkt),
        'protocol': 0,
        'src_ip_int': 0, 'dst_ip_int': 0,
        'src_ip_is_private': 0, 'dst_ip_is_private': 0,
        'src_port': 0, 'dst_port': 0,
        'ip_ttl': 0, 'ip_len': 0, 'ip_tos': 0, 'ip_flags': 0,
        'tcp_flags_syn': 0, 'tcp_flags_ack': 0, 'tcp_flags_fin': 0,
        'tcp_flags_rst': 0, 'tcp_flags_psh': 0, 'tcp_flags_urg': 0,
        'tcp_window': 0, 'tcp_seq': 0, 'tcp_ack': 0,
        'payload_size': 0,
        'icmp_type': 0, 'icmp_code': 0,
    }

    if IP in pkt:
        features['src_ip_int'] = ip_to_int(pkt[IP].src)
        features['dst_ip_int'] = ip_to_int(pkt[IP].dst)
        features['src_ip_is_private'] = is_private_ip(pkt[IP].src)
        features['dst_ip_is_private'] = is_private_ip(pkt[IP].dst)
        features['ip_ttl'] = pkt[IP].ttl
        features['ip_len'] = pkt[IP].len
        features['ip_tos'] = pkt[IP].tos
        features['ip_flags'] = int(pkt[IP].flags)

    if TCP in pkt:
        features['protocol'] = 1
        features['src_port'] = pkt[TCP].sport
        features['dst_port'] = pkt[TCP].dport
        flags = str(pkt[TCP].flags)
        features['tcp_flags_syn'] = 1 if 'S' in flags else 0
        features['tcp_flags_ack'] = 1 if 'A' in flags else 0
        features['tcp_flags_fin'] = 1 if 'F' in flags else 0
        features['tcp_flags_rst'] = 1 if 'R' in flags else 0
        features['tcp_flags_psh'] = 1 if 'P' in flags else 0
        features['tcp_flags_urg'] = 1 if 'U' in flags else 0
        features['tcp_window'] = pkt[TCP].window
        features['tcp_seq'] = pkt[TCP].seq % 1000000
        features['tcp_ack'] = pkt[TCP].ack % 1000000
        if Raw in pkt:
            features['payload_size'] = len(pkt[Raw].load)
    elif UDP in pkt:
        features['protocol'] = 2
        features['src_port'] = pkt[UDP].sport
        features['dst_port'] = pkt[UDP].dport
        if Raw in pkt:
            features['payload_size'] = len(pkt[Raw].load)
    elif ICMP in pkt:
        features['protocol'] = 3
        features['icmp_type'] = pkt[ICMP].type if hasattr(pkt[ICMP], 'type') else 0
        features['icmp_code'] = pkt[ICMP].code if hasattr(pkt[ICMP], 'code') else 0

    return features


def create_sequences_from_packets(packet_features, sequence_length, stride):
    """从包特征列表创建序列（无重叠时 stride = sequence_length）"""
    sequences = []
    for i in range(0, len(packet_features) - sequence_length + 1, stride):
        seq = packet_features[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32) if sequences else None


def process_pcap_stratified(pcap_path, attack_label, base_station, sequence_length, stride):
    """
    处理单个PCAP文件，返回所有序列（后续按类别+基站分层划分）

    返回: (sequences, labels, base_station_ids) 或 (None, None, None)
    """
    print(f"\n  Reading PCAP: {os.path.basename(pcap_path)}")

    try:
        packets = rdpcap(pcap_path)
        print(f"  Total packets: {len(packets):,}")

        if len(packets) < sequence_length:
            print(f"  ⚠️  Not enough packets (need at least {sequence_length})")
            return None, None, None

        # 提取所有包的特征
        base_time = float(packets[0].time)
        packet_features = []
        for pkt in tqdm(packets, desc="  Extracting", leave=False):
            features = extract_packet_features(pkt, base_time)
            packet_features.append(list(features.values()))

        packet_features = np.array(packet_features, dtype=np.float32)

        # 构建序列（无重叠）
        sequences = create_sequences_from_packets(packet_features, sequence_length, stride)

        if sequences is None or len(sequences) == 0:
            print(f"  ⚠️  No sequences created")
            return None, None, None

        n_seqs = len(sequences)
        labels = np.full(n_seqs, attack_label, dtype=np.int64)
        base_station_ids = np.full(n_seqs, base_station, dtype=np.int64)

        print(f"  Sequences created: {n_seqs}")

        return sequences, labels, base_station_ids

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None



def stratified_split_by_group(sequences, labels, base_stations, train_ratio, val_ratio, test_ratio, random_seed):
    """
    按类别+基站分层划分数据

    确保每个 (类别, 基站) 组合在 train/val/test 中都有数据
    """
    np.random.seed(random_seed)

    # 创建分组key: (label, base_station)
    groups = {}
    for i in range(len(sequences)):
        key = (labels[i], base_stations[i])
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    train_indices, val_indices, test_indices = [], [], []

    print(f"\n按 (类别, 基站) 分层划分:")
    for key, indices in sorted(groups.items()):
        label, bs = key
        indices = np.array(indices)
        np.random.shuffle(indices)

        n = len(indices)
        n_train = max(1, int(n * train_ratio))  # 至少保证1个样本
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val

        if n_test < 1:
            # 样本太少时调整
            n_test = 1
            n_train = max(1, n - n_val - n_test)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train+n_val])
        test_indices.extend(indices[n_train+n_val:])

        print(f"  {CLASS_NAMES[label]:<15} BS{bs+1}: total={n:>5}, train={n_train:>5}, val={n_val:>4}, test={n_test:>4}")

    return (np.array(train_indices), np.array(val_indices), np.array(test_indices))


def build_no_leak_dataset(base_dir, output_dir, sequence_length, stride,
                          train_ratio, val_ratio, test_ratio, use_gtp_removed=True):
    """构建无数据泄露的数据集 (分层抽样版本)"""

    os.makedirs(output_dir, exist_ok=True)

    # 确定PCAP目录 (基于实际文件结构)
    if use_gtp_removed:
        bs1_dir = os.path.join(base_dir, 'BS1_GTP_removed')
        bs2_dir = os.path.join(base_dir, 'BS2 GTP removed')
        print("\n📁 Using GTP-removed PCAP files")
    else:
        bs1_dir = os.path.join(base_dir, 'BS1')
        bs2_dir = os.path.join(base_dir, 'BS2')
        print("\n📁 Using original PCAP files")

    # PCAP文件映射 (BS1) - 使用实际文件名
    attack_files_bs1 = {
        'SSH_BS1_nogtp.pcapng': 0,           # Benign
        'Goldeneye_BS1_nogtp.pcapng': 1,     # HTTPFlood
        'ICMPflood_BS1_nogtp.pcapng': 2,     # ICMPFlood
        'SYNflood_BS1_nogtp.pcapng': 3,      # SYNFlood
        'SYNScan_BS1_nogtp.pcapng': 4,       # SYNScan
        'Slowloris_BS1_nogtp.pcapng': 5,     # SlowrateDoS
        'TCPConnect_BS1_nogtp.pcapng': 6,    # TCPConnectScan
        'UDPflood_BS1_nogtp.pcapng': 7,      # UDPFlood
        'UDPScan_BS1_nogtp.pcapng': 8,       # UDPScan
    }

    # PCAP文件映射 (BS2)
    attack_files_bs2 = {
        'SSH_BS2_nogtp.pcapng': 0,           # Benign
        'Goldeneye_BS2_nogtp.pcapng': 1,     # HTTPFlood
        'ICMPflood_BS2_nogtp.pcapng': 2,     # ICMPFlood
        'SYNflood_BS2_nogtp.pcapng': 3,      # SYNFlood
        'SYNscan_BS2_nogtp.pcapng': 4,       # SYNScan (注意小写s)
        'Slowloris_BS2_nogtp.pcapng': 5,     # SlowrateDoS
        'TCPConnect_BS2_nogtp.pcapng': 6,    # TCPConnectScan
        'UDPflood_BS2_nogtp.pcapng': 7,      # UDPFlood
        'UDPScan_BS2_nogtp.pcapng': 8,       # UDPScan
    }

    # 收集所有数据（先不划分）
    all_sequences = []
    all_labels = []
    all_base_stations = []

    # 处理BS1文件 (base_station = 0)
    print("\n" + "="*100)
    print("Processing BS1 Files".center(100))
    print("="*100)

    for filename, attack_label in attack_files_bs1.items():
        pcap_path = os.path.join(bs1_dir, filename)
        if not os.path.exists(pcap_path):
            print(f"\n⚠️  File not found: {filename}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing: {filename} (Label: {attack_label} - {CLASS_NAMES[attack_label]})")
        print(f"{'='*80}")

        seqs, labels, bs_ids = process_pcap_stratified(
            pcap_path, attack_label, base_station=0,
            sequence_length=sequence_length, stride=stride
        )

        if seqs is not None:
            all_sequences.append(seqs)
            all_labels.append(labels)
            all_base_stations.append(bs_ids)

    # 处理BS2文件 (base_station = 1)
    print("\n" + "="*100)
    print("Processing BS2 Files".center(100))
    print("="*100)

    for filename, attack_label in attack_files_bs2.items():
        pcap_path = os.path.join(bs2_dir, filename)
        if not os.path.exists(pcap_path):
            print(f"\n⚠️  File not found: {filename}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing: {filename} (Label: {attack_label} - {CLASS_NAMES[attack_label]})")
        print(f"{'='*80}")

        seqs, labels, bs_ids = process_pcap_stratified(
            pcap_path, attack_label, base_station=1,
            sequence_length=sequence_length, stride=stride
        )

        if seqs is not None:
            all_sequences.append(seqs)
            all_labels.append(labels)
            all_base_stations.append(bs_ids)

    # 合并所有数据
    print("\n" + "="*100)
    print("Combining Data and Stratified Splitting".center(100))
    print("="*100)

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_base_stations = np.concatenate(all_base_stations, axis=0)

    print(f"\nTotal sequences: {len(all_sequences):,}")
    print(f"Shape: {all_sequences.shape}")

    # 按类别+基站分层划分
    train_idx, val_idx, test_idx = stratified_split_by_group(
        all_sequences, all_labels, all_base_stations,
        train_ratio, val_ratio, test_ratio, RANDOM_SEED
    )

    # 打乱顺序
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    train_sequences = all_sequences[train_idx]
    val_sequences = all_sequences[val_idx]
    test_sequences = all_sequences[test_idx]
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]
    test_labels = all_labels[test_idx]

    print(f"\nAfter splitting:")
    print(f"  Train: {train_sequences.shape}")
    print(f"  Val: {val_sequences.shape}")
    print(f"  Test: {test_sequences.shape}")

    # 标准化（只在训练集上拟合）
    print("\nNormalizing...")
    n_train, seq_len, n_features = train_sequences.shape

    scaler = StandardScaler()
    train_flat = train_sequences.reshape(-1, n_features)
    val_flat = val_sequences.reshape(-1, n_features)
    test_flat = test_sequences.reshape(-1, n_features)

    train_flat = scaler.fit_transform(train_flat)
    val_flat = scaler.transform(val_flat)
    test_flat = scaler.transform(test_flat)

    train_sequences = train_flat.reshape(len(train_sequences), seq_len, n_features)
    val_sequences = val_flat.reshape(len(val_sequences), seq_len, n_features)
    test_sequences = test_flat.reshape(len(test_sequences), seq_len, n_features)

    # 保存数据
    print(f"\nSaving to {output_dir}...")
    np.save(os.path.join(output_dir, 'train_sequences.npy'), train_sequences)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'val_sequences.npy'), val_sequences)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(output_dir, 'test_sequences.npy'), test_sequences)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    # 保存scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # 保存metadata
    feature_names = ['timestamp', 'packet_length', 'protocol', 'src_ip_int', 'dst_ip_int',
                     'src_ip_is_private', 'dst_ip_is_private', 'src_port', 'dst_port',
                     'ip_ttl', 'ip_len', 'ip_tos', 'ip_flags',
                     'tcp_flags_syn', 'tcp_flags_ack', 'tcp_flags_fin',
                     'tcp_flags_rst', 'tcp_flags_psh', 'tcp_flags_urg',
                     'tcp_window', 'tcp_seq', 'tcp_ack', 'payload_size',
                     'icmp_type', 'icmp_code']

    metadata = {
        'sequence_length': sequence_length,
        'stride': stride,
        'num_features': n_features,
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'feature_names': feature_names,
        'split_strategy': 'stratified_by_class_and_base_station',
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'use_gtp_removed': use_gtp_removed,
        'standardization': 'StandardScaler (fit on train)',
        'random_seed': RANDOM_SEED,
        'train_samples': len(train_sequences),
        'val_samples': len(val_sequences),
        'test_samples': len(test_sequences),
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # 打印统计
    print(f"\n{'='*100}")
    print("Dataset Statistics".center(100))
    print(f"{'='*100}")
    print(f"\nSplit Strategy: STRATIFIED by (class, base_station)")
    print(f"  - No overlap (stride = sequence_length = {sequence_length})")
    print(f"  - Every (class, BS) combination has data in all splits")
    print(f"  - Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    print(f"\nDataset Size:")
    print(f"  Train: {len(train_sequences):,} samples")
    print(f"  Val: {len(val_sequences):,} samples")
    print(f"  Test: {len(test_sequences):,} samples")
    print(f"  Total: {len(train_sequences)+len(val_sequences)+len(test_sequences):,} samples")

    print(f"\nClass Distribution:")
    print(f"{'Class':<20} {'Train':>10} {'Val':>10} {'Test':>10}")
    print("-"*50)
    for i, name in enumerate(CLASS_NAMES):
        tr = np.sum(train_labels == i)
        va = np.sum(val_labels == i)
        te = np.sum(test_labels == i)
        print(f"{name:<20} {tr:>10,} {va:>10,} {te:>10,}")

    print(f"\n✓ Dataset saved to {output_dir}")


if __name__ == '__main__':
    print("="*100)
    print("Building No-Leak 50-Packet Dataset (V2 - Stratified)".center(100))
    print("="*100)

    BASE_DIR = 'data/5G-NIDD-raw'

    print(f"\nConfiguration:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Stride: {STRIDE} (NO OVERLAP)")
    print(f"  Split: {TRAIN_RATIO*100:.0f}%/{VAL_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}% (stratified by class+BS)")
    print(f"  Random seed: {RANDOM_SEED}")

    build_no_leak_dataset(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        use_gtp_removed=USE_GTP_REMOVED
    )
