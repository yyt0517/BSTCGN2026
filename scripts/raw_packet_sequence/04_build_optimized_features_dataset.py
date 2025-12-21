"""
构建优化特征的无泄露数据集

特征优化：
1. IP地址：从32位整数改为4字节分离编码 (0-255范围)
2. 移除冗余特征：ip_len, payload_size (与packet_length高度相关r>0.99)
3. 移除派生特征：src_ip_is_private, dst_ip_is_private (可从IP字节计算)
4. 移除低方差特征：tcp_flags_fin, tcp_flags_rst, tcp_flags_urg, icmp_code
5. 移除低信息量特征：tcp_seq, tcp_ack (对分类意义不大)

最终特征：21维 (原25维减少4维，但IP编码更合理)
- 通用(3): timestamp, packet_length, protocol
- IP地址(8): src_ip_b0-b3, dst_ip_b0-b3 (替代2个32位整数+2个bool)
- 端口(2): src_port, dst_port
- IP层(3): ip_ttl, ip_tos, ip_flags
- TCP(4): tcp_flags_syn, tcp_flags_ack, tcp_flags_psh, tcp_window
- ICMP(1): icmp_type
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw
from tqdm import tqdm

# 配置
SEQUENCE_LENGTH = 50
STRIDE = 50  # 无重叠
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
OUTPUT_DIR = 'data/processed/raw_packet_sequences_50packets_optimized'
RANDOM_SEED = 42

CLASS_NAMES = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan',
               'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

# 优化后的特征名 (19维)
FEATURE_NAMES = [
    # 通用特征 (3)
    'timestamp', 'packet_length', 'protocol',
    # IP地址 - 4字节编码 (8)
    'src_ip_b0', 'src_ip_b1', 'src_ip_b2', 'src_ip_b3',
    'dst_ip_b0', 'dst_ip_b1', 'dst_ip_b2', 'dst_ip_b3',
    # 端口 (2)
    'src_port', 'dst_port',
    # IP层特征 (3)
    'ip_ttl', 'ip_tos', 'ip_flags',
    # TCP特征 (4) - 只保留高信息量的
    'tcp_flags_syn', 'tcp_flags_ack', 'tcp_flags_psh', 'tcp_window',
    # ICMP (1)
    'icmp_type'
]

NUM_FEATURES = len(FEATURE_NAMES)  # 21 (3+8+2+3+4+1)


def ip_to_bytes(ip_str):
    """将IP地址转换为4个字节 (0-255)"""
    try:
        parts = [int(x) for x in ip_str.split('.')]
        return parts  # [b0, b1, b2, b3]
    except:
        return [0, 0, 0, 0]


def extract_packet_features_optimized(pkt, base_time):
    """提取优化后的19维特征"""
    features = {
        # 通用特征
        'timestamp': float(pkt.time) - base_time,
        'packet_length': len(pkt),
        'protocol': 0,  # 0=OTHER, 1=TCP, 2=UDP, 3=ICMP
        # IP地址字节
        'src_ip_b0': 0, 'src_ip_b1': 0, 'src_ip_b2': 0, 'src_ip_b3': 0,
        'dst_ip_b0': 0, 'dst_ip_b1': 0, 'dst_ip_b2': 0, 'dst_ip_b3': 0,
        # 端口
        'src_port': 0, 'dst_port': 0,
        # IP层
        'ip_ttl': 0, 'ip_tos': 0, 'ip_flags': 0,
        # TCP (精简)
        'tcp_flags_syn': 0, 'tcp_flags_ack': 0, 'tcp_flags_psh': 0, 'tcp_window': 0,
        # ICMP
        'icmp_type': 0,
    }

    # IP层
    if IP in pkt:
        src_bytes = ip_to_bytes(pkt[IP].src)
        dst_bytes = ip_to_bytes(pkt[IP].dst)
        features['src_ip_b0'], features['src_ip_b1'] = src_bytes[0], src_bytes[1]
        features['src_ip_b2'], features['src_ip_b3'] = src_bytes[2], src_bytes[3]
        features['dst_ip_b0'], features['dst_ip_b1'] = dst_bytes[0], dst_bytes[1]
        features['dst_ip_b2'], features['dst_ip_b3'] = dst_bytes[2], dst_bytes[3]
        features['ip_ttl'] = pkt[IP].ttl
        features['ip_tos'] = pkt[IP].tos if hasattr(pkt[IP], 'tos') else 0
        features['ip_flags'] = int(pkt[IP].flags) if hasattr(pkt[IP], 'flags') else 0

    # TCP层
    if TCP in pkt:
        features['protocol'] = 1
        features['src_port'] = pkt[TCP].sport
        features['dst_port'] = pkt[TCP].dport
        flags = str(pkt[TCP].flags)
        features['tcp_flags_syn'] = 1 if 'S' in flags else 0
        features['tcp_flags_ack'] = 1 if 'A' in flags else 0
        features['tcp_flags_psh'] = 1 if 'P' in flags else 0
        features['tcp_window'] = pkt[TCP].window
    # UDP层
    elif UDP in pkt:
        features['protocol'] = 2
        features['src_port'] = pkt[UDP].sport
        features['dst_port'] = pkt[UDP].dport
    # ICMP层
    elif ICMP in pkt:
        features['protocol'] = 3
        features['icmp_type'] = pkt[ICMP].type if hasattr(pkt[ICMP], 'type') else 0

    return features


def create_sequences(packet_features, sequence_length, stride):
    """创建序列（无重叠）"""
    sequences = []
    for i in range(0, len(packet_features) - sequence_length + 1, stride):
        seq = packet_features[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32) if sequences else None


def process_pcap(pcap_path, attack_label, base_station, sequence_length, stride):
    """处理单个PCAP文件"""
    print(f"\n  Processing: {os.path.basename(pcap_path)}")
    
    try:
        packets = rdpcap(pcap_path)
        print(f"  Packets: {len(packets):,}")
        
        if len(packets) < sequence_length:
            print(f"  ⚠️ Not enough packets")
            return None, None, None
        
        # 提取特征
        base_time = float(packets[0].time)
        packet_features = []
        for pkt in tqdm(packets, desc="  Extracting", leave=False):
            feat = extract_packet_features_optimized(pkt, base_time)
            packet_features.append(list(feat.values()))
        
        packet_features = np.array(packet_features, dtype=np.float32)
        
        # 构建序列
        sequences = create_sequences(packet_features, sequence_length, stride)
        if sequences is None:
            return None, None, None
        
        n_seqs = len(sequences)
        labels = np.full(n_seqs, attack_label, dtype=np.int64)
        bs_ids = np.full(n_seqs, base_station, dtype=np.int64)
        
        print(f"  Sequences: {n_seqs}")
        return sequences, labels, bs_ids
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None, None


def stratified_split(sequences, labels, base_stations, train_ratio, val_ratio, test_ratio, seed):
    """按(类别, 基站)分层划分"""
    np.random.seed(seed)

    # 创建分组键
    groups = defaultdict(list)
    for i in range(len(sequences)):
        key = (labels[i], base_stations[i])
        groups[key].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for key, indices in groups.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train+n_val])
        test_idx.extend(indices[n_train+n_val:])

    return (
        sequences[train_idx], labels[train_idx],
        sequences[val_idx], labels[val_idx],
        sequences[test_idx], labels[test_idx]
    )


def build_dataset(base_dir, output_dir, sequence_length=50, stride=50,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """构建完整数据集"""

    os.makedirs(output_dir, exist_ok=True)

    # PCAP文件名到类别的映射 (文件名关键字 -> 类别索引)
    # 类别: Benign(0), HTTPFlood(1), ICMPFlood(2), SYNFlood(3), SYNScan(4),
    #       SlowrateDoS(5), TCPConnectScan(6), UDPFlood(7), UDPScan(8)
    file_to_label = {
        'SSH': 0,           # SSH被视为Benign
        'Goldeneye': 1,     # HTTPFlood
        'Torshammer': 1,    # HTTPFlood
        'ICMPflood': 2,     # ICMPFlood
        'SYNflood': 3,      # SYNFlood
        'SYNScan': 4,       # SYNScan
        'Slowloris': 5,     # SlowrateDoS
        'TCPConnect': 6,    # TCPConnectScan
        'UDPflood': 7,      # UDPFlood
        'UDPScan': 8,       # UDPScan
    }

    base_stations = ['BS1_GTP_removed', 'BS2 GTP removed']

    all_sequences, all_labels, all_bs = [], [], []

    for bs_idx, bs_name in enumerate(base_stations):
        bs_path = os.path.join(base_dir, bs_name)
        if not os.path.exists(bs_path):
            print(f"⚠️ Base station not found: {bs_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {bs_name}")
        print('='*60)

        # 获取所有pcapng文件
        pcap_files = [f for f in os.listdir(bs_path) if f.endswith('.pcapng')]
        print(f"Found {len(pcap_files)} PCAP files")

        for pcap_file in pcap_files:
            # 根据文件名确定攻击类型
            attack_label = None
            for key, label in file_to_label.items():
                if key.lower() in pcap_file.lower():
                    attack_label = label
                    attack_name = CLASS_NAMES[label]
                    break

            if attack_label is None:
                print(f"  ⚠️ Unknown attack type: {pcap_file}")
                continue

            pcap_path = os.path.join(bs_path, pcap_file)
            print(f"\n[{attack_name}] {pcap_file}")

            seqs, lbls, bs_ids = process_pcap(
                pcap_path, attack_label, bs_idx, sequence_length, stride
            )

            if seqs is not None:
                all_sequences.append(seqs)
                all_labels.append(lbls)
                all_bs.append(bs_ids)

    # 合并
    print("\n" + "="*60)
    print("Merging all data...")

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_bs = np.concatenate(all_bs, axis=0)

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Shape: {all_sequences.shape}")

    # 分层划分
    print("\nStratified splitting...")
    (train_seq, train_lbl, val_seq, val_lbl, test_seq, test_lbl) = stratified_split(
        all_sequences, all_labels, all_bs,
        train_ratio, val_ratio, test_ratio, RANDOM_SEED
    )

    print(f"  Train: {len(train_seq)}")
    print(f"  Val: {len(val_seq)}")
    print(f"  Test: {len(test_seq)}")

    # 标准化
    print("\nNormalizing...")
    n_train, seq_len, n_feat = train_seq.shape

    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_seq.reshape(-1, n_feat))
    val_flat = scaler.transform(val_seq.reshape(-1, n_feat))
    test_flat = scaler.transform(test_seq.reshape(-1, n_feat))

    train_seq = train_flat.reshape(len(train_seq), seq_len, n_feat)
    val_seq = val_flat.reshape(len(val_seq), seq_len, n_feat)
    test_seq = test_flat.reshape(len(test_seq), seq_len, n_feat)

    # 保存
    print(f"\nSaving to {output_dir}...")
    np.save(os.path.join(output_dir, 'train_sequences.npy'), train_seq)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_lbl)
    np.save(os.path.join(output_dir, 'val_sequences.npy'), val_seq)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_lbl)
    np.save(os.path.join(output_dir, 'test_sequences.npy'), test_seq)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_lbl)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Metadata
    metadata = {
        'sequence_length': sequence_length,
        'stride': stride,
        'num_features': NUM_FEATURES,
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'feature_names': FEATURE_NAMES,
        'feature_description': {
            'timestamp': '相对时间戳',
            'packet_length': '包长度(字节)',
            'protocol': '协议类型(0=OTHER,1=TCP,2=UDP,3=ICMP)',
            'src_ip_b0-b3': '源IP地址4字节(0-255)',
            'dst_ip_b0-b3': '目标IP地址4字节(0-255)',
            'src_port': '源端口',
            'dst_port': '目标端口',
            'ip_ttl': 'IP生存时间',
            'ip_tos': 'IP服务类型',
            'ip_flags': 'IP标志',
            'tcp_flags_syn': 'TCP SYN标志',
            'tcp_flags_ack': 'TCP ACK标志',
            'tcp_flags_psh': 'TCP PSH标志',
            'tcp_window': 'TCP窗口大小',
            'icmp_type': 'ICMP类型'
        },
        'optimization_notes': [
            '移除ip_len, payload_size (与packet_length相关性>0.99)',
            '移除src/dst_ip_is_private (可从IP字节计算)',
            '移除tcp_flags_fin/rst/urg (低方差<0.15)',
            '移除tcp_seq/ack (对分类无意义)',
            '移除icmp_code (几乎常数)',
            'IP地址从32位整数改为4字节编码(范围0-255)'
        ],
        'train_samples': len(train_seq),
        'val_samples': len(val_seq),
        'test_samples': len(test_seq),
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # 打印统计
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Features: {NUM_FEATURES} (optimized from 25)")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"\nClass distribution (Train):")
    for i, name in enumerate(CLASS_NAMES):
        cnt = (train_lbl == i).sum()
        print(f"  {name:<15}: {cnt:>6} ({cnt/len(train_lbl)*100:>5.2f}%)")

    # 验证特征统计
    print(f"\nFeature statistics (after normalization):")
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10}")
    print("-"*42)
    for i, name in enumerate(FEATURE_NAMES):
        mean = scaler.mean_[i]
        std = scaler.scale_[i]
        print(f"{name:<20} {mean:>10.2f} {std:>10.2f}")

    print(f"\n✓ Dataset saved to {output_dir}")
    return metadata


if __name__ == '__main__':
    print("="*80)
    print("Building Optimized Feature Dataset (19 features)".center(80))
    print("="*80)

    BASE_DIR = 'data/5G-NIDD-raw'

    print(f"\nConfiguration:")
    print(f"  Input: {BASE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Stride: {STRIDE} (no overlap)")
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Split: {TRAIN_RATIO*100:.0f}%/{VAL_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%")

    metadata = build_dataset(
        BASE_DIR, OUTPUT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

