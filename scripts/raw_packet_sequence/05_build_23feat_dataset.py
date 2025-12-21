"""
构建23维特征数据集 (恢复tcp_seq/tcp_ack)

在21维优化特征基础上恢复tcp_seq和tcp_ack特征，测试是否能提升SYNFlood等攻击的分类。

特征 (23维):
- 通用(3): timestamp, packet_length, protocol
- IP地址(8): src_ip_b0-b3, dst_ip_b0-b3 
- 端口(2): src_port, dst_port
- IP层(3): ip_ttl, ip_tos, ip_flags
- TCP(6): tcp_flags_syn, tcp_flags_ack, tcp_flags_psh, tcp_window, tcp_seq, tcp_ack [+2]
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
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
from tqdm import tqdm

# 配置
SEQUENCE_LENGTH = 50
STRIDE = 50  # 无重叠
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
OUTPUT_DIR = 'data/processed/raw_packet_sequences_50packets_23feat'
RANDOM_SEED = 42

CLASS_NAMES = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan',
               'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

FEATURE_NAMES = [
    'timestamp', 'packet_length', 'protocol',
    'src_ip_b0', 'src_ip_b1', 'src_ip_b2', 'src_ip_b3',
    'dst_ip_b0', 'dst_ip_b1', 'dst_ip_b2', 'dst_ip_b3',
    'src_port', 'dst_port',
    'ip_ttl', 'ip_tos', 'ip_flags',
    'tcp_flags_syn', 'tcp_flags_ack', 'tcp_flags_psh', 'tcp_window',
    'tcp_seq', 'tcp_ack',  # 恢复这两个特征
    'icmp_type'
]

NUM_FEATURES = len(FEATURE_NAMES)  # 23


def ip_to_bytes(ip_str):
    """将IP地址转换为4个字节 (0-255)"""
    try:
        parts = [int(x) for x in ip_str.split('.')]
        return parts
    except:
        return [0, 0, 0, 0]


def extract_packet_features(pkt, base_time):
    """提取23维特征"""
    features = {
        'timestamp': float(pkt.time) - base_time,
        'packet_length': len(pkt),
        'protocol': 0,
        'src_ip_b0': 0, 'src_ip_b1': 0, 'src_ip_b2': 0, 'src_ip_b3': 0,
        'dst_ip_b0': 0, 'dst_ip_b1': 0, 'dst_ip_b2': 0, 'dst_ip_b3': 0,
        'src_port': 0, 'dst_port': 0,
        'ip_ttl': 0, 'ip_tos': 0, 'ip_flags': 0,
        'tcp_flags_syn': 0, 'tcp_flags_ack': 0, 'tcp_flags_psh': 0, 'tcp_window': 0,
        'tcp_seq': 0, 'tcp_ack': 0,  # 新增
        'icmp_type': 0,
    }

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

    if TCP in pkt:
        features['protocol'] = 1
        features['src_port'] = pkt[TCP].sport
        features['dst_port'] = pkt[TCP].dport
        flags = str(pkt[TCP].flags)
        features['tcp_flags_syn'] = 1 if 'S' in flags else 0
        features['tcp_flags_ack'] = 1 if 'A' in flags else 0
        features['tcp_flags_psh'] = 1 if 'P' in flags else 0
        features['tcp_window'] = pkt[TCP].window
        # 恢复tcp_seq和tcp_ack (取模以限制范围)
        features['tcp_seq'] = pkt[TCP].seq % 1000000
        features['tcp_ack'] = pkt[TCP].ack % 1000000
    elif UDP in pkt:
        features['protocol'] = 2
        features['src_port'] = pkt[UDP].sport
        features['dst_port'] = pkt[UDP].dport
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
            print(f"  Not enough packets")
            return None, None, None
        
        base_time = float(packets[0].time)
        packet_features = []
        
        for pkt in packets:
            if IP in pkt:
                feat = extract_packet_features(pkt, base_time)
                packet_features.append([feat[name] for name in FEATURE_NAMES])
        
        if len(packet_features) < sequence_length:
            return None, None, None
        
        packet_features = np.array(packet_features, dtype=np.float32)
        sequences = create_sequences(packet_features, sequence_length, stride)
        
        if sequences is None or len(sequences) == 0:
            return None, None, None
        
        labels = np.full(len(sequences), attack_label, dtype=np.int64)
        bs_tags = np.full(len(sequences), base_station, dtype=np.int64)
        
        print(f"  Sequences: {len(sequences)}")
        return sequences, labels, bs_tags
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None


def stratified_split(sequences, labels, base_stations, train_ratio, val_ratio, test_ratio, seed):
    """按(类别, 基站)分层划分"""
    np.random.seed(seed)
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

    return (sequences[train_idx], labels[train_idx],
            sequences[val_idx], labels[val_idx],
            sequences[test_idx], labels[test_idx])


def build_dataset():
    """构建完整数据集"""
    base_dir = 'data/5G-NIDD-raw'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_to_label = {
        'SSH': 0, 'Goldeneye': 1, 'Torshammer': 1, 'ICMPflood': 2,
        'SYNflood': 3, 'SYNScan': 4, 'Slowloris': 5,
        'TCPConnect': 6, 'UDPflood': 7, 'UDPScan': 8,
    }
    base_stations = ['BS1_GTP_removed', 'BS2 GTP removed']

    all_sequences, all_labels, all_bs = [], [], []

    for bs_idx, bs_name in enumerate(base_stations):
        bs_path = os.path.join(base_dir, bs_name)
        if not os.path.exists(bs_path):
            print(f"Base station not found: {bs_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {bs_name}")
        print('='*60)

        pcap_files = [f for f in os.listdir(bs_path) if f.endswith('.pcapng')]
        print(f"Found {len(pcap_files)} PCAP files")

        for pcap_file in pcap_files:
            attack_label = None
            for key, label in file_to_label.items():
                if key.lower() in pcap_file.lower():
                    attack_label = label
                    break

            if attack_label is None:
                print(f"  Unknown attack type: {pcap_file}")
                continue

            pcap_path = os.path.join(bs_path, pcap_file)
            print(f"\n[{CLASS_NAMES[attack_label]}] {pcap_file}")

            seqs, lbls, bs_ids = process_pcap(
                pcap_path, attack_label, bs_idx, SEQUENCE_LENGTH, STRIDE
            )

            if seqs is not None:
                all_sequences.append(seqs)
                all_labels.append(lbls)
                all_bs.append(bs_ids)

    print("\n" + "="*60)
    print("Merging all data...")

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_bs = np.concatenate(all_bs, axis=0)

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Shape: {all_sequences.shape}")

    print("\nStratified splitting...")
    (train_seq, train_lbl, val_seq, val_lbl, test_seq, test_lbl) = stratified_split(
        all_sequences, all_labels, all_bs,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    print(f"  Train: {len(train_seq)}")
    print(f"  Val: {len(val_seq)}")
    print(f"  Test: {len(test_seq)}")

    print("\nNormalizing...")
    n_train, seq_len, n_feat = train_seq.shape
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_seq.reshape(-1, n_feat))
    val_flat = scaler.transform(val_seq.reshape(-1, n_feat))
    test_flat = scaler.transform(test_seq.reshape(-1, n_feat))

    train_seq = train_flat.reshape(len(train_seq), seq_len, n_feat)
    val_seq = val_flat.reshape(len(val_seq), seq_len, n_feat)
    test_seq = test_flat.reshape(len(test_seq), seq_len, n_feat)

    print(f"\nSaving to {OUTPUT_DIR}...")
    np.save(os.path.join(OUTPUT_DIR, 'train_sequences.npy'), train_seq)
    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_lbl)
    np.save(os.path.join(OUTPUT_DIR, 'val_sequences.npy'), val_seq)
    np.save(os.path.join(OUTPUT_DIR, 'val_labels.npy'), val_lbl)
    np.save(os.path.join(OUTPUT_DIR, 'test_sequences.npy'), test_seq)
    np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), test_lbl)

    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'stride': STRIDE,
        'num_features': NUM_FEATURES,
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'feature_names': FEATURE_NAMES,
        'train_samples': len(train_seq),
        'val_samples': len(val_seq),
        'test_samples': len(test_seq),
    }
    with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("\nDone!")
    print(f"Features: {NUM_FEATURES}")
    print(f"Dataset saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    build_dataset()

