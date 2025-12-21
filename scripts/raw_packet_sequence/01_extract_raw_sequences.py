"""
从PCAP文件提取原始包级别的时序数据
不进行时间窗口聚合，保留每个包的原始特征
用于时序预测模型
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("Raw Packet Sequence Extraction from PCAP".center(100))
print("="*100)

# Check if scapy is available
try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw
    print("\n✓ Scapy is available")
except ImportError:
    print("\n✗ Scapy is not installed. Please install it:")
    print("  conda install -c conda-forge scapy")
    exit(1)

# Attack type mapping
ATTACK_MAPPING = {
    'SSH': 0,            # Benign
    'Goldeneye': 1,      # HTTPFlood
    'ICMPflood': 2,      # ICMPFlood
    'SYNflood': 3,       # SYNFlood
    'SYNScan': 4,        # SYNScan
    'Slowloris': 5,      # SlowrateDoS
    'TCPConnect': 6,     # TCPConnectScan
    'UDPflood': 7,       # UDPFlood
    'UDPScan': 8,        # UDPScan
}

CLASS_NAMES = [
    'Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 
    'SYNScan', 'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan'
]

def ip_to_int(ip_str):
    """将IP地址转换为32位整数"""
    try:
        parts = [int(x) for x in ip_str.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except:
        return 0

def is_private_ip(ip_str):
    """判断是否为内网IP"""
    try:
        parts = [int(x) for x in ip_str.split('.')]
        if parts[0] == 10:
            return 1
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return 1
        if parts[0] == 192 and parts[1] == 168:
            return 1
        return 0
    except:
        return 0

def encode_ip_flags(flags):
    """编码IP标志"""
    flags_str = str(flags)
    if 'DF' in flags_str and 'MF' in flags_str:
        return 3  # DF+MF
    elif 'DF' in flags_str:
        return 1  # DF only
    elif 'MF' in flags_str:
        return 2  # MF only
    else:
        return 0  # No flags

def extract_packet_features(pkt, base_time):
    """从单个数据包中提取特征（25个核心特征）"""
    features = {
        # 通用特征 (3个)
        'timestamp': float(pkt.time) - base_time,
        'packet_length': len(pkt),
        'protocol': 0,  # 0=OTHER, 1=TCP, 2=UDP, 3=ICMP

        # IP地址特征 (4个)
        'src_ip_int': 0,
        'dst_ip_int': 0,
        'src_ip_is_private': 0,
        'dst_ip_is_private': 0,

        # 端口特征 (2个)
        'src_port': 0,
        'dst_port': 0,

        # IP层特征 (4个)
        'ip_ttl': 0,
        'ip_len': 0,
        'ip_tos': 0,
        'ip_flags': 0,

        # TCP标志特征 (6个)
        'tcp_flags_syn': 0,
        'tcp_flags_ack': 0,
        'tcp_flags_fin': 0,
        'tcp_flags_rst': 0,
        'tcp_flags_psh': 0,
        'tcp_flags_urg': 0,

        # TCP窗口和序列号 (3个)
        'tcp_window': 0,
        'tcp_seq': 0,
        'tcp_ack': 0,

        # 载荷大小 (1个)
        'payload_size': 0,

        # ICMP特征 (2个)
        'icmp_type': 0,
        'icmp_code': 0,
    }

    # IP layer
    if IP in pkt:
        # IP地址
        features['src_ip_int'] = ip_to_int(pkt[IP].src)
        features['dst_ip_int'] = ip_to_int(pkt[IP].dst)
        features['src_ip_is_private'] = is_private_ip(pkt[IP].src)
        features['dst_ip_is_private'] = is_private_ip(pkt[IP].dst)

        # IP层特征
        features['ip_ttl'] = pkt[IP].ttl
        features['ip_len'] = pkt[IP].len
        features['ip_tos'] = pkt[IP].tos if hasattr(pkt[IP], 'tos') else 0
        features['ip_flags'] = encode_ip_flags(pkt[IP].flags if hasattr(pkt[IP], 'flags') else '')

    # TCP layer
    if TCP in pkt:
        features['protocol'] = 1
        features['src_port'] = pkt[TCP].sport
        features['dst_port'] = pkt[TCP].dport

        # TCP标志
        flags = str(pkt[TCP].flags)
        features['tcp_flags_syn'] = 1 if 'S' in flags else 0
        features['tcp_flags_ack'] = 1 if 'A' in flags else 0
        features['tcp_flags_fin'] = 1 if 'F' in flags else 0
        features['tcp_flags_rst'] = 1 if 'R' in flags else 0
        features['tcp_flags_psh'] = 1 if 'P' in flags else 0
        features['tcp_flags_urg'] = 1 if 'U' in flags else 0

        # TCP窗口和序列号
        features['tcp_window'] = pkt[TCP].window
        features['tcp_seq'] = pkt[TCP].seq % 1000000  # 取模减小范围
        features['tcp_ack'] = pkt[TCP].ack % 1000000  # 取模减小范围

        # 载荷大小
        if Raw in pkt:
            features['payload_size'] = len(pkt[Raw].load)

    # UDP layer
    elif UDP in pkt:
        features['protocol'] = 2
        features['src_port'] = pkt[UDP].sport
        features['dst_port'] = pkt[UDP].dport
        if Raw in pkt:
            features['payload_size'] = len(pkt[Raw].load)

    # ICMP layer
    elif ICMP in pkt:
        features['protocol'] = 3
        features['icmp_type'] = pkt[ICMP].type if hasattr(pkt[ICMP], 'type') else 0
        features['icmp_code'] = pkt[ICMP].code if hasattr(pkt[ICMP], 'code') else 0

    return features

def process_pcap_to_sequences(pcap_path, attack_label, max_packets=None, 
                               sequence_length=100, stride=50):
    """
    处理单个PCAP文件，提取固定长度的包序列
    
    Args:
        pcap_path: PCAP文件路径
        attack_label: 攻击类型标签
        max_packets: 最大读取包数量
        sequence_length: 序列长度（包数量）
        stride: 滑动窗口步长
    
    Returns:
        sequences: (N, sequence_length, num_features)
        labels: (N,)
    """
    print(f"\n  Reading PCAP: {os.path.basename(pcap_path)}")
    
    try:
        # Read packets
        if max_packets:
            packets = rdpcap(pcap_path, count=max_packets)
        else:
            packets = rdpcap(pcap_path)
        
        print(f"  Total packets: {len(packets):,}")
        
        if len(packets) < sequence_length:
            print(f"  ⚠️  Not enough packets (need {sequence_length})")
            return None, None
        
        # Extract features from each packet
        base_time = float(packets[0].time)
        packet_features = []
        
        for pkt in tqdm(packets, desc="  Extracting", leave=False):
            features = extract_packet_features(pkt, base_time)
            packet_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(packet_features)
        
        # Create sequences using sliding window
        sequences = []
        labels = []
        
        for i in range(0, len(df) - sequence_length + 1, stride):
            seq = df.iloc[i:i+sequence_length].values
            sequences.append(seq)
            labels.append(attack_label)
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"  Created {len(sequences):,} sequences")
        print(f"  Sequence shape: {sequences.shape}")
        
        return sequences, labels
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None

def build_raw_sequence_dataset(base_dir, output_dir, sequence_length=100, stride=50,
                                use_gtp_removed=True, max_packets_per_file=None):
    """构造原始包序列数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which directory to use
    if use_gtp_removed:
        bs1_dir = os.path.join(base_dir, 'BS1_GTP_removed')
        bs2_dir = os.path.join(base_dir, 'BS2 GTP removed')
        print("\n📁 Using GTP-removed PCAP files")
    else:
        bs1_dir = os.path.join(base_dir, 'BS1')
        bs2_dir = os.path.join(base_dir, 'BS2')
        print("\n📁 Using original PCAP files")
    
    # Define attack files
    attack_files_bs1 = {
        'SSH_BS1_nogtp.pcapng' if use_gtp_removed else 'SSH_BS1.pcapng': 0,
        'Goldeneye_BS1_nogtp.pcapng' if use_gtp_removed else 'Goldeneye_BS1.pcapng': 1,
        'ICMPflood_BS1_nogtp.pcapng' if use_gtp_removed else 'ICMPflood_BS1.pcapng': 2,
        'SYNflood_BS1_nogtp.pcapng' if use_gtp_removed else 'SYNflood_BS1.pcapng': 3,
        'SYNScan_BS1_nogtp.pcapng' if use_gtp_removed else 'SYNScan_BS1.pcapng': 4,
        'Slowloris_BS1_nogtp.pcapng' if use_gtp_removed else 'Slowloris_BS1.pcapng': 5,
        'TCPConnect_BS1_nogtp.pcapng' if use_gtp_removed else 'TCPConnect_BS1.pcapng': 6,
        'UDPflood_BS1_nogtp.pcapng' if use_gtp_removed else 'UDPflood_BS1.pcapng': 7,
        'UDPScan_BS1_nogtp.pcapng' if use_gtp_removed else 'UDPScan_BS1.pcapng': 8,
    }
    
    attack_files_bs2 = {
        'SSH_BS2_nogtp.pcapng' if use_gtp_removed else 'SSH_BS2.pcapng': 0,
        'Goldeneye_BS2_nogtp.pcapng' if use_gtp_removed else 'Goldeneye_BS2.pcapng': 1,
        'ICMPflood_BS2_nogtp.pcapng' if use_gtp_removed else 'ICMPflood_BS2.pcapng': 2,
        'SYNflood_BS2_nogtp.pcapng' if use_gtp_removed else 'SYNflood_BS2.pcapng': 3,
        'SYNscan_BS2_nogtp.pcapng' if use_gtp_removed else 'SYNScan_BS2.pcapng': 4,
        'Slowloris_BS2_nogtp.pcapng' if use_gtp_removed else 'Slowloris_BS2.pcapng': 5,
        'TCPConnect_BS2_nogtp.pcapng' if use_gtp_removed else 'TCPConnect_BS2.pcapng': 6,
        'UDPflood_BS2_nogtp.pcapng' if use_gtp_removed else 'UDPflood_BS2.pcapng': 7,
        'UDPScan_BS2_nogtp.pcapng' if use_gtp_removed else 'UDPScan_BS2.pcapng': 8,
    }
    
    all_sequences = []
    all_labels = []
    
    # Process BS1 files
    print("\n" + "="*100)
    print("Processing BS1 Files".center(100))
    print("="*100)
    
    for filename, attack_label in attack_files_bs1.items():
        pcap_path = os.path.join(bs1_dir, filename)
        
        if not os.path.exists(pcap_path):
            print(f"\n⚠️  File not found: {filename}")
            continue
        
        print(f"\n{'='*100}")
        print(f"Processing: {filename} (Label: {attack_label} - {CLASS_NAMES[attack_label]})".center(100))
        print(f"{'='*100}")
        
        sequences, labels = process_pcap_to_sequences(
            pcap_path, attack_label, max_packets_per_file, 
            sequence_length, stride
        )
        
        if sequences is not None:
            all_sequences.append(sequences)
            all_labels.append(labels)
            print(f"  ✓ Added {len(sequences):,} sequences")
    
    # Process BS2 files
    print("\n" + "="*100)
    print("Processing BS2 Files".center(100))
    print("="*100)
    
    for filename, attack_label in attack_files_bs2.items():
        pcap_path = os.path.join(bs2_dir, filename)
        
        if not os.path.exists(pcap_path):
            print(f"\n⚠️  File not found: {filename}")
            continue
        
        print(f"\n{'='*100}")
        print(f"Processing: {filename} (Label: {attack_label} - {CLASS_NAMES[attack_label]})".center(100))
        print(f"{'='*100}")
        
        sequences, labels = process_pcap_to_sequences(
            pcap_path, attack_label, max_packets_per_file,
            sequence_length, stride
        )
        
        if sequences is not None:
            all_sequences.append(sequences)
            all_labels.append(labels)
            print(f"  ✓ Added {len(sequences):,} sequences")
    
    # Combine all sequences
    if len(all_sequences) == 0:
        print("\n✗ No data extracted!")
        return
    
    print("\n" + "="*100)
    print("Combining All Sequences".center(100))
    print("="*100)
    
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal sequences: {len(X):,}")
    print(f"Sequence shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"\nLabel distribution:")
    for label in range(9):
        count = (y == label).sum()
        print(f"  {CLASS_NAMES[label]:20s}: {count:,} ({count/len(y)*100:.2f}%)")
    
    # Save data
    np.save(os.path.join(output_dir, 'sequences.npy'), X)
    np.save(os.path.join(output_dir, 'labels.npy'), y)
    
    # Save metadata
    metadata = {
        'sequence_length': sequence_length,
        'stride': stride,
        'num_features': X.shape[2],
        'num_classes': 9,
        'class_names': CLASS_NAMES,
        'feature_names': [
            # 通用特征 (3个)
            'timestamp', 'packet_length', 'protocol',
            # IP地址特征 (4个)
            'src_ip_int', 'dst_ip_int', 'src_ip_is_private', 'dst_ip_is_private',
            # 端口特征 (2个)
            'src_port', 'dst_port',
            # IP层特征 (4个)
            'ip_ttl', 'ip_len', 'ip_tos', 'ip_flags',
            # TCP标志特征 (6个)
            'tcp_flags_syn', 'tcp_flags_ack', 'tcp_flags_fin', 'tcp_flags_rst', 'tcp_flags_psh', 'tcp_flags_urg',
            # TCP窗口和序列号 (3个)
            'tcp_window', 'tcp_seq', 'tcp_ack',
            # 载荷大小 (1个)
            'payload_size',
            # ICMP特征 (2个)
            'icmp_type', 'icmp_code'
        ]
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Data saved to: {output_dir}")
    print(f"  - sequences.npy: {X.nbytes / (1024**2):.2f} MB")
    print(f"  - labels.npy: {y.nbytes / (1024**2):.2f} MB")
    print(f"  - metadata.pkl")

# Main execution
if __name__ == "__main__":
    BASE_DIR = '5G-NIDD-raw'
    OUTPUT_DIR = 'data/processed/raw_packet_sequences'
    SEQUENCE_LENGTH = 100  # 每个序列包含100个包
    STRIDE = 50  # 滑动窗口步长50个包
    USE_GTP_REMOVED = True
    MAX_PACKETS = 50000  # 每个文件最多读取50000个包（测试用）
    
    print(f"\nConfiguration:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sequence length: {SEQUENCE_LENGTH} packets")
    print(f"  Stride: {STRIDE} packets")
    print(f"  Use GTP-removed: {USE_GTP_REMOVED}")
    print(f"  Max packets per file: {MAX_PACKETS if MAX_PACKETS else 'Unlimited'}")
    
    # Build dataset
    build_raw_sequence_dataset(
        BASE_DIR,
        OUTPUT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        use_gtp_removed=USE_GTP_REMOVED,
        max_packets_per_file=MAX_PACKETS
    )
    
    print("\n" + "="*100)
    print("SUCCESS! Raw Packet Sequence Dataset Created".center(100))
    print("="*100)

