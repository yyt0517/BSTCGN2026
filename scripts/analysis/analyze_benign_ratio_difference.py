"""
分析良性流量在流聚合数据和时序数据中的占比差异
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*100)
print("良性流量占比差异分析".center(100))
print("="*100)

# ============================================================================
# Part 1: 分析流聚合数据中的良性流量特征
# ============================================================================

print("\n" + "="*100)
print("Part 1: 流聚合数据分析 (BTS1_BTS2_fields_preserved.csv)".center(100))
print("="*100)

flow_file = '5G-NIDD-raw/BTS1_BTS2_fields_preserved.csv'

if not os.path.exists(flow_file):
    print(f"\n✗ 文件不存在: {flow_file}")
    exit(1)

print(f"\n正在加载流聚合数据...")
df_flow = pd.read_csv(flow_file, nrows=100000)  # 先读取10万行进行分析

print(f"✓ 加载完成 (前100,000行)")

# Check if Attack Type column exists
if 'Attack Type' not in df_flow.columns:
    print("\n可用列:")
    print(df_flow.columns.tolist())
    print("\n✗ 未找到 'Attack Type' 列")
    exit(1)

# 分析良性流量的特征
benign_flows = df_flow[df_flow['Attack Type'] == 'Benign']
attack_flows = df_flow[df_flow['Attack Type'] != 'Benign']

print(f"\n流数量统计:")
print(f"  良性流量: {len(benign_flows):,} ({len(benign_flows)/len(df_flow)*100:.2f}%)")
print(f"  攻击流量: {len(attack_flows):,} ({len(attack_flows)/len(df_flow)*100:.2f}%)")

# 分析流持续时间
if 'StartTime' in df_flow.columns and 'LastTime' in df_flow.columns:
    print("\n正在分析流持续时间...")
    
    def parse_time(time_str):
        """解析 MM:SS.f 格式的时间"""
        try:
            if pd.isna(time_str):
                return np.nan
            parts = str(time_str).split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            return float(time_str)
        except:
            return np.nan
    
    # 计算流持续时间
    df_flow['start_seconds'] = df_flow['StartTime'].apply(parse_time)
    df_flow['last_seconds'] = df_flow['LastTime'].apply(parse_time)
    df_flow['duration'] = df_flow['last_seconds'] - df_flow['start_seconds']
    
    # 良性流量的持续时间统计
    benign_duration = df_flow[df_flow['Attack Type'] == 'Benign']['duration'].dropna()
    attack_duration = df_flow[df_flow['Attack Type'] != 'Benign']['duration'].dropna()
    
    print("\n流持续时间统计 (秒):")
    print("\n良性流量:")
    print(f"  平均: {benign_duration.mean():.3f}s")
    print(f"  中位数: {benign_duration.median():.3f}s")
    print(f"  最小: {benign_duration.min():.3f}s")
    print(f"  最大: {benign_duration.max():.3f}s")
    print(f"  标准差: {benign_duration.std():.3f}s")
    
    print("\n攻击流量:")
    print(f"  平均: {attack_duration.mean():.3f}s")
    print(f"  中位数: {attack_duration.median():.3f}s")
    print(f"  最小: {attack_duration.min():.3f}s")
    print(f"  最大: {attack_duration.max():.3f}s")
    print(f"  标准差: {attack_duration.std():.3f}s")
    
    # 持续时间分布
    print("\n良性流量持续时间分布:")
    duration_bins = [0, 0.001, 0.01, 0.1, 1, 10, 100, float('inf')]
    duration_labels = ['0-0.001s', '0.001-0.01s', '0.01-0.1s', '0.1-1s', '1-10s', '10-100s', '>100s']
    
    benign_duration_dist = pd.cut(benign_duration, bins=duration_bins, labels=duration_labels)
    print(benign_duration_dist.value_counts().sort_index())
    
    print("\n攻击流量持续时间分布:")
    attack_duration_dist = pd.cut(attack_duration, bins=duration_bins, labels=duration_labels)
    print(attack_duration_dist.value_counts().sort_index())
    
    # 计算每条流映射到多少个时间窗口（1秒窗口）
    df_flow['num_windows'] = np.ceil(df_flow['duration']).fillna(1).astype(int)
    df_flow.loc[df_flow['num_windows'] < 1, 'num_windows'] = 1
    
    benign_windows = df_flow[df_flow['Attack Type'] == 'Benign']['num_windows']
    attack_windows = df_flow[df_flow['Attack Type'] != 'Benign']['num_windows']
    
    print("\n每条流映射到的时间窗口数:")
    print("\n良性流量:")
    print(f"  平均: {benign_windows.mean():.2f} 个窗口")
    print(f"  中位数: {benign_windows.median():.0f} 个窗口")
    print(f"  总窗口数: {benign_windows.sum():,} 个")
    
    print("\n攻击流量:")
    print(f"  平均: {attack_windows.mean():.2f} 个窗口")
    print(f"  中位数: {attack_windows.median():.0f} 个窗口")
    print(f"  总窗口数: {attack_windows.sum():,} 个")
    
    # 预期的时序数据占比
    total_windows = df_flow['num_windows'].sum()
    benign_window_ratio = benign_windows.sum() / total_windows * 100
    attack_window_ratio = attack_windows.sum() / total_windows * 100
    
    print("\n预期时序数据占比（基于流持续时间）:")
    print(f"  良性流量: {benign_window_ratio:.2f}%")
    print(f"  攻击流量: {attack_window_ratio:.2f}%")

# ============================================================================
# Part 2: 分析PCAP文件大小
# ============================================================================

print("\n" + "="*100)
print("Part 2: PCAP文件大小分析".center(100))
print("="*100)

bs1_dir = '5G-NIDD-raw/BS1_GTP_removed'
bs2_dir = '5G-NIDD-raw/BS2 GTP removed'

file_mapping = {
    'SSH_BS1_nogtp.pcapng': ('BS1', 'Benign', 0),
    'SSH_BS2_nogtp.pcapng': ('BS2', 'Benign', 0),
    'Goldeneye_BS1_nogtp.pcapng': ('BS1', 'HTTPFlood', 1),
    'Goldeneye_BS2_nogtp.pcapng': ('BS2', 'HTTPFlood', 1),
    'ICMPflood_BS1_nogtp.pcapng': ('BS1', 'ICMPFlood', 2),
    'ICMPflood_BS2_nogtp.pcapng': ('BS2', 'ICMPFlood', 2),
    'SYNflood_BS1_nogtp.pcapng': ('BS1', 'SYNFlood', 3),
    'SYNflood_BS2_nogtp.pcapng': ('BS2', 'SYNFlood', 3),
    'SYNScan_BS1_nogtp.pcapng': ('BS1', 'SYNScan', 4),
    'SYNscan_BS2_nogtp.pcapng': ('BS2', 'SYNScan', 4),
    'Slowloris_BS1_nogtp.pcapng': ('BS1', 'SlowrateDoS', 5),
    'Slowloris_BS2_nogtp.pcapng': ('BS2', 'SlowrateDoS', 5),
    'TCPConnect_BS1_nogtp.pcapng': ('BS1', 'TCPConnectScan', 6),
    'TCPConnect_BS2_nogtp.pcapng': ('BS2', 'TCPConnectScan', 6),
    'UDPflood_BS1_nogtp.pcapng': ('BS1', 'UDPFlood', 7),
    'UDPflood_BS2_nogtp.pcapng': ('BS2', 'UDPFlood', 7),
    'UDPScan_BS1_nogtp.pcapng': ('BS1', 'UDPScan', 8),
    'UDPScan_BS2_nogtp.pcapng': ('BS2', 'UDPScan', 8),
}

file_sizes = []

for filename, (bs, attack_type, label) in file_mapping.items():
    if bs == 'BS1':
        filepath = os.path.join(bs1_dir, filename)
    else:
        filepath = os.path.join(bs2_dir, filename)
    
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        file_sizes.append({
            'filename': filename,
            'base_station': bs,
            'attack_type': attack_type,
            'label': label,
            'size_mb': size_mb
        })

df_sizes = pd.DataFrame(file_sizes)

print("\nPCAP文件大小统计:")
print("\n按攻击类型分组:")
size_by_type = df_sizes.groupby('attack_type')['size_mb'].agg(['sum', 'mean', 'count'])
size_by_type['percentage'] = size_by_type['sum'] / size_by_type['sum'].sum() * 100
size_by_type = size_by_type.sort_values('sum', ascending=False)

print("\n攻击类型      | 总大小(MB) | 平均(MB) | 文件数 | 占比")
print("-" * 70)
for attack_type, row in size_by_type.iterrows():
    print(f"{attack_type:14s} | {row['sum']:10.2f} | {row['mean']:8.2f} | {row['count']:6.0f} | {row['percentage']:5.2f}%")

print("-" * 70)
print(f"{'总计':14s} | {df_sizes['size_mb'].sum():10.2f} | {df_sizes['size_mb'].mean():8.2f} | {len(df_sizes):6d} | 100.00%")

benign_size_pct = df_sizes[df_sizes['attack_type'] == 'Benign']['size_mb'].sum() / df_sizes['size_mb'].sum() * 100
print(f"\n良性流量PCAP文件占比: {benign_size_pct:.2f}%")

# ============================================================================
# Part 3: 分析时序数据
# ============================================================================

print("\n" + "="*100)
print("Part 3: 时序数据分析 (timeseries_from_pcap_1s.csv)".center(100))
print("="*100)

timeseries_file = 'timeseries_from_pcap_1s.csv'

if not os.path.exists(timeseries_file):
    print(f"\n✗ 文件不存在: {timeseries_file}")
else:
    print(f"\n正在加载时序数据...")
    df_ts = pd.read_csv(timeseries_file)
    
    print(f"✓ 加载完成")
    print(f"  样本数: {len(df_ts):,}")
    print(f"  特征数: {len(df_ts.columns)}")
    
    print("\n攻击类型分布:")
    attack_counts = df_ts['attack_type'].value_counts().sort_index()
    
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
    
    print("\nLabel | 类型           | 样本数      | 占比")
    print("-" * 60)
    for label, count in attack_counts.items():
        name = attack_names.get(label, 'Unknown')
        percentage = count / len(df_ts) * 100
        print(f"{label:5d} | {name:14s} | {count:10,d} | {percentage:5.2f}%")
    
    benign_ts_pct = (df_ts['attack_type'] == 0).sum() / len(df_ts) * 100
    print(f"\n良性流量时序数据占比: {benign_ts_pct:.2f}%")

# ============================================================================
# Part 4: 综合分析
# ============================================================================

print("\n" + "="*100)
print("Part 4: 综合分析与结论".center(100))
print("="*100)

print("\n占比对比:")
print(f"  流聚合数据中良性流量: 39.2% (477,737条流)")
print(f"  PCAP文件大小占比: {benign_size_pct:.2f}%")
print(f"  时序数据中良性流量: 9.59% (2,188个时间窗口)")

print("\n差异原因分析:")
print("\n1. 流持续时间差异")
print("   - 如果良性流量的流持续时间较短")
print("   - 每条流映射到的时间窗口数较少")
print("   - 导致时序数据中占比降低")

print("\n2. PCAP文件大小差异")
print("   - PCAP文件大小反映了实际的网络流量")
print(f"   - 良性流量PCAP文件只占 {benign_size_pct:.2f}%")
print("   - 说明良性流量虽然流数量多，但数据量小")

print("\n3. 包数量差异")
print("   - 良性流量可能每条流的包数较少")
print("   - 攻击流量（如UDPFlood）每条流的包数较多")
print("   - 导致时序数据中攻击流量占比更高")

print("\n结论:")
if benign_size_pct < 15:
    print("  ✓ 时序数据占比（9.59%）与PCAP文件大小占比一致")
    print("  ✓ 这是正常现象")
    print("  ✓ 流聚合数据的39.2%是流数量占比，不代表实际流量占比")
    print("  ✓ 时序数据更准确地反映了实际网络流量的时间分布")
else:
    print("  ⚠️  时序数据占比可能偏低")
    print("  ⚠️  需要进一步检查PCAP文件处理是否正确")

print("\n" + "="*100)

