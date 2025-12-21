"""比较原始25维特征和优化后21维特征"""
import numpy as np
import pickle
from pathlib import Path

# 加载两个数据集进行对比
old_dir = Path('data/processed/raw_packet_sequences_50packets_no_leak')
new_dir = Path('data/processed/raw_packet_sequences_50packets_optimized')

old_seq = np.load(old_dir / 'train_sequences.npy')
new_seq = np.load(new_dir / 'train_sequences.npy')

with open(old_dir / 'scaler.pkl', 'rb') as f:
    old_scaler = pickle.load(f)
with open(new_dir / 'scaler.pkl', 'rb') as f:
    new_scaler = pickle.load(f)
with open(old_dir / 'metadata.pkl', 'rb') as f:
    old_meta = pickle.load(f)
with open(new_dir / 'metadata.pkl', 'rb') as f:
    new_meta = pickle.load(f)

print('='*80)
print('特征对比: 25维 vs 21维')
print('='*80)

print(f'\n旧数据集: {old_seq.shape}')
print(f'新数据集: {new_seq.shape}')

print('\n【移除的特征】(25→21)')
old_feats = set(old_meta['feature_names'])
new_feats = set(new_meta['feature_names'])
removed = old_feats - new_feats
added = new_feats - old_feats
print(f'  移除: {sorted(removed)}')
print(f'  新增: {sorted(added)}')

print('\n【IP地址编码改进】')
print('  旧: src_ip_int (方差=864M), dst_ip_int (方差=544M)')
print('  新: src_ip_b0~b3, dst_ip_b0~b3 (每字节范围0-255)')

print('\n【新特征的原始统计】')
header = f"{'特征':<20} {'均值':>12} {'标准差':>12}"
print(f'  {header}')
print('  ' + '-'*48)
for i, name in enumerate(new_meta['feature_names']):
    print(f"  {name:<20} {new_scaler.mean_[i]:>12.2f} {new_scaler.scale_[i]:>12.2f}")

# 计算新数据集的特征相关性
print('\n【新数据集特征相关性】')
new_flat = new_seq.reshape(-1, 21)
corr = np.corrcoef(new_flat.T)

high_corr_found = False
print('  高相关特征对 (|r| > 0.8):')
for i in range(21):
    for j in range(i+1, 21):
        if abs(corr[i,j]) > 0.8:
            high_corr_found = True
            f1 = new_meta['feature_names'][i]
            f2 = new_meta['feature_names'][j]
            print(f'    {f1} <-> {f2}: r={corr[i,j]:.3f}')

if not high_corr_found:
    print('    无 (已成功移除冗余特征)')

print('\n【低方差特征检查 (std < 0.5)】')
low_var_found = False
for i, name in enumerate(new_meta['feature_names']):
    if new_scaler.scale_[i] < 0.5:
        low_var_found = True
        print(f'  {name}: std={new_scaler.scale_[i]:.4f}')

if not low_var_found:
    print('  无 (已移除低方差特征)')

print('\n【改进总结】')
print('  ✓ 移除冗余: ip_len, payload_size (r=0.99)')
print('  ✓ 移除派生: src_ip_is_private, dst_ip_is_private')
print('  ✓ 移除低信息: tcp_flags_fin/rst/urg, tcp_seq/ack, icmp_code')
print('  ✓ IP编码: 32位整数→4字节 (方差更均匀)')
print('  ✓ 特征维度: 25 → 21 (减少16%)')

# 对比IP特征方差
print('\n【IP地址特征方差对比】')
print('  旧版:')
for i, name in enumerate(old_meta['feature_names']):
    if 'ip_int' in name:
        print(f'    {name}: std={old_scaler.scale_[i]:,.0f}')
print('  新版 (4字节编码):')
for i, name in enumerate(new_meta['feature_names']):
    if 'ip_b' in name:
        print(f'    {name}: std={new_scaler.scale_[i]:.2f}')

