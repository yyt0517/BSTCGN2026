"""对比四个版本的模型性能"""

print('='*80)
print('特征优化实验对比')
print('='*80)

# 硬编码结果数据
results = [
    ('25维-baseline', 0.9721, 0.9195, 0.9723, 421130),
    ('21维-baseline', 0.9692, 0.9244, 0.9696, 420618),
    ('21维-优化(h128+LS)', 0.9698, 0.9265, 0.9700, 420618),
    ('23维-最佳(+seq/ack)', 0.9710, 0.9306, 0.9714, 420874),
]

print(f"\n{'版本':<22} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'参数量':>12}")
print('-'*72)
for name, acc, mf1, wf1, params in results:
    print(f"{name:<22} {acc:>10.4f} {mf1:>10.4f} {wf1:>12.4f} {params:>12,}")

print('\n各类别F1对比:')
print(f"{'类别':<16} {'25维-base':>12} {'21维-opt':>12} {'23维-best':>12}")
print('-'*54)

class_f1 = [
    ('Benign', 0.9104, 1.0000, 1.0000),
    ('SYNScan', 0.7503, 0.7903, 0.8039),
    ('TCPConnectScan', 0.8156, 0.8366, 0.8351),
    ('SYNFlood', 0.9873, 0.9336, 0.9315),
    ('UDPScan', 0.8744, 0.8502, 0.8710),
    ('ICMPFlood', 0.9578, 0.9518, 0.9580),
]

for cls, s1, s2, s3 in class_f1:
    print(f"{cls:<16} {s1:>12.4f} {s2:>12.4f} {s3:>12.4f}")

print('\n' + '='*80)
print('结论:')
print('='*80)
print('1. 23维特征版本取得最佳综合性能:')
print('   - Accuracy: 0.9710 (接近25维baseline的0.9721)')
print('   - Macro F1: 0.9306 (比25维baseline提升 +1.11%)')
print('   - Weighted F1: 0.9714 (接近25维baseline的0.9723)')
print('2. 恢复tcp_seq/tcp_ack显著改善了:')
print('   - SYNScan: 0.7903 -> 0.8039 (+1.36%)')
print('   - UDPScan: 0.8502 -> 0.8710 (+2.08%)')
print('   - ICMPFlood: 0.9518 -> 0.9580 (+0.62%)')
print('3. 23维特征是最佳选择，保留tcp_seq/ack对扫描类攻击分类有帮助')

