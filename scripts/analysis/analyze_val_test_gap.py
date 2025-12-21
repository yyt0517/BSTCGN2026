"""
分析验证集和测试集之间性能差距的原因
"""

import numpy as np
import torch
import pickle
import os
import sys
sys.path.insert(0, 'scripts/hybrid_model')
from hybrid_model_improved import GraphSAGEHybridModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

data_dir = 'data/processed/raw_packet_sequences_50packets_no_leak'
model_path = 'baseline/results/hybrid_model_50packets_no_leak/best_model.pth'

# 加载数据
train_seqs = np.load(os.path.join(data_dir, 'train_sequences.npy'))
train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
val_seqs = np.load(os.path.join(data_dir, 'val_sequences.npy'))
val_labels = np.load(os.path.join(data_dir, 'val_labels.npy'))
test_seqs = np.load(os.path.join(data_dir, 'test_sequences.npy'))
test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = GraphSAGEHybridModel(
    input_dim=25,
    hidden_dim=128,
    num_classes=9,
    num_heads=8,
    num_st_blocks=3,
    dropout=0.2,
    k_neighbors=10
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def predict(sequences):
    """分批预测"""
    tensor = torch.FloatTensor(sequences).to(device)
    batch_size = 512
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i+batch_size]
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

class_names = ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan', 
               'SlowrateDoS', 'TCPConnectScan', 'UDPFlood', 'UDPScan']

# 预测
print("预测验证集...")
val_preds = predict(val_seqs)
print("预测测试集...")
test_preds = predict(test_seqs)

print('='*80)
print('验证集 vs 测试集 性能对比')
print('='*80)

print(f"\n{'指标':<20} {'验证集':>15} {'测试集':>15}")
print('-'*50)
print(f"{'Accuracy':<20} {accuracy_score(val_labels, val_preds):>15.4f} {accuracy_score(test_labels, test_preds):>15.4f}")
print(f"{'Macro F1':<20} {f1_score(val_labels, val_preds, average='macro'):>15.4f} {f1_score(test_labels, test_preds, average='macro'):>15.4f}")

print('\n' + '='*80)
print('各类别 Recall 对比')
print('='*80)
print(f"\n{'类别':<20} {'验证集':>15} {'测试集':>15} {'差距':>15}")
print('-'*65)

for i, name in enumerate(class_names):
    val_mask = val_labels == i
    test_mask = test_labels == i
    val_recall = (val_preds[val_mask] == i).mean() if val_mask.sum() > 0 else 0
    test_recall = (test_preds[test_mask] == i).mean() if test_mask.sum() > 0 else 0
    gap = val_recall - test_recall
    print(f"{name:<20} {val_recall:>15.4f} {test_recall:>15.4f} {gap:>+15.4f}")

# 分析特征分布差异
print('\n' + '='*80)
print('特征分布差异分析 (每类的均值差异)')
print('='*80)

feature_names = ['timestamp', 'packet_length', 'protocol', 'src_ip_int', 'dst_ip_int',
                 'src_ip_is_private', 'dst_ip_is_private', 'src_port', 'dst_port',
                 'ip_ttl', 'ip_len', 'ip_tos', 'ip_flags',
                 'tcp_flags_syn', 'tcp_flags_ack', 'tcp_flags_fin',
                 'tcp_flags_rst', 'tcp_flags_psh', 'tcp_flags_urg',
                 'tcp_window', 'tcp_seq', 'tcp_ack', 'payload_size',
                 'icmp_type', 'icmp_code']

# 对几个关键类分析特征差异
print("\n选择几个关键特征分析分布偏移：")
key_features = [0, 1, 7, 8, 9, 22]  # timestamp, packet_length, src_port, dst_port, ip_ttl, payload_size

for class_idx in [1, 2, 7]:  # HTTPFlood, ICMPFlood, UDPFlood
    print(f"\n{class_names[class_idx]}:")
    train_class = train_seqs[train_labels == class_idx]
    val_class = val_seqs[val_labels == class_idx]
    test_class = test_seqs[test_labels == class_idx]
    
    for feat_idx in key_features:
        train_mean = train_class[:, :, feat_idx].mean()
        val_mean = val_class[:, :, feat_idx].mean()
        test_mean = test_class[:, :, feat_idx].mean()
        
        # 计算验证集和测试集相对于训练集的偏移
        val_shift = (val_mean - train_mean) / (abs(train_mean) + 1e-8)
        test_shift = (test_mean - train_mean) / (abs(train_mean) + 1e-8)
        
        if abs(val_shift - test_shift) > 0.1:  # 只显示差异大的
            print(f"  {feature_names[feat_idx]:<15}: train={train_mean:>8.3f}, val={val_mean:>8.3f}, test={test_mean:>8.3f}")
            print(f"    {'':15}  val_shift={val_shift:>+.2%}, test_shift={test_shift:>+.2%}")

