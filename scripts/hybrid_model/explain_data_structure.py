"""
解释数据结构：为什么有 100 个节点？

关键概念：
- 25 个特征 = 每个数据包的特征维度
- 100 个节点 = 每个样本包含的数据包数量（时间步）
"""

import numpy as np
import pickle
from pathlib import Path

def explain_data_structure():
    """详细解释数据结构"""
    
    print("="*80)
    print("数据结构详解：为什么有 100 个节点？")
    print("="*80)
    
    # 加载数据
    data_dir = Path('data/processed/raw_packet_sequences')
    
    # 加载元数据
    with open(data_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("\n📊 元数据信息:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # 加载训练数据
    train_sequences = np.load(data_dir / 'train_sequences.npy')
    train_labels = np.load(data_dir / 'train_labels.npy')
    
    print("\n" + "="*80)
    print("🔍 数据形状分析")
    print("="*80)
    
    print(f"\n训练序列形状: {train_sequences.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    num_samples, seq_len, num_features = train_sequences.shape
    
    print("\n📐 维度解释:")
    print(f"  维度 0 (样本数): {num_samples:,} 个网络流量样本")
    print(f"  维度 1 (序列长度): {seq_len} 个数据包（时间步）")
    print(f"  维度 2 (特征数): {num_features} 个特征/包")
    
    print("\n" + "="*80)
    print("💡 关键概念")
    print("="*80)
    
    print("\n1️⃣  什么是「节点」？")
    print("   在图神经网络中，每个节点代表一个「数据包」")
    print(f"   因此，一个样本有 {seq_len} 个数据包 = {seq_len} 个节点")
    
    print("\n2️⃣  什么是「特征」？")
    print("   每个数据包（节点）有 25 个特征，例如：")
    print("   - 包长度")
    print("   - 协议类型")
    print("   - 端口号")
    print("   - 时间间隔")
    print("   - ... 等共 25 个特征")
    
    print("\n3️⃣  数据结构类比")
    print("   可以把一个样本想象成一个「表格」：")
    print()
    print("   ┌─────────────┬──────────┬──────────┬─────┬──────────┐")
    print("   │   数据包    │  特征1   │  特征2   │ ... │  特征25  │")
    print("   ├─────────────┼──────────┼──────────┼─────┼──────────┤")
    print("   │  包 1 (节点1) │  0.523   │  1.234   │ ... │  0.891   │")
    print("   │  包 2 (节点2) │  0.612   │  0.987   │ ... │  0.745   │")
    print("   │  包 3 (节点3) │  0.789   │  1.456   │ ... │  0.623   │")
    print("   │     ...     │   ...    │   ...    │ ... │   ...    │")
    print("   │ 包100(节点100)│  0.456   │  0.789   │ ... │  0.912   │")
    print("   └─────────────┴──────────┴──────────┴─────┴──────────┘")
    print(f"                  ↑ 共 {num_features} 列（特征）")
    print(f"                  ↑ 共 {seq_len} 行（节点/数据包）")
    
    print("\n4️⃣  图结构的构建")
    print(f"   - 节点数 = {seq_len}（每个数据包是一个节点）")
    print(f"   - 节点特征维度 = {num_features}（每个节点有 25 个特征）")
    print(f"   - 边的构建：基于节点特征的相似度")
    print(f"     * 计算任意两个节点（数据包）之间的相似度")
    print(f"     * 选择最相似的 K=10 个邻居")
    print(f"     * 形成邻接矩阵 ({seq_len} × {seq_len})")
    
    print("\n" + "="*80)
    print("🎯 具体例子")
    print("="*80)
    
    # 取一个样本
    sample = train_sequences[0]
    label = train_labels[0]
    
    print(f"\n样本 0:")
    class_names = metadata.get('class_names', metadata.get('label_names', []))
    if class_names and label < len(class_names):
        print(f"  标签: {label} ({class_names[label]})")
    else:
        print(f"  标签: {label}")
    print(f"  形状: {sample.shape}")
    print(f"  解释: 这个样本包含 {sample.shape[0]} 个数据包，每个包有 {sample.shape[1]} 个特征")
    
    print(f"\n前 3 个数据包的前 5 个特征:")
    print(sample[:3, :5])
    
    print("\n在图神经网络中:")
    print(f"  - 节点 1 = 第 1 个数据包，特征向量 = {sample[0][:5]}... (共 25 维)")
    print(f"  - 节点 2 = 第 2 个数据包，特征向量 = {sample[1][:5]}... (共 25 维)")
    print(f"  - 节点 3 = 第 3 个数据包，特征向量 = {sample[2][:5]}... (共 25 维)")
    print(f"  - ...")
    print(f"  - 节点 100 = 第 100 个数据包，特征向量 = {sample[99][:5]}... (共 25 维)")
    
    print("\n" + "="*80)
    print("📊 图的邻接矩阵")
    print("="*80)
    
    print(f"\n邻接矩阵形状: ({seq_len}, {seq_len})")
    print(f"含义: 描述 {seq_len} 个节点之间的连接关系")
    print()
    print("   节点1  节点2  节点3  ...  节点100")
    print("节点1  [ 1     1     0    ...    0   ]  ← 节点1 连接到节点1(自环)和节点2")
    print("节点2  [ 1     1     1    ...    0   ]  ← 节点2 连接到节点1、2、3")
    print("节点3  [ 0     1     1    ...    1   ]  ← 节点3 连接到节点2、3、100")
    print(" ...   [ ...  ...   ...   ...   ... ]")
    print("节点100 [ 0     0     1    ...    1   ]  ← 节点100 连接到节点3和自己")
    
    print("\n" + "="*80)
    print("🔄 数据流动过程")
    print("="*80)
    
    print("\n输入数据:")
    print(f"  形状: (batch_size, seq_len, num_features)")
    print(f"  例如: (64, 100, 25)")
    print(f"  含义: 64 个样本，每个样本 100 个数据包，每个包 25 个特征")
    
    print("\n经过输入投影:")
    print(f"  形状: (64, 100, 128)")
    print(f"  含义: 64 个样本，每个样本 100 个节点，每个节点 128 维隐藏特征")
    
    print("\n构建图:")
    print(f"  邻接矩阵形状: (64, 100, 100)")
    print(f"  含义: 64 个图，每个图 100×100 的邻接矩阵")
    
    print("\n经过 GATv2:")
    print(f"  输入: x (64, 100, 128) + adj (64, 100, 100)")
    print(f"  输出: x (64, 100, 128)")
    print(f"  含义: 每个节点聚合其邻居的信息，更新自己的特征")
    
    print("\n经过池化:")
    print(f"  输入: (64, 100, 128)")
    print(f"  输出: (64, 128)")
    print(f"  含义: 将 100 个节点的信息聚合成一个图级别的表示")
    
    print("\n经过分类器:")
    print(f"  输入: (64, 128)")
    print(f"  输出: (64, 9)")
    print(f"  含义: 预测 9 个类别的概率")
    
    print("\n" + "="*80)
    print("📈 为什么选择 100 个数据包？")
    print("="*80)
    
    print("\n1. 时序信息的平衡")
    print("   - 太少（如 10 个包）：信息不足，难以捕获攻击模式")
    print("   - 太多（如 1000 个包）：计算量大，且可能包含噪声")
    print("   - 100 个包：经验上的最佳平衡点")
    
    print("\n2. 计算复杂度")
    print(f"   - 图构建: O(L²H) = O(100² × 128) = 1.28M 操作")
    print(f"   - GATv2: O(L²H) = O(100² × 128) = 1.28M 操作")
    print(f"   - 如果 L=1000: O(1000² × 128) = 128M 操作（100倍！）")
    
    print("\n3. 内存占用")
    print(f"   - 邻接矩阵: (64, 100, 100) = 640K 元素 = 2.44 MB")
    print(f"   - 如果 L=1000: (64, 1000, 1000) = 64M 元素 = 244 MB（100倍！）")
    
    print("\n" + "="*80)
    print("✅ 总结")
    print("="*80)
    
    print("\n关键点:")
    print(f"  ✓ 节点数 = 序列长度 = {seq_len}（数据包数量）")
    print(f"  ✓ 节点特征维度 = {num_features}（每个包的特征数）")
    print(f"  ✓ 图的邻接矩阵 = ({seq_len}, {seq_len})（节点间的连接关系）")
    print(f"  ✓ 每个节点代表一个数据包，不是一个特征！")
    
    print("\n类比:")
    print("  如果把网络流量看作一部电影：")
    print(f"  - 100 个数据包 = 100 帧画面（节点）")
    print(f"  - 25 个特征 = 每帧画面的 RGB 颜色、亮度等属性")
    print(f"  - 图结构 = 描述哪些帧之间有相似的内容")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    explain_data_structure()

