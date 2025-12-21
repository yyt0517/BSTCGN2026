"""
可视化改进版 Hybrid 模型的图结构

展示：
1. 邻接矩阵热图
2. 图的度分布
3. 图的连通性
4. 示例子图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.hybrid_model.hybrid_model_improved import ImprovedHybridModel

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def visualize_adjacency_matrix(adj, sample_idx=0, save_path=None):
    """可视化邻接矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 取第一个样本
    adj_sample = adj[sample_idx].cpu().numpy()
    
    # 1. 完整邻接矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(adj_sample, cmap='Blues', aspect='auto')
    ax1.set_title('Complete Adjacency Matrix (100×100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Node Index', fontsize=12)
    ax1.set_ylabel('Node Index', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Edge Weight')
    
    # 2. 局部放大（前 20×20）
    ax2 = axes[1]
    im2 = ax2.imshow(adj_sample[:20, :20], cmap='Blues', aspect='auto')
    ax2.set_title('Zoomed View (First 20×20)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Node Index', fontsize=12)
    ax2.set_ylabel('Node Index', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Edge Weight')
    
    # 3. 度分布
    ax3 = axes[2]
    degrees = adj_sample.sum(axis=1) - 1  # 减去自环
    ax3.hist(degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(degrees.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {degrees.mean():.1f}')
    ax3.set_title('Node Degree Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Degree', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 邻接矩阵可视化已保存到: {save_path}")
    
    plt.show()


def analyze_graph_properties(adj):
    """分析图的属性"""
    adj_np = adj[0].cpu().numpy()
    
    # 基本统计
    num_nodes = adj_np.shape[0]
    num_edges = (adj_np.sum() - num_nodes) / 2  # 减去自环，除以2（无向图）
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_edges / max_edges
    
    # 度统计
    degrees = adj_np.sum(axis=1) - 1  # 减去自环
    
    # 连通性（简化检查）
    # 使用 BFS 检查是否所有节点可达
    visited = np.zeros(num_nodes, dtype=bool)
    queue = [0]
    visited[0] = True
    
    while queue:
        node = queue.pop(0)
        neighbors = np.where(adj_np[node] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    is_connected = visited.all()
    
    return {
        'num_nodes': num_nodes,
        'num_edges': int(num_edges),
        'max_edges': int(max_edges),
        'density': density,
        'avg_degree': degrees.mean(),
        'min_degree': degrees.min(),
        'max_degree': degrees.max(),
        'std_degree': degrees.std(),
        'is_connected': is_connected,
        'num_components': 1 if is_connected else 'Unknown'
    }


def visualize_graph_statistics(adj_list, k_values, save_path=None):
    """可视化不同 K 值下的图统计"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stats_list = [analyze_graph_properties(adj) for adj in adj_list]
    
    # 1. 边数 vs K
    ax1 = axes[0, 0]
    edges = [s['num_edges'] for s in stats_list]
    ax1.plot(k_values, edges, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Edges', fontsize=12, fontweight='bold')
    ax1.set_title('Graph Size vs K', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. 图密度 vs K
    ax2 = axes[0, 1]
    densities = [s['density'] * 100 for s in stats_list]
    ax2.plot(k_values, densities, marker='s', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Graph Density (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Graph Density vs K', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. 平均度数 vs K
    ax3 = axes[1, 0]
    avg_degrees = [s['avg_degree'] for s in stats_list]
    ax3.plot(k_values, avg_degrees, marker='^', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('K (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Degree', fontsize=12, fontweight='bold')
    ax3.set_title('Average Node Degree vs K', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. 统计表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for k, stats in zip(k_values, stats_list):
        table_data.append([
            f"K={k}",
            f"{stats['num_edges']:,}",
            f"{stats['density']*100:.1f}%",
            f"{stats['avg_degree']:.1f}",
            "Yes" if stats['is_connected'] else "No"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['K', 'Edges', 'Density', 'Avg Degree', 'Connected'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Graph Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图统计可视化已保存到: {save_path}")
    
    plt.show()


def visualize_complexity_breakdown(save_path=None):
    """可视化计算复杂度分解"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 各阶段计算量（饼图）
    stages = ['ST-Blocks\n(3层)', 'Transformer\n(1层)', 'Graph\nConstruction', 
              'Pooling', 'Classifier', 'Input Proj']
    operations = [3300, 920, 82, 0.8, 1, 20]  # 单位：百万操作
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
    
    wedges, texts, autotexts = ax1.pie(
        operations, 
        labels=stages, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Computational Cost Breakdown\n(Total: ~4.4B operations)', 
                  fontsize=14, fontweight='bold')
    
    # 2. 各操作详细分解（柱状图）
    operations_detail = {
        'Graph\nConstruction': {
            'Normalize': 0.8,
            'Similarity': 82,
            'Top-K': 2,
            'Build Adj': 0.06,
            'Symmetrize': 0.6
        },
        'GATv2\n(per layer)': {
            'Per-head Proj': 839,
            'Attention Score': 82,
            'Softmax': 5,
            'Aggregation': 82
        },
        'TCN\n(per layer)': {
            'Depthwise Conv': 2.5,
            'Pointwise Conv': 105
        }
    }
    
    x_pos = 0
    colors_detail = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for stage, ops in operations_detail.items():
        values = list(ops.values())
        labels = list(ops.keys())
        x_positions = np.arange(x_pos, x_pos + len(values))
        
        bars = ax2.bar(x_positions, values, color=colors_detail[:len(values)], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}M' if val >= 1 else f'{val*1000:.0f}K',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 添加阶段标签
        ax2.text(x_pos + len(values)/2 - 0.5, -max(values)*0.15, stage,
                ha='center', fontsize=11, fontweight='bold')
        
        x_pos += len(values) + 1
    
    ax2.set_ylabel('Operations (Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Detailed Operation Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 复杂度分解可视化已保存到: {save_path}")
    
    plt.show()


def main():
    print("="*80)
    print("改进版 Hybrid 模型图结构可视化")
    print("="*80)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 生成示例数据
    batch_size = 64
    seq_len = 100
    input_dim = 25
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # 创建模型
    model = ImprovedHybridModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=8,
        num_st_blocks=3,
        num_classes=9,
        dropout=0.2,
        k_neighbors=10
    ).to(device)
    
    model.eval()
    
    print("\n1. 生成邻接矩阵...")
    with torch.no_grad():
        # 前向传播到图构建
        x_proj = model.input_proj(x)
        x_trans = model.transformer(x_proj)
        adj = model.graph_builder(x_trans)
    
    print(f"   邻接矩阵形状: {adj.shape}")
    
    # 分析图属性
    print("\n2. 分析图属性...")
    stats = analyze_graph_properties(adj)
    
    print("\n图的基本属性:")
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  边数: {stats['num_edges']:,}")
    print(f"  最大可能边数: {stats['max_edges']:,}")
    print(f"  图密度: {stats['density']*100:.2f}%")
    print(f"  平均度数: {stats['avg_degree']:.2f}")
    print(f"  度数范围: [{stats['min_degree']:.0f}, {stats['max_degree']:.0f}]")
    print(f"  度数标准差: {stats['std_degree']:.2f}")
    print(f"  是否连通: {'是' if stats['is_connected'] else '否'}")
    
    # 创建输出目录
    output_dir = Path('baseline/results_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化邻接矩阵
    print("\n3. 可视化邻接矩阵...")
    visualize_adjacency_matrix(
        adj, 
        sample_idx=0,
        save_path=output_dir / 'graph_adjacency_matrix.png'
    )
    
    # 测试不同 K 值
    print("\n4. 测试不同 K 值的影响...")
    k_values = [5, 10, 15, 20, 30]
    adj_list = []
    
    for k in k_values:
        model_k = ImprovedHybridModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_heads=8,
            num_st_blocks=3,
            num_classes=9,
            dropout=0.2,
            k_neighbors=k
        ).to(device)
        
        model_k.eval()
        with torch.no_grad():
            x_proj = model_k.input_proj(x)
            x_trans = model_k.transformer(x_proj)
            adj_k = model_k.graph_builder(x_trans)
        
        adj_list.append(adj_k)
        print(f"   K={k}: {analyze_graph_properties(adj_k)['num_edges']:,} edges")
    
    visualize_graph_statistics(
        adj_list, 
        k_values,
        save_path=output_dir / 'graph_statistics_vs_k.png'
    )
    
    # 可视化复杂度分解
    print("\n5. 可视化计算复杂度...")
    visualize_complexity_breakdown(
        save_path=output_dir / 'complexity_breakdown.png'
    )
    
    print("\n✅ 所有可视化完成！")
    print(f"\n输出目录: {output_dir}")


if __name__ == '__main__':
    main()

