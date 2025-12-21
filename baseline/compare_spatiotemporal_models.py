"""
时空模型对比脚本

专门用于对比所有时空结合模型：
- LSTM+GNN
- STGCN
- Hybrid
- BSTGCN

快速运行:
    python baseline/compare_spatiotemporal_models.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """运行时空模型对比"""
    
    print("=" * 100)
    print("时空模型对比实验".center(100))
    print("=" * 100)
    print()
    
    print("将对比以下模型:")
    print("  1. LSTM+GNN  - 串行时空建模")
    print("  2. STGCN     - 并行时空建模")
    print("  3. Hybrid    - Transformer+GNN (主要贡献)")
    print("  4. BSTGCN    - 贝叶斯时空图神经网络 (不确定性量化)")
    print()
    
    # 检查BSTGCN结果是否存在
    bstgcn_results = Path('baseline/results/bstgcn_50packets')
    if not bstgcn_results.exists() or not any(bstgcn_results.iterdir()):
        print("⚠️  警告: BSTGCN结果未找到")
        print("   请先运行: python scripts/BSTGCN/train_bstgcn.py")
        print()
        response = input("是否继续运行其他模型? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    # 运行对比
    print("=" * 100)
    print("开始运行对比...".center(100))
    print("=" * 100)
    print()
    
    cmd = [
        sys.executable,
        'baseline/run_comprehensive_baselines.py',
        '--models', 'lstm_gnn', 'stgcn', 'hybrid', 'bstgcn'
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 1
    
    print()
    print("=" * 100)
    print("对比完成！".center(100))
    print("=" * 100)
    print()
    print("结果已保存到: baseline/results_v2/")
    print("总结文件: baseline/results_v2/summary.json")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

