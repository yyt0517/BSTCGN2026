"""
快速测试不同的模型选择策略

使用方法：
    python scripts/BSTGCN/test_selection_strategies.py --strategy staged
    python scripts/BSTGCN/test_selection_strategies.py --strategy balanced
    python scripts/BSTGCN/test_selection_strategies.py --strategy combined --auroc_weight 0.15
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.BSTGCN.train_bstgcn import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description='测试BSTGCN模型选择策略')
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='staged',
        choices=['f1', 'auroc', 'combined', 'balanced', 'staged'],
        help='模型选择策略'
    )
    
    parser.add_argument(
        '--stage_epoch',
        type=int,
        default=30,
        help='staged策略的切换epoch（仅用于staged策略）'
    )
    
    parser.add_argument(
        '--kl_threshold',
        type=float,
        default=0.0001,
        help='balanced策略的KL权重阈值（仅用于balanced策略）'
    )
    
    parser.add_argument(
        '--auroc_weight',
        type=float,
        default=0.3,
        help='AUROC权重（用于combined/balanced/staged策略）'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=40,
        help='早停耐心值'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("BSTGCN 模型选择策略测试")
    print("=" * 80)
    print(f"策略: {args.strategy}")
    
    if args.strategy == 'staged':
        print(f"切换epoch: {args.stage_epoch}")
        print(f"前期（1-{args.stage_epoch}）: 只看F1")
        print(f"后期（{args.stage_epoch+1}+）: {1-args.auroc_weight:.1f}×F1 + {args.auroc_weight:.1f}×AUROC")
    elif args.strategy == 'balanced':
        print(f"KL阈值: {args.kl_threshold}")
        print(f"KL < {args.kl_threshold}: 只看F1")
        print(f"KL ≥ {args.kl_threshold}: {1-args.auroc_weight:.1f}×F1 + {args.auroc_weight:.1f}×AUROC")
    elif args.strategy == 'combined':
        print(f"全程: {1-args.auroc_weight:.1f}×F1 + {args.auroc_weight:.1f}×AUROC")
    elif args.strategy == 'f1':
        print("全程: 只看F1")
    elif args.strategy == 'auroc':
        print("全程: 只看AUROC")
    
    print(f"训练轮数: {args.epochs}")
    print(f"早停耐心: {args.patience}")
    print("=" * 80)
    
    # 修改配置并运行训练
    import scripts.BSTGCN.train_bstgcn as train_module
    
    # 临时修改配置
    original_main = train_module.main
    
    def modified_main():
        # 这里需要修改train_bstgcn.py中的config
        # 由于config在main函数内部，我们需要通过猴子补丁来修改
        import types
        
        # 获取原始main函数的代码
        original_code = original_main.__code__
        
        # 调用原始main，但我们需要在调用前修改全局变量
        # 更简单的方法：直接提示用户手动修改
        print("\n⚠️  请手动修改 scripts/BSTGCN/train_bstgcn.py 中的配置：")
        print("\n在 main() 函数的 config 字典中修改：")
        print(f"    'selection_metric': '{args.strategy}',")
        
        if args.strategy == 'staged':
            print(f"    'stage_epoch': {args.stage_epoch},")
        elif args.strategy == 'balanced':
            print(f"    'kl_threshold': {args.kl_threshold},")
        
        if args.strategy in ['combined', 'balanced', 'staged']:
            print(f"    'auroc_weight': {args.auroc_weight},")
        
        print(f"    'num_epochs': {args.epochs},")
        print(f"    'patience': {args.patience},")
        
        print("\n然后运行：")
        print("    python scripts/BSTGCN/train_bstgcn.py")
        print("\n或者直接在下方确认后自动运行（需要手动修改配置文件）")
        
        response = input("\n是否已修改配置并继续训练？(y/n): ")
        if response.lower() == 'y':
            original_main()
        else:
            print("已取消训练")
    
    modified_main()


if __name__ == '__main__':
    main()

