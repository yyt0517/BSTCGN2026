"""
自动应用模型选择策略到训练脚本

使用方法：
    # 方案1：分阶段策略（推荐）
    python scripts/BSTGCN/apply_strategy.py staged --stage_epoch 30
    
    # 方案2：平衡策略
    python scripts/BSTGCN/apply_strategy.py balanced --kl_threshold 0.0001
    
    # 方案3：降低AUROC权重
    python scripts/BSTGCN/apply_strategy.py combined --auroc_weight 0.15
    
    # 恢复原始配置
    python scripts/BSTGCN/apply_strategy.py combined --auroc_weight 0.3
"""

import argparse
import re
from pathlib import Path


def modify_config(strategy, stage_epoch=30, kl_threshold=0.0001, auroc_weight=0.3):
    """修改train_bstgcn.py中的配置"""
    
    train_file = Path('scripts/BSTGCN/train_bstgcn.py')
    
    if not train_file.exists():
        print(f"❌ 找不到文件: {train_file}")
        return False
    
    # 读取文件
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改 selection_metric
    content = re.sub(
        r"'selection_metric':\s*'[^']*'",
        f"'selection_metric': '{strategy}'",
        content
    )
    
    # 修改 auroc_weight
    content = re.sub(
        r"'auroc_weight':\s*[\d.]+",
        f"'auroc_weight': {auroc_weight}",
        content
    )
    
    # 检查是否已有 stage_epoch 配置
    if "'stage_epoch':" in content:
        # 修改现有配置
        content = re.sub(
            r"'stage_epoch':\s*\d+",
            f"'stage_epoch': {stage_epoch}",
            content
        )
    else:
        # 在 auroc_weight 后添加
        content = re.sub(
            r"('auroc_weight':\s*[\d.]+,.*?\n)",
            f"\\1        'stage_epoch': {stage_epoch},            # staged策略的切换epoch\n",
            content
        )
    
    # 检查是否已有 kl_threshold 配置
    if "'kl_threshold':" in content:
        # 修改现有配置
        content = re.sub(
            r"'kl_threshold':\s*[\d.]+",
            f"'kl_threshold': {kl_threshold}",
            content
        )
    else:
        # 在 auroc_weight 后添加
        content = re.sub(
            r"('auroc_weight':\s*[\d.]+,.*?\n)",
            f"\\1        'kl_threshold': {kl_threshold},          # balanced策略的KL权重阈值\n",
            content
        )
    
    # 写回文件
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='应用模型选择策略')
    
    parser.add_argument(
        'strategy',
        type=str,
        choices=['f1', 'auroc', 'combined', 'balanced', 'staged'],
        help='模型选择策略'
    )
    
    parser.add_argument(
        '--stage_epoch',
        type=int,
        default=30,
        help='staged策略的切换epoch（默认30）'
    )
    
    parser.add_argument(
        '--kl_threshold',
        type=float,
        default=0.0001,
        help='balanced策略的KL权重阈值（默认0.0001）'
    )
    
    parser.add_argument(
        '--auroc_weight',
        type=float,
        default=0.3,
        help='AUROC权重（默认0.3）'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("应用模型选择策略")
    print("=" * 80)
    print(f"策略: {args.strategy}")
    
    if args.strategy == 'staged':
        print(f"\n配置:")
        print(f"  - stage_epoch: {args.stage_epoch}")
        print(f"  - auroc_weight: {args.auroc_weight}")
        print(f"\n工作方式:")
        print(f"  - Epoch 1-{args.stage_epoch}: 只看 Val F1")
        print(f"  - Epoch {args.stage_epoch+1}+: {1-args.auroc_weight:.1f}×F1 + {args.auroc_weight:.1f}×AUROC")
    
    elif args.strategy == 'balanced':
        print(f"\n配置:")
        print(f"  - kl_threshold: {args.kl_threshold}")
        print(f"  - auroc_weight: {args.auroc_weight}")
        print(f"\n工作方式:")
        print(f"  - KL权重 < {args.kl_threshold}: 只看 Val F1")
        print(f"  - KL权重 ≥ {args.kl_threshold}: {1-args.auroc_weight:.1f}×F1 + {args.auroc_weight:.1f}×AUROC")
    
    elif args.strategy == 'combined':
        print(f"\n配置:")
        print(f"  - auroc_weight: {args.auroc_weight}")
        print(f"\n工作方式:")
        print(f"  - 全程: {1-args.auroc_weight:.1f}×F1 + {args.auroc_weight:.1f}×AUROC")
    
    elif args.strategy == 'f1':
        print(f"\n工作方式:")
        print(f"  - 全程: 只看 Val F1")
    
    elif args.strategy == 'auroc':
        print(f"\n工作方式:")
        print(f"  - 全程: 只看 Val AUROC")
    
    print("=" * 80)
    
    # 应用配置
    success = modify_config(
        args.strategy,
        args.stage_epoch,
        args.kl_threshold,
        args.auroc_weight
    )
    
    if success:
        print("\n✅ 配置已成功应用到 scripts/BSTGCN/train_bstgcn.py")
        print("\n现在可以运行训练：")
        print("    python scripts/BSTGCN/train_bstgcn.py")
    else:
        print("\n❌ 配置应用失败")


if __name__ == '__main__':
    main()

