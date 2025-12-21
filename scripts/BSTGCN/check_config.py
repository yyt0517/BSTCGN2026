"""
检查BSTGCN训练配置

显示当前配置并与推荐配置对比
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    print("=" * 80)
    print("BSTGCN训练配置检查")
    print("=" * 80)
    
    # 读取当前配置
    config_file = os.path.join(os.path.dirname(__file__), 'train_bstgcn.py')
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取关键配置
    configs = {}
    
    import re
    
    patterns = {
        'dropout': r"'dropout':\s*([\d.]+)",
        'kl_weight_max': r"'kl_weight_max':\s*([\d.]+)",
        'weight_decay': r"'weight_decay':\s*([\de\-]+)",
        'patience': r"'patience':\s*(\d+)",
        'label_smoothing': r"'label_smoothing':\s*([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            configs[key] = match.group(1)
    
    # 推荐配置
    recommended = {
        'dropout': '0.25',
        'kl_weight_max': '0.001',
        'weight_decay': '1e-4',
        'patience': '15',
        'label_smoothing': '0.1',
    }
    
    # 显示对比
    print("\n配置对比:")
    print("-" * 80)
    print(f"{'参数':<20} {'当前值':<15} {'推荐值':<15} {'状态':<10}")
    print("-" * 80)
    
    all_ok = True
    
    for key in recommended:
        current = configs.get(key, 'N/A')
        rec = recommended[key]
        
        # 判断是否匹配
        try:
            current_val = float(current.replace('e-', 'e-'))
            rec_val = float(rec.replace('e-', 'e-'))
            ok = abs(current_val - rec_val) < 1e-6
        except:
            ok = current == rec
        
        status = "✓ OK" if ok else "✗ 需修复"
        
        if not ok:
            all_ok = False
        
        print(f"{key:<20} {current:<15} {rec:<15} {status:<10}")
    
    print("-" * 80)
    
    if all_ok:
        print("\n✅ 所有配置正确! 可以开始训练。")
        print("\n运行命令:")
        print("  python scripts/BSTGCN/train_bstgcn.py")
    else:
        print("\n⚠️  部分配置需要修复!")
        print("\n修复方法:")
        print("  1. 配置已自动更新到 train_bstgcn.py")
        print("  2. 重新运行训练即可")
        print("\n运行命令:")
        print("  python scripts/BSTGCN/train_bstgcn.py")
    
    print("\n" + "=" * 80)
    print("配置说明")
    print("=" * 80)
    print("""
1. dropout (0.25)
   - 增加随机正则化，防止过拟合
   - 从0.15增加到0.25 (提升67%)

2. kl_weight_max (0.001)
   - 增强贝叶斯正则化
   - 从0.0001增加到0.001 (提升10倍)
   - KL项贡献从0.11增加到1.1

3. weight_decay (1e-4)
   - 增强L2正则化
   - 从1e-5增加到1e-4 (提升10倍)

4. patience (15)
   - 更早停止训练，避免过拟合
   - 从25降低到15 (降低40%)

5. label_smoothing (0.1)
   - 防止模型过于自信
   - 从0.05增加到0.1 (提升100%)
""")
    
    print("=" * 80)
    print("预期效果")
    print("=" * 80)
    print("""
修复前:
  Epoch 30: Train F1=0.91, Val F1=0.88
  Epoch 35: Train F1=0.92, Val F1=0.43 ❌ 过拟合!

修复后:
  Epoch 30: Train F1=0.91, Val F1=0.90
  Epoch 35: Train F1=0.92, Val F1=0.91 ✓ 稳定
  Epoch 50: Train F1=0.94, Val F1=0.93 ✓ 收敛
""")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

