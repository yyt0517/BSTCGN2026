"""
测试新的结果保存目录结构
"""

from pathlib import Path
from datetime import datetime

# 模拟配置
save_dir_base = 'baseline/results/bstgcn_50packets'

# 创建带时间戳的保存目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = Path(save_dir_base) / timestamp

print(f"基础目录: {save_dir_base}")
print(f"时间戳: {timestamp}")
print(f"完整保存路径: {save_dir}")
print(f"\n目录是否存在: {save_dir.exists()}")

# 创建目录
save_dir.mkdir(parents=True, exist_ok=True)
print(f"创建目录后: {save_dir.exists()}")

# 列出所有时间戳目录
base_path = Path(save_dir_base)
if base_path.exists():
    print(f"\n当前所有运行目录:")
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            print(f"  - {item.name}")

