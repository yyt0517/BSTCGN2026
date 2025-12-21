"""
从PCAP文件提取50包序列数据
对比实验：使用50个包而不是100个包

关键变更：
- SEQUENCE_LENGTH = 50 (vs 100)
- STRIDE = 25 (vs 50) - 保持50%重叠率
- 输出目录: data/processed/raw_packet_sequences_50packets

这样可以：
1. 充分利用所有数据（不浪费后50个包）
2. 保持序列连续性（不强行截断）
3. 生成更多训练样本（stride=25）
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 直接导入原始脚本
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_module",
    os.path.join(os.path.dirname(__file__), '01_extract_raw_sequences.py')
)
extract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_module)

# 获取函数
build_raw_sequence_dataset = extract_module.build_raw_sequence_dataset

if __name__ == '__main__':
    print("="*100)
    print("提取50包序列数据 (对比实验)".center(100))
    print("="*100)

    # 配置
    BASE_DIR = '5G-NIDD-raw'  # 修改为正确的路径
    OUTPUT_DIR = 'data/processed/raw_packet_sequences_50packets'
    SEQUENCE_LENGTH = 50  # 使用50个包
    STRIDE = 25           # 步长25（50%重叠）
    USE_GTP_REMOVED = True
    MAX_PACKETS_PER_FILE = None  # 使用所有包

    print(f"\n配置:")
    print(f"  原始数据目录: {BASE_DIR}")
    print(f"  序列长度: {SEQUENCE_LENGTH} 包")
    print(f"  步长: {STRIDE} 包")
    print(f"  重叠率: {(SEQUENCE_LENGTH - STRIDE) / SEQUENCE_LENGTH * 100:.0f}%")
    print(f"  输出目录: {OUTPUT_DIR}")

    # 构建数据集
    build_raw_sequence_dataset(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        use_gtp_removed=USE_GTP_REMOVED,
        max_packets_per_file=MAX_PACKETS_PER_FILE
    )

    print("\n" + "="*100)
    print("✓ 50包序列数据提取完成！".center(100))
    print("="*100)

