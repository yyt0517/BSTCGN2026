"""
运行完整的原始包序列分类流程
1. 提取原始包序列
2. 准备数据（标准化、划分）
3. 训练Transformer模型
4. 生成对比报告
"""
import os
import sys
import time
import subprocess

print("="*100)
print("Complete Raw Packet Sequence Classification Pipeline".center(100))
print("="*100)

def run_script(script_path, description):
    """运行Python脚本"""
    print(f"\n{'='*100}")
    print(f"{description}".center(100))
    print(f"{'='*100}")
    print(f"运行脚本: {script_path}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', script_path],
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ 完成！耗时: {elapsed:.2f}秒")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 失败！耗时: {elapsed:.2f}秒")
        print(f"错误: {e}")
        return False

def main():
    pipeline_start = time.time()
    
    # Step 1: Extract raw packet sequences
    print("\n" + "="*100)
    print("Step 1/4: Extract Raw Packet Sequences".center(100))
    print("="*100)
    
    if not os.path.exists('data/processed/raw_packet_sequences/sequences.npy'):
        success = run_script(
            'scripts/raw_packet_sequence/01_extract_raw_sequences.py',
            'Extracting raw packet sequences from PCAP files'
        )
        if not success:
            print("\n✗ 提取失败，终止流程")
            return
    else:
        print("\n✓ 原始序列数据已存在，跳过提取步骤")
    
    # Step 2: Prepare data
    print("\n" + "="*100)
    print("Step 2/4: Prepare Data".center(100))
    print("="*100)
    
    if not os.path.exists('data/processed/raw_packet_sequences/train_sequences.npy'):
        success = run_script(
            'scripts/raw_packet_sequence/02_prepare_data.py',
            'Preparing data: normalization and train/test split'
        )
        if not success:
            print("\n✗ 数据准备失败，终止流程")
            return
    else:
        print("\n✓ 训练/测试数据已存在，跳过准备步骤")
    
    # Step 3: Train Transformer model
    print("\n" + "="*100)
    print("Step 3/4: Train Transformer Model".center(100))
    print("="*100)
    
    success = run_script(
        'scripts/raw_packet_sequence/04_train_transformer.py',
        'Training Transformer model on raw packet sequences'
    )
    if not success:
        print("\n✗ 训练失败，终止流程")
        return
    
    # Step 4: Generate comparison report
    print("\n" + "="*100)
    print("Step 4/4: Generate Comparison Report".center(100))
    print("="*100)
    
    success = run_script(
        'scripts/raw_packet_sequence/05_generate_report.py',
        'Generating comparison report with baseline models'
    )
    
    # Pipeline complete
    pipeline_time = time.time() - pipeline_start
    
    print("\n" + "="*100)
    print("PIPELINE COMPLETE!".center(100))
    print("="*100)
    print(f"\n总耗时: {pipeline_time:.2f}秒 ({pipeline_time/60:.2f}分钟)")
    print(f"\n结果保存在:")
    print(f"  - 数据: data/processed/raw_packet_sequences/")
    print(f"  - 模型: baseline/results/transformer_raw/")
    print(f"  - 报告: baseline/results/transformer_raw/evaluation_report.txt")

if __name__ == "__main__":
    main()

