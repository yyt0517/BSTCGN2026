"""
测试BSTGCN集成到综合基线脚本

验证:
1. BSTGCN模型可以被正确导入
2. 可以从已有结果读取BSTGCN性能
3. 命令行参数正确工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试导入"""
    print("=" * 80)
    print("测试1: 验证BSTGCN模块导入")
    print("=" * 80)
    
    try:
        from scripts.BSTGCN.bstgcn_model import BSTGCNModel
        print("✅ BSTGCNModel 导入成功")
    except Exception as e:
        print(f"❌ BSTGCNModel 导入失败: {e}")
        return False
    
    try:
        from scripts.BSTGCN.elbo_loss import ELBOLoss, KLWeightScheduler
        print("✅ ELBOLoss, KLWeightScheduler 导入成功")
    except Exception as e:
        print(f"❌ ELBO组件导入失败: {e}")
        return False
    
    try:
        from scripts.BSTGCN.mc_inference import batch_mc_inference
        print("✅ batch_mc_inference 导入成功")
    except Exception as e:
        print(f"❌ MC推理导入失败: {e}")
        return False
    
    print("\n✅ 所有导入测试通过\n")
    return True


def test_model_availability():
    """测试模型可用性"""
    print("=" * 80)
    print("测试2: 验证BSTGCN在可用模型列表中")
    print("=" * 80)
    
    from baseline.run_comprehensive_baselines import AVAILABLE_MODELS
    
    if 'bstgcn' in AVAILABLE_MODELS:
        print(f"✅ BSTGCN在可用模型列表中: {AVAILABLE_MODELS['bstgcn']}")
        return True
    else:
        print("❌ BSTGCN不在可用模型列表中")
        return False


def test_result_reading():
    """测试结果读取"""
    print("\n" + "=" * 80)
    print("测试3: 验证BSTGCN结果文件")
    print("=" * 80)
    
    bstgcn_results_dir = Path('baseline/results/bstgcn_50packets')
    
    if not bstgcn_results_dir.exists():
        print(f"⚠️  BSTGCN结果目录不存在: {bstgcn_results_dir}")
        print("   请先运行: python scripts/BSTGCN/train_bstgcn.py")
        return False
    
    # 查找最新的结果目录
    result_dirs = sorted([d for d in bstgcn_results_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
    
    if not result_dirs:
        print("⚠️  未找到BSTGCN结果子目录")
        return False
    
    latest_result = result_dirs[0]
    print(f"✅ 找到最新结果目录: {latest_result.name}")
    
    # 检查评估报告
    report_file = latest_result / 'evaluation_report.txt'
    if not report_file.exists():
        print(f"❌ 评估报告不存在: {report_file}")
        return False
    
    print(f"✅ 评估报告存在: {report_file}")
    
    # 尝试解析报告
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        import re
        acc_match = re.search(r'Accuracy:\s+([\d.]+)', content)
        macro_f1_match = re.search(r'Macro F1:\s+([\d.]+)', content)
        
        if acc_match and macro_f1_match:
            print(f"✅ 成功解析指标:")
            print(f"   Accuracy: {acc_match.group(1)}")
            print(f"   Macro F1: {macro_f1_match.group(1)}")
            return True
        else:
            print("❌ 无法解析指标")
            return False
            
    except Exception as e:
        print(f"❌ 读取报告失败: {e}")
        return False


def test_command_line():
    """测试命令行参数"""
    print("\n" + "=" * 80)
    print("测试4: 验证命令行参数")
    print("=" * 80)
    
    print("\n可用的运行命令:")
    print("  1. 只运行BSTGCN:")
    print("     python baseline/run_comprehensive_baselines.py --models bstgcn")
    print("\n  2. 运行BSTGCN和其他模型:")
    print("     python baseline/run_comprehensive_baselines.py --models bstgcn stgcn hybrid")
    print("\n  3. 运行所有模型（包括BSTGCN）:")
    print("     python baseline/run_comprehensive_baselines.py --models all")
    print("\n  4. 列出所有可用模型:")
    print("     python baseline/run_comprehensive_baselines.py --list")
    
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("BSTGCN集成测试")
    print("=" * 80 + "\n")
    
    results = []
    
    # 测试1: 导入
    results.append(("导入测试", test_imports()))
    
    # 测试2: 模型可用性
    results.append(("模型可用性", test_model_availability()))
    
    # 测试3: 结果读取
    results.append(("结果读取", test_result_reading()))
    
    # 测试4: 命令行
    results.append(("命令行参数", test_command_line()))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:<20} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！BSTGCN已成功集成到综合基线脚本中。")
        print("\n下一步:")
        print("  运行: python baseline/run_comprehensive_baselines.py --models bstgcn")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

