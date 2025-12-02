"""
检查实验环境是否配置正确
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - 缺失: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - 缺失: {dirpath}")
        return False

def main():
    print("="*80)
    print("实验环境检查")
    print("="*80)
    
    all_good = True
    
    # 检查目录结构
    print("\n[1] 检查目录结构...")
    dirs_to_check = [
        ("configs", "配置目录"),
        ("data", "数据目录"),
        ("data/dimension_analysis", "维度分析目录"),
        ("results", "结果目录"),
        ("results/strategy_1_random", "策略1目录"),
        ("results/strategy_2_top4", "策略2目录"),
        ("results/strategy_3_balanced", "策略3目录"),
        ("results/strategy_4_triple_order", "策略4目录"),
        ("analysis", "分析目录"),
        ("scripts", "脚本目录"),
    ]
    
    for dirpath, description in dirs_to_check:
        if not check_directory_exists(dirpath, description):
            all_good = False
    
    # 检查配置文件
    print("\n[2] 检查配置文件...")
    config_files = [
        ("configs/base_config.json", "基础配置"),
        ("configs/strategy_1_random.json", "策略1配置"),
        ("configs/strategy_2_top4.json", "策略2配置"),
        ("configs/strategy_3_balanced.json", "策略3配置"),
        ("configs/strategy_4_triple_order.json", "策略4配置"),
    ]
    
    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # 检查脚本文件
    print("\n[3] 检查脚本文件...")
    script_files = [
        ("scripts/step1_analyze_dimensions.py", "Step 1脚本"),
        ("scripts/step2_select_dimensions.py", "Step 2脚本"),
        ("scripts/step3_embed_watermarks.py", "Step 3脚本"),
        ("scripts/step4_extract_with_attacks.py", "Step 4脚本"),
        ("scripts/step5_analyze_results.py", "Step 5脚本"),
        ("scripts/step6_visualize.py", "Step 6脚本"),
    ]
    
    for filepath, description in script_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # 检查主运行脚本
    print("\n[4] 检查主脚本...")
    if not check_file_exists("run_all.py", "主运行脚本"):
        all_good = False
    
    if not check_file_exists("README.md", "说明文档"):
        all_good = False
    
    # 检查依赖模块
    print("\n[5] 检查依赖模块...")
    
    parent_dir = Path(__file__).parent.parent
    
    watermark_dir = parent_dir / "Watermark4code"
    srcmarker_dir = parent_dir / "SrcMarker-main"
    
    if not check_directory_exists(str(watermark_dir), "Watermark4code模块"):
        all_good = False
    
    if not check_directory_exists(str(srcmarker_dir), "SrcMarker-main模块"):
        all_good = False
    
    # 检查模型
    model_dir = parent_dir / "Watermark4code" / "best_model"
    if not check_directory_exists(str(model_dir), "鲁棒编码器模型"):
        all_good = False
    
    # 检查数据集
    anchors_file = parent_dir / "SrcMarker-main" / "contrastive_learning" / "datasets" / "MBXP" / "test_filtered_code.jsonl"
    if not check_file_exists(str(anchors_file), "MBXP测试数据集"):
        all_good = False
    
    # 检查Python环境
    print("\n[6] 检查Python环境...")
    try:
        import torch
        print(f"  ✓ PyTorch已安装 (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  ✓ GPU可用 (device: {torch.cuda.get_device_name(0)})")
        else:
            print(f"  ⚠ GPU不可用，将使用CPU（速度较慢）")
    except ImportError:
        print(f"  ✗ PyTorch未安装")
        all_good = False
    
    try:
        import numpy
        print(f"  ✓ NumPy已安装")
    except ImportError:
        print(f"  ✗ NumPy未安装")
        all_good = False
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib已安装")
    except ImportError:
        print(f"  ✗ Matplotlib未安装")
        all_good = False
    
    # 总结
    print("\n" + "="*80)
    if all_good:
        print("✅ 所有检查通过！可以开始运行实验。")
        print("\n运行方式：")
        print("  python run_all.py")
    else:
        print("❌ 检查失败，请修复上述问题后再运行实验。")
    print("="*80)

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    main()

