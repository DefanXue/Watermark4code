#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一体化实验脚本 - 按顺序执行所有步骤
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime

def run_step(step_name, script_path, extra_args=None):
    """运行单个步骤"""
    print("\n" + "="*80)
    print(f"开始执行: {step_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if extra_args:
        print(f"参数: {' '.join(extra_args)}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # 构建命令行
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "-"*80)
        print(f"✓ {step_name} 完成!")
        print(f"  用时: {hours}小时 {minutes}分钟 {seconds}秒")
        print("-"*80)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print("\n" + "-"*80)
        print(f"✗ {step_name} 失败!")
        print(f"  错误代码: {e.returncode}")
        print(f"  用时: {int(elapsed_time//60)}分钟 {int(elapsed_time%60)}秒")
        print("-"*80)
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="代码水印维度策略对比实验 - 完整流程")
    parser.add_argument('--resume', action='store_true', 
                       help='为支持--resume的步骤添加--resume参数（Step 3, 4, 4b）')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("代码水印维度策略对比实验 - 完整流程")
    if args.resume:
        print("模式: 断点继续（--resume）")
    print("="*80)
    
    total_start = time.time()
    
    # 定义所有步骤（带并发参数）
    steps = [
        ("Step 1: 维度分析", "scripts/step1_analyze_dimensions.py", []),
        ("Step 1b: 训练策略5方向矩阵", "scripts/step1b_train_learned_directions.py", 
         ["--concurrency", "5"]),
        ("Step 2: 维度选择", "scripts/step2_select_dimensions.py", []),
        ("Step 3: 水印嵌入", "scripts/step3_embed_watermarks.py", 
         ["--concurrency", "5"] + (["--resume"] if args.resume else [])),
        ("Step 4: Baseline提取测试", "scripts/step4_extract_with_attacks.py", 
         ["--concurrency", "5"] + (["--resume"] if args.resume else [])),
        ("Step 4b: Alternative方法测试", "scripts/step4b_extract_with_alternative_methods.py", 
         ["--concurrency", "5"] + (["--resume"] if args.resume else [])),
        ("Step 5: 结果分析", "scripts/step5_analyze_results.py", []),
        ("Step 6: 可视化", "scripts/step6_visualize.py", [])
    ]
    
    # 按顺序执行所有步骤
    for step_name, script_path, extra_args in steps:
        success = run_step(step_name, script_path, extra_args)
        if not success:
            print("\n" + "="*80)
            print("实验中止：上一步骤执行失败")
            print("="*80)
            sys.exit(1)
    
    # 计算总用时
    total_elapsed = time.time() - total_start
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    total_seconds = int(total_elapsed % 60)
    
    print("\n" + "="*80)
    print("✓ 所有步骤执行完成!")
    print(f"  总用时: {total_hours}小时 {total_minutes}分钟 {total_seconds}秒")
    print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n结果文件位置:")
    print("  - 维度分析: data/dimension_analysis/")
    print("  - 策略5训练: results/strategy_5_learned/trained_W.pth")
    print("  - 维度选择: results/{strategy}/embedding/")
    print("  - 水印嵌入: results/{strategy}/embedding/")
    print("  - 提取结果: results/{strategy}/extraction/")
    print("  - 分析报告: analysis/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

