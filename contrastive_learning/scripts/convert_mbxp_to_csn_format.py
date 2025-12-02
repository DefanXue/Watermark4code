#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 MBXP 数据集转换为 CSN-Java 的 raw/*.jsonl 格式
这样就可以直接使用现有的 preprocess_csn_java.py 等脚本
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_full_java_code(data_item):
    """
    从 MBXP 数据项中提取完整的 Java 代码
    
    Args:
        data_item: MBXP 数据项字典
        
    Returns:
        完整的 Java 代码字符串，如果提取失败则返回 None
    """
    try:
        # 检查是否为 Java 代码
        if data_item.get('language', '').lower() != 'java':
            return None
        
        # 提取 prompt 和 canonical_solution
        prompt = data_item.get('prompt', '')
        canonical_solution = data_item.get('canonical_solution', '')
        
        if not prompt or not canonical_solution:
            return None
        
        # 拼接完整代码
        full_code = prompt + canonical_solution
        
        return full_code
    except Exception:
        return None


def convert_mbxp_to_csn_format(input_file, output_file):
    """
    将 MBXP 格式转换为 CSN-Java 格式
    
    MBXP 格式:
    {
        "task_id": "MBJP/1",
        "prompt": "...",
        "canonical_solution": "...",
        "language": "java",
        ...
    }
    
    CSN-Java 格式 (与 raw/*.jsonl 一致):
    {
        "code": "完整的Java代码",
        "original_string": "完整的Java代码"
    }
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    converted_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="转换 MBXP 格式"):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # 提取完整 Java 代码
                code = extract_full_java_code(data)
                if not code:
                    continue
                
                # 转换为 CSN-Java 格式 (与 raw/*.jsonl 保持一致)
                csn_format = {
                    "code": code,
                    "original_string": code
                }
                
                f_out.write(json.dumps(csn_format, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    print(f"\n转换完成!")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总样本数: {total_count}")
    print(f"成功转换: {converted_count}")
    print(f"转换率: {converted_count/max(total_count,1):.2%}")
    
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description='将 MBXP 数据集转换为 CSN-Java 的 raw/*.jsonl 格式'
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='../datasets/MBXP/mbjp_release_v1.2.jsonl',
        help='MBXP 原始数据文件路径'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='../datasets/mbxp/raw/test.jsonl',
        help='输出文件路径（CSN-Java raw 格式）'
    )
    
    args = parser.parse_args()
    
    # 确保路径是相对于脚本的
    script_dir = Path(__file__).parent
    input_file = script_dir / args.input_file
    output_file = script_dir / args.output_file
    
    if not input_file.exists():
        print(f"错误: 找不到输入文件 {input_file}")
        return
    
    # 转换格式
    convert_mbxp_to_csn_format(str(input_file), str(output_file))
    
    print(f"\n✓ 转换完成！现在可以使用现有的 CSN-Java 处理脚本:")
    print(f"  1. python preprocess_csn_java.py --raw_dir ../datasets/mbxp/raw --output_dir ../datasets/mbxp")
    print(f"  2. python generate_java_train_data.py --test_input ../datasets/mbxp/test_filtered_code.jsonl --output_dir ../datasets/mbxp --splits test")
    print(f"  3. python simple_pairs_builder.py (如果需要)")


if __name__ == "__main__":
    main()


