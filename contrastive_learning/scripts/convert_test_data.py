#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
转换测试数据格式，将生成的测试数据转换为评估所需的代码对格式。

将形如：
{
  "anchor": "原始代码",
  "positive": "增强后的代码" 
}
或
{
  "anchor": "原始代码",
  "negative": "负样本代码"
}

转换为：
{
  "code1": "代码1",
  "code2": "代码2",
  "label": 1或0
}
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 新增：固定HTTP代理
proxy_address = "http://localhost:8888"
os.environ['HTTP_PROXY'] = proxy_address
os.environ['HTTPS_PROXY'] = proxy_address

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def convert_data(input_file, output_file):
    """
    转换测试数据格式
    
    参数:
        input_file: 输入文件路径（原始生成的测试数据）
        output_file: 输出文件路径（转换后的评估格式数据）
    """
    print(f"转换数据格式: {input_file} -> {output_file}")
    
    # 计数器
    positive_count = 0
    negative_count = 0
    skipped_count = 0
    
    # 读取输入文件并转换格式
    output_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 检查是正样本还是负样本
                if 'anchor' in data and 'positive' in data:
                    # 正样本
                    converted_item = {
                        'code1': data['anchor'],
                        'code2': data['positive'],
                        'label': 1
                    }
                    positive_count += 1
                    output_data.append(converted_item)
                    
                elif 'anchor' in data and 'negative' in data:
                    # 负样本
                    converted_item = {
                        'code1': data['anchor'],
                        'code2': data['negative'],
                        'label': 0
                    }
                    negative_count += 1
                    output_data.append(converted_item)
                    
                else:
                    # 跳过不符合格式的数据
                    print(f"警告: 第{line_num}行数据格式不符合预期，已跳过")
                    skipped_count += 1
                    
            except json.JSONDecodeError:
                print(f"警告: 第{line_num}行不是有效的JSON格式，已跳过")
                skipped_count += 1
                continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    print(f"转换完成:")
    print(f"  - 正样本数量: {positive_count}")
    print(f"  - 负样本数量: {negative_count}")
    print(f"  - 跳过的条目: {skipped_count}")
    print(f"  - 总输出条目: {positive_count + negative_count}")
    print(f"  - 输出文件: {output_file}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="转换测试数据格式为评估所需的代码对格式")
    parser.add_argument('--input', type=str, required=True,
                      help='输入文件路径（原始生成的测试数据）')
    parser.add_argument('--output', type=str, default=None,
                      help='输出文件路径（默认为原文件名_pairs.jsonl）')
    
    args = parser.parse_args()
    
    # 如果未指定输出文件，则使用默认命名
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_pairs{input_path.suffix}"
        args.output = str(output_path)
    
    # 转换数据
    convert_data(args.input, args.output)


if __name__ == "__main__":
    main() 