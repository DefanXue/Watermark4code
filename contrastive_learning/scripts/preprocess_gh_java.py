#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门为 GH-Java 数据集预处理
- 自动包装单个方法为类
- 使用与 MBXP/CSN-Java 相同的筛选标准
"""

import os
import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# 导入现有的预处理模块
from preprocess_csn_java import (
    JavaSyntaxChecker,
    remove_comments,
    find_tree_sitter_lib
)


def is_valid_gh_java_function(code, checker, min_length=20, max_length=2000):
    """
    GH-Java 专用的验证函数
    - 自动包装单个方法为类
    - 使用相同的筛选标准
    """
    # 1. 长度检查
    if not code or len(code) < min_length or len(code) > max_length:
        return False

    # 2. 移除注释
    clean_code = remove_comments(code)

    # 3. 尝试直接解析
    syntax_ok, _ = checker.check_syntax(clean_code)

    # 4. 如果失败，尝试包装为类
    if not syntax_ok:
        wrapped_code = f"public class Wrapper {{\n{clean_code}\n}}"
        syntax_ok, _ = checker.check_syntax(wrapped_code)
        if not syntax_ok:
            return False
        clean_code = wrapped_code

    # 5. AST详细分析
    try:
        tree = checker.parser.parse(bytes(clean_code, "utf8"))
        root_node = tree.root_node

        # 检查方法完整性
        if not checker.check_method_completeness(root_node):
            return False

        # 检查代码复杂度
        is_complex, _ = checker.measure_complexity(root_node)
        if not is_complex:
            return False
    except Exception:
        return False

    return True


def process_gh_java_file(input_path, output_path, checker, min_length=20, max_length=2000):
    """处理 GH-Java 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    valid_count = 0
    total_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc=f"Processing {os.path.basename(input_path)}"):
            total_count += 1
            try:
                data = json.loads(line.strip())

                # GH-Java 使用 'original_string' 字段
                code = data.get('original_string', '')
                if not code:
                    continue

                # 应用筛选标准
                if is_valid_gh_java_function(code, checker, min_length, max_length):
                    # 输出为统一格式
                    f_out.write(json.dumps({"code": code}, ensure_ascii=False) + '\n')
                    valid_count += 1

            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    print(f"Processing complete: {input_path} -> {output_path}")
    print(f"Total: {total_count}, Valid: {valid_count}, Rate: {valid_count/max(total_count,1):.2%}")
    return valid_count, total_count


def main():
    parser = argparse.ArgumentParser(description='Preprocess GH-Java dataset')
    parser.add_argument('--input_dir', type=str,
                       default='../../datasets/github_java_funcs',
                       help='Input directory containing train.jsonl, valid.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                       default='../../datasets/github_java_funcs',
                       help='Output directory')
    parser.add_argument('--min_length', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--tree_sitter_path', type=str, default=None)
    args = parser.parse_args()

    # 确保目录路径是相对于脚本的
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 查找 tree-sitter 语言库
    lib_path = args.tree_sitter_path or find_tree_sitter_lib()
    if not lib_path:
        print("Error: Cannot find tree-sitter language library")
        sys.exit(1)

    # 初始化语法检查器
    try:
        checker = JavaSyntaxChecker(lib_path)
    except Exception as e:
        print(f"Error: Cannot initialize tree-sitter: {e}")
        sys.exit(1)

    # 处理三个拆分
    splits = ['train', 'valid', 'test']
    total_stats = {'valid': 0, 'total': 0}

    for split in splits:
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}_filtered_code.jsonl"

        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue

        valid, total = process_gh_java_file(
            str(input_path),
            str(output_path),
            checker,
            args.min_length,
            args.max_length
        )
        total_stats['valid'] += valid
        total_stats['total'] += total

    print("\nOverall statistics:")
    print(f"Total samples: {total_stats['total']}")
    print(f"Valid samples: {total_stats['valid']}")
    print(f"Overall rate: {total_stats['valid']/max(total_stats['total'],1):.2%}")


if __name__ == "__main__":
    main()
