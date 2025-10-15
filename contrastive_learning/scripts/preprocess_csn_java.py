#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理CSN-Java数据集，严格对标CodeWMBench筛选标准
- 使用tree-sitter进行AST级语法分析
- 检查代码复杂度和结构
- 验证Java函数完整性
"""

import os
import re
import json
import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from tree_sitter import Language, Parser

# Java注释模式，与CodeWMBench的utils.py一致
JAVA_COMMENT_PATTERN = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"'

class JavaSyntaxChecker:
    """严格对标CodeWMBench的Java语法检查器"""
    
    def __init__(self, language_so_path):
        """初始化语法检查器，必须提供tree-sitter语言库路径"""
        if not os.path.exists(language_so_path):
            raise ValueError(f"找不到tree-sitter语言库: {language_so_path}")
            
        try:
            self.JAVA_LANGUAGE = Language(language_so_path, 'java')
            self.parser = Parser()
            self.parser.set_language(self.JAVA_LANGUAGE)
            print(f"成功初始化tree-sitter Java解析器")
        except Exception as e:
            raise RuntimeError(f"初始化tree-sitter失败: {e}")
    
    def check_syntax(self, code):
        """检查Java代码语法，完全对标CodeWMBench实现"""
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return self._check_node_syntax(tree.root_node)
        except Exception as e:
            return False, f"解析错误: {str(e)}"
    
    def _check_node_syntax(self, node):
        """递归检查节点语法，与CodeWMBench实现一致"""
        if node.has_error:
            return False, "Syntax error found in Java code."
        for child in node.children:
            result, message = self._check_node_syntax(child)
            if not result:
                return result, message
        return True, "No syntax errors found in Java code."
    
    def check_method_completeness(self, node):
        """检查是否包含完整方法定义，CodeWMBench隐含要求"""
        # 检查是否包含方法声明
        has_method = False
        
        def traverse(current_node):
            nonlocal has_method
            if current_node.type == "method_declaration":
                has_method = True
                return True
            
            for child in current_node.children:
                if traverse(child):
                    return True
            return False
        
        traverse(node)
        return has_method
    
    def measure_complexity(self, node):
        """测量代码复杂度，对标CodeWMBench筛选标准"""
        # 统计各类节点
        stats = {
            'method_declaration': 0,
            'if_statement': 0, 
            'for_statement': 0,
            'while_statement': 0,
            'try_statement': 0,
            'variable_declarator': 0,
            'method_invocation': 0,
            'total_nodes': 0
        }
        
        def traverse_count(current_node):
            stats['total_nodes'] += 1
            if current_node.type in stats:
                stats[current_node.type] += 1
                
            for child in current_node.children:
                traverse_count(child)
        
        traverse_count(node)
        
        # CodeWMBench隐含的复杂度要求
        control_flow = stats['if_statement'] + stats['for_statement'] + stats['while_statement'] + stats['try_statement']
        
        has_sufficient_complexity = (
            stats['method_declaration'] >= 1 and
            control_flow >= 1 and
            stats['variable_declarator'] >= 1 and
            stats['total_nodes'] >= 15  # 确保代码不是太简单
        )
        
        return has_sufficient_complexity, stats


def remove_comments(source):
    """
    移除Java代码中的注释，保留字符串内容
    完全复制CodeWMBench parser/utils.py中的实现
    """
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " * len(s)  # 注释替换为等长空白
        else:
            return s  # 保留字符串
    
    pattern = re.compile(
        JAVA_COMMENT_PATTERN,
        re.DOTALL | re.MULTILINE
    )
    
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def find_tree_sitter_lib():
    """
    在常见位置查找tree-sitter Java语言库
    
    Returns:
        语言库路径，找不到则返回None
    """
    # 可能的路径列表
    possible_paths = [
        # 当前脚本目录下
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parser', 'my-languages.so'),
        # 项目根目录下
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'parser', 'my-languages.so'),
        # CodeWMBench目录下
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'CodeWMBench-main', 'parser', 'my-languages.so'),
        # SrcMarker-main下
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'CodeWMBench-main', 'parser', 'my-languages.so')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到tree-sitter语言库: {path}")
            return path
            
    return None


def is_valid_java_function(code, checker, min_length=20, max_length=2000):
    """
    对标CodeWMBench的Java函数筛选标准
    
    Args:
        code: 代码字符串
        checker: JavaSyntaxChecker实例
        min_length: 最小代码长度
        max_length: 最大代码长度
        
    Returns:
        布尔值，表示代码是否通过筛选
    """
    # 1. 长度检查
    if not code or len(code) < min_length or len(code) > max_length:
        return False
    
    # 2. 移除注释
    clean_code = remove_comments(code)
    
    # 3. AST级语法检查
    syntax_ok, _ = checker.check_syntax(clean_code)
    if not syntax_ok:
        return False
    
    # 4. AST详细分析
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
        # 解析错误，当作不合格处理
        return False
    
    # 通过所有检查
    return True


def process_file(input_path, output_path, checker, min_length=20, max_length=2000):
    """处理单个JSONL文件，应用对标CodeWMBench的筛选标准"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    valid_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc=f"处理 {os.path.basename(input_path)}"):
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # 提取代码字段（适应不同的字段名）
                code = data.get('code', data.get('original_string', ''))
                if not code:
                    continue
                
                # 应用筛选标准
                if is_valid_java_function(code, checker, min_length, max_length):
                    # 输出为统一格式
                    f_out.write(json.dumps({"code": code}) + '\n')
                    valid_count += 1
            except json.JSONDecodeError:
                continue
            except Exception as e:
                # 处理过程中的任何异常
                continue
    
    print(f"处理完成: {input_path} -> {output_path}")
    print(f"总样本数: {total_count}, 有效样本数: {valid_count}, 筛选率: {valid_count/max(total_count,1):.2%}")
    return valid_count, total_count


def main():
    parser = argparse.ArgumentParser(description='处理CSN-Java数据集，严格对标CodeWMBench筛选标准')
    parser.add_argument('--raw_dir', type=str, default='../datasets/csn_java/raw',
                        help='原始数据目录，包含train.jsonl, valid.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str, default='../datasets/csn_java',
                        help='输出目录')
    parser.add_argument('--min_length', type=int, default=20,
                        help='最小代码长度')
    parser.add_argument('--max_length', type=int, default=2000,
                        help='最大代码长度')
    parser.add_argument('--tree_sitter_path', type=str, default=None,
                        help='tree-sitter语言库路径，如不指定则自动搜索')
    args = parser.parse_args()
    
    # 确保目录路径是相对于脚本的
    script_dir = Path(__file__).parent
    raw_dir = script_dir / args.raw_dir
    output_dir = script_dir / args.output_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找tree-sitter语言库
    lib_path = args.tree_sitter_path or find_tree_sitter_lib()
    if not lib_path:
        print("错误: 找不到tree-sitter语言库。请安装tree-sitter并构建Java语言支持。")
        print("参考步骤:")
        print("1. pip install tree_sitter")
        print("2. git clone https://github.com/tree-sitter/tree-sitter-java.git")
        print("3. python -c \"from tree_sitter import Language; Language.build_library('parser/my-languages.so', ['tree-sitter-java'])\"")
        print("或者使用--tree_sitter_path参数指定语言库路径。")
        sys.exit(1)
    
    # 初始化语法检查器
    try:
        checker = JavaSyntaxChecker(lib_path)
    except Exception as e:
        print(f"错误: 无法初始化tree-sitter: {e}")
        sys.exit(1)
    
    # 处理三个拆分
    splits = ['train', 'valid', 'test']
    total_stats = {'valid': 0, 'total': 0}
    
    for split in splits:
        input_path = raw_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}_filtered_code.jsonl"
        
        if not input_path.exists():
            print(f"警告: 找不到输入文件 {input_path}")
            continue
        
        valid, total = process_file(str(input_path), str(output_path), 
                                   checker, args.min_length, args.max_length)
        total_stats['valid'] += valid
        total_stats['total'] += total
    
    print("\n总体统计:")
    print(f"总样本数: {total_stats['total']}")
    print(f"有效样本数: {total_stats['valid']}")
    print(f"总体筛选率: {total_stats['valid']/max(total_stats['total'],1):.2%}")


if __name__ == "__main__":
    main() 