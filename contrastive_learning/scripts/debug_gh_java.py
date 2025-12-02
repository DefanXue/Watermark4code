#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断 GH-Java 预处理失败的原因
"""

import json
import sys
from pathlib import Path
from tree_sitter import Language, Parser

# 导入预处理模块
from preprocess_csn_java import JavaSyntaxChecker, is_valid_java_function, remove_comments

def test_single_sample(code, checker):
    """测试单个样本"""
    print("=" * 80)
    print("原始代码:")
    print(code[:200] + "..." if len(code) > 200 else code)
    print(f"\n代码长度: {len(code)}")

    # 1. 长度检查
    if len(code) < 20 or len(code) > 2000:
        print(f"[FAIL] Length check: {len(code)}")
        return False
    print(f"[PASS] Length check")

    # 2. 移除注释
    clean_code = remove_comments(code)
    print(f"[PASS] Comment removal")

    # 3. 语法检查
    syntax_ok, msg = checker.check_syntax(clean_code)
    if not syntax_ok:
        print(f"[FAIL] Syntax check: {msg}")

        # 尝试包装为类
        wrapped_code = f"public class Wrapper {{\n{clean_code}\n}}"
        syntax_ok2, msg2 = checker.check_syntax(wrapped_code)
        if syntax_ok2:
            print(f"[PASS] Syntax check after wrapping")
            clean_code = wrapped_code
        else:
            print(f"[FAIL] Still failed after wrapping: {msg2}")
            return False
    else:
        print(f"[PASS] Syntax check")

    # 4. 方法完整性
    tree = checker.parser.parse(bytes(clean_code, "utf8"))
    has_method = checker.check_method_completeness(tree.root_node)
    if not has_method:
        print(f"[FAIL] Method completeness check")
        return False
    print(f"[PASS] Method completeness check")

    # 5. 复杂度检查
    is_complex, stats = checker.measure_complexity(tree.root_node)
    print(f"Complexity stats: {stats}")
    if not is_complex:
        print(f"[FAIL] Complexity check")
        return False
    print(f"[PASS] Complexity check")

    return True

def main():
    # 初始化检查器
    lib_path = "../../parser/languages.so"
    checker = JavaSyntaxChecker(lib_path)

    # 测试前3个样本
    input_file = "../../datasets/github_java_funcs/test.jsonl"

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break

            data = json.loads(line)
            code = data.get('original_string', '')

            print(f"\nSample {i+1}:")
            result = test_single_sample(code, checker)
            print(f"Final result: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    main()
