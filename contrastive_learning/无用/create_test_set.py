#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成代码克隆检测测试数据集的脚本。
该脚本生成一个高质量的测试数据集，用于评估代码克隆检测模型的性能。
"""

import os
import sys
import json
import random
import time
import subprocess
import tempfile
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import argparse

proxy_address = "http://localhost:8888"
os.environ['HTTP_PROXY'] = proxy_address
os.environ['HTTPS_PROXY'] = proxy_address

# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入项目内部模块
from contrastive_learning.augmentor import CodeAugmentor

# 配置Google AI
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Error: 请安装Google Generative AI SDK: pip install google-generativeai")
    sys.exit(1)

# 全局配置
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: 未设置GOOGLE_API_KEY环境变量")
    print("请设置环境变量: export GOOGLE_API_KEY=你的API密钥")
    sys.exit(1)

# 配置Google AI
genai.configure(api_key=API_KEY)

# 文件路径配置
SOURCE_FILE = os.path.join(project_root, "datasets/csn_python/test.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "adversarial_clone_test.jsonl")

# 样本数量配置
NUM_SAMPLES_TO_PROCESS = 2000  # 处理的样本数量（不再用于截断，仅保留）

# LLM模型配置
LLM_MODEL_NAME = 'models/gemma-3-27b-it'  # 设置使用的模型名称

# 安全设置
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# ===== 调试输出辅助函数 =====

def _now_str() -> str:
    return time.strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now_str()}] {message}", flush=True)


def _extract_func_name(code_text: str) -> str:
    try:
        for line in code_text.split("\n"):
            s = line.strip()
            if s.startswith("def "):
                return s[4:].split("(")[-2].split(")")[0].split("(")[0] if "(" in s else s[4:].strip()
    except Exception:
        pass
    return "unknown_func"

# 新增：代码静态预筛（函数定义、长度阈值、且不包含 import/from）
def _is_candidate_function(code_text: str) -> bool:
    if not code_text:
        return False
    stripped = code_text.strip()
    if len(stripped) <= 50:
        return False
    if not stripped.startswith("def "):
        return False
    # 简单排除 import 与 from 导入（整段代码中出现即过滤）
    if "\nimport " in code_text or code_text.strip().startswith("import "):
        return False
    if "\nfrom " in code_text or code_text.strip().startswith("from "):
        return False
    return True


def generate_with_retry(
    model: Any, prompt: str, max_retries: int = 3, temperature: float = 0.2
) -> str:
    """
    使用重试机制调用Google Gemini API，并对结果进行后处理。
    
    参数:
        model: genai.GenerativeModel实例
        prompt: 提示字符串
        max_retries: 最大重试次数
        temperature: 生成温度
        
    返回:
        处理后的生成内容字符串
    """
    # 创建生成配置
    generation_config = {
        "candidate_count": 1,
        "temperature": temperature,
        "max_output_tokens": 4096,
    }
    
    retry_count = 0
    _log(f"[LLM] 发起生成: temp={temperature}, prompt_len={len(prompt)}")
    
    while retry_count < max_retries:
        try:
            _log(f"[LLM] 尝试第 {retry_count + 1}/{max_retries} 次请求...")
            t0 = time.time()
            # 调用模型生成内容
            response = model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS
            )
            dt = time.time() - t0
            
            # 处理结果
            result = response.text.strip()
            
            # 移除可能存在的Markdown代码块标记
            if result.startswith("```python"):
                result = result.replace("```python", "", 1).strip()
            if result.startswith("```"):
                result = result.replace("```", "", 1).strip()
            if result.endswith("```"):
                result = result[:-3].strip()
                
            _log(f"[LLM] 成功返回: text_len={len(result)}, 用时={dt:.2f}s")
            return result
        
        except Exception as e:
            retry_count += 1
            _log(f"[LLM] API调用失败，重试 {retry_count}/{max_retries}，错误: {e}")
            time.sleep(5)  # 等待5秒后重试
    
    # 如果所有重试都失败
    raise RuntimeError("所有API调用重试都失败")


def execute_tests(code_to_test: str, test_code: str) -> Tuple[bool, str]:
    """
    在隔离环境中执行代码和对应的单元测试，返回测试结果。
    
    参数:
        code_to_test: 被测代码字符串
        test_code: 测试代码字符串
        
    返回:
        (是否通过测试, 结果消息)的元组
    """
    # 提取测试类名
    test_class_name = None
    for line in test_code.split("\n"):
        if "class " in line and "(unittest.TestCase)" in line:
            test_class_name = line.split("class ")[1].split("(")[0].strip()
            break
    
    if not test_class_name:
        _log("[TEST] 无法提取测试类名，跳过该样本")
        return False, "无法提取测试类名"
    
    temp_dir_ctx = None
    test_file_path = None
    
    try:
        # 为测试运行创建独立的临时工作目录，避免污染项目目录
        temp_dir_ctx = tempfile.TemporaryDirectory()
        run_dir = temp_dir_ctx.name
        test_file_path = os.path.join(run_dir, "test_run.py")
        
        # 写入完整的测试脚本
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write("import unittest\n\n")
            f.write(code_to_test)
            f.write("\n\n")
            f.write(test_code)
            f.write("\n\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    unittest.main(argv=['first-arg-is-ignored'], exit=False)\n")
        
        _log(f"[TEST] 运行单元测试: class={test_class_name}, 文件={os.path.basename(test_file_path)}")
        t0 = time.time()
        # 在独立工作目录中执行，防止测试在项目目录创建文件/文件夹
        result = subprocess.run(
            [sys.executable, test_file_path],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=run_dir
        )
        dt = time.time() - t0
        
        # 解析结果
        _log(f"[TEST] 结束: returncode={result.returncode}, 用时={dt:.2f}s")
        if result.returncode == 0:
            return True, "Tests passed"
        else:
            tail = (result.stderr or "")[-200:]
            _log(f"[TEST] 错误(stderr最后200字节): {tail}")
            return False, result.stderr
    
    except subprocess.TimeoutExpired:
        _log("[TEST] 超时(>20s)")
        return False, "测试执行超时"
    except Exception as e:
        _log(f"[TEST] 执行时异常: {e}")
        return False, f"执行测试时发生错误: {str(e)}"
    
    finally:
        # 自动清理临时工作目录
        if temp_dir_ctx is not None:
            try:
                temp_dir_ctx.cleanup()
            except Exception:
                pass


def main():
    """主函数：生成代码克隆检测测试数据集"""
    # 新增：解析命令行参数，支持 --max_pairs 控制最终写入的样本对上限
    parser = argparse.ArgumentParser(description="Create adversarial clone test set")
    parser.add_argument('--max_pairs', type=int, default=6000, help='最终写入的样本对总上限（正样本+两类负样本）')
    args = parser.parse_args()
    max_pairs = int(args.max_pairs)

    print(f"====== 开始生成代码克隆检测测试数据集 ======")
    print(f"源文件: {SOURCE_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"目标样本对上限(max_pairs): {max_pairs}")
    print(f"使用模型: {LLM_MODEL_NAME}")
    
    # 初始化模型和增强器
    print("正在初始化LLM模型和代码增强器...")
    llm_model = genai.GenerativeModel(model_name=LLM_MODEL_NAME)
    augmentor = CodeAugmentor()
    
    # 加载源代码
    print(f"正在加载源代码...")
    source_codes = []
    total_read = 0
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # 提取代码，与contrastive_learning/dataset.py中的处理方式保持一致
                    code = None
                    for field in ['code', 'content', 'function', 'source', 'raw']:
                        if field in data and isinstance(data[field], str):
                            code = data[field]
                            break
                    
                    if code is None:
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 10:  # 假设长字符串是代码
                                code = value
                                break
                    
                    total_read += 1
                    # 仅保留：函数级、长度>50、且不含 import/from
                    if code and _is_candidate_function(code):
                        source_codes.append(code)
                except Exception as e:
                    continue
    
    print(f"成功加载 {len(source_codes)} / {total_read} 个候选代码样本(通过静态预筛)")

    # 改为对通过预筛的全集进行打乱遍历（不再固定抽取2000条）
    anchor_codes = list(source_codes)
    random.shuffle(anchor_codes)

    print(f"将遍历 {len(anchor_codes)} 个锚点代码，直到累计写入达到 {max_pairs} 对或遍历完毕")

    # 主生成循环
    successful_pairs = 0
    skipped_env_count = 0  # 环境性跳过计数（ModuleNotFoundError/ImportError）
    pos_count = 0
    neg_simple_count = 0
    neg_hard_count = 0
    qc_fail_count = 0  # QC-1 未通过（非环境依赖）

    stop_early = False
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        for i, anchor_code in enumerate(tqdm(anchor_codes, desc="生成进度")):
            if successful_pairs >= max_pairs:
                stop_early = True
                break
            try:
                func_name = _extract_func_name(anchor_code)
                _log(f"[SAMPLE] 开始处理 {i + 1}/{len(anchor_codes)}: func={func_name}, code_len={len(anchor_code)}")
                # ===== 质量控制阶段 =====
                try:
                    # 构建测试生成提示
                    test_prompt = f"""
# ROLE
You are an expert Python programmer and a senior Quality Assurance (QA) engineer specializing in writing comprehensive unit tests.

# TASK
Your task is to write a set of unit tests for the following Python function using Python's built-in `unittest` library.

# CONSTRAINTS
1.  Create a test class that inherits from `unittest.TestCase`.
2.  Write at least 5 diverse test cases, including edge cases (e.g., empty inputs, zero, negative numbers, special characters if applicable).
3.  Each test case should be a separate method in the test class (e.g., `test_case_1`, `test_edge_case_empty`).
4.  Use `self.assertEqual()`, `self.assertTrue()`, `self.assertFalse()`, etc., for assertions.
5.  DO NOT modify the original function.
6.  ONLY output the raw Python code for the test class. Do not include the original function in your output. Do not include any explanations or markdown formatting.
7.  You may import third-party libraries if needed. Explicitly include all required imports (e.g., `from typing import List`). Avoid network access, external services, heavy file I/O, or long-running operations.

# PYTHON FUNCTION
```python
{anchor_code}
```
                    """
                    _log("[QC] 生成单元测试用例...")
                    t_qc = time.time()
                    # 生成测试用例
                    unit_tests_str = generate_with_retry(llm_model, test_prompt, temperature=0.2)
                    _log(f"[QC] 测试代码生成完成: len={len(unit_tests_str)}, 用时={time.time() - t_qc:.2f}s")
                    
                    # 验证测试用例
                    _log("[QC] 执行单元测试以验证 anchor 代码...")
                    test_passed, test_result = execute_tests(anchor_code, unit_tests_str)
                    _log(f"[QC] 单元测试结果: passed={test_passed}")
                    if not test_passed:
                        # 环境性跳过：缺少第三方库（不计失败）
                        if isinstance(test_result, str) and ("ModuleNotFoundError" in test_result or "ImportError" in test_result):
                            skipped_env_count += 1
                            _log("[QC] 检测到环境性依赖缺失，跳过该样本")
                        else:
                            qc_fail_count += 1
                        continue
                
                except Exception as e:
                    _log(f"[QC] 阶段异常: {e}")
                    qc_fail_count += 1
                    continue
                
                # 每一步写入前都检查剩余额度
                remain = max_pairs - successful_pairs
                if remain <= 0:
                    stop_early = True
                    break
                
                # ===== 正样本生成阶段 =====
                try:
                    # 构建代码重写提示
                    rewrite_prompt = f"""
# ROLE
You are an expert Python programmer specializing in semantic-preserving code refactoring. Your goal is to rewrite the given function to be more efficient, readable, or use a different algorithmic approach, while keeping the functionality identical.

# TASK
Rewrite the following Python function.

# CONSTRAINTS
1.  The new function MUST be functionally identical to the original.
2.  DO NOT change the function name or its signature (the parameters and their names).
3.  You may use third-party libraries when necessary; prefer standard library when feasible, and avoid network access or heavy I/O.
4.  ONLY output the raw Python code for the new function. Do not include any explanations or markdown formatting.

# ORIGINAL CODE
```python
{anchor_code}
```
                    """
                    _log("[POS] 生成功能等价重写代码...")
                    t_pos = time.time()
                    # 生成重写的代码
                    rewritten_code = generate_with_retry(llm_model, rewrite_prompt, temperature=0.7)
                    _log(f"[POS] 重写完成: len={len(rewritten_code)}, 用时={time.time() - t_pos:.2f}s")
                    
                    # 验证重写的代码是否能通过原测试
                    _log("[POS] 使用同一测试验证重写代码...")
                    rewrite_test_passed, rewrite_test_result = execute_tests(rewritten_code, unit_tests_str)
                    _log(f"[POS] 验证结果: passed={rewrite_test_passed}")
                    
                    if rewrite_test_passed and successful_pairs < max_pairs:
                        out_file.write(json.dumps({
                            "code1": anchor_code,
                            "code2": rewritten_code,
                            "label": 1
                        }) + "\n")
                        successful_pairs += 1
                        pos_count += 1
                        _log(f"[POS] 已写入正样本，累计对数={successful_pairs}")
                
                except Exception as e:
                    _log(f"[POS] 阶段异常: {e}")
                    pass
                
                # 再次检查剩余额度
                remain = max_pairs - successful_pairs
                if remain <= 0:
                    stop_early = True
                    break
                
                # ===== 负样本生成阶段 =====
                try:
                    # 简单负样本：从其他代码中随机选择
                    available_negatives = [c for c in source_codes if c != anchor_code]
                    if available_negatives and successful_pairs < max_pairs:
                        negative_code = random.choice(available_negatives)
                        out_file.write(json.dumps({
                            "code1": anchor_code,
                            "code2": negative_code,
                            "label": 0
                        }) + "\n")
                        successful_pairs += 1
                        neg_simple_count += 1
                        _log(f"[NEG] 写入简单负样本，累计对数={successful_pairs}")
                    
                    # 困难负样本：使用增强器
                    try:
                        if successful_pairs < max_pairs:
                            _log("[NEG] 生成困难负样本...")
                            hard_negative_codes = augmentor.create_hard_negative(anchor_code)
                            if hard_negative_codes and isinstance(hard_negative_codes, str) and successful_pairs < max_pairs:
                                hard_negative_code = hard_negative_codes
                                out_file.write(json.dumps({
                                    "code1": anchor_code,
                                    "code2": hard_negative_code,
                                    "label": 0
                                }) + "\n")
                                successful_pairs += 1
                                neg_hard_count += 1
                                _log(f"[NEG] 写入困难负样本，累计对数={successful_pairs}")
                            else:
                                _log("[NEG] 困难负样本生成失败或为空，跳过")
                    except Exception as e:
                        _log(f"[NEG] 困难负样本异常: {e}")
                        pass
                        
                except Exception as e:
                    _log(f"[NEG] 阶段异常: {e}")
                    pass
                    
            except Exception as e:
                _log(f"[SAMPLE] 样本级异常: {e}")
                continue
        
        if stop_early:
            _log(f"[STOP] 达到写入上限 {max_pairs} 对，提前结束遍历")
    
    print(f"====== 生成完成 ======")
    print(f"成功生成了 {successful_pairs} 对样本 (包含正负样本写入统计)")
    print(f"  其中：正样本 {pos_count}，简单负样本 {neg_simple_count}，困难负样本 {neg_hard_count}")
    print(f"环境性跳过样本: {skipped_env_count}")
    print(f"QC未通过样本(非环境原因): {qc_fail_count}")
    print(f"结果已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
