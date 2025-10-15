#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Java代码增强器，用于生成对比学习的正样本和负样本。
严格对标CodeWMBench中的三类扰动生成方式：
1. 语义保持转换（变量重命名、添加注释等）
2. LLM重写（使用与CodeWMBench相同的prompt）
3. 转译攻击（Java→C#→Java）
"""

import random
import re
import os
import time
import json
import threading
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union

# 添加文件锁，确保多线程安全写入
file_lock = threading.Lock()

# 添加API密钥轮换管理
class APIKeyManager:
    """API密钥轮换管理器，在一个密钥达到限额时自动切换到另一个"""
    
    def __init__(self, env_vars=None):
        """
        初始化API密钥管理器
        
        参数:
            env_vars: API密钥环境变量列表，如果为None则使用默认变量
        """
        # 默认使用环境变量GOOGLE_API_KEY和多个备用密钥
        self.env_vars = env_vars or [
            "GOOGLE_API_KEY",           # 主API密钥
            "GOOGLE_API_KEY_BACKUP",    # 备用API密钥1
            "GOOGLE_API_KEY_BACKUP2",   # 备用API密钥2
            "GOOGLE_API_KEY_BACKUP3"    # 备用API密钥3
        ]
        self.current_index = 0
        self._load_keys()
    
    def reload_keys(self):
        """重新从环境变量加载API密钥，用于环境变量更新后刷新密钥列表"""
        self._load_keys()
        
    def _load_keys(self):
        """从环境变量加载API密钥"""
        self.api_keys = []
        for var in self.env_vars:
            key = os.environ.get(var)
            if key:
                self.api_keys.append(key)
        
        if not self.api_keys:
            print(f"警告: 未找到有效的API密钥。请设置环境变量: {', '.join(self.env_vars)}")
        else:
            print(f"已加载 {len(self.api_keys)} 个API密钥")
    
    def get_current_key(self):
        """获取当前使用的API密钥"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_index]
    
    def rotate_key(self):
        """轮换到下一个API密钥"""
        if len(self.api_keys) <= 1:
            print("警告: 没有可用的备用API密钥")
            return False
        
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        print(f"已切换到备用API密钥 {self.current_index + 1}/{len(self.api_keys)}")
        return True

# 全局API密钥管理器实例
api_key_manager = APIKeyManager()

# 线程局部：为本次调用临时覆盖模型名（并发安全）
_model_ctx = threading.local()

def _set_model_override(name: str) -> None:
	try:
		_model_ctx.name_override = name
	except Exception:
		pass

def _clear_model_override() -> None:
	try:
		if hasattr(_model_ctx, "name_override"):
			delattr(_model_ctx, "name_override")
	except Exception:
		pass

# 使用 NewAPI 的模型与基础地址（可由环境变量覆盖）
NEWAPI_MODEL_NAME = os.environ.get("NEWAPI_MODEL", "gpt-5-nano")
NEWAPI_BASE_URL = os.environ.get("NEWAPI_BASE_URL", "https://chat.cloudapi.vip/v1beta")

# 导入安全设置类型
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    # 安全设置（与create_test_set.py保持一致）
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
except ImportError:
    # 如果未安装google.generativeai，定义一个空字典
    SAFETY_SETTINGS = {}

# 导入基础增强器
from contrastive_learning.augmentor import CodeAugmentor


class JavaCodeAugmentor(CodeAugmentor):
    """
    针对Java代码的增强器，继承自通用CodeAugmentor
    实现CodeWMBench中的语义保持转换
    """
    
    def __init__(self, strategies: Optional[Dict[str, float]] = None):
        """
        初始化Java代码增强器
        
        参数:
            strategies: 增强策略字典，mapping策略名称到应用概率
        """
        # Java特定的默认策略
        self.strategies = strategies or {
            "rename_variables": 0.8,          # 变量重命名（不含方法名）
            "insert_comments": 0.4,           # 添加/修改注释
            "reorder_statements": 0.0,        # 重排无依赖语句（已禁用）
            "change_whitespace": 0.7,         # 修改空白/格式
            "change_braces_style": 0.6,       # 修改大括号样式
            "convert_loop": 0.0,              # 转换循环结构（已禁用）
            "add_redundant_parens": 0.5,      # 添加冗余括号
            "transform_boolean_literal": 0.3, # 布尔字面量等价替换
            "transform_zero_literal": 0.2     # 零的等价表达式
        }
        
        # 注册Java特定增强方法
        self.augmentation_methods = {
            "rename_variables": self._rename_java_variables,
            "insert_comments": self._insert_java_comments,
            "change_whitespace": self._change_whitespace,  # 复用基类方法
            "change_braces_style": self._change_java_braces_style,
            "add_redundant_parens": self._add_redundant_parentheses,
            "transform_boolean_literal": self._transform_boolean_literal,
            "transform_zero_literal": self._transform_zero_literal,
        }
    
    def _rename_java_variables(self, code_string: str) -> str:
        """重命名Java变量和方法名"""
        # Java关键字列表
        java_keywords = {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", 
            "class", "const", "continue", "default", "do", "double", "else", "enum", 
            "extends", "final", "finally", "float", "for", "goto", "if", "implements", 
            "import", "instanceof", "int", "interface", "long", "native", "new", "package", 
            "private", "protected", "public", "return", "short", "static", "strictfp", 
            "super", "switch", "synchronized", "this", "throw", "throws", "transient", 
            "try", "void", "volatile", "while", "true", "false", "null"
        }
        
        # 常见Java类名，避免重命名
        common_types = {
            "String", "Integer", "Boolean", "Double", "Float", "List", "Map", 
            "Set", "ArrayList", "HashMap", "Object", "Exception", "RuntimeException"
        }
        
        # 提取变量声明（不包括方法名）
        var_pattern = r'(?<!\.)(\b[a-zA-Z_]\w*\b)(?!\s*\()'  # 变量名（后面不跟括号）
        
        # 识别所有标识符
        identifiers = set(re.findall(var_pattern, code_string))
        
        # 排除Java关键字和常见类型
        identifiers = identifiers - java_keywords - common_types
        
        # 创建重命名映射
        var_mapping = {}
        prefix_options = ["var", "tmp", "arg", "param", "val", "obj", "item"]
        
        # 变量重命名（不重命名方法）
        for name in identifiers:
            # 跳过特殊名称
            if name in ["main", "args"]:
                continue
            
            # 生成新名称
            prefix = random.choice(prefix_options)
            new_name = f"{prefix}_{random.randint(1, 100)}"
            var_mapping[name] = new_name
        
        # 应用重命名（只重命名变量，不重命名方法）
        result = code_string
        
        for old_name, new_name in var_mapping.items():
            # 使用正则确保只替换完整的标识符（不包括方法调用）
            result = re.sub(r'\b' + re.escape(old_name) + r'\b(?!\s*\()', new_name, result)
        
        return result
    
    def _insert_java_comments(self, code_string: str) -> str:
        """在Java代码中插入或修改注释"""
        lines = code_string.split('\n')
        
        # Java风格注释
        comments = [
            "// Process the input data",
            "// Initialize variables",
            "// Update counter",
            "// Check boundary conditions",
            "// Handle edge case",
            "// Main business logic",
            "// Helper method",
            "// Return the result",
            "// Validate parameters",
            "// Apply transformation",
            "// Parse input string",
            "// Calculate result",
            "// Check for null values"
        ]
        
        # 插入注释的次数
        num_insertions = random.randint(1, 3)
        
        # 插入注释
        for _ in range(num_insertions):
            if not lines:
                break
                
            pos = random.randint(0, len(lines) - 1)
            
            # 跳过空行和已有注释的行
            if not lines[pos].strip() or "//" in lines[pos]:
                continue
                
            # 添加注释
            comment = random.choice(comments)
            indentation = re.match(r'^\s*', lines[pos]).group(0)
            
            if random.random() < 0.6:
                # 在行末添加注释
                lines[pos] = lines[pos] + "  " + comment
            else:
                # 在行前添加注释
                lines.insert(pos, indentation + comment)
                
        return '\n'.join(lines)
    
    def _change_java_braces_style(self, code_string: str) -> str:
        """修改Java大括号样式（K&R风格 vs. Allman风格）"""
        # K&R风格: if (condition) {
        # Allman风格: if (condition)
        #            {
        
        # 模式1: 将K&R风格转换为Allman风格
        kr_to_allman = random.random() < 0.5
        
        if kr_to_allman:
            # 匹配行尾的左花括号，将其移到下一行
            pattern = r'(\b(?:if|for|while|switch|try|catch|else)\b.*?)\s*\{'
            
            def replace_kr_to_allman(match):
                return match.group(1) + '\n' + ' ' * (len(match.group(1)) - len(match.group(1).lstrip())) + '{'
            
            return re.sub(pattern, replace_kr_to_allman, code_string)
        else:
            # 匹配单独一行的左花括号，将其移到上一行末尾
            lines = code_string.split('\n')
            result = []
            i = 0
            
            while i < len(lines):
                if i > 0 and lines[i].strip() == '{' and any(keyword in lines[i-1] for keyword in ['if', 'for', 'while', 'switch', 'try', 'catch', 'else']):
                    # 将括号附加到前一行
                    result[-1] = result[-1] + ' {'
                else:
                    result.append(lines[i])
                i += 1
            
            return '\n'.join(result)
    
    def _add_redundant_parentheses(self, code_string: str) -> str:
        """为算术和逻辑表达式添加冗余括号（语义不变）"""
        # 匹配二元运算表达式（避免在字符串/注释中匹配）
        result = code_string
        
        # 模式：变量 运算符 变量/字面量
        patterns = [
            # 算术运算: a + b → (a + b)
            (r'(\w+)\s*\+\s*(\w+)(?!\+)', r'(\1 + \2)'),
            (r'(\w+)\s*-\s*(\w+)(?!>)', r'(\1 - \2)'),
            (r'(\w+)\s*\*\s*(\w+)', r'(\1 * \2)'),
            (r'(\w+)\s*/\s*(\w+)', r'(\1 / \2)'),
            # 比较运算: a < b → (a < b)
            (r'(\w+)\s*<\s*(\w+)(?!=)', r'(\1 < \2)'),
            (r'(\w+)\s*>\s*(\w+)(?!=)', r'(\1 > \2)'),
        ]
        
        # 随机选择1-2个模式应用（避免过度括号化）
        num_transforms = random.randint(1, 2)
        selected_patterns = random.sample(patterns, min(num_transforms, len(patterns)))
        
        for pattern, replacement in selected_patterns:
            # 只替换第一个匹配（避免过度变换）
            result = re.sub(pattern, replacement, result, count=1)
        
        return result
    
    def _transform_boolean_literal(self, code_string: str) -> str:
        """将布尔字面量替换为等价表达式（语义不变）"""
        # 提取字符串字面量的位置（保护区域）
        protected_regions = []
        for match in re.finditer(r'"(?:[^"\\]|\\.)*"', code_string):
            protected_regions.append((match.start(), match.end()))
        for match in re.finditer(r"'(?:[^'\\]|\\.)*'", code_string):
            protected_regions.append((match.start(), match.end()))
        
        def is_protected(pos):
            """检查位置是否在字符串字面量中"""
            for start, end in protected_regions:
                if start <= pos < end:
                    return True
            return False
        
        result = code_string
        
        # 替换 true → !false（50%概率）
        if random.random() < 0.5:
            for match in re.finditer(r'\btrue\b', code_string):
                if not is_protected(match.start()):
                    # 只替换第一个找到的
                    result = result[:match.start()] + '!false' + result[match.end():]
                    break
        
        # 替换 false → !true（50%概率，独立于上面的替换）
        if random.random() < 0.5:
            # 需要重新搜索，因为字符串可能已经改变
            for match in re.finditer(r'\bfalse\b', result):
                if not is_protected(match.start()):
                    result = result[:match.start()] + '!true' + result[match.end():]
                    break
        
        return result
    
    def _transform_zero_literal(self, code_string: str) -> str:
        """将数字0替换为等价算术表达式（语义不变）"""
        # 只匹配赋值语句中的 0: = 0; 或 = 0,  或 = 0)
        pattern = r'=\s*0\s*([;,)])'
        
        def replace_zero(match):
            # 50%概率进行替换
            if random.random() < 0.5:
                # 随机选择等价表达式
                equivalents = [
                    '1 - 1',   # 减法
                    '2 - 2',   # 减法（不同数字）
                    '0 * 1',   # 乘法
                    '1 * 0',   # 乘法（交换）
                ]
                replacement = random.choice(equivalents)
                return f'= {replacement}{match.group(1)}'
            return match.group(0)
        
        result = re.sub(pattern, replace_zero, code_string)
        return result
    
    def create_hard_negative(self, code_string: str) -> str:
        """
        为Java代码生成语义不同但语法相似的困难负样本
        """
        # 替换一些关键操作符
        replacements = {
            " == ": " != ",
            " != ": " == ",
            " > ": " <= ",
            " < ": " >= ",
            " >= ": " < ",
            " <= ": " > ",
            " && ": " || ",
            " || ": " && ",
            "true": "false",
            "false": "true"
        }
        
        # 尝试替换操作符
        modified = code_string
        replacement_made = False
        
        for original, replacement in replacements.items():
            if original in modified:
                # 只替换一次
                pos = modified.find(original)
                modified = modified[:pos] + replacement + modified[pos + len(original):]
                replacement_made = True
                break
        
        # 如果没有找到操作符，尝试修改数值常量
        if not replacement_made:
            # 查找数值常量
            num_pattern = r'\b(\d+)\b'
            matches = list(re.finditer(num_pattern, code_string))
            
            if matches:
                # 选择一个随机数值进行修改
                match = random.choice(matches)
                num = int(match.group(1))
                
                if num == 0:
                    new_num = 1
                elif num == 1:
                    new_num = 0
                else:
                    # 改变符号或加/减一个小值
                    ops = [lambda x: -x, lambda x: x + 1, lambda x: x - 1, lambda x: x * 2]
                    new_num = random.choice(ops)(num)
                
                modified = code_string[:match.start()] + str(new_num) + code_string[match.end():]
                replacement_made = True
        
        # 如果还是没法修改，尝试移除一个重要语句
        if not replacement_made:
            lines = code_string.split('\n')
            for i, line in enumerate(lines):
                # 寻找包含return、赋值或条件语句的行
                if ("return" in line or "=" in line or 
                    "if" in line or "for" in line or 
                    "while" in line) and ";" in line:
                    # 注释掉这行，而不是删除
                    lines[i] = "// " + line
                    modified = '\n'.join(lines)
                    break
        
        return modified


# LLM重写和转译攻击功能，完全对标CodeWMBench的实现

def check_hard_gates(original_code: str, generated_code: str) -> Tuple[bool, str]:
    """
    三条“必要条件”硬门槛：
    1) 返回类型与参数类型恒等（签名解析失败即判失败）；
    2) 返回依赖原始输入/状态（非纯常量/非全路径抛出；允许一跳依赖近似）；
    3) 非全路径抛出/常量返回。
    通过返回 (True, "")，失败返回 (False, 原因)。
    """
    # Stage-0：语法检查（优先 AST 解析；不可用时回退轻量快检）
    def _syntax_ast_check(code: str) -> Tuple[bool, str]:
        # 尝试使用 javalang 做纯语法解析（不编译、不类型检查）
        try:
            import javalang  # type: ignore
        except Exception:
            # 回退到轻量快检
            return _syntax_quick_check(code)

        # 直接解析（候选可能已包含 import/class）
        try:
            javalang.parse.parse(code)
            return True, ""
        except Exception:
            pass

        # 若仅是方法级片段或包含零散 import：提取 import 行，包一层最小类再解析
        try:
            import_lines: List[str] = []
            body_lines: List[str] = []
            for ln in code.split('\n'):
                if re.match(r"^\s*import\s+[^;]+;\s*$", ln):
                    import_lines.append(ln)
                else:
                    body_lines.append(ln)
            wrapped = "\n".join(import_lines) + "\nclass __Tmp__ {\n" + "\n".join(body_lines) + "\n}\n"
            javalang.parse.parse(wrapped)
            return True, ""
        except Exception as e:
            msg = str(e)
            return False, f"SyntaxFail: {msg[:200]}"

    # 轻量语法快检（不依赖外部编译器，仅做显著错误拦截）
    def _syntax_quick_check(code: str) -> Tuple[bool, str]:
        # 0) 非法注释标记：出现 "/ /" 而不是 "//"
        if re.search(r"/\s/", code):
            return False, "SyntaxFail: invalid comment token '/ /'"

        # 1) 括号/花括号/方括号平衡检查
        pairs = {')': '(', ']': '[', '}': '{'}
        stack: List[str] = []
        in_str = False
        esc = False
        for ch in code:
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch in '([{':
                    stack.append(ch)
                elif ch in ')]}':
                    if not stack or stack[-1] != pairs[ch]:
                        return False, "SyntaxFail: unbalanced brackets/braces"
                    stack.pop()
        if in_str or stack:
            return False, "SyntaxFail: unbalanced string or brackets/braces"

        return True, ""

    ok_syn, syn_reason = _syntax_ast_check(generated_code)
    if not ok_syn:
        return False, syn_reason

    # 门槛1：签名比对（仅检查返回类型与参数类型，不强制方法名）
    def extract_signature(code: str):
        pattern = r"(public|private|protected|static|\s)+\s+(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)"
        m = re.search(pattern, code, flags=re.MULTILINE)
        if not m:
            return None, None
        ret_type = m.group(2).strip()
        params = m.group(4).strip()
        param_types = []
        if params:
            for p in params.split(','):
                p = p.strip()
                if p:
                    tm = re.match(r"(\w+(?:<[^>]+>)?)\s+\w+", p)
                    if tm:
                        param_types.append(tm.group(1))
        return ret_type, param_types

    orig_ret, orig_params = extract_signature(original_code)
    gen_ret, gen_params = extract_signature(generated_code)
    if not orig_ret or not gen_ret:
        return False, "门槛1失败: 签名解析失败"
    if orig_ret != gen_ret:
        return False, f"门槛1失败: 返回类型变化 ({orig_ret} -> {gen_ret})"
    if (orig_params or []) != (gen_params or []):
        return False, f"门槛1失败: 参数类型变化 ({orig_params} -> {gen_params})"

    # 门槛2：返回依赖原始输入/状态（粗粒度静态检查）
    def extract_input_deps(code: str):
        deps = set()
        sm = re.search(r"\(([^)]*)\)", code)
        if sm:
            for p in sm.group(1).split(','):
                p = p.strip()
                if p:
                    parts = p.split()
                    if len(parts) >= 2:
                        deps.add(parts[-1])
        deps.add('this')
        return deps

    deps = extract_input_deps(original_code)
    returns = re.findall(r"return\s+([^;]+);", generated_code)
    if returns:
        has_dep = False
        for expr in returns:
            if any(d in expr for d in deps):
                has_dep = True
                break
            if re.search(r"\w+\s*\(", expr):  # 方法调用近似认为可能依赖输入
                has_dep = True
                break
        if not has_dep:
            all_const = all(
                re.match(r"^\s*[\d\"\'\w]+\s*$", e.strip()) or e.strip() in {"null", "true", "false"}
                for e in returns
            )
            if all_const:
                return False, "门槛2失败: 返回为常量且无输入依赖"

    # 门槛3：非全路径抛出
    body_start = generated_code.find('{')
    if body_start != -1:
        body = generated_code[body_start+1:]
        lines = [l.strip() for l in body.split('\n') if l.strip() and not l.strip().startswith('//') and l.strip() != '}']
        if lines and all('throw' in l for l in lines):
            return False, "门槛3失败: 全路径抛出异常"

    # ---- PDG-lite 基线（对所有函数启用）----
    # 目标：return 至少两跳内依赖原始输入/可见状态；且函数体有效使用参数/this
    def _extract_params(src: str) -> List[str]:
        m = re.search(r"\(([^)]*)\)", src, flags=re.MULTILINE)
        names: List[str] = []
        if m:
            raw = m.group(1).strip()
            if raw:
                for seg in raw.split(','):
                    seg = seg.strip()
                    mm = re.match(r"(\w+(?:<[^>]+>)?)\s+(\w+)$", seg)
                    if mm:
                        names.append(mm.group(2))
        return names

    param_names = _extract_params(original_code)
    source_symbols = set(param_names + ['this'])

    # 1) 有效使用：函数体中至少一次出现参数名或 this.
    body_text = generated_code[generated_code.find('{')+1:] if '{' in generated_code else generated_code
    if not any(re.search(rf"\b{re.escape(nm)}\b", body_text) for nm in source_symbols):
        return False, "PDG-BaseNoUse: no parameter/this usage in method body"

    # 2) 两跳依赖：构建简单依赖图 var -> sources (识别两次赋值链)
    assign_pat = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);")
    var_deps: dict[str, set[str]] = {}
    # 初次扫描：直接含有源符号
    for m in assign_pat.finditer(body_text):
        var = m.group(1)
        rhs = m.group(2)
        ids = set(re.findall(r"\b[A-Za-z_]\w*\b", rhs))
        hit = ids & source_symbols
        if hit:
            var_deps.setdefault(var, set()).update(hit)
    # 二次传播：依赖于已知变量的变量
    for _ in range(2):
        changed = False
        for m in assign_pat.finditer(body_text):
            var = m.group(1)
            rhs = m.group(2)
            ids = set(re.findall(r"\b[A-Za-z_]\w*\b", rhs))
            # 依赖已有的变量
            acc: set[str] = set()
            for idt in ids:
                if idt in var_deps:
                    acc.update(var_deps[idt])
                if idt in source_symbols:
                    acc.add(idt)
            if acc:
                prev = var_deps.get(var, set())
                new = prev | acc
                if new != prev:
                    var_deps[var] = new
                    changed = True
        if not changed:
            break

    # 3) return 依赖检查：任意一个 return 表达式需包含源或依赖于源的变量
    ret_ok = False
    for expr in returns:
        ids = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
        if ids & source_symbols:
            ret_ok = True
            break
        for idt in ids:
            if idt in var_deps and (var_deps[idt] & source_symbols):
                ret_ok = True
                break
        if ret_ok:
            break
    if returns and not ret_ok:
        return False, "PDG-BaseNoDep: return does not depend on inputs/state within two hops"

    # NOTE: 本行上方为基础三门槛；以下追加轻量 PDG 快检 + 小契约（仅当原方法命中模式时启用）

    # -------- PDG-lite 与小契约：逐一通知（void + 入参集合迭代 + 元素副作用） --------
    # 轻量解析参数 (类型, 名称)
    def parse_params_with_names(code: str) -> List[Tuple[str, str]]:
        m = re.search(r"\(([^)]*)\)", code, flags=re.MULTILINE)
        if not m:
            return []
        params_src = m.group(1).strip()
        if not params_src:
            return []
        result: List[Tuple[str, str]] = []
        for seg in params_src.split(','):
            seg = seg.strip()
            # 捕获类似: Type<Generic> name 或 Type name
            mm = re.match(r"(\w+(?:<[^>]+>)?)\s+(\w+)$", seg)
            if mm:
                result.append((mm.group(1), mm.group(2)))
        return result

    def is_collection_type(t: str) -> bool:
        t_low = t.lower()
        return ("list" in t_low) or ("collection" in t_low) or ("iterable" in t_low)

    def detect_roles(code: str) -> Tuple[str, str]:
        """返回 (list_param_name, event_like_param_name) 若无法判定返回 ("", "")."""
        params = parse_params_with_names(code)
        list_name = ""
        event_name = ""
        # 优先按类型判定集合
        for tp, nm in params:
            if is_collection_type(tp):
                list_name = nm
                break
        # 退化：若类型未标识集合，但代码中出现 .size()/isEmpty()/get( 对某参数，则认为其为集合
        if not list_name:
            for tp, nm in params:
                if re.search(rf"\b{re.escape(nm)}\s*\.(size|isEmpty|get)\s*\(", code):
                    list_name = nm
                    break
        if list_name:
            # 选一个非集合参数作为 event-like（若仅一个参数则留空）
            for tp, nm in params:
                if nm != list_name:
                    event_name = nm
                    break
        return list_name, event_name

    def has_iteration_over_list(code: str, list_nm: str) -> bool:
        if not list_nm:
            return False
        pat_any = [
            rf"for\s*\(.*:\s*.*{re.escape(list_nm)}.*\)",  # for-each
            rf"{re.escape(list_nm)}\s*\.iterator\s*\(",    # iterator()
            rf"{re.escape(list_nm)}\s*\.forEach\s*\(",     # forEach
            rf"{re.escape(list_nm)}\s*\.stream\s*\(\)\s*\.forEach\s*\(",  # stream().forEach
            rf"for\s*\(\s*int\s+\w+\s*=\s*0;[^;]*{re.escape(list_nm)}\s*\.size\s*\(\)[^;]*;"  # 传统 for i<size
        ]
        return any(re.search(p, code) for p in pat_any)

    def has_element_call_with_event(code: str, event_nm: str) -> bool:
        if not event_nm:
            # 无事件参数时，放宽为存在元素方法调用
            return bool(re.search(r"\.[a-zA-Z_]\w*\s*\(", code))
        # 任意点出现以事件为实参的元素方法调用
        return bool(re.search(rf"\.[a-zA-Z_]\w*\s*\([^)]*\b{re.escape(event_nm)}\b[^)]*\)", code))

    def only_this_listener_path(code: str) -> bool:
        # 存在 this.listener 调用，但无集合迭代
        return ("this.listener" in code) and (not re.search(r"\.iterator\(|forEach\(|stream\(\).*forEach\(|for\s*\(|while\s*\(", code))

    # 仅在原代码命中“逐一通知模式”时启用小契约
    orig_list, orig_event = detect_roles(original_code)
    gen_list, gen_event = detect_roles(generated_code)

    # 原方法是否为逐一通知模式（void + 集合迭代 + 元素副作用）
    is_void = (orig_ret == "void") if (orig_ret is not None) else False
    orig_iter = has_iteration_over_list(original_code, orig_list)
    orig_elem_call = has_element_call_with_event(original_code, orig_event)
    orig_notify_pattern = is_void and orig_list and (orig_iter and orig_elem_call)

    if orig_notify_pattern:
        # 生成代码必须保留：集合迭代 + 元素调用（with event 或宽松）
        gen_iter = has_iteration_over_list(generated_code, gen_list or orig_list)
        gen_elem_call = has_element_call_with_event(generated_code, gen_event or orig_event)
        if not (gen_iter and gen_elem_call):
            # 如果仅出现 this.listener 但无逐一通知，也视为失败
            if only_this_listener_path(generated_code):
                return False, "门槛2-副作用子规则失败: 替换为 this.listener 而未保留逐一通知"
            return False, "门槛2-副作用子规则失败: 未保留入参集合迭代或逐一元素副作用调用"

    # 若未命中模式，不加额外限制，保持通过
    return True, ""

def generate_with_retry(
    model: Any, prompt: str, max_retries: int = 3, temperature: float = 0.2
) -> str:
    """
    使用重试机制调用LLM生成内容，通过NewAPI代理调用Gemini模型
    支持API密钥轮换：当检测到API配额限制时自动切换到备用密钥。
    
    参数:
        model: LLM模型实例
        prompt: 提示字符串
        max_retries: 最大重试次数
        temperature: 生成温度
        
    返回:
        处理后的生成内容字符串
    """
    # 创建生成配置
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": 4096,
    }
    
    # NewAPI的API密钥 - 请在此处替换为您的实际密钥
    NEWAPI_KEY = "sk-ApxCfJ7h1YR0L6nmSHvU0AXbUTs1iA4eiyMVwGaXkHBfJR6W"
    
    # NewAPI服务器地址 - 请替换为您的实际服务器地址
    NEWAPI_SERVER = "你的newapi服务器地址"  # 例如: api.example.com
    
    retry_count = 0
    key_rotations = 0
    max_key_rotations = 3  # 最多尝试轮换密钥次数
    print(f"[LLM] 发起生成: temp={temperature}, prompt_len={len(prompt)}")
    
    while retry_count < max_retries:
        try:
            print(f"[LLM] 尝试第 {retry_count + 1}/{max_retries} 次请求...")
            t0 = time.time()
            
            '''
            # 原有的Google API调用代码（已注释）
            # 使用当前API密钥配置模型
            current_key = api_key_manager.get_current_key()
            if current_key:
                import google.generativeai as genai
                genai.configure(api_key=current_key)
            
            # 调用模型生成内容
            response = model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=SAFETY_SETTINGS
            )
            '''
            
            # 使用NewAPI调用指定模型
            import requests
            import os
            
            # 保存原始代理设置
            original_http_proxy = os.environ.pop('HTTP_PROXY', None)
            original_https_proxy = os.environ.pop('HTTPS_PROXY', None)
            
            try:
                # 构建请求体，根据NewAPI文档示例格式
                payload = {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ]
                }
                
                # 将生成配置添加到请求体
                if generation_config:
                    payload["generationConfig"] = generation_config
                
                # API密钥
                api_key = "sk-ApxCfJ7h1YR0L6nmSHvU0AXbUTs1iA4eiyMVwGaXkHBfJR6W"
                
                # 使用Bearer token认证方式
                # 优先使用线程局部覆盖的模型名，其次回退到默认模型名
                try:
                    model_name_for_call = getattr(_model_ctx, "name_override", None) or NEWAPI_MODEL_NAME
                except Exception:
                    model_name_for_call = NEWAPI_MODEL_NAME
                url = f"{NEWAPI_BASE_URL}/models/{model_name_for_call}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                # 调用API
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(5, 60)
                )
                
                # 检查响应
                response.raise_for_status()
                response_json = response.json()
                
                # 提取文本结果 - 根据NewAPI响应格式调整
                result = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
            finally:
                # 恢复原始代理设置
                if original_http_proxy:
                    os.environ['HTTP_PROXY'] = original_http_proxy
                if original_https_proxy:
                    os.environ['HTTPS_PROXY'] = original_https_proxy
            
            dt = time.time() - t0
            
            # 移除可能存在的Markdown代码块标记
            if result.startswith("```java"):
                result = result.replace("```java", "", 1).strip()
            if result.startswith("```"):
                result = result.replace("```", "", 1).strip()
            if result.endswith("```"):
                result = result[:-3].strip()
                
            print(f"[LLM] 成功返回: text_len={len(result)}, 用时={dt:.2f}s")
            return result
        
        except Exception as e:
            error_str = str(e).lower()
            
            # 检测API配额限制错误
            if ("quota" in error_str or 
                "rate limit" in error_str or 
                "api key not valid" in error_str or
                "api_key_invalid" in error_str):
                
                # 尝试轮换API密钥
                if key_rotations < max_key_rotations and api_key_manager.rotate_key():
                    key_rotations += 1
                    print(f"[LLM] 检测到API限制，已切换密钥 ({key_rotations}/{max_key_rotations})")
                    # 轮换密钥后不增加重试计数
                    continue
            
            # 其他错误或密钥轮换失败后，增加重试计数
            retry_count += 1
            print(f"[LLM] API调用失败，重试 {retry_count}/{max_retries}，错误: {e}")
            time.sleep(5)  # 等待5秒后重试
    
    # 如果所有重试都失败
    raise RuntimeError("所有API调用重试都失败")


def llm_rewrite_java(code_string: str, model, temperature=0.7) -> str:
    """
    使用LLM生成功能等价的Java代码重写版本；追加“硬门槛+功能不变Self-check”，
    生成后执行两次审核，失败抛异常（由上层捕获并跳过样本）。
    """

    # 1) Prompt 模板（5种，保留原有风格集合）
    PROMPT_TEMPLATES = {
        "default": """
# ROLE
You are an expert Java programmer specializing in semantic-preserving code refactoring. Your goal is to rewrite the given function to be more efficient, readable, or use a different algorithmic approach, while keeping the functionality identical.

# TASK
Rewrite the following Java method.

# CONSTRAINTS
1. The new method MUST be functionally identical to the original.
2. DO NOT change the method name, return type, or parameter types/names.
3. You may use standard Java libraries when necessary.
4. ONLY output the raw Java code for the new method. Do not include any explanations or markdown formatting.

# ORIGINAL CODE
```java
{code_string}
```
""",
        "concise_refactor": """
# ROLE
You are an expert Java engineer focused on semantic-preserving refactoring. Your goal is to produce a more concise and maintainable version of the given method without changing its behavior.

# TASK
Rewrite the following Java method to remove redundancy, simplify expressions, and improve readability while preserving all functionality.

# CONSTRAINTS
1. The rewritten method MUST be functionally identical for all inputs and edge cases.
2. DO NOT change the method name, return type, or parameter types/names.
3. Prefer straightforward control flow, clear naming, and idiomatic Java when safe.
4. Do not alter visibility or public APIs; no logging, I/O, or side effects beyond the original.
5. ONLY output the raw Java code for the new method. No explanations or markdown.

# ORIGINAL CODE
```java
{code_string}
```
""",
        "structure_reorder": """
# ROLE
You are a senior Java refactoring specialist. Your goal is to improve structure (e.g., statement ordering, extracting minimal private helpers) without altering observable behavior.

# TASK
Rewrite the following Java method by safely reorganizing code blocks, optionally extracting tiny private helpers, and clarifying control flow, while keeping functionality identical.

# CONSTRAINTS
1. Preserve functional behavior, side effects, and exception semantics exactly.
2. DO NOT change the method name, return type, or parameter types/names.
3. Reordering is allowed only when it does not violate data or side-effect dependencies.
4. Any helper must be private and not change external behavior or APIs.
5. Prefer micro-level algorithmic changes (e.g., replace a loop with a library call, use an equivalent standard algorithm) rather than a complete algorithmic redesign.
6. ONLY output the raw Java code for the new method. No explanations or markdown.

# ORIGINAL CODE
```java
{code_string}
```
""",
        "alt_algorithm_safe": """
# ROLE
You are an expert Java developer. Your goal is to produce an equivalent method that may use an alternative algorithmic approach while strictly preserving behavior and efficiency class.

# TASK
Rewrite the following Java method using an alternative but equivalent approach where appropriate, keeping the same big-O time/space complexity and identical results.

# CONSTRAINTS
1. The rewritten method MUST be functionally identical for all inputs and edge cases.
2. DO NOT change the method name, return type, or parameter types/names.
3. Maintain the same asymptotic complexity; do not introduce worse performance.
4. Use only standard Java libraries; do not add dependencies or change public APIs.
5. ONLY output the raw Java code for the new method. No explanations or markdown.

# ORIGINAL CODE
```java
{code_string}
```
""",
        "robustness_guardrails": """
# ROLE
You are an expert Java maintainer. Your goal is to enhance clarity with minimal inline comments and formatting tweaks, without modifying any behavior.

# TASK
Rewrite the following Java method by slightly clarifying intent (e.g., minimal comments, whitespace) and organizing code, while keeping observable behavior strictly identical.

# CONSTRAINTS
1. The rewritten method MUST be functionally identical, including return values and exception behavior.
2. DO NOT change the method name, return type, or parameter types/names.
3. Do NOT add new branches, guards, or checks that alter control flow or errors.
4. Comments must be minimal and purely explanatory; no logging or I/O.
5. ONLY output the raw Java code for the new method. No explanations or markdown.

# ORIGINAL CODE
```java
{code_string}
```
""",
	}

    # 2) 温度集合（仅 default 模板使用多温度）
    TEMPERATURE_SET = [0.0, 0.2, 0.4, 0.6, 0.8]

    # 3) 模型名集合（硬编码）：默认模型名与 gemini-2.5-flash-lite
    DEFAULT_MODEL_NAME = NEWAPI_MODEL_NAME
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

    # 4) 20 种变体（等概率）：每项为 (prompt_style, temperature, model_name)
    variants = []
    # 当前默认模型：5个 default@温度 + 5个风格@0.7（含 default@0.7）
    for t in TEMPERATURE_SET:
        variants.append(("default", float(t), DEFAULT_MODEL_NAME))
    variants.extend([
        ("default", 0.7, DEFAULT_MODEL_NAME),
        ("concise_refactor", 0.7, DEFAULT_MODEL_NAME),
        ("structure_reorder", 0.7, DEFAULT_MODEL_NAME),
        ("alt_algorithm_safe", 0.7, DEFAULT_MODEL_NAME),
        ("robustness_guardrails", 0.7, DEFAULT_MODEL_NAME),
    ])
	# gemini-2.5-flash-lite：同样的10个
    for t in TEMPERATURE_SET:
        variants.append(("default", float(t), GEMINI_FLASH_LITE))
    variants.extend([
        ("default", 0.7, GEMINI_FLASH_LITE),
        ("concise_refactor", 0.7, GEMINI_FLASH_LITE),
        ("structure_reorder", 0.7, GEMINI_FLASH_LITE),
        ("alt_algorithm_safe", 0.7, GEMINI_FLASH_LITE),
        ("robustness_guardrails", 0.7, GEMINI_FLASH_LITE),
    ])

    # 5) 构建审查对齐的不变式（从原代码轻量提取，用于提示）
    def _extract_sig_for_prompt(src: str):
        m = re.search(r"(public|private|protected|static|\s)+\s+(\w+(?:<[^>]+>)?)\s+\w+\s*\(([^)]*)\)", src, flags=re.MULTILINE)
        if not m:
            return "void", []
        ret = m.group(2).strip()
        params = []
        raw = m.group(3).strip()
        if raw:
            for seg in raw.split(','):
                seg = seg.strip()
                mm = re.match(r"(\w+(?:<[^>]+>)?)\s+\w+$", seg)
                if mm:
                    params.append(mm.group(1))
        return ret, params

    def _detect_roles_for_prompt(src: str):
        # 返回 (list_param_name, event_param_name)
        m = re.search(r"\(([^)]*)\)", src, flags=re.MULTILINE)
        list_nm, event_nm = "", ""
        if not m:
            return list_nm, event_nm
        raw = m.group(1).strip()
        pairs = []
        for seg in raw.split(','):
            seg = seg.strip()
            mm = re.match(r"(\w+(?:<[^>]+>)?)\s+(\w+)$", seg)
            if mm:
                pairs.append((mm.group(1), mm.group(2)))
        # 集合参数
        for tp, nm in pairs:
            tl = tp.lower()
            if ("list" in tl) or ("collection" in tl) or ("iterable" in tl):
                list_nm = nm
                break
        if not list_nm:
            for tp, nm in pairs:
                if re.search(rf"\b{re.escape(nm)}\s*\.(size|isEmpty|get)\s*\(", src):
                    list_nm = nm
                    break
        # 事件参数
        if list_nm:
            for tp, nm in pairs:
                if nm != list_nm:
                    event_nm = nm
                    break
        return list_nm, event_nm

    def _build_audit_invariants_for_prompt(src: str) -> str:
        ret, param_types = _extract_sig_for_prompt(src)
        list_nm, event_nm = _detect_roles_for_prompt(src)
        lines = []
        lines.append(f"- Types: return={ret}; params=[{', '.join(param_types)}]")
        lines.append("- SrcDeps: result MUST depend on original inputs/visible state; constant-only/always-throw forbidden.")
        lines.append("- NormalPath: ensure at least one non-exceptional path.")
        if list_nm:
            ev = event_nm or "<event>"
            lines.append(f"- Iter+SideEffect: explicitly iterate input collection '{list_nm}' and call the element side-effect with '{ev}' for each non-null element; do NOT replace with a single this.listener or remove the iteration.")
        return "\n# AUDIT-ALIGNED INVARIANTS\n" + "\n".join(lines) + "\n"

    audit_block = _build_audit_invariants_for_prompt(code_string)

    # 6) 抽样一个变体并生成（最多两次尝试）
    style, temp_use, model_name = random.choice(variants)
    base_template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["default"]) 
    # 将“方法名不变”的约束弱化为“返回/参数类型不变”（最小侵入：对该句做替换）
    prompt_raw = base_template.format(code_string=code_string)
    prompt_raw = re.sub(r"DO NOT change the method name, return type, or parameter types/names\.",
                        "DO NOT change the return type or parameter types (parameter names may change).",
                        prompt_raw)
    # 追加审查对齐不变式与自检
    prompt_suffix = """

# SELF-CHECK BEFORE OUTPUT
Before outputting, ensure:
- Return type and parameter types are unchanged
- Return value depends on original inputs/state (not constant-only or always-throw)
- Functionality is preserved
If any constraint is violated, regenerate until all are satisfied.
"""
    prompt = prompt_raw + audit_block + prompt_suffix

    last_reason = ""
    for _ in range(2):
        try:
            _set_model_override(model_name)
            generated = generate_with_retry(model, prompt, temperature=float(temp_use))
        finally:
            _clear_model_override()
        ok, reason = check_hard_gates(code_string, generated)
        if ok:
            return generated
        last_reason = reason
    raise RuntimeError(f"LLM rewrite failed hard gates after 2 attempts: {last_reason}")


def retranslate_java(code_string: str, model, intermediate_lang="csharp", temperature=0.3) -> str:
    """
    Java → 中间语言 → Java 的等价转译（方式A：内部等概率抽样20种变体，外部接口不变）。
    """

    # 1) 支持的中间语言集合（按方案：csharp + c/cpp/kotlin/scala）
    INTERMEDIATE_LANGS = ["csharp", "c", "cpp", "kotlin", "scala"]

    # 2) 温度集合（仅 csharp 使用多温度；其它语言固定 0.0）
    TEMPERATURE_SET = [0.1, 0.15, 0.2, 0.25, 0.3]

    # 3) 模型名集合（硬编码）：默认模型名与 gemini-2.5-flash-lite
    DEFAULT_MODEL_NAME = NEWAPI_MODEL_NAME
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

    # 4) Prompt 模板工厂：按语言生成 trans1/trans2 模板（严格保持功能等价、签名不变、仅输出代码）
    def build_trans1_prompt(target_lang: str, code_java: str) -> str:
        return f"""
# ROLE
You are an expert multi-language programmer specialized in precise, semantics-preserving translation.

# TASK
Translate the following Java method to {target_lang} with strictly identical functionality.

# CONSTRAINTS
1. Preserve ALL behavior, side effects, and exception semantics.
2. External contract unchanged: method name, parameter types and order, and return type. Parameter names may change; generics/annotations/modifier orders may vary.
3. The declared throws set must be equivalent (order can differ); do not add or remove checked exceptions.
4. Imports: allow only JDK/Javax standard libraries; do not introduce third-party frameworks or replace original framework types.
5. Do not change package/class or wrap into a new class; keep original context.
6. Keep control-flow skeleton and exception structure; allow minimal statement-level rewrites only.
7. ONLY output the translated method body or the method with the original signature and throws. No package/import/class wrappers, no extra text.
8. Do not add any comments or markers that describe who/how it was generated or which variant it belongs to (e.g., "generated by", "LLM rewrite", "translated", "auto-generated").

# SOURCE CODE (Java)
```java
{code_java}
```
"""

    def build_trans2_prompt(source_lang: str, code_middle: str) -> str:
        return f"""
# ROLE
You are an expert Java programmer specialized in precise back-translation.

# TASK
Translate the following {source_lang} code back to Java with strictly identical functionality.

# CONSTRAINTS
1. Keep the external contract: same method name, parameter types and order, and return type. Parameter names may change; generics/annotations/modifier orders may vary.
2. The declared throws set must be equivalent (order can differ); do not add or remove checked exceptions.
3. Use only JDK/Javax standard libraries; do not introduce third-party frameworks or replace original framework types.
4. Preserve control-flow skeleton and exception structure; allow minimal statement-level rewrites.
5. ONLY output the Java method with the original signature and throws, or only the method body; no package/import/class wrappers and no extra text.
6. Do not add any comments or markers that describe who/how it was generated or which variant it belongs to (e.g., "generated by", "LLM rewrite", "translated", "auto-generated").

# SOURCE CODE ({source_lang})
```
{code_middle}
```
"""

    # 5) 构建 20 种变体（等概率）：每项为 (lang, temperature, model_name)
    variants = []
    # 默认模型：csharp@多温度 + 4语言@0.0
    for t in TEMPERATURE_SET:
        variants.append(("csharp", float(t), DEFAULT_MODEL_NAME))
    variants.extend([
        ("c", 0.0, DEFAULT_MODEL_NAME),
        ("cpp", 0.0, DEFAULT_MODEL_NAME),
        ("kotlin", 0.0, DEFAULT_MODEL_NAME),
        ("scala", 0.0, DEFAULT_MODEL_NAME),
    ])
    # gemini-2.5-flash-lite：相同的集合
    for t in TEMPERATURE_SET:
        variants.append(("csharp", float(t), GEMINI_FLASH_LITE))
    variants.extend([
        ("c", 0.0, GEMINI_FLASH_LITE),
        ("cpp", 0.0, GEMINI_FLASH_LITE),
        ("kotlin", 0.0, GEMINI_FLASH_LITE),
        ("scala", 0.0, GEMINI_FLASH_LITE),
    ])

    # 5.5) 为 trans1/trans2 构建审查对齐不变式（从原 Java 代码）
    def _extract_sig_for_prompt(src: str):
        m = re.search(r"(public|private|protected|static|\s)+\s+(\w+(?:<[^>]+>)?)\s+\w+\s*\(([^)]*)\)", src, flags=re.MULTILINE)
        if not m:
            return "void", []
        ret = m.group(2).strip()
        params = []
        raw = m.group(3).strip()
        if raw:
            for seg in raw.split(','):
                seg = seg.strip()
                mm = re.match(r"(\w+(?:<[^>]+>)?)\s+\w+$", seg)
                if mm:
                    params.append(mm.group(1))
        return ret, params

    def _detect_roles_for_prompt(src: str):
        m = re.search(r"\(([^)]*)\)", src, flags=re.MULTILINE)
        list_nm, event_nm = "", ""
        if not m:
            return list_nm, event_nm
        raw = m.group(1).strip()
        pairs = []
        for seg in raw.split(','):
            seg = seg.strip()
            mm = re.match(r"(\w+(?:<[^>]+>)?)\s+(\w+)$", seg)
            if mm:
                pairs.append((mm.group(1), mm.group(2)))
        for tp, nm in pairs:
            tl = tp.lower()
            if ("list" in tl) or ("collection" in tl) or ("iterable" in tl):
                list_nm = nm
                break
        if not list_nm:
            for tp, nm in pairs:
                if re.search(rf"\b{re.escape(nm)}\s*\.(size|isEmpty|get)\s*\(", src):
                    list_nm = nm
                    break
        if list_nm:
            for tp, nm in pairs:
                if nm != list_nm:
                    event_nm = nm
                    break
        return list_nm, event_nm

    def _build_audit_invariants_for_prompt(src: str) -> str:
        ret, param_types = _extract_sig_for_prompt(src)
        list_nm, event_nm = _detect_roles_for_prompt(src)
        lines = []
        lines.append(f"- Types: return={ret}; params=[{', '.join(param_types)}]")
        lines.append("- SrcDeps: result MUST depend on original inputs/visible state; constant-only/always-throw forbidden.")
        lines.append("- NormalPath: ensure at least one non-exceptional path.")
        if list_nm:
            ev = event_nm or "<event>"
            lines.append(f"- Iter+SideEffect: explicitly iterate input collection '{list_nm}' and call the element side-effect with '{ev}' for each non-null element; do NOT replace with a single this.listener or remove the iteration.")
        return "\n# AUDIT-ALIGNED INVARIANTS\n" + "\n".join(lines) + "\n"

    audit_block = _build_audit_invariants_for_prompt(code_string)

    # 6) 等概率抽样并执行两段转译（最多两次尝试）
    lang, temp_use, model_name = random.choice(variants)
    last_reason = ""
    for _ in range(2):
        try:
            _set_model_override(model_name)
            # trans1: Java → lang
            trans1 = build_trans1_prompt(lang, code_string) + audit_block + "\n# SELF-CHECK BEFORE OUTPUT\nConfirm the above items; if any fails, regenerate and fix within this response. Output only the code.\n"
            middle = generate_with_retry(model, trans1, temperature=float(temp_use))
            # trans2: lang → Java，追加硬门槛与自检说明
            trans2_prompt = build_trans2_prompt(lang, middle) + audit_block + """

# SELF-CHECK BEFORE OUTPUT
Before outputting, ensure:
- Return type and parameter types are unchanged from the original Java code
- Return value depends on original inputs/state (not constant-only or always-throw)
- Functionality is preserved
If any constraint is violated, regenerate until all are satisfied.
"""
            generated = generate_with_retry(model, trans2_prompt, temperature=float(temp_use))
        finally:
            _clear_model_override()

        ok, reason = check_hard_gates(code_string, generated)
        if ok:
            return generated
        last_reason = reason

    raise RuntimeError(f"LLM retranslate failed hard gates after 2 attempts: {last_reason}")


def generate_java_training_data(
    input_file: str,
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    parallel: bool = True,
    workers: int = 4,
    batch_size: int = 50,
    resume: bool = False  # 添加恢复参数
) -> int:
    """
    生成Java对比学习训练数据
    
    参数:
        input_file: 输入Java代码文件路径
        output_file: 输出训练数据文件路径
        model: LLM模型实例
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        max_samples: 最大样本数
        parallel: 是否启用并行处理
        workers: 并行工作线程数量
        batch_size: 每批处理的样本数
        resume: 是否从上次中断点恢复
        
    返回:
        处理的样本总数
    """
    # 统一使用并行实现，避免串行/并行分叉导致的不一致
    parallel = True
    # 如果启用并行处理，调用并行版本
    if parallel:
        return generate_java_training_data_parallel(
            input_file=input_file,
            output_file=output_file,
            model=model,
            split_type=split_type,
            positive_ratio=positive_ratio,
            augmentation_types=augmentation_types,
            max_samples=max_samples,
            num_workers=workers,
            batch_size=batch_size,
            resume=resume  # 传递恢复参数
        )
    
    # 以下是原来的串行版本
    print(f"为{split_type}数据生成增强样本...")
    
    # 默认增强类型概率
    augmentation_types = augmentation_types or {
        "semantic_preserving": 0.4,  # 语义保持转换（变量重命名等）
        "llm_rewrite": 0.4,          # LLM重写（与CodeWMBench一致）
        "retranslate": 0.2           # 转译攻击（Java→C#→Java）
    }
    
    # 根据split_type调整参数
    if split_type == "train":
        # 训练集：多样化，更激进的增强
        temp_multiplier = 1.0
    elif split_type == "valid":
        # 验证集：稍保守
        temp_multiplier = 0.8
    else:  # test
        # 测试集：更严格的评估
        temp_multiplier = 1.2  # 更创新/困难的变体
    
    # 初始化Java代码增强器
    java_augmentor = JavaCodeAugmentor()
    
    # 加载源代码
    source_codes = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    code = data.get("code", "")
                    if code and len(code.strip()) > 0:
                        source_codes.append(code)
                except Exception:
                    continue
    
    # 限制样本数
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个Java代码样本")
    
    # 生成训练数据
    processed = 0
    with open(output_file, 'w', encoding='utf-8') as out_file:
        try:
            for i, anchor_code in enumerate(tqdm(source_codes, desc=f"{split_type}处理进度")):
                # 按比例决定生成正样本还是负样本
                is_positive = random.random() < positive_ratio
                
                try:
                    if is_positive:
                        # 选择增强类型
                        aug_type = random.choices(
                            list(augmentation_types.keys()),
                            weights=list(augmentation_types.values()),
                            k=1
                        )[0]
                        
                        if aug_type == "semantic_preserving":
                            # 使用Java代码增强器生成正样本
                            positive_samples = java_augmentor.augment(anchor_code)
                            if positive_samples:
                                positive_code = positive_samples[0]
                                out_file.write(json.dumps({
                                    "anchor": anchor_code,
                                    "positive": positive_code,
                                    "type": "augment"
                                }) + "\n")
                                processed += 1
                        
                        elif aug_type == "llm_rewrite":
                            # 使用LLM重写生成正样本
                            positive_code = llm_rewrite_java(
                                anchor_code, model, 
                                temperature=0.7 * temp_multiplier
                            )
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "positive": positive_code,
                                "type": "llm_rewrite"
                            }) + "\n")
                            processed += 1
                        
                        elif aug_type == "retranslate":
                            # 使用转译攻击生成正样本
                            positive_code = retranslate_java(
                                anchor_code, model,
                                temperature=0.3 * temp_multiplier
                            )
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "positive": positive_code,
                                "type": "retranslate"
                            }) + "\n")
                            processed += 1
                    
                    else:
                        # 仅使用简单负样本：从全量源代码中随机选择非自身的代码
                        available_negatives = [c for c in source_codes if c is not anchor_code]
                        if available_negatives:
                            negative_code = random.choice(available_negatives)
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "negative": negative_code,
                                "type": "random_negative"
                            }) + "\n")
                            processed += 1
                        
                except Exception as e:
                    print(f"处理样本 {i} 时出错: {e}")
                    continue
        except KeyboardInterrupt:
            print("检测到Ctrl+C，提前结束当前数据集的处理，已安全保存已生成的样本。")
            # 直接跳出循环，with块会确保文件关闭
            pass
    
    print(f"共生成 {processed} 个处理后的样本")
    return processed 


def process_batch(
    batch_codes: List[str],
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    all_codes: List[str] = None
) -> int:
    """
    处理一批代码样本，生成增强样本
    
    参数:
        batch_codes: 待处理的代码样本列表
        output_file: 输出文件路径
        model: LLM模型实例
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        all_codes: 全量源代码池，用于抽取简单负样本
        
    返回:
        成功处理的样本数
    """
    processed = 0
    java_augmentor = JavaCodeAugmentor()
    
    # 回退：如果未提供全量池，则仅使用当前批次作为可选负样本池
    negative_pool = all_codes if all_codes is not None else batch_codes
    
    for anchor_code in batch_codes:
        try:
            # 保持原有的正负样本决策逻辑不变
            is_positive = random.random() < positive_ratio
            
            if is_positive:
                # 选择增强类型
                aug_type = random.choices(
                    list(augmentation_types.keys()),
                    weights=list(augmentation_types.values()),
                    k=1
                )[0]
                
                if aug_type == "semantic_preserving":
                    # 使用Java代码增强器生成正样本（统一出口：三阶段审核）
                    positive_samples = java_augmentor.augment(anchor_code)
                    if not positive_samples:
                        continue
                    selected = None
                    for cand in positive_samples:
                        try:
                            ok, reason = check_hard_gates(anchor_code, cand)
                            if ok:
                                selected = cand
                                break
                        except Exception:
                            continue
                    if selected is None:
                        continue
                    positive_code = selected
                    result = {
                        "anchor": anchor_code,
                        "positive": positive_code,
                        "type": "augment"
                    }
                    processed += 1
                
                elif aug_type == "llm_rewrite":
                    # 使用LLM重写生成正样本
                    temp_multiplier = 1.0 if split_type == "train" else 0.8 if split_type == "valid" else 1.2
                    positive_code = llm_rewrite_java(
                        anchor_code, model, 
                        temperature=0.7 * temp_multiplier
                    )
                    result = {
                        "anchor": anchor_code,
                        "positive": positive_code,
                        "type": "llm_rewrite"
                    }
                    processed += 1
                
                elif aug_type == "retranslate":
                    # 使用转译攻击生成正样本
                    temp_multiplier = 1.0 if split_type == "train" else 0.8 if split_type == "valid" else 1.2
                    positive_code = retranslate_java(
                        anchor_code, model,
                        temperature=0.3 * temp_multiplier
                    )
                    result = {
                        "anchor": anchor_code,
                        "positive": positive_code,
                        "type": "retranslate"
                    }
                    processed += 1
            
            else:
                # 仅使用简单负样本：从全量池随机选择非自身代码
                candidates = [c for c in negative_pool if c is not anchor_code]
                if not candidates:
                    continue
                negative_code = random.choice(candidates)
                result = {
                    "anchor": anchor_code,
                    "negative": negative_code,
                    "type": "random_negative"
                }
                processed += 1
        
            # 使用锁安全写入文件
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + "\n")
                    
        except Exception as e:
            print(f"处理样本时错误: {e}")
            continue
    
    return processed


def generate_java_training_data_parallel(
    input_file: str,
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    num_workers: int = 48,  # 保持原有默认值
    batch_size: int = 50,
    resume: bool = False  # 添加恢复参数
) -> int:
    """
    并行版本的Java训练数据生成，支持中断恢复
    
    参数:
        input_file: 输入Java代码文件路径
        output_file: 输出训练数据文件路径
        model: LLM模型实例
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        max_samples: 最大样本数
        num_workers: 并行工作线程数量
        batch_size: 每批处理的样本数
        resume: 是否从上次中断点恢复
        
    返回:
        处理的样本总数
    """
    print(f"为{split_type}数据并行生成增强样本（使用{num_workers}个工作线程）...")
    
    # 进度记录文件路径
    progress_file = f"{output_file}.progress"
    
    # 默认增强类型概率
    augmentation_types = augmentation_types or {
        "semantic_preserving": 0.4,
        "llm_rewrite": 0.4,
        "retranslate": 0.2
    }
    
    # 加载源代码
    source_codes = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    code = data.get("code", "")
                    if code and len(code.strip()) > 0:
                        source_codes.append(code)
                except Exception:
                    continue
    
    # 限制样本数
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个Java代码样本")
    
    # 将数据分成多个批次
    total_samples = len(source_codes)
    batches = [source_codes[i:i+batch_size] for i in range(0, total_samples, batch_size)]
    
    # 加载上次处理进度
    processed_batches = set()
    total_processed = 0
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                processed_batches = set(progress_data.get('processed_batches', []))
                total_processed = progress_data.get('total_processed', 0)
                print(f"恢复进度：已处理 {len(processed_batches)}/{len(batches)} 批次，共 {total_processed} 样本")
        except Exception as e:
            print(f"读取进度文件失败: {e}，将从头开始处理")
            processed_batches = set()
            total_processed = 0
    
    # 如果首次运行或需要重新开始，清空输出文件
    if not processed_batches:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    # 共享状态：所有API密钥是否已用尽
    all_keys_exhausted = threading.Event()
    
    # 记录进度的函数
    def save_progress():
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_batches': list(processed_batches),
                'total_processed': total_processed,
                'total_batches': len(batches),
                'timestamp': time.time()
            }, f)
    
    # 处理单个批次的函数
    def process_batch_with_progress(batch_index, batch_codes):
        nonlocal total_processed
        
        # 如果批次已处理过或所有API密钥已用尽，则跳过
        if batch_index in processed_batches or all_keys_exhausted.is_set():
            return 0
        
        try:
            # 调用原始的process_batch函数
            result = process_batch(
                batch_codes,
                output_file,
                model,
                split_type,
                positive_ratio,
                augmentation_types,
                all_codes=source_codes
            )
            
            # 更新进度
            with threading.Lock():
                processed_batches.add(batch_index)
                total_processed += result
                # 每完成10个批次保存一次进度
                if len(processed_batches) % 10 == 0:
                    save_progress()
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            # 检测是否所有API密钥都用尽
            if ("quota" in error_str or "rate limit" in error_str or 
                "api key not valid" in error_str or "api_key_invalid" in error_str):
                
                # 尝试轮换所有可用的API密钥
                keys_tried = 0
                max_keys = len(api_key_manager.api_keys)
                
                while keys_tried < max_keys:
                    if api_key_manager.rotate_key():
                        # 密钥轮换成功，再次尝试处理
                        try:
                            return process_batch(
                                batch_codes, output_file, model, 
                                split_type, positive_ratio, augmentation_types,
                                all_codes=source_codes
                            )
                        except Exception:
                            # 如果新密钥也失败，继续轮换
                            keys_tried += 1
                    else:
                        # 轮换失败，表示没有更多密钥
                        keys_tried = max_keys
                
                # 所有密钥都尝试过仍失败，设置标志
                print(f"\n[警告] 所有API密钥都已达到限额或无效，将保存进度并终止")
                all_keys_exhausted.set()
                # 确保保存最新进度
                save_progress()
                return 0
            
            # 其他错误，记录日志但继续执行
            print(f"批次 {batch_index} 处理失败: {e}")
            return 0
    
    # 并行处理未完成的批次
    remaining_batches = [(i, batch) for i, batch in enumerate(batches) if i not in processed_batches]
    
    with tqdm(total=len(batches), desc=f"{split_type}批次处理进度", initial=len(processed_batches)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有未处理的批次
            future_to_batch = {
                executor.submit(process_batch_with_progress, i, batch): i 
                for i, batch in remaining_batches
            }
            
            # 处理完成的任务
            try:
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    
                    # 检查是否所有API密钥都用尽
                    if all_keys_exhausted.is_set():
                        print("\n所有API密钥都已用尽，终止处理")
                        # 取消剩余任务
                        for f in future_to_batch:
                            if not f.done():
                                f.cancel()
                        break
                    
                    try:
                        # 这个批次在process_batch_with_progress中已更新过进度
                        future.result()  # 忽略返回值，因为已在函数内更新
                        pbar.update(1)
                        pbar.set_postfix({"已处理": total_processed, "批次": f"{len(processed_batches)}/{len(batches)}"})
                    except Exception as exc:
                        # 异常已在process_batch_with_progress中处理
                        pbar.update(1)
            
            except KeyboardInterrupt:
                # 捕获Ctrl+C，保存进度后终止
                print("\n检测到Ctrl+C，保存进度并终止...")
                save_progress()
                
                # 尝试取消未完成的任务
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
                
                print(f"\n进度已保存，可使用相同命令恢复。当前完成: {len(processed_batches)}/{len(batches)} 批次")
                raise
    
    # 最终保存进度
    save_progress()
    
    print(f"共生成 {total_processed} 个处理后的样本 ({len(processed_batches)}/{len(batches)} 批次)")
    return total_processed 