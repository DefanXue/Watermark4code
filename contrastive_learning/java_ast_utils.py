"""
基于AST的Java代码分析工具模块
用于替代正则表达式，提供可靠的Java代码解析和分析
"""

import javalang
import re
from typing import Optional, List, Tuple, Set


class JavaASTAnalyzer:
    """基于AST的Java代码分析器"""
    
    def __init__(self, code: str):
        self.code = code
        self.tree = None
        self.method_node = None
        self._parse()
    
    def _parse(self):
        """解析Java代码为AST"""
        # 1. 尝试直接解析
        try:
            self.tree = javalang.parse.parse(self.code)
            self._extract_method()
            return
        except:
            pass
        
        # 2. 尝试包装为类
        try:
            import_lines = []
            body_lines = []
            for line in self.code.split('\n'):
                if re.match(r"^\s*import\s+", line):
                    import_lines.append(line)
                else:
                    body_lines.append(line)
            wrapped = "\n".join(import_lines) + "\nclass __Tmp__ {\n" + "\n".join(body_lines) + "\n}\n"
            self.tree = javalang.parse.parse(wrapped)
            self._extract_method()
            return
        except:
            pass
        
        # 3. 解析失败
        self.tree = None
    
    def _extract_method(self):
        """从AST中提取第一个方法节点"""
        if not self.tree:
            return
        for path, node in self.tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                self.method_node = node
                return
    
    def is_valid(self) -> bool:
        """是否成功解析"""
        return self.tree is not None and self.method_node is not None
    
    # ========== 签名提取（100%可靠）==========
    def get_return_type(self) -> Optional[str]:
        """获取返回类型"""
        if not self.method_node:
            return None
        if self.method_node.return_type is None:
            return "void"
        return self._type_to_string(self.method_node.return_type)
    
    def get_parameter_types(self) -> List[str]:
        """获取参数类型列表"""
        if not self.method_node:
            return []
        return [self._type_to_string(p.type) for p in self.method_node.parameters]
    
    def get_parameter_names(self) -> List[str]:
        """获取参数名列表"""
        if not self.method_node:
            return []
        return [p.name for p in self.method_node.parameters]
    
    def get_method_name(self) -> Optional[str]:
        """获取方法名"""
        if not self.method_node:
            return None
        return self.method_node.name
    
    def _type_to_string(self, type_node) -> str:
        """将类型节点转换为字符串"""
        if isinstance(type_node, javalang.tree.BasicType):
            return type_node.name
        elif isinstance(type_node, javalang.tree.ReferenceType):
            name = type_node.name
            # 处理泛型
            if type_node.arguments:
                args = ", ".join(self._type_to_string(arg.type) if hasattr(arg, 'type') else str(arg) for arg in type_node.arguments)
                name += f"<{args}>"
            # 处理数组
            if type_node.dimensions:
                name += "[]" * len(type_node.dimensions)
            return name
        else:
            return str(type_node)
    
    # ========== 返回语句分析 ==========
    def get_return_expressions(self) -> List[str]:
        """获取所有return表达式的字符串表示"""
        if not self.method_node:
            return []
        
        returns = []
        for path, node in self.method_node:
            if isinstance(node, javalang.tree.ReturnStatement):
                if node.expression:
                    # 获取返回表达式在源代码中的文本
                    expr_str = self._node_to_string(node.expression)
                    returns.append(expr_str)
        return returns
    
    def _node_to_string(self, node) -> str:
        """将AST节点转换为字符串（简化版）"""
        if isinstance(node, javalang.tree.Literal):
            return node.value
        elif isinstance(node, javalang.tree.MemberReference):
            return node.member
        elif isinstance(node, javalang.tree.MethodInvocation):
            return node.member if node.member else "method"
        else:
            # 对于复杂节点，返回占位符
            return "expression"
    
    # ========== 常量判断 ==========
    def is_definite_constant_return(self) -> bool:
        """返回值是否为明确的常量"""
        returns = self.get_return_expressions()
        if not returns:
            return False
        
        for ret_expr in returns:
            if not self._is_literal(ret_expr):
                return False
        return True
    
    def _is_literal(self, expr: str) -> bool:
        """判断表达式是否为字面量常量"""
        expr = expr.strip()
        # 数字字面量
        if re.match(r'^\d+$', expr) or re.match(r'^\d+\.\d+[fFdD]?$', expr):
            return True
        # 字符串字面量
        if re.match(r'^".*"$', expr) or re.match(r"^'.*'$", expr):
            return True
        # 布尔/null字面量
        if expr in {'true', 'false', 'null'}:
            return True
        return False
    
    # ========== 抛异常分析 ==========
    def is_all_paths_throw(self) -> bool:
        """是否所有路径都抛出异常"""
        if not self.method_node:
            return False
        
        # 简化检查：检查方法体中是否只有throw语句
        body_code = self.code[self.code.find('{'):] if '{' in self.code else self.code
        lines = [l.strip() for l in body_code.split('\n') 
                 if l.strip() and not l.strip().startswith('//') and l.strip() not in {'{', '}'} ]
        
        if not lines:
            return False
        
        return all('throw' in l for l in lines)
    
    # ========== 参数使用分析 ==========
    def get_used_parameters(self) -> Set[str]:
        """获取方法体中使用的参数名（基于文本搜索）"""
        param_names = set(self.get_parameter_names())
        used = set()
        
        body_text = self.code[self.code.find('{')+1:] if '{' in self.code else self.code
        
        for param in param_names:
            if re.search(rf"\b{re.escape(param)}\b", body_text):
                used.add(param)
        
        # 检查this的使用
        if re.search(r"\bthis\b", body_text):
            used.add('this')
        
        return used


def is_definite_constant(expr: str) -> bool:
    """判断表达式是否为明确的编译时常量"""
    expr = expr.strip()
    # 数字字面量
    if re.match(r'^\d+$', expr) or re.match(r'^\d+\.\d+[fFdD]?$', expr):
        return True
    # 字符串字面量
    if re.match(r'^".*"$', expr) or re.match(r"^'.*'$", expr):
        return True
    # 布尔/null字面量
    if expr in {'true', 'false', 'null'}:
        return True
    return False

