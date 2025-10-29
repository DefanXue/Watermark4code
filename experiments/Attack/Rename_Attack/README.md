# Rename Attack 模块

这个模块实现了针对代码水印的重命名攻击，用于测试水印的鲁棒性。

## 核心功能

### 1. `AttackConfig`
配置重命名攻击的参数：
- `naming_strategy`: 命名策略
  - `'random'`: 随机字母数字组合 (例如: `var_a3b2`)
  - `'sequential'`: 顺序命名 (例如: `v0`, `v1`, `v2`)
  - `'obfuscated'`: 混淆命名 (例如: `l`, `O`, `I` 等易混淆字符)
- `preserve_semantics`: 是否保持语义 (默认: True)
- `seed`: 随机种子 (默认: 42)

### 2. `JavaVariableRenamer`
Java 变量重命名器：
- 自动提取代码中的变量名
- 根据策略生成新的变量名
- 保留 Java 关键字和常用方法名
- 使用词边界匹配避免部分替换

## 使用示例

```python
from Watermark4code.experiments.Attack.Rename_Attack import JavaVariableRenamer, AttackConfig

# 原始代码
code = """
public static int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
"""

# 方法1: 使用配置类
config = AttackConfig(naming_strategy='random', seed=42)
renamer = JavaVariableRenamer(code)
attacked_code = renamer.apply_renames(config)

# 方法2: 使用便捷函数
from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import rename_variables
attacked_code = rename_variables(code, strategy='sequential', seed=42)
```

## 命名策略对比

### Random 策略
```java
public static int factorial(int var_a3b2) {
    int var_x7k9 = 1;
    for (int var_m2n4 = 1; var_m2n4 <= var_a3b2; var_m2n4++) {
        var_x7k9 *= var_m2n4;
    }
    return var_x7k9;
}
```

### Sequential 策略
```java
public static int factorial(int v0) {
    int v1 = 1;
    for (int v2 = 1; v2 <= v0; v2++) {
        v1 *= v2;
    }
    return v1;
}
```

### Obfuscated 策略
```java
public static int factorial(int l0) {
    int O1 = 1;
    for (int I2 = 1; I2 <= l0; I2++) {
        O1 *= I2;
    }
    return O1;
}
```

## 测试

运行测试文件验证功能：
```bash
python -m pytest Watermark4code/experiments/Attack/Rename_Attack/test_rename.py
```

## 注意事项

1. **语义保持**: 所有重命名都保持代码语义不变
2. **关键字保护**: Java 关键字和常用方法名不会被重命名
3. **词边界匹配**: 使用正则表达式词边界避免部分替换
4. **可重现性**: 使用 seed 参数确保结果可重现

