# Watermark4code - Java Φ (Secret Feature Library)

本目录实现“步骤 1：设计并生成秘密特征表 Φ（Java，仅插入类）”。

- 语言：Java（仅“插入型”片段；不做等价替换）
- 通道策略：
  - 主通道：T1 未使用私有静态 helper 方法、T5 未使用常量、T6 私有静态内部类
  - 次要冗余通道：T2 文档块（JavaDoc 风格但不绑定元素，放类体内末尾）、T3 单行标记
- 放置策略：尽量插入到“类末尾”以最小化顺序影响；仅在类内部插入
- 语法与告警：生成时附加 @SuppressWarnings("unused")；不强制使用 @Generated；静态语法检查仅在生成阶段执行（无单元测试）
- 私钥：seed = SHA-256(secret_key || package || className || path)（路径规范化，大小写与分隔符统一）；签名长度默认 8 hex

目录结构：

- phi/java/templates/*.tmpl 模板片段（含占位符）
- phi/java/catalog.json 模板清单与类别元信息
- phi/java/generator_config.json 生成配置（K、M、签名长度、解析策略、注释长度限制）
- phi/java/instances/*.json 针对目标类实例化的候选清单（生成器输出）
- tools/generate_phi_java.py 候选生成脚本（含命名冲突与静态语法预检）

用法（示例）：

1) 生成候选清单（不会改动原文件）

```bash
python tools/generate_phi_java.py \
  --secret-key "<YOUR_SECRET>" \
  --java-file "<path/to/YourClass.java>" \
  --out "phi/java/instances"
```

- 默认参数：每类 K=2，总上限 M=6；签名 8 hex；优先用 javac 做解析，找不到则回退括号平衡快速检查
- 输出：在 `phi/java/instances/` 下生成一个以上下文哈希命名的 JSON，包含：
  - context（package、className、seed 等）
  - candidates（各模板生成的插入片段、参数、放置策略、抑制注解、静态预检结果）

注意：
- 仅生成候选并做静态预检，不做真实注入、不改动源文件（注入与精排将在后续步骤执行）
- 敏感词黑名单当前为空（`tools/sensitive_words.json`），未来可按需扩充 