# CodeBLEU 实现说明

## 实现方式

CodeBLEU 指标已集成到实验流程中，使用与 SrcMarker 论文完全相同的实现（来自 Microsoft CodeXGLUE）。

## CodeBLEU 定义

CodeBLEU = 0.25 × BLEU + 0.25 × Weighted-BLEU + 0.25 × Syntax-Match + 0.25 × Dataflow-Match

- **BLEU**: n-gram 匹配
- **Weighted-BLEU**: 关键字加权的 n-gram 匹配
- **Syntax-Match**: AST 语法树匹配
- **Dataflow-Match**: 数据流图匹配

## 计算位置

### Step 4 (提取阶段)
- 文件：`scripts/step4_extract_with_attacks.py`
- 计算：水印代码 vs 攻击后代码的 CodeBLEU
- 保存：每个提取结果的 JSON 文件中包含 `codebleu` 字段

### Step 5 (分析阶段)
- 文件：`scripts/step5_analyze_results.py`
- 汇总：计算每个攻击类型的平均 CodeBLEU
- 输出：`analysis/per_strategy_summary.json` 和 `analysis/comparison_table.json`

## 适用数据集

✅ **所有三个数据集都支持 CodeBLEU**：
- MBXP
- CSN-Java
- GH-Java

## 输出示例

```json
{
  "attack_type": "rename_0.40",
  "avg_bit_accuracy": 0.85,
  "avg_codebleu": 0.72,
  "success_rate": 0.65
}
```

## 注意事项

- CodeBLEU 计算可能较慢（需要 AST 解析和数据流分析）
- 如果计算失败，`codebleu` 字段为 `null`
- 语言固定为 Java（`lang='java'`）
