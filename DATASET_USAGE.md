# 数据集选择使用说明

## 支持的数据集

本项目现在支持三个数据集：

1. **MBXP** - 多语言基准测试集（默认）
2. **CSN-Java** - CodeSearchNet Java 数据集
3. **GH-Java** - GitHub Java 函数数据集

## 切换数据集

### 方法：修改配置文件

编辑 `configs/base_config.json`，修改 `dataset.name` 字段：

```json
{
  "dataset": {
    "name": "mbxp",  // 改为 "csn-java" 或 "gh-java"
    "paths": {
      "mbxp": "...",
      "csn-java": "...",
      "gh-java": "..."
    }
  }
}
```

## 数据集统计

| 数据集 | 测试集样本数 | 数据路径 |
|--------|-------------|----------|
| MBXP | 643 | `SrcMarker-main/contrastive_learning/datasets/MBXP/test_filtered_code.jsonl` |
| CSN-Java | 11,826 | `SrcMarker-main/contrastive_learning/datasets/csn_java/test_filtered_code.jsonl` |
| GH-Java | 265 | `SrcMarker-main/datasets/github_java_funcs/test_filtered_code.jsonl` |

## 注意事项

- 所有数据集都经过相同的质量筛选标准
- 数据格式统一为 `{"code": "..."}`
- 切换数据集后需要重新运行实验流程
