# 维度策略对比实验

## 🎯 实验目标

对比4种不同的维度选择策略对水印鲁棒性的影响：

1. **策略1（Baseline）**：随机投影（当前方案）
2. **策略2**：综合得分Top-4（忽略正交性）
3. **策略3**：平衡选择（鲁棒性+正交性）
4. **策略4**：三元顺序优先

---

## 🚀 快速开始

### 运行实验
```bash
cd d:\kyl410\XDF\dimension_strategy_comparison
conda activate XDF
python run_all.py
```

### 分步运行
```bash
python scripts/step1_analyze_dimensions.py     # 分析768个维度
python scripts/step2_select_dimensions.py      # 选择维度并生成方向
python scripts/step3_embed_watermarks.py       # 嵌入水印
python scripts/step4_extract_with_attacks.py   # 攻击并提取
python scripts/step5_analyze_results.py        # 分析结果
python scripts/step6_visualize.py             # 可视化
```

---

## 🔧 修复历史

### 已修复的问题

1. ✅ **目录位置错误** - 移动到正确位置 `d:\kyl410\XDF\dimension_strategy_comparison\`
2. ✅ **Python导入错误** - 修复 `from SrcMarker-main...` 语法
3. ✅ **模块路径错误** - 正确设置 `sys.path`
4. ✅ **函数名错误** - `load_model` → `load_best_model`，修正 `embed_codes` 参数
5. ✅ **类名错误** - `JavaAugmentor` → `JavaCodeAugmentor`，`augment_java()` → `augment()[0]`
6. ✅ **模型路径错误** - 使用绝对路径，与原始水印流程一致
7. ✅ **JavaVariableRenamer初始化错误** - 每次调用时传入code参数
8. ✅ **抽样方式调整** - 改为顺序抽样，与原始水印嵌入保持一致

### 重要说明：测试代码选择

**为什么改用顺序抽样？**

初始实验使用 `random` + `seed=42` 抽样，导致：
- run_0009 对应 `SortMatrix` (数据集索引604)
- 该代码包含匿名内部类，MutableAST不支持
- 与原始水印嵌入的测试集不一致

**改为顺序抽样后**：
- 从数据集第0行开始，依次取前10个
- run_0009 对应 `Frequency` (数据集索引9)
- **与原始水印嵌入的测试集完全一致**
- 避免匿名内部类导致的MutableAST警告

**对应关系**：
```
原始水印嵌入 batch_run_mbxp:
run_0000 = MagicSquareTest (索引0)
run_0001 = DecimalToBinary (索引1)
...
run_0009 = Frequency (索引9)

当前实验（顺序抽样）:
run_0000 = MinCost (索引0)
run_0001 = HeapQueueLargest (索引1)
...
run_0009 = Frequency (索引9) ← 相同！
```

---

### 关键修复：模型路径配置

**配置文件** (`configs/base_config.json`):
```json
{
  "model_dir": "D:\\kyl410\\XDF\\Watermark4code\\best_model",
  "anchors_file": "D:\\kyl410\\XDF\\SrcMarker-main\\contrastive_learning\\datasets\\MBXP\\test_filtered_code.jsonl",
  "num_test_codes": 10,
  "start_index": 0,
  "sample_method": "sequential"  // 顺序抽样，与原始水印嵌入一致
}
```

**脚本修改** (所有step脚本):
```python
# 在main()函数开头添加
project_root = Path(__file__).parent.parent.parent  # d:\kyl410\XDF
os.chdir(project_root)
```

**调用方式**（与原始流程一致）:
```python
model, tokenizer = load_best_model(config['model_dir'])
embeddings = embed_codes(model, tokenizer, code_list, batch_size=..., device=device)
```

---

## 📁 目录结构

```
dimension_strategy_comparison/
├── configs/                          # 配置文件
│   ├── base_config.json             # 基础配置
│   ├── strategy_1_random.json       # 策略1配置
│   ├── strategy_2_top4.json         # 策略2配置
│   ├── strategy_3_balanced.json     # 策略3配置
│   └── strategy_4_triple_order.json # 策略4配置
│
├── data/                             # 数据
│   ├── test_codes.jsonl             # 测试代码（10个）
│   └── dimension_analysis/          # 维度分析结果
│       ├── run_0000_analysis.json
│       └── ...
│
├── results/                          # 实验结果
│   ├── strategy_1_random/
│   ├── strategy_2_top4/
│   ├── strategy_3_balanced/
│   └── strategy_4_triple_order/
│       ├── embedding/               # 嵌入结果
│       │   ├── run_0000/
│       │   │   ├── selected_dimensions.json
│       │   │   ├── final.json
│       │   │   ├── watermarked.java
│       │   │   └── original.java
│       │   └── ...
│       └── extraction/              # 提取结果
│           ├── attack_0.00/
│           ├── attack_0.25/
│           ├── attack_0.50/
│           ├── attack_0.75/
│           └── attack_1.00/
│
├── analysis/                         # 分析结果
│   ├── per_strategy_summary.json    # 各策略汇总
│   ├── comparison_table.json        # 对比表
│   ├── robustness_curves.json       # 鲁棒性曲线数据
│   └── visualizations/              # 可视化图表
│       ├── success_rate_vs_attack.png
│       ├── bit_accuracy_heatmap.png
│       ├── success_rate_comparison.png
│       └── offset_preservation_boxplot.png
│
├── scripts/                          # 执行脚本
│   ├── step1_analyze_dimensions.py
│   ├── step2_select_dimensions.py
│   ├── step3_embed_watermarks.py
│   ├── step4_extract_with_attacks.py
│   ├── step5_analyze_results.py
│   └── step6_visualize.py
│
├── run_all.py                        # 一键运行脚本
└── README.md                         # 本文件
```

---

## 🚀 运行方式

### 方法1：一键运行（推荐）

```bash
cd D:\kyl410\XDF\dimension_strategy_comparison
conda activate XDF
python run_all.py
```

**总耗时：约7-10小时**

### 方法2：分步运行

```bash
cd D:\kyl410\XDF\dimension_strategy_comparison
conda activate XDF

# Step 1: 分析768个维度（约2-3小时）
python scripts/step1_analyze_dimensions.py

# Step 2: 选择维度（约1分钟）
python scripts/step2_select_dimensions.py

# Step 3: 嵌入水印（约1.5-2小时）
python scripts/step3_embed_watermarks.py

# Step 4: 攻击+提取（约3-4小时）
python scripts/step4_extract_with_attacks.py

# Step 5: 分析结果（约1分钟）
python scripts/step5_analyze_results.py

# Step 6: 可视化（约1分钟）
python scripts/step6_visualize.py
```

---

## 📊 评估指标

### 1. 提取成功率
- **无攻击提取成功率**
- **各攻击强度下的提取成功率**（5个级别：0, 0.25, 0.5, 0.75, 1.0）

### 2. 比特准确率
- 提取的比特与原始比特的匹配度
- 完全匹配率（4/4比特正确）

### 3. 偏移保持率
- **符号保持率**：攻击后偏移方向是否保持
- **幅度保持率**：攻击后偏移大小的保留比例

### 4. 鲁棒性曲线
- X轴：攻击强度（rename_ratio）
- Y轴：提取成功率
- 4条曲线对比（每个策略一条）

---

## 🔧 4种策略说明

### 策略1：Random Projection（Baseline）

- **描述**：当前方案，基于secret的随机正交投影
- **方法**：直接使用`derive_directions(secret="XDF")`生成4个随机正交方向
- **特点**：不做维度筛选，完全随机

### 策略2：Top-4 Composite Score

- **描述**：每个代码选择综合得分最高的4个维度，然后正交化
- **方法**：
  1. 计算综合得分：0.5×三元顺序 + 0.4×符号保持 + 0.1×左右顺序
  2. 选择得分最高的4个维度索引
  3. 构造基向量并Gram-Schmidt正交化
- **特点**：优先鲁棒性，忽略正交性

### 策略3：Balanced Selection

- **描述**：贪心选择，优先鲁棒性，但避免高度相关的维度
- **方法**：
  1. 按综合得分排序
  2. 贪心选择：每次选择得分最高且与已选维度相关性<0.90的维度
  3. 如果失败，逐步放宽阈值（0.93, 0.95, 0.98）
  4. 正交化
- **特点**：平衡鲁棒性和正交性

### 策略4：Triple Order Priority

- **描述**：只看三元顺序保持率，选Top-4
- **方法**：
  1. 按三元顺序保持率排序
  2. 选择Top-4维度
  3. 正交化
- **特点**：单一指标，最直接反映攻击鲁棒性

---

## 📈 预期输出

### 分析文件

- `analysis/per_strategy_summary.json`：每个策略在各攻击强度下的详细指标
- `analysis/comparison_table.json`：4个策略的对比表格
- `analysis/robustness_curves.json`：鲁棒性曲线的原始数据

### 可视化图表

1. **success_rate_vs_attack.png**：鲁棒性曲线（核心图表）
   - 4条曲线展示不同策略的鲁棒性
   
2. **bit_accuracy_heatmap.png**：比特准确率热力图
   - 展示各策略在不同攻击强度下的比特准确率
   
3. **success_rate_comparison.png**：成功率对比柱状图
   - 横向对比4个策略的成功率
   
4. **offset_preservation_boxplot.png**：偏移保持率箱线图
   - 展示符号保持率和幅度保持率的分布

---

## ⚠️ 注意事项

1. **环境要求**：必须在XDF环境中运行
   ```bash
   conda activate XDF
   ```

2. **时间预算**：完整实验约需7-10小时，建议提前规划

3. **依赖检查**：
   - Watermark4code模块
   - SrcMarker-main模块
   - GPU加速（推荐）

4. **中断恢复**：如果某个步骤中断，可以从该步骤重新运行，前面的结果会被保留

5. **磁盘空间**：确保有足够的磁盘空间（约5-10GB）

---

## 🐛 故障排除

### 问题1：ModuleNotFoundError

**解决**：确保在XDF环境中运行
```bash
conda activate XDF
```

### 问题2：编码太慢

**原因**：未使用GPU

**检查**：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出`False`，说明在用CPU，耗时会更长（约10-15小时）

### 问题3：某个代码嵌入失败

**影响**：该代码的提取结果会缺失，但不影响其他代码

**处理**：检查`results/strategy_X/embedding/run_XXXX/error.txt`查看错误原因

---

## 📝 实验设计细节

### 测试数据
- **代码数量**：10个
- **来源**：MBXP测试集
- **身份信息**：统一使用`[1, 1, 0, 0]`

### 维度分析
- **分析维度数**：768个（全部）
- **变体数量**：每个代码100个变体
- **攻击次数**：每个代码30次攻击

### 水印嵌入
- **K**：30个候选变体
- **max_iters**：30次迭代
- **quantile**：0.90

### 攻击测试
- **攻击类型**：重命名攻击
- **攻击强度**：5个级别（0, 0.25, 0.5, 0.75, 1.0）
- **重复次数**：每个强度30次（不同随机种子）

---

## 📚 参考

- 鲁棒编码器：`Watermark4code/best_model`
- 变体生成：`SrcMarker-main/contrastive_learning/java_augmentor.py`
- 攻击实现：`Watermark4code/experiments/Attack/Rename_Attack/`
- 维度选择原理：参考`XDF/adaptive_dimension_selection/`的已有分析

---

## ✅ 完成标志

实验成功完成的标志：
1. ✅ `data/dimension_analysis/`包含10个分析文件
2. ✅ `results/strategy_X/embedding/`包含10个嵌入结果
3. ✅ `results/strategy_X/extraction/`包含5×10×30=1500个提取结果
4. ✅ `analysis/visualizations/`包含4张图表
5. ✅ 控制台输出成功率对比表

---

**Good luck! 🚀**

