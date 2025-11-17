# Strategy 5: 可学习方向矩阵 - 运行指南

## 📋 概述

Strategy 5 通过端到端学习优化方向矩阵W，使其最大化符号保持性和簇分离性，是对现有4种基于固定维度选择策略的理论性升级。

---

## 🎯 核心原理

### **与现有策略的对比**

```
策略1-4：从768个固定维度中选择4个
  - 基于预定义指标（符号保持率、簇稳定性等）
  - 方向是one-hot向量的正交化
  - 例如：[0,0,...,1,0,...,0] → 只看第i维

策略5：学习4个任意方向
  - 端到端优化损失函数
  - 方向是768维空间中的任意单位向量
  - 例如：[0.3, -0.1, 0.05, ..., 0.2] → 所有维度的线性组合
```

### **优化目标（损失函数）**

```python
L_total = λ1 * L_sign_preservation      # 符号保持（最重要）
          + λ2 * L_cluster_separation   # 簇分离（有空间嵌入）
          + λ3 * L_orthogonality        # 正交性（避免冗余）
          + λ4 * L_compactness          # 紧致性（辅助）

默认权重：λ1=10.0, λ2=1.0, λ3=0.1, λ4=0.01
```

---

## 🚀 运行步骤

### **前置条件**

确保已完成 Step 1（维度分析），即存在：
```
dimension_strategy_comparison/data/dimension_analysis/run_0000_analysis.json
...
dimension_strategy_comparison/data/dimension_analysis/run_0029_analysis.json
```

如果没有，请先运行：
```bash
cd XDF/dimension_strategy_comparison
python scripts/step1_analyze_dimensions.py
```

---

### **Step 1b: 训练可学习方向矩阵**

```bash
cd XDF/dimension_strategy_comparison
python scripts/step1b_train_learned_directions.py
```

**预期输出**：
```
================================================================================
Step 1b: 训练可学习方向矩阵
================================================================================
使用设备: cuda

加载训练数据...
  加载编码器...
  加载 30 个代码...
  初始化变体生成器...
  处理代码: 100%|██████████| 30/30 [XX:XX<00:00]
训练数据: 30 个代码，每个100个变体+100个攻击样本

开始训练...
  Epoch 1/100: Loss=X.XXXX (Sign=X.XXXX, Sep=X.XXXX, Orth=X.XXXX, Comp=X.XXXX)
  Epoch 10/100: Loss=X.XXXX ...
  ...
  Epoch 100/100: Loss=X.XXXX ...

✓ 已保存W矩阵到: dimension_strategy_comparison/results/strategy_5_learned/trained_W.pth
✓ 已保存训练日志到: dimension_strategy_comparison/results/strategy_5_learned/training_log.json

================================================================================
训练完成！
最终损失: X.XXXX
================================================================================
```

**预计时间**：10-20分钟（取决于GPU）

---

### **Step 2: 选择维度（加载trained_W）**

```bash
python scripts/step2_select_dimensions.py
```

这会处理所有5个策略，包括Strategy 5。对于Strategy 5，它会：
1. 加载 `results/strategy_5_learned/trained_W.pth`
2. 为每个run创建 `results/strategy_5_learned/embedding/run_XXXX/selected_dimensions.json`

**输出示例**：
```
处理策略: strategy_5_learned
  描述: 端到端学习方向矩阵，优化符号保持和簇分离
  run_0000: 加载学习到的方向矩阵 (4, 768) ✓
  run_0001: 加载学习到的方向矩阵 (4, 768) ✓
  ...
```

---

### **Step 3-6: 嵌入、攻击、提取、分析（无需修改）**

```bash
# Step 3: 嵌入水印
python scripts/step3_embed_watermarks.py

# Step 4: 攻击并提取
python scripts/step4_extract_with_attacks.py --concurrency 5

# Step 5: 分析结果
python scripts/step5_analyze_results.py

# Step 6: 可视化
python scripts/step6_visualize.py
```

---

## 📊 结果对比

运行完成后，查看：
```
dimension_strategy_comparison/analysis/comparison_table.json
```

对比5种策略在不同攻击强度下的成功率：
```json
{
  "strategy_1_random": {
    "attack_0.00": 1.00,
    "attack_1.00": 0.328
  },
  "strategy_2_top4": {...},
  "strategy_3_balanced": {...},
  "strategy_4_triple_order": {...},
  "strategy_5_learned": {
    "attack_0.00": 1.00,
    "attack_1.00": ???  // ← 期望 > 0.556 (超越strategy_3)
  }
}
```

---

## 🔧 调整参数

如需调整训练参数，编辑 `configs/strategy_5_learned.json`：

```json
{
  "learning": {
    "epochs": 100,              // 训练轮数（增加可能提升性能）
    "learning_rate": 0.001,     // 学习率
    "loss_weights": {
      "sign_preservation": 10.0,  // ← 最重要，权重最大
      "cluster_separation": 1.0,
      "orthogonality": 0.1,
      "compactness": 0.01
    }
  }
}
```

修改后需重新运行 Step 1b。

---

## ⚠️ 注意事项

1. **GPU推荐**：虽然可以在CPU上运行，但GPU会快很多（~10分钟 vs ~1小时）
2. **数据一致性**：确保使用的是30个代码的dimension_analysis数据（与base_config一致）
3. **断点继续**：如果训练中断，需要重新运行Step 1b（暂不支持断点恢复）

---

## 📝 文件结构

```
dimension_strategy_comparison/
├── configs/
│   └── strategy_5_learned.json          [新建] 配置文件
│
├── scripts/
│   ├── step1b_train_learned_directions.py  [新建] 训练脚本
│   └── step2_select_dimensions.py          [修改] 添加strategy_5支持
│
└── results/
    └── strategy_5_learned/
        ├── trained_W.pth                [输出] 4x768方向矩阵
        ├── training_log.json            [输出] 损失曲线
        └── embedding/
            └── run_XXXX/
                └── selected_dimensions.json  [输出] 与其他策略格式一致
```













