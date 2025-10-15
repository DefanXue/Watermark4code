# 鲁棒代码表示学习

本项目利用对比学习（Contrastive Learning）对预训练的代码大模型进行微调，训练出鲁棒的代码表示编码器。该编码器能够生成高质量的代码表示向量，适用于代码克隆检测等下游任务。

## 项目结构

```
contrastive_learning/
├── __init__.py                # 包初始化
├── model.py                   # 鲁棒编码器模型定义
├── augmentor.py               # 代码增强引擎
├── dataset.py                 # PyTorch数据集
├── losses.py                  # 对比损失函数
├── trainer.py                 # 训练器
├── scripts/                   # 训练和评估脚本
│   ├── __init__.py
│   ├── train.py               # 训练入口脚本
│   └── evaluate.py            # 评估入口脚本
└── configs/                   # 配置文件
    ├── __init__.py
    └── default_config.yaml    # 默认配置
```

## 主要功能

1. **鲁棒编码器模型（model.py）**
   - 基于Hugging Face预训练模型构建（如CodeT5）
   - 支持LoRA和QLoRA参数高效微调
   - 灵活的池化策略和投影头

2. **代码增强引擎（augmentor.py）**
   - 多样化的代码增强策略
   - 支持变量重命名、语句重排序、注释插入等
   - 用于生成对比学习的正样本对

3. **对比损失函数（losses.py）**
   - InfoNCE损失
   - 支持对称对比学习
   - 可调节的温度参数

4. **训练器（trainer.py）**
   - 支持梯度累积
   - 支持混合精度训练
   - 自动保存检查点和日志

## 安装依赖

```bash
pip install -r requirements.txt
```

## QLoRA支持

本项目支持通过QLoRA（4位量化LoRA）进行高效微调，大幅减少显存占用：

- **显存节约**: 相比标准LoRA减少60-70%的显存使用
- **大批量训练**: 允许使用更大的批次大小提高训练效率
- **配置简单**: 通过配置文件或命令行轻松开启/关闭量化

要使用QLoRA，您需要：

1. 确保已安装bitsandbytes和accelerate库
2. 在配置文件中设置`model.use_quantization: true`
3. 或者使用命令行标志`--use_quantization`

## 使用方法

### 训练模型

```bash
# 使用标准LoRA
python -m contrastive_learning.scripts.train \
  --config contrastive_learning/configs/default_config.yaml \
  --output_dir ./output/robust_encoder \
  --no_quantization

# 使用QLoRA (4位量化)
python -m contrastive_learning.scripts.train \
  --config contrastive_learning/configs/default_config.yaml \
  --output_dir ./output/robust_encoder_qlora \
  --use_quantization
```

### 评估模型

```bash
# 评估标准LoRA模型
python -m contrastive_learning.scripts.evaluate \
  --config contrastive_learning/configs/default_config.yaml \
  --model_path ./output/robust_encoder/best_model \
  --test_path datasets/github_java_funcs/test.jsonl \
  --no_quantization

# 评估QLoRA模型
python -m contrastive_learning.scripts.evaluate \
  --config contrastive_learning/configs/default_config.yaml \
  --model_path ./output/robust_encoder_qlora/best_model \
  --test_path datasets/github_java_funcs/test.jsonl \
  --use_quantization
```

## 配置文件

配置文件`default_config.yaml`中包含以下主要部分：

1. **模型配置**：预训练模型名称、投影维度、池化策略、LoRA参数、是否使用量化等
2. **数据配置**：数据路径、最大序列长度等
3. **增强策略**：各种代码增强方法及其应用概率
4. **训练配置**：批量大小、学习率、训练轮数等
5. **评估配置**：评估批量大小等

## 实现细节

### RobustEncoder

鲁棒编码器通过以下步骤构建：

1. 加载预训练代码模型（如CodeT5）作为基础编码器
2. 使用LoRA/QLoRA适配器进行参数高效微调
3. 添加MLP投影头产生最终表示向量
4. 支持多种池化策略：平均池化、CLS池化、最大池化

### 对比学习训练流程

1. 对每个代码片段生成语义相同但语法不同的正样本
2. 将原始代码和增强后的正样本通过编码器获得表示向量
3. 使用InfoNCE损失函数优化模型，使正样本对的表示相似，负样本对的表示不同
4. 定期在验证集上评估模型性能并保存最佳检查点

### 评估方法

使用代码克隆检测任务评估模型性能：

1. 对测试集中的代码对计算余弦相似度
2. 寻找最佳阈值将相似度分数转换为二分类预测
3. 计算准确率、精确率、召回率、F1和AUC等指标 