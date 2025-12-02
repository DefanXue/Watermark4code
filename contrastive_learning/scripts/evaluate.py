#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估入口脚本，用于评估训练好的模型在代码克隆检测任务上的性能。
更新：使用纯编码器输出进行评估，不使用投影头。
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from peft import PeftConfig, PeftModel

# 添加项目根目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contrastive_learning.model import RobustEncoder
from contrastive_learning.dataset import CloneDetectionTestDataset, create_test_collate_fn


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件。
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标。
    
    参数:
        labels: 真实标签
        predictions: 模型预测标签
        scores: 预测分数（用于AUC计算）
        
    返回:
        包含各项指标的字典
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, scores)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    寻找最佳阈值。
    
    参数:
        labels: 真实标签
        scores: 预测分数
        
    返回:
        最佳阈值和对应的指标
    """
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    # 扩大阈值搜索范围，覆盖余弦相似度的全部可能值
    for threshold in np.arange(-1.0, 1.0, 0.01):
        predictions = (scores >= threshold).astype(int)
        metrics = compute_metrics(labels, predictions, scores)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics


def load_model(
    model_path: str, 
    config: Dict[str, Any], 
    device: torch.device,
    eval_mode: bool = True,
    encoder_only: bool = True
) -> Tuple[RobustEncoder, AutoTokenizer]:
    """
    加载模型和分词器。
    
    参数:
        model_path: 模型路径
        config: 配置信息
        device: 运行设备
        eval_mode: 是否以评估模式加载
        encoder_only: 是否只使用编码器部分
        
    返回:
        模型和分词器的元组
    """
    # 检查是否为PEFT模型
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    logger = logging.getLogger(__name__)
    
    use_quantization = config['model'].get('use_quantization', False)
    
    # 加载模型
    if is_peft_model:
        logger.info("检测到PEFT适配器，加载PEFT模型")
        # 加载PEFT配置
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        
        logger.info(f"基础模型: {base_model_name}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 初始化基础模型 - 如果是评估模式，不初始化投影头
        model = RobustEncoder(
            model_name=base_model_name,
            projection_dim=config['model']['projection_dim'],
            projection_hidden_dim=config['model']['projection_hidden_dim'],
            pooling_strategy=config['model']['pooling_strategy'],
            use_quantization=use_quantization,  # 传递QLoRA配置
            eval_mode=eval_mode  # 评估模式时不初始化投影头
        )
        
        # 加载PEFT适配器
        model.encoder = PeftModel.from_pretrained(model.encoder, model_path)
        
        # 如果需要投影头且不是仅编码器模式，尝试加载投影头
        if not encoder_only and not eval_mode and model.projection_head is not None:
            proj_path = os.path.join(model_path, "projection_head.bin")
            if os.path.exists(proj_path):
                logger.info(f"加载投影头: {proj_path}")
                model.projection_head.load_state_dict(torch.load(proj_path, map_location=device))
    else:
        logger.info("加载完整模型")
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        # 初始化模型
        model = RobustEncoder(
            model_name=config['model']['name'],
            projection_dim=config['model']['projection_dim'],
            projection_hidden_dim=config['model']['projection_hidden_dim'],
            pooling_strategy=config['model']['pooling_strategy'],
            use_quantization=use_quantization,  # 传递QLoRA配置
            eval_mode=eval_mode  # 评估模式时不初始化投影头
        )
        
        # 加载模型权重
        model_weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path, map_location=device)
            
            # 如果只使用编码器，过滤掉投影头参数
            if encoder_only:
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                      if not k.startswith("projection_head.")}
                missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"加载编码器部分。缺失键: {missing_keys}, 意外键: {unexpected_keys}")
            else:
                model.load_state_dict(state_dict)
                logger.info(f"加载完整模型: {model_weights_path}")
        else:
            logger.warning(f"模型文件不存在: {model_weights_path}")
    
    return model, tokenizer


def evaluate_model(
    model: RobustEncoder, 
    dataloader: DataLoader, 
    device: torch.device,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """
    评估模型性能。
    
    参数:
        model: 模型
        dataloader: 测试数据加载器
        device: 运行设备
        threshold: 可选的固定阈值，如果提供则使用此阈值
        
    返回:
        (labels, scores, best_threshold, metrics)元组
    """
    model.eval()
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 移动数据到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 提取输入
            inputs1 = {
                "input_ids": batch["input_ids_1"],
                "attention_mask": batch["attention_mask_1"]
            }
            
            inputs2 = {
                "input_ids": batch["input_ids_2"],
                "attention_mask": batch["attention_mask_2"]
            }
            
            # 获取嵌入 - 使用forward_encoder_only方法，确保只使用编码器部分
            embeddings1 = model.forward_encoder_only(**inputs1)
            embeddings2 = model.forward_encoder_only(**inputs2)
            
            # 计算余弦相似度
            similarity_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
            
            # 收集标签和分数
            all_labels.extend(batch["label"].cpu().numpy())
            all_scores.extend(similarity_scores.cpu().numpy())
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 使用提供的阈值或寻找最佳阈值
    if threshold is not None:
        # 使用固定阈值
        predictions = (all_scores >= threshold).astype(int)
        metrics = compute_metrics(all_labels, predictions, all_scores)
        best_threshold = threshold
    else:
        # 寻找最佳阈值
        best_threshold, metrics = find_best_threshold(all_labels, all_scores)
    
    return all_labels, all_scores, best_threshold, metrics


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估鲁棒的代码表示编码器")
    parser.add_argument('--config', type=str, default='contrastive_learning/configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径（PEFT适配器或完整模型）')
    parser.add_argument('--test_path', type=str, default=None,
                        help='测试数据集路径，如果不指定则使用配置文件中的值')
    parser.add_argument('--val_path', type=str, default=None,
                        help='验证数据集路径，用于找到最佳阈值')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批处理大小，如果不指定则使用配置文件中的值')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径，如果不指定则使用默认值')
    parser.add_argument('--use_quantization', action='store_true',
                        help='是否启用量化（覆盖配置文件中的设置）')
    parser.add_argument('--no_quantization', action='store_true',
                        help='是否禁用量化（覆盖配置文件中的设置）')
    parser.add_argument('--encoder_only', action='store_true',
                        help='是否只使用编码器（不使用投影头），默认为True')
    parser.add_argument('--use_validation_threshold', action='store_true',
                        help='是否使用验证集上找到的最佳阈值进行测试')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)

    # 强制使用本地 hf-cache 并启用离线模式（若未预设则写入）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hf_cache = os.path.join(project_root, "hf-cache")
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    
    # 命令行参数覆盖配置文件
    if args.test_path:
        config['data']['test_path'] = args.test_path
    
    if args.batch_size:
        config['evaluation']['batch_size'] = args.batch_size
    
    # 处理量化选项
    if args.use_quantization:
        config['model']['use_quantization'] = True
    elif args.no_quantization:
        config['model']['use_quantization'] = False
        
    use_quantization = config['model'].get('use_quantization', False)
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # 输出配置
    logger.info(f"配置文件: {args.config}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"测试数据集: {config['data']['test_path']}")
    if args.val_path:
        logger.info(f"验证数据集: {args.val_path}")
    if use_quantization:
        logger.info("启用4位量化 (QLoRA)")
    logger.info(f"仅使用编码器: {args.encoder_only}")
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型和分词器
    model, tokenizer = load_model(
        args.model_path, 
        config, 
        device, 
        eval_mode=True, 
        encoder_only=args.encoder_only
    )
    
    # 将模型移至设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    # 变量初始化
    val_threshold = None
    
    # 如果指定了验证集和使用验证集阈值，先在验证集上找到最佳阈值
    if args.use_validation_threshold and args.val_path:
        logger.info(f"在验证集上寻找最佳阈值: {args.val_path}")
        
        # 加载验证数据集
        val_dataset = CloneDetectionTestDataset(
            data_path=args.val_path,
            tokenizer=tokenizer,
            max_length=config['data']['max_length']
        )
        
        # 创建动态填充的collate函数
        val_collate_fn = create_test_collate_fn(
            tokenizer=tokenizer,
            max_length=config['data']['max_length']
        )
        
        # 创建验证数据加载器
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['evaluation'].get('num_workers', 0),
            collate_fn=val_collate_fn
        )
        
        logger.info(f"验证集样本数量: {len(val_dataset)}")
        
        # 在验证集上评估并找到最佳阈值
        _, _, val_threshold, val_metrics = evaluate_model(model, val_dataloader, device)
        
        logger.info(f"验证集最佳阈值: {val_threshold:.3f}")
        logger.info("验证集指标:")
        for metric_name, metric_value in val_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # 加载测试数据集
    logger.info(f"加载测试数据集: {config['data']['test_path']}")
    test_dataset = CloneDetectionTestDataset(
        data_path=config['data']['test_path'],
        tokenizer=tokenizer,
        max_length=config['data']['max_length']
    )
    
    # 创建动态填充的collate函数
    test_collate_fn = create_test_collate_fn(
        tokenizer=tokenizer,
        max_length=config['data']['max_length']
    )
    
    # 创建数据加载器
    logger.info("创建测试数据加载器（使用动态填充策略）")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation'].get('num_workers', 0),
        collate_fn=test_collate_fn  # 使用动态填充的collate函数
    )
    
    logger.info(f"测试样本数量: {len(test_dataset)}")
    
    # 评估模型
    logger.info("开始评估...")
    if args.use_validation_threshold and val_threshold is not None:
        logger.info(f"使用验证集最佳阈值: {val_threshold:.3f}")
        labels, scores, threshold, metrics = evaluate_model(model, test_dataloader, device, val_threshold)
    else:
        labels, scores, threshold, metrics = evaluate_model(model, test_dataloader, device)
    
    # 输出结果
    logger.info(f"最佳阈值: {threshold:.3f}")
    logger.info("评估指标:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # 输出分数分布统计
    logger.info("相似度分数分布:")
    logger.info(f"  最小值: {np.min(scores):.4f}")
    logger.info(f"  最大值: {np.max(scores):.4f}")
    logger.info(f"  平均值: {np.mean(scores):.4f}")
    logger.info(f"  标准差: {np.std(scores):.4f}")
    logger.info(f"  分位数[0.1, 0.25, 0.5, 0.75, 0.9]: {np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9])}")
    
    # 保存结果
    output_file = args.output_file or f"evaluation_results_{os.path.basename(args.model_path)}.json"
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'threshold': threshold,
                'metrics': metrics,
                'config': {
                    'model_path': args.model_path,
                    'test_path': config['data']['test_path'],
                    'val_path': args.val_path if args.val_path else None,
                    'batch_size': config['evaluation']['batch_size'],
                    'use_quantization': use_quantization,
                    'encoder_only': args.encoder_only,
                    'use_validation_threshold': args.use_validation_threshold and val_threshold is not None
                },
                'score_stats': {
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'quantiles': {
                        '10%': float(np.quantile(scores, 0.1)),
                        '25%': float(np.quantile(scores, 0.25)),
                        '50%': float(np.quantile(scores, 0.5)),
                        '75%': float(np.quantile(scores, 0.75)),
                        '90%': float(np.quantile(scores, 0.9))
                    }
                }
            },
            f,
            indent=2
        )
    
    logger.info(f"结果已保存至 {output_file}")
    
    return metrics


if __name__ == "__main__":
    main() 