#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型对比评估脚本，用于评估不同模型在代码克隆检测任务上的性能。
支持评估：
1. 微调后的CodeT5
2. 原始未微调的CodeT5
3. CodeBERT
4. GraphCodeBERT

更新：所有模型均使用编码器输出进行评估（不使用投影头），统一使用均值池化策略。
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
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoConfig
from peft import PeftConfig, PeftModel

# 新增：固定HTTP代理
proxy_address = "http://localhost:8888"
os.environ['HTTP_PROXY'] = proxy_address
os.environ['HTTPS_PROXY'] = proxy_address

# 添加项目根目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contrastive_learning.model import RobustEncoder
from contrastive_learning.dataset import CloneDetectionTestDataset, create_test_collate_fn

# 导入原始评估脚本中的函数
try:
    from evaluate import load_config, compute_metrics, find_best_threshold, evaluate_model, load_model
except ImportError:
    # 如果无法直接导入，导入相对路径
    from .evaluate import load_config, compute_metrics, find_best_threshold, evaluate_model, load_model


class CodeBERTEncoder(torch.nn.Module):
    """
    CodeBERT编码器包装类，保持与RobustEncoder相同的接口
    现在统一使用均值池化以确保公平比较
    """
    def __init__(self, model_name="microsoft/codebert-base", projection_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 添加与RobustEncoder兼容的投影层
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, projection_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(projection_dim * 2, projection_dim),
            torch.nn.LayerNorm(projection_dim)
        )
        
        # 记录评估模式
        self.eval_mode = False
    
    def mean_pooling(self, last_hidden_state, attention_mask=None):
        """实现与RobustEncoder相同的均值池化策略"""
        if attention_mask is not None:
            # Create proper mask for mean calculation
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            # Apply mask and calculate mean
            sum_hidden = torch.sum(last_hidden_state * expanded_mask, dim=1)
            token_count = torch.sum(attention_mask, dim=1, keepdim=True)
            return sum_hidden / token_count.clamp(min=1e-9)
        else:
            return torch.mean(last_hidden_state, dim=1)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # 使用均值池化而非CLS token
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        
        # 如果处于评估模式，直接返回归一化的编码器输出
        if self.eval_mode:
            return torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        
        # 投影和归一化
        projected_output = self.projection_head(pooled_output)
        normalized_output = torch.nn.functional.normalize(projected_output, p=2, dim=1)
        
        return normalized_output
    
    def forward_encoder_only(self, input_ids, attention_mask=None, **kwargs):
        """仅返回编码器输出，不经过投影头，用于评估"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs.last_hidden_state
        
        # 使用均值池化
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        
        # 直接L2归一化，跳过投影头
        normalized_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        
        return normalized_output
    
    def eval(self):
        """重写eval方法，设置评估模式标志"""
        self.eval_mode = True
        self.encoder.eval()
        return self
    
    def train(self, mode=True):
        """重写train方法，重置评估模式标志"""
        self.eval_mode = not mode
        self.encoder.train(mode)
        return self


class GraphCodeBERTEncoder(torch.nn.Module):
    """
    GraphCodeBERT编码器包装类，保持与RobustEncoder相同的接口
    现在统一使用均值池化以确保公平比较
    """
    def __init__(self, model_name="microsoft/graphcodebert-base", projection_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 添加与RobustEncoder兼容的投影层
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, projection_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(projection_dim * 2, projection_dim),
            torch.nn.LayerNorm(projection_dim)
        )
        
        # 记录评估模式
        self.eval_mode = False
    
    def mean_pooling(self, last_hidden_state, attention_mask=None):
        """实现与RobustEncoder相同的均值池化策略"""
        if attention_mask is not None:
            # Create proper mask for mean calculation
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            # Apply mask and calculate mean
            sum_hidden = torch.sum(last_hidden_state * expanded_mask, dim=1)
            token_count = torch.sum(attention_mask, dim=1, keepdim=True)
            return sum_hidden / token_count.clamp(min=1e-9)
        else:
            return torch.mean(last_hidden_state, dim=1)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # 使用均值池化而非CLS token
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        
        # 如果处于评估模式，直接返回归一化的编码器输出
        if self.eval_mode:
            return torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        
        # 投影和归一化
        projected_output = self.projection_head(pooled_output)
        normalized_output = torch.nn.functional.normalize(projected_output, p=2, dim=1)
        
        return normalized_output
    
    def forward_encoder_only(self, input_ids, attention_mask=None, **kwargs):
        """仅返回编码器输出，不经过投影头，用于评估"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = outputs.last_hidden_state
        
        # 使用均值池化
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        
        # 直接L2归一化，跳过投影头
        normalized_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        
        return normalized_output
    
    def eval(self):
        """重写eval方法，设置评估模式标志"""
        self.eval_mode = True
        self.encoder.eval()
        return self
    
    def train(self, mode=True):
        """重写train方法，重置评估模式标志"""
        self.eval_mode = not mode
        self.encoder.train(mode)
        return self


def load_model_for_evaluation(model_key, config, args, device):
    """
    根据模型标识加载相应的模型
    
    参数:
        model_key: 模型标识
        config: 配置信息
        args: 命令行参数
        device: 运行设备
    
    返回:
        model: 加载的模型
        tokenizer: 对应的tokenizer
        model_info: 模型信息
    """
    models_config = {
        'finetuned_codet5': {
            'name': 'Salesforce/codet5-base',
            'path': args.model_path,  # 微调后的模型路径
            'is_finetuned': True,
            'description': "微调后的CodeT5模型"
        },
        'base_codet5': {
            'name': 'Salesforce/codet5-base',
            'path': None,  # 不加载适配器，使用原始模型
            'is_finetuned': False,
            'description': "原始未微调的CodeT5模型"
        },
        'codebert': {
            'name': 'microsoft/codebert-base',
            'path': None,
            'is_finetuned': False,
            'description': "原始CodeBERT模型"
        },
        'graphcodebert': {
            'name': 'microsoft/graphcodebert-base',
            'path': None,
            'is_finetuned': False,
            'description': "原始GraphCodeBERT模型"
        }
    }
    
    model_config = models_config[model_key]
    model_name = model_config['name']
    model_path = model_config['path']
    is_finetuned = model_config['is_finetuned']
    description = model_config['description']
    
    logger = logging.getLogger(__name__)
    logger.info(f"加载模型: {model_key} ({description})")
    
    use_quantization = config['model'].get('use_quantization', False)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 根据模型类型加载模型
    if model_key in ['finetuned_codet5', 'base_codet5']:
        # 对于CodeT5模型，使用新的load_model函数
        if model_key == 'finetuned_codet5':
            model, _ = load_model(
                model_path=model_path,
                config=config,
                device=device,
                eval_mode=True,
                encoder_only=True
            )
        else:
            # 初始化原始未微调的CodeT5
            model = RobustEncoder(
                model_name=model_name,
                projection_dim=config['model']['projection_dim'],
                projection_hidden_dim=config['model']['projection_hidden_dim'],
                pooling_strategy=config['model']['pooling_strategy'],
                use_quantization=use_quantization,
                eval_mode=True  # 评估模式，不初始化投影头
            )
    
    elif model_key == 'codebert':
        logger.info("加载CodeBERT模型 (均值池化版本)")
        model = CodeBERTEncoder(
            model_name=model_name,
            projection_dim=config['model']['projection_dim']
        )
    
    elif model_key == 'graphcodebert':
        logger.info("加载GraphCodeBERT模型 (均值池化版本)")
        model = GraphCodeBERTEncoder(
            model_name=model_name,
            projection_dim=config['model']['projection_dim']
        )
    
    # 将模型移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, model_config


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多模型对比评估脚本")
    parser.add_argument('--config', type=str, default='contrastive_learning/configs/default_config.yaml',
                      help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                      help='微调模型权重路径（PEFT适配器或完整模型）')
    parser.add_argument('--test_path', type=str, default=None,
                      help='测试数据集路径，如果不指定则使用配置文件中的值')
    parser.add_argument('--val_path', type=str, default=None,
                      help='验证数据集路径，用于找到最佳阈值')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='批处理大小，如果不指定则使用配置文件中的值')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                      help='输出目录，用于保存评估结果')
    parser.add_argument('--use_quantization', action='store_true',
                      help='是否启用量化（覆盖配置文件中的设置）')
    parser.add_argument('--no_quantization', action='store_true',
                      help='是否禁用量化（覆盖配置文件中的设置）')
    parser.add_argument('--models', type=str, default='all',
                      help='要评估的模型，用逗号分隔 ("finetuned_codet5,base_codet5,codebert,graphcodebert" 或 "all")')
    parser.add_argument('--use_validation_threshold', action='store_true',
                      help='是否使用验证集上找到的最佳阈值进行测试')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
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
    
    # 解析要评估的模型
    if args.models.lower() == 'all':
        selected_models = ['finetuned_codet5', 'base_codet5', 'codebert', 'graphcodebert']
    else:
        selected_models = [m.strip() for m in args.models.split(',')]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "evaluation.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 输出配置
    logger.info(f"配置文件: {args.config}")
    logger.info(f"微调模型路径: {args.model_path}")
    logger.info(f"测试数据集: {config['data']['test_path']}")
    if args.val_path:
        logger.info(f"验证数据集: {args.val_path}")
    logger.info(f"评估模型: {', '.join(selected_models)}")
    if use_quantization:
        logger.info("启用4位量化 (QLoRA)")
    logger.info("使用均值池化策略对所有模型进行评估")
    logger.info("仅使用编码器输出（不使用投影头）")
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载测试数据集（只加载一次）
    logger.info(f"加载测试数据集: {config['data']['test_path']}")
    
    # 为确保公平比较，使用微调模型的tokenizer加载测试数据
    initial_tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    
    test_dataset = CloneDetectionTestDataset(
        data_path=config['data']['test_path'],
        tokenizer=initial_tokenizer,
        max_length=config['data']['max_length']
    )
    
    logger.info(f"测试样本数量: {len(test_dataset)}")
    
    # 验证数据集（如果指定）
    val_dataset = None
    val_thresholds = {}
    if args.use_validation_threshold and args.val_path:
        logger.info(f"加载验证数据集: {args.val_path}")
        val_dataset = CloneDetectionTestDataset(
            data_path=args.val_path,
            tokenizer=initial_tokenizer,
            max_length=config['data']['max_length']
        )
        logger.info(f"验证集样本数量: {len(val_dataset)}")
    
    # 存储所有模型的评估结果
    all_results = {}
    
    # 逐一评估每个选定的模型
    for model_key in selected_models:
        try:
            # 加载当前模型和tokenizer
            model, tokenizer, model_info = load_model_for_evaluation(
                model_key, config, args, device
            )
            
            # 创建测试数据加载器（使用特定模型的tokenizer）
            test_collate_fn = create_test_collate_fn(
                tokenizer=tokenizer,
                max_length=config['data']['max_length']
            )
            
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=config['evaluation']['batch_size'],
                shuffle=False,
                num_workers=config['evaluation'].get('num_workers', 0),
                collate_fn=test_collate_fn
            )
            
            # 如果使用验证集寻找阈值
            threshold = None
            val_metrics = None
            if args.use_validation_threshold and val_dataset:
                # 创建验证数据加载器
                val_collate_fn = create_test_collate_fn(
                    tokenizer=tokenizer,
                    max_length=config['data']['max_length']
                )
                
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=config['evaluation']['batch_size'],
                    shuffle=False,
                    num_workers=config['evaluation'].get('num_workers', 0),
                    collate_fn=val_collate_fn
                )
                
                # 在验证集上评估并找到最佳阈值
                val_labels, val_scores, threshold, val_metrics = evaluate_model(
                    model, val_dataloader, device
                )
                
                val_thresholds[model_key] = threshold
                
                logger.info(f"验证集最佳阈值 ({model_key}): {threshold:.3f}")
                logger.info(f"验证集指标 ({model_key}):")
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # 评估模型
            if args.use_validation_threshold and threshold is not None:
                logger.info(f"使用验证集阈值 {threshold:.3f} 评估 {model_key}")
                labels, scores, final_threshold, metrics = evaluate_model(
                    model, test_dataloader, device, threshold
                )
            else:
                labels, scores, final_threshold, metrics = evaluate_model(
                    model, test_dataloader, device
                )
            
            # 输出结果
            logger.info(f"模型 {model_key} ({model_info['description']}) 评估结果:")
            logger.info(f"最佳阈值: {final_threshold:.3f}")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # 保存相似度分数分布信息
            score_stats = {
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
            
            logger.info(f"相似度分数分布 ({model_key}):")
            logger.info(f"  最小值: {score_stats['min']:.4f}")
            logger.info(f"  最大值: {score_stats['max']:.4f}")
            logger.info(f"  平均值: {score_stats['mean']:.4f}")
            logger.info(f"  标准差: {score_stats['std']:.4f}")
            logger.info(f"  分位数: {score_stats['quantiles']}")
            
            # 保存结果
            all_results[model_key] = {
                'threshold': final_threshold,
                'metrics': metrics,
                'description': model_info['description'],
                'score_stats': score_stats,
                'val_metrics': val_metrics
            }
            
            # 释放GPU内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"评估模型 {model_key} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 生成比较表格
    logger.info("生成模型比较报告...")
    
    # 创建Markdown格式的比较表格
    md_table = "# 模型评估比较报告\n\n"
    md_table += "## 使用仅编码器输出评估（不使用投影头）\n\n"
    md_table += "| 模型 | 描述 | 阈值 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |\n"
    md_table += "|------|------|------|--------|--------|--------|--------|------|\n"
    
    for model_key, result in all_results.items():
        metrics = result['metrics']
        md_table += f"| {model_key} | {result['description']} | {result['threshold']:.3f} | "
        md_table += f"{metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
        md_table += f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} |\n"
    
    # 保存比较表格
    with open(os.path.join(args.output_dir, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md_table)
    
    # 保存详细JSON结果
    import json
    with open(os.path.join(args.output_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评估结果已保存至目录: {args.output_dir}")
    
    return all_results


if __name__ == "__main__":
    main() 