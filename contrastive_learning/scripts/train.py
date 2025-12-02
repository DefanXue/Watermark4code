#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练入口脚本，负责解析配置并调用Trainer进行模型训练。
"""

import os
import sys
import argparse
import logging
import yaml
from typing import Dict, Any
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from peft import LoraConfig, TaskType

# 添加项目根目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from contrastive_learning.model import RobustEncoder
from contrastive_learning.augmentor import CodeAugmentor
from contrastive_learning.dataset import CloneDetectionTestDataset, create_test_collate_fn
from contrastive_learning.losses import CombinedContrastiveLoss
from contrastive_learning.trainer import ContrastiveTrainer


def set_seed(seed: int):
	"""
	设置随机种子，确保结果可复现。
	
	参数:
		seed: 随机种子
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	

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


def main():
	# 解析命令行参数
	parser = argparse.ArgumentParser(description="训练鲁棒的代码表示编码器")
	parser.add_argument('--config', type=str, default='contrastive_learning/configs/default_config.yaml',
					help='配置文件路径')
	parser.add_argument('--output_dir', type=str, default=None,
					help='输出目录，如果不指定则使用配置文件中的值')
	parser.add_argument('--seed', type=int, default=None,
					help='随机种子，如果不指定则使用配置文件中的值')
	parser.add_argument('--use_quantization', action='store_true',
					help='是否启用量化（覆盖配置文件中的设置）')
	parser.add_argument('--no_quantization', action='store_true',
					help='是否禁用量化（覆盖配置文件中的设置）')
	
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
	if args.output_dir:
		config['training']['output_dir'] = args.output_dir
	
	if args.seed:
		config['training']['seed'] = args.seed
	
	# 处理量化选项
	if args.use_quantization:
		config['model']['use_quantization'] = True
	elif args.no_quantization:
		config['model']['use_quantization'] = False
	
	# 设置随机种子
	set_seed(config['training']['seed'])
	
	# 设置输出目录
	output_dir = config['training']['output_dir']
	os.makedirs(output_dir, exist_ok=True)
	
	# 配置日志
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%Y/%m/%d %H:%M:%S",
		level=logging.INFO,
		handlers=[
			logging.FileHandler(os.path.join(output_dir, "train.log")),
			logging.StreamHandler()
		]
	)
	logger = logging.getLogger(__name__)
	logger.info(f"配置文件: {args.config}")
	logger.info(f"输出目录: {output_dir}")
	
	# 输出完整配置
	logger.info("训练配置:")
	logger.info(yaml.dump(config, default_flow_style=False, allow_unicode=True))
	
	# 初始化 tokenizer
	logger.info(f"加载 tokenizer: {config['model']['name']}")
	tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
	
	# 训练改为使用pairs格式，与评估完全对齐
	logger.info("使用pairs数据(code1, code2, label)进行训练，与评估对齐")
	
	# 加载训练数据集（pairs）
	logger.info(f"加载训练数据集: {config['data']['train_path']}")
	train_dataset = CloneDetectionTestDataset(
		data_path=config['data']['train_path'],
		tokenizer=tokenizer,
		max_length=config['data']['max_length']
	)
	
	# 加载验证数据集（pairs）
	val_dataset = None
	if config['data'].get('val_path'):
		logger.info(f"加载验证数据集: {config['data']['val_path']}")
		val_dataset = CloneDetectionTestDataset(
			data_path=config['data']['val_path'],
			tokenizer=tokenizer,
			max_length=config['data']['max_length']
		)
	
	# 复用评估用collate（pairs）
	train_collate_fn = create_test_collate_fn(
		tokenizer=tokenizer,
		max_length=config['data']['max_length']
	)
	
	# 创建数据加载器
	logger.info("创建数据加载器（使用动态填充策略）")
	train_dataloader = DataLoader(
		train_dataset,
		batch_size=config['training']['batch_size'],
		shuffle=True,
		num_workers=config['training'].get('num_workers', 0),
		pin_memory=True,
		collate_fn=train_collate_fn  # 使用pairs collate
	)
	
	val_dataloader = None
	if val_dataset:
		val_dataloader = DataLoader(
			val_dataset,
			batch_size=config['training']['batch_size'],
			shuffle=False,
			num_workers=config['training'].get('num_workers', 0),
			pin_memory=True,
			collate_fn=train_collate_fn
		)
	
	# 配置 PEFT
	logger.info("配置 PEFT 适配器")
	peft_config = LoraConfig(
		task_type=TaskType.FEATURE_EXTRACTION,
		inference_mode=False,
		r=config['model']['lora_r'],
		lora_alpha=config['model']['lora_alpha'],
		lora_dropout=config['model']['lora_dropout'],
		target_modules=config['model']['target_modules']
	)
	
	# 初始化模型
	logger.info(f"初始化模型: {config['model']['name']}")
	model = RobustEncoder(
		model_name=config['model']['name'],
		peft_config=peft_config,
		projection_dim=config['model']['projection_dim'],
		projection_hidden_dim=config['model']['projection_hidden_dim'],
		pooling_strategy=config['model']['pooling_strategy'],
		use_quantization=config['model'].get('use_quantization', False)  # 传递QLoRA配置
	)
	
	# 优化器将由 ContrastiveTrainer 使用参数分组创建
	logger.info("优化器将使用参数分组（LoRA + 投影头分离学习率）")
	
	# 学习率调度器将由 ContrastiveTrainer 处理（如果需要的话）
	total_steps = len(train_dataloader) * config['training']['num_epochs']
	warmup_steps = int(total_steps * config['training']['warmup_ratio'])
	
	logger.info(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
	lr_scheduler = None  # 由于使用参数分组优化器，暂时不使用学习率调度器
	
	# 初始化双重监督对比损失函数
	logger.info("初始化双重监督对比损失函数（投影InfoNCE + 编码器一致性 + 可选编码器InfoNCE）")
	loss_fn = CombinedContrastiveLoss(
		temperature=config['loss']['temperature'],
		consistency_lambda=config['loss']['consistency_lambda'],
		lambda_warmup_steps=config['loss']['lambda_warmup_steps'],
		normalize_losses=config['loss'].get('normalize_losses', False),
		# 新增encoder端InfoNCE参数
		use_encoder_infonce=config['loss'].get('use_encoder_infonce', True),
		encoder_infonce_lambda=config['loss'].get('encoder_infonce_lambda', 0.5),
		encoder_infonce_warmup_steps=config['loss'].get('encoder_infonce_warmup_steps', 1000),
		temperature_encoder=config['loss'].get('temperature_encoder', config['loss']['temperature']),
		# 归一化策略
		normalization_method=config['loss'].get('normalization_method', 'ema_div'),
		ema_beta=config['loss'].get('ema_beta', 0.99),
		eps=config['loss'].get('eps', 1.0e-8)
	)
	# 挂载 pos_weight 供 trainer 的 BCE 使用（默认1.0）
	loss_fn.pos_weight = config['loss'].get('pos_weight', 1.0)
	
	# 初始化训练器
	logger.info("初始化训练器")
	trainer = ContrastiveTrainer(
		model=model,
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		optimizer=None,  # 将由trainer内部使用参数分组创建
		lr_scheduler=lr_scheduler,
		loss_fn=loss_fn,
		gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
		max_grad_norm=config['training'].get('max_grad_norm', 1.0),
		use_amp=config['training'].get('use_amp', False),
		output_dir=output_dir,
		logging_steps=config['training'].get('logging_steps', 100),
		eval_steps=config['training'].get('eval_steps', 1000),
		save_steps=config['training'].get('save_steps', 1000),
		num_epochs=config['training']['num_epochs'],
		logger=logger,
		optimizer_config=config['optimizer']  # 传入优化器配置用于参数分组
	)
	
	# 开始训练
	logger.info("开始训练")
	metrics = trainer.train()
	
	logger.info("训练完成！")
	return metrics


if __name__ == "__main__":
	main() 