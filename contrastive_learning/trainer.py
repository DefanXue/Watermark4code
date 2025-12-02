"""
Trainer for contrastive learning of robust code representations.
"""

import os
import time
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from transformers import get_scheduler, PreTrainedModel
from peft import PeftModel

from .model import RobustEncoder
from .losses import InfoNCELoss, CombinedContrastiveLoss


class ContrastiveTrainer:
    """
    Trainer for contrastive learning of robust code representations.
    支持双重监督训练与编码器/投影头分离。
    """
    
    def __init__(
        self,
        model: RobustEncoder,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        output_dir: str = "./output",
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        num_epochs: int = 10,
        logger: Optional[logging.Logger] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The RobustEncoder model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            loss_fn: Loss function (defaults to CombinedContrastiveLoss)
            device: Device to train on (defaults to cuda if available)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            use_amp: Whether to use automatic mixed precision
            output_dir: Directory to save model checkpoints
            logging_steps: Number of steps between logging
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            num_epochs: Number of epochs to train for
            logger: Logger instance
            optimizer_config: Configuration for optimizer parameter groups
        """
        # Set up device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        
        # Initialize logging first (needed for optimizer creation)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logger or self._setup_logger()
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup optimizer with parameter groups if not provided
        if optimizer is None and optimizer_config is not None:
            self.optimizer = self._create_optimizer_with_groups(model, optimizer_config)
        else:
            self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=5e-5)
        
        self.lr_scheduler = lr_scheduler
        
        # Loss function - 更新为支持双重监督的组合损失函数
        if loss_fn is None:
            self.loss_fn = CombinedContrastiveLoss(
                temperature=0.07,
                consistency_lambda=0.1,  # 默认λ值
                lambda_warmup_steps=1000,  # 默认λ warmup步数
                normalize_losses=True  # 默认启用损失归一化
            )
        else:
            self.loss_fn = loss_fn
        
        # Training settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        # Output and logging settings
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.num_epochs = num_epochs
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        
        # 新增: 详细损失记录
        self.detailed_losses = defaultdict(list)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger."""
        logger = logging.getLogger("ContrastiveTrainer")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Create file handler
        log_file = os.path.join(self.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_optimizer_with_groups(
        self, 
        model: RobustEncoder, 
        config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """
        创建具有参数分组的优化器，分别为LoRA参数和投影头参数设置不同的学习率。
        
        Args:
            model: 模型
            config: 优化器配置，包含lora_lr和projection_head_lr
            
        Returns:
            具有参数分组的优化器
        """
        # 默认参数
        lora_lr = config.get("lora_lr", 5e-5)
        projection_head_lr = config.get("projection_head_lr", 1e-4)
        weight_decay = config.get("weight_decay", 0.01)
        
        # 分别收集LoRA参数和投影头参数
        lora_params = []
        projection_head_params = []
        
        # 收集LoRA参数 (在encoder中)
        if hasattr(model.encoder, "get_peft_config_as_dict"):
            # 使用PEFT的参数收集API
            for name, param in model.encoder.named_parameters():
                if param.requires_grad:  # 只收集可训练参数
                    lora_params.append(param)
                    self.logger.info(f"LoRA parameter: {name}")
        else:
            # 使用模式匹配查找LoRA参数
            for name, param in model.encoder.named_parameters():
                if "lora" in name.lower() and param.requires_grad:
                    lora_params.append(param)
                    self.logger.info(f"LoRA parameter: {name}")
        
        # 收集投影头参数
        if model.projection_head is not None:
            for name, param in model.projection_head.named_parameters():
                if param.requires_grad:
                    projection_head_params.append(param)
                    self.logger.info(f"Projection head parameter: {name}")
        
        # 创建参数组
        param_groups = [
            {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
            {"params": projection_head_params, "lr": projection_head_lr, "weight_decay": weight_decay}
        ]
        
        # 打印参数组统计信息
        self.logger.info(f"LoRA parameters: {len(lora_params)}, lr={lora_lr}")
        self.logger.info(f"Projection head parameters: {len(projection_head_params)}, lr={projection_head_lr}")
        
        # 创建优化器
        return optim.AdamW(param_groups)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Number of training examples: {len(self.train_dataloader.dataset)}")
        
        if self.val_dataloader:
            self.logger.info(f"Number of validation examples: {len(self.val_dataloader.dataset)}")
        
        # Training loop
        start_time = time.time()
        metrics = {
            "train_loss": [],
            "val_loss": [],
            "epoch": []
        }
        
        # 新增: 打印优化器参数组信息
        if hasattr(self.optimizer, "param_groups"):
            for i, group in enumerate(self.optimizer.param_groups):
                self.logger.info(f"Parameter group {i}: {len(group['params'])} parameters, lr={group['lr']}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            epoch_metrics = self._train_epoch()
            metrics["train_loss"].extend(epoch_metrics["train_loss"])
            metrics["epoch"].extend([epoch] * len(epoch_metrics["train_loss"]))
            
            # 新增: 记录详细损失
            for key, values in epoch_metrics.items():
                if key != "train_loss":  # 已经添加过了
                    metrics[key] = metrics.get(key, []) + values
            
            # Evaluate if validation set is provided
            if self.val_dataloader:
                val_loss = self.evaluate()
                metrics["val_loss"].append(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(name="best_model")
                    self.logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint after each epoch
            self._save_checkpoint(name=f"epoch_{epoch+1}")
        
        # Training complete
        total_time = time.time() - start_time
        self.logger.info(f"Training complete in {total_time:.2f} seconds")
        
        # Save final model
        self._save_checkpoint(name="final_model")
        
        return metrics
    
    def _train_epoch(self) -> Dict[str, List[float]]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    loss, loss_dict = self._compute_loss(batch)
            else:
                loss, loss_dict = self._compute_loss(batch)
                
            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with optional AMP
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Record the full loss for metrics
            epoch_metrics["train_loss"].append((loss * self.gradient_accumulation_steps).item())
            
            # 记录详细的损失组件
            if loss_dict:
                for key, value in loss_dict.items():
                    epoch_metrics[f"train_{key}"].append(value)
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Clear gradients
                self.optimizer.zero_grad()
                
                # LR scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Update progress bar and log
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.logging_steps == 0:
                    avg_loss = np.mean(epoch_metrics["train_loss"][-self.logging_steps:])
                    self.train_losses.append(avg_loss)
                    
                    # 记录和打印详细损失
                    log_message = f"Step {self.global_step}: Train loss = {avg_loss:.4f}"
                    for key in [k for k in epoch_metrics.keys() if k != "train_loss"]:
                        if len(epoch_metrics[key]) >= self.logging_steps:
                            avg_component = np.mean(epoch_metrics[key][-self.logging_steps:])
                            log_message += f", {key.replace('train_', '')} = {avg_component:.4f}"
                            self.detailed_losses[key].append(avg_component)
                    
                    self.logger.info(log_message)
                
                # Evaluate
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    val_loss = self.evaluate()
                    # Return to training mode
                    self.model.train()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(name="best_model")
                        self.logger.info(f"New best validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(name=f"step_{self.global_step}")
                    
                # Update progress bar
                progress_bar.set_postfix({"loss": np.mean(epoch_metrics["train_loss"][-100:])})
        
        return dict(epoch_metrics)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss for a batch.
        支持双输出模型和组合损失，新增pairs模式（BCE）。
        
        Args:
            batch: Batch from dataloader, containing the following keys:
                - input_ids_anchor: Token IDs for anchor code
                - attention_mask_anchor: Attention mask for anchor code
                - input_ids_positive: Token IDs for positive code
                - attention_mask_positive: Attention mask for positive code
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Pairs 模式：检测成对输入与标签
        if (
            "input_ids_1" in batch and "attention_mask_1" in batch and
            "input_ids_2" in batch and "attention_mask_2" in batch and
            "label" in batch
        ):
            inputs1 = {"input_ids": batch["input_ids_1"], "attention_mask": batch["attention_mask_1"]}
            inputs2 = {"input_ids": batch["input_ids_2"], "attention_mask": batch["attention_mask_2"]}
            labels = batch["label"].float()
            # 编码器-only前向（模型内部已L2归一化）
            emb1 = self.model.forward_encoder_only(**inputs1)
            emb2 = self.model.forward_encoder_only(**inputs2)
            # 余弦相似度
            scores = torch.nn.functional.cosine_similarity(emb1, emb2)
            # 温度缩放（与loss配置一致）
            temperature_enc = getattr(self.loss_fn, 'temperature_encoder', None)
            if temperature_enc is None and hasattr(self.loss_fn, 'encoder_infonce_loss_obj'):
                temperature_enc = getattr(self.loss_fn.encoder_infonce_loss_obj, 'temperature', 0.07)
            if temperature_enc is None:
                temperature_enc = 0.07
            logits = scores / temperature_enc
            # BCEWithLogitsLoss
            pos_w = torch.tensor(getattr(self.loss_fn, 'pos_weight', 1.0), device=labels.device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            loss = bce(logits, labels)
            # 统计日志（与评估空间一致）
            with torch.no_grad():
                pos_mask = (labels == 1)
                neg_mask = (labels == 0)
                pos_mean = scores[pos_mask].mean().item() if pos_mask.any() else 0.0
                neg_mean = scores[neg_mask].mean().item() if neg_mask.any() else 0.0
                loss_dict = {
                    "encoder_pos_sim_mean": float(pos_mean),
                    "encoder_neg_sim_mean": float(neg_mean)
                }
            return loss, loss_dict
        
        # 否则走原有anchor-positive（可带显式负样本）的路径
        # 提取anchor和positive的输入
        anchor_inputs = {
            "input_ids": batch["input_ids_anchor"],
            "attention_mask": batch["attention_mask_anchor"]
        }
        
        positive_inputs = {
            "input_ids": batch["input_ids_positive"],
            "attention_mask": batch["attention_mask_positive"]
        }
        
        # 可选：显式负样本
        has_explicit_negs = ("input_ids_negatives" in batch and "attention_mask_negatives" in batch and "neg_ptr" in batch)
        if has_explicit_negs:
            negatives_inputs = {
                "input_ids": batch["input_ids_negatives"],
                "attention_mask": batch["attention_mask_negatives"]
            }
        
        # 获取嵌入向量 - 处理两种可能的输出格式
        anchor_outputs = self.model(**anchor_inputs)
        positive_outputs = self.model(**positive_inputs)
        if has_explicit_negs:
            negatives_outputs = self.model(**negatives_inputs)
        
        # 判断是否为双输出模型
        if isinstance(anchor_outputs, tuple) and len(anchor_outputs) == 2:
            # 双输出模型: (encoder_output, projection_output)
            anchor_encoder, anchor_proj = anchor_outputs
            positive_encoder, positive_proj = positive_outputs
            if has_explicit_negs:
                neg_encoder, neg_proj = negatives_outputs
            
            # 使用组合损失
            if isinstance(self.loss_fn, CombinedContrastiveLoss):
                loss = self.loss_fn(
                    encoder_outputs=(anchor_encoder, positive_encoder),
                    projection_outputs=(anchor_proj, positive_proj)
                )
                # 若存在显式负样本，在投影空间追加一个基于显式negatives的InfoNCE分母项
                if has_explicit_negs:
                    # 构建每个anchor的logits: 正样本相似度 + 该anchor对应的显式负样本相似度
                    B = anchor_proj.size(0)
                    neg_ptr = batch["neg_ptr"].to(anchor_proj.device)  # [B,2] (start, len)
                    # 归一化
                    anchor_proj_n = torch.nn.functional.normalize(anchor_proj, p=2, dim=1)
                    positive_proj_n = torch.nn.functional.normalize(positive_proj, p=2, dim=1)
                    neg_proj_n = torch.nn.functional.normalize(neg_proj, p=2, dim=1) if has_explicit_negs else None
                    temperature = getattr(self.loss_fn.projection_loss, 'temperature', 0.07)
                    extra_losses = []
                    for i in range(B):
                        pos_sim = (anchor_proj_n[i] * positive_proj_n[i]).sum() / temperature
                        start, length = neg_ptr[i].tolist()
                        if length > 0:
                            neg_sims = (neg_proj_n[start:start+length] @ anchor_proj_n[i].unsqueeze(1)).squeeze(1) / temperature
                            logits = torch.cat([pos_sim.view(1), neg_sims], dim=0).unsqueeze(0)  # [1, 1+K]
                            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                            extra_losses.append(torch.nn.functional.cross_entropy(logits, labels, reduction='mean'))
                        else:
                            # 无显式负样本则跳过
                            continue
                    if len(extra_losses) > 0:
                        loss = loss + torch.stack(extra_losses).mean()
                
                # 新增：编码器空间的显式负样本对比项（与评估空间一致）
                if has_explicit_negs:
                    B = anchor_encoder.size(0)
                    neg_ptr = batch["neg_ptr"].to(anchor_encoder.device)
                    anchor_enc_n = torch.nn.functional.normalize(anchor_encoder, p=2, dim=1)
                    positive_enc_n = torch.nn.functional.normalize(positive_encoder, p=2, dim=1)
                    neg_enc_n = torch.nn.functional.normalize(neg_encoder, p=2, dim=1)
                    temperature_enc = getattr(self.loss_fn.encoder_infonce_loss_obj, 'temperature', 0.07) \
                        if hasattr(self.loss_fn, 'encoder_infonce_loss_obj') else 0.07
                    extra_losses_enc = []
                    for i in range(B):
                        pos_sim_enc = (anchor_enc_n[i] * positive_enc_n[i]).sum() / temperature_enc
                        start, length = neg_ptr[i].tolist()
                        if length > 0:
                            neg_sims_enc = (neg_enc_n[start:start+length] @ anchor_enc_n[i].unsqueeze(1)).squeeze(1) / temperature_enc
                            logits_enc = torch.cat([pos_sim_enc.view(1), neg_sims_enc], dim=0).unsqueeze(0)
                            labels_enc = torch.zeros(1, dtype=torch.long, device=logits_enc.device)
                            extra_losses_enc.append(torch.nn.functional.cross_entropy(logits_enc, labels_enc, reduction='mean'))
                        else:
                            continue
                    if len(extra_losses_enc) > 0:
                        # 轻权重加到总损失，避免过度强调负样本（6:4 倾向）
                        loss = loss + torch.stack(extra_losses_enc).mean()
                
                # 获取详细损失组件
                loss_dict = self.loss_fn.get_last_losses()
                # 新增：记录编码器正/负相似度统计
                with torch.no_grad():
                    # anchor_encoder 与 positive_encoder 已为L2归一化表示
                    sim_matrix = torch.matmul(anchor_encoder, positive_encoder.t())  # [B, B]
                    pos_sims = sim_matrix.diag()
                    # 计算非对角元素均值作为负样本相似度均值
                    batch_size = sim_matrix.size(0)
                    if batch_size > 1:
                        neg_sum = sim_matrix.sum() - pos_sims.sum()
                        neg_mean = neg_sum / (batch_size * (batch_size - 1))
                    else:
                        neg_mean = torch.tensor(0.0, device=sim_matrix.device)
                    loss_dict["encoder_pos_sim_mean"] = float(pos_sims.mean().detach())
                    loss_dict["encoder_neg_sim_mean"] = float(neg_mean.detach())
            else:
                # 如果不是组合损失，只使用投影头输出
                loss = self.loss_fn(anchor_proj, positive_proj)
                loss_dict = {}
        else:
            # 单输出模型（向后兼容）
            loss = self.loss_fn(anchor_outputs, positive_outputs)
            loss_dict = {}
        
        return loss, loss_dict
    
    def evaluate(self) -> float:
        """
        Evaluate the model on validation data.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Compute loss
                loss, _ = self._compute_loss(batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def _save_checkpoint(self, name: str = "model"):
        """
        Save model checkpoint.
        增强版，保存完整模型状态包括投影头。
        
        Args:
            name: Name for the checkpoint
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存训练元数据
        metadata = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "lambda": self.loss_fn.current_lambda if hasattr(self.loss_fn, "current_lambda") else None,
            "model_type": type(self.model).__name__,
            "loss_type": type(self.loss_fn).__name__,
        }
        
        # 保存模型
        if hasattr(self.model.encoder, "save_pretrained") and hasattr(self.model.encoder, "peft_config"):
            # 保存PEFT adapter
            self.model.encoder.save_pretrained(checkpoint_dir)
            self.logger.info(f"Saved PEFT adapter to {checkpoint_dir}")
            
            # 如果有投影头，单独保存
            if self.model.projection_head is not None:
                torch.save(
                    self.model.projection_head.state_dict(),
                    os.path.join(checkpoint_dir, "projection_head.bin")
                )
                self.logger.info(f"Saved projection head to {checkpoint_dir}")
        else:
            # 保存完整模型
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
            self.logger.info(f"Saved model to {checkpoint_dir}")
        
        # 保存优化器、调度器和训练状态
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                **metadata
            },
            os.path.join(checkpoint_dir, "optimizer.pt")
        )
        
        # 保存元数据
        import json
        with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Saved checkpoint metadata to {checkpoint_dir}")
    
    def load_checkpoint(
        self, 
        checkpoint_dir: str, 
        eval_mode: bool = False,
        encoder_only: bool = False
    ):
        """
        Load model from checkpoint.
        增强版，支持不同的加载模式。
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            eval_mode: Whether to load in evaluation mode
            encoder_only: Whether to load only the encoder (skip projection head)
        """
        # 加载元数据（如果存在）
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.logger.info(f"Loaded checkpoint metadata: {metadata}")
        
        # 根据模式加载不同组件
        if eval_mode:
            self.logger.info("Loading in evaluation mode")
            
        if encoder_only:
            self.logger.info("Loading encoder only (skipping projection head)")
        
        # If using a PEFT model, load adapter
        if hasattr(self.model.encoder, "load_adapter") and os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
            # Load PEFT adapter
            self.model.encoder.load_adapter(checkpoint_dir)
            self.logger.info(f"Loaded PEFT adapter from {checkpoint_dir}")
            
            # 如果不是仅编码器模式，且投影头存在，则加载投影头
            if not encoder_only and self.model.projection_head is not None:
                proj_path = os.path.join(checkpoint_dir, "projection_head.bin")
                if os.path.exists(proj_path):
                    self.model.projection_head.load_state_dict(
                        torch.load(proj_path, map_location=self.device)
                    )
                    self.logger.info(f"Loaded projection head from {proj_path}")
        else:
            # Load full model
            model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                
                if encoder_only:
                    # 过滤掉投影头的参数
                    filtered_state_dict = {k: v for k, v in state_dict.items() 
                                          if not k.startswith("projection_head.")}
                    # 检查是否需要修改模型结构
                    if hasattr(self.model, "projection_head") and self.model.projection_head is not None:
                        self.logger.info("Setting projection_head to None for encoder-only mode")
                        self.model.projection_head = None
                    
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        filtered_state_dict, strict=False
                    )
                    self.logger.info(f"Loaded encoder only. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
                else:
                    # 加载全部参数
                    self.model.load_state_dict(state_dict)
                    self.logger.info(f"Loaded full model from {model_path}")
        
        # 如果不是评估模式，加载优化器和调度器
        if not eval_mode:
            # Load optimizer and scheduler
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                checkpoint = torch.load(optimizer_path, map_location=self.device)
                
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                if self.lr_scheduler and checkpoint.get("lr_scheduler"):
                    self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                
                self.epoch = checkpoint.get("epoch", 0)
                self.global_step = checkpoint.get("global_step", 0)
                self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                
                # 如果使用组合损失，恢复lambda
                if hasattr(self.loss_fn, "current_lambda") and "lambda" in checkpoint:
                    self.loss_fn.current_lambda = checkpoint["lambda"]
                
                self.logger.info(f"Loaded optimizer state from {optimizer_path}")
                self.logger.info(f"Resuming from epoch {self.epoch + 1}, step {self.global_step}")
        
        # 快速验证加载的模型
        if not eval_mode:
            self._verify_model_loading()
        
        return self
    
    def _verify_model_loading(self):
        """验证模型加载是否成功，检查参数是否可训练且有梯度流"""
        with torch.enable_grad():
            # 创建一个小批量假数据
            batch_size = 2
            seq_len = 32
            
            # 随机生成一个小批量数据
            sample_batch = {
                "input_ids_anchor": torch.randint(0, 1000, (batch_size, seq_len), device=self.device),
                "attention_mask_anchor": torch.ones((batch_size, seq_len), device=self.device),
                "input_ids_positive": torch.randint(0, 1000, (batch_size, seq_len), device=self.device),
                "attention_mask_positive": torch.ones((batch_size, seq_len), device=self.device)
            }
            
            # 清除现有梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            loss, _ = self._compute_loss(sample_batch)
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            lora_has_grad = False
            proj_has_grad = False
            
            # 检查LoRA参数梯度
            for name, param in self.model.encoder.named_parameters():
                if "lora" in name.lower() and param.requires_grad:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        lora_has_grad = True
                        break
            
            # 检查投影头梯度（如果存在）
            if self.model.projection_head is not None:
                for name, param in self.model.projection_head.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None and param.grad.abs().sum() > 0:
                            proj_has_grad = True
                            break
            
            # 清除测试梯度
            self.optimizer.zero_grad()
            
            # 记录结果
            self.logger.info(f"Verification results: LoRA parameters have gradient: {lora_has_grad}")
            if self.model.projection_head is not None:
                self.logger.info(f"Verification results: Projection head parameters have gradient: {proj_has_grad}")
            else:
                self.logger.info("Verification results: No projection head found (encoder-only mode)")
            
            if not lora_has_grad and (not proj_has_grad and self.model.projection_head is not None):
                self.logger.warning("WARNING: No gradients detected in model parameters!")
            
            return lora_has_grad, proj_has_grad 