"""
Contrastive loss functions for training robust code representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Any


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    
    This implementation supports both symmetric and asymmetric versions:
    - Symmetric: treat both elements of each positive pair as anchors
    - Asymmetric: only treat the first element as the anchor
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = "mean"
    ):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter to scale logits
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss using in-batch negatives.
        
        Args:
            anchor_embeddings: Tensor of shape [batch_size, embedding_dim].
                               These are the representations of the original code samples.
            positive_embeddings: Tensor of shape [batch_size, embedding_dim].
                                 These are the representations of the augmented code samples.
                                 
        Returns:
            The final loss value (a scalar tensor).
        """
        # Ensure embeddings are L2 normalized
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        batch_size = anchor_embeddings.size(0)
        
        # Concatenate all embeddings to form a single matrix for efficient computation
        # The shape will be [2 * batch_size, embedding_dim]
        all_embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
        
        # Compute the similarity matrix (all-to-all cosine similarities)
        # The shape will be [2 * batch_size, 2 * batch_size]
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T) / self.temperature
        
        # Create the labels for cross-entropy loss.
        # For each sample i (from 0 to 2*N-1), its positive pair is at position (i + N) % (2*N).
        # For anchors (0 to N-1), their positives are at (N to 2N-1).
        # For positives (N to 2N-1), their anchors are at (0 to N-1).
        labels = torch.arange(batch_size, 2 * batch_size, device=anchor_embeddings.device)
        labels = torch.cat([labels, torch.arange(batch_size, device=anchor_embeddings.device)], dim=0)
        
        # Mask out self-similarity from the logits.
        # A sample should not be compared with itself.
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute the cross-entropy loss.
        # This single calculation handles both directions (anchor -> positive and positive -> anchor)
        # because we constructed the labels and similarity matrix symmetrically.
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        
        return loss


def nt_xent_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    temperature: float = 0.5,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    NT-Xent loss (Normalized Temperature-scaled Cross Entropy)
    Alternative implementation of InfoNCE loss as a function.
    
    Args:
        anchor_embeddings: Tensor of shape [batch_size, embedding_dim]
        positive_embeddings: Tensor of shape [batch_size, embedding_dim]
        temperature: Temperature parameter to scale logits
        reduction: Reduction method ('none', 'mean', 'sum')
        
    Returns:
        Loss value
    """
    batch_size = anchor_embeddings.size(0)
    
    # L2 normalize embeddings
    anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
    positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
    
    # Concatenate representations to create the complete similarity matrix
    representations = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
    
    # Compute similarity matrix
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                           representations.unsqueeze(0), 
                                           dim=2) / temperature
    
    # Create masks for positive and negative pairs
    # Positives: diagonal elements in the off-diagonal quadrants
    positive_mask = torch.zeros_like(similarity_matrix)
    positive_mask[:batch_size, batch_size:] = torch.eye(batch_size)
    positive_mask[batch_size:, :batch_size] = torch.eye(batch_size)
    
    # Exclude self-similarities on the diagonal
    negative_mask = torch.ones_like(similarity_matrix)
    negative_mask[:batch_size, :batch_size].fill_diagonal_(0)
    negative_mask[batch_size:, batch_size:].fill_diagonal_(0)
    negative_mask = negative_mask * (1 - positive_mask)
    
    # Extract positive and negative scores
    positive_scores = similarity_matrix[positive_mask.bool()].view(2*batch_size, 1)
    negative_scores = similarity_matrix[negative_mask.bool()].view(2*batch_size, -1)
    
    # Compute log-probability using logsumexp trick for numerical stability
    logits = torch.cat([positive_scores, negative_scores], dim=1)
    labels = torch.zeros(2*batch_size, device=logits.device, dtype=torch.long)
    
    # Compute cross entropy loss
    loss = F.cross_entropy(logits, labels, reduction=reduction)
    return loss


class TripletMarginLoss(nn.Module):
    """
    Triplet margin loss for contrastive learning.
    Encourages anchors to be closer to positives than to negatives by a margin.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        reduction: str = "mean",
        swap: bool = False
    ):
        """
        Initialize triplet margin loss.
        
        Args:
            margin: Margin between positive and negative distances
            p: Norm degree for distance calculation (default: 2 for Euclidean)
            reduction: Reduction method ('none', 'mean', 'sum')
            swap: Whether to use distance swap for semi-hard negatives
        """
        super().__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        self.swap = swap
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.
        
        Args:
            anchor_embeddings: Tensor of shape [batch_size, embedding_dim]
            positive_embeddings: Tensor of shape [batch_size, embedding_dim]
            negative_embeddings: Tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Loss value
        """
        return F.triplet_margin_loss(
            anchor_embeddings, 
            positive_embeddings, 
            negative_embeddings,
            margin=self.margin,
            p=self.p,
            reduction=self.reduction,
            swap=self.swap
        )
        
        
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.
    This extends contrastive learning to leverage label information.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07
    ):
        """
        Initialize supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter to scale logits
            contrast_mode: 'all' or 'one', whether to compare all positives or just one
            base_temperature: Base temperature for the loss formula
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Tensor of shape [batch_size, n_views, embedding_dim]
                or [batch_size*n_views, embedding_dim]
            labels: Ground truth labels for supervised loss
            mask: Optional mask for specifying positive pairs
            
        Returns:
            Loss value
        """
        device = features.device
        
        # Handle different input shapes
        if features.dim() < 3:
            # If input is [batch_size*n_views, embedding_dim], reshape to 
            # [batch_size, n_views, embedding_dim]
            if labels is not None:
                batch_size = labels.shape[0]
            else:
                batch_size = features.shape[0] // 2  # Assuming 2 views by default
                
            features = features.view(batch_size, -1, features.shape[1])
            
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # Reshape to [batch_size * n_views, embedding_dim]
        features = features.reshape(batch_size * n_views, -1)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Create similarity matrix
        similarity_matrix = torch.matmul(features, features.T)
        
        # Create labels mask if not provided
        if mask is None:
            # Extract labels from initial batch if provided
            if labels is not None:
                # Repeat labels for each view
                labels = labels.repeat(n_views)
                labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
                mask = labels_matrix
            else:
                # If no labels provided, use instance-level identity
                mask = torch.eye(batch_size, dtype=torch.float, device=device)
                mask = mask.repeat(n_views, n_views)
        
        # Mask out self-contrast cases
        logits_mask = torch.ones_like(mask, device=device)
        logits_mask.fill_diagonal_(0)
        
        # Mask for valid positives
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix / self.temperature)
        
        # Mask out self-contrast
        exp_logits = exp_logits * logits_mask
        
        # Compute log_prob with logsumexp trick for stability
        if self.contrast_mode == 'all':
            log_prob = similarity_matrix / self.temperature - \
                       torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        elif self.contrast_mode == 'one':
            log_prob = similarity_matrix / self.temperature - \
                       torch.log(exp_logits + 1e-12)
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Scale by temperature
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss 


class EncoderConsistencyLoss(nn.Module):
    """
    编码器一致性损失，用于计算anchor和positive在编码器输出层级的相似性。
    这个损失直接约束编码器输出，确保编码器能够生成好的表示，即使没有投影头。
    """
    
    def __init__(
        self,
        reduction: str = "mean"
    ):
        """
        初始化编码器一致性损失。
        
        Args:
            reduction: 损失归约方法 ('none', 'mean', 'sum')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        anchor_encoder_outputs: torch.Tensor,
        positive_encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算anchor和positive编码器输出之间的余弦距离损失。
        
        Args:
            anchor_encoder_outputs: 形状为[batch_size, embedding_dim]的张量。
                                   这些是原始代码样本的编码器输出。
            positive_encoder_outputs: 形状为[batch_size, embedding_dim]的张量。
                                     这些是增强代码样本的编码器输出。
                                     
        Returns:
            最终的损失值（标量张量）。
        """
        # 确保输入已经L2归一化
        anchor_encoder_outputs = F.normalize(anchor_encoder_outputs, p=2, dim=1)
        positive_encoder_outputs = F.normalize(positive_encoder_outputs, p=2, dim=1)
        
        # 计算余弦相似度 (1 - cosine_sim)
        cosine_similarity = torch.sum(anchor_encoder_outputs * positive_encoder_outputs, dim=1)
        cosine_distance = 1.0 - cosine_similarity
        
        # 应用归约
        if self.reduction == "mean":
            loss = torch.mean(cosine_distance)
        elif self.reduction == "sum":
            loss = torch.sum(cosine_distance)
        else:  # 'none'
            loss = cosine_distance
            
        return loss


class CombinedContrastiveLoss(nn.Module):
	"""
	组合对比损失，整合InfoNCE损失和编码器一致性损失。
	使用权重λ来平衡这两种损失，支持λ的warmup调度。
	
	总损失 = L_proj + λ * L_enc
	"""
	
	def __init__(
		self,
		temperature: float = 0.07,
		consistency_lambda: float = 0.1,
		lambda_warmup_steps: int = 0,
		normalize_losses: bool = False,
		reduction: str = "mean",
		# 新增：encoder端InfoNCE相关
		use_encoder_infonce: bool = False,
		encoder_infonce_lambda: float = 0.0,
		encoder_infonce_warmup_steps: int = 0,
		temperature_encoder: float = 0.07,
		# 新增：归一化策略（EMA尺度归一）
		normalization_method: str = "ema_div",
		ema_beta: float = 0.99,
		eps: float = 1e-8,
	):
		"""
		初始化组合对比损失。
		
		Args:
			temperature: InfoNCE的温度参数（投影头）
			consistency_lambda: 编码器一致性损失的权重系数
			lambda_warmup_steps: （一致性λ）从0增加到目标值的步数
			normalize_losses: 是否对两种/三种损失进行归一化
			reduction: 损失归约方法 ('none', 'mean', 'sum')
			use_encoder_infonce: 是否启用编码器端的InfoNCE
			encoder_infonce_lambda: 编码器端InfoNCE的权重
			encoder_infonce_warmup_steps: （编码器InfoNCE λ）warmup步数
			temperature_encoder: 编码器端InfoNCE的温度参数
			normalization_method: 归一化方法（当前支持'ema_div'）
			ema_beta: EMA的动量系数
			eps: 除法防止为0的小常数
		"""
		super().__init__()
		self.projection_loss = InfoNCELoss(temperature=temperature, reduction=reduction)
		self.consistency_loss = EncoderConsistencyLoss(reduction=reduction)
		# 新增：encoder端InfoNCE
		self.use_encoder_infonce = use_encoder_infonce
		self.encoder_infonce_lambda_base = encoder_infonce_lambda
		self.encoder_infonce_warmup_steps = encoder_infonce_warmup_steps
		self.encoder_infonce_loss_obj = InfoNCELoss(temperature=temperature_encoder, reduction=reduction)
		
		self.base_lambda = consistency_lambda
		self.current_lambda = 0.0 if lambda_warmup_steps > 0 else consistency_lambda  # 向后兼容：保持字段名
		self.lambda_warmup_steps = lambda_warmup_steps
		self.normalize_losses = normalize_losses
		self.reduction = reduction
		self.step_count = 0
		
		# 新增：分别维护两条分支的当前λ（保持current_lambda作为一致性λ以兼容旧代码）
		self.current_consistency_lambda = self.current_lambda
		self.current_encoder_infonce_lambda = 0.0 if encoder_infonce_warmup_steps > 0 else self.encoder_infonce_lambda_base
		
		# 记录各损失值
		self.last_proj_loss = 0.0
		self.last_enc_loss = 0.0
		self.last_enc_infonce_loss = 0.0
		self.last_total_loss = 0.0
		
		# 新增：EMA统计用于尺度归一
		self.normalization_method = normalization_method
		self.ema_beta = ema_beta
		self.eps = eps
		self.ema_proj = None
		self.ema_enc = None
		self.ema_enc_infonce = None
	
	def _update_warmups(self):
		# 一致性λ warmup
		if self.step_count < self.lambda_warmup_steps:
			self.current_consistency_lambda = self.base_lambda * (self.step_count / max(1, self.lambda_warmup_steps))
		else:
			self.current_consistency_lambda = self.base_lambda
		# 编码器InfoNCE λ warmup
		if self.step_count < self.encoder_infonce_warmup_steps:
			self.current_encoder_infonce_lambda = self.encoder_infonce_lambda_base * (self.step_count / max(1, self.encoder_infonce_warmup_steps))
		else:
			self.current_encoder_infonce_lambda = self.encoder_infonce_lambda_base
		# 向后兼容：保持current_lambda字段与一致性λ一致
		self.current_lambda = self.current_consistency_lambda
	
	def _ema_update(self, ema_value: Optional[torch.Tensor], new_value: torch.Tensor) -> torch.Tensor:
		# 将标量张量进行EMA更新（在CPU上安全detach）
		val = new_value.detach()
		if ema_value is None:
			return val
		return self.ema_beta * ema_value + (1.0 - self.ema_beta) * val
	
	def _maybe_normalize(self, loss_value: torch.Tensor, ema_value: Optional[torch.Tensor]) -> torch.Tensor:
		if not self.normalize_losses:
			return loss_value
		if self.normalization_method == "ema_div":
			if ema_value is None:
				return loss_value
			# 使用上一时刻EMA作为分母进行尺度归一
			denom = torch.clamp(ema_value, min=self.eps)
			return loss_value / denom
		return loss_value
	
	def forward(
		self,
		encoder_outputs: Tuple[torch.Tensor, torch.Tensor],
		projection_outputs: Tuple[torch.Tensor, torch.Tensor]
	) -> torch.Tensor:
		"""
		计算组合损失。
		
		Args:
			encoder_outputs: (anchor_encoder_outputs, positive_encoder_outputs)元组
			projection_outputs: (anchor_projection_outputs, positive_projection_outputs)元组
							 
		Returns:
			最终的组合损失值（标量张量）。
		"""
		anchor_encoder, positive_encoder = encoder_outputs
		anchor_proj, positive_proj = projection_outputs
		
		# 1) 计算投影头InfoNCE
		proj_loss = self.projection_loss(anchor_proj, positive_proj)
		
		# 2) 计算编码器一致性
		enc_loss = self.consistency_loss(anchor_encoder, positive_encoder)
		
		# 3) 计算编码器端InfoNCE（可选）
		if self.use_encoder_infonce and self.encoder_infonce_lambda_base > 0.0:
			enc_infonce_loss = self.encoder_infonce_loss_obj(anchor_encoder, positive_encoder)
		else:
			enc_infonce_loss = torch.zeros_like(proj_loss)
		
		# 归一化（仅做尺度归一，不改变梯度方向）
		proj_loss_norm = self._maybe_normalize(proj_loss, self.ema_proj)
		enc_loss_norm = self._maybe_normalize(enc_loss, self.ema_enc)
		enc_infonce_loss_norm = self._maybe_normalize(enc_infonce_loss, self.ema_enc_infonce)
		
		# 更新warmup λ
		self._update_warmups()
		
		# 组合损失
		total_loss = (
			proj_loss_norm
			+ self.current_consistency_lambda * enc_loss_norm
			+ self.current_encoder_infonce_lambda * enc_infonce_loss_norm
		)
		
		# 保存当前原始损失（用于日志与EMA）
		self.last_proj_loss = proj_loss.detach()
		self.last_enc_loss = enc_loss.detach()
		self.last_enc_infonce_loss = enc_infonce_loss.detach()
		self.last_total_loss = total_loss.detach()
		
		# EMA更新（使用原始未归一化的分量以稳定尺度估计）
		self.ema_proj = self._ema_update(self.ema_proj, proj_loss)
		self.ema_enc = self._ema_update(self.ema_enc, enc_loss)
		self.ema_enc_infonce = self._ema_update(self.ema_enc_infonce, enc_infonce_loss)
		
		# 步数+1
		self.step_count += 1
		
		return total_loss
	
	def get_last_losses(self) -> Dict[str, float]:
		"""
		获取最近一次计算的损失值。
		
		Returns:
			包含各个损失组件的字典。
		"""
		return {
			"projection_loss": float(self.last_proj_loss),
			"encoder_infonce_loss": float(self.last_enc_infonce_loss),
			"encoder_consistency_loss": float(self.last_enc_loss),
			"lambda_encoder_infonce": float(self.current_encoder_infonce_lambda),
			"lambda_consistency": float(self.current_consistency_lambda),
			"ema_proj": float(self.ema_proj) if self.ema_proj is not None else 0.0,
			"ema_encoder_infonce": float(self.ema_enc_infonce) if self.ema_enc_infonce is not None else 0.0,
			"ema_encoder_consistency": float(self.ema_enc) if self.ema_enc is not None else 0.0,
			"total_loss": float(self.last_total_loss)
		} 