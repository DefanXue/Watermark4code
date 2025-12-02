"""
Model implementation for robust code representation encoder.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, T5EncoderModel, BitsAndBytesConfig
from peft import get_peft_model, PeftConfig


class MLP(nn.Module):
    """
    Multi-layer perceptron for projection head
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x


class RobustEncoder(nn.Module):
    """
    Robust code encoder using pre-trained model with PEFT adapters
    and contrastive learning projection head.
    NOW WITH QLoRA SUPPORT.
    
    Modified to support dual outputs (encoder-only and projection outputs)
    for combined loss training and encoder-only evaluation.
    """
    def __init__(
        self, 
        model_name="Salesforce/codet5-base",
        peft_config=None, 
        projection_dim=128,
        projection_hidden_dim=512,
        pooling_strategy="mean",
        use_quantization=False,  # QLoRA开关参数
        eval_mode=False  # 新增：评估模式标志，如果为True则不初始化投影头
    ):
        """
        Initialize the robust encoder.
        
        Args:
            model_name: Hugging Face model identifier
            peft_config: PEFT configuration object
            projection_dim: Dimension of projection head output
            projection_hidden_dim: Hidden dimension for projection head
            pooling_strategy: Strategy for pooling token representations ("mean", "cls", or "max")
            use_quantization: Whether to use 4-bit quantization (QLoRA)
            eval_mode: Whether to initialize in evaluation mode (skip projection head)
        """
        super().__init__()
        
        self.eval_mode = eval_mode
        
        # 根据开关决定是否创建量化配置
        quantization_config = None
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # 使用NF4数据类型，这是QLoRA论文推荐的
                bnb_4bit_compute_dtype=torch.bfloat16,  # 在计算时使用bfloat16
                bnb_4bit_use_double_quant=True,  # 使用双重量化，进一步节省显存
            )
            print("QLoRA (4-bit quantization) is enabled.")
        
        # Load base encoder from Hugging Face
        self.config = AutoConfig.from_pretrained(model_name)
        
        # 根据模型类型和量化配置选择合适的模型加载方式
        if "t5" in model_name.lower():
            # 对于T5模型，直接使用T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if use_quantization else None
            )
        else:
            # 对于其他模型使用AutoModel
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if use_quantization else None
            )
        
        # Apply PEFT if config is provided
        if peft_config is not None:
            # 如果使用了量化，需要为量化后的模型准备PEFT
            if use_quantization:
                from peft import prepare_model_for_kbit_training
                self.encoder = prepare_model_for_kbit_training(self.encoder)
                
            self.encoder = get_peft_model(self.encoder, peft_config)
            
            # 打印可训练参数统计信息
            if hasattr(self.encoder, "print_trainable_parameters"):
                self.encoder.print_trainable_parameters()
            
        # Set up projection head (unless in eval_mode)
        hidden_size = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.d_model
        
        # 在评估模式下不初始化投影头，节省内存
        if not eval_mode:
            self.projection_head = MLP(
                input_dim=hidden_size,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_dim
            )
        else:
            self.projection_head = None
            
        # Set pooling strategy
        self.pooling_strategy = pooling_strategy
        
    def pool_output(self, last_hidden_state, attention_mask=None):
        """
        Apply pooling strategy to encoder output
        """
        if self.pooling_strategy == "cls" and not "t5" in self.encoder.__class__.__name__.lower():
            # Use [CLS] token representation - note: T5 doesn't have a CLS token
            return last_hidden_state[:, 0]
        
        elif self.pooling_strategy == "mean":
            # Apply mean pooling over token dimension
            if attention_mask is not None:
                # Create proper mask for mean calculation
                expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                # Apply mask and calculate mean
                sum_hidden = torch.sum(last_hidden_state * expanded_mask, dim=1)
                token_count = torch.sum(attention_mask, dim=1, keepdim=True)
                return sum_hidden / token_count.clamp(min=1e-9)
            else:
                return torch.mean(last_hidden_state, dim=1)
        
        elif self.pooling_strategy == "max":
            # Apply max pooling over token dimension
            if attention_mask is not None:
                # Create proper mask for max calculation (set padded positions to very negative value)
                mask = (1 - attention_mask).unsqueeze(-1) * -1e9
                masked_hidden = last_hidden_state + mask.expand(last_hidden_state.size())
                return torch.max(masked_hidden, dim=1)[0]
            else:
                return torch.max(last_hidden_state, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass through encoder and projection head.
        Now returns both encoder output and projection output.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            **kwargs: Additional arguments to pass to encoder
            
        Returns:
            Tuple of (encoder_output, projected_output), both normalized
        """
        # Forward pass through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Extract last hidden state - handle different model output structures
        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            # For T5 encoder models
            last_hidden_state = outputs[0]
        
        # Apply pooling to get sequence representation
        pooled_output = self.pool_output(last_hidden_state, attention_mask)
        
        # Normalize encoder output
        normalized_encoder_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        
        # In evaluation mode or if projection head is None, only return encoder output
        if self.eval_mode or self.projection_head is None:
            return normalized_encoder_output
        
        # Project through MLP
        projected_output = self.projection_head(pooled_output)
        
        # Normalize projection output
        normalized_projection_output = torch.nn.functional.normalize(projected_output, p=2, dim=1)
        
        # Return tuple of (encoder_output, projection_output)
        return (normalized_encoder_output, normalized_projection_output)
        
    def forward_encoder_only(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass through encoder only, skipping the projection head.
        Used for evaluation and inference.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            **kwargs: Additional arguments to pass to encoder
            
        Returns:
            L2-normalized encoder output
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Extract last hidden state - handle different model output structures
        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            # For T5 encoder models
            last_hidden_state = outputs[0]
            
        # Apply pooling and normalize
        pooled_output = self.pool_output(last_hidden_state, attention_mask)
        normalized_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        
        return normalized_output
        
    def get_encoder_output(self, input_ids, attention_mask=None, **kwargs):
        """
        Get the raw encoder output before projection head for analysis.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Extract last hidden state - handle different model output structures
        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            # For T5 encoder models
            last_hidden_state = outputs[0]
            
        pooled_output = self.pool_output(last_hidden_state, attention_mask)
        return pooled_output 