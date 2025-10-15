"""
Dataset classes for contrastive learning.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Callable
from transformers import PreTrainedTokenizer

# 新增：用于训练/验证样本的静态预筛（Java函数、长度>min_length、基本结构检查）
def _is_candidate_function(code_text: str, min_length: int = 50) -> bool:
    """
    检查代码是否是有效的Java函数/方法
    
    参数:
        code_text: 代码文本
        min_length: 最小代码长度
        
    返回:
        布尔值，表示代码是否通过筛选
    """
    import re
    
    if not code_text:
        return False
    
    stripped = code_text.strip()
    
    # 长度检查
    if len(stripped) <= min_length:
        return False
    
    # 基本结构检查：必须包含括号和花括号
    if '(' not in stripped or ')' not in stripped or '{' not in stripped or '}' not in stripped:
        return False
    
    # 函数定义模式检查（Java方法）
    java_function_pattern = r'(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *\{'
    if not re.search(java_function_pattern, stripped, re.MULTILINE):
        return False
    
    # 检查括号是否平衡
    if stripped.count('{') != stripped.count('}'):
        return False
    if stripped.count('(') != stripped.count(')'):
        return False
    
    # 检查是否包含至少一个Java关键字
    java_keywords = ['return', 'if', 'for', 'while', 'switch', 'try', 'catch']
    if not any(keyword in stripped for keyword in java_keywords):
        return False
    
    return True


class ContrastiveTrainDataset(Dataset):
	"""
	Dataset for contrastive learning of code representations.
	Each item returns a pair of (original code, positive code).
	"""
	
	def __init__(
		self,
		data_path: str,
		tokenizer: PreTrainedTokenizer,
		augmentor=None,
		max_length: int = 512,
		cache_dir: Optional[str] = None,
		static_filter: bool = False,
		min_length: int = 50
	):
		"""
		Initialize dataset.
		
		Args:
			data_path: Path to JSON/JSONL file or directory containing code samples
			tokenizer: Tokenizer to use for encoding code
			augmentor: CodeAugmentor instance to use for generating positive samples
			max_length: Maximum sequence length
			cache_dir: Directory to cache processed data
			static_filter: Whether to filter samples to only simple, runnable functions
			min_length: Minimum code length for filtering
		"""
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.augmentor = augmentor
		self.static_filter = static_filter
		self.min_length = min_length
		
		# Load code samples
		self.samples = []
		
		if os.path.isfile(data_path):
			# Load from single file
			self._load_file(data_path)
		elif os.path.isdir(data_path):
			# Load from directory
			for filename in os.listdir(data_path):
				if filename.endswith('.json') or filename.endswith('.jsonl'):
					self._load_file(os.path.join(data_path, filename))
		else:
			raise ValueError(f"Invalid data_path: {data_path}")
		
		print(f"Loaded {len(self.samples)} code samples for contrastive learning")
	
	def _load_file(self, file_path: str):
		"""Load code samples from a file."""
		with open(file_path, 'r', encoding='utf-8') as f:
			if file_path.endswith('.jsonl'):
				# JSONL file - one JSON object per line
				for line in f:
					if line.strip():
						data = json.loads(line)
						self._process_sample(data)
			else:
				# JSON file - single object or array
				data = json.load(f)
				if isinstance(data, list):
					for item in data:
						self._process_sample(item)
				else:
					self._process_sample(data)
		
	def _process_sample(self, data: Dict):
		"""Process a single code sample from the data."""
		# Extract code string from common field names
		code = None
		for field in ['code', 'content', 'function', 'source', 'raw']:
			if field in data and isinstance(data[field], str):
				code = data[field]
				break
		
		# If no code found, try to find any string field
		if code is None:
			for key, value in data.items():
				if isinstance(value, str) and len(value) > 10:  # Assume longer strings are code
					code = value
					break
		
		if code is None:
			return  # Skip this sample
		
		# 应用静态预筛选（可选）
		if self.static_filter and not _is_candidate_function(code, self.min_length):
			return
		
		self.samples.append({
			'code': code,
			'lang': data.get('language', data.get('lang', 'unknown')),
			'metadata': {k: v for k, v in data.items() if k not in ['code', 'content', 'function', 'source', 'raw']}
		})
	
	def __len__(self) -> int:
		return len(self.samples)
	
	def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
		"""
		Get a pair of (original, positive) samples without padding.
		Padding will be applied in collate_fn during batch creation.
		
		Returns:
			Dictionary with keys:
			- code: Original code string
			- positive_code: Augmented positive code string
		"""
		code = self.samples[idx]['code']
		
		# Generate positive sample using augmentor
		if self.augmentor:
			augmented_codes = self.augmentor.augment(code)
			positive_code = augmented_codes[0] if augmented_codes else code
		else:
			# If no augmentor, use original code as positive (not ideal for training)
			positive_code = code
		
		# Return original and positive code strings (tokenization will be done in collate_fn)
		return {
			'code': code,
			'positive_code': positive_code
		}


class CloneDetectionTestDataset(Dataset):
	"""
	Dataset for evaluating code clone detection.
	Each item returns a pair of code samples and a label indicating whether they are clones.
	"""
	
	def __init__(
		self,
		data_path: str,
		tokenizer: PreTrainedTokenizer,
		max_length: int = 512
	):
		"""
		Initialize dataset.
		
		Args:
			data_path: Path to JSON/JSONL file with code pairs
			tokenizer: Tokenizer to use for encoding code
			max_length: Maximum sequence length
		"""
		self.tokenizer = tokenizer
		self.max_length = max_length
		
		# Load code pairs and labels
		self.pairs = []
		
		with open(data_path, 'r', encoding='utf-8') as f:
			if data_path.endswith('.jsonl'):
				for line in f:
					if line.strip():
						self._process_pair(json.loads(line))
			else:
				data = json.load(f)
				if isinstance(data, list):
					for item in data:
						self._process_pair(item)
				else:
					raise ValueError("Expected JSON array for clone detection data")
					
		print(f"Loaded {len(self.pairs)} code pairs for clone detection evaluation")
	
	def _process_pair(self, data: Dict):
		"""Process a single code pair from the data."""
		# Extract code pair and label
		code1 = None
		code2 = None
		label = None
		
		# Try to extract from common field structures
		if 'code1' in data and 'code2' in data:
			code1 = data['code1']
			code2 = data['code2']
			label = data.get('label', data.get('clone', data.get('is_clone', None)))
		elif 'pair' in data and isinstance(data['pair'], list) and len(data['pair']) == 2:
			code1, code2 = data['pair']
			label = data.get('label', data.get('clone', data.get('is_clone', None)))
		
		# If label is not found, try common fields
		if label is None:
			for field in ['label', 'clone', 'is_clone', 'is_similar', 'similar']:
				if field in data:
					label = data[field]
					break
		
		# Skip if missing data
		if code1 is None or code2 is None or label is None:
			return
			
		# Convert label to int
		if isinstance(label, bool):
			label = int(label)
		elif isinstance(label, str):
			label = 1 if label.lower() in ['true', 'yes', '1', 'similar', 'clone'] else 0
			
		self.pairs.append({
			'code1': code1,
			'code2': code2,
			'label': label
		})
	
	def __len__(self) -> int:
		return len(self.pairs)
	
	def __getitem__(self, idx: int) -> Dict[str, Union[str, int]]:
		"""
		Get a code pair and its label without padding.
		Padding will be applied in collate_fn during batch creation.
		
		Returns:
			Dictionary with keys:
			- code1: First code string
			- code2: Second code string
			- label: Binary label (1 for clone, 0 for non-clone)
		"""
		pair = self.pairs[idx]
		
		# Return code strings and label (tokenization will be done in collate_fn)
		return {
			'code1': pair['code1'],
			'code2': pair['code2'],
			'label': torch.tensor(pair['label'], dtype=torch.long)
		}


class TrainCollator:
	"""
	可序列化的训练数据整理器，用于多进程数据加载。
	"""
	def __init__(self, tokenizer, max_length=512):
		self.tokenizer = tokenizer
		self.max_length = max_length
		
	def __call__(self, batch):
		# 提取原始代码和正样本代码
		codes = [item['code'] for item in batch]
		positive_codes = [item['positive_code'] for item in batch]
		
		# 批量编码，使用动态填充（以批次中最长的样本为准）
		anchor_encodings = self.tokenizer(
			codes,
			max_length=self.max_length,
			padding=True,  # 动态填充
			truncation=True,
			return_tensors='pt'
		)
		
		positive_encodings = self.tokenizer(
			positive_codes,
			max_length=self.max_length,
			padding=True,  # 动态填充
			truncation=True,
			return_tensors='pt'
		)
		
		# 返回编码后的批次
		return {
			'input_ids_anchor': anchor_encodings['input_ids'],
			'attention_mask_anchor': anchor_encodings['attention_mask'],
			'input_ids_positive': positive_encodings['input_ids'],
			'attention_mask_positive': positive_encodings['attention_mask']
		}


class TestCollator:
	"""
	可序列化的测试数据整理器，用于多进程数据加载。
	"""
	def __init__(self, tokenizer, max_length=512):
		self.tokenizer = tokenizer
		self.max_length = max_length
		
	def __call__(self, batch):
		# 提取代码对和标签
		codes1 = [item['code1'] for item in batch]
		codes2 = [item['code2'] for item in batch]
		labels = [item['label'] for item in batch]
		
		# 批量编码，使用动态填充
		encodings1 = self.tokenizer(
			codes1,
			max_length=self.max_length,
			padding=True,  # 动态填充
			truncation=True,
			return_tensors='pt'
		)
		
		encodings2 = self.tokenizer(
			codes2,
			max_length=self.max_length,
			padding=True,  # 动态填充
			truncation=True,
			return_tensors='pt'
		)
		
		# 返回编码后的批次
		return {
			'input_ids_1': encodings1['input_ids'],
			'attention_mask_1': encodings1['attention_mask'],
			'input_ids_2': encodings2['input_ids'],
			'attention_mask_2': encodings2['attention_mask'],
			'label': torch.stack(labels)
		}


def create_train_collate_fn(tokenizer: PreTrainedTokenizer, max_length: int = 512):
	"""
	创建用于训练数据的collate_fn，动态处理批次填充。
	
	Args:
		tokenizer: 分词器
		max_length: 最大序列长度（用于截断）
		
	Returns:
		collate_fn函数
	"""
	return TrainCollator(tokenizer, max_length)


def create_test_collate_fn(tokenizer: PreTrainedTokenizer, max_length: int = 512):
	"""
	创建用于测试数据的collate_fn，动态处理批次填充。
	
	Args:
		tokenizer: 分词器
		max_length: 最大序列长度（用于截断）
		
	Returns:
		collate_fn函数
	"""
	return TestCollator(tokenizer, max_length) 