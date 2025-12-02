import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple
from transformers import PreTrainedTokenizer


class PairedTrainDataset(Dataset):
	"""
	成对训练数据集：严格消费离线配对监督，使用 anchor-positive 作为正对，
	同时为每个 anchor 收集可用的 negatives（hard_negative/random_negative）。
	输出与现有 Trainer 对齐：默认提供 {'code','positive_code'}；若存在负样本，
	额外提供 'negative_codes'（列表，长度可为0）。
	"""
	def __init__(
		self,
		data_path: str,
		tokenizer: PreTrainedTokenizer,
		max_length: int = 512,
		min_length: int = 1,
		num_negatives_per_anchor: int = 2
	):
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.min_length = min_length
		self.num_negatives_per_anchor = num_negatives_per_anchor
		self.samples: List[Dict[str, Union[str, List[str]]]] = []
		self._build_pairs_with_negatives(data_path)
		print(f"Loaded {len(self.samples)} anchor-positive pairs (with negatives if any) from {data_path}")

	def _build_pairs_with_negatives(self, file_path: str):
		if not os.path.exists(file_path):
			raise ValueError(f"Invalid data_path: {file_path}")
		# 先按anchor聚合positives与negatives
		anchor_to_pos: Dict[str, List[str]] = {}
		anchor_to_neg: Dict[str, List[str]] = {}
		with open(file_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					data = json.loads(line)
				except json.JSONDecodeError:
					continue
				anchor = data.get('anchor')
				if not isinstance(anchor, str) or len(anchor.strip()) < self.min_length:
					continue
				if 'positive' in data and isinstance(data['positive'], str) and len(data['positive'].strip()) >= self.min_length:
					anchor_to_pos.setdefault(anchor, []).append(data['positive'])
				elif 'negative' in data and isinstance(data['negative'], str) and len(data['negative'].strip()) >= self.min_length:
					anchor_to_neg.setdefault(anchor, []).append(data['negative'])
		# 产出样本：每个anchor使用一个positive，附带至多K个negatives
		for anchor, pos_list in anchor_to_pos.items():
			if not pos_list:
				continue
			positive = pos_list[0]
			negs = anchor_to_neg.get(anchor, [])
			if self.num_negatives_per_anchor > 0 and len(negs) > self.num_negatives_per_anchor:
				negs = negs[:self.num_negatives_per_anchor]
			self.samples.append({'code': anchor, 'positive_code': positive, 'negative_codes': negs})

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Union[str, List[str]]]:
		item = self.samples[idx]
		return item


class PairedTrainCollator:
	"""
	打包 anchor/positive 以及展平的 negatives；
	提供 neg_ptr: [B,2] 记录每个样本在展平负样本张量中的起始索引与长度。
	"""
	def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __call__(self, batch: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
		codes = [b['code'] for b in batch]
		pos_codes = [b['positive_code'] for b in batch]
		neg_lists: List[List[str]] = [b.get('negative_codes', []) for b in batch]

		enc_a = self.tokenizer(codes, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
		enc_p = self.tokenizer(pos_codes, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')

		flat_negs: List[str] = []
		neg_ptr = []  # (start, length)
		cursor = 0
		for negs in neg_lists:
			length = len(negs) if isinstance(negs, list) else 0
			if length > 0:
				flat_negs.extend(negs)
				neg_ptr.append((cursor, length))
				cursor += length
			else:
				neg_ptr.append((cursor, 0))

		batch_out = {
			'input_ids_anchor': enc_a['input_ids'],
			'attention_mask_anchor': enc_a['attention_mask'],
			'input_ids_positive': enc_p['input_ids'],
			'attention_mask_positive': enc_p['attention_mask'],
		}

		if len(flat_negs) > 0:
			enc_n = self.tokenizer(flat_negs, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
			batch_out['input_ids_negatives'] = enc_n['input_ids']
			batch_out['attention_mask_negatives'] = enc_n['attention_mask']
			batch_out['neg_ptr'] = torch.tensor(neg_ptr, dtype=torch.long)
		else:
			batch_out['neg_ptr'] = torch.tensor(neg_ptr, dtype=torch.long)

		return batch_out


def create_paired_collate_fn(tokenizer: PreTrainedTokenizer, max_length: int = 512):
	return PairedTrainCollator(tokenizer, max_length) 