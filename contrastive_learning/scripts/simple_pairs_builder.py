#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 *_augmented.jsonl 生成仅包含“简单负样本”的 pairs 文件：*_pairs_simple.jsonl。
- 正样本：保留 (anchor, positive) -> label=1
- 负样本：为每个 anchor 采样 K 个“简单负样本” code_random（与 anchor 明显不相似）-> label=0
- 忽略原始文件中的困难负样本（type=hard_negative）与原生 negative 行

用法示例：
python contrastive_learning/scripts/simple_pairs_builder.py \
  --input contrastive_learning/datasets/csn_java/train_augmented.jsonl \
  --output contrastive_learning/datasets/csn_java/train_augmented_pairs_simple.jsonl \
  --neg_per_anchor 1 --jaccard_threshold 0.2 --seed 42
"""

import os
import re
import json
import argparse
import random
from typing import List, Dict, Set, Tuple


def tokenize(code: str) -> List[str]:
    # 简单 token 切分：字母数字下划线
    return re.findall(r"\w+", code)


def jaccard(a_tokens: Set[str], b_tokens: Set[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    if union == 0:
        return 1.0
    return inter / union


def build_code_pool(input_path: str) -> Tuple[List[Dict], List[str]]:
    """读取 augmented.jsonl，返回原始行对象列表与候选代码池。"""
    rows = []
    pool_set: Set[str] = set()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
            # 将可用字段加入候选池
            for key in ("anchor", "positive", "negative"):
                code = obj.get(key)
                if isinstance(code, str) and code.strip():
                    pool_set.add(code)
    return rows, list(pool_set)


def pick_easy_negative(anchor: str, candidates: List[str], 
                        anchor_tokens: Set[str], jaccard_threshold: float,
                        max_trials: int = 50) -> str:
    """从候选池随机挑选一个与 anchor 不相似的代码片段。"""
    n = len(candidates)
    if n == 0:
        return None
    for _ in range(max_trials):
        code = candidates[random.randrange(n)]
        if code == anchor:
            continue
        # 粗略长度约束：避免几乎相等长度的重复文本，但不过分严格
        if 0.8 * len(anchor) <= len(code) <= 1.25 * len(anchor):
            # 仍允许，但继续用Jaccard过滤
            pass
        code_tokens = set(tokenize(code))
        jac = jaccard(anchor_tokens, code_tokens)
        if jac < jaccard_threshold:
            return code
    # 兜底：放宽策略，直接返回一个不同的样本
    for code in candidates:
        if code != anchor:
            return code
    return None


def convert_to_pairs_simple(input_path: str, output_path: str, 
                            neg_per_anchor: int = 1, 
                            jaccard_threshold: float = 0.2,
                            seed: int = 42) -> Dict[str, int]:
    random.seed(seed)
    rows, pool = build_code_pool(input_path)

    pos_count = 0
    neg_count = 0
    skip_count = 0

    # 为了可复现，打乱候选池
    random.shuffle(pool)

    with open(output_path, 'w', encoding='utf-8') as out:
        for obj in rows:
            anchor = obj.get("anchor")
            if not isinstance(anchor, str) or not anchor.strip():
                skip_count += 1
                continue

            # 1) 写出正样本（若存在）
            positive = obj.get("positive")
            if isinstance(positive, str) and positive.strip():
                out.write(json.dumps({
                    "code1": anchor,
                    "code2": positive,
                    "label": 1
                }, ensure_ascii=False) + "\n")
                pos_count += 1

            # 2) 采样简单负样本（忽略原始 negative/hard_negative）
            anchor_tokens = set(tokenize(anchor))
            for _ in range(max(0, neg_per_anchor)):
                neg_code = pick_easy_negative(anchor, pool, anchor_tokens, jaccard_threshold)
                if neg_code is None:
                    continue
                out.write(json.dumps({
                    "code1": anchor,
                    "code2": neg_code,
                    "label": 0
                }, ensure_ascii=False) + "\n")
                neg_count += 1

    return {
        "positives": pos_count,
        "negatives": neg_count,
        "skipped": skip_count,
        "total_out": pos_count + neg_count
    }


def main():
    parser = argparse.ArgumentParser(description="将 augmented.jsonl 转为仅含简单负样本的 pairs_simple.jsonl")
    parser.add_argument('--input', type=str, required=True, help='输入 *_augmented.jsonl')
    parser.add_argument('--output', type=str, required=True, help='输出 *_pairs_simple.jsonl')
    parser.add_argument('--neg_per_anchor', type=int, default=1, help='每个 anchor 采样的简单负样本数量')
    parser.add_argument('--jaccard_threshold', type=float, default=0.2, help='token Jaccard 上限（越小越不相似）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    stats = convert_to_pairs_simple(
        input_path=args.input,
        output_path=args.output,
        neg_per_anchor=args.neg_per_anchor,
        jaccard_threshold=args.jaccard_threshold,
        seed=args.seed
    )
    print("转换完成:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")


if __name__ == '__main__':
    main() 