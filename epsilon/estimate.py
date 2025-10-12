"""
估计 ε_emp：
- 从 train_filtered_code.jsonl 中随机采样 N 个 anchor
- 为每个 anchor 按 0.2/0.5/0.3 生成 M 个全新变体（静态/LLM重写/转译）
- 过滤无变化或失败样本；计算 d = ||E(anchor)-E(variant)|| 的分布
- 取 p=0.99 分位作为 ε_emp
"""

import os
import json
import random
from typing import Dict, List, Tuple
import concurrent.futures
import numpy as np

from ..api import load_encoder
from ..encoder import embed_codes
from ..keys import derive_directions
from ..utils import project_embeddings
from contrastive_learning.java_augmentor import generate_java_training_data_parallel  # type: ignore


def estimate_epsilon_emp(
    model_dir: str,
    anchors_path: str,
    N: int = 100,
    M: int = 20,
    secret_key: str = "XDF",
    quantized: bool = True,
    max_length: int = 512,
    batch_size: int = 64,
    quantile: float = 0.99,
    enable_semantic_preserving: bool = True,
    enable_llm_rewrite: bool = True,
    enable_retranslate: bool = True,
    seed: int = 42,
) -> Dict:
    """
    返回：
    {
      "quantile": 0.99,
      "epsilon_emp": float,
      "num_pairs": int,
      "weights": {..}
    }
    """
    rng = random.Random(seed)

    # 按“测试阶段”的生成流程：使用 parallel 生成 augmented（仅更换数据来源），split='test'
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'epsilon'), exist_ok=True)
    out_augmented = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'epsilon', 'test_like_augmented.jsonl'))

    augmentation_types = {
        "semantic_preserving": 0.2,
        "llm_rewrite": 0.5,
        "retranslate": 0.3,
    }

    # 生成正/负混合的 augmented；最大样本数用 N 作为上限控制规模（与测试流程对齐的并行实现）
    # 动态缩小批大小以在较小 N 时也产生多批并发
    batch_size_for_parallel = max(1, min(5, N // 5))  # 例如 N=100 -> 20; N=50 -> 10

    generate_java_training_data_parallel(
        input_file=anchors_path,
        output_file=out_augmented,
        model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
        split_type="test",
        positive_ratio=0.5,
        augmentation_types=augmentation_types,
        max_samples=N,
        num_workers=48,
        batch_size=batch_size_for_parallel,
        resume=False,
    )

    # 读取 augmented 中的正样本作为等价变体（与测试正样本一致的读取方式）
    pairs: List[Tuple[str, str]] = []
    with open(out_augmented, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if 'anchor' in obj and 'positive' in obj and isinstance(obj['anchor'], str) and isinstance(obj['positive'], str):
                a = obj['anchor'].strip(); p = obj['positive'].strip()
                if a and p:
                    pairs.append((a, p))

    if not pairs:
        return {"quantile": float(quantile), "per_bit": [0.0, 0.0, 0.0, 0.0], "num_pairs": 0, "weights": augmentation_types}

    model, tokenizer, device = load_encoder(model_dir, use_quantization=quantized)

    # 批量嵌入并计算“沿私钥方向”的逐方向增量分布
    anchors = [a for a, _ in pairs]
    variants = [p for _, p in pairs]
    v1 = embed_codes(model, tokenizer, anchors, max_length=max_length, batch_size=batch_size, device=device)
    v2 = embed_codes(model, tokenizer, variants, max_length=max_length, batch_size=batch_size, device=device)

    d = v1.shape[1]
    W = derive_directions(secret_key=secret_key, d=d, k=4)
    s1 = project_embeddings(v1, W)  # [N,4]
    s2 = project_embeddings(v2, W)  # [N,4]
    delta_dir = np.abs(s2 - s1)     # [N,4]

    if delta_dir.shape[0] == 0:
        per_bit_main = [0.0, 0.0, 0.0, 0.0]
        quantiles_out = {}
    else:
        q_list = [round(q, 2) for q in np.arange(0.50, 1.00, 0.05).tolist()]
        if 0.99 not in q_list:
            q_list.append(0.99)
        q_list = sorted(set(q_list))
        quantiles_out = {}
        for q in q_list:
            per_bit_q = [float(np.quantile(delta_dir[:, i], q)) for i in range(delta_dir.shape[1])]
            quantiles_out[str(q)] = {"per_bit": per_bit_q}
        key = str(round(quantile, 2))
        per_bit_main = quantiles_out[key]["per_bit"] if key in quantiles_out else [0.0, 0.0, 0.0, 0.0]

    return {
        "quantile": float(quantile),
        "per_bit": per_bit_main,
        "num_pairs": int(delta_dir.shape[0]),
        "weights": augmentation_types,
        "quantiles": quantiles_out,
    }


