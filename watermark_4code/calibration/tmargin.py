"""
基于测试集正样本对标定 t_margin：
- 读取 *_pairs_simple.jsonl 中 label=1 的样本对 (code1, code2)
- 计算 Δ[i] = dot(wi, E(code2)) - dot(wi, E(code1))
- 给出每个方向的高分位（默认 p99），并提供标量 t_margin = max_i 分位值
"""

import json
from typing import Dict, Tuple
import numpy as np

from ..api import load_encoder
from ..keys import derive_directions
from ..utils import project_embeddings


def calibrate_tmargin(
    model_dir: str,
    test_pairs_path: str,
    secret_key: str = "XDF",
    quantized: bool = True,
    max_length: int = 512,
    batch_size: int = 64,
    quantile: float = 0.99,
) -> Dict:
    """
    返回：
    {
      "quantile": 0.99,
      "per_bit": [q1, q2, q3, q4],
      "scalar": max(per_bit)
    }
    """
    # 读取正样本对
    code1_list = []
    code2_list = []
    with open(test_pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if int(obj.get('label', 0)) != 1:
                continue
            c1 = obj.get('code1', '')
            c2 = obj.get('code2', '')
            if c1 and c2:
                code1_list.append(c1)
                code2_list.append(c2)

    model, tokenizer, device = load_encoder(model_dir, use_quantization=quantized)

    # 批量嵌入
    from ..encoder import embed_codes
    v1 = embed_codes(model, tokenizer, code1_list, max_length=max_length, batch_size=batch_size, device=device)
    v2 = embed_codes(model, tokenizer, code2_list, max_length=max_length, batch_size=batch_size, device=device)

    # 派生方向
    d = v1.shape[1]
    W = derive_directions(secret_key=secret_key, d=d, k=4)

    s1 = project_embeddings(v1, W)  # [N,4]
    s2 = project_embeddings(v2, W)  # [N,4]
    delta = (s2 - s1)               # [N,4]

    # 主 quantile 结果
    per_bit = [float(np.quantile(delta[:, i], quantile)) for i in range(delta.shape[1])]
    scalar = float(max(per_bit))

    # 追加多分位输出：0.50..0.95 每隔0.05，另含0.99
    q_list = [round(q, 2) for q in np.arange(0.50, 1.00, 0.05).tolist()]
    if 0.99 not in q_list:
        q_list.append(0.99)
    q_list = sorted(set(q_list))

    quantiles_out = {}
    for q in q_list:
        per_bit_q = [float(np.quantile(delta[:, i], q)) for i in range(delta.shape[1])]
        scalar_q = float(max(per_bit_q))
        quantiles_out[str(q)] = {
            "per_bit": per_bit_q,
            "scalar": scalar_q,
        }

    return {
        "quantile": float(quantile),
        "per_bit": per_bit,
        "scalar": scalar,
        "quantiles": quantiles_out,
    }


