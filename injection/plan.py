"""
Watermark injection planning helpers.

Only uses components inside Watermark4code and the upstream generator entrypoint
to strictly match the test-split generation process.
"""

import os
import json
import tempfile
from typing import Dict, List, Tuple
import numpy as np

from ..api import load_encoder, compute_baseline_s0
from ..encoder import embed_codes
from ..keys import derive_directions
from ..utils import project_embeddings
from contrastive_learning.java_augmentor import generate_java_training_data_parallel  # type: ignore


def _get_quantile_entry(obj: Dict, q: float):
    """
    Robustly fetch a quantile entry from an object that stores quantiles under string keys.
    Tries keys like "0.90", "0.9", str(q); if not found, tries tolerant numeric match.
    Returns the entry (could be float or dict) or None if not found.
    """
    if "quantiles" not in obj or not isinstance(obj["quantiles"], dict):
        return None
    qmap = obj["quantiles"]
    candidates = [f"{q:.2f}", f"{q:.1f}", str(q)]
    for k in candidates:
        if k in qmap:
            return qmap[k]
    # tolerant numeric match
    for k, v in qmap.items():
        try:
            if abs(float(k) - q) < 1e-8 or abs(round(float(k), 2) - round(q, 2)) < 1e-8:
                return v
        except Exception:
            continue
    return None


def compute_required_delta(epsilon_json_path: str, tmargin_json_path: str, quantile: float = 0.90) -> List[float]:
    with open(epsilon_json_path, "r", encoding="utf-8") as f:
        eps_obj = json.load(f)
    with open(tmargin_json_path, "r", encoding="utf-8") as f:
        tm_obj = json.load(f)

    # epsilon per-bit
    eps_entry = _get_quantile_entry(eps_obj, quantile)
    if isinstance(eps_entry, dict) and "per_bit" in eps_entry:
        eps_vec = [float(x) for x in eps_entry["per_bit"]]
    else:
        # 若不存在 per_bit，回退为 4 维同值（不保留历史，只为兼容极端情况）
        scalar = float(eps_obj.get("epsilon_emp", 0.0)) if not isinstance(eps_entry, (list, tuple)) else float(eps_entry)
        eps_vec = [scalar, scalar, scalar, scalar]

    # t_margin per-bit
    tm_entry = _get_quantile_entry(tm_obj, quantile)
    if isinstance(tm_entry, dict) and "per_bit" in tm_entry:
        tm_vec = [float(x) for x in tm_entry["per_bit"]]
    else:
        scalar = float(tm_obj.get("scalar", 0.0))
        tm_vec = [scalar, scalar, scalar, scalar]

    return [e + t for e, t in zip(eps_vec, tm_vec)]


def compute_baseline(model_dir: str, anchor_code: str, secret_key: str = "XDF") -> Dict:
    base = compute_baseline_s0(model_dir, [anchor_code], secret_key=secret_key)
    return {
        "s0": base["s0"][0],
        "W": base["directions"],
    }


def _write_anchor_copies_tmp(anchor_code: str, copies: int) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="wm_inject_")
    tmp_path = os.path.join(tmp_dir, "anchors.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for _ in range(max(1, copies)):
            f.write(json.dumps({"code": anchor_code}, ensure_ascii=False) + "\n")
    return tmp_path


def build_candidates_test_like(
    anchor_code: str,
    K: int,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
) -> List[str]:
    """
    Match test-split generation: split_type='test', positive_ratio=1.0, proportions 0.2/0.5/0.3.
    To get K variants, duplicate the same anchor K times as input.
    """
    aug_types = {
        "semantic_preserving": 0.2,
        "llm_rewrite": 0.5,
        "retranslate": 0.3,
    }

    in_file = _write_anchor_copies_tmp(anchor_code, K)
    out_dir = tempfile.mkdtemp(prefix="wm_aug_")
    out_file = os.path.join(out_dir, "augmented.jsonl")

    generate_java_training_data_parallel(
        input_file=in_file,
        output_file=out_file,
        model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
        split_type="test",
        positive_ratio=1.0,
        augmentation_types=aug_types,
        max_samples=K,
        num_workers=num_workers,
        batch_size=batch_size_for_parallel,
        resume=False,
    )

    cands: List[str] = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "anchor" in obj and "positive" in obj and isinstance(obj["positive"], str):
                cand = obj["positive"].strip()
                if cand and cand != anchor_code.strip():
                    cands.append(cand)

    # 去重并截取 K
    uniq: List[str] = []
    seen = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= K:
            break
    return uniq


def build_candidates_by_type(
    anchor_code: str,
    K: int,
    aug_type: str,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
) -> List[str]:
    """
    仅生成指定类别的等价候选，分布对齐 test-split：positive_ratio=1.0。
    aug_type ∈ {"semantic_preserving", "llm_rewrite", "retranslate"}
    """
    assert aug_type in {"semantic_preserving", "llm_rewrite", "retranslate"}

    aug_types = {aug_type: 1.0}

    in_file = _write_anchor_copies_tmp(anchor_code, K)
    out_dir = tempfile.mkdtemp(prefix="wm_aug_type_")
    out_file = os.path.join(out_dir, "augmented.jsonl")

    generate_java_training_data_parallel(
        input_file=in_file,
        output_file=out_file,
        model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
        split_type="test",
        positive_ratio=1.0,
        augmentation_types=aug_types,
        max_samples=K,
        num_workers=num_workers,
        batch_size=batch_size_for_parallel,
        resume=False,
    )

    cands: List[str] = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "anchor" in obj and "positive" in obj and isinstance(obj["positive"], str):
                cand = obj["positive"].strip()
                if cand and cand != anchor_code.strip():
                    cands.append(cand)

    # 去重并截取 K
    uniq: List[str] = []
    seen = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= K:
            break
    return uniq


def compute_required_delta_per_anchor(
    model_dir: str,
    anchor_code: str,
    secret_key: str = "XDF",
    K: int = 50,
    quantile: float = 0.90,
    quantized: bool = True,
    max_length: int = 512,
    batch_size: int = 64,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
) -> Dict:
    """
    针对单个 anchor 代码，按测试分布生成 K 个等价变体，
    计算“有符号Δ”的正/负分布分位，用于双侧门槛构建（2维方案）。
    """
    # 生成候选（阈值估计固定采样 50 次，不随嵌入阶段 K 变化）
    K_thr = 50
    cands = build_candidates_test_like(anchor_code, max(1, K_thr), num_workers=num_workers, batch_size_for_parallel=batch_size_for_parallel)
    # 过滤无变化
    cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != anchor_code.strip()]
    if not cands:
        return {
            "k": 2,
            "q_neg0": 0.0,
            "q_pos1": 0.0,
            "m_pos0": 0.0,
            "m_neg1": 0.0,
            "T_pos0": 0.0,
            "T_neg1": 0.0,
        }

    # 编码并投影
    model, tokenizer, device = load_encoder(model_dir, use_quantization=quantized)
    anchors = [anchor_code]
    v_anchor = embed_codes(model, tokenizer, anchors, max_length=max_length, batch_size=batch_size, device=device)
    v_cands = embed_codes(model, tokenizer, cands, max_length=max_length, batch_size=batch_size, device=device)

    d = v_anchor.shape[1]
    W = derive_directions(secret_key=secret_key, d=d, k=2)
    s_anchor = project_embeddings(v_anchor, W)[0]  # [2]
    s_cands = project_embeddings(v_cands, W)       # [K,2]

    delta = s_cands - s_anchor[None, :]  # [K,2], 有符号

    # 通用二维分布分位（每维的正向/负向分布）
    pos0 = np.maximum(+delta[:, 0], 0.0)
    neg0 = np.maximum(-delta[:, 0], 0.0)
    pos1 = np.maximum(+delta[:, 1], 0.0)
    neg1 = np.maximum(-delta[:, 1], 0.0)

    q_pos = [float(np.quantile(pos0, quantile)), float(np.quantile(pos1, quantile))]
    q_neg = [float(np.quantile(neg0, quantile)), float(np.quantile(neg1, quantile))]

    # 身份阈值（用于提取与判定）：m_pos[i] 用于判 1，m_neg[i] 用于判 0
    m_pos = [0.1 * q_pos[0], 0.1 * q_pos[1]]
    m_neg = [0.1 * q_neg[0], 0.1 * q_neg[1]]

    # 可选目标阈（注入阶段可能使用的达标目标）
    T_pos = [q_pos[0] + m_pos[0], q_pos[1] + m_pos[1]]
    T_neg = [q_neg[0] + m_neg[0], q_neg[1] + m_neg[1]]

    # 兼容旧字段（当前贪心实现使用的口径）：
    # 保留原有非对称定义：
    #  - q_neg0: 第0维的负向分位（用于信息位推进的防守边界）
    #  - q_pos1: 第1维的正向分位（用于非信息位抑制的防守边界）
    #  - m_pos0 = 0.1*q_neg0, T_pos0 = q_neg0 + m_pos0
    #  - m_neg1 = 0.1*q_pos1, T_neg1 = q_pos1 + m_neg1
    q_neg0_legacy = q_neg[0]
    q_pos1_legacy = q_pos[1]
    m_pos0_legacy = 0.1 * q_neg0_legacy
    m_neg1_legacy = 0.1 * q_pos1_legacy
    T_pos0_legacy = q_neg0_legacy + m_pos0_legacy
    T_neg1_legacy = q_pos1_legacy + m_neg1_legacy

    return {
        "k": 2,
        # 旧字段（供现有注入贪心使用）
        "q_neg0": float(q_neg0_legacy),
        "q_pos1": float(q_pos1_legacy),
        "m_pos0": float(m_pos0_legacy),
        "m_neg1": float(m_neg1_legacy),
        "T_pos0": float(T_pos0_legacy),
        "T_neg1": float(T_neg1_legacy),
        # 新阈值结构（供提取/通用判定使用）
        "thresholds": {
            "m_pos": [float(m_pos[0]), float(m_pos[1])],
            "m_neg": [float(m_neg[0]), float(m_neg[1])],
        },
        # 可选统计（审计用，不影响判定）
        "stats_optional": {
            "quantile": float(quantile),
            "q_pos": [float(q_pos[0]), float(q_pos[1])],
            "q_neg": [float(q_neg[0]), float(q_neg[1])],
            "T_pos": [float(T_pos[0]), float(T_pos[1])],
            "T_neg": [float(T_neg[0]), float(T_neg[1])],
        },
    }


