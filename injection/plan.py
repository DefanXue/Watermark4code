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
import sys


def _ensure_srcmarker_on_path() -> None:
    """确保优先解析 XDF/SrcMarker-main 下的 contrastive_learning。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xdf_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    srcmarker_root = os.path.join(xdf_root, "SrcMarker-main")
    if srcmarker_root in sys.path:
        sys.path.remove(srcmarker_root)
    sys.path.insert(0, srcmarker_root)


_ensure_srcmarker_on_path()

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


def compute_required_delta(epsilon_json_path: str, tmargin_json_path: str, quantile: float = 0.90) -> float:
    with open(epsilon_json_path, "r", encoding="utf-8") as f:
        eps_obj = json.load(f)
    with open(tmargin_json_path, "r", encoding="utf-8") as f:
        tm_obj = json.load(f)

    # epsilon: quantiles hold floats
    eps_entry = _get_quantile_entry(eps_obj, quantile)
    eps = float(eps_entry) if eps_entry is not None else float(eps_obj.get("epsilon_emp", 0.0))

    # t_margin: quantiles hold dict with {per_bit, scalar}
    tm_entry = _get_quantile_entry(tm_obj, quantile)
    if isinstance(tm_entry, dict) and "scalar" in tm_entry:
        tm_scalar = float(tm_entry["scalar"])
    else:
        tm_scalar = float(tm_obj.get("scalar", 0.0))

    return eps + tm_scalar


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
) -> Tuple[List[str], Dict]:
    """
    Match test-split generation: split_type='test', positive_ratio=1.0, proportions 0.2/0.5/0.3.
    To get K variants, duplicate the same anchor K times as input.
    
    Returns:
        (cands, stats): 
            cands - 通过审核的变体列表
            stats - {"passed_count": int, "failed_reasons": {"原因1": 计数, ...}}
    """
    aug_types = {
        "semantic_preserving": 0.7,  # 提高到 70% 以降低 token 消耗
        "llm_rewrite": 0.3,           # 降低到 30%
        "retranslate": 0.0,
    }

    in_file = _write_anchor_copies_tmp(anchor_code, K)
    out_dir = tempfile.mkdtemp(prefix="wm_aug_")
    out_file = os.path.join(out_dir, "augmented.jsonl")
    review_stats_file = os.path.join(out_dir, "review_stats.jsonl")

    # 通过环境变量传递统计文件路径
    os.environ['REVIEW_STATS_FILE'] = review_stats_file
    
    try:
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
    finally:
        if 'REVIEW_STATS_FILE' in os.environ:
            del os.environ['REVIEW_STATS_FILE']

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

    # 读取审核统计
    passed_count = 0
    failed_reasons = {}
    if os.path.exists(review_stats_file):
        with open(review_stats_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("passed"):
                        passed_count += 1
                    else:
                        reason = record.get("reason", "未知原因")
                        failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                except Exception:
                    continue
    
    stats = {
        "passed_count": passed_count,
        "failed_reasons": failed_reasons
    }

    # 去重并截取 K
    uniq: List[str] = []
    seen = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= K:
            break
    return uniq, stats


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
    bits: List[int],
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
    针对单个 anchor，按测试分布生成 K_thr 等价候选，计算二维有符号Δ的分布分位，
    返回旧字段（兼容贪心）与通用 thresholds.m_pos/m_neg 以及可选统计信息。
    """
    # 1) 生成候选（门槛由底层生成器负责），阈值估计固定用 50 个样本
    K_thr = 50
    try:
        cands, review_stats = build_candidates_test_like(
            anchor_code,
            max(1, K_thr),
            num_workers=num_workers,
            batch_size_for_parallel=batch_size_for_parallel,
        )
    except Exception:
        cands = []
        review_stats = {"passed_count": 0, "failed_reasons": {}}

    # 过滤无变化
    cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != anchor_code.strip()]
    if not cands:
        result = {"k": 4, "review_stats": review_stats}
        for i in range(4):
            if bits[i] == 1:
                result[f"q_neg{i}"] = 0.0
                result[f"m_pos{i}"] = 0.0
                result[f"T_pos{i}"] = 0.0
            else:
                result[f"q_pos{i}"] = 0.0
                result[f"m_neg{i}"] = 0.0
                result[f"T_neg{i}"] = 0.0
        return result

    # 2) 编码并投影
    model, tokenizer, device = load_encoder(model_dir, use_quantization=quantized)
    v_anchor = embed_codes(model, tokenizer, [anchor_code], max_length=max_length, batch_size=batch_size, device=device)
    v_cands = embed_codes(model, tokenizer, cands, max_length=max_length, batch_size=batch_size, device=device)

    d = v_anchor.shape[1]
    W = derive_directions(secret_key=secret_key, d=int(d), k=4)
    s_anchor = project_embeddings(v_anchor, W)[0]  # [4]
    s_cands = project_embeddings(v_cands, W)       # [K,4]

    delta = s_cands - s_anchor[None, :]  # [K,4]

    # 3) 分布分位（每维的正向/负向组件，根据bits值动态决定）
    result = {"k": 4, "review_stats": review_stats}
    
    for i in range(4):
        pos_i = np.maximum(+delta[:, i], 0.0)
        neg_i = np.maximum(-delta[:, i], 0.0)
        
        if bits[i] == 1:
            # bit=1: 需要推正，计算负向分位和正向阈值
            q_neg_i = float(np.quantile(neg_i, quantile))
            m_pos_i = 0.1 * q_neg_i
            T_pos_i = q_neg_i + m_pos_i
            result[f"q_neg{i}"] = q_neg_i
            result[f"m_pos{i}"] = m_pos_i
            result[f"T_pos{i}"] = T_pos_i
        else:  # bits[i] == 0
            # bit=0: 需要拉负，计算正向分位和负向阈值
            q_pos_i = float(np.quantile(pos_i, quantile))
            m_neg_i = 0.1 * q_pos_i
            T_neg_i = q_pos_i + m_neg_i
            result[f"q_pos{i}"] = q_pos_i
            result[f"m_neg{i}"] = m_neg_i
            result[f"T_neg{i}"] = T_neg_i

    return result

