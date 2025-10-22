"""
Greedy selection loop: measure-then-add iteration.
"""

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..api import load_encoder
from ..encoder import embed_codes
from ..utils import project_embeddings
from .plan import build_candidates_test_like, build_candidates_by_type, compute_baseline
from .evaluate import measure_gains


def select_and_inject(
    model_dir: str,
    anchor_code: str,
    bits: List[int],
    required_delta,
    secret_key: str = "XDF",
    K: int = 50,
    max_iters: int = 8,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
    max_accept_per_round: int = 3,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Return dict with keys: final_code, s0, s_after, gains_trace[List[Dict]].
    """
    model, tokenizer, device = load_encoder(model_dir, use_quantization=True)

    baseline = compute_baseline(model_dir, anchor_code, secret_key=secret_key)
    s0 = baseline["s0"].astype(np.float32)
    W = baseline["W"].astype(np.float32)

    # 2维方案：信息位=第0维（推正），非信息位=第1维（拉负）
    # required_delta 现在是 dict，包含 q_neg0/q_pos1 及 T/m
    q_neg0 = float(required_delta.get("q_neg0", 0.0))
    q_pos1 = float(required_delta.get("q_pos1", 0.0))
    m_pos0 = float(required_delta.get("m_pos0", 0.0))
    m_neg1 = float(required_delta.get("m_neg1", 0.0))
    T_pos0 = float(required_delta.get("T_pos0", q_neg0 + m_pos0))
    T_neg1 = float(required_delta.get("T_neg1", q_pos1 + m_neg1))

    trace: List[Dict] = []
    current_code = anchor_code
    current_s0 = s0.copy()

    for it in range(max_iters):
        # 为三类变体各生成 K 个候选，合并为一个列表（三类并发触发）
        all_cands = []
        all_types = []
        aug_types = ("semantic_preserving", "llm_rewrite")
        with ThreadPoolExecutor(max_workers=len(aug_types)) as ex:
            futs = {
                ex.submit(
                    build_candidates_by_type,
                    current_code,
                    K,
                    aug_type,
                    num_workers,
                    batch_size_for_parallel,
                ): aug_type for aug_type in aug_types
            }
            for fut in as_completed(futs):
                aug_type = futs[fut]
                try:
                    cands = fut.result()
                except Exception:
                    cands = []
                # 过滤无变化
                cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != current_code.strip()]
                if cands:
                    all_cands.extend(cands)
                    all_types.extend([aug_type] * len(cands))

        if not all_cands:
            break  # 无候选，停止迭代

        # 统一计算所有候选的增益
        g = measure_gains(model, tokenizer, W, current_s0, all_cands, device=device)

        # 双侧打分：第0维奖励正向推进至 T_pos0；第1维奖励负向推进至 T_neg1；
        # 惩罚：第0维负向回退、第1维正向外溢。
        penalty_pos_overflow = 0.2
        penalty_neg_back = 0.2
        p = 1.5

        delta_now = current_s0 - s0  # [2]
        # 剩余缺口（信息位正向、非信息位负向）
        rem_pos0 = max(T_pos0 - float(delta_now[0]), 0.0)
        rem_neg1 = max(T_neg1 - float(-delta_now[1]), 0.0)

        # 自适应权重（按缺口大小）
        rem_vec = np.array([rem_pos0, rem_neg1], dtype=np.float32)
        rem_pow = np.power(rem_vec, p)
        denom = rem_pow.sum() + 1e-8
        alpha = rem_pow / denom  # [2]

        # 奖励项
        gain_pos0 = np.minimum(np.clip(g[:, 0], 0.0, None), rem_pos0) * alpha[0]
        gain_neg1 = np.minimum(np.clip(-g[:, 1], 0.0, None), rem_neg1) * alpha[1]
        reward = gain_pos0 + gain_neg1

        # 惩罚项
        punish_back0 = np.clip(-g[:, 0], 0.0, None) * penalty_neg_back
        punish_over1 = np.clip(g[:, 1], 0.0, None) * penalty_pos_overflow
        scores = reward - (punish_back0 + punish_over1)

        try:
            trace.append({
                "iter": it + 1,
                "phase": "all",
                "candidates_gains": g.tolist(),
                "candidates_scores": scores.tolist(),
                "candidates_types": all_types
            })
        except Exception:
            pass

        # 全局选择分数最高且 > 0 的候选
        order = np.argsort(-scores)
        best_idx = None
        for idx in order:
            if scores[idx] > 0:
                best_idx = int(idx)
                break
        if best_idx is None:
            break  # 无正分候选，停止迭代

        best_gain = g[best_idx]
        best_type = all_types[best_idx]
        trace.append({
            "iter": it + 1,
            "phase": best_type,
            "accepted_index": best_idx,
            "gain": best_gain.tolist(),
        })
        best_code = all_cands[best_idx]
        if save_dir:
            try:
                snap_path = os.path.join(save_dir, f"iter_{it+1:04d}.java")
                with open(snap_path, "w", encoding="utf-8") as f:
                    f.write(best_code)
            except Exception:
                pass
        current_code = best_code
        base2 = compute_baseline(model_dir, current_code, secret_key=secret_key)
        current_s0 = base2["s0"].astype(np.float32)

        # 达标检查（双侧）：Δ0≥T_pos0 且 Δ1≤−T_neg1
        delta_now = current_s0 - s0
        if (float(delta_now[0]) >= T_pos0) and (float(delta_now[1]) <= -T_neg1):
            break

    s_after = current_s0
    return {
        "final_code": current_code,
        "s0": s0.tolist(),
        "s_after": s_after.tolist(),
        "trace": trace,
        "q_neg0": q_neg0,
        "q_pos1": q_pos1,
        "m_pos0": m_pos0,
        "m_neg1": m_neg1,
        "T_pos0": T_pos0,
        "T_neg1": T_neg1,
    }


