"""
Java 变体生成器（与训练/测试构造方式对齐）：
- 三类变换：
  1) 语义保持静态规则（参考 contrastive_learning.java_augmentor.JavaCodeAugmentor）
  2) LLM 重写（参考 java_augmentor.llm_rewrite_java 提示与流程）
  3) 转译（参考 java_augmentor.retranslate_java Java→C#→Java）

比例：默认按训练配置 0.2 / 0.5 / 0.3；若关闭某类，则对启用类别按权重归一化重采样。

注意：本模块仅生成“全新变体”，不复用已有 augmented 数据；LLM/转译依赖外部 API，需按环境自行配置。
"""

import os
import sys
import random
from typing import List, Dict, Tuple


def _ensure_contrastive_learning_on_path() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    water_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    xdf_root = os.path.abspath(os.path.join(water_root, os.pardir))
    srcmarker_root = os.path.join(xdf_root, "SrcMarker-main")
    if srcmarker_root not in sys.path:
        sys.path.append(srcmarker_root)


_ensure_contrastive_learning_on_path()

from contrastive_learning.java_augmentor import (  # type: ignore
    JavaCodeAugmentor,
    llm_rewrite_java,
    retranslate_java,
)


DEFAULT_WEIGHTS = {
    "semantic_preserving": 0.2,
    "llm_rewrite": 0.5,
    "retranslate": 0.3,
}


def _normalize_weights(weights: Dict[str, float], enabled: Dict[str, bool]) -> Dict[str, float]:
    active = {k: v for k, v in weights.items() if enabled.get(k, False) and v > 0}
    s = sum(active.values())
    if s <= 0:
        # 兜底：若全被禁用，默认只启用静态规则
        return {"semantic_preserving": 1.0}
    return {k: v / s for k, v in active.items()}


def generate_variants_for_anchor(
    anchor_code: str,
    num_variants: int,
    weights: Dict[str, float] = None,
    enable_semantic_preserving: bool = True,
    enable_llm_rewrite: bool = True,
    enable_retranslate: bool = True,
    intermediate_lang: str = "csharp",
    seed: int = 42,
) -> List[str]:
    """
    为单个 anchor 生成 num_variants 个“全新变体”，按训练同分布抽样：
    - semantic_preserving: 调用 JavaCodeAugmentor().augment(anchor)
    - llm_rewrite: 调用 llm_rewrite_java
    - retranslate: 调用 retranslate_java（Java→中间→Java）
    """
    rng = random.Random(seed)
    weights = weights or DEFAULT_WEIGHTS
    enabled = {
        "semantic_preserving": enable_semantic_preserving,
        "llm_rewrite": enable_llm_rewrite,
        "retranslate": enable_retranslate,
    }
    norm_w = _normalize_weights(weights, enabled)
    methods = list(norm_w.keys())
    probs = [norm_w[m] for m in methods]

    augmentor = JavaCodeAugmentor()
    out: List[str] = []

    for i in range(num_variants):
        choice = rng.choices(methods, weights=probs, k=1)[0]
        try:
            if choice == "semantic_preserving":
                cands = augmentor.augment(anchor_code)
                variant = cands[0] if cands else anchor_code
            elif choice == "llm_rewrite":
                variant = llm_rewrite_java(anchor_code, model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")})
            elif choice == "retranslate":
                variant = retranslate_java(anchor_code, model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")}, intermediate_lang=intermediate_lang)
            else:
                variant = anchor_code
        except Exception:
            # 单条失败则回退为 anchor 原文，后续可由上层过滤掉“无变化”样本
            variant = anchor_code

        out.append(variant)

    return out









