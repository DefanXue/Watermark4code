import os
import json
import argparse
from typing import Dict, List, Tuple

from ..api import load_encoder
from ..encoder import embed_codes
from ..keys.directions import derive_directions
from ..utils.math import project_embeddings


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_final_fields(final_json_path: str) -> Dict:
    with open(final_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 提取 m_pos0/m_neg1
    m_pos0 = obj.get("m_pos0")
    m_neg1 = obj.get("m_neg1")
    if m_pos0 is None or m_neg1 is None:
        req = obj.get("required_delta_vec") or {}
        m_pos0 = req.get("m_pos0", 0.0)
        m_neg1 = req.get("m_neg1", 0.0)
    # 提取 s0/s_after（若存在可直接使用）
    s0 = obj.get("s0")
    s_after = obj.get("s_after")
    return {
        "m_pos0": float(m_pos0),
        "m_neg1": float(m_neg1),
        "s0": s0,
        "s_after": s_after,
    }


def verify_dir(
    root_dir: str,
    model_dir: str,
    secret_key: str = "XDF",
    max_length: int = 512,
    batch_size: int = 2,
) -> Dict:
    """
    扫描 root_dir 下的子目录：每个目录应包含 original.java 与 final.json（可选 final.java）。
    判定口径（2维/身份=10）：Δ0≥m_pos0 且 Δ1≤−m_neg1。
    若 final.json 中含 s0/s_after，直接使用；否则用 original/final 计算。
    返回 summary 字典。
    """
    # 收集 run 目录
    entries = []
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        orig = os.path.join(sub, "original.java")
        fjson = os.path.join(sub, "final.json")
        fcode = os.path.join(sub, "final.java")
        if os.path.isfile(orig) and os.path.isfile(fjson):
            entries.append((name, sub, orig, fjson, fcode if os.path.isfile(fcode) else None))

    summary = {"total": len(entries), "success": 0, "failed": 0, "items": []}

    # 延迟加载 encoder，仅在需要计算 s0/s_after 时加载
    model = tokenizer = device = None

    for name, sub, orig, fjson, fcode in entries:
        try:
            fields = _load_final_fields(fjson)
            m_pos0 = fields["m_pos0"]
            m_neg1 = fields["m_neg1"]

            s0 = fields["s0"]
            s_after = fields["s_after"]

            # 如果 final.json 未内置 s0/s_after，则用 original/final 计算
            if s0 is None or s_after is None:
                if model is None:
                    model, tokenizer, device = load_encoder(model_dir, use_quantization=True)
                codes: List[str] = [_read_text(orig)]
                if fcode:
                    codes.append(_read_text(fcode))
                else:
                    raise RuntimeError("final.java 缺失，且 final.json 未提供 s0/s_after，无法验证")
                embs = embed_codes(model, tokenizer, codes, max_length=max_length, batch_size=batch_size, device=device)
                W = derive_directions(secret_key=secret_key, d=embs.shape[1], k=2)
                proj = project_embeddings(embs, W)
                s0 = proj[0].tolist()
                s_after = proj[1].tolist()

            delta0 = float(s_after[0]) - float(s0[0])
            delta1 = float(s_after[1]) - float(s0[1])
            ok = (delta0 >= m_pos0) and (delta1 <= -m_neg1)
            summary["success" if ok else "failed"] += 1
            summary["items"].append({
                "run": name,
                "ok": ok,
                "delta": [delta0, delta1],
                "m_pos0": m_pos0,
                "m_neg1": m_neg1,
            })
        except Exception as e:
            summary["failed"] += 1
            summary["items"].append({"run": name, "ok": False, "error": str(e)})

    return summary


def main():
    p = argparse.ArgumentParser(description="Verify watermarks for a directory of runs")
    p.add_argument("--dir", type=str, required=True, help="Root directory containing run_* subfolders")
    p.add_argument("--model_dir", type=str, default=r"D:\kyl410\XDF\Watermark4code\best_model")
    p.add_argument("--secret", type=str, default="XDF")
    p.add_argument("--out", type=str, default=None, help="Where to write verify_summary.json; default to --dir")
    args = p.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    summary = verify_dir(root_dir=args.dir, model_dir=args.model_dir, secret_key=args.secret)
    out_path = os.path.join(out_dir, "verify_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Verification summary written to {out_path}")


if __name__ == "__main__":
    main()


