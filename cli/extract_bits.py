import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

from ..api import load_encoder
from ..encoder import embed_codes
from ..keys.directions import derive_directions
from ..utils.math import project_embeddings


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_thresholds_from_final_json(final_json_path: str) -> Tuple[List[float], List[float]]:
    with open(final_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 1) 优先读取 required_delta_vec.thresholds.m_pos/m_neg（数组，长度为2）
    if isinstance(obj.get("required_delta_vec"), dict) and isinstance(obj["required_delta_vec"].get("thresholds"), dict):
        mp = obj["required_delta_vec"]["thresholds"].get("m_pos")
        mn = obj["required_delta_vec"]["thresholds"].get("m_neg")
        if isinstance(mp, list) and isinstance(mn, list) and len(mp) == 2 and len(mn) == 2:
            return [float(mp[0]), float(mp[1])], [float(mn[0]), float(mn[1])]
    # 2) 其次读取顶层 thresholds.m_pos/m_neg（数组，长度为2）
    if isinstance(obj.get("thresholds"), dict):
        mp = obj["thresholds"].get("m_pos")
        mn = obj["thresholds"].get("m_neg")
        if isinstance(mp, list) and isinstance(mn, list) and len(mp) == 2 and len(mn) == 2:
            return [float(mp[0]), float(mp[1])], [float(mn[0]), float(mn[1])]
    # 3) 否则报错（不再回退到旧键）
    raise ValueError("final.json 缺少 required_delta_vec.thresholds 或顶层 thresholds 的 m_pos/m_neg 数组")


def _find_suspect_file(dir_path: str) -> Optional[str]:
    # 优先使用 suspect.java；若不存在则回退 final.java；都没有则返回 None
    p1 = os.path.join(dir_path, "suspect.java")
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(dir_path, "final.java")
    if os.path.isfile(p2):
        return p2
    return None


def extract_bit_for_dir(
    dir_path: str,
    model_dir: str,
    secret_key: str = "XDF",
    max_length: int = 512,
    batch_size: int = 2,
) -> Dict:
    """
    仅进行水印“提取”：
    - 从 final.json 读取身份阈值 m_pos0/m_neg1
    - 使用 encoder-only 表示与私钥方向（k=2）计算 Δ = s_after - s0
    - 提取规则：Δ0 >= m_pos0 -> '1'；否则若 Δ1 <= -m_neg1 -> '0'；否则 'U'
    不读取 final.json 中任何水印内容（bits/s0/s_after）。
    """
    original_path = os.path.join(dir_path, "original.java")
    final_json_path = os.path.join(dir_path, "final.json")
    suspect_path = _find_suspect_file(dir_path)

    result: Dict = {
        "dir": os.path.basename(dir_path.rstrip(os.sep)),
        "files": {
            "original": original_path,
            "suspect": suspect_path,
            "final_json": final_json_path,
        },
        "thresholds": None,
        "delta": None,
        "bits": None,
        "error": None,
        "note": "threshold_source=final.json (identity only)",
    }

    try:
        if not os.path.isfile(original_path):
            raise FileNotFoundError("original.java 缺失")
        if suspect_path is None:
            raise FileNotFoundError("suspect.java 缺失")
        if not os.path.isfile(final_json_path):
            raise FileNotFoundError("final.json 缺失")

        m_pos_vec, m_neg_vec = _read_thresholds_from_final_json(final_json_path)
        result["thresholds"] = {"m_pos": m_pos_vec, "m_neg": m_neg_vec}

        # 编码并计算投影
        model, tokenizer, device = load_encoder(model_dir, use_quantization=True)
        codes: List[str] = [_read_text(original_path), _read_text(suspect_path)]
        embs = embed_codes(model, tokenizer, codes, max_length=max_length, batch_size=batch_size, device=device)

        d = embs.shape[1]
        W = derive_directions(secret_key=secret_key, d=d, k=2)
        s = project_embeddings(embs, W)  # [2,2]
        s0 = s[0]
        s1 = s[1]
        delta0 = float(s1[0] - s0[0])
        delta1 = float(s1[1] - s0[1])
        result["delta"] = [delta0, delta1]

        # 提取位（二维独立口径）：逐维输出并拼接为2位
        def _bit_from_delta(d: float, m_pos: float, m_neg: float) -> str:
            if d >= m_pos:
                return "1"
            if d <= -m_neg:
                return "0"
            return "U"

        bit0 = _bit_from_delta(delta0, m_pos_vec[0], m_neg_vec[0])
        bit1 = _bit_from_delta(delta1, m_pos_vec[1], m_neg_vec[1])
        result["bits"] = f"{bit0}{bit1}"

    except Exception as e:
        result["error"] = str(e)

    return result


def extract_bits_for_root(root_dir: str, model_dir: str, secret_key: str = "XDF", max_length: int = 512, batch_size: int = 2) -> Dict:
    # 收集 run 目录
    entries: List[str] = []
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        has_original = os.path.isfile(os.path.join(sub, "original.java"))
        has_final_json = os.path.isfile(os.path.join(sub, "final.json"))
        has_suspect = os.path.isfile(os.path.join(sub, "suspect.java"))
        has_final_java = os.path.isfile(os.path.join(sub, "final.java"))
        # 放宽条件：original.java 与 final.json 必须存在，且 suspect.java 或 final.java 任一存在即可
        if has_original and has_final_json and (has_suspect or has_final_java):
            entries.append(sub)

    summary = {"total": len(entries), "items": []}

    if not entries:
        return summary

    # 预读阈值与代码文本，构造批处理列表
    batch_records: List[Dict] = []
    for sub in entries:
        original_path = os.path.join(sub, "original.java")
        final_json_path = os.path.join(sub, "final.json")
        suspect_path = _find_suspect_file(sub)

        base_rec = {
            "dir": os.path.basename(sub.rstrip(os.sep)),
            "files": {
                "original": original_path,
                "suspect": suspect_path,
                "final_json": final_json_path,
            }
        }

        try:
            if not os.path.isfile(original_path):
                raise FileNotFoundError("original.java 缺失")
            if suspect_path is None:
                raise FileNotFoundError("suspect.java 与 final.java 均缺失")
            if not os.path.isfile(final_json_path):
                raise FileNotFoundError("final.json 缺失")

            m_pos_vec, m_neg_vec = _read_thresholds_from_final_json(final_json_path)
            code_orig = _read_text(original_path)
            code_susp = _read_text(suspect_path)

            batch_records.append({
                **base_rec,
                "thresholds": {"m_pos": m_pos_vec, "m_neg": m_neg_vec},
                "code_original": code_orig,
                "code_suspect": code_susp,
            })
        except Exception as e:
            summary["items"].append({
                **base_rec,
                "thresholds": None,
                "delta": None,
        "bits": None,
                "error": str(e),
                "note": "threshold_source=final.json (identity only)",
            })

    if not batch_records:
        return summary

    # 单次加载模型，批量嵌入
    model, tokenizer, device = load_encoder(model_dir, use_quantization=True)

    # 拼接批次：[orig1, susp1, orig2, susp2, ...]
    codes: List[str] = []
    for rec in batch_records:
        codes.append(rec["code_original"])
        codes.append(rec["code_suspect"])

    embs = embed_codes(model, tokenizer, codes, max_length=max_length, batch_size=batch_size, device=device)
    d = embs.shape[1]
    W = derive_directions(secret_key=secret_key, d=d, k=2)
    s_all = project_embeddings(embs, W)  # [2*N, 2]

    # 还原逐目录结果
    for i, rec in enumerate(batch_records):
        s0 = s_all[2 * i]
        s1 = s_all[2 * i + 1]
        delta0 = float(s1[0] - s0[0])
        delta1 = float(s1[1] - s0[1])
        m_pos_vec = rec["thresholds"]["m_pos"]
        m_neg_vec = rec["thresholds"]["m_neg"]

        def _bit_from_delta(d: float, m_pos: float, m_neg: float) -> str:
            if d >= m_pos:
                return "1"
            if d <= -m_neg:
                return "0"
            return "U"
        bit0 = _bit_from_delta(delta0, m_pos_vec[0], m_neg_vec[0])
        bit1 = _bit_from_delta(delta1, m_pos_vec[1], m_neg_vec[1])
        bits = f"{bit0}{bit1}"

        summary["items"].append({
            "dir": rec["dir"],
            "files": rec["files"],
            "thresholds": rec["thresholds"],
            "delta": [delta0, delta1],
            "bits": bits,
            "error": None,
            "note": "threshold_source=final.json (identity only)",
        })

    return summary


def main():
    p = argparse.ArgumentParser(description="Extract watermark bit from directories using identity thresholds in final.json")
    p.add_argument("--dir", type=str, required=True, help="Root directory containing run_* subfolders")
    p.add_argument("--model_dir", type=str, default=r"D:\kyl410\XDF\Watermark4code\best_model")
    p.add_argument("--secret", type=str, default="XDF")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--out", type=str, default=None, help="Where to write extract_summary.json; default to --dir")
    args = p.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    summary = extract_bits_for_root(
        root_dir=args.dir,
        model_dir=args.model_dir,
        secret_key=args.secret,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    out_path = os.path.join(out_dir, "extract_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Extraction summary written to {out_path}")


if __name__ == "__main__":
    main()


