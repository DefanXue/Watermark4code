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


def _load_bitwise_thresholds(final_json_path: str) -> Dict:
    with open(final_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    bitwise_thresholds = obj.get("bitwise_thresholds", {})
    if not bitwise_thresholds:
        raise ValueError("final.json 缺少 bitwise_thresholds 字段")
    return bitwise_thresholds


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
    水印提取（基于簇中心）：
    - 从 final.json 读取簇中心 s0
    - 计算 offset = s_suspect - s0（簇中心）
    - 逐维判决：offset > 0 → "1", offset < 0 → "0", else → "U"
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
        "offset": None,
        "bits": "UUUU",
        "uncertain_count": 4,
        "error": None,
    }

    try:
        if suspect_path is None:
            raise FileNotFoundError("suspect.java 缺失")
        if not os.path.isfile(final_json_path):
            raise FileNotFoundError("final.json 缺失")

        # 读取簇中心
        with open(final_json_path, "r", encoding="utf-8") as f:
            final_json = json.load(f)
        
        cluster_center = final_json.get("s0", None)
        if cluster_center is None:
            raise ValueError("final.json 缺少 s0（簇中心）字段")

        # 只编码suspect代码
        model, tokenizer, device = load_encoder(model_dir, use_quantization=True)
        suspect_code = _read_text(suspect_path)
        embs = embed_codes(model, tokenizer, [suspect_code], max_length=max_length, batch_size=batch_size, device=device)

        d = embs.shape[1]
        W = derive_directions(secret_key=secret_key, d=d, k=4)
        s_suspect = project_embeddings(embs, W)[0]  # [4]
        
        # 计算相对于簇中心的偏移
        offset = [float(s_suspect[i] - cluster_center[i]) for i in range(4)]
        result["offset"] = offset

        # 逐维判决（纯粹基于符号）
        bits_result = []
        for i in range(4):
            if offset[i] > 0:
                bits_result.append("1")
            elif offset[i] < 0:
                bits_result.append("0")
            else:
                bits_result.append("U")
        
        bits = "".join(bits_result)
        result["bits"] = bits
        result["uncertain_count"] = bits.count("U")

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

    # 预读簇中心与代码文本，构造批处理列表
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
            if suspect_path is None:
                raise FileNotFoundError("suspect.java 与 final.java 均缺失")
            if not os.path.isfile(final_json_path):
                raise FileNotFoundError("final.json 缺失")

            # 读取簇中心
            with open(final_json_path, "r", encoding="utf-8") as f:
                final_json = json.load(f)
            cluster_center = final_json.get("s0", None)
            if cluster_center is None:
                raise ValueError("final.json 缺少 s0（簇中心）字段")
            
            code_susp = _read_text(suspect_path)

            batch_records.append({
                **base_rec,
                "cluster_center": cluster_center,
                "code_suspect": code_susp,
            })
        except Exception as e:
            summary["items"].append({
                **base_rec,
                "offset": None,
                "bits": "UUUU",
                "uncertain_count": 4,
                "error": str(e),
            })

    if not batch_records:
        return summary

    # 单次加载模型，批量嵌入（只编码suspect代码）
    model, tokenizer, device = load_encoder(model_dir, use_quantization=True)

    # 拼接批次：[susp1, susp2, ...]
    codes: List[str] = []
    for rec in batch_records:
        codes.append(rec["code_suspect"])

    embs = embed_codes(model, tokenizer, codes, max_length=max_length, batch_size=batch_size, device=device)
    d = embs.shape[1]
    W = derive_directions(secret_key=secret_key, d=d, k=4)
    s_all = project_embeddings(embs, W)  # [N, 4]

    # 还原逐目录结果
    for i, rec in enumerate(batch_records):
        s_suspect = s_all[i]
        cluster_center = rec["cluster_center"]
        
        # 计算相对于簇中心的偏移
        offset = [float(s_suspect[j] - cluster_center[j]) for j in range(4)]
        
        # 逐维判决（纯粹基于符号）
        bits_result = []
        for j in range(4):
            if offset[j] > 0:
                bits_result.append("1")
            elif offset[j] < 0:
                bits_result.append("0")
            else:
                bits_result.append("U")
        
        bits = "".join(bits_result)
        
        summary["items"].append({
            "dir": rec["dir"],
            "files": rec["files"],
            "offset": offset,
            "bits": bits,
            "uncertain_count": bits.count("U"),
            "error": None,
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


