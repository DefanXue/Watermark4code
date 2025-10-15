# -*- coding: utf-8 -*-
import json, sys, os, re
import numpy as np

def tokenize(code: str):
    # 简单分词：按非字母数字切分，转小写；过滤空串
    toks = re.split(r"\W+", code.lower())
    return [t for t in toks if t]

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def load_pairs(path):
    total=0; pos=0; neg=0
    neg_jaccs=[]
    same_neg=0
    lengths=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            code1 = d.get("code1") or (d.get("pair")[0] if isinstance(d.get("pair"), list) and len(d["pair"])==2 else None)
            code2 = d.get("code2") or (d.get("pair")[1] if isinstance(d.get("pair"), list) and len(d["pair"])==2 else None)
            lab  = d.get("label", d.get("clone", d.get("is_clone", None)))
            if isinstance(lab, str): lab = 1 if lab.lower() in ["true","yes","1","clone","similar"] else 0
            if code1 is None or code2 is None or lab is None: continue
            total+=1
            if lab==1: pos+=1
            else:
                neg+=1
                if code1 == code2:
                    same_neg += 1
                t1 = tokenize(code1); t2 = tokenize(code2)
                neg_jaccs.append(jaccard(t1,t2))
                lengths.append((len(code1), len(code2)))
    return {
        "total": total, "pos": pos, "neg": neg,
        "neg_jaccs": np.array(neg_jaccs, dtype=float),
        "same_neg": same_neg,
        "lengths": lengths
    }

def summarize(res, tag):
    total, pos, neg = res["total"], res["pos"], res["neg"]
    neg_j = res["neg_jaccs"]
    same_neg = res["same_neg"]

    print(f"=== {tag} pairs summary ===")
    print(f"total={total}, pos={pos} ({pos/max(1,total):.2%}), neg={neg} ({neg/max(1,total):.2%})")
    if neg == 0 or len(neg_j)==0:
        print("No negative pairs or no tokens; cannot estimate.")
        return

    # 简单/困难负样本的启发式阈值（可调）
    # - 简单负样本：词汇Jaccard很低（<0.02）或长度相差悬殊
    # - 困难负样本：Jaccard中高（>=0.02）
    easy_by_jacc = float((neg_j < 0.02).sum()) / len(neg_j)
    # 长度阈值：较短/较长长度比 < 0.5 判为“显著长度差异”
    easy_by_len = 0.0
    if res["lengths"]:
        lens = np.array(res["lengths"], dtype=float)
        r = np.minimum(lens[:,0], lens[:,1]) / np.maximum(lens[:,0], lens[:,1])
        easy_by_len = float((r < 0.5).sum()) / len(r)

    # 合并判定：任一条件为真即判“简单”
    easy_mask = (neg_j < 0.02)
    if res["lengths"]:
        lens = np.array(res["lengths"], dtype=float)
        r = np.minimum(lens[:,0], lens[:,1]) / np.maximum(lens[:,0], lens[:,1])
        easy_mask = np.logical_or(easy_mask, (r < 0.5))
    easy_ratio = float(easy_mask.sum()) / len(neg_j)

    print(f"negatives: jaccard stats -> min={neg_j.min():.4f}, p10={np.quantile(neg_j,0.1):.4f}, p25={np.quantile(neg_j,0.25):.4f}, p50={np.quantile(neg_j,0.5):.4f}, p75={np.quantile(neg_j,0.75):.4f}, p90={np.quantile(neg_j,0.9):.4f}, max={neg_j.max():.4f}")
    print(f"negative identical-pairs (code1==code2): {same_neg} ({same_neg/max(1,neg):.2%})")
    print(f"easy_by_jaccard<0.02: {easy_by_jacc:.2%}")
    print(f"easy_by_length_ratio<0.5: {easy_by_len:.2%}")
    print(f"estimated_easy_negative_ratio (union rule): {easy_ratio:.2%}")
    print(f"estimated_hard_negative_ratio: {1.0 - easy_ratio:.2%}")

if __name__ == "__main__":
    val_path = sys.argv[1] if len(sys.argv) > 1 else "contrastive_learning/datasets/csn_java/valid_augmented_pairs.jsonl"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "contrastive_learning/datasets/csn_java/test_augmented_pairs.jsonl"
    for p, tag in [(val_path,"VALID"), (test_path,"TEST")]:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        summarize(load_pairs(p), tag)