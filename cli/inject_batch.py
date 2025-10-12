import os
import json
import argparse
import random
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..injection.plan import compute_required_delta_per_anchor
from ..injection.greedy import select_and_inject
from ..injection.io import save_text, save_json


def _load_anchors(path: str) -> List[str]:
    anchors: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            code = (obj.get('code') or '').strip()
            if code:
                anchors.append(code)
    return anchors


def process_one_anchor(task):
    """
    子进程任务：处理单个 anchor。参数通过基础类型传递，兼容 Windows spawn。
    输入: (idx, code, args_dict)
    返回: {idx, ok, delta, error}
    """
    idx, code, args_dict = task
    run_dir = os.path.join(args_dict["out_dir"], f"run_{idx:04d}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        # 保存原始代码
        save_text(os.path.join(run_dir, 'original.java'), code)

        # 计算该 anchor 的阈值
        required_delta = compute_required_delta_per_anchor(
            model_dir=args_dict["model_dir"],
            anchor_code=code,
            secret_key=args_dict["secret"],
            K=args_dict["K"],
            quantile=args_dict["quantile"],
            num_workers=args_dict["num_workers"],
            batch_size_for_parallel=args_dict["batch_size_for_parallel"],
        )

        # 注入选择
        res = select_and_inject(
            model_dir=args_dict["model_dir"],
            anchor_code=code,
            bits=args_dict["bits"],
            required_delta=required_delta,
            secret_key=args_dict["secret"],
            K=args_dict["K"],
            max_iters=args_dict["max_iters"],
            num_workers=args_dict["num_workers"],
            batch_size_for_parallel=args_dict["batch_size_for_parallel"],
            save_dir=run_dir,
        )

        # 写结果文件
        save_text(os.path.join(run_dir, 'final.java'), res['final_code'])
        save_json(os.path.join(run_dir, 'final.json'), {
            'bits': args_dict["bits_str"],
            'required_delta_vec': required_delta,
            **{k: v for k, v in res.items() if k != 'final_code'}
        })

        # 判定是否达标（2维方案：信息位第0维推正、非信息位第1维拉负；验证口径只看身份阈值）
        s0 = res['s0']
        sa = res['s_after']
        delta = [a - b for a, b in zip(sa, s0)]
        m_pos0 = float(required_delta.get("m_pos0", 0.0))
        m_neg1 = float(required_delta.get("m_neg1", 0.0))
        ok = (delta[0] >= m_pos0) and (delta[1] <= -m_neg1)
        return {"idx": idx, "ok": ok, "delta": delta, "error": None}
    except Exception as e:
        return {"idx": idx, "ok": False, "delta": None, "error": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=False, default=r"D:\kyl410\XDF\Watermark4code\best_model")
    p.add_argument('--anchors', type=str, required=True, help='csn_java filtered_code.jsonl')
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--sample', type=str, choices=['random','sequential'], default='random')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bits', type=str, required=True)
    # 移除全局 epsilon/tmargin 依赖，按 anchor 现算阈值
    p.add_argument('--quantile', type=float, default=0.90)
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--max_iters', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=48)
    p.add_argument('--batch_size_for_parallel', type=int, default=10)
    p.add_argument('--secret', type=str, default='XDF')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--concurrency', type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # required_delta 将在每个 anchor 上单独计算
    bits = [1 if ch == '1' else 0 for ch in args.bits.strip()]

    # load anchors and choose N
    anchors = _load_anchors(args.anchors)
    if len(anchors) == 0:
        save_json(os.path.join(args.out_dir, 'summary.json'), {
            'error': 'no anchors loaded',
            'anchors_path': args.anchors
        })
        return

    if args.sample == 'random':
        rnd = random.Random(args.seed)
        rnd.shuffle(anchors)
    # sequential -> keep order

    anchors = anchors[:max(0, args.N)]

    summary: Dict = {
        'total': len(anchors),
        'success': 0,
        'failed': 0,
        'items': []
    }

    # 组装任务参数（基础类型，便于跨进程传递）
    args_dict = {
        "model_dir": args.model_dir,
        "out_dir": args.out_dir,
        "secret": args.secret,
        "K": args.K,
        "quantile": args.quantile,
        "max_iters": args.max_iters,
        "num_workers": args.num_workers,
        "batch_size_for_parallel": args.batch_size_for_parallel,
        "bits": bits,
        "bits_str": args.bits,
    }
    tasks = [(idx, code, args_dict) for idx, code in enumerate(anchors)]

    if args.concurrency and args.concurrency > 1:
        with ProcessPoolExecutor(max_workers=int(args.concurrency)) as ex:
            futs = [ex.submit(process_one_anchor, t) for t in tasks]
            results = [f.result() for f in as_completed(futs)]
    else:
        # 顺序执行（与原逻辑等价）
        results = [process_one_anchor(t) for t in tasks]

    # 按 idx 排序并汇总
    results.sort(key=lambda r: r["idx"])  # 保持与 anchors 顺序一致
    for r in results:
        if r["ok"]:
            summary['success'] += 1
        else:
            summary['failed'] += 1
        summary['items'].append({k: r[k] for k in ('idx','ok','delta','error') if k in r})

    save_json(os.path.join(args.out_dir, 'summary.json'), summary)


if __name__ == '__main__':
    main()


