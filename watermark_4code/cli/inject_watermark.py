import os
import json
import argparse

from ..injection.plan import compute_required_delta_per_anchor, compute_baseline, build_candidates_test_like
from ..injection.greedy import select_and_inject
from ..injection.io import save_json, save_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', type=str, required=False, default=r"D:\kyl410\XDF\Watermark4code\best_model")
    p.add_argument('--secret', type=str, default='XDF')
    p.add_argument('--bits', type=str, required=True, help='e.g., 1011')
    # 取消全局 epsilon/tmargin 文件依赖，改为每个 anchor 现算 ε_emp 阈值
    p.add_argument('--quantile', type=float, default=0.90)
    p.add_argument('--K', type=int, default=50)
    p.add_argument('--max_iters', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=48)
    p.add_argument('--batch_size_for_parallel', type=int, default=20)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input_code', type=str)
    g.add_argument('--input_file', type=str)
    p.add_argument('--out_dir', type=str, required=True)
    args = p.parse_args()

    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            anchor_code = f.read()
    else:
        anchor_code = args.input_code

    required_delta = compute_required_delta_per_anchor(
        model_dir=args.model_dir,
        anchor_code=anchor_code,
        secret_key=args.secret,
        K=args.K,
        quantile=args.quantile,
        num_workers=args.num_workers,
        batch_size_for_parallel=args.batch_size_for_parallel,
    )
    bits = [1 if ch == '1' else 0 for ch in args.bits.strip()]

    # 保存原始代码形态
    os.makedirs(args.out_dir, exist_ok=True)
    save_text(os.path.join(args.out_dir, 'original.java'), anchor_code)

    result = select_and_inject(
        model_dir=args.model_dir,
        anchor_code=anchor_code,
        bits=bits,
        required_delta=required_delta,
        secret_key=args.secret,
        K=args.K,
        max_iters=args.max_iters,
        num_workers=args.num_workers,
        batch_size_for_parallel=args.batch_size_for_parallel,
    )

    save_text(os.path.join(args.out_dir, 'final.java'), result['final_code'])
    save_json(os.path.join(args.out_dir, 'final.json'), {
        'bits': args.bits,
        'required_delta_vec': required_delta,
        **{k: v for k, v in result.items() if k != 'final_code'}
    })


if __name__ == '__main__':
    main()


