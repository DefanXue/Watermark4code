import os
import json
import argparse

from ..epsilon.estimate import estimate_epsilon_emp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--anchors', type=str, required=True)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--M', type=int, default=20)
    parser.add_argument('--secret', type=str, default='XDF')
    parser.add_argument('--quantized', action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--quantile', type=float, default=0.99)
    parser.add_argument('--enable_semantic', action='store_true', default=True)
    parser.add_argument('--enable_llm', action='store_true', default=True)
    parser.add_argument('--enable_retranslate', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='experiments/epsilon/epsilon_emp.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = estimate_epsilon_emp(
        model_dir=args.model_dir,
        anchors_path=args.anchors,
        N=args.N,
        M=args.M,
        secret_key=args.secret,
        quantized=args.quantized,
        max_length=args.max_length,
        batch_size=args.batch_size,
        quantile=args.quantile,
        enable_semantic_preserving=args.enable_semantic,
        enable_llm_rewrite=args.enable_llm,
        enable_retranslate=args.enable_retranslate,
        seed=args.seed,
    )
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()









