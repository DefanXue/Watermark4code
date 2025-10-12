import os
import json
import argparse

from ..calibration.tmargin import calibrate_tmargin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_pairs', type=str, required=True)
    parser.add_argument('--secret', type=str, default='XDF')
    parser.add_argument('--quantized', action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--quantile', type=float, default=0.99)
    parser.add_argument('--out', type=str, default='experiments/calibration/t_margin.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = calibrate_tmargin(
        model_dir=args.model_dir,
        test_pairs_path=args.test_pairs,
        secret_key=args.secret,
        quantized=args.quantized,
        max_length=args.max_length,
        batch_size=args.batch_size,
        quantile=args.quantile,
    )
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()









