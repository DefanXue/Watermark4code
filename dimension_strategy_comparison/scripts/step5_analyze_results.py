"""
Step 5: 分析并汇总结果

输入：results/strategy_X/extraction/
输出：analysis/per_strategy_summary.json, comparison_table.json, robustness_curves.json
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def collect_extraction_results(strategy_name, attack_ratios):
    """收集单个策略的所有提取结果（支持新的攻击类型目录结构）"""
    results_by_attack = defaultdict(list)

    extraction_dir = f"results/{strategy_name}/extraction"
    if not os.path.exists(extraction_dir):
        return results_by_attack

    # 遍历所有攻击类型目录
    for attack_dir_name in os.listdir(extraction_dir):
        attack_dir_path = os.path.join(extraction_dir, attack_dir_name)

        if not os.path.isdir(attack_dir_path):
            continue

        # 读取该攻击类型下所有的结果文件
        for filename in os.listdir(attack_dir_path):
            if filename.endswith('.json'):
                filepath = os.path.join(attack_dir_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        # 使用attack_type作为key（来自结果文件中的attack_type字段）
                        attack_type = result.get('attack_type', attack_dir_name)
                        results_by_attack[attack_type].append(result)
                except:
                    continue

    return results_by_attack


def compute_strategy_metrics(results_by_attack):
    """计算策略的各项指标（按攻击类型）"""
    metrics = {}

    for attack_type, results in results_by_attack.items():
        if not results:
            continue

        # 提取成功率
        success_count = sum([1 for r in results if r['success']])
        success_rate = success_count / len(results)

        # 比特准确率
        bit_accuracies = [r['bit_accuracy'] for r in results]
        avg_bit_accuracy = np.mean(bit_accuracies)

        # CodeBLEU
        codebleu_scores = [r['codebleu'] for r in results if r.get('codebleu') is not None]
        avg_codebleu = np.mean(codebleu_scores) if codebleu_scores else None

        # 偏移保持率
        sign_preservation_rates = [
            r['offset_preservation']['sign_preservation_rate']
            for r in results
        ]
        avg_sign_preservation = np.mean(sign_preservation_rates)

        magnitude_retentions = [
            r['offset_preservation']['avg_magnitude_retention']
            for r in results
        ]
        avg_magnitude_retention = np.mean(magnitude_retentions)

        metrics[attack_type] = {
            'attack_type': attack_type,
            'num_samples': len(results),
            'success_rate': float(success_rate),
            'avg_bit_accuracy': float(avg_bit_accuracy),
            'avg_codebleu': float(avg_codebleu) if avg_codebleu is not None else None,
            'offset_preservation': {
                'avg_sign_preservation_rate': float(avg_sign_preservation),
                'avg_magnitude_retention': float(avg_magnitude_retention)
            },
            'success_count': success_count
        }

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Step 5: 分析结果")
    parser.add_argument('--strategy', type=str, default=None, help='指定处理的策略（如果不指定，处理所有策略）')
    parser.add_argument('--output-dir', type=str, default='analysis', help='输出目录（默认: analysis）')
    args = parser.parse_args()

    print("="*80)
    print("Step 5: 分析结果")
    if args.strategy:
        print(f"  仅处理策略: {args.strategy}")
    print("="*80)

    # 加载基础配置
    with open("configs/base_config.json", 'r') as f:
        base_config = json.load(f)

    attack_ratios = base_config['extraction']['attack_ratios']

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要处理的策略
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = [
            "strategy_5_learned",
            "strategy_6_adaptive",
        ]
    
    # 分析每个策略
    print("\n分析各策略...")
    all_strategy_summaries = {}

    for strategy_name in strategies:
        print(f"  {strategy_name}...")

        # 收集结果（自动遍历所有攻击类型目录）
        results_by_attack = collect_extraction_results(strategy_name, attack_ratios)

        # 计算指标
        metrics = compute_strategy_metrics(results_by_attack)

        all_strategy_summaries[strategy_name] = {
            'strategy': strategy_name,
            'metrics_by_attack_type': metrics
        }
    
    # 保存各策略汇总
    summary_path = os.path.join(args.output_dir, "per_strategy_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_strategy_summaries, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 已保存 {summary_path}")
    
    # 生成对比表（按攻击类型）
    print("\n生成对比表...")
    comparison = {
        'strategies': strategies,
        'comparison': {}
    }

    for metric_name in ['success_rate', 'avg_bit_accuracy', 'avg_codebleu']:
        comparison['comparison'][metric_name] = {}

        for strategy_name in strategies:
            comparison['comparison'][metric_name][strategy_name] = {}

            # 遍历该策略的所有攻击类型
            for attack_type, metric_data in all_strategy_summaries[strategy_name]['metrics_by_attack_type'].items():
                comparison['comparison'][metric_name][strategy_name][attack_type] = metric_data.get(metric_name)

    comparison_path = os.path.join(args.output_dir, "comparison_table.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 已保存 {comparison_path}")
    
    # 生成鲁棒性曲线数据（仅针对rename攻击）
    print("\n生成鲁棒性曲线数据...")
    robustness_curves = {
        'x_axis': 'attack_ratio',
        'y_axis': 'success_rate',
        'attack_type': 'rename',
        'curves': {}
    }

    for strategy_name in strategies:
        curve_data = []
        for ratio in sorted(attack_ratios):
            ratio_key = f"rename_{ratio:.2f}"
            if ratio_key in all_strategy_summaries[strategy_name]['metrics_by_attack_type']:
                success_rate = all_strategy_summaries[strategy_name]['metrics_by_attack_type'][ratio_key]['success_rate']
                curve_data.append({
                    'x': ratio,
                    'y': success_rate
                })
            else:
                curve_data.append({
                    'x': ratio,
                    'y': None
                })

        robustness_curves['curves'][strategy_name] = curve_data

    robustness_path = os.path.join(args.output_dir, "robustness_curves.json")
    with open(robustness_path, 'w', encoding='utf-8') as f:
        json.dump(robustness_curves, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 已保存 {robustness_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("结果摘要")
    print("="*80)
    
    print("\n提取成功率对比（按攻击类型）：")

    # 获取所有攻击类型（从任意一个策略的指标中提取）
    if strategies and all_strategy_summaries[strategies[0]]['metrics_by_attack_type']:
        attack_types = sorted(all_strategy_summaries[strategies[0]]['metrics_by_attack_type'].keys())
        attack_types_display = " | ".join([f"{at:<12}" for at in attack_types])
        print(f"{'策略':<25} | {attack_types_display}")
        print("-" * (25 + 3 + 14 * len(attack_types)))

        for strategy_name in strategies:
            row = f"{strategy_name:<25} | "
            for attack_type in attack_types:
                if attack_type in all_strategy_summaries[strategy_name]['metrics_by_attack_type']:
                    success_rate = all_strategy_summaries[strategy_name]['metrics_by_attack_type'][attack_type]['success_rate']
                    row += f"{success_rate:>9.2%}   | "
                else:
                    row += f"{'N/A':>9}   | "
            print(row)
    
    print("\n" + "="*80)
    print("分析完成！结果已保存到 analysis/")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()

