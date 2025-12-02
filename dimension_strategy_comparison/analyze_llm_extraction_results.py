"""
分析Step 4c LLM攻击提取结果
生成成功率、比特准确度等统计指标
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_strategy(strategy_name):
    """分析单个策略的LLM攻击提取结果"""
    extraction_dir = f"results/{strategy_name}/extraction_llm"

    if not os.path.exists(extraction_dir):
        print(f"[警告] {strategy_name} 的结果目录不存在: {extraction_dir}")
        return None

    # 收集所有结果
    results_by_attack_type = defaultdict(list)

    for run_id in range(100):
        result_file = f"{extraction_dir}/run_{run_id:04d}.json"

        if not os.path.exists(result_file):
            continue

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                run_results = json.load(f)

            # run_results是一个list，包含rewrite和retrans两个攻击
            for result in run_results:
                attack_type = result['attack_type']
                results_by_attack_type[attack_type].append(result)

        except Exception as e:
            print(f"[错误] 读取 {result_file} 失败: {e}")
            continue

    # 计算统计指标
    print(f"\n{'='*80}")
    print(f"策略: {strategy_name}")
    print(f"{'='*80}")

    for attack_type in ['rewrite', 'retrans']:
        if attack_type not in results_by_attack_type:
            print(f"\n{attack_type}: 无数据")
            continue

        attack_results = results_by_attack_type[attack_type]

        # 计算成功率和比特准确度
        success_count = sum(1 for r in attack_results if r['success'])
        total_count = len(attack_results)
        success_rate = success_count / total_count if total_count > 0 else 0

        bit_accuracies = [r['bit_accuracy'] for r in attack_results]
        avg_bit_accuracy = np.mean(bit_accuracies) if bit_accuracies else 0

        # 按比特准确度分布
        perfect_count = sum(1 for ba in bit_accuracies if ba == 1.0)
        three_quarters_count = sum(1 for ba in bit_accuracies if ba == 0.75)
        half_count = sum(1 for ba in bit_accuracies if ba == 0.5)

        print(f"\n{attack_type.upper()} 攻击:")
        print(f"  总数:          {total_count}")
        print(f"  成功数:        {success_count}")
        print(f"  成功率:        {success_rate:.2%}")
        print(f"  平均比特准确度: {avg_bit_accuracy:.4f}")
        print(f"  ")
        print(f"  比特准确度分布:")
        print(f"    100% (4/4):  {perfect_count:3d} ({perfect_count/total_count:.2%})")
        print(f"    75%  (3/4):  {three_quarters_count:3d} ({three_quarters_count/total_count:.2%})")
        print(f"    50%  (2/4):  {half_count:3d} ({half_count/total_count:.2%})")

        # 按维度分析成功率
        dimension_success = [0, 0, 0, 0]
        for r in attack_results:
            true_bits = r['true_bits']
            extracted_bits = r['extracted_bits']
            for i in range(4):
                if extracted_bits[i] == true_bits[i]:
                    dimension_success[i] += 1

        print(f"  ")
        print(f"  按维度成功率:")
        for i in range(4):
            dim_success_rate = dimension_success[i] / total_count if total_count > 0 else 0
            print(f"    维度{i}: {dim_success_rate:.2%} ({dimension_success[i]}/{total_count})")


def compare_strategies():
    """比较所有策略的LLM攻击效果"""
    print(f"\n{'='*80}")
    print("LLM攻击提取结果对比")
    print(f"{'='*80}\n")

    strategies = ["strategy_5_learned", "strategy_6_adaptive"]

    summary_data = []

    for strategy_name in strategies:
        extraction_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction_llm"

        if not os.path.exists(extraction_dir):
            continue

        # 统计Rewrite和Retrans的成功率
        rewrite_success = 0
        rewrite_total = 0
        retrans_success = 0
        retrans_total = 0

        for run_id in range(100):
            result_file = f"{extraction_dir}/run_{run_id:04d}.json"

            if not os.path.exists(result_file):
                continue

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    run_results = json.load(f)

                for result in run_results:
                    if result['attack_type'] == 'rewrite':
                        rewrite_total += 1
                        if result['success']:
                            rewrite_success += 1
                    elif result['attack_type'] == 'retrans':
                        retrans_total += 1
                        if result['success']:
                            retrans_success += 1
            except:
                continue

        rewrite_rate = rewrite_success / rewrite_total if rewrite_total > 0 else 0
        retrans_rate = retrans_success / retrans_total if retrans_total > 0 else 0

        summary_data.append({
            'strategy': strategy_name,
            'rewrite_success': rewrite_success,
            'rewrite_total': rewrite_total,
            'rewrite_rate': rewrite_rate,
            'retrans_success': retrans_success,
            'retrans_total': retrans_total,
            'retrans_rate': retrans_rate
        })

    # 打印对比表
    print(f"{'策略':<20} {'Rewrite成功率':<15} {'Retrans成功率':<15}")
    print("-" * 50)
    for data in summary_data:
        print(f"{data['strategy']:<20} {data['rewrite_rate']:.2%}             {data['retrans_rate']:.2%}")


def main():
    print("="*80)
    print("Step 4c: LLM攻击提取结果分析")
    print("="*80)

    project_root = Path(__file__).parent
    os.chdir(project_root)

    # 分析每个策略
    strategies = ["strategy_5_learned", "strategy_6_adaptive"]

    for strategy_name in strategies:
        analyze_strategy(strategy_name)

    # 对比所有策略
    compare_strategies()

    print(f"\n{'='*80}")
    print("分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
