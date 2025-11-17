"""
Step 5b: 对比三种提取方法的性能
  - Baseline (Step4): 与s0比较 + 阈值
  - Median (Step4b): 与median_code被攻击后比较
  - Extreme (Step4b): 与极值被攻击后距离判断
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_baseline_results(strategy_name, attack_ratios):
    """加载Baseline方法的结果（来自Step4）"""
    results_by_ratio = defaultdict(list)
    
    for ratio in attack_ratios:
        result_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction/attack_{ratio:.2f}"
        
        if not os.path.exists(result_dir):
            continue
        
        for filename in os.listdir(result_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(result_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results_by_ratio[ratio].append({
                        'success': data['success'],
                        'bit_accuracy': data['bit_accuracy']
                    })
    
    return results_by_ratio


def load_alternative_results(strategy_name, attack_ratios):
    """加载Median和Extreme方法的结果（来自Step4b）"""
    median_by_ratio = defaultdict(list)
    extreme_by_ratio = defaultdict(list)
    
    for ratio in attack_ratios:
        result_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction_alternative/attack_{ratio:.2f}"
        
        if not os.path.exists(result_dir):
            continue
        
        for filename in os.listdir(result_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(result_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    median_by_ratio[ratio].append({
                        'success': data['method_median']['success'],
                        'bit_accuracy': data['method_median']['bit_accuracy']
                    })
                    extreme_by_ratio[ratio].append({
                        'success': data['method_extreme']['success'],
                        'bit_accuracy': data['method_extreme']['bit_accuracy']
                    })
    
    return median_by_ratio, extreme_by_ratio


def analyze_all_methods():
    """分析所有提取方法"""
    # 读取配置
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    config_path = project_root / "dimension_strategy_comparison" / "configs" / "base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    strategies = ["strategy_1_random", "strategy_2_top4", "strategy_3_balanced", "strategy_4_triple_order", "strategy_5_learned"]
    attack_ratios = base_config['extraction']['attack_ratios']
    
    all_results = {}
    
    for strategy in strategies:
        baseline_results = load_baseline_results(strategy, attack_ratios)
        median_results, extreme_results = load_alternative_results(strategy, attack_ratios)
        
        all_results[strategy] = {}
        
        for ratio in attack_ratios:
            baseline_data = baseline_results.get(ratio, [])
            median_data = median_results.get(ratio, [])
            extreme_data = extreme_results.get(ratio, [])
            
            all_results[strategy][f"attack_{ratio:.2f}"] = {
                'baseline': {
                    'msgacc': float(np.mean([r['success'] for r in baseline_data])) if baseline_data else 0.0,
                    'bitacc': float(np.mean([r['bit_accuracy'] for r in baseline_data])) if baseline_data else 0.0,
                    'num_samples': len(baseline_data)
                },
                'median': {
                    'msgacc': float(np.mean([r['success'] for r in median_data])) if median_data else 0.0,
                    'bitacc': float(np.mean([r['bit_accuracy'] for r in median_data])) if median_data else 0.0,
                    'num_samples': len(median_data)
                },
                'extreme': {
                    'msgacc': float(np.mean([r['success'] for r in extreme_data])) if extreme_data else 0.0,
                    'bitacc': float(np.mean([r['bit_accuracy'] for r in extreme_data])) if extreme_data else 0.0,
                    'num_samples': len(extreme_data)
                }
            }
    
    # 保存结果
    os.makedirs("dimension_strategy_comparison/analysis", exist_ok=True)
    with open("dimension_strategy_comparison/analysis/extraction_methods_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印对比表格
    for strategy in strategies:
        print("\n" + "="*120)
        print(f"提取方法对比 - {strategy}")
        print("="*120)
        print(f"{'Attack':<8} | {'Baseline MsgAcc':<18} | {'Median MsgAcc':<18} | {'Extreme MsgAcc':<18} | {'Best Method':<15}")
        print("-"*120)
        
        for ratio in attack_ratios:
            data = all_results[strategy][f"attack_{ratio:.2f}"]
            baseline_msg = data['baseline']['msgacc']
            median_msg = data['median']['msgacc']
            extreme_msg = data['extreme']['msgacc']
            
            best_method = max(
                [('Baseline', baseline_msg), ('Median', median_msg), ('Extreme', extreme_msg)],
                key=lambda x: x[1]
            )[0]
            
            print(f"{ratio:<8.2f} | {baseline_msg:<18.2%} | {median_msg:<18.2%} | {extreme_msg:<18.2%} | {best_method:<15}")
        
        print("="*120)
    
    print("\n结果已保存到: dimension_strategy_comparison/analysis/extraction_methods_comparison.json")


if __name__ == '__main__':
    analyze_all_methods()












