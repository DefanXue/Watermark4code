"""
Step 6: 可视化结果

输入：analysis/comparison_table.json, robustness_curves.json
输出：analysis/visualizations/*.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_robustness_curves(robustness_data, output_file):
    """绘制鲁棒性曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategy_labels = {
        'strategy_5_learned': 'Learned Directions',
        'strategy_6_adaptive': 'Adaptive Generator'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (strategy_name, curve_data) in enumerate(robustness_data['curves'].items()):
        x = [point['x'] for point in curve_data if point['y'] is not None]
        y = [point['y'] for point in curve_data if point['y'] is not None]
        
        ax.plot(x, y, 
                label=strategy_labels.get(strategy_name, strategy_name),
                color=colors[i],
                marker=markers[i],
                linewidth=2,
                markersize=8)
    
    ax.set_xlabel('Attack Ratio (Rename Ratio)', fontsize=12)
    ax.set_ylabel('Extraction Success Rate', fontsize=12)
    ax.set_title('Robustness Comparison: Success Rate vs Attack Intensity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 已保存 {output_file}")


def plot_bit_accuracy_heatmap(comparison_data, output_file):
    """绘制比特准确率热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = comparison_data['strategies']
    attack_ratios = comparison_data['attack_ratios']
    
    strategy_labels = {
        'strategy_5_learned': 'Learned',
        'strategy_6_adaptive': 'Adaptive'
    }
    
    # 构造热力图数据
    data = []
    for strategy in strategies:
        row = []
        for ratio in attack_ratios:
            value = comparison_data['comparison']['avg_bit_accuracy'][strategy].get(ratio)
            row.append(value if value is not None else 0.0)
        data.append(row)
    
    data = np.array(data)
    
    # 绘制热力图
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax.set_xticks(range(len(attack_ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in attack_ratios])
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([strategy_labels.get(s, s) for s in strategies])
    
    # 添加数值标注
    for i in range(len(strategies)):
        for j in range(len(attack_ratios)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xlabel('Attack Ratio', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    ax.set_title('Bit Accuracy Heatmap', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Bit Accuracy', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 已保存 {output_file}")


def plot_success_rate_comparison(comparison_data, output_file):
    """绘制成功率对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = comparison_data['strategies']
    attack_ratios = comparison_data['attack_ratios']
    
    strategy_labels = {
        'strategy_5_learned': 'Learned',
        'strategy_6_adaptive': 'Adaptive'
    }
    
    x = np.arange(len(attack_ratios))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, strategy in enumerate(strategies):
        values = []
        for ratio in attack_ratios:
            value = comparison_data['comparison']['success_rate'][strategy].get(ratio)
            values.append(value if value is not None else 0.0)
        
        offset = (i - len(strategies)/2 + 0.5) * width
        ax.bar(x + offset, values, width, 
               label=strategy_labels.get(strategy, strategy),
               color=colors[i])
    
    ax.set_xlabel('Attack Ratio', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate Comparison Across Attack Intensities', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in attack_ratios])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 已保存 {output_file}")


def plot_offset_preservation(summary_data, output_file):
    """绘制偏移保持率箱线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    strategies = list(summary_data.keys())
    strategy_labels = {
        'strategy_5_learned': 'Learned',
        'strategy_6_adaptive': 'Adaptive'
    }
    
    # 子图1: 符号保持率
    sign_preservation_data = []
    for strategy in strategies:
        values = []
        for ratio_key, metrics in summary_data[strategy]['metrics_by_attack_ratio'].items():
            values.append(metrics['offset_preservation']['avg_sign_preservation_rate'])
        sign_preservation_data.append(values)
    
    ax1.boxplot(sign_preservation_data, labels=[strategy_labels.get(s, s) for s in strategies])
    ax1.set_ylabel('Sign Preservation Rate', fontsize=11)
    ax1.set_title('Offset Sign Preservation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-0.05, 1.05)
    
    # 子图2: 幅度保持率
    magnitude_retention_data = []
    for strategy in strategies:
        values = []
        for ratio_key, metrics in summary_data[strategy]['metrics_by_attack_ratio'].items():
            values.append(metrics['offset_preservation']['avg_magnitude_retention'])
        magnitude_retention_data.append(values)
    
    ax2.boxplot(magnitude_retention_data, labels=[strategy_labels.get(s, s) for s in strategies])
    ax2.set_ylabel('Magnitude Retention', fontsize=11)
    ax2.set_title('Offset Magnitude Retention', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 已保存 {output_file}")


def main():
    print("="*80)
    print("Step 6: 可视化结果")
    print("="*80)
    
    # 创建输出目录
    os.makedirs("analysis/visualizations", exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    with open("analysis/comparison_table.json", 'r') as f:
        comparison_data = json.load(f)
    
    with open("analysis/robustness_curves.json", 'r') as f:
        robustness_data = json.load(f)
    
    with open("analysis/per_strategy_summary.json", 'r') as f:
        summary_data = json.load(f)
    
    # 生成图表
    print("\n生成图表...")
    
    print("  [1/4] 鲁棒性曲线...")
    plot_robustness_curves(
        robustness_data,
        "analysis/visualizations/success_rate_vs_attack.png"
    )
    
    print("  [2/4] 比特准确率热力图...")
    plot_bit_accuracy_heatmap(
        comparison_data,
        "analysis/visualizations/bit_accuracy_heatmap.png"
    )
    
    print("  [3/4] 成功率对比柱状图...")
    plot_success_rate_comparison(
        comparison_data,
        "analysis/visualizations/success_rate_comparison.png"
    )
    
    print("  [4/4] 偏移保持率箱线图...")
    plot_offset_preservation(
        summary_data,
        "analysis/visualizations/offset_preservation_boxplot.png"
    )
    
    print("\n" + "="*80)
    print("可视化完成！图表已保存到 analysis/visualizations/")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()

