"""Step 6: 生成Table 6对比表
生成SrcMarker论文Table 6格式的对比表
"""

import json
from pathlib import Path


# SrcMarker论文中Table 6的值（CSN-Java, SrcMarkerTE）
SRCMARKER_PAPER_VALUES = {
    "No Attack": {"BitAcc": 97.26, "MsgAcc": 92.74, "MRR": 0.8137, "CB": None},
    "T1": {"BitAcc": 83.50, "MsgAcc": None, "MRR": 0.8168, "CB": 0.8589},
    "T2": {"BitAcc": 77.04, "MsgAcc": None, "MRR": 0.8159, "CB": 0.7907},
    "T3": {"BitAcc": 75.07, "MsgAcc": None, "MRR": 0.8189, "CB": 0.7678},
    "V25": {"BitAcc": 84.08, "MsgAcc": None, "MRR": 0.8145, "CB": 0.9150},
    "V50": {"BitAcc": 79.89, "MsgAcc": None, "MRR": 0.8019, "CB": 0.8853},
    "V75": {"BitAcc": 73.71, "MsgAcc": None, "MRR": 0.7910, "CB": 0.8464},
    "V100": {"BitAcc": 62.68, "MsgAcc": None, "MRR": 0.7845, "CB": 0.7855},
    "DualCh": {"BitAcc": 67.76, "MsgAcc": None, "MRR": 0.7984, "CB": 0.7674},
}


# 攻击类型映射（代码中的键 -> 显示名称）
ATTACK_NAME_MAP = {
    "NoAttack": "No Attack",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3",
    "V25": "V25",
    "V50": "V50",
    "V75": "V75",
    "V100": "V100",
    "DualCh": "DualCh"
}


def generate_table6(metrics_path: str, output_json: str, output_md: str):
    """
    生成Table 6对比表
    
    Args:
        metrics_path: 指标结果文件路径
        output_json: 输出JSON文件路径
        output_md: 输出Markdown文件路径
    """
    # 读取实验指标
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics_results = json.load(f)
    
    # 构建对比数据
    comparison_data = {
        "dataset": "CSN-Java",
        "num_samples": 50,
        "num_bits": 4,
        "watermark": [1, 1, 0, 0],
        "results": {}
    }
    
    for code_key, display_name in ATTACK_NAME_MAP.items():
        if code_key not in metrics_results:
            continue
        
        our_metrics = metrics_results[code_key]
        paper_metrics = SRCMARKER_PAPER_VALUES.get(display_name, {})
        
        comparison_data["results"][display_name] = {
            "SrcMarker (论文)": {
                "BitAcc": paper_metrics.get("BitAcc"),
                "MsgAcc": paper_metrics.get("MsgAcc"),
                "MRR": paper_metrics.get("MRR"),
                "CodeBLEU": paper_metrics.get("CB")
            },
            "Watermark4code (实验)": {
                "BitAcc": our_metrics['bitacc'] * 100,  # 转换为百分比
                "MsgAcc": our_metrics['msgacc'] * 100,
                "MRR": our_metrics['mrr'],
                "CodeBLEU": our_metrics['codebleu']
            }
        }
    
    # 保存JSON
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown表格
    md_content = generate_markdown_table(comparison_data)
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n{'='*60}")
    print(f"✓ Table 6生成完成")
    print(f"✓ JSON: {output_json}")
    print(f"✓ Markdown: {output_md}")
    print(f"{'='*60}\n")
    
    # 打印到控制台
    print(md_content)


def generate_markdown_table(data: dict) -> str:
    """
    生成Markdown格式的对比表
    
    Args:
        data: 对比数据
    
    Returns:
        Markdown格式的表格字符串
    """
    md = []
    
    # 标题
    md.append(f"# Table 6: Robustness Evaluation on {data['dataset']}")
    md.append(f"\n**Dataset**: {data['dataset']}")
    md.append(f"**Samples**: {data['num_samples']}")
    md.append(f"**Watermark**: {data['num_bits']}-bit {data['watermark']}")
    md.append("\n## Comparison Table\n")
    
    # 表头
    md.append("| Attack Type | Method | BitAcc (%) | MsgAcc (%) | MRR | CodeBLEU |")
    md.append("|-------------|--------|------------|------------|-----|----------|")
    
    # 攻击类型顺序
    attack_order = ["No Attack", "T1", "T2", "T3", "V25", "V50", "V75", "V100", "DualCh"]
    
    # 数据行
    for attack_type in attack_order:
        if attack_type not in data['results']:
            continue
        
        results = data['results'][attack_type]
        
        # SrcMarker行
        paper = results['SrcMarker (论文)']
        md.append(
            f"| {attack_type:<11} | SrcMarker (论文) | "
            f"{format_value(paper['BitAcc'])} | "
            f"{format_value(paper['MsgAcc'])} | "
            f"{format_float(paper['MRR'])} | "
            f"{format_float(paper['CodeBLEU'])} |"
        )
        
        # Watermark4code行
        ours = results['Watermark4code (实验)']
        md.append(
            f"| {' '*11} | Watermark4code | "
            f"{format_value(ours['BitAcc'])} | "
            f"{format_value(ours['MsgAcc'])} | "
            f"{format_float(ours['MRR'])} | "
            f"{format_float(ours['CodeBLEU'])} |"
        )
    
    # 说明
    md.append("\n## Notes\n")
    md.append("- **BitAcc**: Bitwise Accuracy (位准确率)")
    md.append("- **MsgAcc**: Message Accuracy (消息准确率，4-bit完全匹配)")
    md.append("- **MRR**: Mean Reciprocal Rank (代码搜索任务的平均倒数排名)")
    md.append("- **CodeBLEU**: Code similarity score (代码相似度)")
    md.append("- **T@N**: Code transformation attack with N transforms")
    md.append("- **V@X%**: Variable renaming attack with X% variables renamed")
    md.append("- **DualCh**: Dual-channel attack (50% variable renaming + 2 transforms)")
    md.append("\n---")
    md.append("\n*SrcMarker论文值来自IEEE S&P 2024论文Table 6 (CSN-Java, SrcMarkerTE)*")
    
    return '\n'.join(md)


def format_value(value) -> str:
    """格式化百分比值"""
    if value is None:
        return "    -    "
    return f"{value:>6.2f}%  "


def format_float(value) -> str:
    """格式化浮点数值"""
    if value is None:
        return "   -   "
    if value == 0.0:
        return "   -   "  # MRR占位值显示为-
    return f"{value:>6.4f} "


if __name__ == '__main__':
    # 指标结果路径
    metrics_path = Path("../analysis/metrics_results.json")
    
    # 输出路径
    output_json = Path("../analysis/table6_comparison.json")
    output_md = Path("../analysis/table6_comparison.md")
    
    # 生成Table 6
    generate_table6(metrics_path, output_json, output_md)










