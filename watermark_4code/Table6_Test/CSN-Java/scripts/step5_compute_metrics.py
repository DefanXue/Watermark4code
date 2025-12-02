"""Step 5: 计算指标
计算BitAcc, MsgAcc, MRR, CodeBLEU
"""

import json
import sys
from pathlib import Path
import numpy as np

# 添加Watermark4code到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def compute_bitacc_msgacc(extraction_results_path: str, test_samples_path: str, attacked_base: str):
    """
    计算BitAcc和MsgAcc
    
    Args:
        extraction_results_path: 提取结果文件路径
        test_samples_path: 测试样本文件路径
        attacked_base: 攻击结果目录（用于读取attack_metadata.json）
    
    Returns:
        dict: 每种攻击类型的准确率
    """
    # 读取提取结果
    with open(extraction_results_path, 'r', encoding='utf-8') as f:
        extraction_results = json.load(f)
    
    # 读取ground truth
    with open(test_samples_path, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f]
    
    # 构建ground truth映射
    gt_map = {sample['id']: sample['watermark'] for sample in test_samples}
    
    attacked_base = Path(attacked_base)
    
    # 计算准确率
    accuracy_results = {}
    
    for attack_type, attack_results in extraction_results.items():
        correct_bits = 0
        total_bits = 0
        correct_messages = 0
        total_messages = 0
        skipped_samples = []
        
        for sample_id, result in attack_results.items():
            if 'error' in result:
                continue
            
            # 检查该样本在该攻击类型下是否成功
            should_skip = False
            
            if attack_type != 'NoAttack':
                metadata_path = attacked_base / attack_type / sample_id / "attack_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 判断攻击是否失败
                    if attack_type in ['T1', 'T2', 'T3']:
                        # T攻击：检查transforms是否为空
                        if not metadata.get('transforms', []):
                            should_skip = True
                    elif attack_type in ['V25', 'V50', 'V75', 'V100']:
                        # V攻击：检查rename_mapping是否为空
                        if not metadata.get('rename_mapping', {}):
                            should_skip = True
                    elif attack_type == 'DualCh':
                        # DualCh：两者都为空才算失败
                        if not metadata.get('transforms', []) and not metadata.get('rename_mapping', {}):
                            should_skip = True
            
            if should_skip:
                skipped_samples.append(sample_id)
                continue
            
            # 获取提取的bits
            extracted_bits = result['bits']
            
            # 获取ground truth
            gt_bits = gt_map[sample_id]
            gt_str = ''.join(str(b) for b in gt_bits)
            
            # 计算BitAcc
            for i in range(len(gt_str)):
                if i < len(extracted_bits) and extracted_bits[i] != 'U':
                    total_bits += 1
                    if extracted_bits[i] == gt_str[i]:
                        correct_bits += 1
            
            # 计算MsgAcc
            total_messages += 1
            if extracted_bits == gt_str:
                correct_messages += 1
        
        bitacc = correct_bits / total_bits if total_bits > 0 else 0.0
        msgacc = correct_messages / total_messages if total_messages > 0 else 0.0
        
        accuracy_results[attack_type] = {
            'bitacc': bitacc,
            'msgacc': msgacc,
            'correct_bits': correct_bits,
            'total_bits': total_bits,
            'correct_messages': correct_messages,
            'total_messages': total_messages,
            'skipped_samples': skipped_samples
        }
    
    return accuracy_results


def compute_codebleu(embedded_dir: str, attacked_base: str):
    """
    计算CodeBLEU（使用SrcMarker的实现）
    
    Args:
        embedded_dir: 嵌入结果目录
        attacked_base: 攻击结果目录
    
    Returns:
        dict: 每种攻击类型的CodeBLEU分数
    """
    # 复制SrcMarker的CodeBLEU实现
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent.parent.parent.parent / "SrcMarker-main"))
    
    try:
        from metrics.calc_code_bleu import evaluate_per_example
    except ImportError:
        print("警告: 无法导入CodeBLEU模块，将返回空结果")
        return {}
    
    embedded_dir = Path(embedded_dir)
    attacked_base = Path(attacked_base)
    
    # 获取所有样本
    sample_dirs = sorted([d for d in embedded_dir.iterdir() if d.is_dir()])
    
    # 攻击类型（不包括NoAttack）
    attack_types = ['T1', 'T2', 'T3', 'V25', 'V50', 'V75', 'V100', 'DualCh']
    
    codebleu_results = {}
    
    for attack_type in attack_types:
        print(f"计算CodeBLEU: {attack_type}")
        scores = []
        
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            
            # 读取原始代码（嵌入前）
            original_path = sample_dir / "original.java"
            if not original_path.exists():
                continue
            
            with open(original_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # 读取攻击后的代码
            attacked_path = attacked_base / attack_type / sample_id / "attacked.java"
            if not attacked_path.exists():
                continue
            
            with open(attacked_path, 'r', encoding='utf-8') as f:
                attacked_code = f.read()
            
            try:
                # 计算CodeBLEU
                result = evaluate_per_example(
                    reference=original_code,
                    hypothesis=attacked_code,
                    lang='java',
                    params="0.25,0.25,0.25,0.25"
                )
                scores.append(result['codebleu'])
            except Exception as e:
                print(f"  警告: {sample_id} CodeBLEU计算失败: {e}")
        
        # 计算平均值
        avg_codebleu = np.mean(scores) if scores else 0.0
        codebleu_results[attack_type] = {
            'codebleu': avg_codebleu,
            'count': len(scores)
        }
    
    return codebleu_results


def compute_mrr(embedded_dir: str, attacked_base: str, model_name: str = "microsoft/codebert-base"):
    """
    计算MRR（Mean Reciprocal Rank）- 代码搜索任务
    
    注意: CSN-Java filtered code没有docstring，这里简化实现
    返回占位值，实际需要完整的代码搜索实现
    
    Args:
        embedded_dir: 嵌入结果目录
        attacked_base: 攻击结果目录
        model_name: 使用的模型名称
    
    Returns:
        dict: 每种攻击类型的MRR分数
    """
    print("警告: MRR计算需要完整的代码搜索实现和docstring")
    print("      由于CSN-Java filtered code缺少docstring，此处返回占位值")
    
    # 攻击类型
    attack_types = ['NoAttack', 'T1', 'T2', 'T3', 'V25', 'V50', 'V75', 'V100', 'DualCh']
    
    # 返回占位值（实际实验中需要完整实现）
    mrr_results = {}
    for attack_type in attack_types:
        mrr_results[attack_type] = {
            'mrr': 0.0,  # 占位值
            'note': 'MRR需要docstring和完整代码搜索实现'
        }
    
    return mrr_results


def compute_all_metrics(
    extraction_results_path: str,
    test_samples_path: str,
    embedded_dir: str,
    attacked_base: str,
    output_path: str
):
    """
    计算所有指标并保存
    
    Args:
        extraction_results_path: 提取结果路径
        test_samples_path: 测试样本路径
        embedded_dir: 嵌入结果目录
        attacked_base: 攻击结果目录
        output_path: 输出文件路径
    """
    print("="*60)
    print("计算指标")
    print("="*60)
    
    # 1. BitAcc和MsgAcc
    print("\n1. 计算BitAcc和MsgAcc...")
    accuracy_results = compute_bitacc_msgacc(extraction_results_path, test_samples_path, attacked_base)
    
    # 2. CodeBLEU
    print("\n2. 计算CodeBLEU...")
    codebleu_results = compute_codebleu(embedded_dir, attacked_base)
    
    # 3. MRR
    print("\n3. 计算MRR...")
    mrr_results = compute_mrr(embedded_dir, attacked_base)
    
    # 合并结果
    all_metrics = {}
    for attack_type in accuracy_results:
        all_metrics[attack_type] = {
            'bitacc': accuracy_results[attack_type]['bitacc'],
            'msgacc': accuracy_results[attack_type]['msgacc'],
            'codebleu': codebleu_results.get(attack_type, {}).get('codebleu', 0.0),
            'mrr': mrr_results.get(attack_type, {}).get('mrr', 0.0),
            'details': {
                'accuracy': accuracy_results[attack_type],
                'codebleu': codebleu_results.get(attack_type, {}),
                'mrr': mrr_results.get(attack_type, {})
            }
        }
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ 指标计算完成")
    print(f"✓ 结果保存到: {output_path}")
    print(f"{'='*60}\n")
    
    # 打印摘要
    print("\n指标摘要:")
    print(f"{'Attack':<10} {'BitAcc':<10} {'MsgAcc':<10} {'CB':<10} {'MRR':<10} {'Valid':<10}")
    print("-" * 60)
    for attack_type, metrics in all_metrics.items():
        valid_samples = metrics['details']['accuracy']['total_messages']
        skipped = len(metrics['details']['accuracy'].get('skipped_samples', []))
        total = valid_samples + skipped
        print(f"{attack_type:<10} "
              f"{metrics['bitacc']*100:>6.2f}%   "
              f"{metrics['msgacc']*100:>6.2f}%   "
              f"{metrics['codebleu']:>6.4f}   "
              f"{metrics['mrr']:>6.4f}   "
              f"{valid_samples}/{total}")


if __name__ == '__main__':
    # 使用绝对路径
    script_dir = Path(__file__).parent
    
    # 提取结果路径
    extraction_results_path = script_dir.parent / "results" / "extracted" / "extraction_results.json"
    
    # 测试样本路径
    test_samples_path = script_dir.parent / "data" / "test_samples.jsonl"
    
    # 嵌入结果目录
    embedded_dir = script_dir.parent / "results" / "embedded"
    
    # 攻击结果目录
    attacked_base = script_dir.parent / "results" / "attacked"
    
    # 输出路径
    output_path = script_dir.parent / "analysis" / "metrics_results.json"
    
    # 计算所有指标
    compute_all_metrics(
        extraction_results_path,
        test_samples_path,
        embedded_dir,
        attacked_base,
        output_path
    )

