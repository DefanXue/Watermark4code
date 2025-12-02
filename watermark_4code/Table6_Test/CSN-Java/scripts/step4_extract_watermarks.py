"""Step 4: 提取水印
从所有攻击版本的代码中提取水印
"""

import json
import sys
import os
import argparse
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # XDF目录
sys.path.insert(0, str(project_root))

from Watermark4code.cli.extract_bits import extract_bit_for_dir
from transformers import AutoTokenizer
from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.keys.directions import derive_directions
from Watermark4code.utils.math import project_embeddings


def process_one_extraction(task):
    """
    并发worker：对单个样本的单个攻击类型提取水印
    
    Args:
        task: (sample_dir, attack_type, attacked_base, model_dir, resume, output_dir)
    
    Returns:
        dict: {'sample_id': str, 'attack_type': str, 'result': dict}
    """
    sample_dir, attack_type, attacked_base, model_dir, resume, output_dir = task
    sample_id = sample_dir.name
    attacked_base = Path(attacked_base)
    output_dir = Path(output_dir)
    
    # 断点续传检查
    if resume:
        result_file = output_dir / f"{attack_type}_{sample_id}.json"
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return {
                'sample_id': sample_id,
                'attack_type': attack_type,
                'result': result,
                'skipped': True
            }
    
    # 确定suspect代码路径
    if attack_type == 'NoAttack':
        suspect_path = sample_dir / "final.java"
    else:
        suspect_path = attacked_base / attack_type / sample_id / "attacked.java"
    
    if not suspect_path.exists():
        return {
            'sample_id': sample_id,
            'attack_type': attack_type,
            'result': {"error": "文件不存在"},
            'skipped': False
        }
    
    try:
        # 读取suspect代码
        with open(suspect_path, 'r', encoding='utf-8') as f:
            suspect_code = f.read()
        
        # 读取median_code
        final_json_path = sample_dir / "final.json"
        with open(final_json_path, 'r', encoding='utf-8') as f:
            final_json = json.load(f)
        
        median_code = final_json.get('median_code', None)
        if median_code is None:
            return {
                'sample_id': sample_id,
                'attack_type': attack_type,
                'result': {"error": "缺少median_code"},
                'skipped': False
            }
        
        # 编码
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_best_model(model_dir, use_quantization=True, device=device)
        codes = [suspect_code, median_code]
        embs = embed_codes(model, tokenizer, codes, max_length=512, batch_size=2, device=device)
        
        # 投影
        d = embs.shape[1]
        W = derive_directions(secret_key="XDF", d=d, k=4)
        projections = project_embeddings(embs, W)
        
        s_suspect = projections[0]
        s_median = projections[1]
        
        # 计算offset
        offset = [float(s_suspect[i] - s_median[i]) for i in range(4)]
        
        # 提取bits
        bits_result = []
        for i in range(4):
            if offset[i] > 0:
                bits_result.append("1")
            elif offset[i] < 0:
                bits_result.append("0")
            else:
                bits_result.append("U")
        
        bits = "".join(bits_result)
        
        # 保存结果
        result = {
            "bits": bits,
            "offset": offset,
            "uncertain_count": bits.count("U")
        }
        
        # 释放GPU内存
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存单个结果
        if resume:
            result_file = output_dir / f"{attack_type}_{sample_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return {
            'sample_id': sample_id,
            'attack_type': attack_type,
            'result': result,
            'skipped': False
        }
    
    except Exception as e:
        return {
            'sample_id': sample_id,
            'attack_type': attack_type,
            'result': {"error": str(e)},
            'skipped': False
        }


def extract_watermarks(embedded_dir: str, attacked_base: str, output_dir: str, model_dir: str, concurrency: int = 1, resume: bool = False):
    """
    提取所有攻击版本的水印
    
    Args:
        embedded_dir: 嵌入结果目录
        attacked_base: 攻击结果基础目录
        output_dir: 提取结果输出目录
        model_dir: CodeT5模型路径
    """
    embedded_dir = Path(embedded_dir)
    attacked_base = Path(attacked_base)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有样本
    sample_dirs = sorted([d for d in embedded_dir.iterdir() if d.is_dir()])
    
    # 所有攻击类型（包括无攻击）
    attack_types = ['NoAttack', 'T1', 'T2', 'T3', 'V25', 'V50', 'V75', 'V100', 'DualCh']
    
    print(f"✅ 找到 {len(sample_dirs)} 个样本")
    print(f"✅ 攻击类型: {len(attack_types)} 种")
    print(f"✅ 并发数: {concurrency}")
    print(f"✅ 断点续传: {'开启' if resume else '关闭'}\n")
    
    # 准备所有任务（样本 × 攻击类型）
    tasks = [
        (sample_dir, attack_type, attacked_base, model_dir, resume, output_dir)
        for attack_type in attack_types
        for sample_dir in sample_dirs
    ]
    
    print(f"✅ 总任务数: {len(tasks)}\n")
    
    # 并发执行
    results = []
    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_one_extraction, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="提取水印") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # 整理结果（按攻击类型分组）
    all_results = {attack_type: {} for attack_type in attack_types}
    for res in results:
        attack_type = res['attack_type']
        sample_id = res['sample_id']
        all_results[attack_type][sample_id] = res['result']
    
    # 保存所有结果
    output_file = output_dir / "extraction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 统计
    skipped_count = sum(1 for r in results if r.get('skipped', False))
    error_count = sum(1 for r in results if 'error' in r['result'])
    success_count = len(results) - error_count - skipped_count
    
    print(f"\n{'='*60}")
    print(f"✓ 水印提取完成")
    print(f"✓ 成功: {success_count}, 错误: {error_count}, 跳过: {skipped_count}")
    print(f"✓ 结果保存到: {output_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='步骤4: 提取水印')
    parser.add_argument('--embedded_dir', type=str, default=None,
                        help='嵌入结果目录 (默认: results/embedded)')
    parser.add_argument('--attacked_base', type=str, default=None,
                        help='攻击结果基础目录 (默认: results/attacked)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='提取结果输出目录 (默认: results/extracted)')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='CodeT5模型路径 (默认: XDF/Watermark4code/best_model)')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='并发数 (默认: 1)')
    parser.add_argument('--resume', action='store_true',
                        help='断点续传 (默认: 关闭)')
    
    args = parser.parse_args()
    
    # 使用绝对路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[3]
    
    # 嵌入结果目录
    embedded_dir = args.embedded_dir if args.embedded_dir else script_dir.parent / "results" / "embedded"
    
    # 攻击结果目录
    attacked_base = args.attacked_base if args.attacked_base else script_dir.parent / "results" / "attacked"
    
    # 提取结果输出目录
    output_dir = args.output_dir if args.output_dir else script_dir.parent / "results" / "extracted"
    
    # CodeT5模型路径（位于 XDF/Watermark4code/best_model）
    model_dir = args.model_dir if args.model_dir else project_root / "Watermark4code" / "best_model"
    
    # 提取水印
    extract_watermarks(
        embedded_dir=embedded_dir,
        attacked_base=attacked_base,
        output_dir=output_dir,
        model_dir=model_dir,
        concurrency=args.concurrency,
        resume=args.resume
    )

