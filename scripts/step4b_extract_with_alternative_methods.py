"""
Step 4b: 测试替代提取方法
  - Method 1 (Baseline): 与簇中心s0比较 + 阈值判断（Step4已实现，此处不重复）
  - Method 2 (Median): 与median_code被攻击后的相对位置
  - Method 3 (Extreme): 与极值代码被攻击后的距离判断
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.utils.math import project_embeddings
from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer


def apply_rename_attack(code, ratio, seed):
    """应用重命名攻击"""
    try:
        renamer = JavaVariableRenamer(code)
        attacked_code = renamer.apply_renames(rename_ratio=ratio, seed=seed)
        return attacked_code
    except Exception:
        return code


def extract_median(attacked_watermarked, median_cluster_center, directions, K, num_workers, batch_size_for_parallel, model, tokenizer, device):
    """方法2：可疑代码的簇中心 vs median的簇中心（不使用阈值）"""
    from Watermark4code.injection.plan import build_candidates_test_like
    import torch
    
    # 生成可疑代码的变体并计算簇中心
    try:
        suspicious_cands = build_candidates_test_like(
            attacked_watermarked,
            max(1, K),
            num_workers=num_workers,
            batch_size_for_parallel=batch_size_for_parallel,
        )
        suspicious_cands = [c for c in suspicious_cands if isinstance(c, str) and c.strip() and c.strip() != attacked_watermarked.strip()]
        
        if suspicious_cands:
            v_suspicious_cands = embed_codes(model, tokenizer, suspicious_cands, device=device)
            s_suspicious_cands = project_embeddings(v_suspicious_cands, directions)
            s_suspicious_cluster = np.array([np.median(s_suspicious_cands[:, i]) for i in range(4)])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # 无候选时使用可疑代码自身的投影
            v_suspicious = embed_codes(model, tokenizer, [attacked_watermarked], device=device)
            s_suspicious_cluster = project_embeddings(v_suspicious, directions)[0]
    except Exception:
        # 出错时使用可疑代码自身的投影
        v_suspicious = embed_codes(model, tokenizer, [attacked_watermarked], device=device)
        s_suspicious_cluster = project_embeddings(v_suspicious, directions)[0]
    
    # 与median簇中心比较
    median_cluster_center = np.array(median_cluster_center)
    offset = s_suspicious_cluster - median_cluster_center
    
    extracted_bits = []
    for i in range(4):
        # 只看相对位置，不使用阈值
        extracted_bits.append(1 if offset[i] > 0 else 0)
    
    return extracted_bits, s_suspicious_cluster.tolist(), median_cluster_center.tolist()


def extract_extreme(attacked_watermarked, extreme_cluster_centers, directions, K, num_workers, batch_size_for_parallel, model, tokenizer, device):
    """方法3：可疑代码的簇中心 vs 极值簇中心距离判断（不使用阈值）"""
    from Watermark4code.injection.plan import build_candidates_test_like
    import torch
    
    # 生成可疑代码的变体并计算簇中心
    try:
        suspicious_cands = build_candidates_test_like(
            attacked_watermarked,
            max(1, K),
            num_workers=num_workers,
            batch_size_for_parallel=batch_size_for_parallel,
        )
        suspicious_cands = [c for c in suspicious_cands if isinstance(c, str) and c.strip() and c.strip() != attacked_watermarked.strip()]
        
        if suspicious_cands:
            v_suspicious_cands = embed_codes(model, tokenizer, suspicious_cands, device=device)
            s_suspicious_cands = project_embeddings(v_suspicious_cands, directions)
            s_suspicious_cluster = np.array([np.median(s_suspicious_cands[:, i]) for i in range(4)])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # 无候选时使用可疑代码自身的投影
            v_suspicious = embed_codes(model, tokenizer, [attacked_watermarked], device=device)
            s_suspicious_cluster = project_embeddings(v_suspicious, directions)[0]
    except Exception:
        # 出错时使用可疑代码自身的投影
        v_suspicious = embed_codes(model, tokenizer, [attacked_watermarked], device=device)
        s_suspicious_cluster = project_embeddings(v_suspicious, directions)[0]
    
    # 与每个维度的极值簇中心比较距离
    extracted_bits = []
    for i in range(4):
        s_pos_cluster = extreme_cluster_centers[str(i)]['pos_cluster']
        s_neg_cluster = extreme_cluster_centers[str(i)]['neg_cluster']
        
        dist_to_pos = abs(s_suspicious_cluster[i] - s_pos_cluster[i])
        dist_to_neg = abs(s_suspicious_cluster[i] - s_neg_cluster[i])
        
        # 距离判断：靠近哪个极值
        extracted_bits.append(1 if dist_to_pos < dist_to_neg else 0)
    
    return extracted_bits, s_suspicious_cluster.tolist()


def process_one_run(task):
    """处理单个run的所有攻击强度"""
    run_id, strategy_name, base_config = task
    
    embedding_dir = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}"
    
    # 加载嵌入结果
    try:
        with open(f"{embedding_dir}/watermarked.java", 'r', encoding='utf-8') as f:
            watermarked_code = f.read()
        
        with open(f"{embedding_dir}/final.json", 'r', encoding='utf-8') as f:
            embed_result = json.load(f)
        
        with open(f"{embedding_dir}/selected_dimensions.json", 'r', encoding='utf-8') as f:
            dim_data = json.load(f)
        
        median_cluster_center = embed_result.get('median_cluster_center')
        extreme_cluster_centers = embed_result.get('extreme_cluster_centers')
        
        if not median_cluster_center or not extreme_cluster_centers:
            return {"run_id": run_id, "success": False, "error": "缺少median_cluster_center或extreme_cluster_centers", "count": 0}
        
        true_bits = base_config['embedding']['bits']
        directions = np.array(dim_data['directions'])
        
        # 加载模型
        model, tokenizer = load_best_model(base_config['model_dir'])
        device = next(model.parameters()).device
        
        results = []
        
        # 使用与step3一致的簇中心计算变体数量
        K = base_config['embedding'].get('cluster_variants', 100)
        num_workers = base_config['embedding']['num_workers']
        batch_size_for_parallel = base_config['embedding']['batch_size_for_parallel']
        
        for ratio in base_config['extraction']['attack_ratios']:
            for seed in range(base_config['extraction']['num_seeds_per_ratio']):
                # 攻击水印代码
                attacked_watermarked = apply_rename_attack(watermarked_code, ratio, seed)
                
                # === 方法2：Median（簇中心 vs 簇中心）===
                bits_median, s_suspicious_m, s_median_ref = extract_median(
                    attacked_watermarked, median_cluster_center, directions, 
                    K, num_workers, batch_size_for_parallel, 
                    model, tokenizer, device
                )
                success_median = (bits_median == true_bits)
                bitacc_median = sum(b1 == b2 for b1, b2 in zip(bits_median, true_bits)) / 4
                
                # === 方法3：Extreme（簇中心 vs 极值簇中心）===
                bits_extreme, s_suspicious_e = extract_extreme(
                    attacked_watermarked, extreme_cluster_centers, directions, 
                    K, num_workers, batch_size_for_parallel, 
                    model, tokenizer, device
                )
                success_extreme = (bits_extreme == true_bits)
                bitacc_extreme = sum(b1 == b2 for b1, b2 in zip(bits_extreme, true_bits)) / 4
                
                # 保存结果
                result = {
                    'run_id': run_id,
                    'attack_ratio': float(ratio),
                    'seed': seed,
                    'true_bits': true_bits,
                    'method_median': {
                        'extracted_bits': bits_median,
                        'success': bool(success_median),
                        'bit_accuracy': float(bitacc_median)
                    },
                    'method_extreme': {
                        'extracted_bits': bits_extreme,
                        'success': bool(success_extreme),
                        'bit_accuracy': float(bitacc_extreme)
                    }
                }
                
                # 保存单个结果
                output_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction_alternative/attack_{ratio:.2f}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/run_{run_id:04d}_seed_{seed:03d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                results.append(result)
        
        return {"run_id": run_id, "success": True, "count": len(results)}
    
    except Exception as e:
        return {"run_id": run_id, "success": False, "error": str(e), "count": 0}


def process_strategy(strategy_name, base_config, concurrency, resume=False):
    """处理单个策略"""
    print(f"\n处理策略: {strategy_name}")
    
    tasks = []
    skipped_count = 0
    
    for run_id in range(base_config['num_test_codes']):
        if resume:
            # 检查是否所有攻击强度都已完成
            all_exist = True
            for ratio in base_config['extraction']['attack_ratios']:
                for seed in range(base_config['extraction']['num_seeds_per_ratio']):
                    output_file = f"dimension_strategy_comparison/results/{strategy_name}/extraction_alternative/attack_{ratio:.2f}/run_{run_id:04d}_seed_{seed:03d}.json"
                    if not os.path.exists(output_file):
                        all_exist = False
                        break
                if not all_exist:
                    break
            
            if all_exist:
                skipped_count += 1
                continue
        
        tasks.append((run_id, strategy_name, base_config))
    
    if resume and skipped_count > 0:
        print(f"  跳过已完成: {skipped_count} 个")
    
    if not tasks:
        print(f"  所有任务已完成，无需处理")
        return
    
    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_run, task) for task in tasks]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  提取进度", ncols=80):
                result = future.result()
                results.append(result)
                if not result['success']:
                    print(f"\n[错误] run_{result['run_id']:04d} 提取失败: {result.get('error', 'Unknown')}")
    else:
        print(f"  使用串行模式")
        results = []
        for task in tqdm(tasks, desc=f"  提取进度", ncols=80):
            result = process_one_run(task)
            results.append(result)
            if not result['success']:
                print(f"\n[错误] run_{result['run_id']:04d} 提取失败: {result.get('error', 'Unknown')}")
    
    success_count = sum(1 for r in results if r['success'])
    total_extractions = sum(r['count'] for r in results)
    print(f"  完成: {success_count}/{len(results)} 成功, 共 {total_extractions} 次提取")


def main():
    parser = argparse.ArgumentParser(description="Step 4b: 测试替代提取方法")
    parser.add_argument('--concurrency', type=int, default=5, help='并发进程数（默认=5）')
    parser.add_argument('--resume', action='store_true', help='断点继续')
    args = parser.parse_args()
    
    print("="*80)
    print(f"Step 4b: 测试替代提取方法 (concurrency={args.concurrency}, resume={args.resume})")
    print("="*80)
    
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    config_path = project_root / "dimension_strategy_comparison" / "configs" / "base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    strategies = [
        "strategy_1_random",
        "strategy_2_top4",
        "strategy_3_balanced",
        "strategy_4_triple_order",
        "strategy_5_learned",
    ]
    
    for strategy_name in strategies:
        process_strategy(strategy_name, base_config, args.concurrency, args.resume)
    
    print("\n" + "="*80)
    print("完成！替代提取方法结果已保存到 results/strategy_X/extraction_alternative/")
    print("="*80)


if __name__ == '__main__':
    main()



