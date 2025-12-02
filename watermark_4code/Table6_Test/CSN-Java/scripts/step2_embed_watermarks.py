"""Step 2: 嵌入水印
使用Watermark4code主方法嵌入水印（Random维度 + Median提取）
"""

import json
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加Watermark4code到路径（将项目根目录 XDF 加入 sys.path）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(project_root))

from Watermark4code.injection.greedy import select_and_inject
from Watermark4code.injection.plan import compute_required_delta_per_anchor


def process_one_sample(task):
    """
    处理单个测试样本（用于多进程并行）
    
    Args:
        task: (sample_id, sample_code, sample_watermark, args_dict)
    
    Returns:
        dict: {'sample_id': str, 'success': bool, 'error': str or None}
    """
    import os
    import json
    from pathlib import Path
    from Watermark4code.injection.greedy import select_and_inject
    from Watermark4code.injection.plan import compute_required_delta_per_anchor
    
    sample_id, sample_code, sample_watermark, args_dict = task
    
    # 创建输出目录
    output_base = Path(args_dict['output_base'])
    sample_dir = output_base / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 断点续传：检查final.json是否存在
    if args_dict.get('resume', False):
        final_json_path = sample_dir / "final.json"
        if final_json_path.exists():
            return {
                'sample_id': sample_id,
                'success': True,
                'error': None,
                'skipped': True
            }
    
    # 保存原始代码
    original_path = sample_dir / "original.java"
    with open(original_path, 'w', encoding='utf-8') as f:
        f.write(sample_code)
    
    try:
        # 1. 计算阈值和median_code
        anchor_info = compute_required_delta_per_anchor(
            model_dir=args_dict['model_dir'],
            anchor_code=sample_code,
            bits=sample_watermark,
            secret_key=args_dict['secret_key'],
            K=args_dict['K'],
            quantile=args_dict['quantile'],
            quantized=True,
            max_length=512,
            batch_size=64,
            num_workers=args_dict['num_workers'],
            batch_size_for_parallel=args_dict['batch_size_for_parallel'],
        )
        
        # 2. 嵌入水印
        result = select_and_inject(
            model_dir=args_dict['model_dir'],
            anchor_code=sample_code,
            bits=sample_watermark,
            required_delta=anchor_info,
            secret_key=args_dict['secret_key'],
            K=args_dict['K'],
            max_iters=args_dict['max_iters'],
            num_workers=args_dict['num_workers'],
            batch_size_for_parallel=args_dict['batch_size_for_parallel'],
            save_dir=str(sample_dir),
        )
        
        # 3. 保存水印代码
        watermarked_path = sample_dir / "final.java"
        with open(watermarked_path, 'w', encoding='utf-8') as f:
            f.write(result['final_code'])
        
        # 4. 保存final.json
        final_json = {
            'bits': ''.join(str(b) for b in sample_watermark),
            's0': anchor_info['s0'],
            'bitwise_thresholds': anchor_info['bitwise_thresholds'],
            's_after': result['s_after'],
            'trace': result['trace'],
            'median_code': anchor_info.get('median_code', ''),
            'extreme_codes': {}
        }
        
        final_json_path = sample_dir / "final.json"
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        
        return {
            'sample_id': sample_id,
            'success': True,
            'error': None,
            'skipped': False,
            'iterations': len(result['trace'])
        }
        
    except Exception as e:
        import traceback
        
        # 保存错误信息
        error_path = sample_dir / "error.txt"
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
        
        return {
            'sample_id': sample_id,
            'success': False,
            'error': str(e),
            'skipped': False
        }


def embed_watermarks(
    test_samples_path: str,
    model_dir: str,
    output_base: str,
    secret_key: str = "XDF",
    concurrency: int = 1,
    resume: bool = False,
    K: int = 50,
    max_iters: int = 100,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20
):
    """
    为所有测试样本嵌入水印（支持并发和断点继续）
    
    Args:
        test_samples_path: 测试样本文件路径
        model_dir: CodeT5模型路径
        output_base: 输出基础目录
        secret_key: 密钥
        concurrency: 并发进程数
        resume: 是否启用断点继续
        K: 候选数量
        max_iters: 最大迭代次数
        num_workers: 内部并行worker数
        batch_size_for_parallel: 内部批处理大小
    """
    # 读取测试样本
    print(f"正在读取测试样本: {test_samples_path}")
    with open(test_samples_path, 'r', encoding='utf-8') as f:
        test_samples = [json.loads(line) for line in f]
    
    print(f"总样本数: {len(test_samples)}")
    print(f"水印位数: 4-bit")
    print(f"固定水印: [1, 1, 0, 0]")
    print(f"并发数: {concurrency}")
    print(f"断点继续: {'启用' if resume else '禁用'}")
    
    output_base_path = Path(output_base)
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # 准备任务列表
    tasks = []
    skipped_count = 0
    
    for sample in test_samples:
        sample_id = sample['id']
        
        # 断点续传预检查（可选，用于统计）
        if resume:
            final_json_path = output_base_path / sample_id / "final.json"
            if final_json_path.exists():
                skipped_count += 1
                continue
        
        # 准备参数字典（用于进程间传递）
        args_dict = {
            'output_base': str(output_base),
            'model_dir': str(model_dir),
            'secret_key': secret_key,
            'K': K,
            'quantile': 0.99,
            'max_iters': max_iters,
            'num_workers': num_workers,
            'batch_size_for_parallel': batch_size_for_parallel,
            'resume': resume
        }
        
        tasks.append((sample['id'], sample['code'], sample['watermark'], args_dict))
    
    if resume and skipped_count > 0:
        print(f"跳过已完成: {skipped_count} 个样本")
    
    if not tasks:
        print("所有样本已完成，无需处理")
        return
    
    print(f"待处理样本: {len(tasks)}")
    print("="*60)
    
    # 执行任务（并发或串行）
    results = []
    
    if concurrency > 1:
        print(f"使用并发模式 (workers={concurrency})")
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_sample, task) for task in tasks]
            
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="嵌入进度", ncols=80):
                result = future.result()
                results.append(result)
                
                # 实时显示失败信息
                if not result['success']:
                    print(f"\n[✗] {result['sample_id']} 失败: {result['error']}")
                elif not result.get('skipped', False):
                    # 成功且非跳过的，显示迭代次数
                    iters = result.get('iterations', '?')
                    print(f"\n[✓] {result['sample_id']} 成功 (迭代: {iters})")
    else:
        print("使用串行模式")
        for task in tqdm(tasks, desc="嵌入进度", ncols=80):
            result = process_one_sample(task)
            results.append(result)
            
            if not result['success']:
                print(f"\n[✗] {result['sample_id']} 失败: {result['error']}")
            elif not result.get('skipped', False):
                iters = result.get('iterations', '?')
                print(f"\n[✓] {result['sample_id']} 成功 (迭代: {iters})")
    
    # 统计结果
    print("\n" + "="*60)
    success_count = sum(1 for r in results if r['success'])
    skipped_in_process = sum(1 for r in results if r.get('skipped', False))
    failed_count = len(results) - success_count
    
    print(f"✓ 水印嵌入完成")
    print(f"✓ 成功: {success_count}/{len(results)}")
    if skipped_in_process > 0:
        print(f"  - 新完成: {success_count - skipped_in_process}")
        print(f"  - 跳过（已存在）: {skipped_in_process}")
    if failed_count > 0:
        print(f"✗ 失败: {failed_count}")
    print(f"✓ 输出目录: {output_base}")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    # 使用绝对路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[3]
    
    # 读取配置文件
    config_path = script_dir.parent / "configs" / "experiment_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 从配置文件获取默认参数
        default_K = config['embedding']['K']
        default_max_iters = config['embedding']['max_iters']
    else:
        # 配置文件不存在时的默认值
        default_K = 50
        default_max_iters = 100
    
    # 命令行参数解析（命令行参数会覆盖配置文件）
    parser = argparse.ArgumentParser(
        description="Step 2: 嵌入水印（支持并发和断点继续）"
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='并发处理的进程数（默认=1，串行）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点继续：跳过已有final.json的样本'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=default_K,
        help=f'候选代码数量（默认={default_K}，来自配置文件）'
    )
    parser.add_argument(
        '--max_iters',
        type=int,
        default=default_max_iters,
        help=f'最大迭代次数（默认={default_max_iters}，来自配置文件）'
    )
    args = parser.parse_args()
    
    # 测试样本路径
    test_samples_path = script_dir.parent / "data" / "test_samples.jsonl"
    
    # CodeT5模型路径（位于 XDF/Watermark4code/best_model）
    model_dir = project_root / "Watermark4code" / "best_model"
    
    # 输出目录
    output_base = script_dir.parent / "results" / "embedded"
    
    # 嵌入水印
    embed_watermarks(
        test_samples_path=test_samples_path,
        model_dir=model_dir,
        output_base=output_base,
        concurrency=args.concurrency,
        resume=args.resume,
        K=args.K,
        max_iters=args.max_iters
    )

