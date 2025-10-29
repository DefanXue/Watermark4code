"""Batch rename attack with automatic extraction support."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from .java_variable_renamer import JavaVariableRenamer
from .attack_config import AttackConfig


def apply_batch_rename_attack(
    injection_root: str,
    output_dir: str,
    strategy: str = 'random',
    seed: int = 42,
    auto_extract: bool = False,
    model_dir: str = None,
    secret: str = None
) -> Dict[str, Any]:
    """Apply rename attack to all injection results in batch.
    
    Args:
        injection_root: Root directory containing multiple injection runs
        output_dir: Directory to save attack results
        strategy: Naming strategy ('random', 'sequential', 'obfuscated')
        seed: Random seed for reproducibility
        auto_extract: Whether to automatically extract watermarks after attack
        model_dir: Path to model directory (required if auto_extract=True)
        secret: Secret key for extraction (required if auto_extract=True)
        
    Returns:
        Dictionary containing batch attack statistics
    """
    injection_root = Path(injection_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all injection runs
    run_dirs = sorted([d for d in injection_root.iterdir() 
                      if d.is_dir() and d.name.startswith('run_')])
    
    if not run_dirs:
        raise ValueError(f"No injection runs found in {injection_root}")
    
    print(f"Found {len(run_dirs)} injection runs")
    print(f"Strategy: {strategy}, Seed: {seed}")
    if auto_extract:
        print(f"Auto-extraction enabled with secret: {secret}")
    
    results = []
    
    for run_dir in tqdm(run_dirs, desc="Applying rename attacks"):
        run_name = run_dir.name
        
        # Load watermarked code
        watermarked_path = run_dir / "final.java"
        if not watermarked_path.exists():
            print(f"Warning: Watermarked code not found in {run_name}, skipping")
            continue
        
        with open(watermarked_path, 'r', encoding='utf-8') as f:
            watermarked_code = f.read()
        
        # Apply attack
        config = AttackConfig(naming_strategy=strategy, seed=seed)
        renamer = JavaVariableRenamer(watermarked_code)
        attacked_code = renamer.apply_renames(config)
        
        # Create output directory for this run
        run_output_dir = output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save attacked code
        attacked_path = run_output_dir / "attacked.java"
        with open(attacked_path, 'w', encoding='utf-8') as f:
            f.write(attacked_code)
        
        # Load injection metadata
        final_json_path = run_dir / "final.json"
        if final_json_path.exists():
            with open(final_json_path, 'r', encoding='utf-8') as f:
                injection_info = json.load(f)
        else:
            injection_info = {}
        
        # Create result entry
        result_entry = {
            'run_name': run_name,
            'strategy': strategy,
            'seed': seed,
            'original_bits': injection_info.get('bits', 'unknown'),
            'attacked_code_path': str(attacked_path)
        }
        
        # Auto-extract if requested
        if auto_extract and model_dir and secret:
            try:
                from Watermark4code.extraction.extractor import extract_watermark
                from Watermark4code.encoder.loader import load_best_model
                
                # Load model
                model, tokenizer = load_best_model(model_dir)
                
                # Extract watermark
                extracted_info = extract_watermark(
                    model=model,
                    tokenizer=tokenizer,
                    code=attacked_code,
                    secret=secret,
                    cluster_info=injection_info.get('cluster_info')
                )
                
                result_entry['extracted_bits'] = extracted_info.get('bits', 'failed')
                result_entry['extraction_success'] = (
                    result_entry['extracted_bits'] == result_entry['original_bits']
                )
                
            except Exception as e:
                print(f"Warning: Extraction failed for {run_name}: {e}")
                result_entry['extracted_bits'] = 'error'
                result_entry['extraction_success'] = False
        
        results.append(result_entry)
        
        # Save individual result
        result_path = run_output_dir / "attack_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_entry, f, indent=2, ensure_ascii=False)
    
    # Compute statistics
    total = len(results)
    if auto_extract:
        successes = sum(1 for r in results if r.get('extraction_success', False))
        success_rate = successes / total if total > 0 else 0
    else:
        successes = 0
        success_rate = 0
    
    summary = {
        'strategy': strategy,
        'seed': seed,
        'total_runs': total,
        'auto_extract': auto_extract,
        'extraction_successes': successes,
        'extraction_success_rate': success_rate,
        'results': results
    }
    
    # Save summary
    summary_path = output_dir / "batch_attack_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch attack completed!")
    print(f"Total runs: {total}")
    if auto_extract:
        print(f"Extraction success rate: {success_rate:.2%} ({successes}/{total})")
    print(f"Results saved to: {output_dir}")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply rename attack to watermarked code in batch'
    )
    parser.add_argument('--injection_root', type=str, required=True,
                       help='Root directory containing injection results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save attack results')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'sequential', 'obfuscated'],
                       help='Naming strategy for attack')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--auto_extract', action='store_true',
                       help='Automatically extract watermarks after attack')
    parser.add_argument('--model_dir', type=str,
                       help='Path to model directory (required if --auto_extract)')
    parser.add_argument('--secret', type=str,
                       help='Secret key for extraction (required if --auto_extract)')
    
    args = parser.parse_args()
    
    if args.auto_extract and (not args.model_dir or not args.secret):
        parser.error("--model_dir and --secret are required when --auto_extract is set")
    
    apply_batch_rename_attack(
        injection_root=args.injection_root,
        output_dir=args.output_dir,
        strategy=args.strategy,
        seed=args.seed,
        auto_extract=args.auto_extract,
        model_dir=args.model_dir,
        secret=args.secret
    )

