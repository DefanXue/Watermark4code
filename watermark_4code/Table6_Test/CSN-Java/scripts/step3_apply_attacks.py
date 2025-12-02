"""Step 3: 应用攻击
应用所有类型的攻击：T@1/2/3, V@25%/50%/75%/100%, DualCh
"""

import json
import sys
import random
import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  # XDF目录
sys.path.insert(0, str(project_root))

from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer
from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig

# 添加SrcMarker-main到路径
srcmarker_path = project_root / "SrcMarker-main"
sys.path.insert(0, str(srcmarker_path))

from mutable_tree.transformers import (
    IfBlockSwapTransformer,
    CompoundIfTransformer,
    ConditionTransformer,
    LoopTransformer,
    InfiniteLoopTransformer,
    UpdateTransformer,
    SameTypeDeclarationTransformer,
    VarDeclLocationTransformer,
    VarInitTransformer,
    VarNameStyleTransformer,
)
from mutable_tree.adaptors import JavaAdaptor
from mutable_tree.stringifiers import JavaStringifier
from code_transform_provider import CodeTransformProvider
import tree_sitter


# 初始化tree-sitter（使用SrcMarker构建的languages.so）
lang_so = project_root / "SrcMarker-main" / "parser" / "languages.so"
PARSER_LANG = tree_sitter.Language(str(lang_so), "java")
ts_parser = tree_sitter.Parser()
ts_parser.set_language(PARSER_LANG)


def get_java_function_root(root):
    """获取Java函数根节点"""
    if root.type != "program":
        return None
    if len(root.children) == 0:
        return None
    class_decl = root.children[0]
    if class_decl.type != "class_declaration":
        return None
    if len(class_decl.children) < 4:
        return None
    class_body = class_decl.children[3]
    if class_body.type != "class_body":
        return None
    if len(class_body.children) < 2:
        return None
    func_root = class_body.children[1]
    return func_root if func_root.type == "method_declaration" else None


def apply_transform_attack(code: str, n_transforms: int):
    """
    应用代码转换攻击
    
    Args:
        code: 原始代码
        n_transforms: 转换数量
    
    Returns:
        (attacked_code, selected_transforms)
    """
    # 所有转换器
    all_transformers = [
        IfBlockSwapTransformer(),
        CompoundIfTransformer(),
        ConditionTransformer(),
        LoopTransformer(),
        InfiniteLoopTransformer(),
        UpdateTransformer(),
        SameTypeDeclarationTransformer(),
        VarDeclLocationTransformer(),
        VarInitTransformer(),
        VarNameStyleTransformer(),
    ]
    
    # 创建 CodeTransformProvider
    try:
        transform_provider = CodeTransformProvider("java", ts_parser, all_transformers)
    except Exception as e:
        print(f"    警告: 无法创建 CodeTransformProvider: {e}")
        return code, []
    
    # 获取可行的转换
    feasible_transforms = []
    for transformer in all_transformers:
        transformer_name = transformer.__class__.__name__
        available_keys = transformer.get_available_transforms()
        
        for key in available_keys:
            try:
                # 尝试应用转换
                new_code = transform_provider.code_transform(code, [key])
                
                # 检查是否有变化
                if new_code != code and len(new_code) > 0:
                    feasible_transforms.append((transformer_name, key))
            except Exception:
                # 转换失败，跳过
                continue
    
    if len(feasible_transforms) == 0:
        print("    警告: 没有可行的转换，返回原代码")
        return code, []
    
    # 随机选择n个转换
    n_to_apply = min(n_transforms, len(feasible_transforms))
    selected = random.sample(feasible_transforms, n_to_apply)
    
    # 构建转换键列表
    selected_keys = [key for _, key in selected]
    
    # 应用所有选定的转换
    try:
        attacked_code = transform_provider.code_transform(code, selected_keys)
        result_keys = [f"{name}:{key}" for name, key in selected]
        return attacked_code, result_keys
    except Exception as e:
        print(f"    警告: 转换应用失败: {e}")
        return code, []


def apply_variable_attack(code: str, proportion: float, seed: int = 42):
    """
    应用变量重命名攻击（使用Watermark4code的重命名器）
    
    Args:
        code: 原始代码
        proportion: 重命名比例 (0.25/0.5/0.75/1.0)
        seed: 随机种子
    
    Returns:
        (attacked_code, rename_mapping)
    """
    config = AttackConfig(
        naming_strategy='random',
        seed=seed,
        rename_ratio=proportion
    )
    
    renamer = JavaVariableRenamer(code)
    attacked_code = renamer.apply_renames(config, rename_ratio=proportion)
    
    return attacked_code, renamer.rename_mapping


def apply_dual_attack(code: str, seed: int = 42):
    """
    应用双通道攻击：50%变量重命名 + 2个代码转换
    
    Args:
        code: 原始代码
        seed: 随机种子（用于变量重命名）
    
    Returns:
        (attacked_code, attack_info)
    """
    # 第一步：变量重命名 (50%)
    code_after_rename, rename_mapping = apply_variable_attack(code, proportion=0.5, seed=seed)
    
    # 第二步：代码转换 (2个)
    code_after_transform, transforms = apply_transform_attack(code_after_rename, n_transforms=2)
    
    # 合并攻击信息
    attack_info = {
        "rename_mapping": rename_mapping,
        "transforms": transforms
    }
    
    return code_after_transform, attack_info


def process_one_sample_attacks(task):
    """
    并发worker：对单个样本应用所有攻击
    
    Args:
        task: (sample_dir, output_base, resume)
    
    Returns:
        dict: {'sample_id': str, 'success': bool, 'attacks_done': int}
    """
    sample_dir, output_base, resume = task
    sample_id = sample_dir.name
    output_base = Path(output_base)
    
    # 读取水印代码
    watermarked_path = sample_dir / "final.java"
    if not watermarked_path.exists():
        return {
            'sample_id': sample_id,
            'success': False,
            'attacks_done': 0,
            'message': '未找到水印代码'
        }
    
    with open(watermarked_path, 'r', encoding='utf-8') as f:
        watermarked_code = f.read()
    
    attacks_done = 0
    attack_types = (
        [f"T{n}" for n in [1, 2, 3]] + 
        [f"V{int(p*100)}" for p in [0.25, 0.5, 0.75, 1.0]] + 
        ["DualCh"]
    )
    
    for attack_type in attack_types:
        attack_dir = output_base / attack_type / sample_id
        attack_dir.mkdir(parents=True, exist_ok=True)
        
        # 断点续传检查
        if resume and (attack_dir / "attacked.java").exists():
            attacks_done += 1
            continue
        
        try:
            if attack_type.startswith('T'):
                n_trans = int(attack_type[1:])
                attacked_code, transforms = apply_transform_attack(watermarked_code, n_trans)
                
                with open(attack_dir / "attacked.java", 'w', encoding='utf-8') as f:
                    f.write(attacked_code)
                with open(attack_dir / "attack_metadata.json", 'w', encoding='utf-8') as f:
                    json.dump({"transforms": transforms}, f, indent=2, ensure_ascii=False)
            
            elif attack_type.startswith('V'):
                proportion = int(attack_type[1:]) / 100.0
                attacked_code, rename_mapping = apply_variable_attack(watermarked_code, proportion)
                
                with open(attack_dir / "attacked.java", 'w', encoding='utf-8') as f:
                    f.write(attacked_code)
                with open(attack_dir / "attack_metadata.json", 'w', encoding='utf-8') as f:
                    json.dump({"rename_mapping": rename_mapping}, f, indent=2, ensure_ascii=False)
            
            elif attack_type == "DualCh":
                attacked_code, dual_info = apply_dual_attack(watermarked_code)
                
                with open(attack_dir / "attacked.java", 'w', encoding='utf-8') as f:
                    f.write(attacked_code)
                with open(attack_dir / "attack_metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(dual_info, f, indent=2, ensure_ascii=False)
            
            attacks_done += 1
        
        except Exception as e:
            with open(attack_dir / "error.txt", 'w', encoding='utf-8') as f:
                f.write(f"Attack failed: {str(e)}\n")
    
    return {
        'sample_id': sample_id,
        'success': True,
        'attacks_done': attacks_done
    }


def apply_all_attacks(embedded_dir: str, output_base: str, concurrency: int = 1, resume: bool = False):
    """
    对所有样本应用所有攻击
    
    Args:
        embedded_dir: 嵌入水印的结果目录
        output_base: 攻击结果输出基础目录
    """
    embedded_dir = Path(embedded_dir)
    output_base = Path(output_base)
    
    # 获取所有样本
    sample_dirs = sorted([d for d in embedded_dir.iterdir() if d.is_dir()])
    
    print(f"✅ 找到 {len(sample_dirs)} 个样本")
    print(f"✅ 并发数: {concurrency}")
    print(f"✅ 断点续传: {'开启' if resume else '关闭'}\n")
    
    # 准备任务列表
    tasks = [(sample_dir, output_base, resume) for sample_dir in sample_dirs]
    
    # 并发执行
    results = []
    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_one_sample_attacks, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="应用攻击") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    total_attacks = sum(r['attacks_done'] for r in results)
    
    print(f"\n{'='*60}")
    print(f"✓ 攻击应用完成")
    print(f"✓ 成功样本: {success_count}/{len(sample_dirs)}")
    print(f"✓ 总攻击数: {total_attacks}")
    print(f"✓ 输出目录: {output_base}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='步骤3: 应用攻击')
    parser.add_argument('--embedded_dir', type=str, default=None,
                        help='嵌入水印的结果目录 (默认: results/embedded)')
    parser.add_argument('--output_base', type=str, default=None,
                        help='攻击结果输出基础目录 (默认: results/attacked)')
    parser.add_argument('--concurrency', type=int, default=1,
                        help='并发数 (默认: 1)')
    parser.add_argument('--resume', action='store_true',
                        help='断点续传 (默认: 关闭)')
    
    args = parser.parse_args()
    
    # 设置全局随机种子（确保可重现性和样本间多样性）
    random.seed(42)
    
    # 使用绝对路径
    script_dir = Path(__file__).parent
    
    # 嵌入结果目录
    embedded_dir = args.embedded_dir if args.embedded_dir else script_dir.parent / "results" / "embedded"
    
    # 攻击结果输出目录
    output_base = args.output_base if args.output_base else script_dir.parent / "results" / "attacked"
    
    # 应用所有攻击
    apply_all_attacks(
        embedded_dir=embedded_dir,
        output_base=output_base,
        concurrency=args.concurrency,
        resume=args.resume
    )

