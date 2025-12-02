"""
Step 1: 为10个测试代码分析768个维度的指标

输入：MBXP测试集
输出：data/dimension_analysis/run_XXXX_analysis.json
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer

# 导入JavaCodeAugmentor需要特殊处理
srcmarker_path = project_root / "SrcMarker-main" / "contrastive_learning"
sys.path.insert(0, str(srcmarker_path))
from java_augmentor import JavaCodeAugmentor


sys.path.insert(0, str(Path(__file__).parent))
from dataset_loader import DatasetLoader


def load_test_codes(config, num_codes=10, start_index=0):
    """加载测试代码"""
    loader = DatasetLoader(config)
    return loader.load_codes(num_codes, start_index)


def generate_variants(code, augmentor, K=100):
    """生成K个语义等价变体"""
    variants = []
    for _ in range(K):
        try:
            # augment方法返回一个列表
            variant_list = augmentor.augment(code, num_augmentations=1)
            if variant_list:
                variants.append(variant_list[0])
            else:
                variants.append(code)
        except Exception as e:
            print(f"[警告] 变体生成失败: {e}")
            variants.append(code)  # 失败时使用原代码
    return variants


def apply_rename_attack(code, seed):
    """应用重命名攻击"""
    try:
        from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig
        renamer = JavaVariableRenamer(code)
        config = AttackConfig(naming_strategy="random", seed=seed)
        attacked = renamer.apply_renames(config, rename_ratio=1.0)
        return attacked
    except Exception as e:
        print(f"[警告] 攻击失败 (seed={seed}): {e}")
        return code


def find_extreme_variants(embeddings, variants, left_idx=20, right_idx=20):
    """
    找到最左和最右的变体
    
    Args:
        embeddings: (N, 768) 编码向量
        variants: 变体列表
        left_idx: 选择前left_idx个变体中的最左
        right_idx: 选择前right_idx个变体中的最右
    
    Returns:
        left_variant, right_variant, left_global_idx, right_global_idx
    """
    # 计算簇中心（所有变体的中位数）
    s0 = np.median(embeddings[1:], axis=0)  # 跳过original
    
    # 计算每个变体相对簇中心的偏移
    offsets = embeddings[1:] - s0  # (K, 768)
    
    # 计算每个变体的平均偏移（作为"左右"的度量）
    avg_offsets = offsets.mean(axis=1)  # (K,)
    
    # 选择前20个变体中的最左和最右
    top_indices = np.argsort(avg_offsets)
    left_global_idx = top_indices[0] + 1  # +1 因为跳过了original
    right_global_idx = top_indices[-1] + 1
    
    return variants[left_global_idx - 1], variants[right_global_idx - 1], left_global_idx, right_global_idx


def compute_metrics_for_dimension(dim_i, variants_768d, original_768d, 
                                   original_attacks_768d, left_attacks_768d, 
                                   right_attacks_768d, left_idx, right_idx):
    """
    计算单个维度的所有指标
    
    返回字典包含：
    - triple_order_preservation: 三元顺序保持率
    - left_vs_right_preservation: 左右顺序保持率
    - overall_sign_preservation: 符号保持率
    - embeddability_value: 可嵌入度
    """
    # 获取三个关键代码在该维度的值
    v_left = variants_768d[left_idx]
    v_original = original_768d[0]
    v_right = variants_768d[right_idx]
    
    val_left = v_left[dim_i]
    val_original = v_original[dim_i]
    val_right = v_right[dim_i]
    
    # 计算簇中心
    s0 = np.median(variants_768d[:, dim_i])
    
    # === 指标1：三元顺序保持率 ===
    triple_count = 0
    left_right_count = 0
    
    # 记录攻击前的顺序
    order_before = sorted([
        ('left', val_left),
        ('original', val_original),
        ('right', val_right)
    ], key=lambda x: x[1])
    order_before_labels = [x[0] for x in order_before]
    
    num_attacks = len(original_attacks_768d)
    
    for seed_idx in range(num_attacks):
        left_att = left_attacks_768d[seed_idx]
        orig_att = original_attacks_768d[seed_idx]
        right_att = right_attacks_768d[seed_idx]
        
        # 攻击后的顺序
        order_after = sorted([
            ('left', left_att[dim_i]),
            ('original', orig_att[dim_i]),
            ('right', right_att[dim_i])
        ], key=lambda x: x[1])
        order_after_labels = [x[0] for x in order_after]
        
        # 三元顺序是否保持
        if order_after_labels == order_before_labels:
            triple_count += 1
        
        # 左右顺序
        if val_left < val_right:
            if left_att[dim_i] < right_att[dim_i]:
                left_right_count += 1
        else:
            if left_att[dim_i] > right_att[dim_i]:
                left_right_count += 1
    
    triple_order_preservation = triple_count / num_attacks
    left_vs_right_preservation = left_right_count / num_attacks
    
    # === 指标2：符号保持率 ===
    left_sign_count = 0
    orig_sign_count = 0
    right_sign_count = 0
    
    offset_before_left = val_left - s0
    offset_before_orig = val_original - s0
    offset_before_right = val_right - s0
    
    for seed_idx in range(num_attacks):
        orig_att = original_attacks_768d[seed_idx]
        
        # 左变体
        left_att = left_attacks_768d[seed_idx]
        offset_after_left = left_att[dim_i] - orig_att[dim_i]
        if np.sign(offset_before_left) == np.sign(offset_after_left):
            left_sign_count += 1
        
        # 原始代码
        offset_after_orig = orig_att[dim_i] - orig_att[dim_i]  # = 0
        if abs(offset_before_orig) < 0.05:  # 原始代码接近簇中心
            orig_sign_count += 1
        elif np.sign(offset_before_orig) == np.sign(offset_after_orig):
            orig_sign_count += 1
        
        # 右变体
        right_att = right_attacks_768d[seed_idx]
        offset_after_right = right_att[dim_i] - orig_att[dim_i]
        if np.sign(offset_before_right) == np.sign(offset_after_right):
            right_sign_count += 1
    
    overall_sign_preservation = (left_sign_count + orig_sign_count + right_sign_count) / (3 * num_attacks)
    
    # === 指标3：可嵌入度 ===
    embeddability = float(variants_768d[:, dim_i].max() - variants_768d[:, dim_i].min())
    
    # === 指标4：簇稳定性（Cluster Stability）===
    # 衡量攻击后簇中心（s0）的变化程度
    cluster_stability_scores = []
    for seed_idx in range(num_attacks):
        orig_att = original_attacks_768d[seed_idx]
        # 近似：用攻击后的原始代码投影估计簇中心的偏移
        # 假设攻击对所有代码的影响相似
        s0_shift = orig_att[dim_i] - val_original
        s0_attacked_approx = s0 + s0_shift
        
        # 计算相对变化（归一化）
        if embeddability > 1e-8:
            relative_change = abs(s0_attacked_approx - s0) / embeddability
            stability = max(0.0, 1.0 - relative_change)
        else:
            stability = 1.0
        cluster_stability_scores.append(stability)
    
    cluster_stability = float(np.mean(cluster_stability_scores))
    
    # === 指标5：极值稳定性（Extreme Stability）===
    # 衡量极值位置（pos和neg）攻击后的稳定性
    extreme_stability_scores = []
    for seed_idx in range(num_attacks):
        # 正极值（right）的稳定性
        right_shift = right_attacks_768d[seed_idx][dim_i] - val_right
        if embeddability > 1e-8:
            pos_stability = max(0.0, 1.0 - abs(right_shift) / embeddability)
        else:
            pos_stability = 1.0
        
        # 负极值（left）的稳定性
        left_shift = left_attacks_768d[seed_idx][dim_i] - val_left
        if embeddability > 1e-8:
            neg_stability = max(0.0, 1.0 - abs(left_shift) / embeddability)
        else:
            neg_stability = 1.0
        
        # 两个极值的平均稳定性
        extreme_stability_scores.append((pos_stability + neg_stability) / 2.0)
    
    extreme_stability = float(np.mean(extreme_stability_scores))
    
    # === 指标6：簇间距离（Cluster Separation）===
    # 找到median_code（距离s0最近的变体）
    distances_to_s0 = np.abs(variants_768d[:, dim_i] - s0)
    median_idx = int(np.argmin(distances_to_s0))
    val_median = variants_768d[median_idx, dim_i]
    
    # 计算三个簇的最小距离（归一化）
    if embeddability > 1e-8:
        dist_pos_median = abs(val_right - val_median)
        dist_neg_median = abs(val_left - val_median)
        dist_pos_neg = abs(val_right - val_left)
        
        min_distance = min(dist_pos_median, dist_neg_median, dist_pos_neg)
        cluster_separation = float(min_distance / embeddability)
    else:
        cluster_separation = 0.0
    
    return {
        'triple_order_preservation': float(triple_order_preservation),
        'left_vs_right_preservation': float(left_vs_right_preservation),
        'overall_sign_preservation': float(overall_sign_preservation),
        'embeddability_value': float(embeddability),
        'cluster_stability': float(cluster_stability),
        'extreme_stability': float(extreme_stability),
        'cluster_separation': float(cluster_separation)
    }


def main():
    print("="*80)
    print("Step 1: 分析768个维度的指标")
    print("="*80)
    
    # 设置工作目录为项目根目录（与原始流程一致）
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # 加载配置
    config_path = project_root / "dimension_strategy_comparison" / "configs" / "base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 初始化模型（使用绝对路径，与原始流程一致）
    print("\n[1/6] 加载模型...")
    model, tokenizer = load_best_model(config['model_dir'])
    device = next(model.parameters()).device
    
    # 初始化工具
    print("[2/6] 初始化工具...")
    augmentor = JavaCodeAugmentor()
    
    # 加载测试代码
    print(f"[3/6] 加载 {config['num_test_codes']} 个测试代码...")
    test_codes = load_test_codes(
        config,
        num_codes=config['num_test_codes'],
        start_index=config['start_index']
    )
    print(f"  已加载 {len(test_codes)} 个代码")
    
    # 保存测试代码
    data_dir = project_root / "dimension_strategy_comparison" / "data"
    os.makedirs(data_dir, exist_ok=True)
    with open(data_dir / "test_codes.jsonl", 'w', encoding='utf-8') as f:
        for i, code in enumerate(test_codes):
            f.write(json.dumps({"run_id": i, "code": code}, ensure_ascii=False) + '\n')
    print(f"  已保存到 data/test_codes.jsonl")
    
    # 处理每个代码
    print(f"\n[4/6] 分析维度...")
    analysis_dir = data_dir / "dimension_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    for run_id, code in enumerate(test_codes):
        print(f"\n处理 run_{run_id:04d} ({run_id+1}/{len(test_codes)})...")
        
        # 生成变体
        print(f"  生成 {config['dimension_analysis']['num_variants']} 个变体...")
        variants = generate_variants(code, augmentor, K=config['dimension_analysis']['num_variants'])
        
        # 编码所有代码（原始 + 变体）
        print(f"  编码代码...")
        all_codes = [code] + variants
        embeddings = embed_codes(
            model, tokenizer, all_codes,
            batch_size=config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (101, 768)
        
        # 找到最左和最右的变体
        print(f"  识别极值变体...")
        left_variant, right_variant, left_idx, right_idx = find_extreme_variants(
            embeddings, variants,
            left_idx=config['dimension_analysis']['num_extreme_variants'],
            right_idx=config['dimension_analysis']['num_extreme_variants']
        )
        
        # 生成攻击版本
        print(f"  生成攻击版本...")
        original_attacks = []
        for seed in range(config['dimension_analysis']['num_attacks_per_code']):
            attacked = apply_rename_attack(code, seed)
            original_attacks.append(attacked)
        
        left_attacks = []
        for seed in range(config['dimension_analysis']['num_attacks_per_variant']):
            attacked = apply_rename_attack(left_variant, seed)
            left_attacks.append(attacked)
        
        right_attacks = []
        for seed in range(config['dimension_analysis']['num_attacks_per_variant']):
            attacked = apply_rename_attack(right_variant, seed)
            right_attacks.append(attacked)
        
        # 编码攻击版本
        print(f"  编码攻击版本...")
        original_attacks_embeddings = embed_codes(
            model, tokenizer, original_attacks,
            batch_size=config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (30, 768)
        
        left_attacks_embeddings = embed_codes(
            model, tokenizer, left_attacks,
            batch_size=config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (30, 768)
        
        right_attacks_embeddings = embed_codes(
            model, tokenizer, right_attacks,
            batch_size=config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (30, 768)
        
        # 计算所有768个维度的指标
        print(f"  计算768个维度的指标...")
        dimension_scores = []
        for dim in tqdm(range(768), desc="  维度进度", ncols=80):
            scores = compute_metrics_for_dimension(
                dim, embeddings, embeddings[:1], 
                original_attacks_embeddings, left_attacks_embeddings, right_attacks_embeddings,
                left_idx, right_idx
            )
            dimension_scores.append({
                'dimension': dim,
                **scores
            })
        
        # 计算相关性矩阵
        print(f"  计算相关性矩阵...")
        correlation_matrix = np.corrcoef(embeddings.T)  # (768, 768)
        
        # 保存结果
        output_file = analysis_dir / f"run_{run_id:04d}_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'run_id': run_id,
                'left_variant_idx': int(left_idx),
                'right_variant_idx': int(right_idx),
                'dimension_scores': dimension_scores,
                'correlation_matrix': correlation_matrix.tolist()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] 已保存到 {output_file}")
    
    print("\n[5/6] 汇总统计...")
    # 统计全局信息
    summary = {
        'num_codes': len(test_codes),
        'num_dimensions': 768,
        'metrics': ['triple_order_preservation', 'left_vs_right_preservation', 
                   'overall_sign_preservation', 'embeddability_value',
                   'cluster_stability', 'extreme_stability', 'cluster_separation']
    }
    
    with open(analysis_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n[6/6] 完成！")
    print("="*80)
    print(f"已分析 {len(test_codes)} 个代码的 768 个维度")
    print(f"结果保存在: data/dimension_analysis/")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()

