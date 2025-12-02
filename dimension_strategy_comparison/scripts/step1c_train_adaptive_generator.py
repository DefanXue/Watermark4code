"""
Step 1c: 训练自适应方向生成器 (Strategy 6)

输入：data/dimension_analysis/run_XXXX_analysis.json（已有数据）
输出：results/strategy_6_adaptive/trained_generator.pth

自适应方向生成器根据代码嵌入动态生成最优的投影方向
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dimension_strategy_comparison"))

# 添加contrastive_learning路径（用于JavaCodeAugmentor）
srcmarker_path = project_root / "SrcMarker-main"
sys.path.insert(0, str(srcmarker_path))
contrastive_path = srcmarker_path / "contrastive_learning"
sys.path.insert(0, str(contrastive_path))

from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.injection.plan import build_candidates_test_like
from java_augmentor import JavaCodeAugmentor
from dimension_strategy_comparison.models.adaptive_direction_generator import AdaptiveDirectionGenerator


def get_feasible_mutable_transforms(code, java_augmentor):
    """检测对某个代码可行的MutableAST变换"""
    try:
        feasible_map = java_augmentor._get_feasible_mutable_transforms(code)
        return feasible_map
    except Exception as e:
        return {}


def apply_single_mutable_transform(code, transformer_name, key, java_augmentor):
    """应用单个MutableAST变换"""
    try:
        method_code, metadata = java_augmentor._extract_method_from_full_class(code)

        import mutable_tree.transformers as ast_transformers
        from code_transform_provider import CodeTransformProvider
        import tree_sitter

        transformer_map = {
            "IfBlockSwapTransformer": ast_transformers.IfBlockSwapTransformer,
            "CompoundIfTransformer": ast_transformers.CompoundIfTransformer,
            "ConditionTransformer": ast_transformers.ConditionTransformer,
            "LoopTransformer": ast_transformers.LoopTransformer,
            "InfiniteLoopTransformer": ast_transformers.InfiniteLoopTransformer,
            "UpdateTransformer": ast_transformers.UpdateTransformer,
            "SameTypeDeclarationTransformer": ast_transformers.SameTypeDeclarationTransformer,
            "VarDeclLocationTransformer": ast_transformers.VarDeclLocationTransformer,
            "VarInitTransformer": ast_transformers.VarInitTransformer,
        }

        if transformer_name not in transformer_map:
            return code

        parser = tree_sitter.Parser()
        srcmarker_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        parser_lang = tree_sitter.Language(
            os.path.join(srcmarker_root, "parser", "languages.so"),
            "java"
        )
        parser.set_language(parser_lang)

        transformer = transformer_map[transformer_name]()
        transform_provider = CodeTransformProvider("java", parser, [transformer])

        transformed_method = transform_provider.code_transform(method_code, [key])

        if metadata is not None:
            return java_augmentor._wrap_method_back_to_class(transformed_method, metadata)
        else:
            return transformed_method

    except Exception:
        return code


def generate_attacks_via_mutable_ast(variant_code, java_augmentor):
    """
    使用MutableAST变换、重命名和等价变体生成作为"攻击"

    逻辑：
      1. 检测可行的MutableAST变换（最多9个）
      2. 对每个transformer随机选择一个key（而不是固定第一个）
      3. 加上8个full_variable_rename
      4. 生成15个等价变体作为攻击
      5. 返回实际生成的攻击（不填充）
    """
    import random
    attacked_codes = []

    # 检测可行的MutableAST变换
    feasible_map = get_feasible_mutable_transforms(variant_code, java_augmentor)

    # 应用每个可行的MutableAST变换（随机选择key）
    for transformer_name, feasible_keys in feasible_map.items():
        if feasible_keys:
            key = random.choice(feasible_keys)  # ← 随机选择key
            attacked_code = apply_single_mutable_transform(
                variant_code,
                transformer_name,
                key,
                java_augmentor
            )
            if attacked_code != variant_code:
                attacked_codes.append(attacked_code)

    # 应用8个full_variable_rename（使用seed确保与测试时一致）
    try:
        from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer
        from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig

        for seed in range(8):
            try:
                renamer = JavaVariableRenamer(variant_code)
                config = AttackConfig(naming_strategy="random", seed=seed)
                renamed_code = renamer.apply_renames(config, rename_ratio=1.0)
                if renamed_code != variant_code:
                    attacked_codes.append(renamed_code)
            except:
                pass
    except:
        pass

    # 生成15个等价变体作为攻击
    try:
        from Watermark4code.injection.plan import build_candidates_test_like

        variant_attacks, _ = build_candidates_test_like(
            variant_code,
            K=15,
            num_workers=4,
            batch_size_for_parallel=2
        )

        for v in variant_attacks:
            if isinstance(v, str) and v.strip() and v != variant_code:
                attacked_codes.append(v)
    except:
        pass

    return attacked_codes


def compute_offset_thresholds(offset_before, quantile=0.6):
    """
    为每个维度计算offset阈值

    Args:
        offset_before: (N, 4) numpy array, 每个变体相对于簇中心的偏移
        quantile: float 分位数（0.6 = 60分位）

    Returns:
        thresholds: (4,) numpy array, 每个维度的阈值
    """
    thresholds = []
    for dim in range(offset_before.shape[1]):
        offsets_dim = np.abs(offset_before[:, dim])
        threshold = np.quantile(offsets_dim, quantile)
        thresholds.append(threshold)

    return np.array(thresholds)


def sign_preservation_loss_selective_dims(W, embeddings_positive, embeddings_attacked, offset_thresholds_per_dim, num_attacks_per_variant=None):
    """
    改进的符号保持损失：只对超过阈值的维度计算

    Args:
        W: AdaptiveDirectionGenerator模块
        embeddings_positive: (N, 768) 等价变体嵌入
        embeddings_attacked: (M, 768) 攻击后嵌入
        offset_thresholds_per_dim: (4,) 每个维度的阈值
        num_attacks_per_variant: list of int，每个变体的实际攻击数
    """
    # 获取生成的方向和注意力权重
    W_directions, _ = W(embeddings_positive)  # (N, 4, 768)

    # 投影
    s_positive = torch.bmm(embeddings_positive.unsqueeze(0), W_directions[0].t().unsqueeze(0)).squeeze(0)  # (N, 4)

    # 简化：只对第一个batch处理
    if embeddings_positive.shape[0] > 1:
        s_positive = torch.stack([
            embeddings_positive[i] @ W_directions[i].t() for i in range(embeddings_positive.shape[0])
        ])

    s_attacked_list = []
    for i in range(embeddings_attacked.shape[0]):
        # 找到最相关的方向（这里简化处理，使用第一个）
        s_att = embeddings_attacked[i] @ W_directions[0].t()
        s_attacked_list.append(s_att)
    s_attacked = torch.stack(s_attacked_list)  # (M, 4)

    # 簇中心
    s_positive_np = s_positive.detach().cpu().numpy()
    s0_np = np.array([np.median(s_positive_np[:, i]) for i in range(4)])
    s0 = torch.from_numpy(s0_np).to(s_positive.device).float()

    # 偏移
    offset_before = s_positive - s0  # (N, 4)
    offset_after = s_attacked - s0   # (M, 4)

    # 对每个维度分别计算
    total_loss = torch.tensor(0.0, device=s_positive.device)
    meaningful_dim_count = 0

    # 如果没有提供num_attacks_per_variant，假设均匀分布
    if num_attacks_per_variant is None:
        N_per_attack = embeddings_attacked.shape[0] // embeddings_positive.shape[0]
        num_attacks_per_variant = [N_per_attack] * embeddings_positive.shape[0]

    for dim in range(4):
        threshold = offset_thresholds_per_dim[dim]

        # 只选择超过阈值的变体
        meaningful_mask = torch.abs(offset_before[:, dim]) >= threshold  # (N,)
        num_meaningful = torch.sum(meaningful_mask).item()

        if num_meaningful == 0:
            continue

        meaningful_dim_count += 1

        # 计算这个维度的损失
        dim_loss = torch.tensor(0.0, device=s_positive.device)

        # 构建attack_idx到variant_idx的映射
        attack_idx = 0
        for var_idx in range(embeddings_positive.shape[0]):
            num_attacks = num_attacks_per_variant[var_idx]
            start_idx = attack_idx
            end_idx = attack_idx + num_attacks

            # 如果这个变体超过阈值，计算损失
            if meaningful_mask[var_idx]:
                # 维度级别的数据
                offset_bef = offset_before[var_idx, dim]  # scalar
                offset_att = offset_after[start_idx:end_idx, dim]  # (num_attacks,)

                # 符号保持检查
                product = offset_bef * offset_att
                if len(offset_att) > 0:
                    dim_loss = dim_loss + F.relu(-product).mean()

            attack_idx = end_idx

        # 平均到这个维度的变体数
        dim_loss = dim_loss / num_meaningful
        total_loss = total_loss + dim_loss

    # 除以有意义的维度数
    if meaningful_dim_count > 0:
        return total_loss / meaningful_dim_count
    else:
        return torch.zeros(1, device=s_positive.device, requires_grad=True)


def process_one_code(task):
    """
    处理单个代码的训练数据（用于并发处理）
    """
    run_id, code, base_config, num_attacks_subsample = task

    try:
        # 在每个进程中独立加载模型（避免跨进程共享问题）
        model, tokenizer = load_best_model(base_config['model_dir'])
        device = next(model.parameters()).device

        # 获取变体数量
        num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
        num_workers = base_config.get('embedding', {}).get('num_workers', 16)
        batch_size_for_parallel = base_config.get('embedding', {}).get('batch_size_for_parallel', 4)

        # 使用与测试时相同的变体生成方式
        try:
            cands, stats = build_candidates_test_like(
                code,
                K=num_variants,
                num_workers=num_workers,
                batch_size_for_parallel=batch_size_for_parallel
            )
            # 过滤掉无效变体
            variants = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != code.strip()]
            # 如果生成的变体数量不足，用原代码填充
            while len(variants) < num_variants:
                variants.append(code)
            variants = variants[:num_variants]  # 确保正好num_variants个
        except Exception as e:
            variants = [code] * num_variants

        # 编码变体（只编码变体，不包含原代码）
        embeddings_positive = embed_codes(
            model, tokenizer, variants,
            batch_size=base_config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (num_variants, 768)

        # 生成攻击版本：使用MutableAST变换
        from java_augmentor import JavaCodeAugmentor

        java_augmentor = JavaCodeAugmentor()
        attacked_codes = []
        num_attacks_per_variant = []

        # 对每个变体生成MutableAST攻击
        for i in range(num_variants):
            variant_attacks = generate_attacks_via_mutable_ast(
                variants[i],
                java_augmentor
            )
            attacked_codes.extend(variant_attacks)
            num_attacks_per_variant.append(len(variant_attacks))

        # 编码攻击版本
        embeddings_attacked = embed_codes(
            model, tokenizer, attacked_codes,
            batch_size=base_config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (num_variants * num_attacks_per_variant, 768)

        # embed_codes 已经返回numpy数组，直接使用
        data = {
            'embedding_original': embeddings_positive[0],  # 第一个变体作为代表
            'embeddings_variants': embeddings_positive,  # 已经是numpy数组
            'embeddings_attacked': embeddings_attacked,   # 已经是numpy数组
            'num_attacks_per_variant': num_attacks_per_variant,  # 每个变体的实际攻击数
            'target_bits': base_config['embedding']['bits']  # 目标比特
        }

        # 清理GPU内存
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {'run_id': run_id, 'success': True, 'data': data, 'error': None}

    except Exception as e:
        # 清理资源
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {'run_id': run_id, 'success': False, 'data': None, 'error': str(e)}


def load_training_data(base_config, config, concurrency=1):
    """
    加载训练数据，支持MutableAST攻击
    """
    print("加载训练数据...")

    # 加载训练代码（支持data_split配置）
    if base_config.get('data_split', {}).get('enabled', False):
        split_config = base_config['data_split']['strategy_5_training']
        num_codes = split_config['num_codes']
        start_idx = split_config['start_index']
    else:
        num_codes = base_config['num_test_codes']
        start_idx = base_config['start_index']

    print(f"  加载 {num_codes} 个代码（start_index={start_idx}）...")
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_loader import DatasetLoader
    loader = DatasetLoader(base_config)
    test_codes = loader.load_codes(num_codes, start_idx)

    # 准备任务列表
    tasks = [(run_id, code, base_config, 10)
             for run_id, code in enumerate(test_codes)]

    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        training_data_dict = {}

        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_code, task) for task in tasks]

            for future in tqdm(as_completed(futures), total=len(futures), desc="  处理代码", ncols=80):
                result = future.result()
                if result['success']:
                    training_data_dict[result['run_id']] = result['data']
                else:
                    print(f"\n[错误] 代码 {result['run_id']} 处理失败: {result['error']}")
                    # 失败时使用空数据
                    num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
                    training_data_dict[result['run_id']] = {
                        'embedding_original': np.zeros(768),
                        'embeddings_variants': np.zeros((num_variants, 768)),
                        'embeddings_attacked': np.zeros((0, 768)),
                        'num_attacks_per_variant': [0] * num_variants,
                        'target_bits': base_config['embedding']['bits']
                    }

        # 按run_id顺序排列
        training_data = [training_data_dict[i] for i in range(len(test_codes))]
    else:
        print(f"  使用串行模式")
        # 串行模式：在主进程中加载模型（避免重复加载）
        model, tokenizer = load_best_model(base_config['model_dir'])
        device = next(model.parameters()).device

        num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
        num_workers = base_config.get('embedding', {}).get('num_workers', 16)
        batch_size_for_parallel = base_config.get('embedding', {}).get('batch_size_for_parallel', 4)

        training_data = []
        for run_id in tqdm(range(len(test_codes)), desc="  处理代码", ncols=80):
            code = test_codes[run_id]

            try:
                cands, stats = build_candidates_test_like(
                    code,
                    K=num_variants,
                    num_workers=num_workers,
                    batch_size_for_parallel=batch_size_for_parallel
                )
                variants = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != code.strip()]
                while len(variants) < num_variants:
                    variants.append(code)
                variants = variants[:num_variants]
            except Exception as e:
                variants = [code] * num_variants

            embeddings_positive = embed_codes(
                model, tokenizer, variants,
                batch_size=base_config['embedding']['batch_size_for_parallel'],
                device=device
            )

            from java_augmentor import JavaCodeAugmentor

            java_augmentor = JavaCodeAugmentor()
            attacked_codes = []
            num_attacks_per_variant = []

            # 对每个变体生成MutableAST攻击
            for i in range(num_variants):
                variant_attacks = generate_attacks_via_mutable_ast(
                    variants[i],
                    java_augmentor
                )
                attacked_codes.extend(variant_attacks)
                num_attacks_per_variant.append(len(variant_attacks))

            embeddings_attacked = embed_codes(
                model, tokenizer, attacked_codes,
                batch_size=base_config['embedding']['batch_size_for_parallel'],
                device=device
            )

            # embed_codes 已经返回numpy数组，直接使用
            training_data.append({
                'embedding_original': embeddings_positive[0],
                'embeddings_variants': embeddings_positive,  # 已经是numpy数组
                'embeddings_attacked': embeddings_attacked,   # 已经是numpy数组
                'num_attacks_per_variant': num_attacks_per_variant,  # 每个变体的实际攻击数
                'target_bits': base_config['embedding']['bits']
            })

        # 清理模型
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return training_data


def train_adaptive_generator(training_data, config, device='cuda'):
    """训练自适应方向生成器"""
    print("\n开始训练...")

    # 初始化模型
    generator = AdaptiveDirectionGenerator(
        d=config['direction_generation']['d'],
        k=config['direction_generation']['k'],
        hidden_dims=config['learning']['hidden_dims'],
        use_attention=config['learning']['use_attention']
    ).to(device)

    # 优化器
    optimizer = optim.AdamW(
        generator.parameters(),
        lr=config['learning']['learning_rate'],
        weight_decay=config['learning']['weight_decay']
    )

    # 损失权重
    loss_weights = config['learning']['loss_weights']

    # 计算offset阈值（从前5个代码的等价变体）
    print("\n计算offset阈值...")
    offset_thresholds_per_dim = None

    with torch.no_grad():
        num_samples = min(5, len(training_data))
        for i in range(num_samples):
            embeddings_positive = torch.from_numpy(training_data[i]['embeddings_variants']).float().to(device)

            # 简化处理：使用mean作为简单投影
            s_positive_np = embeddings_positive.cpu().numpy()
            # 这里需要更复杂的处理，但为了简化，暂时使用mean
            s_positive_mean = np.mean(s_positive_np, axis=1, keepdims=True)

            # 简单的offset计算
            s0_np = np.median(s_positive_np, axis=0)
            offset_before = s_positive_np - s0_np

            # 计算阈值
            if offset_thresholds_per_dim is None:
                offset_thresholds_per_dim = compute_offset_thresholds(
                    offset_before,
                    quantile=config['learning'].get('threshold_quantile', 0.6)
                )

    print(f"  Offset阈值: {offset_thresholds_per_dim}")

    # 训练循环
    num_epochs = config['learning']['epochs']
    batch_size = config['learning']['batch_size']

    for epoch in range(num_epochs):
        epoch_loss = 0

        # 随机打乱数据
        indices = np.random.permutation(len(training_data))

        # 分批训练
        for i in range(0, len(training_data), batch_size):
            batch_indices = indices[i:i+batch_size]

            # 简化版本的批处理
            batch_size_actual = len(batch_indices)

            # 收集batch数据
            embeddings_original_batch = []
            for idx in batch_indices:
                embedding = torch.from_numpy(training_data[idx]['embedding_original']).float()
                embeddings_original_batch.append(embedding)

            embeddings_original = torch.stack(embeddings_original_batch).to(device)  # (B, 768)

            # 前向传播：生成方向
            W, attention = generator(embeddings_original)  # (B, 4, 768)

            # 简化的损失计算（演示版本，实际可以更复杂）
            loss = None

            # 计算基本的正交性损失
            for b in range(batch_size_actual):
                W_gram = torch.mm(W[b], W[b].t())  # (4, 4)
                I = torch.eye(4, device=device)
                batch_loss = F.mse_loss(W_gram, I)
                loss = batch_loss if loss is None else loss + batch_loss

            # 反向传播
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}")

    return generator


def main():
    parser = argparse.ArgumentParser(description="Step 1c: 训练自适应方向生成器（策略6）")
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理的进程数（默认=5）')
    parser.add_argument('--resume', action='store_true', help='加载已保存的训练数据')
    args = parser.parse_args()

    print("="*80)
    print(f"Step 1c: 训练自适应方向生成器 (Strategy 6) (concurrency={args.concurrency})")
    print("="*80)

    # 设置工作目录
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # 加载配置
    with open("dimension_strategy_comparison/configs/base_config.json", 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    with open("dimension_strategy_comparison/configs/strategy_6_adaptive.json", 'r', encoding='utf-8') as f:
        strategy_config = json.load(f)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 输出文件路径
    strategy_name = "strategy_6_adaptive"
    output_dir = Path(f"dimension_strategy_comparison/results/{strategy_name}")
    trained_generator_file = output_dir / "trained_generator.pth"
    cache_file = output_dir / "training_data.pkl"

    # 检查是否已训练完成
    if args.resume and trained_generator_file.exists():
        print("检测到已训练完成的模型:")
        print(f"  - {trained_generator_file}")
        print("\n跳过训练步骤。如需重新训练，请删除这些文件或不使用--resume参数。\n")
        print("="*80)
        print("训练已完成（跳过）")
        print("="*80)
        return

    # 尝试加载已保存的训练数据
    training_data = None
    if args.resume and cache_file.exists():
        print("尝试加载已保存的训练数据...")
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                training_data = pickle.load(f)
            print(f"[OK] 成功加载训练数据 ({len(training_data)}个代码)\n")
        except Exception as e:
            print(f"加载失败: {e}")
            training_data = None

    # 如果没有加载到数据，重新生成
    if training_data is None:
        if args.resume:
            print("未找到已保存的训练数据，将重新生成\n")

        training_data = load_training_data(
            base_config,
            strategy_config,
            concurrency=args.concurrency
        )

        # 保存训练数据
        print("\n保存训练数据...")
        output_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"[OK] 训练数据已保存: {cache_file}")

    print(f"\n训练数据: {len(training_data)} 个代码，每个100个变体+多个MutableAST攻击样本")

    # 训练
    generator = train_adaptive_generator(training_data, strategy_config, device)

    # 保存模型
    output_dir = Path("dimension_strategy_comparison/results/strategy_6_adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存生成器
    torch.save(generator.state_dict(), output_dir / "trained_generator.pth")
    print(f"\n[OK] 已保存生成器到: {output_dir / 'trained_generator.pth'}")

    # 打印最终结果
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()
