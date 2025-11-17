"""
Step 1b: 训练可学习的方向矩阵W

输入：data/dimension_analysis/run_XXXX_analysis.json（已有数据）
输出：results/strategy_5_learned/trained_W.pth
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.injection.plan import build_candidates_test_like


def compute_offset_thresholds(offset_before, quantile=0.85):
    """
    为每个维度计算offset阈值

    Args:
        offset_before: (N, 4) numpy array, 每个变体相对于簇中心的偏移
        quantile: float 分位数（0.85 = 85分位）

    Returns:
        thresholds: (4,) numpy array, 每个维度的阈值
    """
    thresholds = []
    for dim in range(offset_before.shape[1]):
        offsets_dim = np.abs(offset_before[:, dim])
        threshold = np.quantile(offsets_dim, quantile)
        thresholds.append(threshold)

    return np.array(thresholds)


def get_feasible_mutable_transforms(code, java_augmentor):
    """
    检测对某个代码可行的MutableAST变换

    Returns:
        dict: {transformer_name: [feasible_keys]}
    """
    try:
        feasible_map = java_augmentor._get_feasible_mutable_transforms(code)
        return feasible_map
    except Exception as e:
        return {}


def apply_single_mutable_transform(code, transformer_name, key, java_augmentor):
    """
    应用单个MutableAST变换

    Args:
        transformer_name: str, transformer的名称
        key: str, 变换的key

    Returns:
        str, 变换后的代码（失败则返回原代码）
    """
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
      3. 加上full_variable_rename
      4. 生成20个等价变体作为攻击
      5. 返回实际生成的攻击（不填充）

    Args:
        variant_code: str, 等价变体代码
        java_augmentor: JavaCodeAugmentor实例

    Returns:
        list of str, 攻击版本代码（实际数量）
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

    # 应用full_variable_rename
    try:
        renamed_code = java_augmentor._apply_full_variable_rename(variant_code)
        if renamed_code != variant_code:
            attacked_codes.append(renamed_code)
    except:
        pass

    # 生成20个等价变体作为攻击
    try:
        from Watermark4code.injection.plan import build_candidates_test_like

        variant_attacks, _ = build_candidates_test_like(
            variant_code,
            K=20,
            num_workers=4,
            batch_size_for_parallel=2
        )

        for v in variant_attacks:
            if isinstance(v, str) and v.strip() and v != variant_code:
                attacked_codes.append(v)
    except:
        pass

    return attacked_codes


class LearnableDirections(nn.Module):
    """可学习的方向矩阵"""
    def __init__(self, k=4, d=768):
        super().__init__()
        # 初始化：使用正交矩阵
        W_init = torch.randn(k, d)
        W_init = torch.nn.functional.normalize(W_init, p=2, dim=1)  # L2归一化每一行
        self.W = nn.Parameter(W_init)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (N, 768) 代码嵌入
        Returns:
            (N, k) 投影值
        """
        # 对W进行归一化（保持单位向量）
        W_normalized = torch.nn.functional.normalize(self.W, p=2, dim=1)
        return embeddings @ W_normalized.T


def sign_preservation_loss(W, embeddings_positive, embeddings_attacked):
    """
    符号保持损失：确保攻击后offset的符号不变

    Args:
        W: LearnableDirections模块
        embeddings_positive: (N, 768) 等价变体嵌入
        embeddings_attacked: (M, 768) 攻击后嵌入
    """
    # 投影
    s_positive = W(embeddings_positive)  # (N, k)
    s_attacked = W(embeddings_attacked)  # (M, k)

    # 计算簇中心（使用median，与测试时保持一致）
    # 统一使用numpy的median计算方式，确保训练和测试一致
    s_positive_np = s_positive.detach().cpu().numpy()  # (N, k)
    s0_np = np.array([np.median(s_positive_np[:, i]) for i in range(s_positive_np.shape[1])])  # (k,)
    s0 = torch.from_numpy(s0_np).to(s_positive.device).float()  # (k,)

    # 计算偏移
    offset_before = s_positive - s0  # (N, k)
    offset_after = s_attacked - s0   # (M, k)

    # 对于每个positive的攻击版本，计算符号保持损失
    # offset_before和offset_after应该同号：offset_before * offset_after > 0
    # 使用铰链损失：max(0, -offset_before * offset_after)
    N = embeddings_positive.shape[0]
    M_per_N = embeddings_attacked.shape[0] // N  # 每个positive的攻击数量

    loss = 0
    for i in range(N):
        # 第i个positive的攻击版本
        start_idx = i * M_per_N
        end_idx = (i + 1) * M_per_N
        offset_att_i = offset_after[start_idx:end_idx]  # (M_per_N, k)

        # 广播计算
        offset_bef_i = offset_before[i].unsqueeze(0)  # (1, k)
        product = offset_bef_i * offset_att_i  # (M_per_N, k)

        # 铰链损失
        loss += torch.relu(-product).mean()

    return loss / N


def sign_preservation_loss_selective_dims(W, embeddings_positive, embeddings_attacked, offset_thresholds_per_dim, num_attacks_per_variant=None):
    """
    改进的符号保持损失：只对超过阈值的维度计算

    Args:
        W: LearnableDirections模块
        embeddings_positive: (N, 768) 等价变体嵌入
        embeddings_attacked: (M, 768) 攻击后嵌入
        offset_thresholds_per_dim: (4,) 每个维度的阈值
        num_attacks_per_variant: list of int，每个变体的实际攻击数（如果为None，假设均匀分布）

    Returns:
        tensor, 损失值
    """
    s_positive = W(embeddings_positive)  # (N, 4)
    s_attacked = W(embeddings_attacked)  # (M, 4)

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
                    dim_loss = dim_loss + torch.relu(-product).mean()

            attack_idx = end_idx

        # 平均到这个维度的变体数
        dim_loss = dim_loss / num_meaningful
        total_loss = total_loss + dim_loss

    # 除以有意义的维度数
    if meaningful_dim_count > 0:
        return total_loss / meaningful_dim_count
    else:
        return torch.tensor(0.0, device=s_positive.device, requires_grad=True)


def cluster_separation_loss(W, embeddings_positive):
    """
    簇分离损失：鼓励等价变体在每个方向上的投影范围更大
    
    Args:
        W: LearnableDirections模块
        embeddings_positive: (N, 768) 等价变体嵌入
    """
    s_positive = W(embeddings_positive)  # (N, k)
    
    # 每个维度的范围
    s_min = s_positive.min(dim=0).values  # (k,)
    s_max = s_positive.max(dim=0).values  # (k,)
    separation = (s_max - s_min).mean()
    
    # 最大化separation = 最小化负值
    return -separation


def orthogonality_loss(W):
    """
    正交性损失：鼓励W的行向量正交
    
    Args:
        W: LearnableDirections模块
    """
    W_normalized = torch.nn.functional.normalize(W.W, p=2, dim=1)
    gram = W_normalized @ W_normalized.T  # (k, k)
    identity = torch.eye(W.W.shape[0], device=W.W.device)
    
    return torch.mean((gram - identity) ** 2)


def compactness_loss(W, embeddings_positive):
    """
    紧致性损失：鼓励等价变体在每个方向上的投影接近
    
    Args:
        W: LearnableDirections模块
        embeddings_positive: (N, 768) 等价变体嵌入
    """
    s_positive = W(embeddings_positive)  # (N, k)
    variance = torch.var(s_positive, dim=0).mean()
    return variance


def process_one_code(task):
    """
    处理单个代码的训练数据（用于并发处理）

    Args:
        task: tuple (run_id, code, base_config, num_attacks_subsample)

    Returns:
        dict: {'run_id': int, 'success': bool, 'data': dict or None, 'error': str or None}
    """
    # 在子进程中设置路径（确保能找到java_augmentor）
    srcmarker_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "SrcMarker-main", "contrastive_learning")
    if srcmarker_path not in sys.path:
        sys.path.insert(0, srcmarker_path)

    run_id, code, base_config, num_attacks_subsample = task

    try:
        # 在每个进程中独立加载模型（避免跨进程共享问题）
        model, tokenizer = load_best_model(base_config['model_dir'])
        device = next(model.parameters()).device
        
        # 初始化变体生成器
        from Watermark4code.injection.plan import build_candidates_test_like
        
        # 获取变体数量（与测试时保持一致）
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
        )  # (num_variants * num_attacks_per_variant, 768) - 已经是numpy数组
        
        # embed_codes 已经返回numpy数组，直接使用
        data = {
            'embeddings_positive': embeddings_positive,  # 已经是numpy数组
            'embeddings_attacked': embeddings_attacked,   # 已经是numpy数组
            'num_attacks_per_variant': num_attacks_per_variant  # 每个变体的实际攻击数
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
    加载训练数据（改进版），支持MutableAST攻击

    Args:
        base_config: 基础配置
        config: strategy配置（包含learning参数）
        concurrency: 并发处理的进程数（默认1，串行）

    Returns:
        list of dict: 每个元素包含
            - 'embeddings_positive': (num_variants, 768) 等价变体
            - 'embeddings_attacked': (N, 768) 攻击后嵌入（N = num_variants * actual_num_attacks）
            - 'num_attacks_per_variant': int 每个变体的实际攻击数
    """
    print("加载训练数据...")

    # 从config中获取num_attacks_subsample
    num_attacks_subsample = config['learning'].get('num_attacks_subsample', -1)
    if num_attacks_subsample == -1:
        num_attacks_subsample = 10  # 默认10个攻击

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
    tasks = [(run_id, code, base_config, num_attacks_subsample) 
             for run_id, code in enumerate(test_codes)]
    
    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        training_data_dict = {}  # 使用字典保持顺序
        
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_code, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="  处理代码", ncols=80):
                result = future.result()
                if result['success']:
                    training_data_dict[result['run_id']] = result['data']
                else:
                    print(f"\n[错误] 代码 {result['run_id']} 处理失败: {result['error']}")
                    # 失败时使用空数据（避免训练中断）
                    num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
                    training_data_dict[result['run_id']] = {
                        'embeddings_positive': np.zeros((num_variants, 768)),
                        'embeddings_attacked': np.zeros((0, 768)),
                        'num_attacks_per_variant': [0] * num_variants
                    }
        
        # 按run_id顺序排列
        training_data = [training_data_dict[i] for i in range(len(test_codes))]
    else:
        print(f"  使用串行模式")
        # 串行模式：在主进程中加载模型（避免重复加载）
        model, tokenizer = load_best_model(base_config['model_dir'])
        device = next(model.parameters()).device
        
        from Watermark4code.injection.plan import build_candidates_test_like
        
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
                'embeddings_positive': embeddings_positive,  # 已经是numpy数组
                'embeddings_attacked': embeddings_attacked,   # 已经是numpy数组
                'num_attacks_per_variant': num_attacks_per_variant  # 每个变体的实际攻击数
            })
        
        # 清理模型
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return training_data


def train_learned_directions(training_data, config, device='cuda'):
    """训练可学习方向矩阵"""
    print("\n开始训练...")
    
    # 初始化模型
    W = LearnableDirections(
        k=config['direction_generation']['k'],
        d=config['direction_generation']['d']
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(
        W.parameters(),
        lr=config['learning']['learning_rate'],
        weight_decay=config['learning']['weight_decay']
    )
    
    # 损失权重
    lambda_sign = config['learning']['loss_weights']['sign_preservation']
    lambda_sep = config['learning']['loss_weights']['cluster_separation']
    lambda_orth = config['learning']['loss_weights']['orthogonality']
    lambda_comp = config['learning']['loss_weights']['compactness']
    
    # 计算offset阈值（从前5个代码的等价变体）
    print("\n计算offset阈值...")
    offset_thresholds_per_dim = None

    with torch.no_grad():
        num_samples = min(5, len(training_data))
        for i in range(num_samples):
            batch_positive = torch.from_numpy(training_data[i]['embeddings_positive']).float().to(device)
            s_positive = W(batch_positive)  # (num_variants, k)

            # 计算簇中心
            s_positive_np = s_positive.detach().cpu().numpy()
            s0_np = np.array([np.median(s_positive_np[:, j]) for j in range(s_positive_np.shape[1])])
            offset_before = s_positive_np - s0_np  # (num_variants, k)

            # 计算阈值
            if offset_thresholds_per_dim is None:
                offset_thresholds_per_dim = compute_offset_thresholds(
                    offset_before,
                    quantile=config['learning']['threshold_strategy']['quantile']
                )
            else:
                # 累积多个代码的offset，取更新阈值
                offset_thresholds_per_dim = np.maximum(
                    offset_thresholds_per_dim,
                    compute_offset_thresholds(
                        offset_before,
                        quantile=config['learning']['threshold_strategy']['quantile']
                    )
                )

    print(f"  Offset阈值: {offset_thresholds_per_dim}")

    # 计算损失归一化系数
    print("\n计算损失归一化系数（预运行5个batch）...")
    scale_sign_list = []
    scale_sep_list = []
    scale_orth_list = []
    scale_comp_list = []

    with torch.no_grad():
        num_samples = min(5, len(training_data))
        for i in range(num_samples):
            batch_positive = torch.from_numpy(training_data[i]['embeddings_positive']).float().to(device)
            batch_attacked = torch.from_numpy(training_data[i]['embeddings_attacked']).float().to(device)
            num_attacks = training_data[i]['num_attacks_per_variant']

            # 使用selective_dims loss计算归一化系数
            loss_sign = sign_preservation_loss_selective_dims(
                W, batch_positive, batch_attacked, offset_thresholds_per_dim, num_attacks
            ).item()
            loss_sep = abs(cluster_separation_loss(W, batch_positive).item())
            loss_orth = orthogonality_loss(W).item()
            loss_comp = compactness_loss(W, batch_positive).item()

            scale_sign_list.append(loss_sign)
            scale_sep_list.append(loss_sep)
            scale_orth_list.append(loss_orth)
            scale_comp_list.append(loss_comp)
    
    # 使用均值作为归一化系数
    norm_sign = np.mean(scale_sign_list) + 1e-8
    norm_sep = np.mean(scale_sep_list) + 1e-8
    norm_orth = np.mean(scale_orth_list) + 1e-8
    norm_comp = np.mean(scale_comp_list) + 1e-8
    
    print(f"  符号保持归一化系数: {norm_sign:.6f}")
    print(f"  簇分离归一化系数:   {norm_sep:.6f}")
    print(f"  正交性归一化系数:   {norm_orth:.6f}")
    print(f"  紧致性归一化系数:   {norm_comp:.6f}")
    print(f"  量级比 (sep/sign):  {norm_sep/norm_sign:.1f}x")
    
    # 训练日志
    training_log = {
        'epochs': [],
        'loss_history': {
            'total': [],
            'sign_preservation': [],
            'sign_preservation_normalized': [],
            'cluster_separation': [],
            'cluster_separation_normalized': [],
            'orthogonality': [],
            'orthogonality_normalized': [],
            'compactness': [],
            'compactness_normalized': []
        },
        'normalization_scales': {
            'sign_preservation': float(norm_sign),
            'cluster_separation': float(norm_sep),
            'orthogonality': float(norm_orth),
            'compactness': float(norm_comp)
        }
    }
    
    # 训练循环
    num_epochs = config['learning']['epochs']
    batch_size = config['learning']['batch_size']
    
    for epoch in range(num_epochs):
        epoch_losses = {
            'total': 0,
            'sign_preservation': 0,
            'sign_preservation_normalized': 0,
            'cluster_separation': 0,
            'cluster_separation_normalized': 0,
            'orthogonality': 0,
            'orthogonality_normalized': 0,
            'compactness': 0,
            'compactness_normalized': 0
        }
        
        # 随机打乱数据
        indices = torch.randperm(len(training_data))
        
        # 分批训练
        for i in range(0, len(training_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # 收集batch数据
            batch_positive = []
            batch_attacked = []
            batch_num_attacks = []
            for idx in batch_indices:
                batch_positive.append(torch.from_numpy(training_data[idx]['embeddings_positive']).float())
                batch_attacked.append(torch.from_numpy(training_data[idx]['embeddings_attacked']).float())
                batch_num_attacks.extend(training_data[idx]['num_attacks_per_variant'])

            # 拼接
            embeddings_positive = torch.cat(batch_positive, dim=0).to(device)  # (batch_size*100, 768)
            embeddings_attacked = torch.cat(batch_attacked, dim=0).to(device)  # (batch_size*M, 768) M取决于实际攻击数

            # 前向传播
            optimizer.zero_grad()

            # 计算各项损失（原始值）
            loss_sign = sign_preservation_loss_selective_dims(
                W, embeddings_positive, embeddings_attacked, offset_thresholds_per_dim, batch_num_attacks
            )
            loss_sep = cluster_separation_loss(W, embeddings_positive)
            loss_orth = orthogonality_loss(W)
            loss_comp = compactness_loss(W, embeddings_positive)
            
            # 归一化损失
            loss_sign_norm = loss_sign / norm_sign
            loss_sep_norm = loss_sep / norm_sep
            loss_orth_norm = loss_orth / norm_orth
            loss_comp_norm = loss_comp / norm_comp
            
            # 总损失（使用归一化后的值）
            loss = (
                lambda_sign * loss_sign_norm +
                lambda_sep * loss_sep_norm +
                lambda_orth * loss_orth_norm +
                lambda_comp * loss_comp_norm
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失（原始值和归一化值）
            epoch_losses['total'] += loss.item()
            epoch_losses['sign_preservation'] += loss_sign.item()
            epoch_losses['sign_preservation_normalized'] += loss_sign_norm.item()
            epoch_losses['cluster_separation'] += loss_sep.item()
            epoch_losses['cluster_separation_normalized'] += loss_sep_norm.item()
            epoch_losses['orthogonality'] += loss_orth.item()
            epoch_losses['orthogonality_normalized'] += loss_orth_norm.item()
            epoch_losses['compactness'] += loss_comp.item()
            epoch_losses['compactness_normalized'] += loss_comp_norm.item()
        
        # 平均损失
        num_batches = (len(training_data) + batch_size - 1) // batch_size
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # 记录日志
        training_log['epochs'].append(epoch)
        for key in epoch_losses:
            training_log['loss_history'][key].append(epoch_losses[key])
        
        # 打印进度（显示归一化后的值）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={epoch_losses['total']:.4f} "
                  f"[归一化后: Sign={epoch_losses['sign_preservation_normalized']:.4f}, "
                  f"Sep={abs(epoch_losses['cluster_separation_normalized']):.4f}, "
                  f"Orth={epoch_losses['orthogonality_normalized']:.4f}, "
                  f"Comp={epoch_losses['compactness_normalized']:.4f}]")
    
    return W, training_log


def main():
    parser = argparse.ArgumentParser(description="Step 1b: 训练可学习方向矩阵（支持并发）")
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理的进程数（默认=5）')
    parser.add_argument('--resume', action='store_true', help='加载已保存的训练数据')
    args = parser.parse_args()
    
    print("="*80)
    print(f"Step 1b: 训练可学习方向矩阵 (concurrency={args.concurrency})")
    print("="*80)
    
    # 设置工作目录
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # 加载配置
    with open("dimension_strategy_comparison/configs/base_config.json", 'r', encoding='utf-8') as f:
        base_config = json.load(f)
    
    with open("dimension_strategy_comparison/configs/strategy_5_learned.json", 'r', encoding='utf-8') as f:
        strategy_config = json.load(f)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 输出文件路径
    strategy_name = "strategy_5_learned"
    output_dir = Path(f"dimension_strategy_comparison/results/{strategy_name}")
    trained_W_file = output_dir / "trained_W.pth"
    training_log_file = output_dir / "training_log.json"
    cache_file = output_dir / "training_data.pkl"

    # 检查是否已训练完成
    if args.resume and trained_W_file.exists() and training_log_file.exists():
        print("检测到已训练完成的模型:")
        print(f"  - {trained_W_file}")
        print(f"  - {training_log_file}")
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
    W, training_log = train_learned_directions(training_data, strategy_config, device)
    
    # 保存模型
    output_dir = Path("dimension_strategy_comparison/results/strategy_5_learned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存W矩阵
    W_normalized = torch.nn.functional.normalize(W.W, p=2, dim=1)
    torch.save(W_normalized.cpu().detach(), output_dir / "trained_W.pth")
    print(f"\n[OK] 已保存W矩阵到: {output_dir / 'trained_W.pth'}")
    
    # 保存训练日志
    with open(output_dir / "training_log.json", 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
    print(f"[OK] 已保存训练日志到: {output_dir / 'training_log.json'}")
    
    # 打印最终结果
    print("\n" + "="*80)
    print("训练完成！")
    print(f"最终损失: {training_log['loss_history']['total'][-1]:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()

