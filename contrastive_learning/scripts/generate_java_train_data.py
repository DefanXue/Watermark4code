#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为Java代码生成对比学习训练数据脚本
严格对标CodeWMBench的三类攻击方式：
1. 语义保持变换（变量重命名等）
2. LLM重写（使用与CodeWMBench相同的prompt）
3. 转译攻击（Java→C#→Java）
"""

import os
import sys
import json
import random
import argparse
from tqdm import tqdm

proxy_address = "http://localhost:8888"
os.environ['HTTP_PROXY'] = proxy_address
os.environ['HTTPS_PROXY'] = proxy_address


# 导入Google Generative AI
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("Error: 请安装 Google Generative AI SDK: pip install google-generativeai")
    sys.exit(1)

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入自定义模块
from contrastive_learning.java_augmentor import (
    JavaCodeAugmentor, 
    llm_rewrite_java, 
    retranslate_java,
    generate_java_training_data
)

# 安全设置（与create_test_set.py保持一致）
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}


def main():
    """主函数：根据参数生成Java训练、验证或测试数据"""
    parser = argparse.ArgumentParser(description="为Java代码生成对比学习训练数据")
    parser.add_argument("--train_input", type=str, default="../datasets/csn_java/train_filtered_code.jsonl",
                      help="训练集输入文件路径")
    parser.add_argument("--valid_input", type=str, default="../datasets/csn_java/valid_filtered_code.jsonl",
                      help="验证集输入文件路径")
    parser.add_argument("--test_input", type=str, default="../datasets/csn_java/test_filtered_code.jsonl", 
                      help="测试集输入文件路径")
    parser.add_argument("--output_dir", type=str, default="../datasets/csn_java",
                      help="输出目录")
    parser.add_argument("--splits", type=str, default="train,valid,test",
                      help="要生成的数据集类型，用逗号分隔")
    parser.add_argument("--model", type=str, default=None,
                      help="（可选）模型名称，占位不直接调用")
    parser.add_argument("--name_tag", type=str, default="",
                      help="输出文件名追加的标签（例如 gpt-5-nano）")
    parser.add_argument("--positive_ratio", type=float, default=0.7,
                      help="正样本比例")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="每个集合最大样本数，None表示不限制")
    parser.add_argument("--api_key", type=str, default=None,
                      help="主API密钥，覆盖环境变量GOOGLE_API_KEY")
    parser.add_argument("--backup_api_key", type=str, default=None,
                      help="备用API密钥1，覆盖环境变量GOOGLE_API_KEY_BACKUP")
    parser.add_argument("--backup_api_key2", type=str, default=None,
                      help="备用API密钥2，覆盖环境变量GOOGLE_API_KEY_BACKUP2")
    parser.add_argument("--backup_api_key3", type=str, default=None,
                      help="备用API密钥3，覆盖环境变量GOOGLE_API_KEY_BACKUP3")
    parser.add_argument("--parallel", action="store_true", default=True,
                      help="启用并行处理（默认开启）")
    parser.add_argument("--workers", type=int, default=48,
                      help="并行工作线程数量（推荐设置为CPU核心数的1-2倍）")
    parser.add_argument("--batch_size", type=int, default=100,
                      help="每批处理的样本数")
    parser.add_argument("--resume", action="store_true", 
                      help="从上次中断点恢复处理，而不是从头开始")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 处理API密钥设置
    if args.api_key:
        os.environ["GOOGLE_API_KEY"] = args.api_key
        print("已使用命令行参数设置主API密钥")

    if args.backup_api_key:
        os.environ["GOOGLE_API_KEY_BACKUP"] = args.backup_api_key
        print("已使用命令行参数设置备用API密钥1")

    if args.backup_api_key2:
        os.environ["GOOGLE_API_KEY_BACKUP2"] = args.backup_api_key2
        print("已使用命令行参数设置备用API密钥2")

    if args.backup_api_key3:
        os.environ["GOOGLE_API_KEY_BACKUP3"] = args.backup_api_key3
        print("已使用命令行参数设置备用API密钥3")

    # 重新加载API密钥管理器，确保新设置的环境变量生效
    from contrastive_learning.java_augmentor import api_key_manager
    api_key_manager.reload_keys()

    # 检查API密钥
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    BACKUP_API_KEY = os.environ.get("GOOGLE_API_KEY_BACKUP")
    BACKUP_API_KEY2 = os.environ.get("GOOGLE_API_KEY_BACKUP2")
    BACKUP_API_KEY3 = os.environ.get("GOOGLE_API_KEY_BACKUP3")

    if not API_KEY:
        print("Error: 未设置GOOGLE_API_KEY环境变量")
        print("请设置环境变量: export GOOGLE_API_KEY=你的API密钥")
        print("或使用命令行参数: --api_key=你的API密钥")
        sys.exit(1)

    backup_keys_count = sum(1 for k in [BACKUP_API_KEY, BACKUP_API_KEY2, BACKUP_API_KEY3] if k)
    if backup_keys_count == 0:
        print("警告: 未设置任何备用API密钥。API额度用尽时将无法继续。")
        print("如需设置备用密钥，请使用以下环境变量或命令行参数:")
        print("  - GOOGLE_API_KEY_BACKUP   (--backup_api_key)")
        print("  - GOOGLE_API_KEY_BACKUP2  (--backup_api_key2)")
        print("  - GOOGLE_API_KEY_BACKUP3  (--backup_api_key3)")
    else:
        print(f"已配置 {backup_keys_count} 个备用API密钥")
    
    '''
    # 配置Google AI（使用主API密钥）
    genai.configure(api_key=API_KEY)
    
    # 初始化模型
    print(f"初始化LLM模型：{args.model}")
    model = genai.GenerativeModel(model_name=args.model, safety_settings=SAFETY_SETTINGS)
    '''
    
    # 使用NewAPI代替Google API
    from contrastive_learning.java_augmentor import NEWAPI_MODEL_NAME
    print(f"使用NewAPI调用模型：{NEWAPI_MODEL_NAME}")
    print("请确保在java_augmentor.py的generate_with_retry函数中已设置有效的NewAPI密钥与URL")
    # 这里只是一个占位对象，实际调用由generate_with_retry函数处理
    model = {"name": NEWAPI_MODEL_NAME}
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析要生成的数据集类型
    splits = [s.strip() for s in args.splits.split(",")]
    
    # 定义不同集合的增强类型分布
    augmentation_configs = {
        "train": {
            "semantic_preserving": 0.2,
            "llm_rewrite": 0.5,
            "retranslate": 0.3
        },
        "valid": {
            "semantic_preserving": 0.2,
            "llm_rewrite": 0.5,
            "retranslate": 0.3
        },
        "test": {
            "semantic_preserving": 0.2,
            "llm_rewrite": 0.5,
            "retranslate": 0.3
        }
    }
    
    # 为各个数据集生成增强数据
    for split in splits:
        if split == "train":
            input_file = args.train_input
            output_name = "train_augmented.jsonl" if not args.name_tag else f"train_augmented__{args.name_tag}.jsonl"
            output_file = os.path.join(args.output_dir, output_name)
            pos_ratio = args.positive_ratio
        elif split == "valid":
            input_file = args.valid_input
            output_name = "valid_augmented.jsonl" if not args.name_tag else f"valid_augmented__{args.name_tag}.jsonl"
            output_file = os.path.join(args.output_dir, output_name)
            pos_ratio = args.positive_ratio - 0.1  # 验证集正样本比例略低
        elif split == "test":
            input_file = args.test_input
            output_name = "test_augmented.jsonl" if not args.name_tag else f"test_augmented__{args.name_tag}.jsonl"
            output_file = os.path.join(args.output_dir, output_name)
            pos_ratio = 0.5  # 测试集正负样本均衡
        else:
            print(f"跳过未知数据集类型：{split}")
            continue
        
        print(f"处理{split}数据集：{input_file} -> {output_file}")
        
        # 调用生成函数
        count = generate_java_training_data(
            input_file=input_file,
            output_file=output_file,
            model=model,
            split_type=split,
            positive_ratio=pos_ratio,
            augmentation_types=augmentation_configs[split],
            max_samples=args.max_samples,
            parallel=args.parallel,
            workers=args.workers,
            batch_size=args.batch_size,
            resume=args.resume  # 传递恢复参数
        )
        
        print(f"{split}数据集处理完成，生成了{count}个样本")
    
    print("所有数据集处理完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("检测到Ctrl+C，中止当前任务。已生成的数据已写入输出文件。")
        sys.exit(130) 