"""Step 1: 准备测试数据
从CSN-Java测试集中抽取50个样本，固定水印为[1,1,0,0]
"""

import json
import random
from pathlib import Path


def prepare_test_data(csn_java_path: str, output_path: str, n_samples: int = 50):
    """
    从CSN-Java抽取测试样本，固定水印为1100
    
    Args:
        csn_java_path: CSN-Java测试集路径
        output_path: 输出文件路径
        n_samples: 样本数量 (默认50)
    """
    # 读取CSN-Java测试集
    print(f"正在读取CSN-Java测试集: {csn_java_path}")
    with open(csn_java_path, 'r', encoding='utf-8') as f:
        all_samples = [json.loads(line) for line in f]
    
    print(f"总样本数: {len(all_samples)}")
    
    # 随机抽取50个样本
    random.seed(42)
    selected = random.sample(all_samples, n_samples)
    
    # 准备测试数据 - 固定水印为[1,1,0,0]
    test_samples = []
    for i, sample in enumerate(selected):
        test_sample = {
            "id": f"test#{i:03d}",
            "code": sample['code'],
            "watermark": [1, 1, 0, 0],  # 固定水印
            "docstring": ""  # CSN-Java filtered code没有docstring，后续代码搜索需要单独处理
        }
        test_samples.append(test_sample)
    
    # 保存
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"✓ 准备了 {len(test_samples)} 个测试样本")
    print(f"✓ 固定水印: [1, 1, 0, 0]")
    print(f"✓ 保存到: {output_file}")
    print(f"{'='*60}\n")
    
    return test_samples


if __name__ == '__main__':
    # CSN-Java测试集路径（绝对路径）
    script_dir = Path(__file__).parent
    csn_java_path = script_dir.parent.parent.parent.parent / "SrcMarker-main" / "contrastive_learning" / "datasets" / "csn_java" / "test_filtered_code.jsonl"
    
    # 输出路径
    output_path = script_dir.parent / "data" / "test_samples.jsonl"
    
    # 读取配置文件
    config_path = script_dir.parent / "configs" / "experiment_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        n_samples = config.get('num_samples', 50)
    else:
        # 配置文件不存在时使用默认值
        n_samples = 50
    
    # 准备数据
    prepare_test_data(csn_java_path, output_path, n_samples=n_samples)

