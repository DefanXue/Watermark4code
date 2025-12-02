import os
import json
from datasets import load_dataset
from tqdm import tqdm

# 创建输出目录
output_dir = "../datasets/csn_python"
os.makedirs(output_dir, exist_ok=True)

# 加载CodeSearchNet的Python部分
print("加载CodeSearchNet Python数据集...")
dataset = load_dataset("code_search_net", "python")

# 只保留部分数据作为Mini版本(如需完整数据集，移除limit参数)
def process_split(split_name, limit=None):
    print(f"处理{split_name}数据...")
    data = dataset[split_name]
    
    # 打印样本结构以便调试（仅在第一次处理时）
    if split_name == "train":
        print("样本结构:", list(data[0].keys()))
    
    # 限制数据量创建Mini版本
    if limit and len(data) > limit:
        data = data.select(range(limit))
    
    # 将数据转换为所需格式并保存为JSONL
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data):
            # 提取代码和元数据
            processed_item = {
                'code': item['whole_func_string'],
                'language': 'python',
                'func_name': item['func_name'],
                'path': item['func_path_in_repository']
            }
            
            # 添加repo相关信息，检查不同可能的字段名
            if 'repo' in item:
                processed_item['repo'] = item['repo']
            elif 'repo_name' in item:
                processed_item['repo'] = item['repo_name']
            elif 'repository_name' in item:
                processed_item['repo'] = item['repository_name']
            
            f.write(json.dumps(processed_item) + "\n")
    
    print(f"已保存到 {output_file}，共 {len(data)} 条数据")

# 处理各个数据集分割
process_split("train", limit=40000)  # 训练集限制为4万条
process_split("validation", limit=5000)  # 验证集限制为5000条
process_split("test", limit=5000)  # 测试集限制为5000条

print("数据集准备完成!")