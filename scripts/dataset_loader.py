"""
统一的数据集加载器
"""
import json

class DatasetLoader:
    def __init__(self, config):
        self.dataset_name = config['dataset']['name']
        self.data_path = config['dataset']['paths'][self.dataset_name]

    def load_codes(self, num_codes, start_index=0):
        """加载代码"""
        codes = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_index:
                    continue
                if len(codes) >= num_codes:
                    break
                data = json.loads(line)
                codes.append(data['code'])
        return codes
