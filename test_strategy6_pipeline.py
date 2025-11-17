#!/usr/bin/env python3
"""
Strategy 6 完整测试脚本
从Step 1到Step 5运行完整的水印嵌入和提取流程

支持功能：
- --resume: 断点继续（对有此功能的步骤）
- --concurrency: 并发处理进程数
- --test-mode: 快速测试模式（用少量数据）
- --skip-steps: 跳过某些步骤
- --verbose: 详细输出
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 设置项目路径
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dimension_strategy_comparison"))


class TestPipeline:
    """Strategy 6完整测试流程管理器"""

    def __init__(self, args):
        self.args = args
        self.project_root = project_root
        self.scripts_dir = project_root / "dimension_strategy_comparison" / "scripts"
        self.configs_dir = project_root / "dimension_strategy_comparison" / "configs"
        self.results_dir = project_root / "dimension_strategy_comparison" / "results" / "strategy_6_adaptive"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 测试报告
        self.report = {
            'start_time': datetime.now().isoformat(),
            'steps': {},
            'metrics': {},
            'errors': [],
            'summary': {}
        }

        # 设置编码以支持Windows
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # 时间统计
        self.step_times = {}

        # 加载配置
        self.load_configs()

    def load_configs(self):
        """加载配置文件"""
        base_config_path = self.configs_dir / "base_config.json"
        strategy_config_path = self.configs_dir / "strategy_6_adaptive.json"

        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)

        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            self.strategy_config = json.load(f)

        if self.args.verbose:
            print(f"[OK] 配置加载成功")
            print(f"  - Base config: {base_config_path}")
            print(f"  - Strategy config: {strategy_config_path}")

    def run_step(self, step_num, script_name, step_name, extra_args=None):
        """运行单个步骤"""
        if f"step{step_num}" in self.args.skip_steps:
            print(f"\n[SKIP] 跳过 Step {step_num}: {step_name}")
            return True

        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            error_msg = f"脚本不存在: {script_path}"
            self.report['errors'].append(f"Step {step_num}: {error_msg}")
            print(f"[FAIL] Step {step_num} ({step_name}): {error_msg}")
            return False

        print(f"\n{'='*80}")
        print(f"Step {step_num}: {step_name}")
        print(f"{'='*80}")

        # 构建命令
        cmd = [sys.executable, str(script_path)]

        # 添加通用参数
        if extra_args:
            cmd.extend(extra_args)

        # 根据步骤添加特定参数
        if step_num in [1, 1, 3, 4]:  # step1c, step3, step4支持concurrency
            cmd.extend(['--concurrency', str(self.args.concurrency)])

        if step_num in [1, 3, 4]:  # step1c, step3, step4支持resume
            if self.args.resume:
                cmd.append('--resume')

        if step_num in [2, 3, 4, 5]:  # step2, step3, step4, step5仅处理strategy_6_adaptive
            cmd.extend(['--strategy', 'strategy_6_adaptive'])

        if step_num == 5:  # step5指定输出目录
            cmd.extend(['--output-dir', 'analysis/strategy_6_adaptive'])

        if self.args.verbose:
            print(f"执行命令: {' '.join(cmd)}")

        # 执行步骤
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                timeout=43200  # 6小时超时
            )

            elapsed_time = time.time() - start_time
            self.step_times[f"step{step_num}"] = elapsed_time

            if result.returncode == 0:
                print(f"[OK] Step {step_num} 完成 (耗时: {elapsed_time:.1f}s)")
                self.report['steps'][f"step{step_num}"] = {
                    'name': step_name,
                    'status': 'success',
                    'duration': elapsed_time
                }
                return True
            else:
                error_msg = f"返回码: {result.returncode}"
                self.report['errors'].append(f"Step {step_num}: {error_msg}")
                print(f"[FAIL] Step {step_num} 失败: {error_msg}")
                self.report['steps'][f"step{step_num}"] = {
                    'name': step_name,
                    'status': 'failed',
                    'duration': elapsed_time,
                    'error': error_msg
                }
                return False

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            error_msg = "执行超时"
            self.report['errors'].append(f"Step {step_num}: {error_msg}")
            print(f"[FAIL] Step {step_num} 超时 (> 3600s)")
            return False

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            self.report['errors'].append(f"Step {step_num}: {error_msg}")
            print(f"[FAIL] Step {step_num} 异常: {error_msg}")
            return False

    def verify_outputs(self):
        """验证各步骤的输出文件"""
        print(f"\n{'='*80}")
        print("验证输出文件")
        print(f"{'='*80}")

        verification = {
            'step1c': ['trained_generator.pth'],
            'step2': ['selected_dimensions.json'],
            'step3': ['embedded_watermarks.json'],
            'step4': ['extraction_results.json'],
            'step5': ['analysis_report.json']
        }

        all_valid = True
        for step, files in verification.items():
            if step in self.args.skip_steps:
                continue

            for file in files:
                file_path = self.results_dir / file

                if file_path.exists():
                    print(f"  [OK] {step}/{file}")
                else:
                    print(f"  [FAIL] {step}/{file} (不存在)")
                    all_valid = False

        return all_valid

    def collect_metrics(self):
        """收集性能指标"""
        print(f"\n{'='*80}")
        print("性能指标")
        print(f"{'='*80}")

        metrics = {
            'total_time': sum(self.step_times.values()),
            'step_times': self.step_times,
            'training_codes': self.base_config.get('data_split', {}).get('strategy_5_training', {}).get('num_codes', 200),
            'test_codes': self.base_config.get('data_split', {}).get('embedding_and_testing', {}).get('num_codes', 30),
        }

        # 尝试加载训练日志
        training_log_path = self.results_dir / "training_log.json"
        if training_log_path.exists():
            with open(training_log_path, 'r', encoding='utf-8') as f:
                training_log = json.load(f)
                if 'loss_history' in training_log and 'total' in training_log['loss_history']:
                    metrics['final_loss'] = training_log['loss_history']['total'][-1]

        # 尝试加载提取结果
        extraction_results_path = self.results_dir / "extraction_results.json"
        if extraction_results_path.exists():
            with open(extraction_results_path, 'r', encoding='utf-8') as f:
                extraction_results = json.load(f)
                if 'overall_accuracy' in extraction_results:
                    metrics['extraction_accuracy'] = extraction_results['overall_accuracy']

        self.report['metrics'] = metrics

        # 打印指标
        print(f"\n训练配置:")
        print(f"  - 训练代码数: {metrics['training_codes']}")
        print(f"  - 测试代码数: {metrics['test_codes']}")
        print(f"  - 每代码变体数: 100")

        print(f"\n执行时间:")
        for step, elapsed in sorted(self.step_times.items()):
            print(f"  - {step}: {elapsed:.1f}s")
        print(f"  - 总耗时: {metrics['total_time']:.1f}s ({metrics['total_time']/60:.1f} min)")

        if 'final_loss' in metrics:
            print(f"\n训练结果:")
            print(f"  - 最终损失: {metrics['final_loss']:.6f}")

        if 'extraction_accuracy' in metrics:
            print(f"\n水印提取:")
            print(f"  - 提取准确率: {metrics['extraction_accuracy']:.2%}")

    def generate_report(self):
        """生成最终报告"""
        print(f"\n{'='*80}")
        print("测试报告生成")
        print(f"{'='*80}")

        self.report['end_time'] = datetime.now().isoformat()

        # 确定整体状态
        failed_steps = [s for s, info in self.report['steps'].items() if info['status'] == 'failed']
        self.report['summary']['status'] = 'failed' if failed_steps else 'success'
        self.report['summary']['failed_steps'] = failed_steps
        self.report['summary']['error_count'] = len(self.report['errors'])

        # 保存报告
        report_path = self.results_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] 测试报告已保存: {report_path}")

        # 打印摘要
        print(f"\n{'='*80}")
        if self.report['summary']['status'] == 'success':
            print("[SUCCESS] 所有测试通过！")
        else:
            print(f"[WARN] 部分步骤失败: {', '.join(failed_steps)}")

        if self.report['errors']:
            print(f"\n错误列表 ({len(self.report['errors'])}):")
            for error in self.report['errors']:
                print(f"  - {error}")

        print(f"{'='*80}\n")

    def run(self):
        """执行完整测试流程"""
        print(f"\n{'#'*80}")
        print(f"# Strategy 6 完整测试流程")
        print(f"# 开始时间: {self.report['start_time']}")
        print(f"{'#'*80}\n")

        # 运行各个步骤
        steps = [
            (1, 'step1c_train_adaptive_generator.py', '训练自适应方向生成器'),
            (2, 'step2_select_dimensions.py', '选择维度'),
            (3, 'step3_embed_watermarks.py', '嵌入水印'),
            (4, 'step4_extract_with_attacks.py', '攻击提取'),
            (5, 'step5_analyze_results.py', '分析结果')
        ]

        all_success = True
        for step_num, script_name, step_name in steps:
            # 构建step标签（step1c特殊处理）
            step_label = 'step1' if step_num == 1 and 'c' not in script_name else \
                        'step1c' if 'c' in script_name else f'step{step_num}'

            if step_label in self.args.skip_steps:
                continue

            success = self.run_step(step_num, script_name, step_name)
            if not success and not self.args.continue_on_error:
                print(f"\n[WARN] Step {step_num} 失败，停止执行")
                all_success = False
                break

        # 验证输出
        verify_success = self.verify_outputs()

        # 收集指标
        self.collect_metrics()

        # 生成报告
        self.generate_report()

        return all_success and verify_success


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Strategy 6 完整测试流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整测试
  python test_strategy6_pipeline.py

  # 快速测试模式
  python test_strategy6_pipeline.py --test-mode

  # 支持断点续传
  python test_strategy6_pipeline.py --resume

  # 跳过某些步骤
  python test_strategy6_pipeline.py --skip-steps step1,step2

  # 并发处理
  python test_strategy6_pipeline.py --concurrency 10

  # 详细输出
  python test_strategy6_pipeline.py --verbose
        """
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='并发处理的进程数（默认=5，对step1c/3/4有效）'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点继续：跳过已有结果的任务（对step3/4有效）'
    )

    parser.add_argument(
        '--skip-steps',
        type=str,
        default='',
        help='跳过的步骤，用逗号分隔（如：step1,step2）'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='快速测试模式（用少量数据）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='错误时继续执行（默认遇到错误停止）'
    )

    args = parser.parse_args()

    # 解析skip-steps
    args.skip_steps = set(s.strip() for s in args.skip_steps.split(',') if s.strip())

    return args


def main():
    """主入口函数"""
    args = parse_arguments()

    # 创建测试管道
    pipeline = TestPipeline(args)

    # 运行测试
    success = pipeline.run()

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
