"""
Step 4c: 使用LLM攻击提取水印
  - Rewrite Attack: 使用CodeLlama重写代码
  - Retrans Attack: 使用CodeLlama进行代码转移（Java→Csharp→Java等）
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.utils.math import project_embeddings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==================== LLM Model Management ====================

_llm_model_cache = None
_llm_tokenizer_cache = None


def load_llm_model(model_path):
    """加载CodeLlama模型"""
    global _llm_model_cache, _llm_tokenizer_cache

    if _llm_model_cache is not None:
        return _llm_model_cache, _llm_tokenizer_cache

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, CodeLlamaTokenizer

        # 检查设备
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # 加载分词器
        try:
            tokenizer = CodeLlamaTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        if device == "cuda":
            llm_model = LlamaForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif device == "mps":
            llm_model = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            llm_model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )

        llm_model.config.pad_token_id = tokenizer.pad_token_id
        llm_model.half()
        llm_model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            llm_model = torch.compile(llm_model)

        _llm_model_cache = llm_model
        _llm_tokenizer_cache = tokenizer

        return llm_model, tokenizer

    except Exception as e:
        print(f"[错误] 加载LLM模型失败: {e}")
        raise


def unload_llm_model():
    """卸载LLM模型释放内存"""
    global _llm_model_cache, _llm_tokenizer_cache

    if _llm_model_cache is not None:
        del _llm_model_cache
        _llm_model_cache = None

    if _llm_tokenizer_cache is not None:
        del _llm_tokenizer_cache
        _llm_tokenizer_cache = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== LLM Attack Functions ====================

def generate_rewrite_prompt(code, lang):
    """生成重写代码的提示"""
    prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n  Here is a {lang} program code. Please rewrite it to enhance its efficiency, readability, and runnability. The revised code should be of the same length (in terms of lines) as the original code. No explanation is needed; only the revised code should be provided. Here's the original code:\n```\n{code}\n```\n\n### Response:"
    )
    return prompt_template.format(lang=lang, code=code)


def generate_retrans_prompt_trans1(code, lang):
    """生成转移提示（lang→target_lang）"""
    lang_dic = {'C++': 'Rust', 'Java': 'Csharp', 'Python': 'Golang'}
    lang2 = lang_dic.get(lang, lang)

    prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n Here is a {lang} program code. Please translate it into equivalent {lang2} code. The translated code should maintain the same functionality as the original {lang} code. No explanation is needed; only the translated {lang2} code should be provided. Here's the original {lang} code:\n```\n{code}\n```\n\n### Response:"
    )
    return prompt_template.format(lang=lang, lang2=lang2, code=code), lang2


def generate_retrans_prompt_trans2(code, lang2, original_lang):
    """生成反向转移提示（target_lang→lang）"""
    prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n Here is a {lang2} program code. Please translate it into equivalent {lang} code. The translated code should maintain the same functionality as the original {lang2} code. No explanation is needed; only the translated {lang} code should be provided. Here's the original {lang2} code:\n```\n{code}\n```\n\n### Response:"
    )
    return prompt_template.format(lang=original_lang, lang2=lang2, code=code)


def llm_evaluate(prompt, llm_model, tokenizer, max_new_tokens=1024):
    """使用LLM进行评估（推理）"""
    try:
        MAX_INPUT_LEN = 4096

        # 获取设备
        device = next(llm_model.parameters()).device

        # 分词
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True, padding=False)
        input_ids = inputs["input_ids"].to(device)

        # 生成配置
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            top_k=10,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 生成
        with torch.no_grad():
            generation_output = llm_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        # 解码
        output = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        return output[0]

    except Exception as e:
        print(f"[警告] LLM推理失败: {e}")
        return None


def extract_code(text):
    """从LLM响应中提取markdown代码块（CodeWMBench方式）"""
    import re

    pattern = r'```[a-z]*\n([\s\S]*?)```|```([\s\S]*?)```'
    matches = re.findall(pattern, text, re.MULTILINE)

    extracted_codes = []
    for match in matches:
        extracted_code = match[0] if match[0] else match[1]
        if extracted_code.strip():
            extracted_codes.append(extracted_code.strip())

    return extracted_codes


def apply_rewrite_attack(code, lang, llm_model, tokenizer):
    """应用Rewrite攻击"""
    try:
        prompt = generate_rewrite_prompt(code, lang)
        response = llm_evaluate(prompt, llm_model, tokenizer)

        if response is None:
            return None

        # 第一步：去除prompt（CodeWMBench方式）
        if "### Response:" in response:
            final_output = response.split("### Response:")[1].strip()
        else:
            final_output = response.strip()

        # 第二步：提取markdown代码块
        extracted = extract_code(final_output)
        if extracted:
            final_output = extracted[0]

        # 检查是否得到有效的代码
        if final_output and len(final_output) > 0:
            return final_output
        else:
            return None

    except Exception as e:
        print(f"[警告] Rewrite攻击失败: {e}")
        return code


def apply_retrans_attack(code, lang, llm_model, tokenizer):
    """应用Retrans攻击（lang→target_lang→lang）"""
    try:
        # 第一步：lang → target_lang
        prompt1, lang2 = generate_retrans_prompt_trans1(code, lang)
        response1 = llm_evaluate(prompt1, llm_model, tokenizer)

        if response1 is None:
            return None

        # 去除prompt
        if "### Response:" in response1:
            final_output1 = response1.split("### Response:")[1].strip()
        else:
            final_output1 = response1.strip()

        # 从markdown代码块中提取代码（CodeWMBench方式）
        matches1 = extract_code(final_output1)
        try:
            code_trans = matches1[0]
        except:
            code_trans = final_output1

        if not code_trans or len(code_trans) == 0:
            return None

        # 第二步：target_lang → lang
        prompt2 = generate_retrans_prompt_trans2(code_trans, lang2, lang)
        response2 = llm_evaluate(prompt2, llm_model, tokenizer)

        if response2 is None:
            return None

        # 去除prompt
        if "### Response:" in response2:
            final_output2 = response2.split("### Response:")[1].strip()
        else:
            final_output2 = response2.strip()

        # 从markdown代码块中提取代码（CodeWMBench方式）
        matches2 = extract_code(final_output2)
        try:
            final_code = matches2[0]
        except:
            final_code = final_output1

        if final_code and len(final_code) > 0:
            return final_code
        else:
            return None

    except Exception as e:
        print(f"[警告] Retrans攻击失败: {e}")
        return code


# ==================== Extraction Logic ====================

def extract_bits_from_embedding(attacked_code_embedding, directions, true_bits, s0, threshold=0.0):
    """从被攻击代码的嵌入中提取水印位"""
    # 确保嵌入是2D数组 [N, D]
    if len(attacked_code_embedding.shape) == 1:
        attacked_code_embedding = attacked_code_embedding[np.newaxis, :]

    # 投影到watermark空间
    s = project_embeddings(attacked_code_embedding, directions)[0]

    # ✅ 计算相对于簇中心s0的偏移（修复）
    offset = s - np.array(s0)

    # ✅ 基于偏移提取位
    extracted_bits = []
    for i in range(4):
        extracted_bits.append(1 if offset[i] > threshold else 0)

    success = (extracted_bits == true_bits)
    bit_accuracy = sum(b1 == b2 for b1, b2 in zip(extracted_bits, true_bits)) / 4

    return extracted_bits, offset.tolist(), success, bit_accuracy


def process_one_run(task):
    """处理单个run的LLM攻击提取"""
    run_id, strategy_name, base_config, llm_model_path = task

    embedding_dir = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}"

    try:
        # 加载嵌入结果
        with open(f"{embedding_dir}/watermarked.java", 'r', encoding='utf-8') as f:
            watermarked_code = f.read()

        with open(f"{embedding_dir}/final.json", 'r', encoding='utf-8') as f:
            embed_result = json.load(f)

        with open(f"{embedding_dir}/selected_dimensions.json", 'r', encoding='utf-8') as f:
            dim_data = json.load(f)

        true_bits = base_config['embedding']['bits']
        directions = np.array(dim_data['directions'])

        # ✅ 加载簇中心s0
        s0 = embed_result.get('s0', None)
        if s0 is None:
            print(f"[警告] run_{run_id:04d} 缺少簇中心s0，无法进行提取")
            return {"run_id": run_id, "success": False, "error": "缺少s0簇中心", "count": 0}

        # 加载watermark模型和分词器
        watermark_model, watermark_tokenizer = load_best_model(base_config['model_dir'])
        wm_device = next(watermark_model.parameters()).device

        # 加载LLM模型
        llm_model, llm_tokenizer = load_llm_model(llm_model_path)

        results = []

        # 获取代码的语言类型
        lang = embed_result.get('lang', 'Java')

        # ===== Rewrite Attack =====
        rewritten_code = apply_rewrite_attack(watermarked_code, lang, llm_model, llm_tokenizer)

        if rewritten_code and rewritten_code != watermarked_code:
            # 嵌入并提取
            try:
                rewritten_embedding = embed_codes(watermark_model, watermark_tokenizer, [rewritten_code], device=wm_device)[0]
                extracted_bits, s_proj, success, bit_acc = extract_bits_from_embedding(
                    rewritten_embedding, directions, true_bits, s0
                )

                result_rewrite = {
                    'run_id': run_id,
                    'attack_type': 'rewrite',
                    'true_bits': true_bits,
                    'extracted_bits': extracted_bits,
                    'projection': s_proj,
                    'success': bool(success),
                    'bit_accuracy': float(bit_acc),
                    'attacked_code': rewritten_code
                }
                results.append(result_rewrite)
            except Exception as e:
                print(f"[警告] run_{run_id:04d} Rewrite提取失败: {e}")

        # ===== Retrans Attack =====
        retrans_code = apply_retrans_attack(watermarked_code, lang, llm_model, llm_tokenizer)

        # 从markdown代码块中提取代码（CodeWMBench在retrans.py保存时做的）
        if retrans_code:
            extracted = extract_code(retrans_code)
            if extracted:
                retrans_code = max(extracted, key=len)

        if retrans_code and retrans_code != watermarked_code:
            # 嵌入并提取
            try:
                retrans_embedding = embed_codes(watermark_model, watermark_tokenizer, [retrans_code], device=wm_device)[0]
                extracted_bits, s_proj, success, bit_acc = extract_bits_from_embedding(
                    retrans_embedding, directions, true_bits, s0
                )

                result_retrans = {
                    'run_id': run_id,
                    'attack_type': 'retrans',
                    'true_bits': true_bits,
                    'extracted_bits': extracted_bits,
                    'projection': s_proj,
                    'success': bool(success),
                    'bit_accuracy': float(bit_acc),
                    'attacked_code': retrans_code
                }
                results.append(result_retrans)
            except Exception as e:
                print(f"[警告] run_{run_id:04d} Retrans提取失败: {e}")

        # 保存结果
        if results:
            output_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction_llm"
            os.makedirs(output_dir, exist_ok=True)

            output_file = f"{output_dir}/run_{run_id:04d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return {"run_id": run_id, "success": True, "count": len(results)}

    except Exception as e:
        return {"run_id": run_id, "success": False, "error": str(e), "count": 0}

    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_strategy(strategy_name, base_config, llm_model_path, concurrency, resume=False):
    """处理单个策略"""
    print(f"\n处理策略: {strategy_name}")

    tasks = []
    skipped_count = 0

    for run_id in range(base_config['num_test_codes']):
        if resume:
            output_file = f"dimension_strategy_comparison/results/{strategy_name}/extraction_llm/run_{run_id:04d}.json"
            if os.path.exists(output_file):
                skipped_count += 1
                continue

        tasks.append((run_id, strategy_name, base_config, llm_model_path))

    if resume and skipped_count > 0:
        print(f"  跳过已完成: {skipped_count} 个")

    if not tasks:
        print(f"  所有任务已完成，无需处理")
        return

    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_run, task) for task in tasks]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  提取进度", ncols=80):
                try:
                    result = future.result()
                    results.append(result)
                    if not result['success']:
                        print(f"\n[错误] run_{result['run_id']:04d} 提取失败: {result.get('error', 'Unknown')}")
                except Exception as e:
                    print(f"\n[错误] 并发任务异常: {e}")
    else:
        print(f"  使用串行模式")
        results = []
        for task in tqdm(tasks, desc=f"  提取进度", ncols=80):
            result = process_one_run(task)
            results.append(result)
            if not result['success']:
                print(f"\n[错误] run_{result['run_id']:04d} 提取失败: {result.get('error', 'Unknown')}")

    success_count = sum(1 for r in results if r['success'])
    total_extractions = sum(r['count'] for r in results)
    print(f"  完成: {success_count}/{len(results)} 成功, 共 {total_extractions} 次提取")


def main():
    parser = argparse.ArgumentParser(description="Step 4c: 使用LLM攻击提取水印")
    parser.add_argument('--concurrency', type=int, default=1, help='并发进程数（默认=1，LLM模型较大建议=1）')
    parser.add_argument('--resume', action='store_true', help='断点继续')
    parser.add_argument('--llm-model', type=str, default=None,
                        help='LLM模型路径（默认查找当前项目下的models文件夹）')
    args = parser.parse_args()

    print("="*80)
    print(f"Step 4c: 使用LLM攻击提取水印 (concurrency={args.concurrency}, resume={args.resume})")
    print("="*80)

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # 确定LLM模型路径
    if args.llm_model is None:
        # 自动在项目下的models文件夹中查找CodeLlama-7b-Instruct-hf
        llm_model_path = project_root / "dimension_strategy_comparison" / "models" / "CodeLlama-7b-Instruct-hf"
        if not llm_model_path.exists():
            print(f"[错误] LLM模型不存在: {llm_model_path}")
            print(f"请先下载模型到该目录")
            return
    else:
        llm_model_path = Path(args.llm_model)

    llm_model_path = str(llm_model_path)

    # 检查PyTorch可用性
    if not TORCH_AVAILABLE:
        print("[错误] PyTorch不可用，无法运行LLM攻击")
        return

    config_path = project_root / "dimension_strategy_comparison" / "configs" / "base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    strategies = [
        "strategy_5_learned",
        "strategy_6_adaptive",
    ]

    for strategy_name in strategies:
        process_strategy(strategy_name, base_config, llm_model_path, args.concurrency, args.resume)

    # 卸载LLM模型
    unload_llm_model()

    print("\n" + "="*80)
    print("完成！LLM攻击提取结果已保存到 results/strategy_X/extraction_llm/")
    print("="*80)


if __name__ == '__main__':
    main()
