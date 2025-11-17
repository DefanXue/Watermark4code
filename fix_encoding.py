"""批量修复所有脚本的编码问题"""
import re
import os

scripts_dir = "scripts"
scripts = [
    "step1_analyze_dimensions.py",
    "step3_embed_watermarks.py",
    "step4_extract_with_attacks.py",
    "step5_analyze_results.py",
    "step6_visualize.py",
]

patterns_to_fix = [
    # 修复读取JSON时没有encoding的情况
    (r"with open\(([^,]+), 'r'\) as", r"with open(\1, 'r', encoding='utf-8') as"),
    # 修复写入JSON时没有encoding的情况
    (r"with open\(([^,]+), 'w'\) as", r"with open(\1, 'w', encoding='utf-8') as"),
]

for script in scripts:
    filepath = os.path.join(scripts_dir, script)
    if not os.path.exists(filepath):
        print(f"跳过 {script} (不存在)")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for pattern, replacement in patterns_to_fix:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 修复 {script}")
    else:
        print(f"- {script} 无需修复")

print("\n完成！")














