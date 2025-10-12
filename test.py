from api import compute_baseline_s0, detect_bits_for_codes

model_dir = r"D:\kyl410\XDF\Watermark4code\best_model"

codes = [
    "public int add(int a,int b){ return a+b; }",
    "public int sub(int a,int b){ return a-b; }"
]

# 1) 计算基线：返回 embeddings[N,768], directions[4,768], s0[N,4]
baseline = compute_baseline_s0(model_dir, codes, secret_key="XDF")
s0 = baseline["s0"]
dirs = baseline["directions"]

# 2) 对同一批代码进行投影与比特判定（默认 t_margin=0.10）
projections, bits = detect_bits_for_codes(model_dir, codes, directions=dirs, s0=s0, t_margin=0.10)

print("s0 shape:", s0.shape)                 # 期望 (2, 4)
print("projections shape:", projections.shape) # 期望 (2, 4)
print("bits:\n", bits)     