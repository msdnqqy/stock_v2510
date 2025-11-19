import torch
import flash_attn

print(f"Flash Attention 版本: {flash_attn.__version__}")

# 简单的测试以确保能运行
try:
    q = torch.randn(1, 128, 32, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 128, 32, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 128, 32, 64, device='cuda', dtype=torch.float16)

    from flash_attn import flash_attn_func

    out = flash_attn_func(q, k, v)
    print("✅ Flash Attention 2 运行测试成功！")
except Exception as e:
    print(f"❌ 运行失败: {e}")


import torch
import time
from flash_attn import flash_attn_func
import torch.nn.functional as F

# 配置
BATCH = 8
SEQ_LEN = 4096  # 长序列才能看出 FA 的优势
HEADS = 32
DIM = 128
DTYPE = torch.float16
DEVICE = "cuda"

print(f"正在 RTX 5070 Ti 上准备数据 (Seq Len: {SEQ_LEN})...")

# 生成随机数据
q = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, device=DEVICE, dtype=DTYPE, requires_grad=False)
k = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, device=DEVICE, dtype=DTYPE, requires_grad=False)
v = torch.randn(BATCH, SEQ_LEN, HEADS, DIM, device=DEVICE, dtype=DTYPE, requires_grad=False)

# 预热 GPU
for _ in range(10):
    _ = F.scaled_dot_product_attention(q, k, v)
    _ = flash_attn_func(q, k, v)

torch.cuda.synchronize()

# --- 测试 1: 标准 PyTorch Attention (SDPA) ---
start_time = time.time()
for _ in range(100):
    # PyTorch 2.0+ 的 SDPA 可能会自动优化，但我们这里强行对比
    out_ref = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
std_time = (time.time() - start_time) * 1000
print(f"Standard Attention 耗时: {std_time:.2f} ms")

# --- 测试 2: Flash Attention 2 ---
start_time = time.time()
for _ in range(100):
    out_fa = flash_attn_func(q, k, v)
torch.cuda.synchronize()
fa_time = (time.time() - start_time) * 1000
print(f"Flash Attention 2  耗时: {fa_time:.2f} ms")

# --- 结果对比 ---
print(f"\n🚀 加速比: {std_time / fa_time:.2f}x")
if std_time / fa_time > 1.5:
    print("✅ 确认 Flash Attention 已生效且正在加速！")
else:
    print("⚠️ 加速不明显，可能是 PyTorch 原生 SDPA 已经自动使用了类似的优化内核。")