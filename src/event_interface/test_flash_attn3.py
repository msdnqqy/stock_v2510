import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel, SDPBackend

# 硬件设定
device = "cuda"
dtype = torch.float16
BATCH, SEQ, HEADS, DIM = 4, 4096, 16, 128

q = torch.randn(BATCH, SEQ, HEADS, DIM, device=device, dtype=dtype)
k = torch.randn(BATCH, SEQ, HEADS, DIM, device=device, dtype=dtype)
v = torch.randn(BATCH, SEQ, HEADS, DIM, device=device, dtype=dtype)

print(f"PyTorch Version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n--- 🕵️‍♂️ SDPA 后端侦探 ---")

# 1. 强制使用 Flash Attention
try:
    with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        # 预热
        F.scaled_dot_product_attention(q, k, v)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(50):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        print(f"✅ SDPA (强制 FlashAttention): {start.elapsed_time(end):.2f} ms")
except RuntimeError as e:
    print(f"❌ SDPA 无法使用 FlashAttention: {e}")

# 2. 强制使用 Math (标准慢速 Attention)
try:
    with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(50):  # 跑少一点，因为真的很慢
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        print(f"🐢 SDPA (强制 Math/标准): {start.elapsed_time(end):.2f} ms")
except RuntimeError:
    print("无法使用 Math Attention")