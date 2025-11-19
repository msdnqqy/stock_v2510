import torch
import sys
import time

print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")

# 1. 基础导入测试
try:
    import flash_attn

    print(f"✅ flash-attn 库导入成功，版本: {flash_attn.__version__}")
except ImportError as e:
    print(f"❌ 无法导入 flash-attn: {e}")
    sys.exit(1)

# 2. 检查 CUDA 编译情况
try:
    # 尝试获取安装时的 CUDA 构建信息（如果可用）
    raw_version = flash_attn.__version__
    print(f"ℹ️  当前安装版本: {raw_version}")
except Exception as e:
    print(f"⚠️ 无法获取详细构建信息: {e}")

# 3. 运行功能测试 (Forward + Backward)
print("\n正在进行功能测试 (Forward + Backward)...")
device = "cuda"
dtype = torch.float16

try:
    q = torch.randn(2, 128, 8, 64, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 128, 8, 64, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 128, 8, 64, device=device, dtype=dtype, requires_grad=True)

    from flash_attn import flash_attn_func

    # Forward
    out = flash_attn_func(q, k, v)
    print("✅ Forward Pass 成功")

    # Backward
    loss = out.sum()
    loss.backward()
    print("✅ Backward Pass 成功")

except Exception as e:
    print(f"❌ 功能测试失败: {e}")
    print("这通常意味着编译版本与当前硬件或 PyTorch 版本不匹配。")
    sys.exit(1)

# 4. 简易性能冒烟测试
print("\n正在进行性能冒烟测试 (对比 SDPA)...")
BATCH, SEQ, HEADS, DIM = 4, 4096, 16, 128
q = torch.randn(BATCH, SEQ, HEADS, DIM, device=device, dtype=dtype)
k = torch.randn(BATCH, SEQ, HEADS, DIM, device=device, dtype=dtype)
v = torch.randn(BATCH, SEQ, HEADS, DIM, device=device, dtype=dtype)

# 预热
for _ in range(5):
    flash_attn_func(q, k, v)

torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    flash_attn_func(q, k, v)
torch.cuda.synchronize()
fa_time = (time.time() - start) * 1000

print(f"⏱️  Flash Attention (50 iter): {fa_time:.2f} ms")
print("如果你看到的这个时间远低于之前的 2400ms (例如在 100-300ms 之间)，说明安装修复成功！")

# --- 测试 1: 标准 PyTorch Attention (SDPA) ---
import torch.nn.functional as F
start_time = time.time()
for _ in range(50):
    # PyTorch 2.0+ 的 SDPA 可能会自动优化，但我们这里强行对比
    out_ref = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
std_time = (time.time() - start_time) * 1000
print(f"Standard Attention 耗时: {std_time:.2f} ms")