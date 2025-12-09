from llama_cpp import Llama

# 替换为你实际的本地文件路径
# 注意：只需要指向 "split-00001" 这个文件即可，程序会自动寻找 00002
MODEL_PATH = "/home/shangong/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Thinking-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,           # 上下文长度，根据需要调整
    n_gpu_layers=20,      # 根据你的 RTX 5070Ti (16G) 调整，F16 版本很大，显存塞不下太多层
    verbose=True          # 显示加载日志，方便排错
)

# 测试一下
output = llm("Hello Qwen!", max_tokens=32)
print(output)