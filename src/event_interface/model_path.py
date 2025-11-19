import os
from huggingface_hub import try_to_load_from_cache

repo_id = "Qwen/Qwen3-30B-A3B-GGUF"
filename = "Qwen3-30B-A3B-Q4_K_M.gguf"

# 获取缓存路径
filepath = try_to_load_from_cache(repo_id=repo_id, filename=filename)

print(f"📂 模型实际存储路径:\n{filepath}")