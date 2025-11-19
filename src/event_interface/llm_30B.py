import ollama
import json
import re

# 实验结果： https://www.yuque.com/g/u21187361/eakfyp/oprwk6b1p7tkkewl/collaborator/join?token=bq4jCYMk6Pre5ufL&source=doc_collaborator# 《因果关系提取-v1112-few shot版》

# --- 1. 推荐模型 ---
# 14B 模型对于“遵循 CoT + JSON”指令的能力远胜 8B
# MODEL_NAME = "deepseek-r1:14b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
# MODEL_NAME = "deepseek-r1:8b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
# MODEL_NAME = "gemma3:27b-it-qat"  # https://ollama.com/library/gemma3
# MODEL_NAME = "gemma3:12b-it-qat"  # https://ollama.com/library/gemma3
# MODEL_NAME = "qwen3:8b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
# MODEL_NAME = "qwen3:14b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
# MODEL_NAME = "qwen3:30b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
import sys
from llama_cpp import Llama



# ================= 配置区域 =================
# 把这里换成你下载的 GGUF 模型路径
# 推荐: Qwen2.5-32B-Instruct-Q4_K_M.gguf
# MODEL_PATH = "Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
# MODEL_PATH = "/home/shangong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf"
# 核心参数调优 (针对 RTX 5070 Ti 16G)
# 建议从 50 开始尝试，慢慢增加，直到不报错为止
# Qwen 32B 总层数约 64 层
import sys
# 1. 必须导入这个常量
from llama_cpp import Llama, GGML_TYPE_Q8_0

# ================= 配置区域 =================
MODEL_PATH = "/home/shangong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf"

# --- 显存策略调整 (针对 RTX 5070 Ti 16G) ---
# Qwen 30B Q4 模型大小约 18.5 GB + KV Cache (8192 context) 约 2-3 GB = 需要 ~21 GB
# 5070 Ti 只有 16 GB。
# 因此，你不能把 60 层全放进去。建议先放 35 层，剩下的用 CPU (你内存大，够快)。
N_GPU_LAYERS = 35

CONTEXT_SIZE = 8192


# ===========================================

def init_model():
    print(f"正在加载模型: {MODEL_PATH}...")
    print(f"尝试加载 GPU 层数: {N_GPU_LAYERS} (利用 RTX 5070 Ti 16G)")

    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=CONTEXT_SIZE,
            n_batch=512,
            flash_attn=True,  # 必开！大幅提升速度
            # 【修正点】这里不能用字符串，必须用导入的常量
            type_k=GGML_TYPE_Q8_0,
            type_v=GGML_TYPE_Q8_0,

            verbose=True
        )
        return llm
    except Exception as e:
        print("\n❌ 模型加载失败！")
        print(f"错误详情: {e}")
        sys.exit(1)


# # 运行测试
# if __name__ == "__main__":
#     model = init_model()
#     print("✅ 模型加载成功！")


def chat_stream(llm, prompt):
    """
    流式对话函数
    """
    messages = [
        {"role": "system", "content": "你是一个乐于助人的智能助手。"},
        {"role": "user", "content": prompt}
    ]

    # 发起推理请求
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,  # 最大输出长度
        temperature=0.7,  # 创造性
        stream=True  # <--- 开启流式输出的关键
    )

    print("\nAI 回复: ", end="", flush=True)

    # 循环获取流式块
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            content = delta['content']
            # 实时打印，flush=True 确保不缓存，立刻显示
            print(content, end="", flush=True)

    print("\n\n[生成结束]")


if __name__ == "__main__":
    # 1. 初始化
    my_llm = init_model()

    # 2. 进入循环对话
    while True:
        user_input = input("\n请输入问题 (输入 'exit' 退出): ")
        if user_input.lower() == 'exit':
            break

        if not user_input.strip():
            continue

        chat_stream(my_llm, user_input)
