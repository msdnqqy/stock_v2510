from pyexpat import model
import sys
import time  # <--- 新增 1: 导入 time 模块
from llama_cpp import Llama, GGML_TYPE_Q8_0
from config import default_user_prompt_template,user_input_d,news_context_text
import concurrent.futures

# ================= 配置区域 =================
# 你的配置保持不变
MODEL_PATH = "/home/shangong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf"
# N_GPU_LAYERS = 35
N_GPU_LAYERS = 42
CONTEXT_SIZE = 8192



def init_model():
    # ... (你的 init_model 代码保持不变) ...
    print(f"正在加载模型: {MODEL_PATH}...")
    print(f"尝试加载 GPU 层数: {N_GPU_LAYERS} (利用 RTX 5070 Ti 16G)")

    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=CONTEXT_SIZE,
            n_batch=512,
            flash_attn=True,
            type_k=GGML_TYPE_Q8_0,
            type_v=GGML_TYPE_Q8_0,
            verbose=True
        )
        return llm
    except Exception as e:
        print("\n❌ 模型加载失败！")
        print(f"错误详情: {e}")
        sys.exit(1)


my_llm = init_model()
print("模型加载成功:\t",MODEL_PATH,my_llm)


def chat_stream( prompt, prompt_template = None):
    """
    带速度统计的流式对话函数
    """
    

    messages = [
        # {"role": "system", "content": "你是一个乐于助人的智能助手。"},
        {"role": "user", "content":  prompt_template.replace('[prompt]', prompt) if prompt_template is not None else prompt}
    ]

    print("\n正在思考...", end="", flush=True)

    # 记录开始处理的时间
    start_process_time = time.time()

    # 发起推理请求
    stream = my_llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
        stream=True
    )

    print("\rAI 回复: ", end="", flush=True)  # \r 把前面的"正在思考"覆盖掉

    token_count = 0
    first_token_time = None
    start_gen_time = None

    result = ''
    # 循环获取流式块
    for chunk in stream:
        # print("chunk:", chunk)
        delta = chunk['choices'][0]['delta']

        if 'content' in delta:
            content = delta['content']

            # 捕获第一个 token 的时间
            if first_token_time is None:
                first_token_time = time.time()
                start_gen_time = first_token_time  # 开始生成的计时起点
                # 计算首字延迟 (Time to First Token)
                ttft = first_token_time - start_process_time

            print(content, end="", flush=True)
            result += content
            token_count += 1

    # 结束计时
    end_time = time.time()

    print("\n\n" + "=" * 30)

    # --- 统计计算 ---
    if token_count > 0 and start_gen_time:
        # 纯生成耗时 (扣除首字等待时间)
        gen_duration = end_time - start_gen_time
        # 首字延迟
        ttft = first_token_time - start_process_time

        # 计算速度 (Tokens Per Second)
        # 防止除以0 (虽然不太可能)
        speed = token_count / gen_duration if gen_duration > 0 else 0

        print(f"📊 统计报告:")
        print(f"   - 生成长度: {token_count} tokens")
        print(f"   - 首字延迟 (TTFT): {ttft:.2f} s (预处理耗时)")
        print(f"   - 生成耗时: {gen_duration:.2f} s")
        print(f"   - 平均速度: \033[1;32m{speed:.2f} tokens/s\033[0m")  # 绿色高亮显示速度
    else:
        print("未生成有效内容。")
    print("=" * 30 + "\n")
    result = result.strip()
    return result