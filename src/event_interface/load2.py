import ollama
import sys

# --- 1. 定义模型 ---
# 我们使用 llama 3 8B 作为示例
# DeepSeek-R1-0528-Qwen3-8B
# from https://ollama.com/library/deepseek-r1:14b
MODEL_TO_USE = "deepseek-r1:8b"


def run_inference():
    print(f"--- 正在连接 Ollama 并使用模型: {MODEL_TO_USE} ---")

    try:
        # 确保模型在本地存在
        # (这是一个好习惯，Ollama 客户端会自动检查)
        print(f"正在检查模型 '{MODEL_TO_USE}' 是否存在...")
        # 调用 ollama.list() 会触发与服务器的连接
        models_list = ollama.list()

        # 检查模型是否在列表中
        model_exists = any(model['name'] == MODEL_TO_USE for model in models_list['models'])

        if not model_exists:
            print(f"本地未找到模型 '{MODEL_TO_USE}'。")
            print("正在尝试从服务器拉取 (这可能需要几分钟)...")
            ollama.pull(MODEL_TO_USE)
            print("模型拉取成功。")
        else:
            print("模型已存在于本地。")

        prompt_xml = """
        请你按照以下的 XML 格式来回答我的问题。

        <think>
        [在这里写下你的思考过程和分析步骤]
        </think>
        <answer>
        [在这里写下你的最终答案]
        </answer>

        问题: 100 只鸡和兔子在同一个笼子里，总共有 260 只脚。请问鸡和兔子各有多少只？
        Let's think step by step,请一步一步思考，最后给出答案。
        """

        # --- 2. 准备聊天输入 ---
        messages = [
            {'role': 'system', 'content': '你是一个有帮助的助手。'},
            {'role': 'user', 'content': '为什么天空是蓝色的？请用中文简洁回答。'}
        ]

        print("\n--- 正在向模型发送请求... ---")

        # --- 3. 调用聊天 API (非流式) ---
        response = ollama.chat(
            model=MODEL_TO_USE,
            messages=messages,
            options={
                'num_ctx': 8192  # 关键：告诉 Ollama "我的上下文窗口上限是 8192"
            },
            extra_body={"enable_thinking": True},
        )

        print("--- 收到模型回复 ---")

        # --- 4. 解析并打印回复 ---
        response_content = response['message']['content']
        print(response_content)

        print("\n--- 性能数据 ---")
        total_duration_s = response.get('total_duration', 0) / 1e9
        prompt_tokens = response.get('prompt_eval_count', 0)
        response_tokens = response.get('eval_count', 0)

        print(f"总耗时: {total_duration_s:.2f} 秒")
        print(f"提示 Token: {prompt_tokens}")
        print(f"生成 Token: {response_tokens}")
        if total_duration_s > 0 and response_tokens > 0:
            print(
                f"估算速度: {response_tokens / total_duration_s :.2f} tokens/秒")  # 注意：这个速度不准，因为 total_duration 包括了提示处理

    except Exception as e:
        print(f"\n[错误] 无法连接到 Ollama 服务器或执行推理。", file=sys.stderr)
        print("请确保 Ollama 应用程序正在后台运行。", file=sys.stderr)
        print("您可以尝试在另一个终端运行 'ollama run llama3:8b' 来启动它。", file=sys.stderr)
        print(f"详细错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_inference()