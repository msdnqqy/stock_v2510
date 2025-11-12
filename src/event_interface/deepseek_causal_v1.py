import ollama
import json
import re

# --- 1. 推荐模型 ---
# 14B 模型对于“遵循 CoT + JSON”指令的能力远胜 8B
# MODEL_NAME = "deepseek-r1:14b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
# MODEL_NAME = "deepseek-r1:8b"  # 确保您已导入或 'ollama pull qwen1.5:14b'
MODEL_NAME = "gemma3:27b-it-qat"  # https://ollama.com/library/gemma3

# --- 2. 示例新闻 (模拟输入) ---
NEWS_ARTICLE = """
路透社 11 月 12 日报道，由于北海主要油田“巨魔 B”平台发生意外停电，
运营商 Equinor 宣布该油田（日产 30 万桶）将临时关闭。
此消息一出，国际原油价格应声上涨，布伦特原油期货价格飙升 4.5%，
突破每桶 90 美元大关。分析师担心，此次停产将加剧全球能源供应的紧张局势。
"""


# --- 3. 构建“主提示” (Master Prompt with CoT) ---
def create_master_prompt_with_cot(article):
    # 这是我们上面设计的模板
    return f"""
        你是一个专业的财经新闻分析师。你的任务是分两步完成分析，**思考过程和最终输出都必须提供**。
        
        **步骤 1: 思考 (Think)**
        在一个 `<think>` XML 标签中，请你先进行“思维链”分析：
        1.  **摘要分析**: 识别出新闻中的所有关键主体（公司、人物）、关键事件、关键数据（数字、百分比、金额）和地点。
        2.  **因果分析**: 寻找文中的因果关系。寻找“A 导致 B”的明确表述（例如：由于...导致...，因此...，结果是...）。明确 A (事件) 和 B (主体/影响)。
        
        **步骤 2: 输出 (Output)**
        在你的思考分析完成后，请根据你在 `<think>` 标签中的分析结果，严格按照以下的 JSON 格式输出：
        
        {{
          "summary": "（根据步骤 1 的摘要分析，生成包含所有关键信息的摘要）",
          "causal_events": [
            {{
              "event": "（根据步骤 1 的因果分析，填入【事件】）",
              "subject": "（根据步骤 1 的因果分析，填入【主体】）",
              "effect": "（根据步骤 1 的因果分析，填入【影响内容】）"
            }}
          ]
        }}
        
        ### 指南
        * 如果新闻中没有因果关系，`causal_events` 字段应返回空列表 `[]`。
        * 你的最终输出**必须**以 `<think>` 标签开始，然后是 JSON 代码块。
        
        ### 新闻正文
        {article}
        
        Let's think step by step,请一步一步思考，最后给出中文答案。
        ### 分析开始:
        """


# --- 4. 执行推理并解析结果 (更新的解析逻辑) ---
def extract_causal_info_with_cot(article_text):
    print(f"--- 正在使用模型 {MODEL_NAME} 进行 CoT 分析... ---")

    prompt = create_master_prompt_with_cot(article_text)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'num_ctx': 8192,
                'temperature': 0.0  # 分析任务用 0.0
            },
        )

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


        raw_response = response['message']['content']
        # print("--- 模型回复 ---")
        # print(raw_response)
        # print("\n--- 模型回复end ---")

        # --- 新的解析逻辑 ---
        # 1. 提取 <think> 块
        think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else "No <think> block found."

        # 2. 提取 JSON 块 (它应该在 </think> 之后)
        # 我们查找第一个 { 和最后一个 }
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not json_match:
            # 备用方案：查找 ```json
            if "```json" in raw_response:
                json_string = raw_response.split("```json")[1].split("```")[0]
            else:
                raise ValueError("No JSON block found in the response.")
        else:
            json_string = json_match.group(0)

        parsed_data = json.loads(json_string)

        if parsed_data:
            print("\n--- ✅ 分析成功 ---")

            print("\n【步骤 1: 模型的“思维链”(CoT)】")
            print(think_text)

            print("\n【步骤 2: 最终 JSON 输出】")
            print("\n【目标 1：结构化摘要】")
            print(parsed_data.get("summary"))

            print("\n【目标 2：因果关系提取】")
            print(json.dumps(parsed_data.get("causal_events"), indent=2, ensure_ascii=False))

        return think_text, parsed_data

    except Exception as e:
        print(f"错误：无法解析模型输出或连接 Ollama。 {e}")
        # print(f"原始回复: {raw_response}")
        return None, None


# --- 5. 运行 ---
if __name__ == "__main__":
    think_process, extracted_data = extract_causal_info_with_cot(NEWS_ARTICLE)
