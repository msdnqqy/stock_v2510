import sys
import time  # <--- 新增 1: 导入 time 模块
from llama_cpp import Llama, GGML_TYPE_Q8_0
from config import default_user_prompt_template,user_input_d,news_context_text
import concurrent.futures
import json

# ================= 配置区域 =================
# 你的配置保持不变
MODEL_PATH = "/home/shangong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GGUF/snapshots/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf"
# N_GPU_LAYERS = 35
N_GPU_LAYERS = 42
CONTEXT_SIZE = 8192


# ===========================================

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


def chat_stream(llm, prompt, prompt_template = None):
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
    stream = llm.create_chat_completion(
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


# ================= 工具函数：安全的 JSON 解析 =================
def clean_and_parse_json(content):
    """尝试清理 markdown 标记并解析 JSON"""
    try:
        # 去掉 ```json 和 ``` 以及首尾空白
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ JSON 解析警告: {e}")
        return None


# ================= 模块 1: 带反思的规划者 (Reflective Planner) =================

def generate_initial_plan(topic):
    """第一步：生成初稿"""
    print(f"💡 [Draft] 正在起草初步计划: {topic} ...")
    prompt = f"""
    用户想研究的主题是：“{topic}”。
    请列出 3-5 个核心子问题，帮助全面理解这个主题。
    只返回 JSON 字符串列表，例如：["问题1", "问题2"]
    """
    # response = client.chat.completions.create(
    #     model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.7
    # )
    result = chat_stream(my_llm, prompt)
    return clean_and_parse_json(result)

def refine_plan(topic, initial_plan):
    """第二步：反思与修订 (Critique & Refine)"""
    print(f"🤔 [Refine] 正在反思并优化计划...")
    
    prompt = f"""
    你是一个资深的主编。
    用户的研究主题是：“{topic}”。
    
    这是初级研究员提出的初步搜索计划：
    {json.dumps(initial_plan, ensure_ascii=False)}
    
    请你批判性地审视这个计划：
    1. 是否有遗漏的关键角度？
    2. 是否有重复的内容？
    3. 问题的颗粒度是否合适？(不要太宽泛，也不要太细节)
    
    请给出一个**修订后**的、更完美的子问题列表。
    保持 JSON 列表格式输出。
    """
    
    # response = client.chat.completions.create(
    #     model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.5
    # )
    
    # refined_plan = clean_and_parse_json(response.choices[0].message.content)
    
    result = chat_stream(my_llm, prompt)
    refined_plan= clean_and_parse_json(result)

    if refined_plan:
        print(f"✨ [Refine] 计划已优化，从 {len(initial_plan)} 个任务调整为 {len(refined_plan)} 个。")
        return refined_plan
    else:
        print("⚠️ 优化解析失败，使用原计划。")
        return initial_plan


def get_research_plan(topic):
    """规划总入口"""
    # 1. 起草
    draft = generate_initial_plan(topic)
    if not draft: return [topic] # 兜底
    
    # 2. 修订
    final_plan = refine_plan(topic, draft)
    return final_plan



# ================= 模块 2: 执行者 (Worker) =================

def execute_step(sub_question):
    """单线程工作单元：搜索 -> 总结"""
    try:
        print(f"🔍 [Search] 开始搜: {sub_question}")
        
        # TODO：搜索 (Tavily)
        # search_result = tavily.search(query=sub_question, search_depth="advanced", max_results=3)
        # context_text = "\n".join([f"- 来源: {r['url']}\n  内容: {r['content']}" for r in search_result['results']])
        
        # 总结 (Qwen)
        print(f"📖 [Read] 正在阅读并总结: {sub_question} ...")
        summary_prompt = f"""
        针对问题：“{sub_question}”
        请阅读以下联网搜索到的原始数据，写一段结构清晰的笔记（约 300 字）。
        笔记必须包含具体的**数据、日期、人名或关键事实**。不要写废话。
        
        原始数据：
        {news_context_text}
        """
        
        response = chat_stream(my_llm, summary_prompt)
        #  client.chat.completions.create(
        #     model=MODEL_NAME, messages=[{"role": "user", "content": summary_prompt}], temperature=0.3
        # )
        return f"### 子课题：{sub_question}\n{response}\n"
        
    except Exception as e:
        print(f"❌ 任务失败: {sub_question}, 错误: {e}")
        return f"### 子课题：{sub_question}\n(该部分搜索失败)\n"

# ================= 模块 3: 整合者 (Writer) =================

def write_final_report(topic, all_notes):
    print(f"✍️ [Write] 正在撰写最终报告...")
    prompt = f"""
    你是一名顶级行业分析师。请基于以下调研笔记，写一份关于“{topic}”的深度报告。
    
    调研笔记：
    {all_notes}
    
    写作要求：
    1. 标题必须专业，使用 Markdown 格式。
    2. 开头先给出“核心结论 (Executive Summary)”。
    3. 正文逻辑分层，引用笔记中的数据支持你的观点。
    4. 语气客观、专业。
    """
    
    response = chat_stream(my_llm, prompt)
    #  client.chat.completions.create(
    #     model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.5
    # )
    return response

# ================= 主流程 =================

def run_deep_research_v2(topic):
    print(f"🚀 启动 Deep Research V2 (含反思与并发)...")
    
    # 1. 智能规划 (含反思)
    plan = get_research_plan(topic)
    print(f"📋 最终执行清单: {plan}")
    
    # 2. 并发执行 (大大提升速度)
    all_results = []
    # 使用 ThreadPoolExecutor 实现并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # 提交所有任务
        future_to_query = {executor.submit(execute_step, q): q for q in plan}
        
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                data = future.result()
                all_results.append(data)
            except Exception as exc:
                print(f"任务 {query} 抛出异常: {exc}")

    # 将所有笔记拼接
    full_notes = "\n".join(all_results)
    
    # 3. 最终写作
    report = write_final_report(topic, full_notes)
    
    print("\n" + "="*40)
    print(report)
    print("="*40)
    return report



if __name__ == "__main__":
    # 1. 初始化
    # my_llm = init_model()
    # chat_stream(my_llm, user_input_d)

    run_deep_research_v2("评估三一重工股票未来的走势")

    # 2. 进入循环对话
    while True:
        try:
            user_input = input("\n请输入问题 (输入 'exit' 退出): ")
            if user_input.lower() == 'exit':
                break

            if not user_input.strip():
                continue

            chat_stream(my_llm, user_input)
        except KeyboardInterrupt:
            print("\n\n用户中断对话")