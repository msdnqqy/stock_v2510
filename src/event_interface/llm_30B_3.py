import sys
import time  # <--- 新增 1: 导入 time 模块
from llama_cpp import Llama, GGML_TYPE_Q8_0

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


def chat_stream(llm, prompt):
    """
    带速度统计的流式对话函数
    """

    user_prompt = f"""
    # Role
    你是一个资深的机械行业分析师和风控专家。

    # Goal
    你需要评估给定的【输入新闻】对目标公司【三一重工 (600031.SH)】的经营和股价的影响程度（权重）。

    # Context: 三一重工业务画像 (Reference Knowledge)
    在打分前，请务必参考以下公司的核心敏感点：
    1. **主营业务**：工程机械（挖掘机、混凝土机械、起重机）。
    2. **上游敏感度**：极度依赖【钢材】价格（成本占比高）、锂电池价格（电动化产品）。
    3. **下游敏感度**：极度依赖【房地产开工】、【基建投资】、【采矿业】。
    4. **宏观敏感度**：
        - **出口**：海外收入占比高，对【汇率】、【关税】、【一带一路政策】敏感。
        - **利率**：作为重资产行业，对【贷款利率/降准降息】敏感。
    5. **竞争对手**：徐工机械、中联重科、卡特彼勒 (Caterpillar)。

    
    # Chain of Thought (推理步骤)
    1. **主体检测**：新闻是否提到三一重工？如果没有，是否提到其上下游（地产、基建、钢铁）或宏观因子（利率、汇率）？
    2. **传导分析**：如果新闻发生，通过什么逻辑链条传导给三一重工？链条越短，权重越高。
    3. **量级评估**：该新闻描述的事件是“常规波动”还是“历史性变革”？

    # Output Requirements (输出要求)
        1. **score (权重)**: 0.0-1.0,
            - **0.0 (Irrelevant)**: 完全无关（如：食品、医药、娱乐新闻）。
            - **0.1 - 0.3 (Low)**: 泛行业新闻，或逻辑链条过长，影响微乎其微（如：某省份的小型环保检查）。
            - **0.4 - 0.7 (Medium)**: 行业级重磅消息，直接影响上下游供需或成本（如：钢材价格暴跌、房地产重磅救市政策、大规模基建计划）。
            - **0.8 - 1.0 (High)**: 直接提及三一重工，或针对其核心市场的颠覆性政策（如：三一重工财报发布、针对中国工程机械的关税制裁、国家级万亿特别国债直投基建）。

        2. **impact_direction (方向)**:
            - "Positive": 利好（如业绩增长、降息、原材料降价）。
            - "Negative": 利空（如加税、地产暴雷、原材料涨价）。
            - "Neutral": 中性/影响极小/多空对冲。

        3. **key_factors (关键要素)**: 提取新闻中支撑判断的**核心数据**或**具体事件**（请简练概括，提取 2-3 点）。

    # Input News
    {prompt}

    # Output Format (JSON)
        {{
            "score": <float, 0.0-1.0>,
            "reasoning": "<用一句话解释传导逻辑>",
            "category": "<宏观|行业|个股>",
            "impact_direction": "<Positive|Negative|Neutral>",
            "key_factors": [
                "<关键要素1>",
                "<关键要素2>"
            ]
        }}
    """

    messages = [
        # {"role": "system", "content": "你是一个乐于助人的智能助手。"},
        {"role": "user", "content": user_prompt}
    ]

    print("\n正在思考...", end="", flush=True)

    # 记录开始处理的时间
    start_process_time = time.time()

    # 发起推理请求
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
        stream=True
    )

    print("\rAI 回复: ", end="", flush=True)  # \r 把前面的"正在思考"覆盖掉

    token_count = 0
    first_token_time = None
    start_gen_time = None

    # 循环获取流式块
    for chunk in stream:
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



user_input_d = """
在美联储议息会议前发布的任何经济数据都在引发市场的关注。今日美国发布的周度初请失业金人数下降，表明就业情况并未恶化。虽然机构普遍预期，美联储12月降息的概率非常大，但也有机构认为，美联储仍有机会暂停降息。
　　失业数据好于预期
　　美国劳工部周三公布的数据显示，截至11月22日当周，初请失业金人数经季节调整后减少6000人，至21.6万人。经济学家此前预计，上周初请失业金人数为22.5万人。

　　美国初请失业金人数(百万人)、续请失业金人数(百万人)和基准利率走势
　　初请失业金是美国衡量劳动力市场状况的核心指标，特指统计周期内首次申请失业保险救济金的人数，由美国劳工部每周四定期发布。该指标通过追踪新失业者申领救济的动态，直接反映企业裁员趋势和就业市场短期波动，被视为经济周期的先行指标。由于本周四是感恩节假期，该报告提前一天发布。
　　此外，续请失业金人数(衡量正在领取失业金人数的指标)则在前一周微升至196万人。自9月以来，续请失业金人数总体呈上升趋势，目前仍接近疫情后劳动力市场复苏时期的水平。
　　分析认为，特朗普激进的贸易和移民政策创造了一个企业不愿裁员或雇佣更多工人的环境，导致了他们和政策制定者所说的“不雇不炒”的劳动力市场。但由于人工智能的盛行，包括亚马逊在内的大型企业正在加大裁员力度。经济学家预计，这些裁员可能会体现在明年的初请失业金数据中，不过过去申请失业金的人数并不总是随着宣布裁员而增加。
　　美联储降息概率提高
　　德国商业银行报告指出，周三公布的美国周度初请失业金数据因其他劳动力市场数据缺失而获得额外关注。但由于市场已基本消化12月降息预期，该数据不太可能引发美元大幅波动。
　　事实上，近期的就业数据没有恶化迹象。但市场普遍认为美联储12月降息的概率极大。芝加哥市场交易所美联储观察(FedWatch)工具显示，市场预计美联储有近85%的概率会降幅2.5个百分点。
　　高盛最新预计，美联储将在12月的会议上实施连续第三次降息。该行认为，通胀放缓以及劳动力市场降温，为政策制定者进一步放松货币政策提供了空间。“明年的风险倾向于进行更多次降息，因为核心通胀方面的消息一直有利，而就业市场的恶化可能难以通过我们预期的温和的周期性增长加速来遏制。”高盛还预计，美联储2026年将再降息两次，分别在3月和6月，最终将联邦基金利率降至3.00%—3.25%的区间。
　　但也有机构认为，美联储降息并非板上钉钉。
　　摩根大通就认为，美联储下月会议的决策“极为接近”，比去年9月的情况更难判断。此前的经济数据为鹰派和鸽派都提供了充足的辩论材料，无论最终决定如何都可能出现多位委员投反对票。
　　该行认为，美联储主席鲍威尔和会议纪要此前已暗示委员会希望放缓降息节奏，9月就业报告为暂停降息提供了理由。但摩根大通也提醒，可能低估了决策中的政治因素，且判断可能过度依赖鹰派非投票委员的言论。
　　摩根大通称，按照修正后的预测路径，美联储将在12月暂停降息，随后在明年1月和5月分别降息，然后进入观望期。
"""

if __name__ == "__main__":
    # 1. 初始化
    my_llm = init_model()
    chat_stream(my_llm, user_input_d)

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