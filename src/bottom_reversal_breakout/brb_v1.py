from openai import OpenAI
import os
import base64
import sys
import time  # <--- 新增 1: 导入 time 模块
import mimetypes  # 引入这个库来自动判断文件类型
from config import *


# 1. 定义一个函数将本地图片转为 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


image_path = './dataset/sample2/img.png'

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"错误：找不到文件 {image_path}")
    exit()

few_shot_1_path = './dataset/sample1/img.png'
few_shot_1_cot = """
    {
      "step_1_context": "图表左侧显示明显且持续的下跌趋势，均线系统呈空头排列，价格处于相对低位。",
      "step_2_pattern": "在下跌后，价格进入一个黄框标识的箱体震荡区域，短期与中期均线在此处发生粘合，显示市场成本趋于一致，正在蓄势。",
      "step_3_breakout": "震荡末端出现了一根实体巨大的红色大阳线，一举突破了箱体上沿和多条均线的压制，收盘价站稳在阻力位之上。",
      "step_4_volume": "在大阳线出现的当天，下方的成交量柱（红色）剧烈放大，是前几日平均成交量的数倍，属于典型的放量突破。",
      "is_bottom_reversal": true,
      "confidence_score": 95,
      "reasoning_summary": "该图完美符合底部反转特征：下跌趋势背景 + 底部箱体蓄势 + 放量大阳线突破 + 均线金叉发散，确认趋势由跌转涨。"
    }
"""


def get_type(path):
    mime_type_temp, _ = mimetypes.guess_type(path)
    if mime_type_temp is None:
        mime_type_temp = 'image/jpeg'  # 默认回退到 jpeg
    return mime_type_temp


client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx", timeout=6000)

token_count = 0
first_token_time = None
start_gen_time = None

result = ''
# 记录开始处理的时间
start_process_time = time.time()
response = client.chat.completions.create(
    max_tokens=16192,  # 视觉任务通常需要多一点 token 输出
    stream=True,
    model="qwen3-vl",
    temperature=0.15,  # 保持低温
    # 【核心修改】加入重复惩罚
    frequency_penalty=1.5,  # 防止复读
    presence_penalty=0.1,  # 【改为0】不要惩罚话题重复，JSON需要重复Key
    stop=[
        "<|im_end|>",
        "<|im_start|>",
        "<|im_end|>",  # Qwen 标准结束符
        "<|endoftext|>",  # 通用结束符
        "```json",  # 防止它输出完代码块后继续废话
        # "}"                # 【绝招】如果你只需要一个 JSON，可以在检测到右大括号时强制停止（需慎用，防止嵌套结构未闭合）
    ],
    # response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            # 使用加强版的 Prompt
            "content": SYSTEM_PROMPT_1
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_type(few_shot_1_path)};base64,{encode_image(few_shot_1_path)}"
                    }
                },
                {
                    "type": "text",
                    "text": "分析这张图表。"
                }
            ]
        },
        {
            "role": "assistant",
            # 确保你的 few_shot_1_cot 是纯净的 JSON 字符串，没有任何Markdown或废话
            "content": few_shot_1_cot
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_type(image_path)};base64,{encode_image(image_path)}"
                    }
                },
                {
                    "type": "text",
                    "text": "参考上述示例逻辑，分析这张新图表中第三个黄色框的位置是否满足底部反转形态。直接输出JSON。"
                }
            ]
        }
    ],

)

print("回答：", end="", flush=True)
think_content = ""
for chunk in response:
    # print(chunk)
    # print('reasoning_content' , chunk.choices[0].delta)

    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
        result += chunk.choices[0].delta.content
        token_count += 1
        # 捕获第一个 token 的时间
        if first_token_time is None:
            first_token_time = time.time()
            start_gen_time = first_token_time  # 开始生成的计时起点
            # 计算首字延迟 (Time to First Token)
            ttft = first_token_time - start_process_time
    elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
        think_content += chunk.choices[0].delta.reasoning_content
        token_count += 1
        # 捕获第一个 token 的时间
        if first_token_time is None:
            first_token_time = time.time()
            start_gen_time = first_token_time  # 开始生成的计时起点
            # 计算首字延迟 (Time to First Token)
            ttft = first_token_time - start_process_time

print("\n")
print("==" * 20)
print("result: ", result)
print("==" * 20)

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
