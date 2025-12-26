from openai import OpenAI
import base64
import io
import os
import time

from PIL import Image
from openai import OpenAI

from config import *
from openai import OpenAI
import os
import base64
import sys
import time  # <--- 新增 1: 导入 time 模块
import mimetypes  # 引入这个库来自动判断文件类型
from config import *


client = OpenAI(
    api_key="sk-no-key-required",
    base_url="http://localhost:8080/v1"
)

token_count = 0
first_token_time = None
start_gen_time = None


result = ''
# 记录开始处理的时间
start_process_time = time.time()

response = client.chat.completions.create(
    model="qwen3-vl-32b-thinking",
    stop=[
            "<|im_end|>",
            "<|im_start|>",
            "<|im_end|>",  # Qwen 标准结束符
            "<|endoftext|>",  # 通用结束符
            # "```json",  # 防止它输出完代码块后继续废话
            # "}"                # 【绝招】如果你只需要一个 JSON，可以在检测到右大括号时强制停止（需慎用，防止嵌套结构未闭合）
        ],
    
    # --- 核心参数配置（解决死循环） ---
    temperature=0.6,          # 官方建议思维模型使用 0.6
    top_p=0.95,
    max_tokens=16384,          # 给思考留出足够空间
    
    # 必须显式设为 0，防止干扰思维链逻辑词
    frequency_penalty=0.0,    
    presence_penalty=0.0,
    stream=True,
    # 额外说明：llama.cpp 的 OpenAI 接口目前主要通过 extra_body 传非标参数
    extra_body={
        "repeat_penalty": 1.0, # 彻底禁用重复惩罚
        "min_p": 0.05          # 强力过滤噪声 Token
    },
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请分析这个逻辑难题：9.11 和 9.9 哪个大？"},
                # 如果有图片，按如下格式添加：
                # {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ],
        }
    ],
)

# print(response.choices[0].message.content)

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