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


def analyze_image(image_path):
    # --- 1. 增强版图片处理函数 (防止坏图导致 500) ---

    def encode_image(p):
        with open(p, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def get_type(path):
        mime_type_temp, _ = mimetypes.guess_type(path)
        if mime_type_temp is None:
            mime_type_temp = 'image/jpeg'  # 默认回退到 jpeg
        return mime_type_temp

    # image_path = './dataset/sample2/img.png'

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        exit()



    # def get_type(path):
    #     mime_type_temp, _ = mimetypes.guess_type(path)
    #     if mime_type_temp is None:
    #         mime_type_temp = 'image/jpeg'  # 默认回退到 jpeg
    #     return mime_type_temp


    client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx", timeout=6000)

    token_count = 0
    first_token_time = None
    start_gen_time = None

    print("length:",len(encode_image(image_path)))

    prompt="""
    “你需要化身为高精度扫描仪。请从左往右，依次找出图中所有的蜡烛线。 对于每一根蜡烛线，请按顺序编号并提供其归一化坐标。

    <think>
        扫描步骤思考
    </think>

    输出要求:在输出中不要输出任何无关内容，务必确保编号连续，最后统计总数。
    <output>
    <ref>K1</ref><box>[...]</box> [实心绿色 | 空心红色 ]
    <ref>K2</ref><box>[...]</box> [实心绿色 | 空心红色 ]
    ... 
    </output>
    """

    result = ''
    # 记录开始处理的时间
    start_process_time = time.time()
    response = client.chat.completions.create(
        max_tokens=16192,  # 视觉任务通常需要多一点 token 输出
        stream=True,
        model="qwen3-vl",
        temperature=0.1,  # 保持低温
        # 【核心修改】加入重复惩罚
        frequency_penalty=1.5,  # 防止复读
        presence_penalty=0.1,  # 【改为0】不要惩罚话题重复，JSON需要重复Key
        stop=[
            "<|im_end|>",
            "<|im_start|>",
            "<|im_end|>",  # Qwen 标准结束符
            "<|endoftext|>",  # 通用结束符
            # "```json",  # 防止它输出完代码块后继续废话
            # "}"                # 【绝招】如果你只需要一个 JSON，可以在检测到右大括号时强制停止（需慎用，防止嵌套结构未闭合）
        ],
        # response_format={"type": "json_object"},
        messages=[
            # {
            #     "role": "system",
            #     # 使用加强版的 Prompt
            #     "content": SYSTEM_PROMPT_2
            # },

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
                        "text": prompt
                    }
                ]
            }
        ],

    )
    print("image_path:", image_path)
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
    return result


if __name__ == "__main__":
    image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_2/image4.png"
    result = analyze_image(image_path)

    # image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031/frame_000360.jpg"
    # result = analyze_image(image_path)
    print(result)