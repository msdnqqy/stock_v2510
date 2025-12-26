import base64
import io
import os
from re import T
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
    <ref>K1</ref><box>[...]</box> <color>红色还是绿色</color> <solid>空心还是实心</solid><tag>柱子的标号</tag>
    <ref>K2</ref><box>[...]</box> <color>红色还是绿色</color> <solid>空心还是实心</solid><tag>柱子的标号</tag>

    ... 
    </output>
    """

    prompt2="""
    Scanning the Image

I've begun analyzing the stock chart image, image6.png. I'm focusing on accurately identifying and extracting candlestick data. My initial focus is on correctly isolating each candlestick from left to right. Once I've verified my approach, I'll move on to normalizing the coordinates. I'm aiming for a highly precise and reliable scan.

Detailing Candlesticks

I've been meticulously examining the image6.png chart, focusing on individual candlestick characteristics and their associated labels. I've categorized each candlestick by color (red or green), solidity (solid or hollow), and the presence of a red number label. My current analysis reveals twelve distinct candlesticks with corresponding labels ranging from 11 to 22, each precisely identified within the chart's structure.

Defining Output Structure

I'm now focusing on formatting the output data according to the established template: <ref>Kn</ref><box>[ymin, xmin, ymax, xmax]</box> <color>...</color> <solid>...</solid><tag>...</tag>. I've extracted all the necessary details, including candlestick color, solidity, and the associated tag. The coordinates, ranging from roughly 100x100 to 950x950, are being normalized to fit the 1000x1000 image dimensions. My priority now is ensuring the output format is precise and compliant with the given structure.

Confirming Final Candlestick Count

I've revised the candlestick data to reflect image-relative coordinates. After processing, I'm verifying the final tally, ensuring a total of twelve candlesticks are represented in the output data. Now, I'm focusing on ensuring data integrity prior to final output.

Finalizing the Data

I'm now checking the solidity of the candlesticks, confirming they're all solid as observed. I've double-checked all coordinates and tag assignments, from K1 to K12. I'm satisfied that everything is correct. The final output is ready to be produced!
    """
    prompt3 = """
当你把图片和这段 Prompt 发给我时，我的处理流程如下：

视觉锚定： 在坐标系中识别出所有符合“柱状+细线”特征的物体。

逻辑对齐： 寻找这些物体下方的红色数字，并将其与视觉特征进行 1:1 绑定。

顺序校验： 按照 X 坐标的值对识别出的物体进行降序排列。

翻译输出： 将每一个物体的视觉包围框（Bounding Box）转换成归一化后的数字序列。"""


    prompt4 = """你现在是一个高精度视觉提取引擎。
任务：识别图中红色数字 11 到 22 对应的蜡烛线属性。

规则定义（必须严格遵守）：
1. 识别顺序：必须按照红色数字 11, 12, 13... 的递增顺序识别。
2. 属性判定：绿色柱子 = <color>绿色</color> <solid>实心</solid>；红色柱子 = <color>红色</color>。
3. 坐标要求：提供蜡烛线（含上下影线）的 [ymin, xmin, ymax, xmax] 的坐标。

<think>
1. 定位数字 N 的位置。
2. 寻找数字 N 垂直正上方的蜡烛线实体。
3. 测量该实体的边界框。
4. 记录颜色和填充。
</think>

输出格式要求：
直接按顺序输出结果，不要包含任何总结性废话。
<output>
<ref>K1</ref><box>[...]</box> <color>...</color> <tag>数字 N 的值</tag>
...
</output>"""
    result = ''
    # 记录开始处理的时间
    start_process_time = time.time()
    response = client.chat.completions.create(
        max_tokens=16192,  # 视觉任务通常需要多一点 token 输出
        stream=True,
        model="qwen3-vl",
        temperature=0.0,  # 保持低温
        top_p=0.1,
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
                        # "text": prompt + prompt3
                        "text": prompt4
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
    image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_2/image6.png"
    result = analyze_image(image_path)

    # image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031/frame_000360.jpg"
    # result = analyze_image(image_path)
    print(result)