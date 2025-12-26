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
import cv2
import numpy as np
import base64


def analyze_image(image_path):
    def encode_image(p):
        with open(p, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_type(path):
        mime_type_temp, _ = mimetypes.guess_type(path)
        if mime_type_temp is None:
            mime_type_temp = 'image/jpeg'  # 默认回退到 jpeg
        return mime_type_temp
    # --- 1. 增强版图片处理函数 (防止坏图导致 500) ---

    def prepare_for_qwen_base64(image_path, target_size=(1536, 1536)):
        """
        使用 OpenCV 读取图片，等比例缩放并填充至 target_size，返回 Base64 字符串。
        针对 Qwen3-VL 的 32x32 语义单元进行优化。
        """
        # 1. 使用 OpenCV 读取图片
        # cv2.imread 默认读取的是 BGR 格式
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"无法读取图片路径: {image_path}")

        # 2. BGR 转 RGB (视觉模型必须使用 RGB)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        h, w = rgb_img.shape[:2]
        target_w, target_h = target_size

        # 3. 计算缩放比例 (保持长宽比)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 4. 执行缩放
        # 使用 INTER_LANCZOS4 这种高质量插值，对保留影线等细小物体效果最好
        resized_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 5. 创建黑色背景画布 (确保符合 32 像素网格对齐)
        # 使用 np.zeros 创建全黑画布
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 6. 将缩放后的图片粘贴到画布左上角 (或居中)
        # 建议贴在 (0,0)，这样坐标计算最直观，减少模型坐标偏移
        canvas[0:new_h, 0:new_w] = resized_img

        # 7. 将处理后的 RGB 图片转回 BGR 以便进行存储编码 (cv2.imencode 默认 BGR)
        final_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        # 8. 转换为 Base64 编码
        # 对于 K 线图，建议使用 .png 保证无损，或者高质量 .jpg (95以上)
        retval, buffer = cv2.imencode('.png', final_bgr)
        if not retval:
            raise ValueError("图片编码失败")

        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str

    # image_path = './dataset/sample2/img.png'

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        exit()


    client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx", timeout=6000)

    token_count = 0
    first_token_time = None
    start_gen_time = None

    print("length:",len(prepare_for_qwen_base64(image_path)))

   
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

            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:{get_type(image_path)};base64,{encode_image(image_path)}"
                    #     }
                    # },
                    {
                        "type": "text",
                        # "text": prompt + prompt3
                        # "text": "读取图中所有k线图，输出其归一化坐标（0-1之间的小数），输出开盘价、收盘价、k线颜色。重新回答一下"
                        # "text":"""
                        # 你需要化身非常细心不会疏漏的扫描仪，读这张折线图，从左到右，依次输出每个点的坐标，输出格式 (x,y)...
                        # """
                        # "text":"读取图中所有的k线，输出每个k线的颜色，格式为： (序号，颜色)"
                        "text":"""
Price	Date	Close	High	Low	Open	Volume	MA5	MA20	MA60
0	2025-11-28	20.320000	20.360001	19.950001	20.059999	42242601	20.456	20.9685	21.499267
1	2025-12-01	20.120001	20.379999	19.959999	20.379999	66326195	20.378	20.8805	21.488816
2	2025-12-02	20.160000	20.260000	19.990000	20.110001	50792825	20.322	20.8030	21.476894
3	2025-12-03	20.400000	20.639999	20.100000	20.230000	63343064	20.226	20.7320	21.472426
4	2025-12-04	20.969999	20.990000	20.340000	20.410000	83791248	20.394	20.6810	21.479923
5	2025-12-05	21.299999	21.350000	20.900000	20.940001	81925980	20.590	20.6395	21.501635
6	2025-12-08	21.230000	21.629999	21.030001	21.430000	65289833	20.812	20.6445	21.515604
7	2025-12-09	21.180000	21.430000	21.040001	21.150000	41468064	21.016	20.6580	21.524628
8	2025-12-10	21.469999	21.549999	21.030001	21.170000	51653148	21.230	20.6990	21.532567
9	2025-12-11	21.090000	21.610001	21.030001	21.450001	50144032	21.254	20.7090	21.536310
10	2025-12-12	21.320000	21.389999	21.090000	21.190001	53330151	21.258	20.7420	21.542900
11	2025-12-15	21.070000	21.570000	21.059999	21.309999	44677139	21.226	20.7670	21.546802
12	2025-12-16	20.760000	21.040001	20.620001	21.040001	42009714	21.142	20.7620	21.553102
13	2025-12-17	21.070000	21.190001	20.629999	20.790001	48258808	21.062	20.7770	21.564733
14	2025-12-18	21.000000	21.180000	20.700001	20.900000	38853047	21.044	20.7840	21.569935
15	2025-12-19	21.240000	21.500000	20.950001	20.990000	52237337	21.028	20.8330	21.582097
16	2025-12-22	20.889999	21.379999	20.860001	21.299999	58059980	20.992	20.8520	21.570339
17	2025-12-23	20.680000	20.930000	20.440001	20.850000	52012927	20.976	20.8640	21.560179
18	2025-12-24	20.799999	20.900000	20.559999	20.690001	31632194	20.922	20.8600	21.551196
19	2025-12-25	20.670000	20.860001	20.620001	20.799999	32538096	20.856	20.8870	21.529852

解释这段k线的走势
                        """
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
    image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_3/image6.png"
    result = analyze_image(image_path)

    # image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031/frame_000360.jpg"
    # result = analyze_image(image_path)
    print(result)