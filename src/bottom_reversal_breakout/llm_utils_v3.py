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
        with Image.open(p) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # 限制最大尺寸 (可选，防止显存爆显存)
            img.thumbnail((768, 768))

            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG', quality=95)  # 统一转为 JPEG
            return base64.b64encode(byte_arr.getvalue()).decode('utf-8')


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

    prompt =f"""
    你正在做 K线图识别任务。你需要化身为高精度扫描仪。图中有大于 10 个k线实体，请从左往右，依次找出图中所有的K线实体并给出如下信息，注意不要遗漏任何一条 k线实体。
    首先，识别图中所有的矩形实体部分（Body）。 然后，以每个实体为中心，找到其上影线，标记其最高点坐标为 high; 找到其下影线，标记其最低点坐标为 low。
    
    ## Output
    请以 JSON 格式输出所有识别到的 K线实体，每个实体包含：
    - Index: 实体在图中的序号（从左到右，从上到下）
    - High: 实体上方 Wick 的最高坐标
    - Low: 实体下方 Wick 的最低坐标
    - Body: 实体的矩形框坐标（[x1, y1, x2, y2]）
    - Color: Body 的矩形框的颜色，一般是红色或绿色
    - Solid: Body 的矩形框是实心还是空心
    
    示例输出：
    ```json 数组
    {
        {"Index": 1, "High": 100, "Low": 50, "Body": [10, 20, 30, 40], "Color": "红", "Solid": "空心"},
        {"Index": 2, "High": 150, "Low": 100, "Body": [40, 50, 60, 70], "Color": "绿", "Solid": "实心"}
    }
    ```
    """

    prompt1="""
    找出 MA5、MA20 的交点坐标,输出像素位置，左上角为(0,0)
    """

    prompt2="""
    Role: 你现在是一台高精度的金融图像扫描仪，专门负责从 K 线图中提取结构化数据。
    Task: 识别图中所有的 K 线实体（Candlestick Bodies），按从左到右的顺序依次编号。
    Requirements:
        坐标规范： 使用归一化坐标系 $[ymin, xmin, ymax, xmax]$，取值范围 $0$ 到 $1000$。坐标框应精准包裹蜡烛实体（Body），不含上下影线。
        属性识别： 识别每根 K 线的颜色（如红色、青色/蓝色）以及其实体状态（实心/空心）。
        空间定位： 观察背景的红色虚线网格作为水平参考，确保 Y 轴坐标的逻辑一致性。
    输出格式： 
        严禁任何解释性文字或分析。
        请直接按以下格式输出列表：
        <ref>K1</ref><box>[ymin, xmin, ymax, xmax]</box> <color>颜色</color> 
        <ref>K2</ref><box>[ymin, xmin, ymax, xmax]</box> <color>颜色</color> 
        ...（以此类推）
    Constraint: 
        请保持高度专注，不要遗漏任何一根微小的 K 线。
    """
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
        response_format={"type": "json_object"},
        messages=[

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
                        "text": prompt2
                        # "text": "描述你在图中看到的内容"
                        # "text": """是一张股票K线图，背景为黑色。图表中有多个彩色的K线（蜡烛线），包括红色和蓝色的实体部分，代表不同的价格走势。此外，还有多条不同颜色的均线（如绿色、黄色、紫色等），这些均线显示了价格的趋势变化。
                        # 你需要化身非常细心不会疏漏的扫描仪，读这张k线图，从左到右，依次输出每个红色蓝色的实体部分的中心点的坐标，输出格式 (x,y)...
                        # 注意不要漏掉任何一个k线实体，在上面得到结果后，再从右到左重新检查一遍"""
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
    image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_3/image4.png"
    result = analyze_image(image_path)

    # image_path = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031/frame_000360.jpg"
    # result = analyze_image(image_path)
    print(result)