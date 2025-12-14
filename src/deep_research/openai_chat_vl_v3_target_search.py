from openai import OpenAI
import os
import base64
import sys
import time  # <--- 新增 1: 导入 time 模块
import mimetypes  # 引入这个库来自动判断文件类型


# 1. 定义一个函数将本地图片转为 Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 2. 设置图片路径 (根据你的实际位置修改，建议使用绝对路径以免出错)
# 假设你的 python 脚本在 stock_v2510 根目录下运行
# image_path = "/mnt/d/projects/stock_v2510/src/deep_research/qwen3vl_arc.jpg"
# image_path = '/mnt/d/projects/stock_v2510/src/deep_research/image.png'
# image_path = '/mnt/d/projects/stock_v2510/src/deep_research/image_1.png'
# image_path = '/mnt/d/projects/stock_v2510/src/deep_research/image_2.png'
# image_path = '/mnt/d/projects/stock_v2510/src/deep_research/image_target_search/img.png'
image_path = './image_target_search/img.png'
image_path_BF1 = './image_target_search/BF1.jpg'
image_path_BF2 = './image_target_search/BF2.jpg'
image_path_BF3 = './image_target_search/BF3.jpg'
image_path_BF4 = './image_target_search/BF4.jpg'

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"错误：找不到文件 {image_path}")
    exit()

# 3. 获取 Base64 字符串
base64_image = encode_image(image_path)
# 1. 自动获取 MIME 类型 (例如 image/jpeg 或 image/png)
mime_type, _ = mimetypes.guess_type(image_path)
if mime_type is None:
    mime_type = 'image/jpeg'  # 默认回退到 jpeg
print("mime_type", mime_type, "base64:", len(base64_image))



def get_type(path):
    mime_type_temp, _ = mimetypes.guess_type(image_path)
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
    model="qwen3-vl",
    messages=[
        {"role": "user", "content": [
            {"type": "text",
             "text": """<role>你是一个从业二十年且逻辑清晰的金融投资专家,擅长根据股票日线、MACD、VOL、KDJ 等指标对股票的趋势进行推断，擅长寻找买点与卖点。</role>
                    <knowledge>
                        1. 股票日线：股票的收盘价趋势图，展示股票价格的长期趋势。
                        2. MACD：移动平均线收敛 divergence 指标，用于判断股票的趋势方向。
                        3. VOL：成交量指标，用于判断股票的交易活跃度。
                        4. KDJ：随机指标，用于判断股票的超买超卖情况。
                        5、BF 法则
                    </knowledge>
                    
                    ### Input Data
                        1、图1~图4，是BF法则的表述
                        2、图5，是要推理是否满足BF法则的图
                        
                    <think>
                        让我们一步步思考：
                        1、先理解图1 到 图4 中表述的BF法则，包括 “黄昏星”、“几乎没有逆势力量”、“零零星星的非趋势K线”、“收在低位，被空头压制的趋势K线”、“”等含义
                        2、理解图4中表述的BF法则的四个必要条件
                        3、图5输入的图片蓝框的位置先下跌然后上扬，这段是否满足 BF 法则，说出推理过程
                    </think>

                    请按照下面的格式输出结论:
                    { 
                        "result": 是否符合BF结构,
                        "reason": 推理原因
                    }
                """},
            {
                "type": "image_url",
                "image_url": {
                    # 关键点：使用 data URI scheme 格式
                    "url": f"data:{get_type(image_path_BF1)};base64,{encode_image(image_path_BF1)}"
                },
            },

            {
                "type": "image_url",
                "image_url": {
                    # 关键点：使用 data URI scheme 格式
                    "url": f"data:{get_type(image_path_BF2)};base64,{encode_image(image_path_BF2)}"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    # 关键点：使用 data URI scheme 格式
                    "url": f"data:{get_type(image_path_BF3)};base64,{encode_image(image_path_BF3)}"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    # 关键点：使用 data URI scheme 格式
                    "url": f"data:{get_type(image_path_BF4)};base64,{encode_image(image_path_BF4)}"
                },
            },

            {
                "type": "image_url",
                "image_url": {
                    # 关键点：使用 data URI scheme 格式
                    "url": f"data:{get_type(image_path)};base64,{encode_image(image_path)}"
                },
            },

        ]}
    ],
    max_tokens=8192,  # 视觉任务通常需要多一点 token 输出
    stream=True
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
print("=="*20)
print("result: ", result)
print("=="*20)

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
