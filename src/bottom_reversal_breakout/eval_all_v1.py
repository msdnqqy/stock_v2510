import glob
from llm_utils import analyze_image
import os
import json

# 1. 先切分视频
# video_to_frames("stock.mp4", "frames_temp", frame_interval=30)

import json

def record(new_items,filename = "dataset_1_600021_result.json"):
    # filename = 'products.json'

    # 1. 新的数据（要追加进去的）
    # new_items = [
    #     {"id": 4, "name": "Grape", "price": 8.0},
    #     {"id": 5, "name": "Melon", "price": 10.0}
    # ]

    # 2. 读取现有数据（如果文件存在）
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                # 读取旧列表
                current_list = json.load(f)
                # 确保读取出来的是列表，防止报错
                if not isinstance(current_list, list):
                    print("警告：原文件内容不是列表，将覆盖为新列表")
                    current_list = []
            except json.JSONDecodeError:
                # 文件为空或格式错误
                current_list = []
    else:
        # 文件不存在，初始化空列表
        current_list = []

    # 3. 将新数据追加到内存中的列表里
    # 使用 extend 合并两个列表，或者用 append 添加单个元素
    current_list.extend(new_items) 

    # 4. 将合并后的完整列表写回文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(current_list, f, indent=4, ensure_ascii=False)

    print(f"追加完成，当前共有 {len(current_list)} 条数据")

# 2. 获取所有图片路径并排序
image_files = sorted(glob.glob("/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031/frame_*.jpg"))
for image_path in image_files:
    print(image_path)



# 3. 循环你的 LLM 代码
for img_path in image_files:
    # 调用你之前的 encode_image 和 LLM 请求逻辑...\
    result = analyze_image(img_path)
    result_d = {'path':img_path,'result':json.loads(result)}
    # result_list = []
    # result_list.append(result_d)

    record([result_d],filename = "dataset_1_600021_result_32B_v1221.json")