import glob

# 1. 先切分视频
# video_to_frames("stock.mp4", "frames_temp", frame_interval=30)

# 2. 获取所有图片路径并排序
image_files = sorted(glob.glob("/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031/frame_*.jpg"))
for image_path in image_files:
    print(image_path)

# 3. 循环你的 LLM 代码
for img_path in image_files:
    # 调用你之前的 encode_image 和 LLM 请求逻辑...
    pass