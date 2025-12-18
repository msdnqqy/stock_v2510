import cv2
import os


def video_to_frames(video_path, output_folder, frame_interval=1, start_frame=0):
    """
    将视频转换为图片
    :param video_path: 视频文件路径
    :param output_folder: 图片保存的文件夹
    :param frame_interval: 保存间隔 (例如 30 代表每 30 帧保存一张，即每秒一张)
    :param start_frame: 跳过起始的帧数 (默认 0)
    """

    # 1. 检查视频是否存在
    if not os.path.exists(video_path):
        print(f"❌ 错误: 找不到视频文件 {video_path}")
        return

    # 2. 创建输出文件夹 (如果不存在)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📂 已创建输出文件夹: {output_folder}")

    # 3. 读取视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    print(f"ℹ️  视频帧率: {fps} FPS | 总帧数: {total_frames}")

    # 设置起始帧
    if start_frame > 0:
        if start_frame >= total_frames:
            print(f"⚠️  起始帧 {start_frame} 超过总帧数 {total_frames}，无法处理。")
            cap.release()
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"⏭️  跳过前 {start_frame} 帧，从第 {start_frame} 帧开始处理...")

    # frame_count = start_frame
    frame_count = 0
    saved_count = 0

    print("🚀 开始处理...")

    while True:
        ret, frame = cap.read()

        # 如果读不到帧了（视频结束），退出循环
        if not ret:
            break

        # 按照设定的间隔保存图片
        if frame_count % frame_interval == 0:
            # 生成文件名，例如: frame_00001.jpg
            image_name = f"frame_{frame_count:06d}.jpg"
            save_path = os.path.join(output_folder, image_name)

            # 保存图片 (默认质量)
            # cv2.imwrite(save_path, frame)

            # 如果需要更高质量的 JPG，可以用下面的代码替代上一行：
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            saved_count += 1
            # 简单的进度打印
            if saved_count % 10 == 0:
                print(f"   -> 已保存 {saved_count} 张图片...", end='\r')

        frame_count += 1

    cap.release()
    print(f"\n✅ 处理完成！共保存了 {saved_count} 张图片到 '{output_folder}'")


# --- 使用示例 ---
if __name__ == "__main__":
    # 视频路径
    my_video = "/mnt/d/projects/stock_v2510/src/deep_research/video/600031_one_year.mp4"

    # 保存路径
    save_dir = "/mnt/d/projects/stock_v2510/src/bottom_reversal_breakout/dataset_1/sh600031"

    # 设置采样率
    # 如果视频是 30FPS:
    # interval = 1   -> 每一帧都存 (结果巨多)
    # interval = 30  -> 每秒存一张 (推荐)
    # interval = 15  -> 每 0.5秒存一张
    video_to_frames(my_video, save_dir, frame_interval=60,start_frame=90)