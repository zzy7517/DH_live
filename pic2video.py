import numpy as np
import cv2

# 输入图片路径
image_path = '3d.png'
# 输出视频路径
output_video_path = 'output.mp4'
# 视频帧率
fps = 30
# 视频时长（秒）
duration = 5



if __name__ == '__main__':
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # 获取图片的宽度和高度
    height, width, _ = image.shape

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 计算总帧数
    total_frames = fps * duration

    # 将图片写入视频的每一帧
    for _ in range(total_frames):
        video_writer.write(image)

    # 释放VideoWriter对象
    video_writer.release()

    print(f"Video saved as {output_video_path}")