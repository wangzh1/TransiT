import cv2
import os

def images_to_video(image_folder, output_video, fps=10):
    # 获取图片列表
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 按名称排序，确保顺序正确

    if not images:
        print("No images found in the specified folder.")
        return

    # 获取第一张图片以确定视频尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 写入每一张图片到视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    print(f"Video saved as {output_video}")

# 使用示例
image_folder = '/home/user/NLOS-Video/logs/video_R'  # 替换为你的图片文件夹路径
output_video = '/home/user/NLOS-Video/logs/temp/output_video.mp4'  # 替换为输出视频文件名
images_to_video(image_folder, output_video, fps=10)
