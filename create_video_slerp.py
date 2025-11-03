import cv2
import os
import re # 导入正则表达式库来帮助排序

print(f"OpenCV 版本: {cv2.__version__}")

# --- 配置 ---
INPUT_DIR = 'outputs/morphing_slerp_frames'  # 生成帧的文件夹
OUTPUT_VIDEO_PATH = 'outputs/morphing_video_slerp.mp4' # 最终视频的名字
FPS = 30  # 保持和 morphing_slerp.py 中的 FPS 一致
# ------------

def get_frame_dimensions(frame_dir):
    """读取第一张图片来获取视频的宽度和高度"""
    # 找到第一张图片 (这里是frame_0000.png)
    test_image_name = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])[0]
    test_image_path = os.path.join(frame_dir, test_image_name)
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"错误: 无法读取第一帧图像于: {test_image_path}")
        return None, None
        
    height, width, layers = img.shape
    print(f"检测到视频尺寸: {width} x {height}")
    return (width, height)

def natural_sort_key(s):
    """
    创建一个“自然排序”的键 (e.g. 'frame_10.png' 会排在 'frame_2.png' 之后)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def create_video(frame_dir, video_path, fps):
    """从帧文件夹创建视频"""
    
    # 1. 获取视频尺寸
    frame_size = get_frame_dimensions(frame_dir)
    if frame_size[0] is None:
        return

    # 2. 定义视频编码器 (H.264)
    # 'mp4v' 对应 .mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 3. 初始化 VideoWriter 对象
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    # 4. 获取所有帧并正确排序
    frames = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
    frames.sort(key=natural_sort_key) # 按 frame_0000, 0001, ... 排序
    
    print(f"正在从 {len(frames)} 帧图像创建视频...")
    
    # 5. 循环读取每一帧并写入视频
    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path)
        out.write(img) # 将帧写入
        
    # 6. 释放资源
    out.release()
    print(f"\n成功！ 视频已保存到: {video_path}")

# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 找不到输入文件夹 '{INPUT_DIR}'")
        print("请先运行 morphing_slerp.py 来生成帧。")
    else:
        create_video(INPUT_DIR, OUTPUT_VIDEO_PATH, FPS)
