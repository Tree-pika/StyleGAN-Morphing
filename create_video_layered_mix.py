import cv2
import os
import re

print(f"OpenCV 版本: {cv2.__version__}")


# 为了简化，默认统一输入和输出路径，若想不覆盖，可以在二次运行前修改下面的参数
INPUT_DIR = 'outputs/morphing_layered_mix_frames'
OUTPUT_VIDEO_PATH = 'outputs/morphing_video_layered_mix.mp4'
FPS = 30

def get_frame_dimensions(frame_dir):
    try:
        test_image_name = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])[0]
    except IndexError:
        print(f"错误: 文件夹 '{frame_dir}' 为空或不包含 .png 文件。")
        return None, None
        
    test_image_path = os.path.join(frame_dir, test_image_name)
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"错误: 无法读取第一帧图像于: {test_image_path}")
        return None, None
        
    height, width, layers = img.shape
    print(f"检测到视频尺寸: {width} x {height}")
    return (width, height)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def create_layered_mix_video(frame_dir, video_path, fps):
    
    frame_size = get_frame_dimensions(frame_dir)
    if frame_size[0] is None:
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    frames = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
    frames.sort(key=natural_sort_key)
    
    if not frames:
         print(f"错误: 在 '{frame_dir}' 中未找到任何 .png 帧。")
         out.release()
         return
         
    print(f"正在从 {len(frames)} 帧图像创建视频...")
    
    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path)
        out.write(img)
        
    out.release()
    print(f"\n成功！ 视频已保存到: {video_path}")

# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 找不到输入文件夹 '{INPUT_DIR}'")
        print("请先运行 morphing_layered_mix.py 来生成帧。")
    else:
        create_layered_mix_video(INPUT_DIR, OUTPUT_VIDEO_PATH, FPS)