import cv2
import os
import numpy as np # 拼接图像
from tqdm import tqdm # 进度条

print(f"OpenCV 版本: {cv2.__version__}")
print(f"Numpy 版本: {np.__version__}")

# --- 配置 ---
# [输入]
VIDEO_LERP_PATH = 'outputs/morphing_video.mp4'
VIDEO_SLERP_PATH = 'outputs/morphing_video_slerp.mp4'
# [输出]
OUTPUT_VIDEO_PATH = 'outputs/comparison_Lerp_vs_Slerp.mp4'
FPS = 30  # 和原视频保持FPS 一致
# ------------

def create_comparison(video_path_a, video_path_b, output_path, fps):
    """
    将两个视频(A和B)按 A|B 的形式并排拼接
    """
    
    # 1. 检查输入文件是否存在
    if not os.path.exists(video_path_a):
        print(f"错误: 找不到视频 '{video_path_a}'")
        return
    if not os.path.exists(video_path_b):
        print(f"错误: 找不到视频 '{video_path_b}'")
        return

    # 2. 初始化视频读取器
    cap_a = cv2.VideoCapture(video_path_a)
    cap_b = cv2.VideoCapture(video_path_b)

    # 3. 获取视频 A 的属性 (假设 B 和 A 属性相同)
    frame_width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 4. 计算输出视频的尺寸
    # 宽度加倍 (width*2), 高度不变
    output_size = (frame_width * 2, frame_height)
    
    print(f"输入视频尺寸: {frame_width}x{frame_height}")
    print(f"输出视频尺寸: {output_size[0]}x{output_size[1]}")
    
    # 5. 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    print(f"正在拼接 {total_frames} 帧...")

    # 6. 循环读取、拼接、写入
    for i in tqdm(range(total_frames)):
        ret_a, frame_a = cap_a.read() # 读取 A 的第 i 帧
        ret_b, frame_b = cap_b.read() # 读取 B 的第 i 帧
        
        if not ret_a or not ret_b:
            print(f"在第 {i} 帧读取失败，提前结束。")
            break
            
        # 7. 使用 numpy.hstack (Horizontal Stack) 水平拼接
        # [frame_a] + [frame_b] => [frame_a | frame_b]
        comparison_frame = np.hstack((frame_a, frame_b))
        
        # 在视频上添加标签：左上角(Lerp)和中上角(Slerp)
        cv2.putText(comparison_frame, 
                    'Lerp (Linear)', 
                    (50, 70), # 坐标 (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, # 字体大小
                    (255, 255, 255), # 颜色 (白色)
                    3) # 粗细
                    
        cv2.putText(comparison_frame, 
                    'Slerp (Spherical)', 
                    (frame_width + 50, 70), # x坐标偏移一个视频的宽度
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, 
                    (0, 255, 0), # 颜色 (绿色)
                    3)
        
        # 8. 写入拼接后的帧
        out.write(comparison_frame)

    # 9. 释放所有资源
    cap_a.release()
    cap_b.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n成功！ 对比视频已保存到: {output_path}")

# --- 主程序 ---
if __name__ == "__main__":
    create_comparison(VIDEO_LERP_PATH, VIDEO_SLERP_PATH, OUTPUT_VIDEO_PATH, FPS)
