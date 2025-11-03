import torch
import numpy as np
import time
import os
from tqdm import tqdm
import argparse # 解析命令行参数

# 导入自定义库
from stylegan_utils import load_model, tensor_to_pil, slerp, load_w_from_source

# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slerp (球面插值) Morphing")
    # 命令行接受 --a 和 --b 参数:起始和目标人脸
    parser.add_argument('--a', default='100', help="输入A (Seed 或 .pt 文件)")
    parser.add_argument('--b', default='200', help="输入B (Seed 或 .pt 文件)")


    parser.add_argument('--psi', type=float, default=0.7, help="截断值 (仅用于从seed生成时)")
    args = parser.parse_args()


    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs/morphing_slerp_frames' 
    DURATION_SEC = 5
    FPS = 30
    TRUNCATION_PSI = args.psi
    # ------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有 [Slerp] 帧将保存到: {OUTPUT_DIR} (将被覆盖)")
    print(f"输入 A: {args.a}, 输入 B: {args.b}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_model(MODEL_PATH, device)
    if not G: exit() 

    # 
    w_A = load_w_from_source(G, device, args.a, TRUNCATION_PSI)
    w_B = load_w_from_source(G, device, args.b, TRUNCATION_PSI)
    if w_A is None or w_B is None: exit()

    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧 [Slerp] 动画...")
    
    start_time = time.time()

    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)
        # 调用 utils 的 slerp 函数
        w_interp = slerp(w_A, w_B, t)

        img_tensor = G.synthesis(w_interp, noise_mode='const')

        img_pil = tensor_to_pil(img_tensor)
        img_pil.save(os.path.join(OUTPUT_DIR, f'frame_{i:04d}.png'))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧 [Slerp] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")