import torch
import numpy as np
import time
import os
from tqdm import tqdm

# 导入自定义库
from stylegan_utils import load_model, tensor_to_pil, get_w_from_seed, slerp

# --- 主程序 ---
if __name__ == "__main__":
    
    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs/morphing_slerp_frames' 
    SEED_A = 100
    SEED_B = 200
    DURATION_SEC = 5
    FPS = 30
    TRUNCATION_PSI = 0.7 
    # ------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有 [Slerp] 帧将保存到: {OUTPUT_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")
    
    # 调用 utils 函数
    G = load_model(MODEL_PATH, device)
    if not G:
        exit() 

    # 调用 utils 函数
    w_A = get_w_from_seed(G, device, SEED_A, TRUNCATION_PSI)
    w_B = get_w_from_seed(G, device, SEED_B, TRUNCATION_PSI)
    
    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧 [Slerp] 动画...")
    
    start_time = time.time()
    
    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)
        
        # 调用 utils 的 slerp 函数
        w_interp = slerp(w_A, w_B, t)
        
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        
        # 调用 utils 函数
        img_pil = tensor_to_pil(img_tensor)
        
        filename = f'frame_{i:04d}.png'
        img_pil.save(os.path.join(OUTPUT_DIR, filename))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧 [Slerp] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")