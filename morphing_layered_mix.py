import torch
import numpy as np
import time
import os
from tqdm import tqdm
import argparse # 解析命令行参数

# 导入自定义库
from stylegan_utils import load_model, tensor_to_pil, get_w_from_seed, slerp

# --- 主程序 ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="StyleGAN2 Layered-Mixing Morphing (结构A + 风格B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-l', '--layer', 
        type=int, 
        default=8, 
        help="风格混合的起始层 (0-17)。 "
             "0-3: 姿态/脸型, 4-7: 五官, 8-17: 皮肤/头发/光照"
    )
    parser.add_argument('--seed_a', type=int, default=100, help="起始人脸 (结构源)")
    parser.add_argument('--seed_b', type=int, default=200, help="目标人脸 (风格源)")
    
    args = parser.parse_args()
    
    
    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    
    # 统一输出文件夹
    OUTPUT_DIR = 'outputs/morphing_layered_mix_frames'
    
    SEED_A = args.seed_a
    SEED_B = args.seed_b
    DURATION_SEC = 5      
    FPS = 30              
    TRUNCATION_PSI = 0.7
    
    LAYERED_MIX_START_LAYER = args.layer 

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 告知用户文件将被覆盖
    print(f"--- [警告] ---")
    print(f"所有帧将保存到统一文件夹: {OUTPUT_DIR}")
    print(f"如果使用不同 --layer 运行，此文件夹将被覆盖。")
    print(f"---------------")
    print(f"当前设置: 结构源 (Seed A): {SEED_A}, 风格源 (Seed B): {SEED_B}")
    print(f"混合起始层 (--layer): {LAYERED_MIX_START_LAYER}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")
    
    G = load_model(MODEL_PATH, device)
    if not G: exit() 

    w_A = get_w_from_seed(G, device, SEED_A, TRUNCATION_PSI)
    w_B = get_w_from_seed(G, device, SEED_B, TRUNCATION_PSI)
    
    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧 [Layered Mix] 动画...")
    
    start_time = time.time()
    
    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)
        
        w_full_slerp = slerp(w_A, w_B, t)

        w_structure = w_A[:, :LAYERED_MIX_START_LAYER, :]
        w_style = w_full_slerp[:, LAYERED_MIX_START_LAYER:, :]
        
        w_interp = torch.cat([w_structure, w_style], dim=1)
        
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        img_pil = tensor_to_pil(img_tensor)
        
        filename = f'frame_{i:04d}.png'
        img_pil.save(os.path.join(OUTPUT_DIR, filename))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧 [Layered Mix] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")