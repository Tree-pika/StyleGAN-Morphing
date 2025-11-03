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
    parser = argparse.ArgumentParser(description="StyleGAN2 Layered-Mixing Morphing")
    # 命令行接受 --a 和 --b 参数:起始和目标人脸
    parser.add_argument('--a', default='100', help="结构源A (Seed 或 .pt 文件)")
    parser.add_argument('--b', default='200', help="风格源B (Seed 或 .pt 文件)")
    # 命令行接受 --start 和 --end 参数:想要混合的起始和结束层
    parser.add_argument('--start', type=int, default=8, help="混合的[起始层] (包含)")
    parser.add_argument('--end', type=int, default=18, help="混合的[结束层] (不包含)")
    parser.add_argument('--psi', type=float, default=0.7, help="截断值 (仅用于从seed生成时)")
    args = parser.parse_args()
    # -----------------------------------------------------------

    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs/morphing_layered_mix_frames'# 保持统一输出

    DURATION_SEC = 5      
    FPS = 30              
    TRUNCATION_PSI = args.psi
    MIX_START_LAYER = args.start
    MIX_END_LAYER = args.end
    
    # 参数校验
    if not (0 <= MIX_START_LAYER < 18): 
        exit("错误: --start 必须在 0-17")
    if not (1 <= MIX_END_LAYER <= 18): 
        exit("错误: --end 必须在 1-18")
    if MIX_START_LAYER >= MIX_END_LAYER: 
        exit("错误: --start 必须小于 --end")
       
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- [警告] ---")
    print(f"所有帧将保存到统一文件夹: {OUTPUT_DIR} (将被覆盖)")
    print(f"---------------")
    print(f"结构源 (A): {args.a}, 风格源 (B): {args.b}")
    print(f"混合层范围: {MIX_START_LAYER} 到 {MIX_END_LAYER - 1}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = load_model(MODEL_PATH, device)
    if not G: exit() 

    # 
    w_A = load_w_from_source(G, device, args.a, TRUNCATION_PSI)
    w_B = load_w_from_source(G, device, args.b, TRUNCATION_PSI)
    if w_A is None or w_B is None: exit()
   
    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧 [Layered Mix] 动画...")
    
    start_time = time.time()

    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)
        w_full_slerp = slerp(w_A, w_B, t)
        # 三段式拼接 (Pre-Struct, Mixed, Post-Struct)
        
        # 1. 混合前的结构 (来自 A)
        w_structure_pre = w_A[:, :MIX_START_LAYER, :]

        # 2. 混合中的部分 (来自 Slerp)
        w_mixed_part = w_full_slerp[:, MIX_START_LAYER:MIX_END_LAYER, :]

        # 3. 混合后的结构 (来自 A)
        w_structure_post = w_A[:, MIX_END_LAYER:, :]

        # 4. 把它们拼回来
        w_interp = torch.cat([w_structure_pre, w_mixed_part, w_structure_post], dim=1)
        
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        img_pil = tensor_to_pil(img_tensor)
        img_pil.save(os.path.join(OUTPUT_DIR, f'frame_{i:04d}.png'))

    end_time = time.time()

    print(f"\n成功！ {total_frames} 帧 [Layered Mix] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")