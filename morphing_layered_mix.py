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
    
    # 命令行接受 --start 和 --end 参数
    parser = argparse.ArgumentParser(
        description="StyleGAN2 Layered-Mixing Morphing (结构A + 风格B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--start', 
        type=int, 
        default=8, 
        help="混合的[起始层] (包含)。范围 0-17。"
    )
    parser.add_argument(
        '--end', 
        type=int, 
        default=18, 
        help="混合的[结束层] (不包含)。范围 1-18。"
    )
    
    parser.add_argument('--seed_a', type=int, default=100, help="起始人脸 (结构源)")
    parser.add_argument('--seed_b', type=int, default=200, help="目标人脸 (风格源)")
    
    args = parser.parse_args()
    # -----------------------------------------------------------
    
    
    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs/morphing_layered_mix_frames' # 保持统一输出
    
    SEED_A = args.seed_a
    SEED_B = args.seed_b
    DURATION_SEC = 5      
    FPS = 30              
    TRUNCATION_PSI = 0.7
    
    # 从 args 读取 start 和 end
    MIX_START_LAYER = args.start
    MIX_END_LAYER = args.end
    # ------------
    
    # 参数校验
    if not (0 <= MIX_START_LAYER < 18):
        print(f"错误: --start 必须在 0-17 之间。你输入的是 {MIX_START_LAYER}")
        exit()
    if not (1 <= MIX_END_LAYER <= 18):
        print(f"错误: --end 必须在 1-18 之间。你输入的是 {MIX_END_LAYER}")
        exit()
    if MIX_START_LAYER >= MIX_END_LAYER:
        print(f"错误: --start ({MIX_START_LAYER}) 必须小于 --end ({MIX_END_LAYER})")
        exit()
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- [警告] ---")
    print(f"所有帧将保存到统一文件夹: {OUTPUT_DIR} (将被覆盖)")
    print(f"---------------")
    print(f"当前设置: 结构源 (Seed A): {SEED_A}, 风格源 (Seed B): {SEED_B}")
    
    # 打印层信息
    print(f"混合层范围: {MIX_START_LAYER} 到 {MIX_END_LAYER - 1} (共 {MIX_END_LAYER - MIX_START_LAYER} 层)")
    print("层级参考: [0-3] 姿态, [4-7] 五官, [8-17] 风格")


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

        # 三段式拼接 (Pre-Struct, Mixed, Post-Struct)
        
        # 1. 混合前的结构 (来自 A)
        # e.g., --start 4 -> w_structure_pre 形状为 [1, 4, 512] (层 0,1,2,3)
        w_structure_pre = w_A[:, :MIX_START_LAYER, :]
        
        # 2. 混合中的部分 (来自 Slerp)
        # e.g., --start 4 --end 8 -> w_mixed_part 形状为 [1, 4, 512] (层 4,5,6,7)
        w_mixed_part = w_full_slerp[:, MIX_START_LAYER:MIX_END_LAYER, :]
        
        # 3. 混合后的结构 (来自 A)
        # e.g., --end 8 -> w_structure_post 形状为 [1, 10, 512] (层 8-17)
        w_structure_post = w_A[:, MIX_END_LAYER:, :]
        
        # 4. 把它们拼回来
        w_interp = torch.cat([w_structure_pre, w_mixed_part, w_structure_post], dim=1)
        
        
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        img_pil = tensor_to_pil(img_tensor)
        
        filename = f'frame_{i:04d}.png'
        img_pil.save(os.path.join(OUTPUT_DIR, filename))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧 [Layered Mix] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")