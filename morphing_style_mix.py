import torch
import numpy as np
import PIL.Image
import pickle
import time
import os
from tqdm import tqdm
import torch.nn.functional as F

def load_model(model_path):
    print(f"正在从 {model_path} 加载模型...")
    try:
        with open(model_path, 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device)
            return G
    except Exception as e: print(f"加载模型失败: {e}"); return None

def tensor_to_pil(img_tensor):
    img_tensor = (img_tensor + 1) * (255/2)
    img_tensor = img_tensor.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    img_np = img_tensor[0].cpu().numpy()
    img = PIL.Image.fromarray(img_np) 
    return img

def get_w_from_seed(G, device, seed, truncation_psi):
    print(f"正在计算 Seed {seed} 的 W 向量...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    z = torch.randn([1, G.z_dim]).to(device)
    label = torch.zeros([1, G.c_dim]).to(device)
    w = G.mapping(z, label, truncation_psi=truncation_psi)
    return w

def slerp(v0, v1, t, Epsilon=1e-5):
    v_lerp = (1.0 - t) * v0 + t * v1
    v0_norm = F.normalize(v0, p=2, dim=-1)
    v1_norm = F.normalize(v1, p=2, dim=-1)
    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    v_slerp = (a * v0) + (b * v1)
    mask = (sin_omega < Epsilon).expand_as(v0)
    w_interp = torch.where(mask, v_lerp, v_slerp)
    return w_interp


# --- 主程序 ---
if __name__ == "__main__":
    
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs/morphing_style_mix_frames' 
    
    SEED_A = 100          # 起始人脸 (结构)
    SEED_B = 200          # 目标人脸 (风格)
    DURATION_SEC = 5      
    FPS = 30              
    TRUNCATION_PSI = 0.7 
    
    # 定义分层点
    # 锁定前 8 层 (0-7 = coarse+medium)，只插值后 10 层 (8-17 = fine/style)
    STYLE_MIX_LAYER = 8 
    # ------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有 [Style Mix] 帧将保存到: {OUTPUT_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")
    
    G = load_model(MODEL_PATH)
    if not G: exit() 

    w_A = get_w_from_seed(G, device, SEED_A, TRUNCATION_PSI)
    w_B = get_w_from_seed(G, device, SEED_B, TRUNCATION_PSI)
    
    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧 [Style Mix] 动画...")
    
    start_time = time.time()
    
    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)
        
        # 1. 计算完整的 Slerp 插值向量 
        w_full_slerp = slerp(w_A, w_B, t)

        # 2. 取 w_A (起始脸) 的 "结构" 层 (0 到 7)
        # w_A 的形状是 [1, 18, 512]
        # w_structure 的形状是 [1, 8, 512]
        w_structure = w_A[:, :STYLE_MIX_LAYER, :]
        
        # 3. 取 Slerp 插值向量的 "风格" 层 (8 到 17)
        # w_style 的形状是 [1, 10, 512]
        w_style = w_full_slerp[:, STYLE_MIX_LAYER:, :]
        
        # 4. 把它们拼回来 (dim=1 是层的维度)
        # 得到最终的 w_interp, 形状 [1, 18, 512]
        w_interp = torch.cat([w_structure, w_style], dim=1)
        
        
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        img_pil = tensor_to_pil(img_tensor)
        
        filename = f'frame_{i:04d}.png'
        img_pil.save(os.path.join(OUTPUT_DIR, filename))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧 [Style Mix] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
