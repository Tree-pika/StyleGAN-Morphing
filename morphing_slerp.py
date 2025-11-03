import torch
import numpy as np
import PIL.Image
import pickle
import time
import os
from tqdm import tqdm
import torch.nn.functional as F # 我们需要 F.normalize

# --- 辅助函数 (来自 generate_face.py) ---

def load_model(model_path):
    print(f"正在从 {model_path} 加载模型...")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到! '{model_path}'")
        return None
    try:
        with open(model_path, 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device) # 直接加载并送到device
            return G
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

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
 
# 手写实现 Slerp 函数

def slerp(v0, v1, t, Epsilon=1e-5):
    """
    实现球面线性插值 (Spherical Linear Interpolation)
    v0, v1: 起始和结束向量 (形状: [1, 18, 512])
    t: 插值系数 (float, 0.0 to 1.0)
    Epsilon: 一个很小的数，用于处理数值不稳定
    """
    
    # (1) LERP (线性插值)，把它作为 Slerp 不稳定时的备用方案
    v_lerp = (1.0 - t) * v0 + t * v1
    
    # (2) SLERP (球面插值)
    # 归一化向量，得到方向 (形状: [1, 18, 512])
    v0_norm = F.normalize(v0, p=2, dim=-1)
    v1_norm = F.normalize(v1, p=2, dim=-1)
    
    # (3) 计算点积 (Dot Product)，即 cos(omega)
    # 需要在18个层上独立计算，所以在最后一个维度(dim=-1)上求和
    # keepdim=True 保持形状为 [1, 18, 1]，以便后续广播
    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    
    # (4) 计算夹角 omega
    omega = torch.acos(dot) # 形状: [1, 18, 1]
    
    # (5) 计算 sin(omega)
    sin_omega = torch.sin(omega) # 形状: [1, 18, 1]
    
    # (6) 计算 Slerp 权重
    # t 是一个标量 (float)，PyTorch 会自动广播
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    
    # (7) 计算 Slerp 结果 (应用权重)
    v_slerp = (a * v0) + (b * v1) # 形状: [1, 18, 512]
    
    # (8) 处理特殊情况
    # 如果 sin_omega 非常接近 0 (即 omega 接近 0 或 180 度)，
    # (a / sin_omega) 和 (b / sin_omega) 的计算会除以0，导致 NaN (数值不稳定)
    #
    # 当 sin_omega < Epsilon 时，说明两个向量几乎共线，
    # 此时退回到使用更稳定的 LERP (线性插值)
    #
    # 需要扩展 mask 来匹配 v0/v1 的形状 [1, 18, 512]
    mask = (sin_omega < Epsilon).expand_as(v0)
    
    # `torch.where(condition, value_if_true, value_if_false)`
    # 如果 mask 为 True (sin_omega 很小), 使用 v_lerp
    # 否则，使用计算出的 v_slerp
    w_interp = torch.where(mask, v_lerp, v_slerp)
    
    return w_interp


# --- 主程序 ---
if __name__ == "__main__":
    
    # --- 配置 (与 morphing.py 相同) ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    
    # 将 Slerp 的结果保存到新文件夹
    OUTPUT_DIR = 'outputs/morphing_slerp_frames' 
    
    SEED_A = 100          # 起始人脸
    SEED_B = 200          # 目标人脸
    DURATION_SEC = 5      # 动画时长 (秒)
    FPS = 30              # 动画帧率 (fps)
    TRUNCATION_PSI = 0.7 
    # ------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有 [Slerp] 帧将保存到: {OUTPUT_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")
    
    G = load_model(MODEL_PATH)
    if not G:
        exit() 

    w_A = get_w_from_seed(G, device, SEED_A, TRUNCATION_PSI)
    w_B = get_w_from_seed(G, device, SEED_B, TRUNCATION_PSI)
    
    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧 [Slerp] 动画...")
    
    start_time = time.time()
    
    for i in tqdm(range(total_frames)):
        t = i / (total_frames - 1)
        
        w_interp = slerp(w_A, w_B, t)
        
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        img_pil = tensor_to_pil(img_tensor)
        
        filename = f'frame_{i:04d}.png'
        img_pil.save(os.path.join(OUTPUT_DIR, filename))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧 [Slerp] 已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
