"""
StyleGAN Morphing 项目的自定义库
"""

import torch
import numpy as np
import PIL.Image
import pickle
import os
import torch.nn.functional as F

def load_model(model_path, device):
    """
    加载 StyleGAN2-ADA 模型并将其发送到指定设备
    """
    print(f"正在从 {model_path} 加载模型...")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到! '{model_path}'")
        return None
    try:
        with open(model_path, 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device) 
            return G
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def tensor_to_pil(img_tensor):
    """
    将 StyleGAN 的输出张量 (-1 到 1) 转换为 PIL 图像 (0 到 255)
    """
    img_tensor = (img_tensor + 1) * (255/2)
    img_tensor = img_tensor.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    img_np = img_tensor[0].cpu().numpy()
    img = PIL.Image.fromarray(img_np) 
    return img

def get_w_from_seed(G, device, seed, truncation_psi):
    """
    通过 G.mapping 从一个 seed 计算出 W 潜在向量
    (G 必须已经在 device 上)
    """
    print(f"正在从 Seed {seed} (PSI={truncation_psi}) 计算 W 向量...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. 创建 z 向量 (噪声)
    z = torch.randn([1, G.z_dim]).to(device)
    # 2. 创建 label 
    label = torch.zeros([1, G.c_dim]).to(device)
    
    # 3. 通过 mapping network 计算 w
    # w 的形状是 [1, 18, 512]，代表18层控制
    w = G.mapping(z, label, truncation_psi=truncation_psi)
    return w

def slerp(v0, v1, t, Epsilon=1e-5):
    """
    手动实现球面线性插值 (Spherical Linear Interpolation)
    v0, v1: 起始和结束向量 (形状: [1, 18, 512])
    t: 插值系数 (float, 0.0 to 1.0)
    """
    
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

def load_w_from_source(G, device, w_source, truncation_psi):
    """
    智能加载W向量：
    - 如果 w_source 是一个数字 (e.g., '100'), 就从seed生成.
    - 如果 w_source 是一个.pt文件 (e.g., 'my_face.pt'), 就从文件加载.
    """
    if str(w_source).endswith('.pt'):
        if not os.path.exists(w_source):
            print(f"错误: 找不到W文件: {w_source}")
            return None
        print(f"正在从文件加载 W: {w_source}...")
        w = torch.load(w_source).to(device)
        return w
    else:
        try:
            seed = int(w_source)
            return get_w_from_seed(G, device, seed, truncation_psi)
        except ValueError:
            print(f"错误: 输入 '{w_source}' 不是一个有效的 seed (数字) 或 .pt 文件。")
            return None