import torch
import numpy as np
import PIL.Image
import pickle
import time
import os
from tqdm import tqdm # 进度条库

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
    """
    将 StyleGAN 的输出张量 (-1 到 1) 转换为 PIL 图像 (0 到 255)
    并修复 generate_face.py 中的 DeprecationWarning
    """
    img_tensor = (img_tensor + 1) * (255/2)
    img_tensor = img_tensor.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    img_np = img_tensor[0].cpu().numpy()
    
    # 修复 DeprecationWarning：PIL 可以自动从 'RGB' 形状的 numpy 数组推断模式
    img = PIL.Image.fromarray(img_np) 
    return img

def get_w_from_seed(G, device, seed, truncation_psi):
    """
    通过 G.mapping 从一个 seed 计算出 W 潜在向量
    """
    print(f"正在计算 Seed {seed} 的 W 向量...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. 创建 z 向量 (噪声)
    z = torch.randn([1, G.z_dim]).to(device)
    # 2. 创建 label (不需要)
    label = torch.zeros([1, G.c_dim]).to(device)
    
    # 3. 通过 mapping network 计算 w
    # w 的形状是 [1, 18, 512]，代表18层控制
    w = G.mapping(z, label, truncation_psi=truncation_psi)
    return w

# --- 主程序 ---
if __name__ == "__main__":
    
    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs/morphing_frames' # 新的输出文件夹
    
    SEED_A = 100          # 起始人脸 
    SEED_B = 200          # 目标人脸 
    
    DURATION_SEC = 5      # 动画时长 (秒)
    FPS = 30              # 动画帧率 (fps)
    
    # 截断值：< 1.0 的值能产生更"常规"、质量更高的人脸
    # 1.0 会产生更多样化但可能更奇怪的人脸
    TRUNCATION_PSI = 0.7 
    # ------------
    
    # 1. 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有帧将保存到: {OUTPUT_DIR}")

    # 2. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")
    
    # 3. 加载模型
    G = load_model(MODEL_PATH)
    if not G:
        exit() # 加载失败则退出

    # 4. 计算两个端点的 W 向量
    w_A = get_w_from_seed(G, device, SEED_A, TRUNCATION_PSI)
    w_B = get_w_from_seed(G, device, SEED_B, TRUNCATION_PSI)
    
    # 5. 执行插值并生成每一帧
    total_frames = DURATION_SEC * FPS
    print(f"即将开始生成 {total_frames} 帧动画 (时长 {DURATION_SEC}s @ {FPS}fps)...")
    
    start_time = time.time()
    
    # 使用 tqdm 来显示进度条
    for i in tqdm(range(total_frames)):
        
        # t 是插值系数，从 0.0 (A) 变为 1.0 (B)
        t = i / (total_frames - 1)
        
        # 6. 线性插值 (Lerp)
        # torch.lerp(start, end, weight)
        w_interp = torch.lerp(w_A, w_B, t)
        
        # 7. 用 G.synthesis (合成网络) 从 w 生成图像
        # 此时不需要 G.mapping 了，因为已经有了 w
        img_tensor = G.synthesis(w_interp, noise_mode='const')
        
        # 8. 转换并保存
        img_pil = tensor_to_pil(img_tensor)
        
        # 确保命名要规范：用0补齐 (e.g., frame_000.png, frame_001.png)
        filename = f'frame_{i:04d}.png'
        img_pil.save(os.path.join(OUTPUT_DIR, filename))

    end_time = time.time()
    print(f"\n成功！ {total_frames} 帧已全部生成。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每帧耗时: {(end_time - start_time) / total_frames * 1000:.2f} 毫秒")
