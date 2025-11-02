import torch
import numpy as np
import PIL.Image
import pickle
import time
import os # 导入 os 库来处理文件路径

def load_model(model_path):
    """加载 StyleGAN2-ADA 模型"""
    print(f"正在从 {model_path} 加载模型...")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到! 请确保 '{model_path}' 已经下载。")
        return None
        
    try:
        with open(model_path, 'rb') as f:
            network_dict = pickle.load(f)
            G = network_dict['G_ema'] # G_ema 是效果最好的生成器
            return G
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def generate_image(G, device, seed):
    """使用生成器 G 在指定设备上生成一张图片"""
    print(f"正在使用设备: {device} (Seed: {seed}) 生成图片...")
    
    # 1. 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 2. 创建输入噪声向量 z (latent code)
    # z 的形状: [batch_size, z_dim] -> [1, 512]
    z = torch.randn([1, G.z_dim]).to(device)
    
    # 3. 创建标签 (FFHQ模型不需要，设为0)
    label = torch.zeros([1, G.c_dim]).to(device)
    
    # 4. 生成图像
    #    G(z, label) 会输出形状为 [1, 3, 1024, 1024] 的图像张量
    #    值域在 -1 到 1 之间
    start_time = time.time()
    img_tensor = G(z, label, truncation_psi=1.0, noise_mode='const')
    end_time = time.time()
    print(f"图片生成耗时: {end_time - start_time:.3f} 秒")

    # 5. 将张量转换回 PIL 图像 (0-255范围)
    img_tensor = (img_tensor + 1) * (255/2) # 转换范围从 [-1, 1] 到 [0, 255]
    img_tensor = img_tensor.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    img_np = img_tensor[0].cpu().numpy()
    img = PIL.Image.fromarray(img_np, 'RGB')
    return img

# --- 主程序 ---
if __name__ == "__main__":
    
    # --- 配置 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    OUTPUT_DIR = 'outputs' # 保存图片的文件夹
    RANDOM_SEED = 100      # 定义生成不同的人脸的数量
    # ------------
    
    # 1. 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")
    
    if (device.type == 'cpu'):
        print("警告: 正在使用 CPU 运行。生成过程会非常缓慢。")
        
    # 3. 加载模型
    G = load_model(MODEL_PATH)
    
    if G:
        # 4. 将模型移动到目标设备 (GPU)
        G.to(device)
        
        # 5. 生成图片
        generated_image = generate_image(G, device, seed=RANDOM_SEED)
        
        # 6. 保存图片
        output_filename = f'generated_face_seed{RANDOM_SEED}.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        generated_image.save(output_path)
        
        print(f"\n成功！图片已保存到: {output_path}")
