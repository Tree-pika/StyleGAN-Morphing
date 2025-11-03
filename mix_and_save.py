import torch
import os
import argparse
# 导入自定义库
from stylegan_utils import load_model, get_w_from_seed

# 用于加载W
def load_w_from_source(G, device, w_source, truncation_psi):
    """
    智能加载W向量：
    - 如果 w_source 是一个数字 (e.g., '100'), 就从seed生成.
    - 如果 w_source 是一个.pt文件 (e.g., 'my_face.pt'), 就从文件加载.
    """
    if w_source.endswith('.pt'):
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

# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StyleGAN W-Vector Mixer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 定义参数
    parser.add_argument('--struct', default='100', help="结构源 (Seed 或 .pt 文件) [层 0-3]")
    parser.add_argument('--face', default='100', help="五官源 (Seed 或 .pt 文件) [层 4-7]")
    parser.add_argument('--style', default='100', help="风格源 (Seed 或 .pt 文件) [层 8-17]")

    parser.add_argument('-o', '--output', default='outputs/mixed_w.pt', help="输出的 .pt 文件路径")
    parser.add_argument('--psi', type=float, default=0.7, help="截断值 (仅用于从seed生成时)")

    args = parser.parse_args()

    # --- 执行 ---
    MODEL_PATH = 'stylegan2-ffhq.pkl'
    TRUNCATION_PSI = args.psi
    os.makedirs('outputs', exist_ok=True) # 确保outputs存在

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到可用设备: {device}")

    G = load_model(MODEL_PATH, device)
    if not G: exit()

    # 1. 加载所有需要的 W 向量
    w_struct = load_w_from_source(G, device, args.struct, TRUNCATION_PSI)
    w_face   = load_w_from_source(G, device, args.face, TRUNCATION_PSI)
    w_style  = load_w_from_source(G, device, args.style, TRUNCATION_PSI)

    if w_struct is None or w_face is None or w_style is None:
        print("一个或多个W向量加载失败，退出。")
        exit()

    # 2. 按照参数定义拼接它们
    # 形状 [1, 4, 512]
    part_struct = w_struct[:, :4, :] 
    # 形状 [1, 4, 512]
    part_face = w_face[:, 4:8, :]
    # 形状 [1, 10, 512]
    part_style = w_style[:, 8:, :]

    # 拼接成 [1, 18, 512]
    w_new = torch.cat([part_struct, part_face, part_style], dim=1)

    # 3. 保存到文件
    torch.save(w_new, args.output)

    print(f"\n成功！")
    print(f"  结构源: {args.struct}")
    print(f"  五官源: {args.face}")
    print(f"  风格源: {args.style}")
    print(f"已混合并保存到: {args.output}")
