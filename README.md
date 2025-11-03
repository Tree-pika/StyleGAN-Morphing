# StyleGAN2 肖像渐变 (Portrait Morphing)

本项目为浙江大学《计算机动画》课程的大程设计，实现了一个基于NVIDIA StyleGAN2预训练模型的W空间肖像渐变工具。

与传统的基于网格或特征点的Morphing方法不同，本项目在StyleGAN的W潜在空间(W-Latent Space)中进行插值，能够产生极其平滑、高质量、高保真度的人脸渐变动画。

## 🌟 项目特色

* **线性插值 (Lerp):** 实现了W空间中的标准线性插值作为基线。
* **球面插值 (Slerp):** **从零开始实现**了球面插值 (`slerp`) 算法，以产生在视觉上更稳定、更平滑的过渡路径，有效避免了Lerp可能导致的中间帧模糊问题。
* **高级分层混合 (Layered Mix):** 基于对StyleGAN 18层W向量的理解，**原创性地实现了分层混合功能**。用户可以指定任意层范围（例如`--start 4 --end 8`）来进行插值，从而实现“只变换五官”、“只变换风格/皮肤”等高级效果。
* **W向量工具箱 (W-Mixer):** 提供了一个 `mix_and_save.py` 脚本，允许用户将不同人脸的特征（如A的结构 + B的五官 + C的风格）“缝合”成一个新的`W`向量 (`.pt` 文件)。
* **解耦的工程设计 (DRY):**
    1.  所有核心脚本（`lerp`, `slerp`, `layered_mix`）均被重构，支持从`seed`或`.pt`文件加载输入。
    2.  所有公共函数（模型加载、`slerp`算法等）均被提取到 `stylegan_utils.py` 中，遵循DRY原则。
* **灵活的命令行界面 (CLI):** 所有脚本均使用 `argparse` 构建，提供了清晰易用的命令行接口。

## 🎨 最终效果展示


### 1. Slerp (右) vs. Lerp (左) 对比
展示了Slerp（球面插值）在过渡中相比Lerp（线性插值）能更好地保持特征的清晰度。

![Slerp vs Lerp Comparison](outputs/comparison_Lerp_vs_Slerp.gif)

### 2. 高级功能：分层混合 (Layered Mix)
**仅混合五官 (层 4-7):** 人脸的结构、姿态、发型和皮肤风格保持不变，只有五官（眼、鼻、口）从A源渐变为B源。
*(使用 `--start 4 --end 8` 运行 `morphing_layered_mix.py` 并生成)*

**仅混合风格 (层 8-17):** 人脸的结构和五官保持不变，只有皮肤纹理、头发颜色、光照和背景风格从A源渐变为B源。

![Layered Mix - Style Only](outputs/morphing_layered_mix.gif)

### 3. W向量混合与Morphing
我们首先创建一个“缝合脸” (`my_face.pt` = Seed 100的结构 + Seed 200的五官 + Seed 300的风格)，然后实现了从 Seed 100 到这个“缝合脸”的Slerp渐变。

![Slerp Morph to Mixed W](outputs/morphing_video_slerp.gif)

PS：[mp4 to GIf online transformer](https://ezgif.com/video-to-gif)
## 🗂️ 文件结构

```
StyleGAN-Morphing/
│
│── outputs/                  # 存放呈现效果的GIF等文件
├── stylegan_utils.py         # 核心工具箱 (DRY): 包含模型加载, W生成, Slerp算法等
│
├── generate_face.py          # (工具) 测试环境, 生成单张人脸
├── mix_and_save.py           # (工具) W向量混合器, 用于创建 "缝合脸" (.pt)
│
├── morphing.py               # (核心) 基础版, 使用 Lerp (线性插值)
├── morphing_slerp.py         # (核心) 进阶版, 使用 Slerp (球面插值)
├── morphing_layered_mix.py   # (核心) 高级版, 支持任意层范围 (--start, --end) 混合
│
├── create_video.py           # (视频工具) 合成 Lerp 动画
├── create_video_slerp.py     # (视频工具) 合成 Slerp 动画
├── create_video_layered_mix.py # (视频工具) 合成 Layered Mix 动画
├── create_comparison_video.py  # (视频工具) 创建 Lerp vs Slerp 对比视频
│
├── .gitignore                # Git 忽略配置
└── README.md                 # 本文档
```

## 🚀 如何运行

### 1. 克隆与环境配置

```bash
# 1. 克隆仓库
git clone [https://github.com/Tree-pika/StyleGAN-Morphing.git](https://github.com/Tree-pika/StyleGAN-Morphing.git)
cd StyleGAN-Morphing

# 2. 创建并激活 Conda 环境
conda create --name CG python=3.10
conda activate CG

# 3. 安装 PyTorch (需NVIDIA GPU, 以CUDA 12.6为例)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# 4. 安装其他 Python 库
pip install numpy pillow opencv-python tqdm
```

### 2. 下载模型与依赖

本项目需要NVIDIA的预训练模型 (`.pkl`) 和它所依赖的辅助代码 (`dnnlib`, `torch_utils`)。
为方便起见，你可以在项目根目录创建一个 `download_deps.sh` 脚本来自动完成此操作。

```bash
# nano download_deps.sh
# 粘贴以下内容并保存:
#!/bin/bash
echo "Downloading StyleGAN2 FFHQ Model..."
wget [https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) -O stylegan2-ffhq.pkl

echo "Downloading dnnlib dependencies..."
mkdir dnnlib && cd dnnlib
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/dnnlib/__init__.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/dnnlib/__init__.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/dnnlib/util.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/dnnlib/util.py)
cd ..

echo "Downloading torch_utils dependencies..."
mkdir torch_utils && cd torch_utils
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/__init__.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/__init__.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/persistence.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/persistence.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/misc.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/misc.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/custom_ops.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/custom_ops.py)

mkdir ops && cd ops
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/__init__.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/__init__.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/bias_act.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/bias_act.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/conv2d_gradfix.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/conv2d_gradfix.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/conv2d_resample.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/conv2d_resample.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/fma.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/fma.py)
wget [https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/upfirdn2d.py](https://raw.githubusercontent.com/NVlabs/stylegan2-ada-pytorch/main/torch_utils/ops/upfirdn2d.py)
cd ../..
echo "All dependencies downloaded."

# ---
# 运行这个脚本
chmod +x download_deps.sh
./download_deps.sh
```

### 3. 使用示例

(确保你的 `(CG)` 环境已激活)

#### 用例 1: 生成一个标准的Slerp渐变 (Seed 100 -> 200)

```bash
# 1. 生成 Slerp 帧 (使用 --a 和 --b 指定输入)
python morphing_slerp.py --a 100 --b 200

# 2. 合成视频
python create_video_slerp.py

# 3. 结果: outputs/morphing_video_slerp.mp4
```

#### 用例 2: 生成一个“仅五官”渐变 (层 4-7)

```bash
# 1. 生成 Layered Mix 帧 (指定 --start 和 --end)
python morphing_layered_mix.py --a 100 --b 200 --start 4 --end 8

# 2. 合成视频
python create_video_layered_mix.py

# 3. 结果: outputs/morphing_layered_mix.mp4
```

#### 用例 3: 创建并使用一个“混合W向量”

```bash
# 1. 创建一个“缝合脸” W 向量:
# (Seed 100的结构 + Seed 200的五官 + Seed 300的风格)
python mix_and_save.py --struct 100 --face 200 --style 300 -o outputs/my_face.pt

# 2. 将 Seed 100 渐变到这个新创建的脸上
# (注意 --b 参数现在是一个 .pt 文件)
python morphing_slerp.py --a 100 --b outputs/my_face.pt

# 3. 合成视频
python create_video_slerp.py

# 4. 结果: outputs/morphing_video_slerp.mp4 (被覆盖)
```

## 🚧 局限性 & 未来工作

本项目目前从 `seed` 或混合的 `.pt` 文件生成W向量。它**不支持**将用户提供的真实照片（如 `my_photo.jpg`）作为输入。

这项功能被称为 **"GAN Inversion"**，是一个非常复杂的高级课题，它需要解决两个主要的额外挑战：
1.  **人脸对齐 (Face Alignment):** 需要使用 `dlib` 等库来检测用户照片中的面部关键点，并将其裁剪、旋转、缩放，以匹配StyleGAN训练时所用的FFHQ数据集的标准对齐方式。
2.  **W向量优化 (Latent Vector Optimization):** 这是一个反向传播优化问题。需要将 `W` 向量设为可训练参数，使用 `Adam` 优化器和 `LPIPS` (感知损失) 迭代约1000步，来“猜”出一个 `W` 向量，使其生成的图像与对齐后的真实照片在“感知上”最相似。

这仍然是一个令人兴奋的未来探索方向。

## 致谢
* 本项目基于NVIDIA Research的 [StyleGAN2-ADA Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) 实现。
